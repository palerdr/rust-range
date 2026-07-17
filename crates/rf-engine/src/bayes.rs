//! Numerically stable Bayesian updates for observed postflop actions.

use crate::action::{Action, ActionLikelihoodModel, ActionObservation, DecisionKind, ModelError};
use crate::distribution::{stable_sum, DistributionError, RangeDistribution};
use crate::metrics::entropy;
use rf_core::{features::extract_features, Board, ComboWeights, HoleCards, KnownState, RfError, Street};

const LOG_2: f64 = std::f64::consts::LN_2;

#[derive(Clone, Debug, PartialEq)]
pub enum InferenceError {
    Core(RfError),
    Distribution(DistributionError),
    Model(ModelError),
    InvalidObservation(String),
    IllegalAction { action: Action, decision: DecisionKind },
    ImpossibleEvidence { observation: usize, action: Action },
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core(err) => write!(f, "{err}"),
            Self::Distribution(err) => write!(f, "{err}"),
            Self::Model(err) => write!(f, "model error: {err:?}"),
            Self::InvalidObservation(message) => write!(f, "invalid observation: {message}"),
            Self::IllegalAction { action, decision } => {
                write!(f, "action {action:?} is illegal for {decision:?}")
            }
            Self::ImpossibleEvidence { observation, action } => {
                write!(f, "observation {observation} ({action:?}) has zero probability")
            }
        }
    }
}

impl std::error::Error for InferenceError {}
impl From<RfError> for InferenceError {
    fn from(value: RfError) -> Self {
        Self::Core(value)
    }
}
impl From<DistributionError> for InferenceError {
    fn from(value: DistributionError) -> Self {
        Self::Distribution(value)
    }
}
impl From<ModelError> for InferenceError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ObservationTrace {
    pub board: String,
    pub decision: DecisionKind,
    pub action: Action,
    pub predictive_probability: f64,
    pub surprisal_bits: f64,
    pub entropy_before_bits: f64,
    pub entropy_after_bits: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceTrace {
    pub prior: RangeDistribution,
    pub posterior: RangeDistribution,
    pub observations: Vec<ObservationTrace>,
    pub log_evidence: f64,
}

/// Apply chronological action observations to a prior conditioned on known cards.
pub fn infer(
    hero: HoleCards,
    current_board: Board,
    prior: &RangeDistribution,
    observations: &[ActionObservation],
    model: &dyn ActionLikelihoodModel,
) -> Result<InferenceTrace, InferenceError> {
    let state = KnownState::new(hero, current_board)?;
    let conditioned = prior.clone().condition_on_dead(state.dead_cards())?;

    let mut current = conditioned.clone();
    let mut traces = Vec::with_capacity(observations.len());
    let mut total_log_evidence = 0.0;

    for (observation_index, observation) in observations.iter().enumerate() {
        validate_observation(current_board, observation)?;
        let before_entropy = entropy(&current);
        let (next, log_probability) = update_once(&current, observation, model, observation_index)?;
        let predictive_probability = log_probability.exp();
        let after_entropy = entropy(&next);

        traces.push(ObservationTrace {
            board: observation.board.to_string(),
            decision: observation.decision,
            action: observation.action,
            predictive_probability,
            surprisal_bits: -log_probability / LOG_2,
            entropy_before_bits: before_entropy,
            entropy_after_bits: after_entropy,
        });
        total_log_evidence += log_probability;
        current = next;
    }

    Ok(InferenceTrace {
        prior: conditioned,
        posterior: current,
        observations: traces,
        log_evidence: total_log_evidence,
    })
}

/// Normalize `prior(h) * likelihood(h)` using log-sum-exp.
pub fn posterior_from_likelihoods(
    prior: &RangeDistribution,
    likelihoods: &[f64],
) -> Result<(RangeDistribution, f64), InferenceError> {
    if likelihoods.len() != rf_core::NUM_HOLE_COMBOS {
        return Err(InferenceError::InvalidObservation(format!(
            "expected {} likelihoods, got {}",
            rf_core::NUM_HOLE_COMBOS,
            likelihoods.len()
        )));
    }

    let mut log_weights = vec![f64::NEG_INFINITY; likelihoods.len()];
    for (idx, (probability, &likelihood)) in prior.iter().map(|(_, p)| p).zip(likelihoods.iter()).enumerate() {
        if !likelihood.is_finite() || !(0.0..=1.0).contains(&likelihood) {
            return Err(InferenceError::InvalidObservation(format!(
                "likelihood at combo {idx} is invalid: {likelihood}"
            )));
        }
        if probability > 0.0 && likelihood > 0.0 {
            log_weights[idx] = probability.ln() + likelihood.ln();
        }
    }

    let max_log = log_weights
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_log.is_finite() {
        return Err(InferenceError::ImpossibleEvidence {
            observation: 0,
            action: Action::Check,
        });
    }

    let exp_sum = stable_sum(
        &log_weights
            .iter()
            .map(|value| {
                if value.is_finite() {
                    (*value - max_log).exp()
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>(),
    );
    let log_evidence = max_log + exp_sum.ln();
    let mut weights = ComboWeights::zeros();
    for (idx, value) in log_weights.iter().enumerate() {
        let normalized = if value.is_finite() {
            (*value - log_evidence).exp()
        } else {
            0.0
        };
        weights.as_mut_slice()[idx] = normalized;
    }
    Ok((RangeDistribution::from_weights(weights)?, log_evidence))
}

fn update_once(
    prior: &RangeDistribution,
    observation: &ActionObservation,
    model: &dyn ActionLikelihoodModel,
    observation_index: usize,
) -> Result<(RangeDistribution, f64), InferenceError> {
    let action_distribution = model.distribution(
        observation.board.street(),
        observation.decision,
        rf_core::features::ModelBucket::Air,
    );
    if let Err(ModelError::UnsupportedStreet) = action_distribution {
        return Err(InferenceError::InvalidObservation(
            "preflop actions are unsupported".to_string(),
        ));
    }

    let mut likelihoods = vec![0.0; rf_core::NUM_HOLE_COMBOS];
    for (id, probability) in prior.iter() {
        if probability == 0.0 {
            continue;
        }
        let features = extract_features(id.hole_cards(), &observation.board)?;
        let row = model.distribution(observation.board.street(), observation.decision, features.bucket)?;
        likelihoods[id.index()] = row
            .iter()
            .find(|(candidate, _)| *candidate == observation.action)
            .map(|(_, probability)| *probability)
            .ok_or(InferenceError::IllegalAction {
                action: observation.action,
                decision: observation.decision,
            })?;
    }

    match posterior_from_likelihoods(prior, &likelihoods) {
        Ok(value) => Ok(value),
        Err(InferenceError::ImpossibleEvidence { .. }) => Err(InferenceError::ImpossibleEvidence {
            observation: observation_index,
            action: observation.action,
        }),
        Err(err) => Err(err),
    }
}

fn validate_observation(current_board: Board, observation: &ActionObservation) -> Result<(), InferenceError> {
    if !matches!(observation.board.street(), Street::Flop | Street::Turn | Street::River) {
        return Err(InferenceError::InvalidObservation(
            "observations must be postflop".to_string(),
        ));
    }
    if !observation.board.is_prefix_of(&current_board) {
        return Err(InferenceError::InvalidObservation(format!(
            "observation board {} is not a prefix of current board {}",
            observation.board, current_board
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{toy_model, Action, DecisionKind};
    use rf_core::{CardMask, ComboWeights};

    fn two_state_prior() -> RangeDistribution {
        let mut weights = ComboWeights::zeros();
        weights.as_mut_slice()[0] = 1.0;
        weights.as_mut_slice()[1] = 3.0;
        RangeDistribution::from_weights(weights).unwrap()
    }

    #[test]
    fn log_space_update_matches_two_state_bayes_example() {
        let prior = two_state_prior();
        let likelihoods = (0..rf_core::NUM_HOLE_COMBOS)
            .map(|idx| match idx {
                0 => 0.25,
                1 => 0.75,
                _ => 0.0,
            })
            .collect::<Vec<_>>();
        let (posterior, evidence) = posterior_from_likelihoods(&prior, &likelihoods).unwrap();
        assert!((posterior.probability(rf_core::ComboId::from_raw(0).unwrap()) - 1.0 / 10.0).abs() < 1e-12);
        assert!((posterior.probability(rf_core::ComboId::from_raw(1).unwrap()) - 9.0 / 10.0).abs() < 1e-12);
        assert!((evidence.exp() - 0.625).abs() < 1e-12);
    }

    #[test]
    fn constant_likelihood_leaves_distribution_unchanged() {
        let prior = two_state_prior();
        let likelihoods = vec![0.5; rf_core::NUM_HOLE_COMBOS];
        let (posterior, _) = posterior_from_likelihoods(&prior, &likelihoods).unwrap();
        for (id, p) in prior.iter() {
            assert!((p - posterior.probability(id)).abs() < 1e-12);
        }
    }

    #[test]
    fn impossible_likelihood_is_typed() {
        let prior = two_state_prior();
        let err = posterior_from_likelihoods(&prior, &vec![0.0; rf_core::NUM_HOLE_COMBOS]).unwrap_err();
        assert!(matches!(err, InferenceError::ImpossibleEvidence { .. }));
    }

    #[test]
    fn real_model_updates_a_postflop_range() {
        let hero = HoleCards::from_strs("As", "Kh").unwrap();
        let board = Board::parse("Qs Jh 2c").unwrap();
        let prior = RangeDistribution::uniform_conditioned(hero.mask().union(board.mask())).unwrap();
        let observations = [ActionObservation {
            board,
            decision: DecisionKind::Unopened,
            action: Action::LargeBet,
        }];
        let trace = infer(hero, board, &prior, &observations, &toy_model().unwrap()).unwrap();
        assert_eq!(trace.posterior.support_size(), prior.support_size());
        assert!(trace.log_evidence.is_finite());
        assert!(trace.observations[0].predictive_probability > 0.0);
        assert_eq!(
            trace.posterior.probability(rf_core::ComboId::from_hole_cards(
                HoleCards::from_strs("As", "2d").unwrap()
            )),
            0.0
        );
        let _ = CardMask::EMPTY;
    }
}

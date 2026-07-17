//! Expected information from future actions and public-card reveals.

use crate::action::{legal_actions, Action, ActionLikelihoodModel, DecisionKind, ModelError};
use crate::bayes::{posterior_from_likelihoods, InferenceError};
use crate::distribution::{DistributionError, RangeDistribution};
use crate::metrics::entropy;
use rf_core::{features::extract_features, Board, Card, CardMask, HoleCards, KnownState, RfError, Street};

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ActionInformationRow {
    pub action: Action,
    pub predictive_probability: f64,
    pub posterior_entropy_bits: Option<f64>,
    pub information_gain_bits: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ActionInformationReport {
    pub decision: DecisionKind,
    pub current_entropy_bits: f64,
    pub expected_information_bits: f64,
    pub rows: Vec<ActionInformationRow>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct NextCardInformationRow {
    pub card: String,
    pub predictive_probability: f64,
    pub information_gain_bits: f64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct NextCardInformationReport {
    pub current_entropy_bits: f64,
    pub expected_information_bits: f64,
    pub rows: Vec<NextCardInformationRow>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InformationError {
    Core(RfError),
    Distribution(DistributionError),
    Inference(InferenceError),
    Model(ModelError),
    UnsupportedStreet,
}
impl std::fmt::Display for InformationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core(err) => write!(f, "{err}"),
            Self::Distribution(err) => write!(f, "{err}"),
            Self::Inference(err) => write!(f, "{err}"),
            Self::Model(err) => write!(f, "model error: {err:?}"),
            Self::UnsupportedStreet => write!(f, "next-card information is supported only on flop and turn"),
        }
    }
}
impl std::error::Error for InformationError {}
impl From<RfError> for InformationError {
    fn from(value: RfError) -> Self {
        Self::Core(value)
    }
}
impl From<DistributionError> for InformationError {
    fn from(value: DistributionError) -> Self {
        Self::Distribution(value)
    }
}
impl From<InferenceError> for InformationError {
    fn from(value: InferenceError) -> Self {
        Self::Inference(value)
    }
}
impl From<ModelError> for InformationError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

/// Compute predictive action probabilities and expected entropy reduction.
pub fn action_information(
    hero: HoleCards,
    board: Board,
    posterior: &RangeDistribution,
    decision: DecisionKind,
    model: &dyn ActionLikelihoodModel,
) -> Result<ActionInformationReport, InformationError> {
    if board.street() == Street::Preflop {
        return Err(InformationError::UnsupportedStreet);
    }
    let current_entropy_bits = entropy(posterior);
    let mut rows = Vec::with_capacity(3);
    let mut expected = 0.0;
    for action in legal_actions(decision) {
        let likelihoods = action_likelihoods(hero, board, posterior, decision, action, model)?;
        let predictive_probability = posterior
            .iter()
            .zip(likelihoods.iter())
            .map(|((_, p), likelihood)| p * likelihood)
            .sum::<f64>();
        let (posterior_entropy_bits, information_gain_bits) = if predictive_probability > 0.0 {
            let (conditional, _) = posterior_from_likelihoods(posterior, &likelihoods)?;
            let conditional_entropy = entropy(&conditional);
            let gain = (current_entropy_bits - conditional_entropy).max(0.0);
            expected += predictive_probability * gain;
            (Some(conditional_entropy), Some(gain))
        } else {
            (None, None)
        };
        rows.push(ActionInformationRow {
            action,
            predictive_probability,
            posterior_entropy_bits,
            information_gain_bits,
        });
    }
    Ok(ActionInformationReport {
        decision,
        current_entropy_bits,
        expected_information_bits: expected.max(0.0),
        rows,
    })
}

/// Compute information value for each possible next card on flop or turn.
pub fn next_card_information(
    hero: HoleCards,
    board: Board,
    posterior: &RangeDistribution,
) -> Result<NextCardInformationReport, InformationError> {
    if !matches!(board.street(), Street::Flop | Street::Turn) {
        return Err(InformationError::UnsupportedStreet);
    }
    let current_entropy_bits = entropy(posterior);
    let candidates: Vec<Card> = rf_core::remaining_cards(hero.mask().union(board.mask())).collect();
    let mut rows = Vec::with_capacity(candidates.len());
    let mut expected = 0.0;
    for card in candidates {
        let mut probability = 0.0;
        for (id, p) in posterior.iter() {
            if p == 0.0 || id.hole_cards().mask().contains(card) {
                continue;
            }
            let legal_next_cards = 52u32 - hero.mask().count() - board.mask().count() - id.hole_cards().mask().count();
            probability += p / legal_next_cards as f64;
        }
        let conditional = posterior.clone().condition_on_dead(CardMask::from_cards([card]))?;
        let gain = (current_entropy_bits - entropy(&conditional)).max(0.0);
        expected += probability * gain;
        rows.push(NextCardInformationRow {
            card: card.to_string(),
            predictive_probability: probability,
            information_gain_bits: gain,
        });
    }
    Ok(NextCardInformationReport {
        current_entropy_bits,
        expected_information_bits: expected.max(0.0),
        rows,
    })
}

fn action_likelihoods(
    hero: HoleCards,
    board: Board,
    posterior: &RangeDistribution,
    decision: DecisionKind,
    action: Action,
    model: &dyn ActionLikelihoodModel,
) -> Result<Vec<f64>, InformationError> {
    KnownState::new(hero, board)?;
    let mut out = vec![0.0; rf_core::NUM_HOLE_COMBOS];
    for (id, p) in posterior.iter() {
        if p == 0.0 {
            continue;
        }
        let features = extract_features(id.hole_cards(), &board)?;
        let row = model.distribution(board.street(), decision, features.bucket)?;
        out[id.index()] = row
            .iter()
            .find(|(candidate, _)| *candidate == action)
            .map(|(_, probability)| *probability)
            .ok_or(InferenceError::IllegalAction { action, decision })?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::toy_model;
    use rf_core::{ComboId, ComboWeights};

    fn small_range() -> (HoleCards, Board, RangeDistribution) {
        let hero = HoleCards::from_strs("As", "Kd").unwrap();
        let board = Board::parse("Qs Jh 2c").unwrap();
        let mut weights = ComboWeights::zeros();
        for cards in [(5u8, 6u8), (5, 7), (5, 8), (6, 7)] {
            let hole = HoleCards::new(
                rf_core::Card::from_parts(cards.0, 0).unwrap(),
                rf_core::Card::from_parts(cards.1, 1).unwrap(),
            )
            .unwrap();
            weights.as_mut_slice()[ComboId::from_hole_cards(hole).index()] = 1.0;
        }
        (
            hero,
            board,
            RangeDistribution::from_weights(weights)
                .unwrap()
                .condition_on_dead(hero.mask().union(board.mask()))
                .unwrap(),
        )
    }

    #[test]
    fn action_rows_are_stable_and_sum_to_one() {
        let (hero, board, posterior) = small_range();
        let report =
            action_information(hero, board, &posterior, DecisionKind::Unopened, &toy_model().unwrap()).unwrap();
        let sum = report.rows.iter().map(|row| row.predictive_probability).sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-12);
        assert_eq!(report.rows[0].action, Action::Check);
        assert!(report.expected_information_bits >= 0.0);
    }

    #[test]
    fn next_card_probabilities_sum_to_one() {
        let (hero, board, posterior) = small_range();
        let report = next_card_information(hero, board, &posterior).unwrap();
        let sum = report.rows.iter().map(|row| row.predictive_probability).sum::<f64>();
        assert!((sum - 1.0).abs() < 1e-12);
        assert_eq!(report.rows.len(), 47);
    }

    #[test]
    fn next_card_information_rejects_river() {
        let hero = HoleCards::from_strs("As", "Kd").unwrap();
        let board = Board::parse("Qs Jh 2c 3d 4s").unwrap();
        let posterior = RangeDistribution::uniform_conditioned(hero.mask().union(board.mask())).unwrap();
        assert_eq!(
            next_card_information(hero, board, &posterior),
            Err(InformationError::UnsupportedStreet)
        );
    }
}

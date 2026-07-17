//! Scenario parsing and one-pipeline analysis reports.

use crate::action::{
    legal_actions, Action, ActionLikelihoodModel, ActionModelMetadata, ActionObservation, DecisionKind, ModelError,
};
use crate::bayes::{infer, InferenceError};
use crate::distribution::{DistributionError, RangeDistribution};
use crate::equity::{exact_equity, monte_carlo_equity, EquityError, EquityResult};
use crate::information::{
    action_information, next_card_information, ActionInformationReport, InformationError, NextCardInformationReport,
};
use crate::metrics::{bucket_masses, effective_hand_count, entropy, kl_bits, total_variation, MetricError};
use rf_core::{expand_range, Board, HoleCards, RangeSpec, RfError, Street};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ScenarioInput {
    pub hero: String,
    pub board: String,
    pub prior: PriorInput,
    #[serde(default)]
    pub observations: Vec<ObservationInput>,
    pub equity: EquityInput,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct PriorInput {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub value: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ObservationInput {
    pub board: String,
    pub decision: String,
    pub action: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct EquityInput {
    pub method: String,
    #[serde(default = "default_samples")]
    pub samples: u64,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_samples() -> u64 {
    100_000
}
fn default_seed() -> u64 {
    42
}

#[derive(Clone, Debug, PartialEq)]
pub struct Scenario {
    pub hero: HoleCards,
    pub board: Board,
    pub prior: RangeSpec,
    pub observations: Vec<ActionObservation>,
    pub equity: EquityInput,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScenarioError {
    Json(String),
    Core(RfError),
    Distribution(DistributionError),
    Inference(InferenceError),
    Equity(EquityError),
    Information(InformationError),
    Metric(MetricError),
    Model(ModelError),
    InvalidInput(String),
}
impl std::fmt::Display for ScenarioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json(message) => write!(f, "invalid scenario JSON: {message}"),
            Self::Core(err) => write!(f, "{err}"),
            Self::Distribution(err) => write!(f, "{err}"),
            Self::Inference(err) => write!(f, "{err}"),
            Self::Equity(err) => write!(f, "{err}"),
            Self::Information(err) => write!(f, "{err}"),
            Self::Metric(err) => write!(f, "metric error: {err:?}"),
            Self::Model(err) => write!(f, "model error: {err:?}"),
            Self::InvalidInput(message) => write!(f, "invalid scenario: {message}"),
        }
    }
}
impl std::error::Error for ScenarioError {}
impl From<RfError> for ScenarioError {
    fn from(value: RfError) -> Self {
        Self::Core(value)
    }
}
impl From<DistributionError> for ScenarioError {
    fn from(value: DistributionError) -> Self {
        Self::Distribution(value)
    }
}
impl From<InferenceError> for ScenarioError {
    fn from(value: InferenceError) -> Self {
        Self::Inference(value)
    }
}
impl From<EquityError> for ScenarioError {
    fn from(value: EquityError) -> Self {
        Self::Equity(value)
    }
}
impl From<InformationError> for ScenarioError {
    fn from(value: InformationError) -> Self {
        Self::Information(value)
    }
}
impl From<MetricError> for ScenarioError {
    fn from(value: MetricError) -> Self {
        Self::Metric(value)
    }
}
impl From<ModelError> for ScenarioError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

impl ScenarioInput {
    pub fn parse(text: &str) -> Result<Self, ScenarioError> {
        serde_json::from_str(text).map_err(|err| ScenarioError::Json(err.to_string()))
    }

    pub fn validate(&self) -> Result<Scenario, ScenarioError> {
        let hero = parse_hole_cards(&self.hero)?;
        let board = Board::parse(&self.board)?;
        rf_core::KnownState::new(hero, board)?;
        let prior = match self.prior.kind.trim().to_ascii_lowercase().as_str() {
            "random" => RangeSpec::Random,
            "notation" => RangeSpec::Notation(self.prior.value.clone()),
            other => return Err(ScenarioError::InvalidInput(format!("unsupported prior type: {other}"))),
        };
        let mut observations = Vec::with_capacity(self.observations.len());
        for input in &self.observations {
            let observation_board = Board::parse(&input.board)?;
            let decision = parse_decision(&input.decision)?;
            let action = parse_action(&input.action)?;
            if !legal_actions(decision).contains(&action) {
                return Err(ScenarioError::InvalidInput(format!(
                    "action {action:?} is illegal for {decision:?}"
                )));
            }
            if !matches!(observation_board.street(), Street::Flop | Street::Turn | Street::River) {
                return Err(ScenarioError::InvalidInput("observations must be postflop".to_string()));
            }
            if !observation_board.is_prefix_of(&board) {
                return Err(ScenarioError::InvalidInput(format!(
                    "observation board {} is not a prefix of current board {}",
                    observation_board, board
                )));
            }
            observations.push(ActionObservation {
                board: observation_board,
                decision,
                action,
            });
        }
        for pair in observations.windows(2) {
            if !pair[0].board.is_prefix_of(&pair[1].board) {
                return Err(ScenarioError::InvalidInput(
                    "observations must be chronological board prefixes".to_string(),
                ));
            }
        }
        let method = self.equity.method.trim().to_ascii_lowercase();
        if method != "exact" && method != "monte_carlo" && method != "monte-carlo" {
            return Err(ScenarioError::InvalidInput(format!(
                "unsupported equity method: {method}"
            )));
        }
        if self.equity.samples == 0 && method != "exact" {
            return Err(ScenarioError::InvalidInput(
                "Monte Carlo samples must be greater than zero".to_string(),
            ));
        }
        if method == "exact" && board.street() == Street::Preflop {
            return Err(ScenarioError::InvalidInput(
                "exact equity is unsupported preflop; use monte_carlo".to_string(),
            ));
        }
        Ok(Scenario {
            hero,
            board,
            prior,
            observations,
            equity: self.equity.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TopCombo {
    pub combo: String,
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DistributionSummary {
    pub support_size: usize,
    pub entropy_bits: f64,
    pub effective_hand_count: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnalysisReport {
    pub schema_version: String,
    pub model_id: String,
    pub model_metadata: Option<ActionModelMetadata>,
    pub hero: String,
    pub board: String,
    pub street: String,
    pub prior: DistributionSummary,
    pub posterior: DistributionSummary,
    pub total_variation: f64,
    pub kl_divergence_bits: f64,
    pub log_evidence: f64,
    pub action_trace: Vec<crate::bayes::ObservationTrace>,
    pub top_combos: Vec<TopCombo>,
    pub bucket_masses: [f64; 6],
    pub equity: EquityResult,
    pub action_information: Option<ActionInformationReport>,
    pub next_card_information: Option<NextCardInformationReport>,
    pub warnings: Vec<String>,
}

/// Run parsing-independent engine work and create the single report consumed by both CLI formats.
pub fn analyze(
    scenario: &Scenario,
    model: &dyn ActionLikelihoodModel,
    top_count: usize,
) -> Result<AnalysisReport, ScenarioError> {
    let weights = expand_range(&scenario.prior)?;
    let prior = RangeDistribution::from_weights(weights)?;
    let trace = infer(scenario.hero, scenario.board, &prior, &scenario.observations, model)?;
    let posterior = &trace.posterior;
    let equity = match scenario.equity.method.trim().to_ascii_lowercase().as_str() {
        "exact" => exact_equity(scenario.hero, scenario.board, posterior)?,
        _ => monte_carlo_equity(
            scenario.hero,
            scenario.board,
            posterior,
            scenario.equity.samples,
            scenario.equity.seed,
        )?,
    };
    let last_decision = scenario
        .observations
        .last()
        .map(|observation| observation.decision)
        .unwrap_or(crate::action::DecisionKind::Unopened);
    let action_report = if matches!(scenario.board.street(), Street::Flop | Street::Turn | Street::River) {
        Some(action_information(
            scenario.hero,
            scenario.board,
            posterior,
            last_decision,
            model,
        )?)
    } else {
        None
    };
    let next_report = if matches!(scenario.board.street(), Street::Flop | Street::Turn) {
        Some(next_card_information(scenario.hero, scenario.board, posterior)?)
    } else {
        None
    };
    let top_combos = top_combos(posterior, top_count);
    let bucket_masses = if matches!(scenario.board.street(), Street::Flop | Street::Turn | Street::River) {
        bucket_masses(posterior, &scenario.board)?
    } else {
        [0.0; 6]
    };
    let mut warnings = Vec::new();
    if model.metadata().map(|metadata| metadata.calibration.as_str()) != Some("calibrated") {
        warnings.push("The configured action model is illustrative; it is not a population prediction.".to_string());
    }
    Ok(AnalysisReport {
        schema_version: "0.1".to_string(),
        model_id: model.model_id().to_string(),
        model_metadata: model.metadata().cloned(),
        hero: scenario.hero.to_string(),
        board: scenario.board.to_string(),
        street: format!("{:?}", scenario.board.street()).to_ascii_lowercase(),
        prior: summary(&trace.prior),
        posterior: summary(posterior),
        total_variation: total_variation(&trace.prior, posterior),
        kl_divergence_bits: kl_bits(posterior, &trace.prior)?,
        log_evidence: trace.log_evidence,
        action_trace: trace.observations.clone(),
        top_combos,
        bucket_masses,
        equity,
        action_information: action_report,
        next_card_information: next_report,
        warnings,
    })
}

fn summary(distribution: &RangeDistribution) -> DistributionSummary {
    DistributionSummary {
        support_size: distribution.support_size(),
        entropy_bits: entropy(distribution),
        effective_hand_count: effective_hand_count(distribution),
    }
}

fn top_combos(distribution: &RangeDistribution, count: usize) -> Vec<TopCombo> {
    let mut items: Vec<_> = distribution
        .iter()
        .filter(|(_, probability)| *probability > 0.0)
        .collect();
    items.sort_by(|(left_id, left_probability), (right_id, right_probability)| {
        right_probability
            .partial_cmp(left_probability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left_id.index().cmp(&right_id.index()))
    });
    items.truncate(count);
    items
        .into_iter()
        .map(|(id, probability)| TopCombo {
            combo: id.hole_cards().to_string(),
            probability,
        })
        .collect()
}

fn parse_hole_cards(text: &str) -> Result<HoleCards, ScenarioError> {
    let compact = text.trim();
    if compact.len() != 4 {
        return Err(ScenarioError::InvalidInput(
            "hero must contain exactly two cards, for example AsKh".to_string(),
        ));
    }
    Ok(HoleCards::from_strs(&compact[0..2], &compact[2..4])?)
}

fn parse_decision(text: &str) -> Result<DecisionKind, ScenarioError> {
    match text.trim().to_ascii_lowercase().as_str() {
        "unopened" => Ok(DecisionKind::Unopened),
        "facing_bet" | "facingbet" => Ok(DecisionKind::FacingBet),
        other => Err(ScenarioError::InvalidInput(format!("unknown decision: {other}"))),
    }
}

fn parse_action(text: &str) -> Result<Action, ScenarioError> {
    match text.trim().to_ascii_lowercase().as_str() {
        "check" => Ok(Action::Check),
        "smallbet" | "small_bet" | "bet_small" => Ok(Action::SmallBet),
        "largebet" | "large_bet" | "bet_large" => Ok(Action::LargeBet),
        "fold" => Ok(Action::Fold),
        "call" => Ok(Action::Call),
        "raise" => Ok(Action::Raise),
        other => Err(ScenarioError::InvalidInput(format!("unknown action: {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::toy_model;

    fn scenario_text(method: &str) -> String {
        format!(
            r#"{{
            "hero": "AsKh",
            "board": "QsJh2c",
            "prior": {{"type":"notation", "value":"random"}},
            "observations": [{{"board":"QsJh2c", "decision":"unopened", "action":"bet_large"}}],
            "equity": {{"method":"{method}", "samples":1000, "seed":7}}
        }}"#
        )
    }

    #[test]
    fn parses_and_analyzes_a_flop_scenario() {
        let input = ScenarioInput::parse(&scenario_text("exact")).unwrap();
        let scenario = input.validate().unwrap();
        let report = analyze(&scenario, &toy_model().unwrap(), 5).unwrap();
        assert_eq!(report.schema_version, "0.1");
        assert_eq!(report.top_combos.len(), 5);
        assert!(report.equity.equity.is_finite());
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn rejects_invalid_prior_type() {
        let input = ScenarioInput::parse(&scenario_text("exact")).unwrap();
        let mut invalid = input;
        invalid.prior.kind = "percentage".to_string();
        assert!(matches!(invalid.validate(), Err(ScenarioError::InvalidInput(_))));
    }

    #[test]
    fn preflop_monte_carlo_is_allowed() {
        let text = r#"{
            "hero":"AsKh", "board":"", "prior":{"type":"random","value":""},
            "observations":[], "equity":{"method":"monte_carlo","samples":10,"seed":1}
        }"#;
        let scenario = ScenarioInput::parse(text).unwrap().validate().unwrap();
        let report = analyze(&scenario, &toy_model().unwrap(), 3).unwrap();
        assert_eq!(report.equity.method, "monte_carlo");
        assert!(report.action_information.is_none());
    }
}

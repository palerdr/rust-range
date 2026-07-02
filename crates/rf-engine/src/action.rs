use rf_core::{features::ModelBucket, Board, Street};
use serde::{Deserialize, Serialize};

pub(crate) const ROW_COUNT: usize = 36;
pub(crate) const ROW_SUM_EPS: f64 = 1e-9;

#[derive(Clone, Debug, PartialEq)]
pub enum ModelError {
    UnsupportedStreet,
    InvalidMetadata(&'static str),
    MissingRow {
        street: Street,
        decision: DecisionKind,
        bucket: ModelBucket,
    },
    DuplicateRow {
        street: Street,
        decision: DecisionKind,
        bucket: ModelBucket,
    },
    InvalidRow(String),
    Json(String),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DecisionKind {
    Unopened,
    FacingBet,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Action {
    Check,
    SmallBet,
    LargeBet,
    Fold,
    Call,
    Raise,
}

pub struct ActionObservation {
    pub board: Board,
    pub decision: DecisionKind,
    pub action: Action,
}

pub type ActionDistribution = [(Action, f64); 3];

pub trait ActionLikelihoodModel: Send + Sync {
    fn distribution(
        &self,
        street: Street,
        decision: DecisionKind,
        bucket: ModelBucket,
    ) -> Result<ActionDistribution, ModelError>;

    fn model_id(&self) -> &str;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ActionModelMetadata {
    pub id: String,
    pub version: String,
    pub description: String,
    pub calibration: String,
}

#[derive(Debug)]
pub struct BucketedActionModel {
    metadata: ActionModelMetadata,
    rows: [ActionDistribution; ROW_COUNT],
}

impl BucketedActionModel {
    pub fn from_rows(
        metadata: ActionModelMetadata,
        rows: Vec<(Street, DecisionKind, ModelBucket, ActionDistribution)>,
    ) -> Result<Self, ModelError> {
        validate_metadata(&metadata)?;

        let mut pending: [Option<ActionDistribution>; ROW_COUNT] = [None; ROW_COUNT];

        for (street, decision, bucket, row) in rows {
            validate_row(row, decision)?;
            let idx = row_index(street, decision, bucket)?;

            if pending[idx].is_some() {
                return Err(ModelError::DuplicateRow {
                    street,
                    decision,
                    bucket,
                });
            }

            pending[idx] = Some(row);
        }

        let mut out = [[(Action::Check, 0.0); 3]; ROW_COUNT];
        for idx in 0..ROW_COUNT {
            out[idx] = match pending[idx] {
                Some(row) => row,
                None => {
                    let (street, decision, bucket) = row_key(idx);
                    return Err(ModelError::MissingRow {
                        street,
                        decision,
                        bucket,
                    });
                }
            };
        }

        Ok(Self {
            metadata,
            rows: out,
        })
    }

    pub fn metadata(&self) -> &ActionModelMetadata {
        &self.metadata
    }
}

impl ActionLikelihoodModel for BucketedActionModel {
    fn distribution(
        &self,
        street: Street,
        decision: DecisionKind,
        bucket: ModelBucket,
    ) -> Result<ActionDistribution, ModelError> {
        let idx = row_index(street, decision, bucket)?;
        Ok(self.rows[idx])
    }

    fn model_id(&self) -> &str {
        &self.metadata.id
    }
}

pub fn toy_model() -> Result<BucketedActionModel, ModelError> {
    let metadata = ActionModelMetadata {
        id: "toy-action-model".to_string(),
        version: "0.1.0".to_string(),
        description: "Illustrative bucketed action model for demos and tests".to_string(),
        calibration: "illustrative".to_string(),
    };

    let mut rows = Vec::with_capacity(ROW_COUNT);
    for street in [Street::Flop, Street::Turn, Street::River] {
        for decision in [DecisionKind::Unopened, DecisionKind::FacingBet] {
            for bucket in buckets() {
                rows.push((street, decision, bucket, toy_distribution(decision, bucket)));
            }
        }
    }

    BucketedActionModel::from_rows(metadata, rows)
}

pub fn legal_actions(decision: DecisionKind) -> [Action; 3] {
    match decision {
        DecisionKind::Unopened => [Action::Check, Action::SmallBet, Action::LargeBet],
        DecisionKind::FacingBet => [Action::Fold, Action::Call, Action::Raise],
    }
}

pub fn require_postflop(street: Street) -> Result<(), ModelError> {
    match street {
        Street::Flop | Street::Turn | Street::River => Ok(()),
        Street::Preflop => Err(ModelError::UnsupportedStreet),
    }
}

pub fn row_index(
    street: Street,
    decision: DecisionKind,
    bucket: ModelBucket,
) -> Result<usize, ModelError> {
    let si = match street {
        Street::Flop => 0,
        Street::Turn => 1,
        Street::River => 2,
        Street::Preflop => return Err(ModelError::UnsupportedStreet),
    };

    let di = match decision {
        DecisionKind::Unopened => 0,
        DecisionKind::FacingBet => 1,
    };

    Ok((si * 2 + di) * 6 + bucket_index(bucket))
}

pub fn validate_row(row: ActionDistribution, decision: DecisionKind) -> Result<(), ModelError> {
    let legal = legal_actions(decision);
    let mut sum = 0.0;

    for i in 0..3 {
        let (action, prob) = row[i];

        if action != legal[i] {
            return Err(ModelError::InvalidRow(format!(
                "{action:?} is not legal at position {i} for {decision:?}"
            )));
        }

        validate_probability(action, prob)?;
        sum += prob;
    }

    validate_sum(sum)
}

fn toy_distribution(decision: DecisionKind, bucket: ModelBucket) -> ActionDistribution {
    let probs = match (decision, bucket) {
        (DecisionKind::Unopened, ModelBucket::Air) => [0.70, 0.25, 0.05],
        (DecisionKind::Unopened, ModelBucket::Draw) => [0.45, 0.40, 0.15],
        (DecisionKind::Unopened, ModelBucket::OnePair) => [0.45, 0.40, 0.15],
        (DecisionKind::Unopened, ModelBucket::TwoPairOrTrips) => [0.25, 0.45, 0.30],
        (DecisionKind::Unopened, ModelBucket::StraightOrFlush) => [0.20, 0.45, 0.35],
        (DecisionKind::Unopened, ModelBucket::FullHousePlus) => [0.10, 0.35, 0.55],
        (DecisionKind::FacingBet, ModelBucket::Air) => [0.70, 0.25, 0.05],
        (DecisionKind::FacingBet, ModelBucket::Draw) => [0.30, 0.55, 0.15],
        (DecisionKind::FacingBet, ModelBucket::OnePair) => [0.25, 0.65, 0.10],
        (DecisionKind::FacingBet, ModelBucket::TwoPairOrTrips) => [0.10, 0.65, 0.25],
        (DecisionKind::FacingBet, ModelBucket::StraightOrFlush) => [0.05, 0.55, 0.40],
        (DecisionKind::FacingBet, ModelBucket::FullHousePlus) => [0.02, 0.38, 0.60],
    };

    let actions = legal_actions(decision);
    [
        (actions[0], probs[0]),
        (actions[1], probs[1]),
        (actions[2], probs[2]),
    ]
}

fn validate_metadata(metadata: &ActionModelMetadata) -> Result<(), ModelError> {
    if metadata.id.trim().is_empty() {
        return Err(ModelError::InvalidMetadata("id"));
    }
    if metadata.version.trim().is_empty() {
        return Err(ModelError::InvalidMetadata("version"));
    }
    if metadata.description.trim().is_empty() {
        return Err(ModelError::InvalidMetadata("description"));
    }
    if metadata.calibration.trim().is_empty() {
        return Err(ModelError::InvalidMetadata("calibration"));
    }
    Ok(())
}

pub(crate) fn validate_probability(action: Action, prob: f64) -> Result<(), ModelError> {
    if !prob.is_finite() {
        return Err(ModelError::InvalidRow(format!(
            "{action:?} probability is not finite: {prob}"
        )));
    }

    if !(0.0..=1.0).contains(&prob) {
        return Err(ModelError::InvalidRow(format!(
            "{action:?} probability is outside [0, 1]: {prob}"
        )));
    }

    Ok(())
}

pub(crate) fn validate_sum(sum: f64) -> Result<(), ModelError> {
    if (sum - 1.0).abs() > ROW_SUM_EPS {
        return Err(ModelError::InvalidRow(format!(
            "row probabilities sum to {sum}, not 1.0"
        )));
    }
    Ok(())
}

fn row_key(index: usize) -> (Street, DecisionKind, ModelBucket) {
    let street = match index / 12 {
        0 => Street::Flop,
        1 => Street::Turn,
        _ => Street::River,
    };

    let decision = match (index % 12) / 6 {
        0 => DecisionKind::Unopened,
        _ => DecisionKind::FacingBet,
    };

    (street, decision, buckets()[index % 6])
}

fn bucket_index(bucket: ModelBucket) -> usize {
    match bucket {
        ModelBucket::Air => 0,
        ModelBucket::Draw => 1,
        ModelBucket::OnePair => 2,
        ModelBucket::TwoPairOrTrips => 3,
        ModelBucket::StraightOrFlush => 4,
        ModelBucket::FullHousePlus => 5,
    }
}

pub(crate) fn buckets() -> [ModelBucket; 6] {
    [
        ModelBucket::Air,
        ModelBucket::Draw,
        ModelBucket::OnePair,
        ModelBucket::TwoPairOrTrips,
        ModelBucket::StraightOrFlush,
        ModelBucket::FullHousePlus,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> ActionModelMetadata {
        ActionModelMetadata {
            id: "test-model".to_string(),
            version: "0.1.0".to_string(),
            description: "test action model".to_string(),
            calibration: "illustrative".to_string(),
        }
    }

    fn dist(decision: DecisionKind) -> ActionDistribution {
        match decision {
            DecisionKind::Unopened => [
                (Action::Check, 0.60),
                (Action::SmallBet, 0.30),
                (Action::LargeBet, 0.10),
            ],
            DecisionKind::FacingBet => [
                (Action::Fold, 0.25),
                (Action::Call, 0.60),
                (Action::Raise, 0.15),
            ],
        }
    }

    fn full_rows() -> Vec<(Street, DecisionKind, ModelBucket, ActionDistribution)> {
        let mut rows = Vec::new();
        for street in [Street::Flop, Street::Turn, Street::River] {
            for decision in [DecisionKind::Unopened, DecisionKind::FacingBet] {
                for bucket in buckets() {
                    rows.push((street, decision, bucket, dist(decision)));
                }
            }
        }
        rows
    }

    #[test]
    fn rejects_missing_row() {
        let mut rows = full_rows();
        rows.pop();

        let err = BucketedActionModel::from_rows(metadata(), rows).unwrap_err();
        assert!(matches!(err, ModelError::MissingRow { .. }));
    }

    #[test]
    fn rejects_invalid_action_for_context() {
        let bad = [
            (Action::Check, 0.50),
            (Action::Call, 0.40),
            (Action::LargeBet, 0.10),
        ];

        assert!(matches!(
            validate_row(bad, DecisionKind::Unopened),
            Err(ModelError::InvalidRow(_))
        ));
    }

    #[test]
    fn rejects_negative_probability() {
        let bad = [
            (Action::Check, 0.60),
            (Action::SmallBet, -0.10),
            (Action::LargeBet, 0.50),
        ];

        assert!(matches!(
            validate_row(bad, DecisionKind::Unopened),
            Err(ModelError::InvalidRow(_))
        ));
    }

    #[test]
    fn rejects_nan_and_infinity_probability() {
        let nan = [
            (Action::Check, f64::NAN),
            (Action::SmallBet, 0.30),
            (Action::LargeBet, 0.70),
        ];
        let inf = [
            (Action::Check, 0.60),
            (Action::SmallBet, f64::INFINITY),
            (Action::LargeBet, 0.40),
        ];

        assert!(matches!(
            validate_row(nan, DecisionKind::Unopened),
            Err(ModelError::InvalidRow(_))
        ));
        assert!(matches!(
            validate_row(inf, DecisionKind::Unopened),
            Err(ModelError::InvalidRow(_))
        ));
    }

    #[test]
    fn rejects_row_sums_outside_tolerance() {
        let bad = [
            (Action::Check, 0.60),
            (Action::SmallBet, 0.30),
            (Action::LargeBet, 0.11),
        ];

        assert!(matches!(
            validate_row(bad, DecisionKind::Unopened),
            Err(ModelError::InvalidRow(_))
        ));
    }

    #[test]
    fn every_supported_row_returns_actions_that_sum_to_one() {
        let model = toy_model().unwrap();

        for street in [Street::Flop, Street::Turn, Street::River] {
            for decision in [DecisionKind::Unopened, DecisionKind::FacingBet] {
                let legal = legal_actions(decision);

                for bucket in buckets() {
                    let row = model.distribution(street, decision, bucket).unwrap();
                    let sum = row.iter().map(|(_, probability)| probability).sum::<f64>();

                    assert_eq!(row.map(|(action, _)| action), legal);
                    assert!((sum - 1.0).abs() <= ROW_SUM_EPS);
                }
            }
        }
    }

    #[test]
    fn toy_model_declares_itself_illustrative() {
        let model = toy_model().unwrap();

        assert_eq!(model.metadata().calibration, "illustrative");
        assert_ne!(model.metadata().calibration, "calibrated");
    }
}

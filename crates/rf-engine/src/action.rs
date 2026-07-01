use crate::{action::{Action::{LargeBet, SmallBet}, DecisionKind::{FacingBet, Unopened}}, distribution::RangeDistribution};
use rf_core::{
    Board, ComboId, ComboWeights, Street, features::{ModelBucket, extract_features},
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ModelError {
    UnsupportedStreet,
    MissingRow,
}
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DecisionKind {
    Unopened,
    FacingBet,
}
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Action {
    Check,
    SmallBet,
    LargeBet,
    Fold,
    Call,
    Raise,
}

#[derive(Clone, Copy)]
pub struct BoardTexture{
    pub paired: bool,
    pub monotone: bool,
    pub twotone: bool,
    pub rainbow : bool,
    pub connected: bool,
    pub disconnected: bool,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionModelMetadata {
    pub id: String,
    pub version: String,
    pub description: String,
    pub calibration: String,
}

pub struct BucketedActionModel {
    metadata: ActionModelMetadata,
    rows: [ActionDistribution; 36],
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

pub fn legal_actions(decision: DecisionKind) -> Vec<Action> {
    match decision {
        DecisionKind::Unopened => {
            vec![Action::Check, Action::SmallBet, Action::LargeBet]
        }
        DecisionKind::FacingBet => {
            vec![Action::Fold, Action::Call, Action::Raise]
        }
    }
}

pub fn require_postflop(street: Street) -> Result<(), ModelError> {
    match street {
        Street::Flop | Street::Turn | Street::River => {Ok(())}
        _ => Err(ModelError::UnsupportedStreet)
    }
}

pub fn row_index(street: Street, decision: DecisionKind, bucket: ModelBucket) -> Result<usize, ModelError> {
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

    let bi = match bucket {
        ModelBucket::Air => 0,
        ModelBucket::Draw => 1,
        ModelBucket::OnePair => 2,
        ModelBucket::TwoPairOrTrips => 3,
        ModelBucket::StraightOrFlush => 4,
        ModelBucket::FullHousePlus => 5,
    };

    Ok((si * 2 + di) * 6 + bi)
}

pub fn validate_row(row: ActionDistribution, decision: DecisionKind) -> Result<(), ModelError> {
    let legal = legal_actions(decision);
    let mut sum = 0.0;

    for i in 0..3 {
        let (action, prob) = row[i];

        if action != legal[i] {
            return Err(ModelError::InvalidActionForDecision);
        }

        if !prob.is_finite() {
            return Err(ModelError::NonFiniteProbability);
        }

        if prob < 0.0 || prob > 1.0 {
            return Err(ModelError::ProbabilityOutOfRange);
        }

        sum += prob;
    }

    if (sum - 1.0).abs() > 1e-9 {
        return Err(ModelError::RowDoesNotSumToOne);
    }

    Ok(())
}
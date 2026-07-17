use crate::action::{
    buckets, legal_actions, validate_probability, validate_sum, Action, ActionDistribution, ActionLikelihoodModel,
    ActionModelMetadata, BucketedActionModel, DecisionKind, ModelError, ROW_COUNT,
};
use rf_core::{features::ModelBucket, Street};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ActionModelFile {
    metadata: ActionModelMetadata,
    rows: Vec<ActionModelRow>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ActionModelRow {
    street: String,
    decision: String,
    bucket: String,
    probs: BTreeMap<String, f64>,
}

impl BucketedActionModel {
    pub fn from_json_str(text: &str) -> Result<Self, ModelError> {
        let file: ActionModelFile = serde_json::from_str(text).map_err(|err| ModelError::Json(err.to_string()))?;
        let mut rows = Vec::with_capacity(file.rows.len());

        for row in file.rows {
            let street = parse_street(&row.street)?;
            let decision = parse_decision(&row.decision)?;
            let bucket = parse_bucket(&row.bucket)?;
            let dist = distribution_from_probs(decision, row.probs)?;
            rows.push((street, decision, bucket, dist));
        }

        Self::from_rows(file.metadata, rows)
    }

    pub fn to_json_string(&self) -> Result<String, ModelError> {
        let mut rows = Vec::with_capacity(ROW_COUNT);

        for street in [Street::Flop, Street::Turn, Street::River] {
            for decision in [DecisionKind::Unopened, DecisionKind::FacingBet] {
                for bucket in buckets() {
                    let mut probs = BTreeMap::new();
                    for (action, probability) in self.distribution(street, decision, bucket)? {
                        probs.insert(action_name(action).to_string(), probability);
                    }

                    rows.push(ActionModelRow {
                        street: street_name(street).to_string(),
                        decision: decision_name(decision).to_string(),
                        bucket: bucket_name(bucket).to_string(),
                        probs,
                    });
                }
            }
        }

        let file = ActionModelFile {
            metadata: self.metadata().clone(),
            rows,
        };

        serde_json::to_string_pretty(&file).map_err(|err| ModelError::Json(err.to_string()))
    }
}

fn distribution_from_probs(
    decision: DecisionKind,
    probs: BTreeMap<String, f64>,
) -> Result<ActionDistribution, ModelError> {
    let legal = legal_actions(decision);
    let mut out = [(legal[0], 0.0), (legal[1], 0.0), (legal[2], 0.0)];
    let mut seen = [false; 3];
    let mut sum = 0.0;

    for (name, prob) in probs {
        let action = parse_action(&name)?;
        let idx = legal
            .iter()
            .position(|&candidate| candidate == action)
            .ok_or_else(|| ModelError::InvalidRow(format!("{action:?} is not legal for {decision:?}")))?;

        if seen[idx] {
            return Err(ModelError::InvalidRow(format!("duplicate action: {action:?}")));
        }

        validate_probability(action, prob)?;
        out[idx] = (action, prob);
        seen[idx] = true;
        sum += prob;
    }

    for i in 0..3 {
        if !seen[i] {
            return Err(ModelError::InvalidRow(format!(
                "missing action {:?} for {decision:?}",
                legal[i]
            )));
        }
    }

    validate_sum(sum)?;
    Ok(out)
}

fn parse_street(text: &str) -> Result<Street, ModelError> {
    match text {
        "flop" | "Flop" => Ok(Street::Flop),
        "turn" | "Turn" => Ok(Street::Turn),
        "river" | "River" => Ok(Street::River),
        "preflop" | "Preflop" => Err(ModelError::UnsupportedStreet),
        other => Err(ModelError::InvalidRow(format!("unknown street: {other}"))),
    }
}

fn parse_decision(text: &str) -> Result<DecisionKind, ModelError> {
    match text {
        "unopened" | "Unopened" => Ok(DecisionKind::Unopened),
        "facing_bet" | "FacingBet" => Ok(DecisionKind::FacingBet),
        other => Err(ModelError::InvalidRow(format!("unknown decision: {other}"))),
    }
}

fn parse_bucket(text: &str) -> Result<ModelBucket, ModelError> {
    match text {
        "Air" | "air" => Ok(ModelBucket::Air),
        "Draw" | "draw" => Ok(ModelBucket::Draw),
        "OnePair" | "one_pair" => Ok(ModelBucket::OnePair),
        "TwoPairOrTrips" | "two_pair_or_trips" => Ok(ModelBucket::TwoPairOrTrips),
        "StraightOrFlush" | "straight_or_flush" => Ok(ModelBucket::StraightOrFlush),
        "FullHousePlus" | "full_house_plus" => Ok(ModelBucket::FullHousePlus),
        other => Err(ModelError::InvalidRow(format!("unknown bucket: {other}"))),
    }
}

fn parse_action(text: &str) -> Result<Action, ModelError> {
    match text {
        "Check" | "check" => Ok(Action::Check),
        "SmallBet" | "small_bet" | "bet_small" => Ok(Action::SmallBet),
        "LargeBet" | "large_bet" | "bet_large" => Ok(Action::LargeBet),
        "Fold" | "fold" => Ok(Action::Fold),
        "Call" | "call" => Ok(Action::Call),
        "Raise" | "raise" => Ok(Action::Raise),
        other => Err(ModelError::InvalidRow(format!("unknown action: {other}"))),
    }
}

fn street_name(street: Street) -> &'static str {
    match street {
        Street::Flop => "flop",
        Street::Turn => "turn",
        Street::River => "river",
        Street::Preflop => "preflop",
    }
}

fn decision_name(decision: DecisionKind) -> &'static str {
    match decision {
        DecisionKind::Unopened => "unopened",
        DecisionKind::FacingBet => "facing_bet",
    }
}

fn bucket_name(bucket: ModelBucket) -> &'static str {
    match bucket {
        ModelBucket::Air => "Air",
        ModelBucket::Draw => "Draw",
        ModelBucket::OnePair => "OnePair",
        ModelBucket::TwoPairOrTrips => "TwoPairOrTrips",
        ModelBucket::StraightOrFlush => "StraightOrFlush",
        ModelBucket::FullHousePlus => "FullHousePlus",
    }
}

fn action_name(action: Action) -> &'static str {
    match action {
        Action::Check => "Check",
        Action::SmallBet => "SmallBet",
        Action::LargeBet => "LargeBet",
        Action::Fold => "Fold",
        Action::Call => "Call",
        Action::Raise => "Raise",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::toy_model;

    #[test]
    fn rejects_extra_unknown_action() {
        let json = r#"{
            "metadata": {
                "id": "bad",
                "version": "0.1.0",
                "description": "bad model",
                "calibration": "illustrative"
            },
            "rows": [
                {
                    "street": "flop",
                    "decision": "unopened",
                    "bucket": "Air",
                    "probs": {
                        "Check": 0.6,
                        "SmallBet": 0.3,
                        "LargeBet": 0.1,
                        "Jam": 0.0
                    }
                }
            ]
        }"#;

        assert!(matches!(
            BucketedActionModel::from_json_str(json),
            Err(ModelError::InvalidRow(_))
        ));
    }

    #[test]
    fn round_trips_model_through_serde_without_semantic_change() {
        let model = toy_model().unwrap();
        let json = model.to_json_string().unwrap();
        let loaded = BucketedActionModel::from_json_str(&json).unwrap();

        assert_eq!(loaded.metadata(), model.metadata());

        for street in [Street::Flop, Street::Turn, Street::River] {
            for decision in [DecisionKind::Unopened, DecisionKind::FacingBet] {
                for bucket in buckets() {
                    assert_eq!(
                        loaded.distribution(street, decision, bucket).unwrap(),
                        model.distribution(street, decision, bucket).unwrap()
                    );
                }
            }
        }
    }
}

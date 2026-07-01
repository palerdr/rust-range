use crate::distribution::RangeDistribution;
use rf_core::{
    features::{extract_features, ModelBucket},
    Board, ComboId, ComboWeights,
};

#[derive(Debug, Clone, PartialEq)]
pub enum MetricError {
    SupportMismatch { combo: ComboId, q: f64, p: f64 },
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

pub fn entropy(p: &RangeDistribution) -> f64 {
    let mut total = 0.0f64;
    for (_, prob) in p.iter() {
        if prob > 0.0 {
            total += prob * prob.log2();
        }
    }
    -total
}

pub fn total_variation(p: &RangeDistribution, q: &RangeDistribution) -> f64 {
    let mut tv = 0.0f64;
    for ((_, pi), (_, qi)) in p.iter().zip(q.iter()) {
        tv += (pi - qi).abs();
    }
    (1.0 / 2.0) * tv
}

pub fn kl_div(q: &RangeDistribution, p: &RangeDistribution) -> Result<f64, MetricError> {
    kl_bits(q, p)
}

pub fn kl_bits(q: &RangeDistribution, p: &RangeDistribution) -> Result<f64, MetricError> {
    let mut dkl = 0.0f64;
    for (id, q_prob) in q.iter() {
        let p_prob = p.probability(id);

        if q_prob == 0.0 && p_prob == 0.0 {
            continue;
        }

        if q_prob == 0.0 {
            continue;
        }

        if p_prob == 0.0 {
            return Err(MetricError::SupportMismatch {
                combo: id,
                q: q_prob,
                p: p_prob,
            });
        }

        dkl += q_prob * (q_prob / p_prob).log2();
    }
    Ok(dkl)
}

pub fn effective_hand_count(p: &RangeDistribution) -> f64 {
    2f64.powf(entropy(p))
}

pub fn top_n(p: &RangeDistribution, n: usize) -> Option<RangeDistribution> {
    if n == 0 {
        return None;
    }
    let mut items: Vec<(ComboId, f64)> = p.iter().collect();

    items.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.index().cmp(&b.0.index()))
    });

    let mut out = ComboWeights::zeros();
    for (id, prob) in items.into_iter().take(n).filter(|(_, prob)| *prob > 0.0) {
        out.as_mut_slice()[id.index()] = prob;
    }

    match RangeDistribution::from_weights(out) {
        Ok(dist) => Some(dist),
        Err(_) => None,
    }
}

pub fn bucket_masses(p: &RangeDistribution, board: &Board) -> Result<[f64; 6], rf_core::RfError> {
    let mut masses = [0.0f64; 6];

    let mut items: Vec<(ComboId, f64)> = p.iter().collect();
    items.sort_by_key(|(id, _)| id.index());

    for (id, prob) in items {
        if prob == 0.0 {
            continue;
        }

        let hole = id.hole_cards();
        if board.mask().intersects(hole.mask()) {
            continue;
        }
        let features = extract_features(hole, board)?;
        let idx = bucket_index(features.bucket);
        masses[idx] += prob;
    }
    Ok(masses)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rf_core::ComboWeights;

    fn uniform_twoway(id_a: usize, id_b: usize) -> RangeDistribution {
        let mut weights = ComboWeights::zeros();
        let slice = weights.as_mut_slice();
        slice[id_a] = 1.0;
        slice[id_b] = 1.0;
        RangeDistribution::from_weights(weights).unwrap()
    }

    fn board_three_of_clubs() -> Board {
        Board::parse("As Ks Qh").unwrap()
    }

    #[test]
    fn entropy_treats_zero_as_zero() {
        let dist = uniform_twoway(0, 1);

        assert!((entropy(&dist) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn effective_hand_count_matches_entropy() {
        let dist = uniform_twoway(0, 1);

        assert!((effective_hand_count(&dist) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn total_variation_self_is_zero() {
        let dist = uniform_twoway(0, 1);
        assert_eq!(total_variation(&dist, &dist), 0.0);
    }

    #[test]
    fn kl_bits_self_is_zero() {
        let dist = uniform_twoway(0, 1);
        assert_eq!(kl_bits(&dist, &dist).unwrap(), 0.0);
    }

    #[test]
    fn kl_bits_support_mismatch() {
        let p = {
            let mut w = ComboWeights::zeros();
            w.as_mut_slice()[0] = 1.0;
            RangeDistribution::from_weights(w).unwrap()
        };
        let q = {
            let mut w = ComboWeights::zeros();
            w.as_mut_slice()[1] = 1.0;
            RangeDistribution::from_weights(w).unwrap()
        };

        let err = kl_bits(&q, &p).unwrap_err();
        assert!(matches!(
            err,
            MetricError::SupportMismatch { combo, q: 1.0, p: 0.0 } if combo.index() == 1
        ));
    }

    #[test]
    fn top_n_is_deterministic_on_ties() {
        let mut w = ComboWeights::zeros();
        for i in 0..3 {
            w.as_mut_slice()[i] = 1.0;
        }
        let dist = RangeDistribution::from_weights(w).unwrap();

        let top = top_n(&dist, 2).unwrap();
        let ids: Vec<usize> = top.iter().filter(|(_, p)| *p > 0.0).map(|(id, _)| id.index()).collect();
        let probs: Vec<f64> = top.iter().filter(|(_, p)| *p > 0.0).map(|(_, p)| p).collect();

        assert_eq!(ids, vec![0, 1]);
        assert_eq!(probs, vec![0.5, 0.5]);
    }

    #[test]
    fn bucket_masses_deterministic_output_order() {
        let board = board_three_of_clubs();
        let dist = RangeDistribution::uniform_conditioned(board.mask()).unwrap();

        let mut masses = bucket_masses(&dist, &board).unwrap();
        let mut expected = [0.0f64; 6];

        for (id, prob) in dist.iter() {
            if prob == 0.0 {
                continue;
            }
            if board.mask().intersects(id.hole_cards().mask()) {
                continue;
            }
            let features = rf_core::features::extract_features(id.hole_cards(), &board).unwrap();
            expected[bucket_index(features.bucket)] += prob;
        }

        assert_eq!(masses.len(), expected.len());
        for i in 0..6 {
            masses[i] = (masses[i] * 1e12).round() / 1e12;
            expected[i] = (expected[i] * 1e12).round() / 1e12;
        }
        assert_eq!(masses, expected);
    }

    #[test]
    fn bucket_masses_is_consistent_when_recomputed() {
        let board = board_three_of_clubs();
        let dist = {
            let mut w = ComboWeights::zeros();
            for i in [0usize, 5, 10, 20, 30, 40].iter() {
                w.as_mut_slice()[*i] = 1.0;
            }
            RangeDistribution::from_weights(w).unwrap()
        };

        let first = bucket_masses(&dist, &board).unwrap();
        let second = bucket_masses(&dist, &board).unwrap();
        assert_eq!(first, second);
    }
}

//! Normalized probability tables over the fixed 1,326 two-card combinations.

use rf_core::{all_combos, CardMask, ComboId, ComboWeights, NUM_HOLE_COMBOS};

pub const NUM_COMBOS: usize = NUM_HOLE_COMBOS;
const NORMALIZATION_EPS: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum DistributionError {
    WrongLength { expected: usize, got: usize },
    NegativeWeight { index: usize, value: f64 },
    NonFiniteWeight { index: usize, value: f64 },
    AllZeroWeights,
    NonFiniteTotal,
    NormalizationFailed { sum: f64 },
    NoUnblockedCombos,
}
impl std::fmt::Display for DistributionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongLength { expected, got } => write!(f, "expected {expected} weights, got {got}"),
            Self::NegativeWeight { index, value } => write!(f, "weight {index} is negative: {value}"),
            Self::NonFiniteWeight { index, value } => write!(f, "weight {index} is not finite: {value}"),
            Self::AllZeroWeights => write!(f, "all weights are zero"),
            Self::NonFiniteTotal => write!(f, "weight total is not finite"),
            Self::NormalizationFailed { sum } => write!(f, "normalization failed; sum is {sum}"),
            Self::NoUnblockedCombos => write!(f, "no unblocked combinations remain"),
        }
    }
}

impl std::error::Error for DistributionError {}

/// A finite, nonnegative probability distribution indexed by [`ComboId`].
#[derive(Clone, Debug, PartialEq)]
pub struct RangeDistribution {
    probs: Vec<f64>,
}

impl RangeDistribution {
    pub fn from_weights(weights: ComboWeights) -> Result<Self, DistributionError> {
        let check_weight_values = |weights: &[f64]| -> Result<(), DistributionError> {
            if weights.len() != NUM_COMBOS {
                return Err(DistributionError::WrongLength {
                    expected: NUM_COMBOS,
                    got: weights.len(),
                });
            }

            for (idx, &w) in weights.iter().enumerate() {
                if !w.is_finite() {
                    return Err(DistributionError::NonFiniteWeight { index: idx, value: w });
                }
                if w < 0.0 {
                    return Err(DistributionError::NegativeWeight { index: idx, value: w });
                }
            }

            Ok(())
        };

        let normalize = |weights: Vec<f64>| -> Result<Vec<f64>, DistributionError> {
            check_weight_values(&weights)?;
            let total = stable_sum(&weights);

            if !total.is_finite() {
                return Err(DistributionError::NonFiniteTotal);
            }

            if total == 0.0 {
                return Err(DistributionError::AllZeroWeights);
            }

            let scale = 1.0 / total;
            let probs: Vec<f64> = weights.into_iter().map(|w| w * scale).collect();

            let final_sum = stable_sum(&probs);
            if !final_sum.is_finite() {
                return Err(DistributionError::NonFiniteTotal);
            }
            if (final_sum - 1.0).abs() >= NORMALIZATION_EPS {
                return Err(DistributionError::NormalizationFailed { sum: final_sum });
            }

            Ok(probs)
        };

        let probs = normalize(weights.as_slice().to_vec())?;
        Ok(RangeDistribution { probs })
    }

    pub fn uniform_conditioned(dead: CardMask) -> Result<Self, DistributionError> {
        let mut out = ComboWeights::zeros();

        for (id, cards) in all_combos() {
            if !dead.intersects(cards.mask()) {
                out.as_mut_slice()[id.index()] = 1.0;
            }
        }

        match RangeDistribution::from_weights(out) {
            Ok(dist) => Ok(dist),
            Err(DistributionError::AllZeroWeights) => Err(DistributionError::NoUnblockedCombos),
            Err(err) => Err(err),
        }
    }

    pub fn condition_on_dead(self, dead: CardMask) -> Result<Self, DistributionError> {
        let mut updated = self;

        for (id, cards) in all_combos() {
            if dead.intersects(cards.mask()) {
                updated.probs[id.index()] = 0.0;
            }
        }

        let total = stable_sum(&updated.probs);
        if !total.is_finite() {
            return Err(DistributionError::NonFiniteTotal);
        }
        if total == 0.0 {
            return Err(DistributionError::NoUnblockedCombos);
        }

        let scale = 1.0 / total;
        for p in &mut updated.probs {
            *p *= scale;
        }

        let final_sum = stable_sum(&updated.probs);
        if !final_sum.is_finite() {
            return Err(DistributionError::NonFiniteTotal);
        }
        if (final_sum - 1.0).abs() >= NORMALIZATION_EPS {
            return Err(DistributionError::NormalizationFailed { sum: final_sum });
        }

        Ok(updated)
    }
    pub fn probability(&self, id: ComboId) -> f64 {
        let idx = id.index();
        self.probs[idx]
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (ComboId, f64)> + '_ {
        self.probs.iter().copied().enumerate().map(|(i, p)| {
            let id = ComboId::from_raw(i as u16).unwrap();
            (id, p)
        })
    }

    pub fn support_size(&self) -> usize {
        self.probs.iter().filter(|&&p| p > 0.0).count()
    }
}

/// Sum values with compensation to reduce floating-point accumulation error.
pub fn stable_sum(xs: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;

    for &x in xs {
        let y = x - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    sum
}

pub fn validate_weights(weights: &[f64]) -> Result<(), DistributionError> {
    if weights.len() != NUM_COMBOS {
        return Err(DistributionError::WrongLength {
            expected: NUM_COMBOS,
            got: weights.len(),
        });
    }

    for (idx, &w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(DistributionError::NonFiniteWeight { index: idx, value: w });
        }
        if w < 0.0 {
            return Err(DistributionError::NegativeWeight { index: idx, value: w });
        }
    }

    let total = stable_sum(weights);

    if !total.is_finite() {
        return Err(DistributionError::NonFiniteTotal);
    }

    if total == 0.0 {
        return Err(DistributionError::AllZeroWeights);
    }

    let is_close = (total - 1.0).abs() < NORMALIZATION_EPS;

    if !is_close {
        return Err(DistributionError::NormalizationFailed { sum: total });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rf_core::Card;

    fn build_uniform_vec(v: f64) -> ComboWeights {
        let mut out = ComboWeights::zeros();
        for w in out.as_mut_slice().iter_mut() {
            *w = v;
        }
        out
    }

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *seed
    }

    fn pseudo_weights(seed: &mut u64, plus_one: bool) -> ComboWeights {
        let mut out = ComboWeights::zeros();
        let slice = out.as_mut_slice();
        for x in slice.iter_mut() {
            let sample = lcg(seed) >> 11;
            let v = (sample as f64) / ((1u64 << 53) as f64);
            *x = if plus_one { v + 1.0 } else { v };
        }
        out
    }

    #[test]
    fn validate_weights_wrong_length() {
        let err = validate_weights(&[1.0; NUM_COMBOS - 1]);
        assert_eq!(
            err,
            Err(DistributionError::WrongLength {
                expected: NUM_COMBOS,
                got: NUM_COMBOS - 1
            })
        );
    }

    #[test]
    fn validate_weights_negative() {
        let mut weights = vec![1.0; NUM_COMBOS];
        weights[7] = -0.25;
        assert_eq!(
            validate_weights(&weights),
            Err(DistributionError::NegativeWeight { index: 7, value: -0.25 })
        );
    }

    #[test]
    fn validate_weights_nan() {
        let mut weights = vec![1.0; NUM_COMBOS];
        weights[13] = f64::NAN;
        assert!(matches!(
            validate_weights(&weights),
            Err(DistributionError::NonFiniteWeight {
                index: 13,
                value
            }) if !value.is_finite()
        ));
    }

    #[test]
    fn validate_weights_infinity() {
        let mut weights = vec![1.0; NUM_COMBOS];
        weights[13] = f64::INFINITY;
        assert!(matches!(
            validate_weights(&weights),
            Err(DistributionError::NonFiniteWeight {
                index: 13,
                value
            }) if !value.is_finite()
        ));
    }

    #[test]
    fn validate_weights_all_zero() {
        let weights = vec![0.0; NUM_COMBOS];
        assert_eq!(validate_weights(&weights), Err(DistributionError::AllZeroWeights));
    }

    #[test]
    fn from_weights_rejects_negative() {
        let mut weights = build_uniform_vec(1.0);
        weights.as_mut_slice()[9] = -0.25;
        let err = match RangeDistribution::from_weights(weights) {
            Ok(_) => panic!("expected rejection for negative weight"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            DistributionError::NegativeWeight { index: 9, value: -0.25 }
        ));
    }

    #[test]
    fn from_weights_rejects_nan() {
        let mut weights = build_uniform_vec(1.0);
        weights.as_mut_slice()[13] = f64::NAN;
        let err = match RangeDistribution::from_weights(weights) {
            Ok(_) => panic!("expected rejection for NaN weight"),
            Err(err) => err,
        };
        assert!(matches!(err, DistributionError::NonFiniteWeight { index: 13, value } if !value.is_finite()));
    }

    #[test]
    fn from_weights_rejects_infinity() {
        let mut weights = build_uniform_vec(1.0);
        weights.as_mut_slice()[13] = f64::INFINITY;
        let err = match RangeDistribution::from_weights(weights) {
            Ok(_) => panic!("expected rejection for infinite weight"),
            Err(err) => err,
        };
        assert!(matches!(err, DistributionError::NonFiniteWeight { index: 13, value } if !value.is_finite()));
    }

    #[test]
    fn from_weights_rejects_all_zero() {
        let weights = ComboWeights::zeros();
        let err = match RangeDistribution::from_weights(weights) {
            Ok(_) => panic!("expected rejection for all-zero weights"),
            Err(err) => err,
        };
        assert!(matches!(err, DistributionError::AllZeroWeights));
    }

    #[test]
    fn from_weights_successful_normalization() {
        let weights = build_uniform_vec(2.0);
        let dist = RangeDistribution::from_weights(weights).unwrap();

        assert_eq!(dist.support_size(), NUM_COMBOS);
        let expected = 1.0f64 / NUM_COMBOS as f64;
        let total = dist.iter().map(|(_, p)| p).sum::<f64>();
        assert!((total - 1.0).abs() < NORMALIZATION_EPS);
        assert!(dist.iter().all(|(_, p)| (p - expected).abs() < 1e-12));
    }

    #[test]
    fn from_weights_normalizes_to_one() {
        let dist = RangeDistribution::from_weights(build_uniform_vec(17.0)).unwrap();
        let total = dist.iter().map(|(_, p)| p).sum::<f64>();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn uniform_conditioned_empty_dead_is_identity() {
        let baseline = RangeDistribution::from_weights(ComboWeights::from_uniform()).unwrap();
        let empty_dead = RangeDistribution::uniform_conditioned(CardMask::EMPTY).unwrap();

        assert_eq!(baseline.support_size(), empty_dead.support_size());
        let baseline_sum = baseline.iter().map(|(_, p)| p).sum::<f64>();
        let empty_sum = empty_dead.iter().map(|(_, p)| p).sum::<f64>();
        assert!((baseline_sum - 1.0).abs() < NORMALIZATION_EPS);
        assert!((empty_sum - 1.0).abs() < NORMALIZATION_EPS);

        for (id, p_empty) in empty_dead.iter() {
            let p_base = baseline.probability(id);
            assert!((p_empty - p_base).abs() < 1e-12);
        }
    }

    #[test]
    fn uniform_conditioned_one_dead_card_removes_51_combos() {
        let dead = CardMask::from_cards([Card::from_parts(12, 0).unwrap()]);
        let dist = RangeDistribution::uniform_conditioned(dead).unwrap();
        assert_eq!(dist.support_size(), NUM_COMBOS - 51);
    }

    #[test]
    fn uniform_conditioned_two_dead_cards_leaves_1225_combos() {
        let dead = CardMask::from_cards([Card::from_parts(12, 0).unwrap(), Card::from_parts(11, 1).unwrap()]);
        let dist = RangeDistribution::uniform_conditioned(dead).unwrap();
        assert_eq!(dist.support_size(), 1225);
    }

    #[test]
    fn condition_on_dead_preserves_sum_to_one() {
        let dist = RangeDistribution::from_weights(build_uniform_vec(1.0)).unwrap();
        let dead = CardMask::from_cards([Card::from_parts(7, 2).unwrap()]);
        let conditioned = dist.condition_on_dead(dead).unwrap();
        let total = conditioned.iter().map(|(_, p)| p).sum::<f64>();

        assert_eq!(conditioned.support_size(), NUM_COMBOS - 51);
        assert!((total - 1.0).abs() < NORMALIZATION_EPS);
    }

    #[test]
    fn condition_on_dead_zeroes_dead_combos() {
        let dist = RangeDistribution::from_weights(build_uniform_vec(1.0)).unwrap();
        let dead = CardMask::from_cards([Card::from_parts(12, 0).unwrap(), Card::from_parts(11, 1).unwrap()]);
        let conditioned = dist.condition_on_dead(dead).unwrap();

        let mut intersected = 0usize;
        let mut zeroed = 0usize;

        for (id, combo) in all_combos() {
            if dead.intersects(combo.mask()) {
                intersected += 1;
                assert_eq!(conditioned.probability(id), 0.0);
            }
            if conditioned.probability(id) == 0.0 {
                zeroed += 1;
            }
        }

        assert_eq!(intersected, zeroed);
    }

    #[test]
    fn condition_on_dead_fails_when_no_support_survives() {
        let dist = RangeDistribution::from_weights(build_uniform_vec(1.0)).unwrap();
        let all_dead = CardMask::from_bits((1u64 << 52) - 1);
        assert!(matches!(
            dist.condition_on_dead(all_dead),
            Err(DistributionError::NoUnblockedCombos)
        ));
    }

    #[test]
    fn iter_returns_exactly_1326_pairs() {
        let dist = RangeDistribution::from_weights(build_uniform_vec(1.0)).unwrap();
        let items: Vec<_> = dist.iter().collect();
        assert_eq!(items.len(), NUM_COMBOS);
        for (expected, (id, _)) in items.into_iter().enumerate() {
            assert_eq!(id.index(), expected);
        }
    }

    #[test]
    fn probability_can_be_read_for_iter_ids() {
        let dist = RangeDistribution::from_weights(build_uniform_vec(1.0)).unwrap();
        for (id, prob) in dist.iter() {
            assert_eq!(dist.probability(id), prob);
            assert!(id.index() < NUM_COMBOS);
        }
    }

    #[test]
    fn support_size_counts_positive() {
        let mut w = build_uniform_vec(1.0);
        w.as_mut_slice()[0] = 0.0;
        w.as_mut_slice()[1] = 0.0;
        w.as_mut_slice()[2] = 0.0;
        let dist = RangeDistribution::from_weights(w).unwrap();
        assert_eq!(dist.support_size(), NUM_COMBOS - 3);
    }

    #[test]
    fn generated_distributions_are_finite_nonnegative_normalized() {
        let mut seed = 0x1234_5678_9abc_def0_u64;
        for _ in 0..16 {
            let weights = pseudo_weights(&mut seed, true);
            let dist = RangeDistribution::from_weights(weights).unwrap();
            let total = dist.iter().map(|(_, p)| p).sum::<f64>();
            assert!((total - 1.0).abs() < NORMALIZATION_EPS);
            assert!(total.is_finite());
            for (_, prob) in dist.iter() {
                assert!(prob >= 0.0);
                assert!(!prob.is_nan());
            }
        }
    }

    #[test]
    fn generated_distributions_after_conditioning_are_finite_nonnegative_normalized() {
        let mut seed = 0xAABBCCDDEEFF0011_u64;
        let mut dist = RangeDistribution::from_weights(pseudo_weights(&mut seed, true)).unwrap();
        let dead = CardMask::from_cards([Card::from_parts(12, 0).unwrap()]);
        dist = dist.condition_on_dead(dead).unwrap();

        let total = dist.iter().map(|(_, p)| p).sum::<f64>();
        assert!((total - 1.0).abs() < NORMALIZATION_EPS);
        for (_, prob) in dist.iter() {
            assert!(prob.is_finite());
            assert!(prob >= 0.0);
        }
    }
}

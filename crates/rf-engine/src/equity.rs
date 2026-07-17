//! Exact postflop and seeded Monte Carlo heads-up equity.

use rand::{Rng, SeedableRng};
use rf_core::{evaluate_best, unordered_pairs, Board, Card, CardMask, HoleCards, RfError, Street};

use crate::distribution::RangeDistribution;

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct EquityResult {
    pub win_probability: f64,
    pub tie_probability: f64,
    pub loss_probability: f64,
    pub equity: f64,
    pub evaluated_cases: u64,
    pub method: String,
    pub samples: Option<u64>,
    pub seed: Option<u64>,
    pub standard_error: Option<f64>,
    pub confidence_interval_95: Option<[f64; 2]>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EquityError {
    Core(RfError),
    UnsupportedPreflopExact,
    NoLegalVillainHands,
    NoLegalRunouts,
    InvalidSampleCount,
}

impl std::fmt::Display for EquityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core(err) => write!(f, "{err}"),
            Self::UnsupportedPreflopExact => write!(f, "exact equity is supported only postflop"),
            Self::NoLegalVillainHands => write!(f, "no legal villain hands remain"),
            Self::NoLegalRunouts => write!(f, "no legal board runouts remain"),
            Self::InvalidSampleCount => write!(f, "Monte Carlo sample count must be greater than zero"),
        }
    }
}
impl std::error::Error for EquityError {}
impl From<RfError> for EquityError {
    fn from(value: RfError) -> Self {
        Self::Core(value)
    }
}

/// Enumerate every legal villain hand and future board runout.
pub fn exact_equity(hero: HoleCards, board: Board, villain: &RangeDistribution) -> Result<EquityResult, EquityError> {
    if board.street() == Street::Preflop {
        return Err(EquityError::UnsupportedPreflopExact);
    }
    let dead = hero.mask().union(board.mask());
    let mut total_weight = 0.0;
    let mut wins = 0.0;
    let mut ties = 0.0;
    let mut losses = 0.0;
    let mut evaluated_cases = 0u64;

    for (id, probability) in villain.iter() {
        if probability == 0.0 {
            continue;
        }
        let villain_hand = id.hole_cards();
        if dead.intersects(villain_hand.mask()) {
            continue;
        }
        let runouts = legal_runouts(dead.union(villain_hand.mask()), board.len());
        if runouts.is_empty() {
            return Err(EquityError::NoLegalRunouts);
        }
        let per_runout = probability / runouts.len() as f64;
        total_weight += probability;
        evaluated_cases += runouts.len() as u64;
        for runout in runouts {
            let runout_mask = CardMask::from_cards(runout.iter().copied());
            let final_board = board.mask().union(runout_mask);
            let hero_rank = evaluate_best(hero.mask().union(final_board))?;
            let villain_rank = evaluate_best(villain_hand.mask().union(final_board))?;
            if hero_rank > villain_rank {
                wins += per_runout;
            } else if hero_rank == villain_rank {
                ties += per_runout;
            } else {
                losses += per_runout;
            }
        }
    }

    if total_weight == 0.0 {
        return Err(EquityError::NoLegalVillainHands);
    }
    let scale = 1.0 / total_weight;
    let (win_probability, tie_probability, loss_probability) = (wins * scale, ties * scale, losses * scale);
    Ok(EquityResult {
        win_probability,
        tie_probability,
        loss_probability,
        equity: win_probability + 0.5 * tie_probability,
        evaluated_cases,
        method: "exact".to_string(),
        samples: None,
        seed: None,
        standard_error: None,
        confidence_interval_95: None,
    })
}

/// Estimate equity with reproducible sampling from the posterior and deck.
pub fn monte_carlo_equity(
    hero: HoleCards,
    board: Board,
    villain: &RangeDistribution,
    samples: u64,
    seed: u64,
) -> Result<EquityResult, EquityError> {
    if samples == 0 {
        return Err(EquityError::InvalidSampleCount);
    }
    let dead = hero.mask().union(board.mask());
    let mut support = Vec::new();
    let mut cumulative = Vec::new();
    let mut total = 0.0;
    for (id, probability) in villain.iter() {
        if probability > 0.0 && !dead.intersects(id.hole_cards().mask()) {
            total += probability;
            support.push(id.hole_cards());
            cumulative.push(total);
        }
    }
    if support.is_empty() || total == 0.0 {
        return Err(EquityError::NoLegalVillainHands);
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut wins = 0u64;
    let mut ties = 0u64;
    let mut losses = 0u64;
    for sample_index in 0..samples {
        let target = rng.gen::<f64>() * total;
        let combo_index = cumulative.partition_point(|&value| value < target);
        let villain_hand = support[combo_index.min(support.len() - 1)];
        let mut cards: Vec<Card> = rf_core::remaining_cards(dead.union(villain_hand.mask())).collect();
        let needed = 5usize - board.len() as usize;
        if cards.len() < needed {
            return Err(EquityError::NoLegalRunouts);
        }
        let mut runout = Vec::with_capacity(needed);
        for _ in 0..needed {
            let index = rng.gen_range(0..cards.len());
            runout.push(cards.swap_remove(index));
        }
        let final_board = board.mask().union(CardMask::from_cards(runout));
        let hero_rank = evaluate_best(hero.mask().union(final_board))?;
        let villain_rank = evaluate_best(villain_hand.mask().union(final_board))?;
        let payoff = if hero_rank > villain_rank {
            wins += 1;
            1.0
        } else if hero_rank == villain_rank {
            ties += 1;
            0.5
        } else {
            losses += 1;
            0.0
        };
        let n = (sample_index + 1) as f64;
        let delta = payoff - mean;
        mean += delta / n;
        m2 += delta * (payoff - mean);
    }

    let variance = if samples > 1 { m2 / (samples - 1) as f64 } else { 0.0 };
    let standard_error = (variance / samples as f64).sqrt();
    let interval = [
        (mean - 1.96 * standard_error).clamp(0.0, 1.0),
        (mean + 1.96 * standard_error).clamp(0.0, 1.0),
    ];
    Ok(EquityResult {
        win_probability: wins as f64 / samples as f64,
        tie_probability: ties as f64 / samples as f64,
        loss_probability: losses as f64 / samples as f64,
        equity: mean,
        evaluated_cases: samples,
        method: "monte_carlo".to_string(),
        samples: Some(samples),
        seed: Some(seed),
        standard_error: Some(standard_error),
        confidence_interval_95: Some(interval),
    })
}

fn legal_runouts(dead: CardMask, board_len: u8) -> Vec<Vec<Card>> {
    let cards: Vec<Card> = rf_core::remaining_cards(dead).collect();
    match board_len {
        3 => unordered_pairs(&cards).map(|pair| pair.to_vec()).collect(),
        4 => cards.into_iter().map(|card| vec![card]).collect(),
        5 => vec![Vec::new()],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rf_core::{ComboId, ComboWeights};

    fn point_mass(hero: &str, villain: &str, board: &str) -> (HoleCards, HoleCards, Board, RangeDistribution) {
        let hero_cards = HoleCards::from_strs(&hero[0..2], &hero[2..4]).unwrap();
        let villain_cards = HoleCards::from_strs(&villain[0..2], &villain[2..4]).unwrap();
        let board_cards = Board::parse(board).unwrap();
        let mut weights = ComboWeights::zeros();
        weights.as_mut_slice()[ComboId::from_hole_cards(villain_cards).index()] = 1.0;
        (
            hero_cards,
            villain_cards,
            board_cards,
            RangeDistribution::from_weights(weights).unwrap(),
        )
    }

    #[test]
    fn river_point_mass_matches_direct_comparison() {
        let (hero, villain, board, range) = point_mass("AsKs", "2c3d", "Qs Js Ts 9h 8c");
        let result = exact_equity(hero, board, &range).unwrap();
        assert_eq!(result.win_probability, 1.0);
        assert_eq!(result.tie_probability, 0.0);
        assert_eq!(result.equity, 1.0);
        assert_eq!(result.evaluated_cases, 1);
        assert!(
            evaluate_best(hero.mask().union(board.mask())).unwrap()
                > evaluate_best(villain.mask().union(board.mask())).unwrap()
        );
    }

    #[test]
    fn board_forces_a_tie() {
        let (hero, _, board, range) = point_mass("AsKd", "2c3d", "Ah Kh Qh Jh Th");
        let result = exact_equity(hero, board, &range).unwrap();
        assert_eq!(result.tie_probability, 1.0);
        assert_eq!(result.equity, 0.5);
    }

    #[test]
    fn monte_carlo_is_reproducible() {
        let (hero, _, board, range) = point_mass("AsKs", "2c3d", "Qs Js Ts 9h 8c");
        let a = monte_carlo_equity(hero, board, &range, 100, 42).unwrap();
        let b = monte_carlo_equity(hero, board, &range, 100, 42).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.standard_error, Some(0.0));
    }

    #[test]
    fn exact_equity_rejects_preflop() {
        let (hero, _, board, range) = point_mass("AsKs", "2c3d", "");
        assert_eq!(
            exact_equity(hero, board, &range),
            Err(EquityError::UnsupportedPreflopExact)
        );
    }
}

use crate::error::RfError;
use crate::game::CardMask;

const CARD_RANK: [u8; 52] = build_card_rank_table();
const CARD_SUIT: [u8; 52] = build_card_suit_table();
const COUNT_WEIGHT: [u16; 5] = [0, 1, 3, 7, 15];
const STRAIGHT_HIGH_BY_MASK: [u8; 8192] = build_straight_high_table();

const fn build_card_rank_table() -> [u8; 52] {
    let mut out = [0u8; 52];
    let mut i = 0;
    while i < 52 {
        out[i] = (i % 13) as u8;
        i += 1;
    }
    out
}

const fn build_card_suit_table() -> [u8; 52] {
    let mut out = [0u8; 52];
    let mut i = 0;
    while i < 52 {
        out[i] = (i / 13) as u8;
        i += 1;
    }
    out
}

const fn build_straight_high_table() -> [u8; 8192] {
    let mut out = [0u8; 8192];
    let mut mask = 0u16;
    while mask < 8192 {
        let mut straight_high = 0u8;
        if (mask & 0x100F) == 0x100F {
            // Wheel: A-2-3-4-5 -> high-rank index 3 (i.e., 5 high in 2..A indexing).
            straight_high = 4;
        } else {
            let mut high = 12i8;
            while high >= 4 {
                if ((mask >> ((high as u16) - 4)) & 0x1F) == 0x1F {
                    straight_high = (high as u8) + 1;
                    break;
                }
                high -= 1;
            }
        }
        out[mask as usize] = straight_high;
        mask += 1;
    }
    out
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
#[repr(u8)]
pub enum HandCategory {
    HighCard,
    OnePair,
    TwoPair,
    ThreeOfAKind,
    Straight,
    Flush,
    FullHouse,
    FourOfAKind,
    StraightFlush,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct HandRank(u32);

impl HandRank {
    pub(crate) const fn pack(cat: u8, rs: &[u8; 5]) -> Self {
        Self(
            (cat as u32) << 24
                | (rs[0] as u32) << 20
                | (rs[1] as u32) << 16
                | (rs[2] as u32) << 12
                | (rs[3] as u32) << 8
                | (rs[4] as u32) << 4,
        )
    }

    pub fn category(self) -> HandCategory {
        match ((self.0 >> 24) & 0xFF) as u8 {
            0 => HandCategory::HighCard,
            1 => HandCategory::OnePair,
            2 => HandCategory::TwoPair,
            3 => HandCategory::ThreeOfAKind,
            4 => HandCategory::Straight,
            5 => HandCategory::Flush,
            6 => HandCategory::FullHouse,
            7 => HandCategory::FourOfAKind,
            8 => HandCategory::StraightFlush,
            _ => HandCategory::HighCard,
        }
    }

    #[allow(dead_code)]
    pub(crate) const fn bits(self) -> u32 {
        self.0
    }
}

#[inline]
pub fn category_of_rank(rank: HandRank) -> HandCategory {
    rank.category()
}

pub fn category_of_cards(cards: CardMask) -> Result<HandCategory, RfError> {
    Ok(evaluate_best(cards)?.category())
}

/// Evaluate exactly five cards and return a rank where higher value means stronger hand.
///
/// - rank index uses 0..12 as 2..A.
/// - wheel straight (A-2-3-4-5) is encoded as rank 3 (five-high).
pub fn evaluate_five(cards: CardMask) -> Result<HandRank, RfError> {
    if cards.count() != 5 {
        return Err(RfError::EvalError("evaluate_five requires exactly 5 cards".to_string()));
    }

    let mut indices = [0u8; 5];
    let mut bits = cards.bits();
    let mut n = 0usize;
    while bits != 0 {
        let lsb = bits & bits.wrapping_neg();
        indices[n] = lsb.trailing_zeros() as u8;
        bits ^= lsb;
        n += 1;
    }

    Ok(evaluate_five_indices(indices))
}

#[inline(always)]
fn evaluate_five_indices(indices: [u8; 5]) -> HandRank {
    let mut rank_count = [0u8; 13];
    let mut s0 = 0u8;
    let mut s1 = 0u8;
    let mut s2 = 0u8;
    let mut s3 = 0u8;
    let mut rank_mask: u16 = 0;

    for idx in indices {
        let idx = idx as usize;
        let r = CARD_RANK[idx] as usize;
        let s = CARD_SUIT[idx] as usize;

        rank_count[r] += 1;
        match s {
            0 => s0 += 1,
            1 => s1 += 1,
            2 => s2 += 1,
            _ => s3 += 1,
        }
        rank_mask |= 1u16 << r;
    }

    let is_flush = s0 == 5 || s1 == 5 || s2 == 5 || s3 == 5;
    let straight_high_from_rank_mask = |m: u16| -> Option<u8> {
        let value = STRAIGHT_HIGH_BY_MASK[m as usize];
        if value == 0 {
            None
        } else {
            Some(value - 1)
        }
    };
    let straight_hi = straight_high_from_rank_mask(rank_mask);
    let mut sig: u16 = 0;
    for &c in &rank_count {
        sig += COUNT_WEIGHT[c as usize];
    }

    let class_mod = (sig % 15) as u8;
    let mut singles = [0u8; 5];
    let mut pairs = [0u8; 2];
    let mut trips = [0u8; 1];
    let mut quads = [0u8; 1];
    let mut si = 0usize;
    let mut pa = 0usize;
    let mut tr = 0usize;
    let mut qu = 0usize;

    for r in (0..13).rev() {
        match rank_count[r] {
            4 => {
                quads[qu] = r as u8;
                qu += 1;
            }
            3 => {
                trips[tr] = r as u8;
                tr += 1;
            }
            2 => {
                pairs[pa] = r as u8;
                pa += 1;
            }
            1 => {
                singles[si] = r as u8;
                si += 1;
            }
            _ => {}
        }
    }

    let (category, rs) = if is_flush && straight_hi.is_some() {
        (HandCategory::StraightFlush, [straight_hi.unwrap_or(0), 0, 0, 0, 0])
    } else {
        match class_mod {
            1 => {
                let mut v = [0u8; 5];
                v[0] = quads[0];
                v[1] = singles[0];
                (HandCategory::FourOfAKind, v)
            }
            10 => {
                let mut v = [0u8; 5];
                v[0] = trips[0];
                v[1] = pairs[0];
                (HandCategory::FullHouse, v)
            }
            9 => {
                let mut v = [0u8; 5];
                v[0] = trips[0];
                v[1] = singles[0];
                v[2] = singles[1];
                (HandCategory::ThreeOfAKind, v)
            }
            7 => {
                let mut v = [0u8; 5];
                v[0] = pairs[0];
                v[1] = pairs[1];
                v[2] = singles[0];
                (HandCategory::TwoPair, v)
            }
            6 => {
                let mut v = [0u8; 5];
                v[0] = pairs[0];
                v[1] = singles[0];
                v[2] = singles[1];
                v[3] = singles[2];
                (HandCategory::OnePair, v)
            }
            5 => {
                if let Some(straight_rank) = straight_hi {
                    (HandCategory::Straight, [straight_rank, 0, 0, 0, 0])
                } else if is_flush {
                    let mut v = [0u8; 5];
                    v.copy_from_slice(&singles);
                    (HandCategory::Flush, v)
                } else {
                    let mut v = [0u8; 5];
                    v.copy_from_slice(&singles);
                    (HandCategory::HighCard, v)
                }
            }
            _ => {
                let mut v = [0u8; 5];
                v.copy_from_slice(&singles);
                if is_flush {
                    (HandCategory::Flush, v)
                } else if let Some(straight_rank) = straight_hi {
                    (HandCategory::Straight, [straight_rank, 0, 0, 0, 0])
                } else {
                    (HandCategory::HighCard, v)
                }
            }
        }
    };

    HandRank::pack(category as u8, &rs)
}

/// Evaluate five, six, or seven cards by selecting the best five-card hand.
pub fn evaluate_best(cards: CardMask) -> Result<HandRank, RfError> {
    let count = cards.count();
    if !(5..=7).contains(&count) {
        return Err(RfError::EvalError(format!(
            "evaluate_best expects 5, 6, or 7 cards, got {count}"
        )));
    }

    let mut card_indices = [0u8; 7];
    let mut idx = 0usize;
    let mut bits = cards.bits();
    while bits != 0 {
        let bit = bits & bits.wrapping_neg();
        card_indices[idx] = bit.trailing_zeros() as u8;
        bits ^= bit;
        idx += 1;
    }

    const SUBSETS_5_OF_5: [[usize; 5]; 1] = [[0, 1, 2, 3, 4]];
    const SUBSETS_5_OF_6: [[usize; 5]; 6] = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 5],
        [0, 1, 2, 4, 5],
        [0, 1, 3, 4, 5],
        [0, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ];
    const SUBSETS_5_OF_7: [[usize; 5]; 21] = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 5],
        [0, 1, 2, 3, 6],
        [0, 1, 2, 4, 5],
        [0, 1, 2, 4, 6],
        [0, 1, 2, 5, 6],
        [0, 1, 3, 4, 5],
        [0, 1, 3, 4, 6],
        [0, 1, 3, 5, 6],
        [0, 1, 4, 5, 6],
        [0, 2, 3, 4, 5],
        [0, 2, 3, 4, 6],
        [0, 2, 3, 5, 6],
        [0, 2, 4, 5, 6],
        [0, 3, 4, 5, 6],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 6],
        [1, 2, 3, 5, 6],
        [1, 2, 4, 5, 6],
        [1, 3, 4, 5, 6],
        [2, 3, 4, 5, 6],
    ];

    let subsets: &[[usize; 5]] = match count {
        5 => &SUBSETS_5_OF_5,
        6 => &SUBSETS_5_OF_6,
        _ => &SUBSETS_5_OF_7,
    };

    let mut best = HandRank::pack(0, &[0, 0, 0, 0, 0]);
    for [i0, i1, i2, i3, i4] in subsets {
        let score = evaluate_five_indices([
            card_indices[*i0],
            card_indices[*i1],
            card_indices[*i2],
            card_indices[*i3],
            card_indices[*i4],
        ]);
        if score > best {
            best = score;
        }
    }

    Ok(best)
}

/// Backward-compatible name for existing call sites.
pub fn evaluate(cards: CardMask) -> Result<HandRank, RfError> {
    evaluate_best(cards)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Card;
    use std::str::FromStr;
    use std::time::Instant;

    fn mask(text: &str) -> CardMask {
        CardMask::from_str(text).unwrap()
    }

    fn c(rank: u8, suit: u8) -> Card {
        Card::from_parts(rank, suit).unwrap()
    }

    #[test]
    fn hand_rank_category_is_decoded_from_pack_bits() {
        let high = evaluate_five(mask("Ac Kd Qh 9s 3c")).unwrap();
        let pair = evaluate_five(mask("Ac As Kd Qh Jh")).unwrap();
        let two_pair = evaluate_five(mask("Ac As Kd Kc Jh")).unwrap();
        let trips = evaluate_five(mask("Ac As Ad Kd Qh")).unwrap();
        let straight = evaluate_five(mask("Ac Kc Qd Jh Td")).unwrap();
        let flush = evaluate_five(mask("Ac Kc Qc 9c 2c")).unwrap();
        let full_house = evaluate_five(mask("Ac Ad Ah Kc Kd")).unwrap();
        let quads = evaluate_five(mask("Ac Ad Ah As Kd")).unwrap();
        let straight_flush = evaluate_five(mask("Ac Kc Qc Jc Tc")).unwrap();

        assert!(high < pair);
        assert!(pair < two_pair);
        assert!(two_pair < trips);
        assert!(trips < straight);
        assert!(straight < flush);
        assert!(flush < full_house);
        assert!(full_house < quads);
        assert!(quads < straight_flush);
    }

    #[test]
    fn evaluate_supports_5_6_7_cards() {
        let five = mask("Ac Kd Qh 9s 3c");
        let six = mask("Ac Kd Qh 9s 3c 2d");
        let seven = mask("Ac Kd Qh 9s 3c 2d 2h");

        let five_eval = evaluate_best(five).unwrap();
        let six_eval = evaluate_best(six).unwrap();
        let seven_eval = evaluate_best(seven).unwrap();

        assert_eq!(five_eval, evaluate_five(five).unwrap());
        assert_eq!(six_eval, evaluate_five(five).unwrap());
        assert_eq!(seven_eval, evaluate_five(mask("Ac Kd Qh 2d 2h")).unwrap());
    }

    #[test]
    fn six_card_evaluator_works_and_chooses_best_with_irrelevant() {
        let hand = mask("Ac Ad As 9c 8c 2d");
        let best = evaluate_best(hand).unwrap();
        let expected = evaluate_five(mask("Ac Ad As 9c 8c")).unwrap();
        assert_eq!(best, expected);
    }

    #[test]
    fn one_pair_tiebreaker_uses_high_kickers() {
        let pair_kicker = evaluate_five(mask("Ac Ad Kc 3h 2s")).unwrap();
        let weaker = evaluate_five(mask("Ac Ad Qc Jh 2s")).unwrap();
        assert!(pair_kicker > weaker);

        let pair_kicker2 = evaluate_five(mask("Ac Ad Kc Qh Js")).unwrap();
        let weaker2 = evaluate_five(mask("Ac Ad Kc Qh Ts")).unwrap();
        assert!(pair_kicker2 > weaker2);
    }

    #[test]
    fn compare_wheel_and_six_high_straight() {
        let wheel = evaluate_five(mask("5h 4d 3c 2s As")).unwrap();
        let six_high = evaluate_five(mask("6h 5d 4c 3s 2d")).unwrap();
        assert!(six_high > wheel);
    }

    #[test]
    fn straight_flush_orders_higher_than_below() {
        let k_high_sf = evaluate_five(mask("Ac Kc Qc Jc Tc")).unwrap();
        let q_high_sf = evaluate_five(mask("Qc Jc Tc 9c 8c")).unwrap();
        assert!(k_high_sf > q_high_sf);
    }

    #[test]
    fn flush_and_highcard_kickers_are_lexicographic() {
        let better_highcard = evaluate_five(mask("Ac Qh 9d 6s 3c")).unwrap();
        let weaker_highcard = evaluate_five(mask("Ac Jd 9h 6c 3s")).unwrap();
        assert!(better_highcard > weaker_highcard);

        let better_flush = evaluate_five(mask("Ac Qc 9c 6c 3c")).unwrap();
        let weaker_flush = evaluate_five(mask("Ac Jc 8c 6c 4c")).unwrap();
        assert!(better_flush > weaker_flush);
    }

    #[test]
    fn trips_and_full_house_ordering() {
        let higher_trips = evaluate_five(mask("Ac As Ad Kd Qh")).unwrap();
        let lower_trips = evaluate_five(mask("Kc Ks Kd 5h 4d")).unwrap();
        assert!(higher_trips > lower_trips);

        let higher_full_house = evaluate_five(mask("Ac Ad Ah Kd Kc")).unwrap();
        let lower_full_house = evaluate_five(mask("Qc Qd Qh Js Jd")).unwrap();
        assert!(higher_full_house > lower_full_house);
    }

    #[test]
    fn quads_with_higher_rank_beats_and_kicker_order() {
        let better = evaluate_five(mask("Ac As Ah Ad Kd")).unwrap();
        let worse = evaluate_five(mask("Kc Ks Kh Kd 2h")).unwrap();
        assert!(better > worse);
    }

    #[test]
    fn invalid_hand_sizes_return_errors() {
        assert!(matches!(evaluate_best(mask("Ac Kd Qh")), Err(RfError::EvalError(_))));
        assert!(matches!(
            evaluate_best(mask("Ac Kd Qh 9s 3c 7h 2d 4c")),
            Err(RfError::EvalError(_))
        ));
    }

    #[test]
    fn card_order_has_no_effect_on_evaluation() {
        let canonical = evaluate_five(mask("Ac Kd Qh 9s 3c")).unwrap();
        let shuffled = evaluate_five(mask("9s Qh Ac 3c Kd")).unwrap();
        assert_eq!(canonical, shuffled);
    }

    #[test]
    fn evaluate_works_with_7_card_full_range() {
        let all = CardMask::from_cards([c(0, 0), c(5, 0), c(8, 1), c(10, 1), c(12, 2), c(11, 3), c(3, 3)]);
        let result = evaluate_best(all).unwrap();
        let direct = evaluate_five(CardMask::from_cards([c(8, 1), c(10, 1), c(12, 2), c(11, 3), c(5, 0)])).unwrap();
        assert_eq!(result, direct);
    }

    #[test]
    #[ignore = "performance: run with --release -- --ignored"]
    fn perf_1_000_000_seven_card_evaluations() {
        const TRIALS: usize = 1_000_000;
        let mut hands = Vec::with_capacity(TRIALS);

        for base in 0..TRIALS {
            let b = (base % 52) as u8;
            let subset = [
                b % 52,
                (b + 7) % 52,
                (b + 14) % 52,
                (b + 21) % 52,
                (b + 28) % 52,
                (b + 35) % 52,
                (b + 42) % 52,
            ]
            .map(Card::from_index_unchecked);
            hands.push(CardMask::from_cards(subset));
        }

        let start = Instant::now();
        let mut sink = 0u64;
        for cards in hands {
            let score = evaluate_best(cards).unwrap();
            sink ^= score.bits() as u64;
        }
        let elapsed = start.elapsed();
        let secs = elapsed.as_secs_f64();
        let throughput = (TRIALS as f64) / secs;
        println!(
            "evaluated {TRIALS} 7-card hands in {:?} => {:.2} evals/sec, sink={sink:#x}",
            elapsed, throughput
        );
        if cfg!(not(debug_assertions)) {
            assert!(
                throughput >= 1_000_000.0,
                "release throughput {throughput:.2} < 1_000_000 evals/sec"
            );
        }
    }

    #[test]
    #[ignore = "slow exhaustive oracle; enable in dedicated perf/validation runs"]
    fn exhaustive_five_card_category_counts_match_reference() {
        use std::collections::HashMap;

        let mut seen: HashMap<u8, u64> = HashMap::new();

        for a in 0u8..48 {
            for b in (a + 1)..49 {
                for c in (b + 1)..50 {
                    for d in (c + 1)..51 {
                        for e in (d + 1)..52 {
                            let cards = CardMask::from_cards([
                                Card::from_index_unchecked(a),
                                Card::from_index_unchecked(b),
                                Card::from_index_unchecked(c),
                                Card::from_index_unchecked(d),
                                Card::from_index_unchecked(e),
                            ]);
                            let rank = evaluate_five(cards).unwrap();
                            let class = (rank.bits() >> 24) as u8;
                            let entry = seen.entry(class).or_insert(0);
                            *entry += 1;
                        }
                    }
                }
            }
        }

        assert_eq!(seen.get(&0).copied().unwrap_or(0), 1_302_540);
        assert_eq!(seen.get(&1).copied().unwrap_or(0), 1_098_240);
        assert_eq!(seen.get(&2).copied().unwrap_or(0), 123_552);
        assert_eq!(seen.get(&3).copied().unwrap_or(0), 54_912);
        assert_eq!(seen.get(&4).copied().unwrap_or(0), 10_200);
        assert_eq!(seen.get(&5).copied().unwrap_or(0), 5_108);
        assert_eq!(seen.get(&6).copied().unwrap_or(0), 3_744);
        assert_eq!(seen.get(&7).copied().unwrap_or(0), 624);
        assert_eq!(seen.get(&8).copied().unwrap_or(0), 40);
        assert_eq!(seen.values().copied().sum::<u64>(), 2_598_960);
    }
}

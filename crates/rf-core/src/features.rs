use crate::{category_of_rank, evaluate_best, Board, HandCategory, HoleCards, RfError};

pub type FeatureError = RfError;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct DrawFlags {
    pub flush_draw: bool,
    pub straight_draw: bool,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ModelBucket {
    Air,
    Draw,
    OnePair,
    TwoPairOrTrips,
    StraightOrFlush,
    FullHousePlus,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct HandFeatures {
    pub category: HandCategory,
    pub draws: DrawFlags,
    pub bucket: ModelBucket,
}

pub fn extract_features(hole: HoleCards, board: &Board) -> Result<HandFeatures, FeatureError> {
    let flush_draw_flag = |mask: crate::CardMask| -> bool {
        let mut suit_counts = [0u8; 4];
        for card in mask.iter() {
            suit_counts[card.suit() as usize] += 1;
        }
        suit_counts.contains(&4)
    };

    let straight_draw_flag = |mask: crate::CardMask| -> bool {
        let mut rank_mask: u16 = 0;
        for card in mask.iter() {
            rank_mask |= 1u16 << card.rank();
        }

        if (rank_mask & 0x100F).count_ones() == 4 {
            return true;
        }

        for high in 4u8..=12u8 {
            let window: u16 = 0x1Fu16 << (high - 4);
            if (rank_mask & window).count_ones() == 4 {
                return true;
            }
        }

        false
    };

    let bucket_for_category = |category: HandCategory, draws: DrawFlags| -> ModelBucket {
        match category {
            HandCategory::HighCard if draws.flush_draw || draws.straight_draw => ModelBucket::Draw,
            HandCategory::HighCard => ModelBucket::Air,
            HandCategory::OnePair => ModelBucket::OnePair,
            HandCategory::TwoPair | HandCategory::ThreeOfAKind => ModelBucket::TwoPairOrTrips,
            HandCategory::Straight | HandCategory::Flush | HandCategory::StraightFlush => ModelBucket::StraightOrFlush,
            HandCategory::FullHouse | HandCategory::FourOfAKind => ModelBucket::FullHousePlus,
        }
    };

    let len = board.len();
    if !(3u8..=5u8).contains(&len) {
        return Err(RfError::InvalidBoardLength(len as usize));
    }

    let all_cards = hole.mask().union(board.mask());
    let category = category_of_rank(evaluate_best(all_cards)?);

    let mut draws = DrawFlags {
        flush_draw: flush_draw_flag(all_cards),
        straight_draw: straight_draw_flag(all_cards),
    };

    if len == 5
        || matches!(
            category,
            HandCategory::Straight
                | HandCategory::Flush
                | HandCategory::FullHouse
                | HandCategory::FourOfAKind
                | HandCategory::StraightFlush
        )
    {
        draws = DrawFlags {
            flush_draw: false,
            straight_draw: false,
        };
    }

    Ok(HandFeatures {
        category,
        draws,
        bucket: bucket_for_category(category, draws),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct FeatureCase {
        hole_a: &'static str,
        hole_b: &'static str,
        board: &'static str,
        category: HandCategory,
        flush_draw: bool,
        straight_draw: bool,
        bucket: ModelBucket,
    }

    #[test]
    fn extract_features_matches_required_matrix() {
        let cases = [
            FeatureCase {
                hole_a: "As",
                hole_b: "Kd",
                board: "7c 4h 2s",
                category: HandCategory::HighCard,
                flush_draw: false,
                straight_draw: false,
                bucket: ModelBucket::Air,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "Kh",
                board: "Qh 7h 2c",
                category: HandCategory::HighCard,
                flush_draw: true,
                straight_draw: false,
                bucket: ModelBucket::Draw,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "Kd",
                board: "Qs Jh 2c",
                category: HandCategory::HighCard,
                flush_draw: false,
                straight_draw: true,
                bucket: ModelBucket::Draw,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "Ad",
                board: "Ks 7h 2c",
                category: HandCategory::OnePair,
                flush_draw: false,
                straight_draw: false,
                bucket: ModelBucket::OnePair,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "Ad",
                board: "Ks Kd 2c",
                category: HandCategory::TwoPair,
                flush_draw: false,
                straight_draw: false,
                bucket: ModelBucket::TwoPairOrTrips,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "Ad",
                board: "As Kd 2c",
                category: HandCategory::ThreeOfAKind,
                flush_draw: false,
                straight_draw: false,
                bucket: ModelBucket::TwoPairOrTrips,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "Kd",
                board: "Qs Jh Tc",
                category: HandCategory::Straight,
                flush_draw: false,
                straight_draw: false,
                bucket: ModelBucket::StraightOrFlush,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "Kh",
                board: "Qh Jh 2h",
                category: HandCategory::Flush,
                flush_draw: false,
                straight_draw: false,
                bucket: ModelBucket::StraightOrFlush,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "Ad",
                board: "As Kd Kc",
                category: HandCategory::FullHouse,
                flush_draw: false,
                straight_draw: false,
                bucket: ModelBucket::FullHousePlus,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "Ad",
                board: "As Ac Kd",
                category: HandCategory::FourOfAKind,
                flush_draw: false,
                straight_draw: false,
                bucket: ModelBucket::FullHousePlus,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "Ad",
                board: "Kh Qh Jh",
                category: HandCategory::OnePair,
                flush_draw: true,
                straight_draw: true,
                bucket: ModelBucket::OnePair,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "Kh",
                board: "Qh Jh 2c 7s 8d",
                category: HandCategory::HighCard,
                flush_draw: false,
                straight_draw: false,
                bucket: ModelBucket::Air,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "2d",
                board: "3s 4c Kd",
                category: HandCategory::HighCard,
                flush_draw: false,
                straight_draw: true,
                bucket: ModelBucket::Draw,
            },
            FeatureCase {
                hole_a: "Ah",
                hole_b: "2d",
                board: "3s 4c 5d",
                category: HandCategory::Straight,
                flush_draw: false,
                straight_draw: false,
                bucket: ModelBucket::StraightOrFlush,
            },
        ];

        for case in cases {
            let hole = HoleCards::from_strs(case.hole_a, case.hole_b).unwrap();
            let board = Board::parse(case.board).unwrap();
            let features = extract_features(hole, &board).unwrap();

            assert_eq!(features.category, case.category, "{}", case.board);
            assert_eq!(features.draws.flush_draw, case.flush_draw, "{}", case.board);
            assert_eq!(features.draws.straight_draw, case.straight_draw, "{}", case.board);
            assert_eq!(features.bucket, case.bucket, "{}", case.board);
        }
    }

    #[test]
    fn extract_features_rejects_invalid_board_lengths() {
        let too_short = Board::parse("7c 4h");
        assert!(matches!(too_short, Err(crate::RfError::InvalidBoardLength(2))));

        let too_long = Board::parse("7c 4h 2s 9d 3c 5h");
        assert!(matches!(too_long, Err(crate::RfError::InvalidBoardLength(6))));
    }

    #[test]
    fn extract_features_pair_plus_draw_is_classed_as_one_pair() {
        let hole = HoleCards::from_strs("Ah", "Ad").unwrap();
        let board = Board::parse("Kh Qh Jh").unwrap();
        let features = extract_features(hole, &board).unwrap();
        assert_eq!(features.category, HandCategory::OnePair);
        assert!(features.draws.flush_draw);
        assert!(features.draws.straight_draw);
        assert_eq!(features.bucket, ModelBucket::OnePair);
    }

    #[test]
    fn extract_features_prefers_no_draws_on_made_straight_or_flush() {
        let pair_board = "Qh Jh Ts 9h 2d";
        let pair_hole = HoleCards::from_strs("Ah", "Kc").unwrap();
        let pair_features = extract_features(pair_hole, &Board::parse(pair_board).unwrap()).unwrap();
        assert_eq!(pair_features.bucket, ModelBucket::StraightOrFlush);
        assert_eq!(pair_features.category, HandCategory::Straight);
        assert!(!pair_features.draws.flush_draw);
        assert!(!pair_features.draws.straight_draw);

        let flush_board = "Ah Kh Qh Jh 2h";
        let flush_hole = HoleCards::from_strs("2c", "7d").unwrap();
        let flush_features = extract_features(flush_hole, &Board::parse(flush_board).unwrap()).unwrap();
        assert_eq!(flush_features.bucket, ModelBucket::StraightOrFlush);
        assert_eq!(flush_features.category, HandCategory::Flush);
        assert!(!flush_features.draws.flush_draw);
        assert!(!flush_features.draws.straight_draw);
    }

    #[test]
    fn extract_features_river_forces_no_draws_regardless_of_potential() {
        let hole = HoleCards::from_strs("Ah", "Kd").unwrap();
        let board = Board::parse("Qh Jh 2c 7s 8d").unwrap();
        let f = extract_features(hole, &board).unwrap();
        assert_eq!(f.bucket, ModelBucket::Air);
        assert_eq!(f.category, HandCategory::HighCard);
        assert!(!f.draws.flush_draw);
        assert!(!f.draws.straight_draw);
    }

    #[test]
    fn extract_features_wheel_related_straight_draw_case() {
        let wheel_draw_hole = HoleCards::from_strs("Ah", "2d").unwrap();
        let wheel_draw_board = Board::parse("3s 4c Kd").unwrap();
        let draw_features = extract_features(wheel_draw_hole, &wheel_draw_board).unwrap();
        assert_eq!(draw_features.category, HandCategory::HighCard);
        assert!(draw_features.draws.straight_draw);
        assert_eq!(draw_features.bucket, ModelBucket::Draw);

        let wheel_made_board = Board::parse("3s 4c 5d").unwrap();
        let made_features = extract_features(wheel_draw_hole, &wheel_made_board).unwrap();
        assert_eq!(made_features.category, HandCategory::Straight);
        assert!(!made_features.draws.straight_draw);
        assert_eq!(made_features.bucket, ModelBucket::StraightOrFlush);
    }

    #[test]
    fn extract_features_flush_draw_requires_exactly_four_suits() {
        let hole = HoleCards::from_strs("Ac", "Kd").unwrap();
        let no_draw = Board::parse("Qh Jh 2h 9h 4h").unwrap();
        let no_draw_features = extract_features(hole, &no_draw).unwrap();
        assert!(!no_draw_features.draws.flush_draw);
        assert_eq!(no_draw_features.bucket, ModelBucket::StraightOrFlush);

        let with_draw = Board::parse("Qh Jh 2h 9h").unwrap();
        let with_draw_features = extract_features(hole, &with_draw).unwrap();
        assert!(with_draw_features.draws.flush_draw);
        assert_eq!(with_draw_features.bucket, ModelBucket::Draw);
    }

    #[test]
    fn extract_features_straight_flush_trumps_pair_plus_draws() {
        let hole = HoleCards::from_strs("Ah", "Kh").unwrap();
        let board = Board::parse("Qh Jh Th 9h").unwrap();
        let features = extract_features(hole, &board).unwrap();
        assert_eq!(features.category, HandCategory::StraightFlush);
        assert_eq!(features.bucket, ModelBucket::StraightOrFlush);
        assert!(!features.draws.flush_draw);
        assert!(!features.draws.straight_draw);
    }

    #[test]
    fn extract_features_inherits_duplicate_rejection_from_state() {
        use crate::KnownState;

        let hero = HoleCards::from_strs("Ah", "Kd").unwrap();
        let board_overlap = Board::parse("Ah 3c 7d").unwrap();
        let state_err = KnownState::new(hero, board_overlap).unwrap_err();
        assert!(matches!(state_err, crate::RfError::DuplicateCard));
    }

    #[test]
    fn extract_features_turn_straight_draw_still_works() {
        let hole = HoleCards::from_strs("Ac", "2d").unwrap();
        let board = Board::parse("3s 4c 6d 7h").unwrap();
        let features = extract_features(hole, &board).unwrap();
        assert!(features.draws.straight_draw);
        assert_eq!(features.bucket, ModelBucket::Draw);
    }
}

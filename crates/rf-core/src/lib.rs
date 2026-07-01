//! A minimal, practical poker core: no extra abstraction layers, just the pieces
//! needed to run a small Bayesian range engine.

pub mod combo;
pub mod error;
pub mod evaluator;
pub mod features;
pub mod game;

pub use combo::{
    all_combos, expand_range, ordered_pairs, remaining_cards, unordered_pairs, ComboId, ComboWeights, RangeSpec,
    NUM_HOLE_COMBOS,
};
pub use error::RfError;
pub use evaluator::{
    category_of_cards, category_of_rank, evaluate, evaluate_best, evaluate_five, HandCategory, HandRank,
};
pub use game::{Board, Card, CardMask, HoleCards, KnownState, Street};

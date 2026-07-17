//! Core poker game objects: cards, card masks, hole cards, boards, streets, and known state.

use crate::error::RfError;
use std::fmt;
use std::str::FromStr;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Card(u8);

impl Card {
    pub fn from_parts(rank: u8, suit: u8) -> Result<Self, RfError> {
        if rank >= 13 || suit >= 4 {
            return Err(RfError::InvalidCard(format!("rank={rank} suit={suit}")));
        }

        Ok(Self::from_parts_unchecked(rank, suit))
    }

    pub(crate) const fn from_parts_unchecked(rank: u8, suit: u8) -> Self {
        Self(suit * 13 + rank)
    }

    pub(crate) const fn from_index_unchecked(index: u8) -> Self {
        Self(index)
    }

    pub const fn index(self) -> u8 {
        self.0
    }

    pub const fn rank(self) -> u8 {
        self.0 % 13
    }

    pub const fn suit(self) -> u8 {
        self.0 / 13
    }

    pub const fn bit(self) -> u64 {
        1u64 << self.0
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rank_ch = match self.rank() {
            0 => '2',
            1 => '3',
            2 => '4',
            3 => '5',
            4 => '6',
            5 => '7',
            6 => '8',
            7 => '9',
            8 => 'T',
            9 => 'J',
            10 => 'Q',
            11 => 'K',
            12 => 'A',
            _ => '?',
        };
        let suit_ch = match self.suit() {
            0 => 'c',
            1 => 'd',
            2 => 'h',
            3 => 's',
            _ => '?',
        };
        write!(f, "{rank_ch}{suit_ch}")
    }
}

impl FromStr for Card {
    type Err = RfError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut it = s.chars();
        let rank_char = it.next().ok_or_else(|| RfError::InvalidCard(s.to_string()))?;
        let rank = match rank_char {
            '2' => 0,
            '3' => 1,
            '4' => 2,
            '5' => 3,
            '6' => 4,
            '7' => 5,
            '8' => 6,
            '9' => 7,
            'T' | 't' => 8,
            'J' | 'j' => 9,
            'Q' | 'q' => 10,
            'K' | 'k' => 11,
            'A' | 'a' => 12,
            _ => return Err(RfError::InvalidCard(s.to_string())),
        };

        let suit_char = it.next().ok_or_else(|| RfError::InvalidCard(s.to_string()))?;
        let suit = match suit_char {
            'c' | 'C' => 0,
            'd' | 'D' => 1,
            'h' | 'H' => 2,
            's' | 'S' => 3,
            _ => return Err(RfError::InvalidCard(s.to_string())),
        };

        if it.next().is_some() {
            return Err(RfError::InvalidCard(s.to_string()));
        }

        Ok(Card::from_parts_unchecked(rank, suit))
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct CardMask(u64);

impl CardMask {
    pub const EMPTY: Self = Self(0);

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn from_bits(bits: u64) -> Self {
        Self(bits & ((1u64 << 52) - 1))
    }

    pub const fn contains(self, card: Card) -> bool {
        (self.0 & card.bit()) != 0
    }

    pub fn insert(&mut self, card: Card) -> bool {
        let bit = card.bit();
        let was = (self.0 & bit) != 0;
        self.0 |= bit;
        !was
    }

    pub fn remove(&mut self, card: Card) {
        self.0 &= !card.bit();
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    pub fn iter(self) -> impl Iterator<Item = Card> {
        let mut bits = self.0;
        std::iter::from_fn(move || {
            if bits == 0 {
                return None;
            }
            let idx = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            Some(Card(idx))
        })
    }

    pub fn from_cards(cards: impl IntoIterator<Item = Card>) -> Self {
        let mut mask = Self::EMPTY;
        for card in cards {
            let _ = mask.insert(card);
        }
        mask
    }
}

impl FromStr for CardMask {
    type Err = RfError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut mask = CardMask::EMPTY;
        for token in s.split_whitespace() {
            let card = token.parse::<Card>()?;
            if !mask.insert(card) {
                return Err(RfError::DuplicateCard);
            }
        }
        Ok(mask)
    }
}

impl Default for CardMask {
    fn default() -> Self {
        Self::EMPTY
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HoleCards {
    cards: [Card; 2],
}

impl HoleCards {
    pub fn new(a: Card, b: Card) -> Result<Self, RfError> {
        if a == b {
            return Err(RfError::DuplicateCard);
        }
        let mut cards = [a, b];
        cards.sort_by_key(|c| c.index());
        Ok(Self { cards })
    }

    pub fn from_strs(a: &str, b: &str) -> Result<Self, RfError> {
        Self::new(a.parse()?, b.parse()?)
    }

    pub const fn cards(self) -> [Card; 2] {
        self.cards
    }

    pub const fn mask(self) -> CardMask {
        CardMask::from_bits(self.cards[0].bit() | self.cards[1].bit())
    }
}

impl fmt::Display for HoleCards {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.cards[0], self.cards[1])
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Board {
    cards: [Card; 5],
    len: u8,
    mask: CardMask,
}

impl Board {
    pub fn parse(text: &str) -> Result<Self, RfError> {
        let parse_board_cards = |text: &str| -> Result<Vec<Card>, RfError> {
            let t = text.trim();
            if t.is_empty() {
                return Ok(Vec::new());
            }

            if t.chars().any(|c| c.is_whitespace()) {
                let mut out = Vec::new();
                for token in t.split_whitespace() {
                    out.push(token.parse::<Card>()?);
                }
                return Ok(out);
            }

            if !t.len().is_multiple_of(2) {
                return Err(RfError::InvalidBoardLength(t.len()));
            }

            let bytes = t.as_bytes();
            let mut out = Vec::with_capacity(t.len() / 2);
            for i in (0..bytes.len()).step_by(2) {
                let token = std::str::from_utf8(&bytes[i..i + 2]).map_err(|_| RfError::InvalidCard(t.to_string()))?;
                out.push(token.parse::<Card>()?);
            }
            Ok(out)
        };

        let cards = parse_board_cards(text)?;
        let len = cards.len();
        if !matches!(len, 0 | 3 | 4 | 5) {
            return Err(RfError::InvalidBoardLength(len));
        }

        let mut fixed = [Card::from_parts_unchecked(0, 0); 5];
        let mut mask = CardMask::EMPTY;
        for (idx, card) in cards.iter().copied().enumerate() {
            if mask.contains(card) {
                return Err(RfError::DuplicateCard);
            }
            fixed[idx] = card;
            let _ = mask.insert(card);
        }

        Ok(Self {
            cards: fixed,
            len: len as u8,
            mask,
        })
    }

    pub fn cards(&self) -> &[Card] {
        &self.cards[..self.len as usize]
    }

    pub const fn mask(self) -> CardMask {
        self.mask
    }

    pub const fn street(self) -> Street {
        match self.len {
            0 => Street::Preflop,
            3 => Street::Flop,
            4 => Street::Turn,
            5 => Street::River,
            _ => Street::Preflop,
        }
    }

    pub fn len(self) -> u8 {
        self.len
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn is_prefix_of(&self, other: &Board) -> bool {
        let a = self.len as usize;
        let b = other.len as usize;
        a <= b && self.cards[..a] == other.cards[..a]
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for card in self.cards() {
            write!(f, "{card}")?;
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KnownState {
    pub hero: HoleCards,
    pub board: Board,
}

impl KnownState {
    pub fn new(hero: HoleCards, board: Board) -> Result<Self, RfError> {
        if hero.mask().intersects(board.mask()) {
            return Err(RfError::DuplicateCard);
        }
        Ok(Self { hero, board })
    }

    pub const fn dead_cards(self) -> CardMask {
        self.hero.mask().union(self.board.mask())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> Card {
        Card::from_parts(rank, suit).unwrap()
    }

    #[test]
    fn card_parts_roundtrip() {
        let card = Card::from_parts(8, 2).unwrap();
        assert_eq!(card.index(), 34);
        assert_eq!(card.rank(), 8);
        assert_eq!(card.suit(), 2);
        assert_eq!(card.bit(), 1u64 << 34);
        assert_eq!(card.to_string(), "Th");
    }

    #[test]
    fn card_from_str_accepts_and_rejects() {
        assert_eq!("Ac".parse::<Card>(), Ok(c(12, 0)));
        assert_eq!("kh".parse::<Card>(), Ok(c(11, 2)));
        assert_eq!("td".parse::<Card>(), Ok(c(8, 1)));
        assert_eq!("2c".parse::<Card>(), Ok(c(0, 0)));

        assert!(matches!("".parse::<Card>(), Err(RfError::InvalidCard(_))));
        assert!(matches!("A".parse::<Card>(), Err(RfError::InvalidCard(_))));
        assert!(matches!("ZZ".parse::<Card>(), Err(RfError::InvalidCard(_))));
        assert!(matches!("10c".parse::<Card>(), Err(RfError::InvalidCard(_))));
        assert!(matches!("A1".parse::<Card>(), Err(RfError::InvalidCard(_))));
        assert!(matches!("A c".parse::<Card>(), Err(RfError::InvalidCard(_))));
    }

    #[test]
    fn card_mask_basic_ops() {
        let mut mask = CardMask::EMPTY;
        let a = c(0, 0);
        let b = c(12, 3);

        assert!(!mask.contains(a));
        assert!(mask.insert(a));
        assert!(mask.contains(a));
        assert!(!mask.insert(a));

        assert_eq!(mask.count(), 1);
        mask.insert(b);
        assert_eq!(mask.count(), 2);
        assert!(mask.intersects(CardMask::from_bits(a.bit())));
        assert_eq!(mask.union(CardMask::from_bits(a.bit())), mask);
        assert_eq!(mask.count(), 2);

        mask.remove(a);
        assert!(!mask.contains(a));
        assert_eq!(mask.count(), 1);
    }

    #[test]
    fn card_mask_from_bits_masks_all_but_52() {
        let mask = CardMask::from_bits(u64::MAX);
        assert_eq!(mask.count(), 52);
        assert!(!mask.intersects(CardMask::from_bits(1u64 << 52)));
    }

    #[test]
    fn card_mask_iter_returns_sorted_cards() {
        let a = c(12, 3);
        let b = c(0, 0);
        let c = c(5, 1);
        let mask = CardMask::from_cards([c, a, b]);
        let collected: Vec<u8> = mask.iter().map(|card| card.index()).collect();
        assert_eq!(collected, vec![0, 18, 51]);
    }

    #[test]
    fn card_mask_from_str_parses_whitespace_tokens() {
        let mask = "As Ks Qh".parse::<CardMask>().unwrap();
        assert!(mask.contains(c(12, 3)));
        assert!(mask.contains(c(11, 3)));
        assert!(mask.contains(c(10, 2)));
    }

    #[test]
    fn hole_cards_keep_canonical_order_and_reject_duplicates() {
        let cards = HoleCards::new(c(12, 3), c(0, 0)).unwrap();
        assert_eq!(cards.cards(), [c(0, 0), c(12, 3)]);
        assert_eq!(cards.mask().count(), 2);

        let reversed = HoleCards::new(c(0, 0), c(12, 3)).unwrap().cards();
        assert_eq!(cards, HoleCards::new(reversed[0], reversed[1]).unwrap());
        assert!(HoleCards::new(c(9, 1), c(9, 1)).is_err());
    }

    #[test]
    fn hole_cards_from_strs_works() {
        let hole = HoleCards::from_strs("Ac", "Kd").unwrap();
        assert_eq!(hole.cards(), [c(12, 0), c(11, 1)]);
    }

    #[test]
    fn parse_board_cards_whitespace() {
        let cards = Board::parse("As Ks Qh").unwrap().cards().to_vec();
        assert_eq!(cards, vec![c(12, 3), c(11, 3), c(10, 2)]);
    }

    #[test]
    fn parse_board_cards_compact() {
        let cards = Board::parse("AcKsQh").unwrap().cards().to_vec();
        assert_eq!(cards, vec![c(12, 0), c(11, 3), c(10, 2)]);
    }

    #[test]
    fn parse_board_cards_rejects_odd_len_compact() {
        assert!(matches!(
            Board::parse("AhK").expect_err("odd compact len should fail"),
            RfError::InvalidBoardLength(3)
        ));
    }

    #[test]
    fn board_parse_valid_lengths() {
        let pre = Board::parse("").unwrap();
        assert_eq!(pre.street(), Street::Preflop);
        assert_eq!(pre.cards().len(), 0);

        let flop = Board::parse("AdKdQd").unwrap();
        assert_eq!(flop.street(), Street::Flop);
        assert_eq!(flop.cards().len(), 3);

        let turn = Board::parse("Ad Kd Qd Js").unwrap();
        assert_eq!(turn.street(), Street::Turn);
        assert_eq!(turn.cards().len(), 4);

        let river = Board::parse("AdKdQdJsQc").unwrap();
        assert_eq!(river.street(), Street::River);
        assert_eq!(river.cards().len(), 5);
    }

    #[test]
    fn board_parse_rejects_bad_sizes_and_dupes() {
        assert_eq!(Board::parse("Ah").unwrap_err(), RfError::InvalidBoardLength(1));
        assert_eq!(
            Board::parse("Ah Kc Qd Js Qc 7h").unwrap_err(),
            RfError::InvalidBoardLength(6)
        );
        assert_eq!(Board::parse("AhKcQdJsQ").unwrap_err(), RfError::InvalidBoardLength(9));
        assert!(matches!(Board::parse("As As 2c"), Err(RfError::DuplicateCard)));
    }

    #[test]
    fn board_prefix_checks() {
        let flop = Board::parse("AcKdQs").unwrap();
        let turn = Board::parse("AcKdQsJh").unwrap();
        assert!(flop.is_prefix_of(&turn));
        assert!(!turn.is_prefix_of(&flop));
    }

    #[test]
    fn board_prefix_requires_order_and_length() {
        let flop = Board::parse("AcKdQs").unwrap();
        let reordered = Board::parse("KdAcQs").unwrap();
        let unrelated = Board::parse("QcJdTh").unwrap();
        assert!(!reordered.is_prefix_of(&flop));
        assert!(!flop.is_prefix_of(&unrelated));
        assert!(!unrelated.is_prefix_of(&flop));
    }

    #[test]
    fn board_from_str_invalid_board_tokens() {
        assert!(matches!(Board::parse("AhKcQdXx").unwrap_err(), RfError::InvalidCard(_)));
    }

    #[test]
    fn known_state_valid_and_invalid() {
        let hero = HoleCards::new(c(0, 0), c(1, 0)).unwrap();
        let board = Board::parse("AsKcQd").unwrap();
        let state = KnownState::new(hero, board).unwrap();
        assert_eq!(state.dead_cards(), hero.mask().union(board.mask()));

        let overlap_board = Board::parse("2c 3d 4h").unwrap();
        assert!(matches!(
            KnownState::new(hero, overlap_board),
            Err(RfError::DuplicateCard)
        ));

        let overlap_board_first = Board::parse("2c Jh 4d").unwrap();
        assert!(matches!(
            KnownState::new(hero, overlap_board_first),
            Err(RfError::DuplicateCard)
        ));

        let overlap_board_second = Board::parse("3c 2d 5h").unwrap();
        assert!(matches!(
            KnownState::new(HoleCards::new(c(2, 0), c(0, 1)).unwrap(), overlap_board_second),
            Err(RfError::DuplicateCard)
        ));
    }

    #[test]
    fn card_mask_default_is_empty() {
        let mask = CardMask::default();
        assert_eq!(mask.bits(), 0);
    }
}

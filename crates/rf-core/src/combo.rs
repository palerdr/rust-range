use crate::error::RfError;
use crate::game::{Card, CardMask, HoleCards};

pub const NUM_HOLE_COMBOS: usize = 1326;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ComboId(u16);

impl ComboId {
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn from_raw(raw: u16) -> Result<Self, RfError> {
        if usize::from(raw) >= NUM_HOLE_COMBOS {
            return Err(RfError::InvalidComboId(raw));
        }
        Ok(Self(raw))
    }

    pub(crate) const fn from_raw_unchecked(raw: u16) -> Self {
        Self(raw)
    }

    pub fn from_hole_cards(hole: HoleCards) -> Self {
        let [c1, c2] = hole.cards();
        let a = c1.index().min(c2.index()) as usize;
        let b = c1.index().max(c2.index()) as usize;
        let before = if a == 0 { 0 } else { a * 51 - (a * (a - 1)) / 2 };
        Self((before + (b - a - 1)) as u16)
    }

    pub fn hole_cards(self) -> HoleCards {
        debug_assert!(self.index() < NUM_HOLE_COMBOS);
        let mut idx = self.0;
        let mut a: u16 = 0;
        while idx >= (51 - a) {
            idx -= 51 - a;
            a += 1;
        }
        let b = a + 1 + idx;
        let c1 = Card::from_index_unchecked(a as u8);
        let c2 = Card::from_index_unchecked(b as u8);
        HoleCards::new(c1, c2).unwrap()
    }

    pub fn try_hole_cards(self) -> Result<HoleCards, RfError> {
        if self.index() >= NUM_HOLE_COMBOS {
            return Err(RfError::InvalidComboId(self.0));
        }
        Ok(self.hole_cards())
    }
}

#[derive(Clone, Debug)]
pub struct ComboWeights([f64; NUM_HOLE_COMBOS]);

impl ComboWeights {
    pub fn zeros() -> Self {
        Self([0.0; NUM_HOLE_COMBOS])
    }

    pub fn from_uniform() -> Self {
        Self([1.0; NUM_HOLE_COMBOS])
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.0
    }

    pub fn get(&self, id: ComboId) -> f64 {
        self.0[id.index()]
    }

    pub fn set(&mut self, id: ComboId, value: f64) -> Result<(), RfError> {
        if !value.is_finite() || value < 0.0 {
            return Err(RfError::InvalidRangeWeight(value.to_string()));
        }
        self.0[id.index()] = value;
        Ok(())
    }

    pub fn iter_nonzero(&self) -> impl Iterator<Item = (ComboId, f64)> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter(|(_, weight)| **weight != 0.0)
            .map(|(i, weight)| (ComboId::from_raw_unchecked(i as u16), *weight))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RangeSpec {
    Random,
    Notation(String),
    Explicit(Vec<(HoleCards, f64)>),
}

pub fn expand_range(spec: &RangeSpec) -> Result<ComboWeights, RfError> {
    let mut out = ComboWeights::zeros();
    let mut used = [false; NUM_HOLE_COMBOS];

    let add = |id: ComboId, w: f64, out: &mut ComboWeights, used: &mut [bool; NUM_HOLE_COMBOS]| {
        let idx = id.index();
        if idx >= NUM_HOLE_COMBOS {
            return Err(RfError::InvalidComboId(id.0));
        }
        if used[idx] {
            return Err(RfError::DuplicateRangeCombo(id.0.to_string()));
        }
        if !w.is_finite() || w < 0.0 {
            return Err(RfError::InvalidRangeWeight(w.to_string()));
        }
        out.0[idx] += w;
        used[idx] = true;
        Ok(())
    };

    let parse_weight = |token: &str| -> Result<(String, f64), RfError> {
        match token.split_once(':') {
            None => Ok((token.to_string(), 1.0)),
            Some((term, raw)) => {
                let w = raw
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| RfError::InvalidRangeWeight(raw.to_string()))?;
                if !w.is_finite() || w < 0.0 {
                    return Err(RfError::InvalidRangeWeight(raw.to_string()));
                }
                Ok((term.trim().to_string(), w))
            }
        }
    };

    let parse_rank = |ch: char| -> Result<u8, RfError> {
        let rank = match ch {
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
            _ => return Err(RfError::InvalidRangeSpec(ch.to_string())),
        };
        Ok(rank)
    };

    let pair_rank = |rank: u8| -> Vec<ComboId> {
        let mut out = Vec::with_capacity(6);
        for s1 in 0..4 {
            for s2 in (s1 + 1)..4 {
                let c1 = Card::from_parts(rank, s1).unwrap();
                let c2 = Card::from_parts(rank, s2).unwrap();
                let id = ComboId::from_hole_cards(HoleCards::new(c1, c2).unwrap());
                out.push(id);
            }
        }
        out
    };

    let suited_only = |a: u8, b: u8| -> Vec<ComboId> {
        let mut out = Vec::with_capacity(4);
        for suit in 0..4 {
            let c1 = Card::from_parts(a, suit).unwrap();
            let c2 = Card::from_parts(b, suit).unwrap();
            let id = ComboId::from_hole_cards(HoleCards::new(c1, c2).unwrap());
            out.push(id);
        }
        out
    };

    let offsuit_only = |a: u8, b: u8| -> Vec<ComboId> {
        let mut out = Vec::with_capacity(12);
        for s1 in 0..4 {
            for s2 in 0..4 {
                if s1 == s2 {
                    continue;
                }
                let c1 = Card::from_parts(a, s1).unwrap();
                let c2 = Card::from_parts(b, s2).unwrap();
                let id = ComboId::from_hole_cards(HoleCards::new(c1, c2).unwrap());
                out.push(id);
            }
        }
        out
    };

    let all_suited_and_offsuit = |a: u8, b: u8| -> Vec<ComboId> {
        let mut out = suited_only(a, b);
        out.extend(offsuit_only(a, b));
        out
    };

    let expand_notation_term = |term: &str| -> Result<Vec<ComboId>, RfError> {
        let t = term.trim();
        if t.eq_ignore_ascii_case("random") {
            return Ok(all_combos().map(|(id, _)| id).collect());
        }

        if t.len() == 4 {
            let left = &t[0..2];
            let right = &t[2..4];
            let c1 = left
                .parse::<Card>()
                .map_err(|_| RfError::InvalidRangeSpec(term.to_string()))?;
            let c2 = right
                .parse::<Card>()
                .map_err(|_| RfError::InvalidRangeSpec(term.to_string()))?;
            if c1 == c2 {
                return Err(RfError::InvalidRangeSpec(term.to_string()));
            }
            return Ok(vec![ComboId::from_hole_cards(HoleCards::new(c1, c2).unwrap())]);
        }

        let chars: Vec<char> = t.chars().collect();
        if chars.len() == 2 {
            let a = parse_rank(chars[0])?;
            let b = parse_rank(chars[1])?;
            if a == b {
                return Ok(pair_rank(a));
            }
            return Ok(all_suited_and_offsuit(a, b));
        }

        if chars.len() == 3 {
            let a = parse_rank(chars[0])?;
            let b = parse_rank(chars[1])?;
            if a == b {
                return Err(RfError::InvalidRangeSpec(t.to_string()));
            }
            return match chars[2] {
                's' | 'S' => Ok(suited_only(a, b)),
                'o' | 'O' => Ok(offsuit_only(a, b)),
                _ => Err(RfError::InvalidRangeSpec(t.to_string())),
            };
        }

        Err(RfError::InvalidRangeSpec(t.to_string()))
    };

    match spec {
        RangeSpec::Random => {
            for (id, _) in all_combos() {
                out.set(id, 1.0)?;
            }
            Ok(out)
        }
        RangeSpec::Explicit(vec) => {
            for (cards, w) in vec {
                add(ComboId::from_hole_cards(*cards), *w, &mut out, &mut used)?;
            }
            Ok(out)
        }
        RangeSpec::Notation(text) => {
            for token in text.split(',') {
                let token = token.trim();
                if token.is_empty() {
                    continue;
                }
                let (term, weight) = parse_weight(token)?;
                let ids = expand_notation_term(term.as_str())?;
                for id in ids {
                    add(id, weight, &mut out, &mut used)?;
                }
            }
            Ok(out)
        }
    }
}

pub fn all_combos() -> impl ExactSizeIterator<Item = (ComboId, HoleCards)> {
    let mut out = Vec::with_capacity(NUM_HOLE_COMBOS);
    for i in 0u8..52u8 {
        for j in (i + 1)..52u8 {
            let h =
                HoleCards::new(Card::from_index_unchecked(i), Card::from_index_unchecked(j)).expect("valid card pair");
            out.push((ComboId::from_hole_cards(h), h));
        }
    }
    out.into_iter()
}

pub fn remaining_cards(dead: CardMask) -> impl ExactSizeIterator<Item = Card> {
    let out: Vec<Card> = (0u8..52)
        .map(Card::from_index_unchecked)
        .filter(|c| !dead.contains(*c))
        .collect();
    out.into_iter()
}

pub fn unordered_pairs(cards: &[Card]) -> impl ExactSizeIterator<Item = [Card; 2]> {
    let mut out = Vec::with_capacity(cards.len() * (cards.len().saturating_sub(1)) / 2);
    for i in 0..cards.len() {
        for j in (i + 1)..cards.len() {
            out.push([cards[i], cards[j]]);
        }
    }
    out.into_iter()
}

pub fn ordered_pairs(cards: &[Card]) -> impl ExactSizeIterator<Item = [Card; 2]> {
    let mut out = Vec::with_capacity(cards.len() * cards.len().saturating_sub(1));
    for i in 0..cards.len() {
        for j in 0..cards.len() {
            if i == j {
                continue;
            }
            out.push([cards[i], cards[j]]);
        }
    }
    out.into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> Card {
        Card::from_parts(rank, suit).unwrap()
    }

    fn id(raw: u16) -> ComboId {
        ComboId::from_raw(raw).unwrap()
    }

    #[test]
    fn combo_id_mapping_corner_cases() {
        let first = ComboId::from_hole_cards(
            HoleCards::new(Card::from_index_unchecked(0), Card::from_index_unchecked(1)).unwrap(),
        );
        let first_rev = ComboId::from_hole_cards(
            HoleCards::new(Card::from_index_unchecked(1), Card::from_index_unchecked(0)).unwrap(),
        );
        let near_last = ComboId::from_hole_cards(
            HoleCards::new(Card::from_index_unchecked(0), Card::from_index_unchecked(51)).unwrap(),
        );
        let last = ComboId::from_hole_cards(
            HoleCards::new(Card::from_index_unchecked(50), Card::from_index_unchecked(51)).unwrap(),
        );

        assert_eq!(first.index(), 0);
        assert_eq!(first_rev.index(), 0);
        assert_eq!(near_last.index(), 50);
        assert_eq!(last.index(), 1325);
    }

    #[test]
    fn combo_id_roundtrip_for_all() {
        for i in 0u8..52u8 {
            for j in (i + 1)..52u8 {
                let hole = HoleCards::new(Card::from_index_unchecked(i), Card::from_index_unchecked(j)).unwrap();
                let id = ComboId::from_hole_cards(hole);
                assert_eq!(id, ComboId::from_hole_cards(id.hole_cards()));
            }
        }
    }

    #[test]
    fn combo_id_roundtrip_cards_are_sorted_by_index() {
        let id = ComboId::from_hole_cards(
            HoleCards::new(Card::from_index_unchecked(3), Card::from_index_unchecked(12)).unwrap(),
        );
        let [c1, c2] = id.hole_cards().cards();
        assert_eq!(c1.index(), 3);
        assert_eq!(c2.index(), 12);
    }

    #[test]
    fn combo_weights_zero_and_uniform_and_accessors() {
        let mut w = ComboWeights::zeros();
        assert_eq!(w.get(id(0)), 0.0);
        assert!(w.set(id(0), 2.5).is_ok());
        assert_eq!(w.get(id(0)), 2.5);

        let uni = ComboWeights::from_uniform();
        assert_eq!(uni.get(id(1325)), 1.0);
    }

    #[test]
    fn expand_range_random_is_all_ones() {
        let out = expand_range(&RangeSpec::Random).unwrap();
        assert_eq!(out.as_slice().len(), NUM_HOLE_COMBOS);
        assert!(out.as_slice().iter().all(|w| *w == 1.0));
        assert_eq!(out.as_slice().iter().sum::<f64>(), NUM_HOLE_COMBOS as f64);
    }

    #[test]
    fn expand_range_explicit_valid_and_invalid() {
        let c1 = HoleCards::new(c(12, 0), c(11, 1)).unwrap();
        let c2 = HoleCards::new(c(10, 0), c(9, 1)).unwrap();
        let spec = RangeSpec::Explicit(vec![(c1, 0.7), (c2, 1.2)]);
        let out = expand_range(&spec).unwrap();
        assert_eq!(out.get(ComboId::from_hole_cards(c1)), 0.7);
        assert_eq!(out.get(ComboId::from_hole_cards(c2)), 1.2);

        let dup = RangeSpec::Explicit(vec![(c1, 1.0), (c1, 1.0)]);
        assert!(matches!(expand_range(&dup), Err(RfError::DuplicateRangeCombo(_))));

        let neg = RangeSpec::Explicit(vec![(c1, -1.0)]);
        assert!(matches!(expand_range(&neg), Err(RfError::InvalidRangeWeight(_))));
    }

    #[test]
    fn expand_range_notation_terms() {
        assert!(matches!(
            expand_range(&RangeSpec::Notation("QQs".to_string())),
            Err(RfError::InvalidRangeSpec(_))
        ));
        assert!(matches!(
            expand_range(&RangeSpec::Notation("QQo".to_string())),
            Err(RfError::InvalidRangeSpec(_))
        ));

        let aa = expand_range(&RangeSpec::Notation("AA".to_string())).unwrap();
        assert_eq!(aa.as_slice().iter().filter(|w| **w != 0.0).count(), 6);

        let aks = expand_range(&RangeSpec::Notation("AKs".to_string())).unwrap();
        assert_eq!(aks.as_slice().iter().filter(|w| **w != 0.0).count(), 4);

        let ako = expand_range(&RangeSpec::Notation("AKo".to_string())).unwrap();
        assert_eq!(ako.as_slice().iter().filter(|w| **w != 0.0).count(), 12);

        let ak = expand_range(&RangeSpec::Notation("AK".to_string())).unwrap();
        assert_eq!(ak.as_slice().iter().filter(|w| **w != 0.0).count(), 16);

        let exact = expand_range(&RangeSpec::Notation("AsKd".to_string())).unwrap();
        let exact_count = exact.as_slice().iter().filter(|w| **w != 0.0).count();
        assert_eq!(exact_count, 1);

        let weighted = expand_range(&RangeSpec::Notation("AA:0.5 , AKs:0.25".to_string())).unwrap();
        assert_eq!(
            weighted.as_slice().iter().filter(|w| (*w - 0.5).abs() < 1e-12).count(),
            6
        );
    }

    #[test]
    fn expand_range_notation_duplicate_and_invalid_inputs() {
        let dup = RangeSpec::Notation("AA, AA".to_string());
        assert!(matches!(expand_range(&dup), Err(RfError::DuplicateRangeCombo(_))));

        let bad_weight = RangeSpec::Notation("AA:abc".to_string());
        assert!(matches!(expand_range(&bad_weight), Err(RfError::InvalidRangeWeight(_))));

        let bad_term = RangeSpec::Notation("ZZ".to_string());
        assert!(matches!(expand_range(&bad_term), Err(RfError::InvalidRangeSpec(_))));

        let bad_exact = RangeSpec::Notation("As".to_string());
        assert!(matches!(expand_range(&bad_exact), Err(RfError::InvalidRangeSpec(_))));
    }

    #[test]
    fn expand_notation_term_private_cases() {
        let all = expand_range(&RangeSpec::Notation("random".to_string())).unwrap();
        assert_eq!(all.as_slice().iter().filter(|w| **w > 0.0).count(), NUM_HOLE_COMBOS);

        let pair = expand_range(&RangeSpec::Notation("AA".to_string())).unwrap();
        assert_eq!(pair.as_slice().iter().filter(|w| **w > 0.0).count(), 6);

        let suited = expand_range(&RangeSpec::Notation("AKs".to_string())).unwrap();
        assert_eq!(suited.as_slice().iter().filter(|w| **w > 0.0).count(), 4);

        let offsuit = expand_range(&RangeSpec::Notation("AKo".to_string())).unwrap();
        assert_eq!(offsuit.as_slice().iter().filter(|w| **w > 0.0).count(), 12);

        let mixed = expand_range(&RangeSpec::Notation("AK".to_string())).unwrap();
        assert_eq!(mixed.as_slice().iter().filter(|w| **w > 0.0).count(), 16);

        let exact = expand_range(&RangeSpec::Notation("AsKd".to_string())).unwrap();
        assert_eq!(exact.as_slice().iter().filter(|w| **w > 0.0).count(), 1);

        let exact_cards_same = expand_range(&RangeSpec::Notation("AsAs".to_string()));
        assert!(matches!(exact_cards_same, Err(RfError::InvalidRangeSpec(_))));
    }

    #[test]
    fn expand_notation_random_and_duplicates_do_not_overlap() {
        let a = RangeSpec::Notation("AA".to_string());
        let aa = expand_range(&a).unwrap();
        let b = expand_range(&RangeSpec::Notation("AK".to_string())).unwrap();
        assert_ne!(aa.as_slice(), b.as_slice());
    }

    #[test]
    fn generator_functions_work() {
        let c0 = Card::from_index_unchecked(0);
        let c1 = Card::from_index_unchecked(1);
        let c2 = Card::from_index_unchecked(2);
        let cards = vec![c0, c1, c2];
        let unordered: Vec<[Card; 2]> = unordered_pairs(&cards).collect();
        let ordered: Vec<[Card; 2]> = ordered_pairs(&cards).collect();

        assert_eq!(unordered.len(), 3);
        assert_eq!(ordered.len(), 6);
        assert_eq!(unordered, vec![[c0, c1], [c0, c2], [c1, c2]]);
        assert_eq!(
            ordered,
            vec![[c0, c1], [c0, c2], [c1, c0], [c1, c2], [c2, c0], [c2, c1]]
        );
    }

    #[test]
    fn remaining_cards_respects_dead_cards() {
        let dead = CardMask::from_cards([c(0, 0), c(12, 3)]);
        let remaining: Vec<Card> = remaining_cards(dead).collect();
        assert_eq!(remaining.len(), 50);
        assert!(!remaining.contains(&c(0, 0)));
        assert!(!remaining.contains(&c(12, 3)));
        assert_eq!(remaining[0].index(), 1);
        assert_eq!(remaining[49].index(), 50);
    }

    #[test]
    fn all_combos_iterator_shape() {
        let all: Vec<(ComboId, HoleCards)> = all_combos().collect();
        assert_eq!(all.len(), NUM_HOLE_COMBOS);
        assert_eq!(all.first().unwrap().0.index(), 0);
        assert_eq!(all.last().unwrap().0.index(), 1325);
    }
}

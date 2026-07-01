use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum RfError {
    InvalidCard(String),
    DuplicateCard,
    InvalidBoardLength(usize),
    InvalidComboId(u16),
    InvalidRangeSpec(String),
    InvalidRangeWeight(String),
    DuplicateRangeCombo(String),
    EvalError(String),
}

impl fmt::Display for RfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RfError::InvalidCard(s) => write!(f, "invalid card: {s}"),
            RfError::DuplicateCard => write!(f, "duplicate card"),
            RfError::InvalidBoardLength(len) => write!(f, "invalid board length: {len}"),
            RfError::InvalidComboId(id) => write!(f, "invalid combo id: {id}"),
            RfError::InvalidRangeSpec(s) => write!(f, "invalid range spec: {s}"),
            RfError::InvalidRangeWeight(s) => write!(f, "invalid range weight: {s}"),
            RfError::DuplicateRangeCombo(s) => write!(f, "duplicate combo in range spec: {s}"),
            RfError::EvalError(s) => write!(f, "evaluation error with current hand: {s}"),
        }
    }
}

impl std::error::Error for RfError {}

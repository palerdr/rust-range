//! RangeForge's probability engine: distributions, inference, equity, and reports.

pub mod action;
mod action_json;
pub mod bayes;
pub mod distribution;
pub mod equity;
pub mod information;
pub mod metrics;
pub mod scenario;

pub use distribution::{DistributionError, RangeDistribution};

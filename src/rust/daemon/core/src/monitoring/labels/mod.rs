//! Metric-label utilities (PRD A1+).
//!
//! Helpers that keep Prometheus label cardinality bounded before values are
//! attached to collectors.

pub mod cardinality;

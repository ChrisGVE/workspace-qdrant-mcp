//! Cross-boundary graph-RAG traversal configuration (`search.graph_rag.*`).
//!
//! Controls the Rust-side fan-out caps applied to `query_cross_boundary`
//! results after the recursive CTE over-fetches (SQLite cannot rank-limit per
//! recursion level), plus the fusion alpha and the two-sided-confidence gate.

use serde::{Deserialize, Serialize};

/// Default fan-out cap: expanded nodes per direct (hop-1) source hit.
fn default_max_per_hit() -> usize {
    5
}

/// Default fan-out cap: nodes leaving any single ConceptNode.
fn default_max_per_concept() -> usize {
    8
}

/// Default fan-out cap: total expanded nodes across the whole result set.
fn default_max_total() -> usize {
    50
}

/// Default fusion weight on the vector score (`1 - alpha` weights graph proximity).
fn default_alpha() -> f64 {
    0.7
}

/// Configuration for cross-boundary graph-RAG traversal and fusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRagConfig {
    /// Maximum expanded nodes kept per direct (hop-1) source hit, top-K by
    /// `edge_confidence`.
    #[serde(default = "default_max_per_hit")]
    pub max_per_hit: usize,
    /// Maximum nodes kept leaving any single ConceptNode, top-K by
    /// `edge_confidence` — tames concept supernodes.
    #[serde(default = "default_max_per_concept")]
    pub max_per_concept: usize,
    /// Maximum total expanded nodes across the entire result set.
    #[serde(default = "default_max_total")]
    pub max_total: usize,
    /// Fusion weight on the vector score; graph proximity gets `1 - alpha`.
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    /// When set, only surface a cross-domain expansion when both the code-side
    /// and doc-side concept edges exceed the confidence floor (precision gate).
    #[serde(default)]
    pub two_sided_confidence: bool,
}

impl Default for GraphRagConfig {
    fn default() -> Self {
        Self {
            max_per_hit: default_max_per_hit(),
            max_per_concept: default_max_per_concept(),
            max_total: default_max_total(),
            alpha: default_alpha(),
            two_sided_confidence: false,
        }
    }
}

impl GraphRagConfig {
    /// Validate configuration settings.
    ///
    /// All caps must be non-zero and `alpha` must be in `[0.0, 1.0]`.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_per_hit == 0 {
            return Err("max_per_hit must be greater than 0".to_string());
        }
        if self.max_per_concept == 0 {
            return Err("max_per_concept must be greater than 0".to_string());
        }
        if self.max_total == 0 {
            return Err("max_total must be greater than 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(format!("alpha must be in [0.0, 1.0], got {}", self.alpha));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let c = GraphRagConfig::default();
        assert_eq!(c.max_per_hit, 5);
        assert_eq!(c.max_per_concept, 8);
        assert_eq!(c.max_total, 50);
        assert_eq!(c.alpha, 0.7);
        assert!(!c.two_sided_confidence);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_zero_caps() {
        for c in [
            GraphRagConfig {
                max_per_hit: 0,
                ..Default::default()
            },
            GraphRagConfig {
                max_per_concept: 0,
                ..Default::default()
            },
            GraphRagConfig {
                max_total: 0,
                ..Default::default()
            },
        ] {
            assert!(c.validate().is_err());
        }
    }

    #[test]
    fn test_validate_rejects_out_of_range_alpha() {
        let c = GraphRagConfig {
            alpha: 1.5,
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_serde_roundtrip_and_field_defaults() {
        let c: GraphRagConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c.max_per_hit, 5);
        assert_eq!(c.max_per_concept, 8);
        assert_eq!(c.max_total, 50);
        assert_eq!(c.alpha, 0.7);
        assert!(!c.two_sided_confidence);
    }
}

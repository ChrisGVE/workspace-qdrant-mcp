//! Concept-edge (IMPLEMENTS_CONCEPT / COVERS_TOPIC) configuration.
//!
//! Controls how symbol- and section-granular concept edges are emitted from
//! Tier-2 taxonomy classification during file ingestion.

use serde::{Deserialize, Serialize};

/// Default minimum cosine confidence for emitting a concept edge.
fn default_min_confidence() -> f64 {
    0.35
}

/// Default maximum number of concept edges emitted per code/narrative unit.
fn default_max_per_unit() -> usize {
    5
}

/// Configuration for concept-edge emission during ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptConfig {
    /// Minimum cosine similarity between a unit's embedding and a taxonomy
    /// term for an IMPLEMENTS_CONCEPT / COVERS_TOPIC edge to be created.
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f64,
    /// Maximum number of concept edges emitted per unit (symbol or section),
    /// keeping the highest-scoring matches.
    #[serde(default = "default_max_per_unit")]
    pub max_per_unit: usize,
}

impl Default for ConceptConfig {
    fn default() -> Self {
        Self {
            min_confidence: default_min_confidence(),
            max_per_unit: default_max_per_unit(),
        }
    }
}

impl ConceptConfig {
    /// Validate configuration settings.
    ///
    /// `min_confidence` must be in `[0.0, 1.0]` (a cosine threshold) and
    /// `max_per_unit` must be non-zero.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(format!(
                "min_confidence must be in [0.0, 1.0], got {}",
                self.min_confidence
            ));
        }
        if self.max_per_unit == 0 {
            return Err("max_per_unit must be greater than 0".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let c = ConceptConfig::default();
        assert_eq!(c.min_confidence, 0.35);
        assert_eq!(c.max_per_unit, 5);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_out_of_range_confidence() {
        let c = ConceptConfig {
            min_confidence: 1.5,
            max_per_unit: 5,
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_cap() {
        let c = ConceptConfig {
            min_confidence: 0.35,
            max_per_unit: 0,
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_serde_roundtrip_and_field_defaults() {
        // Missing fields fall back to defaults.
        let c: ConceptConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c.min_confidence, 0.35);
        assert_eq!(c.max_per_unit, 5);
    }
}

//! Narrative-extraction configuration.
//!
//! Controls thresholds and safety limits for the narrative graph layer
//! (DocumentSection / CodeComment / Docstring nodes and EXPLAINS edges).

use serde::{Deserialize, Serialize};

fn default_enabled() -> bool {
    true
}

fn default_max_input_kb() -> usize {
    256
}

fn default_explains_min_symbol_length() -> usize {
    4
}

fn default_explains_max_per_section() -> usize {
    10
}

fn default_comment_min_lines() -> usize {
    3
}

fn default_comment_symbol_proximity() -> usize {
    5
}

/// Configuration for the narrative extraction pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeConfig {
    /// Master switch for narrative extraction.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Skip narrative extraction for files larger than this (KB), bounding
    /// Aho-Corasick latency on pathological inputs.
    #[serde(default = "default_max_input_kb")]
    pub max_input_kb: usize,
    /// Minimum symbol-name length for an EXPLAINS edge (shorter names are too
    /// noisy to match reliably in prose).
    #[serde(default = "default_explains_min_symbol_length")]
    pub explains_min_symbol_length: usize,
    /// Maximum EXPLAINS edges emitted per document section.
    #[serde(default = "default_explains_max_per_section")]
    pub explains_max_per_section: usize,
    /// Minimum contiguous comment lines to form a CodeComment node.
    #[serde(default = "default_comment_min_lines")]
    pub comment_min_lines: usize,
    /// Maximum line distance from a comment block to the nearest symbol for a
    /// DESCRIBES/EXPLAINS link.
    #[serde(default = "default_comment_symbol_proximity")]
    pub comment_symbol_proximity: usize,
}

impl Default for NarrativeConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            max_input_kb: default_max_input_kb(),
            explains_min_symbol_length: default_explains_min_symbol_length(),
            explains_max_per_section: default_explains_max_per_section(),
            comment_min_lines: default_comment_min_lines(),
            comment_symbol_proximity: default_comment_symbol_proximity(),
        }
    }
}

impl NarrativeConfig {
    /// Maximum input size in bytes, derived from `max_input_kb`.
    pub fn max_input_bytes(&self) -> usize {
        self.max_input_kb.saturating_mul(1024)
    }

    /// Validate configuration settings.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_input_kb == 0 {
            return Err("max_input_kb must be greater than 0".to_string());
        }
        if self.explains_min_symbol_length == 0 {
            return Err("explains_min_symbol_length must be greater than 0".to_string());
        }
        if self.explains_max_per_section == 0 {
            return Err("explains_max_per_section must be greater than 0".to_string());
        }
        if self.comment_min_lines == 0 {
            return Err("comment_min_lines must be greater than 0".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let c = NarrativeConfig::default();
        assert!(c.enabled);
        assert_eq!(c.max_input_kb, 256);
        assert_eq!(c.explains_min_symbol_length, 4);
        assert_eq!(c.explains_max_per_section, 10);
        assert_eq!(c.comment_min_lines, 3);
        assert_eq!(c.comment_symbol_proximity, 5);
        assert_eq!(c.max_input_bytes(), 256 * 1024);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_zero_input_kb() {
        let mut c = NarrativeConfig::default();
        c.max_input_kb = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_caps() {
        let mut c = NarrativeConfig::default();
        c.explains_max_per_section = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_serde_field_defaults() {
        let c: NarrativeConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c.max_input_kb, 256);
        assert_eq!(c.explains_min_symbol_length, 4);
        assert!(c.enabled);
    }
}

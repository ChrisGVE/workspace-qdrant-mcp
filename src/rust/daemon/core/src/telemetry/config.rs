//! Configuration for the granular telemetry system
//!
//! Provides [`GranularTelemetryConfig`] with per-module level overrides and
//! serde support for YAML/JSON configuration files.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::levels::TelemetryLevel;

/// Configuration for the L0-L4 granular telemetry system.
///
/// The `default_level` applies to all modules that do not have an explicit
/// override in `module_overrides`.
///
/// # Example (YAML)
///
/// ```yaml
/// granular_telemetry:
///   enabled: true
///   default_level: l1
///   module_overrides:
///     storage: l2
///     processing: l3
///     watching: l0
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GranularTelemetryConfig {
    /// Master switch. When `false`, the [`TelemetryLayer`](super::TelemetryLayer)
    /// suppresses all telemetry events regardless of level.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// The baseline level applied to modules without an explicit override.
    #[serde(default = "default_level")]
    pub default_level: TelemetryLevel,

    /// Per-module level overrides.
    ///
    /// Keys are module names (e.g. `"storage"`, `"processing"`, `"watching"`).
    /// Values are the maximum telemetry level to emit for that module.
    #[serde(default)]
    pub module_overrides: HashMap<String, TelemetryLevel>,
}

fn default_enabled() -> bool {
    true
}

fn default_level() -> TelemetryLevel {
    TelemetryLevel::L0
}

impl Default for GranularTelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            default_level: default_level(),
            module_overrides: HashMap::new(),
        }
    }
}

impl GranularTelemetryConfig {
    /// Return the effective telemetry level for a given module path.
    ///
    /// Checks `module_overrides` first against the full path, then each
    /// segment of the module path (e.g. for `"crate::storage::upsert"`,
    /// tries `"crate::storage::upsert"`, `"storage::upsert"`, `"upsert"`,
    /// and each individual segment `"crate"`, `"storage"`, `"upsert"`).
    /// Falls back to `default_level`.
    pub fn effective_level(&self, module_path: &str) -> TelemetryLevel {
        if !self.enabled {
            // When disabled, nothing passes; return the lowest possible
            // level so the subscriber can reject everything above L0.
            return TelemetryLevel::L0;
        }

        // Try the full module path first.
        if let Some(&level) = self.module_overrides.get(module_path) {
            return level;
        }

        // Try each individual segment of the path (e.g. "storage" from
        // "workspace_qdrant_core::storage::upsert").
        for segment in module_path.split("::") {
            if let Some(&level) = self.module_overrides.get(segment) {
                return level;
            }
        }

        self.default_level
    }

    /// Validate the configuration.
    ///
    /// Returns an error message if any module override key is empty.
    pub fn validate(&self) -> Result<(), String> {
        for key in self.module_overrides.keys() {
            if key.is_empty() {
                return Err(
                    "module_overrides contains an empty key".to_string(),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_enabled_at_l0() {
        let cfg = GranularTelemetryConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.default_level, TelemetryLevel::L0);
        assert!(cfg.module_overrides.is_empty());
    }

    #[test]
    fn effective_level_uses_default() {
        let cfg = GranularTelemetryConfig {
            default_level: TelemetryLevel::L1,
            ..Default::default()
        };
        assert_eq!(cfg.effective_level("storage"), TelemetryLevel::L1);
    }

    #[test]
    fn effective_level_with_override() {
        let mut cfg = GranularTelemetryConfig::default();
        cfg.module_overrides
            .insert("storage".to_string(), TelemetryLevel::L3);
        assert_eq!(cfg.effective_level("storage"), TelemetryLevel::L3);
        // Other modules fall back to default
        assert_eq!(cfg.effective_level("processing"), TelemetryLevel::L0);
    }

    #[test]
    fn effective_level_matches_path_segment() {
        let mut cfg = GranularTelemetryConfig::default();
        cfg.module_overrides
            .insert("storage".to_string(), TelemetryLevel::L2);
        // Should match "storage" as a segment within the full path.
        assert_eq!(
            cfg.effective_level("workspace_qdrant_core::storage::upsert"),
            TelemetryLevel::L2
        );
    }

    #[test]
    fn effective_level_full_path_takes_precedence() {
        let mut cfg = GranularTelemetryConfig::default();
        cfg.module_overrides
            .insert("storage".to_string(), TelemetryLevel::L1);
        cfg.module_overrides.insert(
            "workspace_qdrant_core::storage::upsert".to_string(),
            TelemetryLevel::L4,
        );
        assert_eq!(
            cfg.effective_level("workspace_qdrant_core::storage::upsert"),
            TelemetryLevel::L4
        );
    }

    #[test]
    fn disabled_config_returns_l0() {
        let cfg = GranularTelemetryConfig {
            enabled: false,
            default_level: TelemetryLevel::L4,
            module_overrides: HashMap::new(),
        };
        assert_eq!(cfg.effective_level("anything"), TelemetryLevel::L0);
    }

    #[test]
    fn serde_roundtrip_json() {
        let mut cfg = GranularTelemetryConfig::default();
        cfg.default_level = TelemetryLevel::L2;
        cfg.module_overrides
            .insert("watching".to_string(), TelemetryLevel::L0);

        let json = serde_json::to_string(&cfg).unwrap();
        let back: GranularTelemetryConfig =
            serde_json::from_str(&json).unwrap();

        assert_eq!(back.enabled, cfg.enabled);
        assert_eq!(back.default_level, cfg.default_level);
        assert_eq!(
            back.module_overrides.get("watching"),
            Some(&TelemetryLevel::L0)
        );
    }

    #[test]
    fn validate_rejects_empty_key() {
        let mut cfg = GranularTelemetryConfig::default();
        cfg.module_overrides
            .insert(String::new(), TelemetryLevel::L1);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_normal_config() {
        let mut cfg = GranularTelemetryConfig::default();
        cfg.module_overrides
            .insert("storage".to_string(), TelemetryLevel::L2);
        assert!(cfg.validate().is_ok());
    }
}

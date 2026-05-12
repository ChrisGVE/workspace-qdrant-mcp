//! Auto-ingestion configuration

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Auto-ingestion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoIngestionConfig {
    pub enabled: bool,
    pub auto_create_watches: bool,
    pub include_common_files: bool,
    pub include_source_files: bool,
    pub target_collection_suffix: String,
    pub max_files_per_batch: usize,
    pub batch_delay_seconds: f64,
    pub max_file_size_mb: usize,
    pub recursive_depth: usize,
    pub debounce_seconds: u64,
}

/// Per-extension maximum ingestion size limits.
///
/// Key: lowercase extension without leading dot (e.g. "json", "yaml", "csv").
/// Value: size limit in KB. Absent = no limit. 0 = skip all files of that extension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionLimitsConfig {
    #[serde(default = "default_extension_size_limits")]
    pub extension_size_limits_kb: HashMap<String, u64>,
}

fn default_extension_size_limits() -> HashMap<String, u64> {
    [
        ("json", 500),
        ("jsonc", 500),
        ("json5", 500),
        ("jsonl", 500),
        ("ndjson", 500),
        ("yaml", 500),
        ("yml", 500),
        ("toml", 500),
        ("xml", 500),
        ("xsl", 500),
        ("xslt", 500),
        ("csv", 500),
        ("tsv", 500),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

impl Default for IngestionLimitsConfig {
    fn default() -> Self {
        Self {
            extension_size_limits_kb: default_extension_size_limits(),
        }
    }
}

impl IngestionLimitsConfig {
    /// Returns the size limit in bytes for a given extension, or None if unlimited.
    ///
    /// Accepts extension with or without leading dot (e.g. "json" or ".json").
    pub fn size_limit_bytes(&self, extension: &str) -> Option<u64> {
        self.extension_size_limits_kb
            .get(extension.trim_start_matches('.'))
            .map(|kb| kb * 1024)
    }

    /// Validate configuration settings.
    ///
    /// Extension names must be non-empty and size limits must be in
    /// the range (0, 1_048_576] KB (max 1 GB). A value of 0 means "skip
    /// all files of that extension" and is disallowed; use the absence of
    /// the key to express "no limit" instead.
    pub fn validate(&self) -> Result<(), String> {
        const MAX_LIMIT_KB: u64 = 1_048_576; // 1 GB
        for (ext, &limit_kb) in &self.extension_size_limits_kb {
            if ext.is_empty() {
                return Err("extension_size_limits_kb contains an empty extension key".to_string());
            }
            if limit_kb == 0 {
                return Err(format!(
                    "extension_size_limits_kb[{ext}] must be greater than 0 (use key absence for \
                     no-limit)"
                ));
            }
            if limit_kb > MAX_LIMIT_KB {
                return Err(format!(
                    "extension_size_limits_kb[{ext}] must not exceed {MAX_LIMIT_KB} KB (1 GB), \
                     got {limit_kb}"
                ));
            }
        }
        Ok(())
    }
}

impl Default for AutoIngestionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_create_watches: true,
            include_common_files: true,
            include_source_files: true,
            target_collection_suffix: "scratchbook".to_string(),
            max_files_per_batch: 5,
            batch_delay_seconds: 2.0,
            max_file_size_mb: 50,
            recursive_depth: 5,
            debounce_seconds: 10,
        }
    }
}

impl AutoIngestionConfig {
    /// Validate configuration settings.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_files_per_batch == 0 {
            return Err("max_files_per_batch must be greater than 0".to_string());
        }
        if !self.batch_delay_seconds.is_finite() || self.batch_delay_seconds < 0.0 {
            return Err("batch_delay_seconds must be a finite non-negative number".to_string());
        }
        if self.max_file_size_mb == 0 {
            return Err("max_file_size_mb must be greater than 0".to_string());
        }
        if self.max_file_size_mb >= 1000 {
            return Err("max_file_size_mb must be less than 1000".to_string());
        }
        if self.recursive_depth > 20 {
            return Err("recursive_depth must not exceed 20".to_string());
        }
        if self.debounce_seconds > 600 {
            return Err("debounce_seconds must not exceed 600".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingestion_limits_default_has_13_extensions() {
        let cfg = IngestionLimitsConfig::default();
        assert_eq!(cfg.extension_size_limits_kb.len(), 13);
        for ext in &[
            "json", "jsonc", "json5", "jsonl", "ndjson", "yaml", "yml", "toml", "xml", "xsl",
            "xslt", "csv", "tsv",
        ] {
            assert!(
                cfg.extension_size_limits_kb.contains_key(*ext),
                "missing extension: {ext}"
            );
            assert_eq!(cfg.extension_size_limits_kb[*ext], 500);
        }
    }

    #[test]
    fn test_size_limit_bytes_present() {
        let cfg = IngestionLimitsConfig::default();
        assert_eq!(cfg.size_limit_bytes("json"), Some(500 * 1024));
    }

    #[test]
    fn test_size_limit_bytes_with_leading_dot() {
        let cfg = IngestionLimitsConfig::default();
        assert_eq!(cfg.size_limit_bytes(".json"), Some(500 * 1024));
        assert_eq!(cfg.size_limit_bytes(".yaml"), Some(500 * 1024));
    }

    #[test]
    fn test_size_limit_bytes_absent_returns_none() {
        let cfg = IngestionLimitsConfig::default();
        assert_eq!(cfg.size_limit_bytes("py"), None);
        assert_eq!(cfg.size_limit_bytes("rs"), None);
        assert_eq!(cfg.size_limit_bytes("d.ts"), None);
    }

    #[test]
    fn test_size_limit_bytes_zero_returns_zero() {
        let mut cfg = IngestionLimitsConfig::default();
        cfg.extension_size_limits_kb.insert("json".to_string(), 0);
        assert_eq!(cfg.size_limit_bytes("json"), Some(0));
    }

    #[test]
    fn test_serde_roundtrip() {
        let cfg = IngestionLimitsConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: IngestionLimitsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(
            cfg.extension_size_limits_kb.len(),
            cfg2.extension_size_limits_kb.len()
        );
        assert_eq!(cfg2.size_limit_bytes("json"), Some(500 * 1024));
    }

    // ── IngestionLimitsConfig::validate ─────────────────────────────────────

    #[test]
    fn test_ingestion_limits_validate_default_ok() {
        assert!(IngestionLimitsConfig::default().validate().is_ok());
    }

    #[test]
    fn test_ingestion_limits_validate_rejects_zero_limit() {
        let mut cfg = IngestionLimitsConfig::default();
        cfg.extension_size_limits_kb.insert("rs".to_string(), 0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_ingestion_limits_validate_rejects_over_max() {
        let mut cfg = IngestionLimitsConfig::default();
        cfg.extension_size_limits_kb
            .insert("rs".to_string(), 1_048_577);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_ingestion_limits_validate_accepts_max_boundary() {
        let mut cfg = IngestionLimitsConfig::default();
        cfg.extension_size_limits_kb
            .insert("rs".to_string(), 1_048_576);
        assert!(cfg.validate().is_ok());
    }

    // ── AutoIngestionConfig::validate ───────────────────────────────────────

    #[test]
    fn test_auto_ingestion_validate_default_ok() {
        assert!(AutoIngestionConfig::default().validate().is_ok());
    }

    #[test]
    fn test_auto_ingestion_validate_rejects_zero_batch() {
        let mut cfg = AutoIngestionConfig::default();
        cfg.max_files_per_batch = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_auto_ingestion_validate_rejects_negative_batch_delay() {
        let mut cfg = AutoIngestionConfig::default();
        cfg.batch_delay_seconds = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_auto_ingestion_validate_rejects_infinite_batch_delay() {
        let mut cfg = AutoIngestionConfig::default();
        cfg.batch_delay_seconds = f64::INFINITY;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_auto_ingestion_validate_rejects_zero_file_size() {
        let mut cfg = AutoIngestionConfig::default();
        cfg.max_file_size_mb = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_auto_ingestion_validate_rejects_file_size_gte_1000() {
        let mut cfg = AutoIngestionConfig::default();
        cfg.max_file_size_mb = 1000;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_auto_ingestion_validate_rejects_recursive_depth_over_20() {
        let mut cfg = AutoIngestionConfig::default();
        cfg.recursive_depth = 21;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_auto_ingestion_validate_accepts_recursive_depth_20() {
        let mut cfg = AutoIngestionConfig::default();
        cfg.recursive_depth = 20;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_auto_ingestion_validate_rejects_debounce_over_600() {
        let mut cfg = AutoIngestionConfig::default();
        cfg.debounce_seconds = 601;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_auto_ingestion_validate_accepts_debounce_600() {
        let mut cfg = AutoIngestionConfig::default();
        cfg.debounce_seconds = 600;
        assert!(cfg.validate().is_ok());
    }
}

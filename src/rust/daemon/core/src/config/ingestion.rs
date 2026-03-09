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
}

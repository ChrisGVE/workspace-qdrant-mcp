//! Embedded YAML defaults parsed at startup
//!
//! Provides `DEFAULT_YAML` (the raw YAML string) and `DEFAULT_YAML_CONFIG`
//! (a lazily-parsed struct). Both daemon and CLI import these to derive their
//! defaults from the single source of truth: `assets/default_configuration.yaml`.

mod infrastructure;
mod processing;

pub use infrastructure::*;
pub use processing::*;

use serde::Deserialize;
use std::sync::LazyLock;

/// The raw default configuration YAML, embedded at compile time.
pub static DEFAULT_YAML: &str = include_str!("../../../../../assets/default_configuration.yaml");

/// Parsed default configuration, initialized once on first access.
pub static DEFAULT_YAML_CONFIG: LazyLock<YamlConfig> = LazyLock::new(|| {
    serde_yml::from_str(DEFAULT_YAML).expect("default_configuration.yaml must parse successfully")
});

// =============================================================================
// Duration string parsing
// =============================================================================

/// Parse a duration string like "30s", "5m", "1h" into milliseconds.
///
/// Supported suffixes: `ms`, `s`, `m`, `h`.
/// Bare numbers are treated as seconds.
pub fn parse_duration_to_ms(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    if let Some(num) = s.strip_suffix("ms") {
        return num.trim().parse::<u64>().ok();
    }
    if let Some(num) = s.strip_suffix('s') {
        return num.trim().parse::<u64>().ok().map(|n| n * 1000);
    }
    if let Some(num) = s.strip_suffix('m') {
        return num.trim().parse::<u64>().ok().map(|n| n * 60_000);
    }
    if let Some(num) = s.strip_suffix('h') {
        return num.trim().parse::<u64>().ok().map(|n| n * 3_600_000);
    }

    // Bare number → seconds
    s.parse::<u64>().ok().map(|n| n * 1000)
}

/// Parse a size string like "50MB", "100KB" into bytes.
pub fn parse_size_to_bytes(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    if let Some(num) = s.strip_suffix("GB") {
        return num.trim().parse::<u64>().ok().map(|n| n * 1_073_741_824);
    }
    if let Some(num) = s.strip_suffix("MB") {
        return num.trim().parse::<u64>().ok().map(|n| n * 1_048_576);
    }
    if let Some(num) = s.strip_suffix("KB") {
        return num.trim().parse::<u64>().ok().map(|n| n * 1024);
    }
    if let Some(num) = s.strip_suffix('B') {
        return num.trim().parse::<u64>().ok();
    }

    // Bare number → bytes
    s.parse::<u64>().ok()
}

// =============================================================================
// Serde helpers for duration strings
// =============================================================================

pub(crate) mod duration_serde {
    use serde::{self, Deserialize, Deserializer};

    /// Deserialize a duration string (e.g. "30s") into milliseconds (u64).
    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        super::parse_duration_to_ms(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("invalid duration: {}", s)))
    }
}

#[allow(dead_code)]
pub(crate) mod optional_duration_serde {
    use serde::{self, Deserialize, Deserializer};

    /// Deserialize an optional duration string into `Option<u64>` milliseconds.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt: Option<String> = Option::deserialize(deserializer)?;
        match opt {
            Some(s) => super::parse_duration_to_ms(&s)
                .map(Some)
                .ok_or_else(|| serde::de::Error::custom(format!("invalid duration: {}", s))),
            None => Ok(None),
        }
    }
}

// =============================================================================
// Top-level YAML config struct
// =============================================================================

/// Top-level configuration parsed from `default_configuration.yaml`.
///
/// Only the sections used by daemon and CLI are included. Unknown sections
/// are silently ignored via `#[serde(default)]`.
// Note: Do NOT parse DEFAULT_YAML in Default — that would recurse infinitely
// because serde's `#[serde(default)]` calls Default for missing fields during parsing.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct YamlConfig {
    pub qdrant: YamlQdrantConfig,
    pub grpc: YamlGrpcConfig,
    pub auto_ingestion: YamlAutoIngestionConfig,
    pub queue_processor: YamlQueueProcessorConfig,
    pub git: YamlGitConfig,
    pub embedding: YamlEmbeddingConfig,
    pub lsp: YamlLspConfig,
    pub grammars: YamlGrammarsConfig,
    pub updates: YamlUpdatesConfig,
    pub performance: YamlPerformanceConfig,
    pub watching: YamlWatchingConfig,
    pub observability: YamlObservabilityConfig,
    pub resource_limits: YamlResourceLimitsConfig,
    pub tagging: YamlTaggingConfig,
}

#[cfg(test)]
mod tests;

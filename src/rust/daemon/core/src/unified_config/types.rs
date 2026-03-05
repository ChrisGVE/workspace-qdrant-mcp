//! Error and format types for unified configuration

use std::path::Path;
use std::path::PathBuf;

/// Error types for unified configuration operations
#[derive(Debug, thiserror::Error)]
pub enum UnifiedConfigError {
    #[error("Configuration file not found: {0}")]
    FileNotFound(PathBuf),

    #[error("Configuration parsing error: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("YAML parsing error: {0}")]
    YamlError(String),

    #[error("Configuration validation error: {0}")]
    ValidationError(String),
}

/// Supported configuration formats (YAML only)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFormat {
    Yaml,
}

impl ConfigFormat {
    /// Detect format from file extension (YAML only)
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("yaml") | Some("yml") => ConfigFormat::Yaml,
            _ => ConfigFormat::Yaml, // Default to YAML
        }
    }
}

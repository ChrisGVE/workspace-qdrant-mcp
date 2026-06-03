//! Daemon configuration loading errors (WI-a2).
//!
//! Replaces the deleted `unified_config::UnifiedConfigError`. The loader maps
//! the shared `wqm_common::config` failures and IO/parse problems into this
//! single daemon-local enum; `error::mod` converts it into `WorkspaceError`.

use std::path::PathBuf;

/// Errors produced while discovering, parsing, or validating daemon config.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// An explicitly requested config file does not exist.
    #[error("Configuration file not found: {0}")]
    FileNotFound(PathBuf),

    /// Reading a config file failed.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// The config file content could not be parsed.
    #[error("Configuration parsing error: {0}")]
    Parse(String),

    /// The resolved configuration failed validation.
    #[error("Configuration validation error: {0}")]
    Validation(String),
}

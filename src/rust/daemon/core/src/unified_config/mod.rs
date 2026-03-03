//! Unified configuration integration for Rust daemon
//!
//! This module loads configuration from the canonical search paths (provided
//! by `wqm_common::paths`), applies environment variable overrides, and
//! validates the result. No project-local `.workspace-qdrant.yaml` is searched.

mod env_overrides;
mod manager;
mod tests;
mod types;
mod validation;

// Public API — all paths remain identical to the previous flat module
pub use manager::UnifiedConfigManager;
pub use types::{ConfigFormat, UnifiedConfigError};

// Re-export from wqm_common for backward compatibility
pub use wqm_common::env_expand::{expand_env_vars, expand_path_env_vars};

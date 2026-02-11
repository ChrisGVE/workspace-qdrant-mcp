//! Shared configuration and path resolution for workspace-qdrant-mcp
//!
//! This crate provides canonical implementations used by both the daemon and CLI:
//! - Configuration file search paths
//! - Database and config directory resolution
//! - Environment variable expansion
//! - YAML-derived default configuration values

pub mod env_expand;
pub mod paths;
pub mod yaml_defaults;

//! LSP Configuration Module
//!
//! This module handles configuration management for LSP servers including
//! language-specific settings, server parameters, and timeout configurations.
//!
//! Split into submodules by responsibility:
//! - `lsp_config` - Main configuration struct, feature flags, validation, file I/O
//! - `language_config` - Per-language defaults and configuration types
//! - `server_config` - Per-server configuration, capabilities, restart policies

mod language_config;
mod lsp_config;
mod server_config;

pub use language_config::LanguageConfig;
pub use lsp_config::{LspConfig, LspFeatures};
pub use server_config::{RestartPolicyOverride, ServerCapabilitiesOverride, ServerConfig};

//! LSP (Language Server Protocol) Integration Module
//!
//! This module provides comprehensive LSP server detection, lifecycle management,
//! and communication capabilities for the workspace-qdrant-mcp daemon.
//!
//! # Architecture Overview
//!
//! The LSP integration is designed around several key components:
//!
//! - **Server Detection**: Automatic discovery of LSP servers via PATH scanning
//! - **Lifecycle Management**: Server startup, health monitoring, restart, and shutdown
//! - **Communication**: JSON-RPC protocol abstraction over stdio/TCP
//! - **Configuration**: Per-language server configuration and parameters
//! - **Error Handling**: Circuit breaker pattern and resilient operation
//!
//! # Supported Languages
//!
//! The system is designed to support multiple programming languages:
//!
//! - **Python**: `ruff-lsp`, `pylsp`, `pyright-langserver`
//! - **Rust**: `rust-analyzer`
//! - **TypeScript/JavaScript**: `typescript-language-server`, `vscode-json-languageserver`
//! - **C/C++**: `clangd`, `ccls`
//! - **Go**: `gopls`
//! - **Java**: `jdtls`
//! - **And more extensibly**
//!
//! # Usage Example
//!
//! ```rust,no_run
//! use workspace_qdrant_core::lsp::{LanguageServerManager, ProjectLspConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ProjectLspConfig::default();
//!     let manager = LanguageServerManager::new(config).await?;
//!
//!     // The manager handles per-project LSP server lifecycle
//!
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod detection;
pub mod lifecycle;
pub mod communication;
pub mod config;
pub mod project_manager;

// NOTE: The `state` module (StateManager) was removed as part of 3-table SQLite compliance.
// The LspManager struct that used StateManager was never instantiated in production.
// The daemon uses LanguageServerManager directly, which works without SQLite persistence.

#[cfg(test)]
mod tests;

pub use detection::{
    LspServerDetector, DetectedServer, ServerCapabilities,
    ProjectLanguageDetector, ProjectLanguageResult, LanguageMarker,
};
pub use lifecycle::{LspServerManager, ServerInstance, ServerStatus};
pub use communication::{JsonRpcClient, JsonRpcMessage, JsonRpcRequest, JsonRpcResponse};
pub use config::{LspConfig, LanguageConfig, ServerConfig};
pub use project_manager::{
    LanguageServerManager, ProjectLspConfig, ProjectLspError, ProjectLspResult,
    ProjectLanguageKey, ProjectServerState, ProjectLspStats,
    Reference, TypeInfo, ResolvedImport, LspEnrichment, EnrichmentStatus,
};

/// Main errors that can occur in the LSP subsystem
#[derive(Error, Debug)]
pub enum LspError {
    #[error("Server not found: {server_name}")]
    ServerNotFound { server_name: String },

    #[error("Communication error: {message}")]
    Communication { message: String },

    #[error("Server initialization failed: {server_name} - {reason}")]
    InitializationFailed { server_name: String, reason: String },

    #[error("Health check failed: {server_name}")]
    HealthCheckFailed { server_name: String },

    #[error("Configuration error: {message}")]
    Configuration { message: String },

    #[error("State management error: {message}")]
    StateManagement { message: String },

    #[error("JSON-RPC error: {message}")]
    JsonRpc { message: String },

    #[error("Timeout occurred: {operation}")]
    Timeout { operation: String },

    #[error("IO error: {source}")]
    Io { #[from] source: std::io::Error },

    #[error("Database error: {source}")]
    Database { #[from] source: sqlx::Error },

    #[error("Serialization error: {source}")]
    Serialization { #[from] source: serde_json::Error },

    #[error("Date parsing error: {source}")]
    DateParsing { #[from] source: chrono::ParseError },

    #[error("UUID parsing error: {source}")]
    UuidParsing { #[from] source: uuid::Error },
}

/// Result type for LSP operations
pub type LspResult<T> = Result<T, LspError>;

/// Language identifiers supported by the LSP system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    Python,
    Rust,
    TypeScript,
    JavaScript,
    Json,
    Go,
    Java,
    C,
    Cpp,
    Ruby,
    Php,
    Shell,
    Yaml,
    Toml,
    Xml,
    Html,
    Css,
    Sql,
    Other(String),
}

impl Language {
    /// Get the language identifier string used by LSP servers
    pub fn identifier(&self) -> &str {
        match self {
            Language::Python => "python",
            Language::Rust => "rust",
            Language::TypeScript => "typescript",
            Language::JavaScript => "javascript",
            Language::Json => "json",
            Language::Go => "go",
            Language::Java => "java",
            Language::C => "c",
            Language::Cpp => "cpp",
            Language::Ruby => "ruby",
            Language::Php => "php",
            Language::Shell => "shellscript",
            Language::Yaml => "yaml",
            Language::Toml => "toml",
            Language::Xml => "xml",
            Language::Html => "html",
            Language::Css => "css",
            Language::Sql => "sql",
            Language::Other(s) => s,
        }
    }

    /// Create language from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "py" => Language::Python,
            "rs" => Language::Rust,
            "ts" => Language::TypeScript,
            "js" | "mjs" => Language::JavaScript,
            "json" => Language::Json,
            "go" => Language::Go,
            "java" => Language::Java,
            "c" | "h" => Language::C,
            "cpp" | "cc" | "cxx" | "hpp" => Language::Cpp,
            "rb" => Language::Ruby,
            "php" => Language::Php,
            "sh" | "bash" => Language::Shell,
            "yaml" | "yml" => Language::Yaml,
            "toml" => Language::Toml,
            "xml" => Language::Xml,
            "html" | "htm" => Language::Html,
            "css" => Language::Css,
            "sql" => Language::Sql,
            other => Language::Other(other.to_string()),
        }
    }

    /// Get common file extensions for this language
    pub fn extensions(&self) -> &[&str] {
        match self {
            Language::Python => &["py", "pyw", "pyi"],
            Language::Rust => &["rs"],
            Language::TypeScript => &["ts", "tsx"],
            Language::JavaScript => &["js", "mjs", "jsx"],
            Language::Json => &["json"],
            Language::Go => &["go"],
            Language::Java => &["java"],
            Language::C => &["c", "h"],
            Language::Cpp => &["cpp", "cc", "cxx", "hpp", "hxx"],
            Language::Ruby => &["rb"],
            Language::Php => &["php"],
            Language::Shell => &["sh", "bash"],
            Language::Yaml => &["yaml", "yml"],
            Language::Toml => &["toml"],
            Language::Xml => &["xml"],
            Language::Html => &["html", "htm"],
            Language::Css => &["css", "scss", "sass"],
            Language::Sql => &["sql"],
            Language::Other(_) => &[],
        }
    }
}

/// Priority levels for LSP operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LspPriority {
    /// Critical operations that must complete immediately
    Critical = 0,
    /// High priority operations for active development
    High = 1,
    /// Medium priority for background analysis
    Medium = 2,
    /// Low priority for maintenance tasks
    Low = 3,
}

// NOTE: LspManager struct was removed as part of 3-table SQLite compliance.
// It used StateManager which created 5 non-compliant SQLite tables.
// The daemon uses LanguageServerManager directly (from project_manager module)
// which provides per-project LSP lifecycle management without SQLite persistence.


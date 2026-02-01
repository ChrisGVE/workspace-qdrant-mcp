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
//! - **State Management**: SQLite-based persistence for server metadata
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
//! use workspace_qdrant_core::lsp::{LspManager, LspConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = LspConfig::default();
//!     let mut manager = LspManager::new(config).await?;
//!     
//!     // Start LSP manager
//!     manager.start().await?;
//!     
//!     // The manager will automatically detect and manage LSP servers
//!     
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{RwLock, Mutex};

pub mod detection;
pub mod lifecycle;
pub mod communication;
pub mod state;
pub mod config;
pub mod project_manager;

#[cfg(test)]
mod tests;

pub use detection::{LspServerDetector, DetectedServer, ServerCapabilities};
pub use lifecycle::{LspServerManager, ServerInstance, ServerStatus};
pub use communication::{JsonRpcClient, JsonRpcMessage, JsonRpcRequest, JsonRpcResponse};
pub use state::{StateManager};
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

/// Central LSP management system
pub struct LspManager {
    config: LspConfig,
    detector: LspServerDetector,
    manager: LspServerManager,
    state: StateManager,
    servers: RwLock<HashMap<Language, ServerInstance>>,
    shutdown_signal: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

impl LspManager {
    /// Create a new LSP manager with the given configuration
    pub async fn new(config: LspConfig) -> LspResult<Self> {
        tracing::info!("Initializing LSP Manager with configuration");
        
        let detector = LspServerDetector::new();
        let manager = LspServerManager::new(config.clone()).await?;
        let state = StateManager::new(&config.database_path).await?;
        
        Ok(Self {
            config,
            detector,
            manager,
            state,
            servers: RwLock::new(HashMap::new()),
            shutdown_signal: Mutex::new(None),
        })
    }

    /// Start the LSP manager
    pub async fn start(&mut self) -> LspResult<()> {
        tracing::info!("Starting LSP Manager");

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        *self.shutdown_signal.lock().await = Some(shutdown_tx);

        // Initialize state database
        self.state.initialize().await?;

        // Detect available LSP servers
        let detected_servers = self.detector.detect_servers().await?;
        tracing::info!("Detected {} LSP servers", detected_servers.len());

        // Initialize detected servers
        for server in detected_servers {
            if let Err(e) = self.initialize_server(server).await {
                tracing::warn!("Failed to initialize server: {}", e);
            }
        }

        // Start background tasks
        self.start_background_tasks(shutdown_rx).await?;

        tracing::info!("LSP Manager started successfully");
        Ok(())
    }

    /// Initialize a detected LSP server
    async fn initialize_server(&mut self, detected: DetectedServer) -> LspResult<()> {
        tracing::info!("Initializing LSP server: {} for {:?}", 
                      detected.name, detected.languages);

        // Create server instance
        let instance = self.manager.create_instance(detected).await?;
        
        // Store metadata in state database
        self.state.store_server_metadata(&instance.metadata()).await?;

        // Add to active servers for primary language
        if let Some(primary_lang) = instance.primary_language() {
            self.servers.write().await.insert(primary_lang, instance);
        }

        Ok(())
    }

    /// Start background maintenance tasks
    async fn start_background_tasks(
        &self,
        mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
    ) -> LspResult<()> {
        let health_check_interval = self.config.health_check_interval;
        let state = self.state.clone();
        let manager = self.manager.clone();

        // Health check task
        let health_check_task = async move {
            let mut interval = tokio::time::interval(health_check_interval);
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => {
                        tracing::info!("Shutting down LSP health check task");
                        break;
                    }
                    _ = interval.tick() => {
                        if let Err(e) = Self::perform_health_checks(&state, &manager).await {
                            tracing::warn!("Health check failed: {}", e);
                        }
                    }
                }
            }
        };

        // Spawn health check task
        tokio::spawn(health_check_task);

        Ok(())
    }

    /// Perform health checks on all active servers
    async fn perform_health_checks(
        state: &StateManager,
        manager: &LspServerManager,
    ) -> LspResult<()> {
        let servers = manager.get_all_instances().await;
        
        for instance in servers {
            match instance.health_check().await {
                Ok(metrics) => {
                    state.update_health_metrics(&instance.id(), &metrics).await?;
                }
                Err(e) => {
                    tracing::warn!("Health check failed for server {}: {}", 
                                 instance.id(), e);
                    
                    // Note: Restart would be handled by a separate task with mutable access
                }
            }
        }

        Ok(())
    }

    /// Get LSP server for a specific language
    pub async fn get_server(&self, language: &Language) -> Option<ServerInstance> {
        self.servers.read().await.get(language).cloned()
    }

    /// Get all active servers
    pub async fn get_all_servers(&self) -> HashMap<Language, ServerInstance> {
        self.servers.read().await.clone()
    }

    /// Get manager statistics
    pub async fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        let servers = self.servers.read().await;
        stats.insert("active_servers".to_string(), 
                    serde_json::Value::Number(servers.len().into()));
        
        let mut language_stats = HashMap::new();
        for (lang, instance) in servers.iter() {
            language_stats.insert(
                lang.identifier().to_string(),
                serde_json::json!({
                    "status": instance.status().await,
                    "uptime_seconds": instance.uptime().await.as_secs(),
                })
            );
        }
        stats.insert("languages".to_string(), serde_json::Value::Object(
            language_stats.into_iter().map(|(k, v)| (k, v)).collect()
        ));
        
        stats
    }

    /// Graceful shutdown
    pub async fn shutdown(&mut self) -> LspResult<()> {
        tracing::info!("Shutting down LSP Manager");

        // Signal shutdown to background tasks
        if let Some(shutdown_tx) = self.shutdown_signal.lock().await.take() {
            let _ = shutdown_tx.send(());
        }

        // Shutdown all server instances
        let mut servers = self.servers.write().await;
        let servers_to_shutdown: Vec<_> = servers.drain().collect();
        drop(servers); // Release the write lock
        
        for (language, mut instance) in servers_to_shutdown {
            tracing::debug!("Shutting down LSP server for {:?}", language);
            if let Err(e) = instance.shutdown().await {
                tracing::warn!("Error shutting down server for {:?}: {}", language, e);
            }
        }

        // Close state database
        self.state.close().await?;

        tracing::info!("LSP Manager shutdown complete");
        Ok(())
    }
}


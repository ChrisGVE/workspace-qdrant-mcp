//! Project-specific LSP Server Management
//!
//! This module provides per-project LSP server management, enabling
//! code intelligence features for active projects. It extends the base
//! LSP module with project-specific lifecycle management and enrichment
//! query capabilities.
//!
//! # Architecture
//!
//! The `LanguageServerManager` maintains a mapping of (project_id, language)
//! to active language server instances. Servers are started when a project
//! becomes active and stopped when the project is deprioritized.
//!
//! # Module structure
//!
//! - [`lifecycle`]: Start/stop/restart server delegation
//! - [`enrichment`]: Chunk enrichment, references, type info, imports
//! - [`health`]: Periodic health checks, crash handling, backoff
//! - [`activity`]: Project activation and deactivation tracking
//! - [`metrics`]: Usage metrics and statistics

mod activity;
mod enrichment;
mod health;
mod imports;
mod lifecycle;
mod metrics;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;

use super::{Language, LspConfig, LspError, ServerInstance, ServerStatus};
use crate::config::LspSettings;

// Re-export public types from submodules for backward compatibility
pub use metrics::ProjectLspStats;

// NOTE: StateManager and persistence removed as part of 3-table SQLite compliance.
// The LanguageServerManager now operates entirely in-memory without SQLite persistence.

/// Errors specific to project-level LSP management
#[derive(Error, Debug)]
pub enum ProjectLspError {
    #[error("Project not found: {project_id}")]
    ProjectNotFound { project_id: String },

    #[error("Language not supported: {language:?}")]
    LanguageNotSupported { language: Language },

    #[error("Server unavailable for project {project_id}, language {language:?}")]
    ServerUnavailable { project_id: String, language: Language },

    #[error("Query failed: {message}")]
    QueryFailed { message: String },

    #[error("LSP error: {0}")]
    Lsp(#[from] LspError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type ProjectLspResult<T> = Result<T, ProjectLspError>;

/// Unique identifier for a project-language server combination
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProjectLanguageKey {
    pub project_id: String,
    pub language: Language,
}

impl ProjectLanguageKey {
    pub fn new(project_id: impl Into<String>, language: Language) -> Self {
        Self {
            project_id: project_id.into(),
            language,
        }
    }
}

/// State of a project's LSP server
#[derive(Debug, Clone)]
pub struct ProjectServerState {
    /// Project ID (tenant_id)
    pub project_id: String,
    /// Language for this server
    pub language: Language,
    /// Project root path
    pub project_root: PathBuf,
    /// Current server status
    pub status: ServerStatus,
    /// Number of restart attempts
    pub restart_count: u32,
    /// Last error message if any
    pub last_error: Option<String>,
    /// Whether the project is currently active
    pub is_active: bool,
    /// Time when server was last healthy (for stability reset)
    pub last_healthy_time: Option<DateTime<Utc>>,
    /// Whether server has been marked unavailable after max restarts
    pub marked_unavailable: bool,
}

/// Configuration for per-project LSP management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectLspConfig {
    /// Base LSP configuration
    pub lsp_config: LspConfig,
    /// User PATH for finding language servers
    pub user_path: Option<String>,
    /// Maximum servers per project
    pub max_servers_per_project: usize,
    /// Whether to auto-start servers on project activation
    pub auto_start_on_activation: bool,
    /// Delay before stopping servers after project deactivation (seconds)
    pub deactivation_delay_secs: u64,
    /// Enable enrichment caching
    pub enable_enrichment_cache: bool,
    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
    /// Health check interval in seconds (default 30)
    pub health_check_interval_secs: u64,
    /// Maximum restart attempts before marking server unavailable (default 3)
    pub max_restarts: u32,
    /// Stability period in seconds before resetting restart count (default 3600)
    pub stability_reset_secs: u64,
    /// Enable auto-restart of failed servers
    pub enable_auto_restart: bool,
}

impl Default for ProjectLspConfig {
    fn default() -> Self {
        Self {
            lsp_config: LspConfig::default(),
            user_path: None,
            max_servers_per_project: 3,
            auto_start_on_activation: true,
            deactivation_delay_secs: 60,
            enable_enrichment_cache: true,
            cache_ttl_secs: 300,
            health_check_interval_secs: 30,
            max_restarts: 3,
            stability_reset_secs: 3600,
            enable_auto_restart: true,
        }
    }
}

impl From<LspSettings> for ProjectLspConfig {
    /// Convert daemon `LspSettings` to `ProjectLspConfig`
    fn from(settings: LspSettings) -> Self {
        let mut lsp_config = LspConfig::default();

        lsp_config.startup_timeout = Duration::from_secs(settings.startup_timeout_secs);
        lsp_config.request_timeout = Duration::from_secs(settings.request_timeout_secs);
        lsp_config.health_check_interval =
            Duration::from_secs(settings.health_check_interval_secs);
        lsp_config.enable_auto_restart = settings.enable_auto_restart;
        lsp_config.max_restart_attempts = settings.max_restart_attempts;
        lsp_config.restart_backoff_multiplier = settings.restart_backoff_multiplier;

        Self {
            lsp_config,
            user_path: settings.user_path,
            max_servers_per_project: settings.max_servers_per_project,
            auto_start_on_activation: settings.auto_start_on_activation,
            deactivation_delay_secs: settings.deactivation_delay_secs,
            enable_enrichment_cache: settings.enable_enrichment_cache,
            cache_ttl_secs: settings.cache_ttl_secs,
            health_check_interval_secs: settings.health_check_interval_secs,
            max_restarts: settings.max_restart_attempts,
            stability_reset_secs: settings.stability_reset_secs,
            enable_auto_restart: settings.enable_auto_restart,
        }
    }
}

/// A reference to a symbol location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    /// File containing the reference
    pub file: String,
    /// Line number (0-indexed)
    pub line: u32,
    /// Column number (0-indexed)
    pub column: u32,
    /// End line (if range available)
    pub end_line: Option<u32>,
    /// End column (if range available)
    pub end_column: Option<u32>,
}

/// Type information for a symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    /// Type signature
    pub type_signature: String,
    /// Documentation if available
    pub documentation: Option<String>,
    /// Kind (function, class, variable, etc.)
    pub kind: String,
    /// Container (parent class, module, etc.)
    pub container: Option<String>,
}

/// Resolved import information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedImport {
    /// Import name as written
    pub import_name: String,
    /// Resolved target file
    pub target_file: Option<String>,
    /// Resolved target symbol
    pub target_symbol: Option<String>,
    /// Whether this is a standard library import
    pub is_stdlib: bool,
    /// Whether the import could be resolved
    pub resolved: bool,
}

/// LSP enrichment data for a semantic chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspEnrichment {
    /// References to this symbol
    pub references: Vec<Reference>,
    /// Type information
    pub type_info: Option<TypeInfo>,
    /// Resolved imports within this chunk
    pub resolved_imports: Vec<ResolvedImport>,
    /// Definition location if this references something
    pub definition: Option<Reference>,
    /// Whether enrichment was successful
    pub enrichment_status: EnrichmentStatus,
    /// Error message if enrichment failed
    pub error_message: Option<String>,
}

/// Status of enrichment operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EnrichmentStatus {
    /// Enrichment completed successfully
    Success,
    /// Partial enrichment (some queries failed)
    Partial,
    /// Enrichment failed (LSP unavailable or error)
    Failed,
    /// Enrichment skipped (project inactive)
    Skipped,
}

impl EnrichmentStatus {
    /// Return a clean lowercase string for storage in Qdrant payloads.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Partial => "partial",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }
}

/// LSP usage metrics tracking (Task 1.17)
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct LspMetrics {
    /// Total enrichment queries made
    pub total_enrichment_queries: u64,
    /// Successful enrichments (full data returned)
    pub successful_enrichments: u64,
    /// Partial enrichments (some queries failed or no data)
    pub partial_enrichments: u64,
    /// Failed enrichments (all queries failed)
    pub failed_enrichments: u64,
    /// Skipped enrichments (project inactive)
    pub skipped_enrichments: u64,
    /// Cache hits (enrichment returned from cache)
    pub cache_hits: u64,
    /// Cache misses (enrichment not in cache)
    pub cache_misses: u64,
    /// Total references queries made
    pub total_references_queries: u64,
    /// Total type info queries made
    pub total_type_info_queries: u64,
    /// Total import resolution queries made
    pub total_import_queries: u64,
    /// Total successful server starts
    pub total_server_starts: u64,
    /// Total server restarts
    pub total_server_restarts: u64,
    /// Total server stops
    pub total_server_stops: u64,
}

impl LspMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a snapshot of current metrics
    pub fn snapshot(&self) -> Self {
        self.clone()
    }

    /// Get cache hit rate as a percentage (0-100)
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            (self.cache_hits as f64 / total as f64) * 100.0
        }
    }

    /// Get enrichment success rate as a percentage (0-100)
    pub fn enrichment_success_rate(&self) -> f64 {
        if self.total_enrichment_queries == 0 {
            0.0
        } else {
            (self.successful_enrichments as f64 / self.total_enrichment_queries as f64) * 100.0
        }
    }
}

/// Manages LSP servers for all active projects
pub struct LanguageServerManager {
    /// Configuration
    pub(crate) config: ProjectLspConfig,
    /// Server state by (project_id, language)
    pub(crate) servers:
        Arc<RwLock<HashMap<ProjectLanguageKey, ProjectServerState>>>,
    /// Running server instances by (project_id, language)
    pub(crate) instances:
        Arc<RwLock<HashMap<ProjectLanguageKey, Arc<tokio::sync::Mutex<ServerInstance>>>>>,
    /// Enrichment cache: (project_id, file_path, position) -> enrichment
    pub(crate) cache: Arc<RwLock<HashMap<String, LspEnrichment>>>,
    /// Detected available servers by language
    pub(crate) available_servers: Arc<RwLock<HashMap<Language, Vec<String>>>>,
    /// Running flag
    pub(crate) running: Arc<RwLock<bool>>,
    /// Usage metrics (Task 1.17)
    pub(crate) metrics: Arc<RwLock<LspMetrics>>,
    /// Set of currently active project IDs (for restore filtering)
    pub(crate) active_projects: Arc<RwLock<HashSet<String>>>,
}

impl LanguageServerManager {
    /// Create a new project LSP manager
    pub async fn new(config: ProjectLspConfig) -> ProjectLspResult<Self> {
        Ok(Self {
            config,
            servers: Arc::new(RwLock::new(HashMap::new())),
            instances: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            available_servers: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
            metrics: Arc::new(RwLock::new(LspMetrics::new())),
            active_projects: Arc::new(RwLock::new(HashSet::new())),
        })
    }

    /// Initialize the manager and detect available servers
    pub async fn initialize(&mut self) -> ProjectLspResult<()> {
        *self.running.write().await = true;

        // Update PATH if user_path is configured
        if let Some(ref user_path) = self.config.user_path {
            std::env::set_var(
                "PATH",
                format!(
                    "{}:{}",
                    user_path,
                    std::env::var("PATH").unwrap_or_default()
                ),
            );
        }

        // Detect available servers
        self.detect_available_servers().await?;

        // Start health check background task
        if self.config.enable_auto_restart {
            self.start_health_check_task();
        }

        tracing::info!("LanguageServerManager initialized");
        Ok(())
    }

    /// Detect available language servers on the system
    async fn detect_available_servers(&self) -> ProjectLspResult<()> {
        let mut available = self.available_servers.write().await;

        let servers_to_check = vec![
            (Language::Rust, vec!["rust-analyzer"]),
            (
                Language::Python,
                vec!["pyright", "pyright-langserver", "pylsp", "ruff-lsp"],
            ),
            (
                Language::TypeScript,
                vec!["typescript-language-server", "tsserver"],
            ),
            (
                Language::JavaScript,
                vec!["typescript-language-server", "tsserver"],
            ),
            (Language::Go, vec!["gopls"]),
            (Language::Java, vec!["jdtls"]),
            (Language::C, vec!["clangd", "ccls"]),
            (Language::Cpp, vec!["clangd", "ccls"]),
        ];

        for (language, candidates) in servers_to_check {
            let found: Vec<String> = candidates
                .iter()
                .filter(|name| which::which(name).is_ok())
                .map(|s| s.to_string())
                .collect();

            if !found.is_empty() {
                tracing::info!(
                    language = ?language,
                    servers = ?found,
                    "Detected language servers"
                );
                available.insert(language, found);
            }
        }

        Ok(())
    }
}

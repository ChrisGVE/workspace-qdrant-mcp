//! Project-specific LSP Server Management
//!
//! This module provides per-project LSP server management, enabling
//! code intelligence features for active projects. It extends the base
//! LSP module with project-specific lifecycle management and enrichment
//! query capabilities.
//!
//! # Architecture
//!
//! The LanguageServerManager maintains a mapping of (project_id, language)
//! to active language server instances. Servers are started when a project
//! becomes active and stopped when the project is deprioritized.
//!
//! # Features
//!
//! - Automatic server lifecycle tied to project activation
//! - Enrichment queries: references, type info, imports
//! - Graceful degradation when LSP unavailable
//! - Cache-aware result management

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use chrono::{DateTime, Utc};

use super::{
    Language, LspConfig, LspError,
    LspServerDetector, ServerInstance, ServerStatus,
};
use crate::config::LspSettings;

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

    /// Stability period in seconds before resetting restart count (default 3600 = 1 hour)
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
            stability_reset_secs: 3600, // 1 hour
            enable_auto_restart: true,
        }
    }
}

impl From<LspSettings> for ProjectLspConfig {
    /// Convert daemon LspSettings to ProjectLspConfig
    ///
    /// This allows the daemon config to be used directly when creating
    /// the LanguageServerManager.
    fn from(settings: LspSettings) -> Self {
        let mut lsp_config = LspConfig::default();

        // Apply settings to base LspConfig
        lsp_config.startup_timeout = Duration::from_secs(settings.startup_timeout_secs);
        lsp_config.request_timeout = Duration::from_secs(settings.request_timeout_secs);
        lsp_config.health_check_interval = Duration::from_secs(settings.health_check_interval_secs);
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

/// Manages LSP servers for all active projects
pub struct LanguageServerManager {
    /// Configuration
    config: ProjectLspConfig,

    /// Server detector
    detector: LspServerDetector,

    /// Server state by (project_id, language)
    servers: Arc<RwLock<HashMap<ProjectLanguageKey, ProjectServerState>>>,

    /// Running server instances by (project_id, language)
    instances: Arc<RwLock<HashMap<ProjectLanguageKey, Arc<tokio::sync::Mutex<ServerInstance>>>>>,

    /// Enrichment cache: (project_id, file_path, position) -> enrichment
    cache: Arc<RwLock<HashMap<String, LspEnrichment>>>,

    /// Detected available servers by language
    available_servers: Arc<RwLock<HashMap<Language, Vec<String>>>>,

    /// Running flag
    running: Arc<RwLock<bool>>,
}

impl LanguageServerManager {
    /// Create a new project LSP manager
    pub async fn new(config: ProjectLspConfig) -> ProjectLspResult<Self> {
        let detector = LspServerDetector::new();

        Ok(Self {
            config,
            detector,
            servers: Arc::new(RwLock::new(HashMap::new())),
            instances: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            available_servers: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(RwLock::new(false)),
        })
    }

    /// Initialize the manager and detect available servers
    pub async fn initialize(&mut self) -> ProjectLspResult<()> {
        *self.running.write().await = true;

        // Update PATH if user_path is configured
        if let Some(ref user_path) = self.config.user_path {
            std::env::set_var(
                "PATH",
                format!("{}:{}", user_path, std::env::var("PATH").unwrap_or_default())
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

    /// Start the background health check task
    fn start_health_check_task(&self) {
        let interval = Duration::from_secs(self.config.health_check_interval_secs);
        let max_restarts = self.config.max_restarts;
        let stability_reset = Duration::from_secs(self.config.stability_reset_secs);
        let instances = Arc::clone(&self.instances);
        let servers = Arc::clone(&self.servers);
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // Check if manager is still running
                if !*running.read().await {
                    tracing::info!("Health check task shutting down");
                    break;
                }

                // Perform health checks on all active servers
                Self::perform_health_checks(
                    &instances,
                    &servers,
                    max_restarts,
                    stability_reset,
                ).await;
            }
        });

        tracing::info!(
            interval_secs = self.config.health_check_interval_secs,
            max_restarts = self.config.max_restarts,
            "Health check background task started"
        );
    }

    /// Perform health checks on all active server instances
    async fn perform_health_checks(
        instances: &Arc<RwLock<HashMap<ProjectLanguageKey, Arc<tokio::sync::Mutex<ServerInstance>>>>>,
        servers: &Arc<RwLock<HashMap<ProjectLanguageKey, ProjectServerState>>>,
        max_restarts: u32,
        stability_reset: Duration,
    ) {
        let keys: Vec<_> = {
            let inst = instances.read().await;
            inst.keys().cloned().collect()
        };

        for key in keys {
            let instance = {
                let inst = instances.read().await;
                inst.get(&key).cloned()
            };

            let Some(instance) = instance else {
                continue;
            };

            // Perform health check
            let mut inst_guard = instance.lock().await;
            let health_result = inst_guard.health_check().await;

            match health_result {
                Ok(metrics) => {
                    let is_healthy = matches!(metrics.status, ServerStatus::Running);
                    let mut servers_guard = servers.write().await;

                    if let Some(state) = servers_guard.get_mut(&key) {
                        if is_healthy {
                            state.status = ServerStatus::Running;
                            state.last_healthy_time = Some(Utc::now());

                            // Reset restart count after stability period
                            if state.restart_count > 0 {
                                if let Some(last_healthy) = state.last_healthy_time {
                                    let stable_for = Utc::now() - last_healthy;
                                    if stable_for > chrono::Duration::from_std(stability_reset).unwrap_or_default() {
                                        tracing::info!(
                                            project_id = %state.project_id,
                                            language = ?state.language,
                                            old_count = state.restart_count,
                                            "Resetting restart count after stability period"
                                        );
                                        state.restart_count = 0;
                                        state.marked_unavailable = false;
                                        inst_guard.reset_restart_attempts();
                                    }
                                }
                            }
                        } else if !state.marked_unavailable {
                            // Server failed health check
                            state.status = ServerStatus::Failed;
                            state.last_error = Some(format!("Health check failed: {:?}", metrics.status));

                            if state.restart_count < max_restarts {
                                // Attempt restart
                                tracing::warn!(
                                    project_id = %state.project_id,
                                    language = ?state.language,
                                    restart_count = state.restart_count + 1,
                                    max_restarts = max_restarts,
                                    "Server failed health check, attempting restart"
                                );

                                drop(servers_guard); // Release lock before restart

                                match inst_guard.restart().await {
                                    Ok(()) => {
                                        let mut servers_guard = servers.write().await;
                                        if let Some(state) = servers_guard.get_mut(&key) {
                                            state.restart_count += 1;
                                            state.status = ServerStatus::Initializing;
                                            state.last_error = None;
                                        }
                                        tracing::info!(
                                            project_id = %key.project_id,
                                            language = ?key.language,
                                            "Server restarted successfully"
                                        );
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            project_id = %key.project_id,
                                            language = ?key.language,
                                            error = %e,
                                            "Failed to restart server"
                                        );
                                        let mut servers_guard = servers.write().await;
                                        if let Some(state) = servers_guard.get_mut(&key) {
                                            state.restart_count += 1;
                                            state.last_error = Some(format!("Restart failed: {}", e));
                                        }
                                    }
                                }
                            } else {
                                // Max restarts reached, mark as unavailable
                                tracing::error!(
                                    project_id = %state.project_id,
                                    language = ?state.language,
                                    restart_count = state.restart_count,
                                    "Server permanently failed after max restart attempts"
                                );
                                state.marked_unavailable = true;
                                state.status = ServerStatus::Failed;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        project_id = %key.project_id,
                        language = ?key.language,
                        error = %e,
                        "Health check failed with error"
                    );
                }
            }
        }
    }

    /// Detect available language servers on the system
    async fn detect_available_servers(&self) -> ProjectLspResult<()> {
        let mut available = self.available_servers.write().await;

        // Check for common language servers
        let servers_to_check = vec![
            (Language::Rust, vec!["rust-analyzer"]),
            (Language::Python, vec!["pyright", "pyright-langserver", "pylsp", "ruff-lsp"]),
            (Language::TypeScript, vec!["typescript-language-server", "tsserver"]),
            (Language::JavaScript, vec!["typescript-language-server", "tsserver"]),
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

    /// Start a server for a specific project and language
    pub async fn start_server(
        &self,
        project_id: &str,
        language: Language,
        project_root: &Path,
    ) -> ProjectLspResult<Arc<ServerInstance>> {
        let key = ProjectLanguageKey::new(project_id, language.clone());

        // Check if server already exists and is running
        {
            let instances = self.instances.read().await;
            if let Some(instance) = instances.get(&key) {
                let inst = instance.lock().await;
                let status = inst.status().await;
                if matches!(status, ServerStatus::Running | ServerStatus::Initializing) {
                    tracing::debug!(
                        project_id = project_id,
                        language = ?language,
                        "Server already running"
                    );
                    // Return the existing instance
                    drop(inst);
                    return Ok(Arc::new(instance.lock().await.clone()));
                }
            }
        }

        // Check if we have a server available for this language
        let available = self.available_servers.read().await;
        let server_names = available.get(&language).ok_or_else(|| {
            ProjectLspError::LanguageNotSupported { language: language.clone() }
        })?;

        if server_names.is_empty() {
            return Err(ProjectLspError::LanguageNotSupported { language: language.clone() });
        }

        let server_name = &server_names[0];

        // Find the server executable path
        let server_path = which::which(server_name).map_err(|_| {
            ProjectLspError::ServerUnavailable {
                project_id: project_id.to_string(),
                language: language.clone(),
            }
        })?;

        tracing::info!(
            project_id = project_id,
            language = ?language,
            server = server_name,
            path = %server_path.display(),
            "Starting language server"
        );

        // Create DetectedServer for the ServerInstance
        use super::detection::{DetectedServer, ServerCapabilities};
        let detected = DetectedServer {
            name: server_name.clone(),
            path: server_path,
            languages: vec![language.clone()],
            version: None,
            capabilities: ServerCapabilities::default(),
            priority: 1,
        };

        // Create and start the server instance with project root as working directory
        let mut instance = ServerInstance::new(detected, self.config.lsp_config.clone())
            .await
            .map_err(|e| ProjectLspError::Lsp(e))?
            .with_working_directory(project_root.to_path_buf());

        tracing::debug!(
            project_id = project_id,
            language = ?language,
            working_directory = %project_root.display(),
            "Created server instance with project root"
        );

        if let Err(e) = instance.start().await {
            tracing::warn!(
                project_id = project_id,
                language = ?language,
                error = %e,
                "Failed to start LSP server"
            );

            // Update state to failed
            let mut servers = self.servers.write().await;
            if let Some(state) = servers.get_mut(&key) {
                state.status = ServerStatus::Failed;
                state.last_error = Some(e.to_string());
            }

            return Err(ProjectLspError::Lsp(e));
        }

        // Create server state
        let state = ProjectServerState {
            project_id: project_id.to_string(),
            language: language.clone(),
            project_root: project_root.to_path_buf(),
            status: ServerStatus::Running,
            restart_count: 0,
            last_error: None,
            is_active: true,
            last_healthy_time: Some(Utc::now()),
            marked_unavailable: false,
        };

        // Store the state and instance
        {
            let mut servers = self.servers.write().await;
            servers.insert(key.clone(), state);
        }
        {
            let mut instances = self.instances.write().await;
            instances.insert(key.clone(), Arc::new(tokio::sync::Mutex::new(instance.clone())));
        }

        tracing::info!(
            project_id = project_id,
            language = ?language,
            "Language server started successfully"
        );

        Ok(Arc::new(instance))
    }

    /// Stop a server for a specific project and language
    pub async fn stop_server(
        &self,
        project_id: &str,
        language: Language,
    ) -> ProjectLspResult<()> {
        let key = ProjectLanguageKey::new(project_id, language.clone());

        // Update state to stopping
        {
            let mut servers = self.servers.write().await;
            if let Some(state) = servers.get_mut(&key) {
                state.status = ServerStatus::Stopping;
                state.is_active = false;
            }
        }

        // Get and shutdown the server instance
        let instance_opt = {
            let mut instances = self.instances.write().await;
            instances.remove(&key)
        };

        if let Some(instance) = instance_opt {
            tracing::info!(
                project_id = project_id,
                language = ?language,
                "Stopping language server"
            );

            let mut inst = instance.lock().await;
            if let Err(e) = inst.shutdown().await {
                tracing::warn!(
                    project_id = project_id,
                    language = ?language,
                    error = %e,
                    "Error during LSP server shutdown"
                );
            }

            tracing::info!(
                project_id = project_id,
                language = ?language,
                "Language server stopped"
            );
        }

        // Remove state
        {
            let mut servers = self.servers.write().await;
            servers.remove(&key);
        }

        Ok(())
    }

    /// Get a server instance for a specific project and language
    ///
    /// Returns the ServerInstance if it exists and is running.
    pub async fn get_server(
        &self,
        project_id: &str,
        language: Language,
    ) -> Option<Arc<tokio::sync::Mutex<ServerInstance>>> {
        let key = ProjectLanguageKey::new(project_id, language);
        let instances = self.instances.read().await;
        instances.get(&key).cloned()
    }

    /// Get the state for a specific project and language
    pub async fn get_server_state(
        &self,
        project_id: &str,
        language: Language,
    ) -> Option<ProjectServerState> {
        let key = ProjectLanguageKey::new(project_id, language);
        let servers = self.servers.read().await;
        servers.get(&key).cloned()
    }

    /// Get all running servers for a project
    pub async fn get_project_servers(
        &self,
        project_id: &str,
    ) -> Vec<(Language, Arc<tokio::sync::Mutex<ServerInstance>>)> {
        let instances = self.instances.read().await;
        instances
            .iter()
            .filter_map(|(key, instance)| {
                if key.project_id == project_id {
                    Some((key.language.clone(), Arc::clone(instance)))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Check if a server is running for a specific project and language
    pub async fn is_server_running(
        &self,
        project_id: &str,
        language: Language,
    ) -> bool {
        if let Some(state) = self.get_server_state(project_id, language.clone()).await {
            matches!(state.status, ServerStatus::Running)
        } else {
            false
        }
    }

    /// Find a running server instance for a file based on its language
    async fn find_server_for_file(
        &self,
        project_id: &str,
        file: &Path,
    ) -> Option<Arc<tokio::sync::Mutex<ServerInstance>>> {
        // Determine language from file extension
        let language = file.extension()
            .and_then(|ext| ext.to_str())
            .map(Language::from_extension)?;

        // Look for a running instance for this project and language
        let key = ProjectLanguageKey::new(project_id, language);
        let instances = self.instances.read().await;
        instances.get(&key).cloned()
    }

    /// Convert file path to LSP URI
    fn file_to_uri(file: &Path) -> String {
        format!("file://{}", file.display())
    }

    /// Parse LSP Location response into Reference
    fn parse_location(location: &serde_json::Value) -> Option<Reference> {
        let uri = location.get("uri")?.as_str()?;
        let range = location.get("range")?;
        let start = range.get("start")?;

        // Extract file path from URI
        let file = uri.strip_prefix("file://").unwrap_or(uri);

        Some(Reference {
            file: file.to_string(),
            line: start.get("line")?.as_u64()? as u32,
            column: start.get("character")?.as_u64()? as u32,
            end_line: range.get("end").and_then(|e| e.get("line")).and_then(|l| l.as_u64()).map(|l| l as u32),
            end_column: range.get("end").and_then(|e| e.get("character")).and_then(|c| c.as_u64()).map(|c| c as u32),
        })
    }

    /// Get references for a symbol at a specific position
    pub async fn get_references(
        &self,
        file: &Path,
        line: u32,
        column: u32,
    ) -> ProjectLspResult<Vec<Reference>> {
        // Check cache first
        let cache_key = format!("refs:{}:{}:{}", file.display(), line, column);
        {
            let cache = self.cache.read().await;
            if let Some(enrichment) = cache.get(&cache_key) {
                return Ok(enrichment.references.clone());
            }
        }

        // Try to find a server for this file
        // We need to know the project_id - for now, check all projects
        let instances = self.instances.read().await;
        let file_language = file.extension()
            .and_then(|ext| ext.to_str())
            .map(Language::from_extension);

        let server_instance = if let Some(language) = file_language {
            // Find any instance that matches this language
            instances.iter()
                .find(|(k, _)| k.language == language)
                .map(|(_, v)| v.clone())
        } else {
            None
        };

        drop(instances);

        let Some(instance) = server_instance else {
            tracing::debug!(
                file = %file.display(),
                "No LSP server available for file"
            );
            return Ok(Vec::new());
        };

        // Prepare textDocument/references request
        let params = serde_json::json!({
            "textDocument": {
                "uri": Self::file_to_uri(file)
            },
            "position": {
                "line": line,
                "character": column
            },
            "context": {
                "includeDeclaration": true
            }
        });

        // Send request
        let inst = instance.lock().await;
        let rpc_client = inst.rpc_client();

        let response = match rpc_client.send_request("textDocument/references", params).await {
            Ok(resp) => resp,
            Err(e) => {
                tracing::debug!(
                    file = %file.display(),
                    error = %e,
                    "Failed to get references from LSP"
                );
                return Ok(Vec::new());
            }
        };

        // Parse response
        let references: Vec<Reference> = if let Some(result) = response.result {
            if let Some(locations) = result.as_array() {
                locations.iter()
                    .filter_map(|loc| Self::parse_location(loc))
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        tracing::debug!(
            file = %file.display(),
            line = line,
            column = column,
            count = references.len(),
            "Got references from LSP"
        );

        // Cache the result
        if !references.is_empty() {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, LspEnrichment {
                references: references.clone(),
                type_info: None,
                resolved_imports: Vec::new(),
                definition: None,
                enrichment_status: EnrichmentStatus::Success,
                error_message: None,
            });
        }

        Ok(references)
    }

    /// Parse hover response into TypeInfo
    fn parse_hover_response(hover: &serde_json::Value) -> Option<TypeInfo> {
        let contents = hover.get("contents")?;

        // Handle MarkupContent format
        let type_signature = if contents.is_object() {
            contents.get("value")?.as_str()?.to_string()
        } else if contents.is_string() {
            contents.as_str()?.to_string()
        } else if contents.is_array() {
            // Handle MarkedString[] format
            contents.as_array()?
                .iter()
                .filter_map(|c| {
                    if c.is_string() {
                        c.as_str().map(|s| s.to_string())
                    } else {
                        c.get("value").and_then(|v| v.as_str()).map(|s| s.to_string())
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            return None;
        };

        // Try to extract kind from the type signature
        let kind = if type_signature.contains("fn ") || type_signature.contains("function") {
            "function"
        } else if type_signature.contains("struct ") || type_signature.contains("class") {
            "class"
        } else if type_signature.contains("trait ") || type_signature.contains("interface") {
            "interface"
        } else if type_signature.contains("type ") {
            "type"
        } else if type_signature.contains("const ") || type_signature.contains("let ") {
            "variable"
        } else {
            "unknown"
        };

        Some(TypeInfo {
            type_signature,
            documentation: None, // Could be extracted from contents if present
            kind: kind.to_string(),
            container: None,
        })
    }

    /// Get type information for a symbol at a specific position
    pub async fn get_type_info(
        &self,
        file: &Path,
        line: u32,
        column: u32,
    ) -> ProjectLspResult<Option<TypeInfo>> {
        // Check cache first
        let cache_key = format!("type:{}:{}:{}", file.display(), line, column);
        {
            let cache = self.cache.read().await;
            if let Some(enrichment) = cache.get(&cache_key) {
                return Ok(enrichment.type_info.clone());
            }
        }

        // Try to find a server for this file
        let instances = self.instances.read().await;
        let file_language = file.extension()
            .and_then(|ext| ext.to_str())
            .map(Language::from_extension);

        let server_instance = if let Some(language) = file_language {
            instances.iter()
                .find(|(k, _)| k.language == language)
                .map(|(_, v)| v.clone())
        } else {
            None
        };

        drop(instances);

        let Some(instance) = server_instance else {
            tracing::debug!(
                file = %file.display(),
                "No LSP server available for file"
            );
            return Ok(None);
        };

        // Prepare textDocument/hover request
        let params = serde_json::json!({
            "textDocument": {
                "uri": Self::file_to_uri(file)
            },
            "position": {
                "line": line,
                "character": column
            }
        });

        // Send request
        let inst = instance.lock().await;
        let rpc_client = inst.rpc_client();

        let response = match rpc_client.send_request("textDocument/hover", params).await {
            Ok(resp) => resp,
            Err(e) => {
                tracing::debug!(
                    file = %file.display(),
                    error = %e,
                    "Failed to get hover info from LSP"
                );
                return Ok(None);
            }
        };

        // Parse response
        let type_info = response.result
            .as_ref()
            .and_then(|r| Self::parse_hover_response(r));

        tracing::debug!(
            file = %file.display(),
            line = line,
            column = column,
            has_type_info = type_info.is_some(),
            "Got type info from LSP"
        );

        // Cache the result
        if type_info.is_some() {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, LspEnrichment {
                references: Vec::new(),
                type_info: type_info.clone(),
                resolved_imports: Vec::new(),
                definition: None,
                enrichment_status: EnrichmentStatus::Success,
                error_message: None,
            });
        }

        Ok(type_info)
    }

    /// Parse a definition location into a ResolvedImport
    fn parse_definition_response(
        import_name: &str,
        definition: Option<&serde_json::Value>,
    ) -> ResolvedImport {
        let (target_file, resolved) = if let Some(def) = definition {
            // Handle Location or Location[] response
            let location = if def.is_array() {
                def.as_array().and_then(|arr| arr.first())
            } else {
                Some(def)
            };

            if let Some(loc) = location {
                let uri = loc.get("uri").and_then(|u| u.as_str());
                let target = uri.map(|u| u.strip_prefix("file://").unwrap_or(u).to_string());
                (target, uri.is_some())
            } else {
                (None, false)
            }
        } else {
            (None, false)
        };

        // Determine if stdlib based on path patterns
        let is_stdlib = target_file.as_ref()
            .map(|p| {
                p.contains("/site-packages/") ||
                p.contains("/.rustup/") ||
                p.contains("/lib/rustlib/") ||
                p.contains("/node_modules/@types/") ||
                p.contains("/usr/lib/") ||
                p.contains("/Library/Developer/")
            })
            .unwrap_or(false);

        ResolvedImport {
            import_name: import_name.to_string(),
            target_file,
            target_symbol: None, // Would require additional parsing
            is_stdlib,
            resolved,
        }
    }

    /// Extract import statements from file content (basic pattern matching)
    fn extract_imports(content: &str, language: &Language) -> Vec<String> {
        let mut imports = Vec::new();

        let import_patterns = match language {
            Language::Python => vec![
                (r"^import\s+(\S+)", 1),
                (r"^from\s+(\S+)\s+import", 1),
            ],
            Language::Rust => vec![
                (r"^use\s+([^;]+)", 1),
            ],
            Language::TypeScript | Language::JavaScript => vec![
                (r#"import\s+.*\s+from\s+['"]([^'"]+)['"]"#, 1),
                (r#"require\s*\(\s*['"]([^'"]+)['"]"#, 1),
            ],
            Language::Go => vec![
                (r#"import\s+["']([^"']+)["']"#, 1),
                (r#"^\s*"([^"]+)"$"#, 1), // Inside import block
            ],
            _ => vec![],
        };

        for line in content.lines() {
            for (pattern, group) in &import_patterns {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if let Some(captures) = re.captures(line) {
                        if let Some(import) = captures.get(*group) {
                            imports.push(import.as_str().to_string());
                        }
                    }
                }
            }
        }

        imports
    }

    /// Resolve imports in a file
    pub async fn resolve_imports(
        &self,
        file: &Path,
    ) -> ProjectLspResult<Vec<ResolvedImport>> {
        // Check cache first
        let cache_key = format!("imports:{}", file.display());
        {
            let cache = self.cache.read().await;
            if let Some(enrichment) = cache.get(&cache_key) {
                return Ok(enrichment.resolved_imports.clone());
            }
        }

        // Try to find a server for this file
        let instances = self.instances.read().await;
        let file_language = file.extension()
            .and_then(|ext| ext.to_str())
            .map(Language::from_extension);

        let Some(language) = file_language else {
            return Ok(Vec::new());
        };

        let server_instance = instances.iter()
            .find(|(k, _)| k.language == language)
            .map(|(_, v)| v.clone());

        drop(instances);

        // Read file content to extract imports
        let content = match tokio::fs::read_to_string(file).await {
            Ok(c) => c,
            Err(e) => {
                tracing::debug!(
                    file = %file.display(),
                    error = %e,
                    "Failed to read file for import extraction"
                );
                return Ok(Vec::new());
            }
        };

        // Extract import statements
        let import_names = Self::extract_imports(&content, &language);
        if import_names.is_empty() {
            return Ok(Vec::new());
        }

        tracing::debug!(
            file = %file.display(),
            imports_found = import_names.len(),
            "Extracted imports from file"
        );

        let mut resolved_imports = Vec::new();

        // If we have an LSP server, try to resolve each import
        if let Some(instance) = server_instance {
            let inst = instance.lock().await;
            let rpc_client = inst.rpc_client();

            // For each import, try to find its definition
            // We approximate by looking at lines that contain the import
            for (line_idx, line) in content.lines().enumerate() {
                for import_name in &import_names {
                    if line.contains(import_name) {
                        // Find the column where the import name starts
                        let column = line.find(import_name).unwrap_or(0) as u32;

                        // Send textDocument/definition request
                        let params = serde_json::json!({
                            "textDocument": {
                                "uri": Self::file_to_uri(file)
                            },
                            "position": {
                                "line": line_idx as u32,
                                "character": column
                            }
                        });

                        match rpc_client.send_request("textDocument/definition", params).await {
                            Ok(response) => {
                                let resolved = Self::parse_definition_response(
                                    import_name,
                                    response.result.as_ref()
                                );
                                resolved_imports.push(resolved);
                            }
                            Err(e) => {
                                tracing::debug!(
                                    import = import_name,
                                    error = %e,
                                    "Failed to resolve import via LSP"
                                );
                                // Add unresolved import
                                resolved_imports.push(ResolvedImport {
                                    import_name: import_name.clone(),
                                    target_file: None,
                                    target_symbol: None,
                                    is_stdlib: false,
                                    resolved: false,
                                });
                            }
                        }

                        break; // Only resolve once per import name
                    }
                }
            }
        } else {
            // No LSP server available, return unresolved imports
            for import_name in import_names {
                resolved_imports.push(ResolvedImport {
                    import_name,
                    target_file: None,
                    target_symbol: None,
                    is_stdlib: false,
                    resolved: false,
                });
            }
        }

        tracing::debug!(
            file = %file.display(),
            resolved = resolved_imports.iter().filter(|i| i.resolved).count(),
            total = resolved_imports.len(),
            "Import resolution complete"
        );

        // Cache the result
        if !resolved_imports.is_empty() {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, LspEnrichment {
                references: Vec::new(),
                type_info: None,
                resolved_imports: resolved_imports.clone(),
                definition: None,
                enrichment_status: EnrichmentStatus::Success,
                error_message: None,
            });
        }

        Ok(resolved_imports)
    }

    /// Enrich a semantic chunk with LSP data
    ///
    /// This is the main entry point for queue processor integration.
    /// Returns enrichment data or gracefully degrades if LSP unavailable.
    ///
    /// Status determination:
    /// - Skipped: project not active
    /// - Success: at least one query returned data
    /// - Partial: some queries failed but at least one succeeded
    /// - Failed: all queries failed
    pub async fn enrich_chunk(
        &self,
        project_id: &str,
        file: &Path,
        _symbol_name: &str,
        start_line: u32,
        _end_line: u32,
        is_project_active: bool,
    ) -> LspEnrichment {
        // Skip enrichment if project is not active
        if !is_project_active {
            return LspEnrichment {
                references: Vec::new(),
                type_info: None,
                resolved_imports: Vec::new(),
                definition: None,
                enrichment_status: EnrichmentStatus::Skipped,
                error_message: Some("Project not active".to_string()),
            };
        }

        // Track successes and failures for each query
        let mut errors: Vec<String> = Vec::new();
        let mut successes = 0;

        // Try to get references
        let references = match self.get_references(file, start_line, 0).await {
            Ok(refs) => {
                successes += 1;
                refs
            }
            Err(e) => {
                let error_msg = format!("get_references: {}", e);
                tracing::debug!(
                    project_id = project_id,
                    file = %file.display(),
                    error = %e,
                    "Failed to get references"
                );
                errors.push(error_msg);
                Vec::new()
            }
        };

        // Try to get type info
        let type_info = match self.get_type_info(file, start_line, 0).await {
            Ok(info) => {
                successes += 1;
                info
            }
            Err(e) => {
                let error_msg = format!("get_type_info: {}", e);
                tracing::debug!(
                    project_id = project_id,
                    file = %file.display(),
                    error = %e,
                    "Failed to get type info"
                );
                errors.push(error_msg);
                None
            }
        };

        // Try to resolve imports
        let resolved_imports = match self.resolve_imports(file).await {
            Ok(imports) => {
                successes += 1;
                imports
            }
            Err(e) => {
                let error_msg = format!("resolve_imports: {}", e);
                tracing::debug!(
                    project_id = project_id,
                    file = %file.display(),
                    error = %e,
                    "Failed to resolve imports"
                );
                errors.push(error_msg);
                Vec::new()
            }
        };

        // Determine enrichment status based on successes and data
        let has_data = !references.is_empty() || type_info.is_some() || !resolved_imports.is_empty();
        let all_failed = successes == 0;
        let some_failed = !errors.is_empty() && successes > 0;

        let (status, error_message) = if all_failed {
            // All queries failed
            (
                EnrichmentStatus::Failed,
                Some(format!("All queries failed: {}", errors.join("; "))),
            )
        } else if has_data {
            // At least one query returned data
            (EnrichmentStatus::Success, None)
        } else if some_failed {
            // Some queries failed but at least one succeeded (with no data)
            (
                EnrichmentStatus::Partial,
                Some(format!("Some queries failed: {}", errors.join("; "))),
            )
        } else {
            // All queries succeeded but no data
            (EnrichmentStatus::Partial, None)
        };

        LspEnrichment {
            references,
            type_info,
            resolved_imports,
            definition: None,
            enrichment_status: status,
            error_message,
        }
    }

    /// Check if a project has active LSP servers
    pub async fn has_active_servers(&self, project_id: &str) -> bool {
        let servers = self.servers.read().await;
        servers.iter().any(|(key, state)| {
            key.project_id == project_id
                && state.is_active
                && matches!(state.status, ServerStatus::Running)
        })
    }

    /// Get statistics for the manager
    pub async fn stats(&self) -> ProjectLspStats {
        let servers = self.servers.read().await;
        let available = self.available_servers.read().await;
        let cache = self.cache.read().await;

        ProjectLspStats {
            active_servers: servers.values().filter(|s| s.is_active).count(),
            total_servers: servers.len(),
            available_languages: available.len(),
            cache_entries: cache.len(),
        }
    }

    /// Check health of all active servers and restart crashed ones
    ///
    /// Returns summary of health check: (checked_count, restarted_count, failed_count)
    pub async fn check_all_servers_health(&self) -> (usize, usize, usize) {
        let mut checked = 0;
        let mut restarted = 0;
        let mut failed = 0;

        // Get list of active server keys
        let keys: Vec<ProjectLanguageKey> = {
            let servers = self.servers.read().await;
            servers.iter()
                .filter(|(_, state)| state.is_active)
                .map(|(k, _)| k.clone())
                .collect()
        };

        tracing::debug!(
            "Checking health of {} active LSP servers",
            keys.len()
        );

        // Check each server
        for key in keys {
            checked += 1;

            // Get the instance and check if it's alive
            let restart_needed = {
                let instances = self.instances.read().await;
                if let Some(instance_arc) = instances.get(&key) {
                    let instance = instance_arc.lock().await;
                    !instance.is_alive().await
                } else {
                    // No instance - might need to start one
                    false
                }
            };

            if restart_needed {
                tracing::info!(
                    "LSP server for {:?} in project {} has crashed, attempting restart",
                    key.language, key.project_id
                );

                // Attempt restart
                let mut instances = self.instances.write().await;
                if let Some(instance_arc) = instances.get(&key) {
                    let mut instance = instance_arc.lock().await;
                    match instance.check_and_restart_if_needed().await {
                        Ok(true) => {
                            restarted += 1;
                            tracing::info!(
                                "Successfully restarted LSP server for {:?} in project {}",
                                key.language, key.project_id
                            );
                        }
                        Ok(false) => {
                            // Couldn't restart (exceeded attempts or disabled)
                            failed += 1;

                            // Update server state
                            let mut servers = self.servers.write().await;
                            if let Some(state) = servers.get_mut(&key) {
                                state.status = ServerStatus::Failed;
                                state.is_active = false;
                            }
                        }
                        Err(e) => {
                            failed += 1;
                            tracing::error!(
                                "Failed to restart LSP server for {:?} in project {}: {}",
                                key.language, key.project_id, e
                            );

                            // Update server state
                            let mut servers = self.servers.write().await;
                            if let Some(state) = servers.get_mut(&key) {
                                state.status = ServerStatus::Failed;
                                state.is_active = false;
                            }
                        }
                    }
                }
            }
        }

        if restarted > 0 || failed > 0 {
            tracing::info!(
                "Health check complete: {} checked, {} restarted, {} failed",
                checked, restarted, failed
            );
        }

        (checked, restarted, failed)
    }

    /// Check health of a specific project's servers
    pub async fn check_project_servers_health(&self, project_id: &str) -> (usize, usize, usize) {
        let mut checked = 0;
        let mut restarted = 0;
        let mut failed = 0;

        // Get keys for this project
        let keys: Vec<ProjectLanguageKey> = {
            let servers = self.servers.read().await;
            servers.iter()
                .filter(|(k, state)| k.project_id == project_id && state.is_active)
                .map(|(k, _)| k.clone())
                .collect()
        };

        for key in keys {
            checked += 1;

            let restart_needed = {
                let instances = self.instances.read().await;
                if let Some(instance_arc) = instances.get(&key) {
                    let instance = instance_arc.lock().await;
                    !instance.is_alive().await
                } else {
                    false
                }
            };

            if restart_needed {
                let mut instances = self.instances.write().await;
                if let Some(instance_arc) = instances.get(&key) {
                    let mut instance = instance_arc.lock().await;
                    match instance.check_and_restart_if_needed().await {
                        Ok(true) => restarted += 1,
                        Ok(false) | Err(_) => {
                            failed += 1;
                            let mut servers = self.servers.write().await;
                            if let Some(state) = servers.get_mut(&key) {
                                state.status = ServerStatus::Failed;
                                state.is_active = false;
                            }
                        }
                    }
                }
            }
        }

        (checked, restarted, failed)
    }

    /// Shutdown the manager and all servers
    pub async fn shutdown(&self) -> ProjectLspResult<()> {
        *self.running.write().await = false;

        // Stop all servers
        let keys: Vec<_> = {
            let servers = self.servers.read().await;
            servers.keys().cloned().collect()
        };

        for key in keys {
            self.stop_server(&key.project_id, key.language).await?;
        }

        // Clear cache
        self.cache.write().await.clear();

        tracing::info!("LanguageServerManager shutdown complete");
        Ok(())
    }
}

/// Statistics for the project LSP manager
#[derive(Debug, Clone, Default)]
pub struct ProjectLspStats {
    pub active_servers: usize,
    pub total_servers: usize,
    pub available_languages: usize,
    pub cache_entries: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_project_language_key() {
        let key1 = ProjectLanguageKey::new("project-1", Language::Rust);
        let key2 = ProjectLanguageKey::new("project-1", Language::Rust);
        let key3 = ProjectLanguageKey::new("project-2", Language::Rust);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[tokio::test]
    async fn test_enrichment_status() {
        let status = EnrichmentStatus::Success;
        assert_eq!(status, EnrichmentStatus::Success);
    }

    #[tokio::test]
    async fn test_project_lsp_config_default() {
        let config = ProjectLspConfig::default();
        assert_eq!(config.max_servers_per_project, 3);
        assert!(config.auto_start_on_activation);
        assert_eq!(config.deactivation_delay_secs, 60);
    }

    #[tokio::test]
    async fn test_reference_serialization() {
        let reference = Reference {
            file: "src/main.rs".to_string(),
            line: 10,
            column: 5,
            end_line: Some(10),
            end_column: Some(15),
        };

        let json = serde_json::to_string(&reference).unwrap();
        let deserialized: Reference = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.file, "src/main.rs");
        assert_eq!(deserialized.line, 10);
    }

    #[tokio::test]
    async fn test_lsp_enrichment_skipped() {
        let enrichment = LspEnrichment {
            references: Vec::new(),
            type_info: None,
            resolved_imports: Vec::new(),
            definition: None,
            enrichment_status: EnrichmentStatus::Skipped,
            error_message: Some("Project not active".to_string()),
        };

        assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Skipped);
        assert!(enrichment.error_message.is_some());
    }

    #[tokio::test]
    async fn test_manager_creation() {
        let config = ProjectLspConfig::default();
        let manager = LanguageServerManager::new(config).await.unwrap();

        let stats = manager.stats().await;
        assert_eq!(stats.active_servers, 0);
        assert_eq!(stats.total_servers, 0);
    }

    #[tokio::test]
    async fn test_has_active_servers_empty() {
        let config = ProjectLspConfig::default();
        let manager = LanguageServerManager::new(config).await.unwrap();

        let has_active = manager.has_active_servers("project-1").await;
        assert!(!has_active);
    }

    #[tokio::test]
    async fn test_health_check_no_servers() {
        let config = ProjectLspConfig::default();
        let manager = LanguageServerManager::new(config).await.unwrap();

        // Should return (0, 0, 0) when no servers exist
        let (checked, restarted, failed) = manager.check_all_servers_health().await;
        assert_eq!(checked, 0);
        assert_eq!(restarted, 0);
        assert_eq!(failed, 0);
    }

    #[tokio::test]
    async fn test_project_health_check_no_servers() {
        let config = ProjectLspConfig::default();
        let manager = LanguageServerManager::new(config).await.unwrap();

        // Should return (0, 0, 0) for non-existent project
        let (checked, restarted, failed) = manager.check_project_servers_health("non-existent").await;
        assert_eq!(checked, 0);
        assert_eq!(restarted, 0);
        assert_eq!(failed, 0);
    }

    #[tokio::test]
    async fn test_type_info_structure() {
        let type_info = TypeInfo {
            type_signature: "fn foo() -> i32".to_string(),
            documentation: Some("Returns a number".to_string()),
            kind: "function".to_string(),
            container: Some("MyModule".to_string()),
        };

        assert_eq!(type_info.type_signature, "fn foo() -> i32");
        assert!(type_info.documentation.is_some());
        assert_eq!(type_info.kind, "function");
        assert!(type_info.container.is_some());
    }

    #[tokio::test]
    async fn test_resolved_import_structure() {
        let import = ResolvedImport {
            import_name: "std::collections::HashMap".to_string(),
            target_file: Some("/usr/lib/rust/std/collections/hash_map.rs".to_string()),
            target_symbol: Some("HashMap".to_string()),
            is_stdlib: true,
            resolved: true,
        };

        assert!(import.is_stdlib);
        assert!(import.resolved);
        assert!(import.target_file.is_some());
    }

    #[tokio::test]
    async fn test_resolved_import_unresolved() {
        let import = ResolvedImport {
            import_name: "unknown_crate::Thing".to_string(),
            target_file: None,
            target_symbol: None,
            is_stdlib: false,
            resolved: false,
        };

        assert!(!import.is_stdlib);
        assert!(!import.resolved);
        assert!(import.target_file.is_none());
    }

    #[tokio::test]
    async fn test_lsp_enrichment_success() {
        let enrichment = LspEnrichment {
            references: vec![
                Reference {
                    file: "src/main.rs".to_string(),
                    line: 10,
                    column: 5,
                    end_line: Some(10),
                    end_column: Some(15),
                },
            ],
            type_info: Some(TypeInfo {
                type_signature: "fn main()".to_string(),
                documentation: None,
                kind: "function".to_string(),
                container: None,
            }),
            resolved_imports: vec![],
            definition: None,
            enrichment_status: EnrichmentStatus::Success,
            error_message: None,
        };

        assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Success);
        assert_eq!(enrichment.references.len(), 1);
        assert!(enrichment.type_info.is_some());
        assert!(enrichment.error_message.is_none());
    }

    #[tokio::test]
    async fn test_lsp_enrichment_partial() {
        let enrichment = LspEnrichment {
            references: vec![],
            type_info: Some(TypeInfo {
                type_signature: "i32".to_string(),
                documentation: None,
                kind: "type".to_string(),
                container: None,
            }),
            resolved_imports: vec![],
            definition: None,
            enrichment_status: EnrichmentStatus::Partial,
            error_message: None,
        };

        assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Partial);
    }

    #[tokio::test]
    async fn test_lsp_enrichment_failed() {
        let enrichment = LspEnrichment {
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            enrichment_status: EnrichmentStatus::Failed,
            error_message: Some("LSP server crashed".to_string()),
        };

        assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Failed);
        assert!(enrichment.error_message.is_some());
    }

    #[tokio::test]
    async fn test_project_server_state_initial() {
        let state = ProjectServerState {
            project_id: "test-project".to_string(),
            language: Language::Rust,
            project_root: PathBuf::from("/test/path"),
            status: ServerStatus::Initializing,
            restart_count: 0,
            last_error: None,
            is_active: false,
            last_healthy_time: None,
            marked_unavailable: false,
        };

        assert_eq!(state.status, ServerStatus::Initializing);
        assert!(!state.is_active);
        assert_eq!(state.restart_count, 0);
        assert!(state.last_error.is_none());
    }

    #[tokio::test]
    async fn test_project_server_state_running() {
        let state = ProjectServerState {
            project_id: "test-project".to_string(),
            language: Language::Python,
            project_root: PathBuf::from("/test/path"),
            status: ServerStatus::Running,
            restart_count: 0,
            last_error: None,
            is_active: true,
            last_healthy_time: Some(Utc::now()),
            marked_unavailable: false,
        };

        assert_eq!(state.status, ServerStatus::Running);
        assert!(state.is_active);
        assert_eq!(state.project_id, "test-project");
    }

    #[tokio::test]
    async fn test_enrichment_status_equality() {
        assert_eq!(EnrichmentStatus::Success, EnrichmentStatus::Success);
        assert_eq!(EnrichmentStatus::Partial, EnrichmentStatus::Partial);
        assert_eq!(EnrichmentStatus::Failed, EnrichmentStatus::Failed);
        assert_eq!(EnrichmentStatus::Skipped, EnrichmentStatus::Skipped);

        assert_ne!(EnrichmentStatus::Success, EnrichmentStatus::Failed);
        assert_ne!(EnrichmentStatus::Partial, EnrichmentStatus::Skipped);
    }

    #[tokio::test]
    async fn test_project_lsp_stats_default() {
        let stats = ProjectLspStats::default();
        assert_eq!(stats.active_servers, 0);
        assert_eq!(stats.total_servers, 0);
        assert_eq!(stats.available_languages, 0);
        assert_eq!(stats.cache_entries, 0);
    }

    #[tokio::test]
    async fn test_lsp_enrichment_serialization() {
        let enrichment = LspEnrichment {
            references: vec![Reference {
                file: "test.rs".to_string(),
                line: 1,
                column: 0,
                end_line: None,
                end_column: None,
            }],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            enrichment_status: EnrichmentStatus::Success,
            error_message: None,
        };

        let json = serde_json::to_string(&enrichment).unwrap();
        let deserialized: LspEnrichment = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.references.len(), 1);
        assert_eq!(deserialized.enrichment_status, EnrichmentStatus::Success);
    }

    #[tokio::test]
    async fn test_get_server_not_found() {
        let config = ProjectLspConfig::default();
        let manager = LanguageServerManager::new(config).await.unwrap();

        // Server doesn't exist
        let server = manager.get_server("project-1", Language::Rust).await;
        assert!(server.is_none());
    }

    #[tokio::test]
    async fn test_get_server_state_not_found() {
        let config = ProjectLspConfig::default();
        let manager = LanguageServerManager::new(config).await.unwrap();

        // State doesn't exist
        let state = manager.get_server_state("project-1", Language::Rust).await;
        assert!(state.is_none());
    }

    #[tokio::test]
    async fn test_get_project_servers_empty() {
        let config = ProjectLspConfig::default();
        let manager = LanguageServerManager::new(config).await.unwrap();

        // No servers for this project
        let servers = manager.get_project_servers("project-1").await;
        assert!(servers.is_empty());
    }

    #[tokio::test]
    async fn test_is_server_running_no_server() {
        let config = ProjectLspConfig::default();
        let manager = LanguageServerManager::new(config).await.unwrap();

        // No server exists
        let running = manager.is_server_running("project-1", Language::Rust).await;
        assert!(!running);
    }

    #[tokio::test]
    async fn test_health_monitoring_config_defaults() {
        let config = ProjectLspConfig::default();

        // Verify health monitoring defaults
        assert_eq!(config.health_check_interval_secs, 30);
        assert_eq!(config.max_restarts, 3);
        assert_eq!(config.stability_reset_secs, 3600); // 1 hour
        assert!(config.enable_auto_restart);
    }

    #[tokio::test]
    async fn test_project_server_state_health_tracking() {
        let state = ProjectServerState {
            project_id: "test-project".to_string(),
            language: Language::Rust,
            project_root: PathBuf::from("/test/path"),
            status: ServerStatus::Running,
            restart_count: 0,
            last_error: None,
            is_active: true,
            last_healthy_time: Some(Utc::now()),
            marked_unavailable: false,
        };

        assert!(state.last_healthy_time.is_some());
        assert!(!state.marked_unavailable);
        assert_eq!(state.restart_count, 0);
    }

    #[tokio::test]
    async fn test_project_server_state_restart_tracking() {
        let mut state = ProjectServerState {
            project_id: "test-project".to_string(),
            language: Language::Python,
            project_root: PathBuf::from("/test/path"),
            status: ServerStatus::Failed,
            restart_count: 2,
            last_error: Some("Connection failed".to_string()),
            is_active: true,
            last_healthy_time: None,
            marked_unavailable: false,
        };

        // Simulate restart count increment
        state.restart_count += 1;
        assert_eq!(state.restart_count, 3);

        // After max restarts, mark unavailable
        state.marked_unavailable = true;
        assert!(state.marked_unavailable);
    }

    #[tokio::test]
    async fn test_manager_health_check_disabled() {
        let config = ProjectLspConfig {
            enable_auto_restart: false,
            ..Default::default()
        };

        // Manager should create without starting health check task
        let mut manager = LanguageServerManager::new(config).await.unwrap();
        let result = manager.initialize().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_project_lsp_config_from_lsp_settings() {
        let settings = LspSettings {
            user_path: Some("/usr/local/bin".to_string()),
            max_servers_per_project: 5,
            auto_start_on_activation: false,
            deactivation_delay_secs: 120,
            enable_enrichment_cache: false,
            cache_ttl_secs: 600,
            startup_timeout_secs: 45,
            request_timeout_secs: 15,
            health_check_interval_secs: 90,
            max_restart_attempts: 5,
            restart_backoff_multiplier: 3.0,
            enable_auto_restart: false,
            stability_reset_secs: 7200,
        };

        let config = ProjectLspConfig::from(settings);

        // Verify all settings were converted correctly
        assert_eq!(config.user_path, Some("/usr/local/bin".to_string()));
        assert_eq!(config.max_servers_per_project, 5);
        assert!(!config.auto_start_on_activation);
        assert_eq!(config.deactivation_delay_secs, 120);
        assert!(!config.enable_enrichment_cache);
        assert_eq!(config.cache_ttl_secs, 600);
        assert_eq!(config.health_check_interval_secs, 90);
        assert_eq!(config.max_restarts, 5);
        assert!(!config.enable_auto_restart);
        assert_eq!(config.stability_reset_secs, 7200);

        // Verify nested LspConfig was updated
        assert_eq!(config.lsp_config.startup_timeout.as_secs(), 45);
        assert_eq!(config.lsp_config.request_timeout.as_secs(), 15);
        assert_eq!(config.lsp_config.health_check_interval.as_secs(), 90);
        assert!(!config.lsp_config.enable_auto_restart);
        assert_eq!(config.lsp_config.max_restart_attempts, 5);
        assert_eq!(config.lsp_config.restart_backoff_multiplier, 3.0);
    }

    #[tokio::test]
    async fn test_enrich_chunk_skipped_inactive_project() {
        let config = ProjectLspConfig::default();
        let manager = LanguageServerManager::new(config).await.unwrap();

        let result = manager.enrich_chunk(
            "test-project",
            Path::new("/test/file.rs"),
            "test_symbol",
            10,
            20,
            false, // project not active
        ).await;

        assert_eq!(result.enrichment_status, EnrichmentStatus::Skipped);
        assert!(result.error_message.is_some());
        assert!(result.error_message.unwrap().contains("not active"));
        assert!(result.references.is_empty());
        assert!(result.type_info.is_none());
        assert!(result.resolved_imports.is_empty());
    }

    #[tokio::test]
    async fn test_enrich_chunk_returns_enrichment_structure() {
        let config = ProjectLspConfig::default();
        let manager = LanguageServerManager::new(config).await.unwrap();

        // Even with no servers, should return a valid structure
        let result = manager.enrich_chunk(
            "test-project",
            Path::new("/test/file.rs"),
            "test_symbol",
            10,
            20,
            true, // project active
        ).await;

        // Without any servers, queries succeed but return no data → Partial status
        // (queries succeed with empty results, not errors)
        assert_eq!(result.enrichment_status, EnrichmentStatus::Partial);
        // No error_message when queries succeed but return no data
        assert!(result.error_message.is_none());
        // Verify empty results
        assert!(result.references.is_empty());
        assert!(result.type_info.is_none());
        assert!(result.resolved_imports.is_empty());
    }

    #[tokio::test]
    async fn test_enrichment_status_display() {
        // Test that status values have expected string representations
        assert_eq!(format!("{:?}", EnrichmentStatus::Success), "Success");
        assert_eq!(format!("{:?}", EnrichmentStatus::Partial), "Partial");
        assert_eq!(format!("{:?}", EnrichmentStatus::Failed), "Failed");
        assert_eq!(format!("{:?}", EnrichmentStatus::Skipped), "Skipped");
    }

    #[tokio::test]
    async fn test_lsp_enrichment_structure_complete() {
        // Test creating a complete enrichment structure
        let enrichment = LspEnrichment {
            references: vec![
                Reference {
                    file: "src/lib.rs".to_string(),
                    line: 10,
                    column: 5,
                    end_line: Some(10),
                    end_column: Some(15),
                },
                Reference {
                    file: "src/main.rs".to_string(),
                    line: 25,
                    column: 8,
                    end_line: None,
                    end_column: None,
                },
            ],
            type_info: Some(TypeInfo {
                type_signature: "fn process() -> Result<(), Error>".to_string(),
                documentation: Some("Process the data".to_string()),
                kind: "function".to_string(),
                container: Some("MyModule".to_string()),
            }),
            resolved_imports: vec![
                ResolvedImport {
                    import_name: "std::collections::HashMap".to_string(),
                    target_file: Some("/rustlib/std/collections/hash_map.rs".to_string()),
                    target_symbol: Some("HashMap".to_string()),
                    is_stdlib: true,
                    resolved: true,
                },
            ],
            definition: None,
            enrichment_status: EnrichmentStatus::Success,
            error_message: None,
        };

        // Verify structure
        assert_eq!(enrichment.references.len(), 2);
        assert!(enrichment.type_info.is_some());
        assert_eq!(enrichment.resolved_imports.len(), 1);
        assert!(enrichment.resolved_imports[0].is_stdlib);
        assert_eq!(enrichment.enrichment_status, EnrichmentStatus::Success);
        assert!(enrichment.error_message.is_none());
    }
}

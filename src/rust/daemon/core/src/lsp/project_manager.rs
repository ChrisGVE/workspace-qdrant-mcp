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
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    Language, LspConfig, LspError, LspResult, LspServerManager,
    LspServerDetector, ServerInstance, ServerStatus,
};

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

        tracing::info!("LanguageServerManager initialized");
        Ok(())
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

        // Create and start the server instance
        let mut instance = ServerInstance::new(detected, self.config.lsp_config.clone())
            .await
            .map_err(|e| ProjectLspError::Lsp(e))?;

        // Set the working directory to the project root
        // Note: ServerInstance uses metadata.working_directory internally
        // We need to reinitialize with the correct project root
        // For now, start the server with the default working directory
        // and let it re-initialize when we send textDocument/didOpen

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

        // In full implementation, we would:
        // 1. Find the project containing this file
        // 2. Find the running server for the file's language
        // 3. Send textDocument/references request
        // 4. Parse the response
        // 5. Cache and return results

        tracing::debug!(
            file = %file.display(),
            line = line,
            column = column,
            "References query (not yet implemented)"
        );

        Ok(Vec::new())
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

        // In full implementation, we would:
        // 1. Find the project containing this file
        // 2. Find the running server for the file's language
        // 3. Send textDocument/hover request
        // 4. Parse the response for type info
        // 5. Cache and return results

        tracing::debug!(
            file = %file.display(),
            line = line,
            column = column,
            "Type info query (not yet implemented)"
        );

        Ok(None)
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

        // In full implementation, we would:
        // 1. Parse the file to find import statements
        // 2. For each import, send textDocument/definition request
        // 3. Collect resolved locations
        // 4. Determine if stdlib based on resolved path
        // 5. Cache and return results

        tracing::debug!(
            file = %file.display(),
            "Import resolution query (not yet implemented)"
        );

        Ok(Vec::new())
    }

    /// Enrich a semantic chunk with LSP data
    ///
    /// This is the main entry point for queue processor integration.
    /// Returns enrichment data or gracefully degrades if LSP unavailable.
    pub async fn enrich_chunk(
        &self,
        project_id: &str,
        file: &Path,
        symbol_name: &str,
        start_line: u32,
        end_line: u32,
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

        // Try to get references
        let references = match self.get_references(file, start_line, 0).await {
            Ok(refs) => refs,
            Err(e) => {
                tracing::debug!(
                    project_id = project_id,
                    file = %file.display(),
                    error = %e,
                    "Failed to get references"
                );
                Vec::new()
            }
        };

        // Try to get type info
        let type_info = match self.get_type_info(file, start_line, 0).await {
            Ok(info) => info,
            Err(e) => {
                tracing::debug!(
                    project_id = project_id,
                    file = %file.display(),
                    error = %e,
                    "Failed to get type info"
                );
                None
            }
        };

        // Try to resolve imports
        let resolved_imports = match self.resolve_imports(file).await {
            Ok(imports) => imports,
            Err(e) => {
                tracing::debug!(
                    project_id = project_id,
                    file = %file.display(),
                    error = %e,
                    "Failed to resolve imports"
                );
                Vec::new()
            }
        };

        // Determine enrichment status
        let status = if !references.is_empty() || type_info.is_some() || !resolved_imports.is_empty() {
            EnrichmentStatus::Success
        } else {
            EnrichmentStatus::Partial
        };

        LspEnrichment {
            references,
            type_info,
            resolved_imports,
            definition: None,
            enrichment_status: status,
            error_message: None,
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
}

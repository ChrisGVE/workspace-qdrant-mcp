//! Server lifecycle management: start, stop, shutdown, and server queries.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;

use super::{
    LanguageServerManager, ProjectLanguageKey, ProjectLspError, ProjectLspResult,
    ProjectServerState, ServerInstance, ServerStatus,
};
use crate::lsp::Language;

/// Per-language warm-up grace before a freshly-started LSP server is treated as
/// query-ready. Heavy indexers need longer before their first request stops
/// timing out (the server is still building its initial index); light servers
/// are ready almost immediately. `floor` is the configured global
/// `lsp.warmup_grace_secs`, applied as a minimum to every language.
fn warmup_grace_for(language: &Language, floor: Duration) -> Duration {
    let lang_min = match language {
        // jdtls indexes the whole workspace + resolves the build before answering.
        Language::Java => Duration::from_secs(120),
        // rust-analyzer runs `cargo metadata` and indexes the crate graph.
        Language::Rust => Duration::from_secs(60),
        // gopls loads the module/workspace; clangd parses the compile DB.
        Language::Go | Language::C | Language::Cpp => Duration::from_secs(30),
        // pyright / tsserver / bash-ls / dart / intelephense: fast to answer.
        _ => Duration::from_secs(5),
    };
    floor.max(lang_min)
}

impl LanguageServerManager {
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
                    drop(inst);
                    return Ok(Arc::new(instance.lock().await.clone()));
                }
            }
        }

        // Global fan-out guard: LSP servers are multi-GB each, so cap the TOTAL
        // running across all projects. Without this, N active projects × M
        // languages eagerly spawn N×M heavyweight processes and exhaust host
        // memory (the OOM/host-freeze runaway). Counts already-registered
        // running/initializing servers; a fresh server is only created below.
        // This is a check-then-act under a read lock (not a reservation), so
        // highly-concurrent starts may overshoot the cap by the number of
        // in-flight calls — acceptable for a safety valve whose job is to
        // prevent a dozen servers, not to enforce an exact count.
        {
            let servers = self.servers.read().await;
            let running = servers
                .values()
                .filter(|s| {
                    matches!(s.status, ServerStatus::Running | ServerStatus::Initializing)
                })
                .count();
            if running >= self.config.max_global_servers {
                tracing::warn!(
                    project_id = project_id,
                    language = ?language,
                    running,
                    max = self.config.max_global_servers,
                    "Global LSP server cap reached; refusing to start new server"
                );
                return Err(ProjectLspError::ServerUnavailable {
                    project_id: project_id.to_string(),
                    language: language.clone(),
                });
            }
        }

        // Check if we have a server available for this language
        let available = self.available_servers.read().await;
        let server_names =
            available
                .get(&language)
                .ok_or_else(|| ProjectLspError::LanguageNotSupported {
                    language: language.clone(),
                })?;

        if server_names.is_empty() {
            return Err(ProjectLspError::LanguageNotSupported {
                language: language.clone(),
            });
        }

        let server_name = &server_names[0];

        // Find the server executable path
        let server_path =
            which::which(server_name).map_err(|_| ProjectLspError::ServerUnavailable {
                project_id: project_id.to_string(),
                language: language.clone(),
            })?;

        tracing::info!(
            project_id = project_id,
            language = ?language,
            server = server_name,
            path = %server_path.display(),
            "Starting language server"
        );

        // Create and start the server instance
        let instance = self
            .create_and_start_instance(
                project_id,
                &language,
                &key,
                server_name,
                server_path,
                project_root,
            )
            .await?;

        // Store the state and instance
        self.register_server(&key, project_id, &language, project_root, &instance)
            .await;

        Ok(Arc::new(instance))
    }

    /// Create a server instance and start it
    async fn create_and_start_instance(
        &self,
        project_id: &str,
        language: &Language,
        key: &ProjectLanguageKey,
        server_name: &str,
        server_path: std::path::PathBuf,
        project_root: &Path,
    ) -> ProjectLspResult<ServerInstance> {
        use super::super::detection::{DetectedServer, ServerCapabilities};

        let detected = DetectedServer {
            name: server_name.to_string(),
            path: server_path,
            languages: vec![language.clone()],
            version: None,
            capabilities: ServerCapabilities::default(),
            priority: 1,
        };

        let mut instance = ServerInstance::new(detected, self.config.lsp_config.clone())
            .await
            .map_err(ProjectLspError::Lsp)?
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
            let mut servers = self.servers.write().await;
            if let Some(state) = servers.get_mut(key) {
                state.status = ServerStatus::Failed;
                state.last_error = Some(e.to_string());
            }
            return Err(ProjectLspError::Lsp(e));
        }

        Ok(instance)
    }

    /// Register a successfully started server in the state and instance maps
    async fn register_server(
        &self,
        key: &ProjectLanguageKey,
        project_id: &str,
        language: &Language,
        project_root: &Path,
        instance: &ServerInstance,
    ) {
        let state = ProjectServerState {
            project_id: project_id.to_string(),
            language: language.clone(),
            project_root: project_root.to_path_buf(),
            status: ServerStatus::Running,
            restart_count: 0,
            last_error: None,
            is_active: true,
            last_healthy_time: Some(Utc::now()),
            last_enrichment_at: Some(std::time::Instant::now()),
            marked_unavailable: false,
        };

        {
            let mut servers = self.servers.write().await;
            servers.insert(key.clone(), state);
        }
        {
            let mut instances = self.instances.write().await;
            instances.insert(
                key.clone(),
                Arc::new(tokio::sync::Mutex::new(instance.clone())),
            );
        }
        {
            // Defer enrichment for this server until it has had time to index,
            // so the first query doesn't time out against a still-warming server.
            let grace = warmup_grace_for(language, self.config.lsp_config.warmup_grace);
            let mut ready_at = self.ready_at.write().await;
            ready_at.insert(key.clone(), Instant::now() + grace);
        }

        tracing::info!(
            project_id = project_id,
            language = ?language,
            warmup_grace_secs =
                warmup_grace_for(language, self.config.lsp_config.warmup_grace).as_secs(),
            "Language server started successfully"
        );

        {
            let mut metrics = self.metrics.write().await;
            metrics.total_server_starts += 1;
        }
    }

    /// Stop a server for a specific project and language
    pub async fn stop_server(&self, project_id: &str, language: Language) -> ProjectLspResult<()> {
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
        {
            let mut ready_at = self.ready_at.write().await;
            ready_at.remove(&key);
        }

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

        // Track server stop metric
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_server_stops += 1;
        }

        Ok(())
    }

    /// Get a server instance for a specific project and language
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
    pub async fn is_server_running(&self, project_id: &str, language: Language) -> bool {
        if let Some(state) = self.get_server_state(project_id, language).await {
            matches!(state.status, ServerStatus::Running)
        } else {
            false
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

    /// Shutdown the manager and all servers
    pub async fn shutdown(&self) -> ProjectLspResult<()> {
        *self.running.write().await = false;

        let keys: Vec<_> = {
            let servers = self.servers.read().await;
            servers.keys().cloned().collect()
        };

        for key in keys {
            self.stop_server(&key.project_id, key.language).await?;
        }

        // Clear cache
        self.cache.lock().await.clear();

        tracing::info!("LanguageServerManager shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod warmup_grace_tests {
    use super::warmup_grace_for;
    use crate::lsp::Language;
    use std::time::Duration;

    #[test]
    fn heavy_languages_wait_longer_than_the_floor() {
        let floor = Duration::from_secs(15);
        assert_eq!(
            warmup_grace_for(&Language::Java, floor),
            Duration::from_secs(120)
        );
        assert_eq!(
            warmup_grace_for(&Language::Rust, floor),
            Duration::from_secs(60)
        );
        assert_eq!(
            warmup_grace_for(&Language::Go, floor),
            Duration::from_secs(30)
        );
        assert_eq!(
            warmup_grace_for(&Language::Cpp, floor),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn light_languages_use_the_configured_floor() {
        // Floor (15s) above the light minimum (5s) → floor wins.
        assert_eq!(
            warmup_grace_for(&Language::Python, Duration::from_secs(15)),
            Duration::from_secs(15)
        );
        // Floor below the light minimum → the 5s minimum wins.
        assert_eq!(
            warmup_grace_for(&Language::Python, Duration::from_secs(2)),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn a_high_floor_raises_even_heavy_languages() {
        assert_eq!(
            warmup_grace_for(&Language::Go, Duration::from_secs(90)),
            Duration::from_secs(90)
        );
    }
}

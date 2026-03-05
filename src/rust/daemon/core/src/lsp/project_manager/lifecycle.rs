//! Server lifecycle management: start, stop, shutdown, and server queries.

use std::path::Path;
use std::sync::Arc;

use chrono::Utc;

use super::{
    LanguageServerManager, ProjectLanguageKey, ProjectLspError, ProjectLspResult,
    ProjectServerState, ServerInstance, ServerStatus,
};
use crate::lsp::Language;

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

        tracing::info!(
            project_id = project_id,
            language = ?language,
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
        self.cache.write().await.clear();

        tracing::info!("LanguageServerManager shutdown complete");
        Ok(())
    }
}

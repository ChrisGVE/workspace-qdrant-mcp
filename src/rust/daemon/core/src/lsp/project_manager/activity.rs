//! Project activation and deactivation tracking.

use super::{Language, LanguageServerManager, ProjectLspResult};

impl LanguageServerManager {
    /// Mark a project as active (Task 1.18)
    ///
    /// This enables state persistence and allows server recovery for this project.
    pub async fn mark_project_active(&self, project_id: &str) {
        let mut active = self.active_projects.write().await;
        active.insert(project_id.to_string());
        tracing::debug!(project_id = project_id, "Marked project as active");
    }

    /// Mark a project as inactive (Task 1.18)
    ///
    /// This disables state persistence and server recovery for this project.
    pub async fn mark_project_inactive(&self, project_id: &str) {
        let mut active = self.active_projects.write().await;
        active.remove(project_id);
        tracing::debug!(project_id = project_id, "Marked project as inactive");
    }

    /// Check if a project is currently active (Task 1.18)
    pub async fn is_project_active(&self, project_id: &str) -> bool {
        let active = self.active_projects.read().await;
        active.contains(project_id)
    }

    /// Evict LSP servers that haven't been used within `idle_timeout`.
    ///
    /// Returns the list of (project_id, language) pairs that were stopped.
    pub async fn evict_idle_servers(
        &self,
        idle_timeout: std::time::Duration,
    ) -> Vec<(String, Language)> {
        let now = std::time::Instant::now();

        // Collect keys of idle servers
        let idle_keys: Vec<_> = {
            let servers = self.servers.read().await;
            servers
                .iter()
                .filter(|(_, state)| {
                    if let Some(last) = state.last_enrichment_at {
                        now.duration_since(last) > idle_timeout
                    } else {
                        // Never enriched — treat as idle
                        true
                    }
                })
                .map(|(key, _)| key.clone())
                .collect()
        };

        let mut evicted = Vec::new();
        for key in idle_keys {
            tracing::info!(
                project_id = %key.project_id,
                language = ?key.language,
                "Evicting idle LSP server"
            );
            if self
                .stop_server(&key.project_id, key.language.clone())
                .await
                .is_ok()
            {
                evicted.push((key.project_id.clone(), key.language));
            }
        }
        evicted
    }

    /// Restore servers for a project that was previously active.
    ///
    /// Note: Server state persistence has been removed as part of 3-table SQLite
    /// compliance. This method now returns an empty list as there is no persisted
    /// state to restore. Servers will be started fresh when `start_server` is called.
    pub async fn restore_project_servers(
        &self,
        _project_id: &str,
    ) -> ProjectLspResult<Vec<Language>> {
        Ok(vec![])
    }
}

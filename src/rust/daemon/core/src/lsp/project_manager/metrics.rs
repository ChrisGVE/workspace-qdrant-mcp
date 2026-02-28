//! Usage metrics, statistics, and available-language queries.

use serde::{Deserialize, Serialize};

use super::{LanguageServerManager, LspMetrics};
use crate::lsp::Language;

/// Statistics for the project LSP manager
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectLspStats {
    /// Number of currently active LSP servers
    pub active_servers: usize,
    /// Total number of LSP servers (active + inactive)
    pub total_servers: usize,
    /// Number of languages with available servers
    pub available_languages: usize,
    /// Number of entries in the enrichment cache
    pub cache_entries: usize,
    /// LSP usage metrics
    pub metrics: LspMetrics,
}

impl LanguageServerManager {
    /// Get a snapshot of current LSP metrics
    pub async fn get_metrics(&self) -> LspMetrics {
        self.metrics.read().await.snapshot()
    }

    /// Reset all metrics to zero
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = LspMetrics::new();
    }

    /// Get statistics for the manager
    pub async fn stats(&self) -> ProjectLspStats {
        let servers = self.servers.read().await;
        let available = self.available_servers.read().await;
        let cache = self.cache.read().await;
        let metrics = self.metrics.read().await;

        ProjectLspStats {
            active_servers: servers.values().filter(|s| s.is_active).count(),
            total_servers: servers.len(),
            available_languages: available.len(),
            cache_entries: cache.len(),
            metrics: metrics.snapshot(),
        }
    }

    /// Get list of languages that have available servers (Task 1.19)
    pub async fn available_languages(&self) -> Vec<Language> {
        let available = self.available_servers.read().await;
        available.keys().cloned().collect()
    }
}

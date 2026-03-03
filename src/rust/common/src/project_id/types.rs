//! Configuration types for project ID calculation

use serde::{Deserialize, Serialize};

/// Configuration for project ID calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisambiguationConfig {
    /// Length of project ID hash suffix
    pub id_hash_length: usize,

    /// Whether to include disambiguation in project_id
    pub enable_disambiguation: bool,

    /// Alias retention period in days
    pub alias_retention_days: u32,
}

impl Default for DisambiguationConfig {
    fn default() -> Self {
        Self {
            id_hash_length: 12,
            enable_disambiguation: true,
            alias_retention_days: 30,
        }
    }
}

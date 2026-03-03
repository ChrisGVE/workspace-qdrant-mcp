//! Types for the hierarchy builder: config, results, and errors.

/// Configuration for the hierarchy rebuild job.
#[derive(Debug, Clone)]
pub struct HierarchyRebuildConfig {
    /// Minimum number of concept tags for a tenant to trigger rebuild
    pub min_tags_threshold: usize,
    /// Canonical tag clustering config
    pub canonical: super::super::canonical_tags::CanonicalConfig,
    /// Collection to operate on (default: "projects")
    pub collection: String,
}

impl Default for HierarchyRebuildConfig {
    fn default() -> Self {
        Self {
            min_tags_threshold: 10,
            canonical: super::super::canonical_tags::CanonicalConfig::default(),
            collection: "projects".to_string(),
        }
    }
}

/// Result of a hierarchy rebuild for a single tenant.
#[derive(Debug, Clone)]
pub struct RebuildResult {
    pub tenant_id: String,
    pub tags_collected: usize,
    pub level3_count: usize,
    pub level2_count: usize,
    pub level1_count: usize,
    pub edges_created: usize,
}

/// Result of a full rebuild across all tenants.
#[derive(Debug, Clone)]
pub struct FullRebuildResult {
    pub tenants_processed: usize,
    pub tenants_skipped: usize,
    pub total_canonical_tags: usize,
    pub total_edges: usize,
    pub tenant_results: Vec<RebuildResult>,
}

/// Errors from hierarchy builder operations.
#[derive(Debug, thiserror::Error)]
pub enum HierarchyError {
    #[error("Database error: {0}")]
    Database(sqlx::Error),

    #[error("Embedding error: {0}")]
    Embedding(crate::embedding::EmbeddingError),
}

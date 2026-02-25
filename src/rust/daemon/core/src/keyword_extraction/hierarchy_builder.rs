//! Nightly canonical tag hierarchy rebuild.
//!
//! Periodically rebuilds the canonical tag graph per tenant by:
//! 1. Collecting all concept tags from the `tags` table
//! 2. Embedding tag phrases for vector similarity
//! 3. Running `canonical_tags::build_hierarchy()` (dedup → cluster)
//! 4. Writing results to `canonical_tags` and `tag_hierarchy_edges` tables
//!
//! Triggers:
//! - Scheduled nightly at 2 AM local time
//! - Manual via CLI: `wqm tags rebuild --tenant <id>`

use std::sync::Arc;

use sqlx::{Row, SqlitePool};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use crate::embedding::EmbeddingGenerator;
use wqm_common::timestamps::now_utc;

use super::canonical_tags::{self, CanonicalConfig, CanonicalHierarchy, TagWithVector};

/// Configuration for the hierarchy rebuild job.
#[derive(Debug, Clone)]
pub struct HierarchyRebuildConfig {
    /// Minimum number of concept tags for a tenant to trigger rebuild
    pub min_tags_threshold: usize,
    /// Canonical tag clustering config
    pub canonical: CanonicalConfig,
    /// Collection to operate on (default: "projects")
    pub collection: String,
}

impl Default for HierarchyRebuildConfig {
    fn default() -> Self {
        Self {
            min_tags_threshold: 10,
            canonical: CanonicalConfig::default(),
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

/// Hierarchy builder for nightly canonical tag rebuilds.
pub struct HierarchyBuilder {
    pool: SqlitePool,
    embedding_generator: Arc<EmbeddingGenerator>,
    config: HierarchyRebuildConfig,
}

impl HierarchyBuilder {
    /// Create a new hierarchy builder.
    pub fn new(
        pool: SqlitePool,
        embedding_generator: Arc<EmbeddingGenerator>,
        config: HierarchyRebuildConfig,
    ) -> Self {
        Self {
            pool,
            embedding_generator,
            config,
        }
    }

    /// Rebuild the canonical tag hierarchy for a single tenant.
    ///
    /// Returns `None` if the tenant has fewer tags than the threshold.
    pub async fn rebuild_tenant(&self, tenant_id: &str) -> Result<Option<RebuildResult>, HierarchyError> {
        info!(tenant_id, "Starting canonical tag hierarchy rebuild");

        // Step 1: Collect distinct concept tags with doc counts
        let tag_rows: Vec<(String, u32)> = self.collect_concept_tags(tenant_id).await?;

        if tag_rows.len() < self.config.min_tags_threshold {
            debug!(
                tenant_id,
                tag_count = tag_rows.len(),
                threshold = self.config.min_tags_threshold,
                "Skipping tenant: below tag threshold"
            );
            return Ok(None);
        }

        info!(
            tenant_id,
            tag_count = tag_rows.len(),
            "Collected concept tags, generating embeddings"
        );

        // Step 2: Embed all tag phrases
        let phrases: Vec<String> = tag_rows.iter().map(|(phrase, _)| phrase.clone()).collect();
        let vectors = self.embed_phrases(&phrases).await?;

        // Step 3: Build TagWithVector collection
        let tags_with_vectors: Vec<TagWithVector> = tag_rows
            .into_iter()
            .zip(vectors)
            .map(|((phrase, doc_count), vector)| TagWithVector {
                phrase,
                vector,
                doc_count,
            })
            .collect();

        let tags_collected = tags_with_vectors.len();

        // Step 4: Build hierarchy
        let hierarchy = canonical_tags::build_hierarchy(&tags_with_vectors, &self.config.canonical);

        info!(
            tenant_id,
            level3 = hierarchy.level3.len(),
            level2 = hierarchy.level2.len(),
            level1 = hierarchy.level1.len(),
            "Hierarchy built, persisting to SQLite"
        );

        // Step 5: Persist to SQLite (in a transaction)
        let edges_created = self.persist_hierarchy(tenant_id, &hierarchy).await?;

        let result = RebuildResult {
            tenant_id: tenant_id.to_string(),
            tags_collected,
            level3_count: hierarchy.level3.len(),
            level2_count: hierarchy.level2.len(),
            level1_count: hierarchy.level1.len(),
            edges_created,
        };

        info!(
            tenant_id,
            tags_collected = result.tags_collected,
            canonical_tags = result.level3_count + result.level2_count + result.level1_count,
            edges = result.edges_created,
            "Canonical tag hierarchy rebuild complete"
        );

        Ok(Some(result))
    }

    /// Rebuild the canonical tag hierarchy for all tenants with sufficient tags.
    pub async fn rebuild_all(&self) -> Result<FullRebuildResult, HierarchyError> {
        info!("Starting full canonical tag hierarchy rebuild for all tenants");

        let tenants = self.get_active_tenants().await?;
        info!(tenant_count = tenants.len(), "Found active tenants");

        let mut result = FullRebuildResult {
            tenants_processed: 0,
            tenants_skipped: 0,
            total_canonical_tags: 0,
            total_edges: 0,
            tenant_results: Vec::new(),
        };

        for tenant_id in &tenants {
            match self.rebuild_tenant(tenant_id).await {
                Ok(Some(tenant_result)) => {
                    result.total_canonical_tags += tenant_result.level3_count
                        + tenant_result.level2_count
                        + tenant_result.level1_count;
                    result.total_edges += tenant_result.edges_created;
                    result.tenant_results.push(tenant_result);
                    result.tenants_processed += 1;
                }
                Ok(None) => {
                    result.tenants_skipped += 1;
                }
                Err(e) => {
                    error!(tenant_id, error = %e, "Failed to rebuild hierarchy for tenant");
                    result.tenants_skipped += 1;
                }
            }
        }

        info!(
            processed = result.tenants_processed,
            skipped = result.tenants_skipped,
            canonical_tags = result.total_canonical_tags,
            edges = result.total_edges,
            "Full canonical tag hierarchy rebuild complete"
        );

        Ok(result)
    }

    /// Start the nightly scheduled rebuild job.
    ///
    /// Spawns a background task that runs at approximately 2 AM local time.
    /// Returns a cancellation token that can be used to stop the job.
    pub fn start_scheduled(self: Arc<Self>, cancel: CancellationToken) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                let delay = delay_until_next_2am();
                info!(
                    delay_secs = delay.as_secs(),
                    "Hierarchy rebuild scheduled (next run in {}h {}m)",
                    delay.as_secs() / 3600,
                    (delay.as_secs() % 3600) / 60
                );

                tokio::select! {
                    _ = tokio::time::sleep(delay) => {
                        info!("Running scheduled nightly hierarchy rebuild");
                        match self.rebuild_all().await {
                            Ok(result) => {
                                info!(
                                    "Nightly rebuild complete: {} tenants processed, {} canonical tags",
                                    result.tenants_processed,
                                    result.total_canonical_tags
                                );
                            }
                            Err(e) => {
                                error!("Nightly hierarchy rebuild failed: {}", e);
                            }
                        }
                    }
                    _ = cancel.cancelled() => {
                        info!("Hierarchy rebuild scheduler stopped");
                        return;
                    }
                }
            }
        })
    }

    /// Check if a rebuild is needed (tags exist but no canonical hierarchy).
    pub async fn needs_rebuild(&self) -> bool {
        let has_tags: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM tags WHERE tag_type = 'concept' LIMIT 1)",
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(false);

        if !has_tags {
            return false;
        }

        let has_canonical: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM canonical_tags LIMIT 1)",
        )
        .fetch_one(&self.pool)
        .await
        .unwrap_or(true); // default true to skip rebuild on error

        !has_canonical
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Collect distinct concept tags with their doc counts for a tenant.
    async fn collect_concept_tags(&self, tenant_id: &str) -> Result<Vec<(String, u32)>, HierarchyError> {
        let rows = sqlx::query(
            "SELECT tag, COUNT(DISTINCT doc_id) as doc_count \
             FROM tags \
             WHERE tenant_id = ?1 AND collection = ?2 AND tag_type = 'concept' \
             GROUP BY tag \
             ORDER BY doc_count DESC"
        )
        .bind(tenant_id)
        .bind(&self.config.collection)
        .fetch_all(&self.pool)
        .await
        .map_err(HierarchyError::Database)?;

        Ok(rows
            .iter()
            .map(|row| {
                let tag: String = row.get("tag");
                let doc_count: i32 = row.get("doc_count");
                (tag, doc_count as u32)
            })
            .collect())
    }

    /// Get all active tenants that have concept tags in the collection.
    async fn get_active_tenants(&self) -> Result<Vec<String>, HierarchyError> {
        let rows = sqlx::query(
            "SELECT DISTINCT tenant_id FROM tags \
             WHERE collection = ?1 AND tag_type = 'concept' \
             GROUP BY tenant_id \
             HAVING COUNT(DISTINCT tag) >= ?2"
        )
        .bind(&self.config.collection)
        .bind(self.config.min_tags_threshold as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(HierarchyError::Database)?;

        Ok(rows.iter().map(|row| row.get("tenant_id")).collect())
    }

    /// Embed a batch of tag phrases.
    async fn embed_phrases(&self, phrases: &[String]) -> Result<Vec<Vec<f32>>, HierarchyError> {
        if phrases.is_empty() {
            return Ok(Vec::new());
        }

        let results = self
            .embedding_generator
            .generate_embeddings_batch(phrases, "all-MiniLM-L6-v2")
            .await
            .map_err(HierarchyError::Embedding)?;

        Ok(results.into_iter().map(|r| r.dense.vector).collect())
    }

    /// Persist a canonical hierarchy to SQLite, replacing previous data for this tenant.
    async fn persist_hierarchy(
        &self,
        tenant_id: &str,
        hierarchy: &CanonicalHierarchy,
    ) -> Result<usize, HierarchyError> {
        let mut tx = self.pool.begin().await.map_err(HierarchyError::Database)?;
        let now_str = now_utc();

        // Clear existing canonical tags and edges for this tenant
        // Edges are ON DELETE CASCADE so deleting canonical_tags clears them
        sqlx::query("DELETE FROM canonical_tags WHERE tenant_id = ?1 AND collection = ?2")
            .bind(tenant_id)
            .bind(&self.config.collection)
            .execute(&mut *tx)
            .await
            .map_err(HierarchyError::Database)?;

        let mut edges_created = 0usize;

        // Insert level 1 (broad) tags first (no parents)
        let mut level1_ids: Vec<i64> = Vec::with_capacity(hierarchy.level1.len());
        for tag in &hierarchy.level1 {
            let result = sqlx::query(
                "INSERT INTO canonical_tags (canonical_name, level, parent_id, tenant_id, collection, created_at) \
                 VALUES (?1, ?2, NULL, ?3, ?4, ?5)"
            )
            .bind(&tag.label)
            .bind(1_i32)
            .bind(tenant_id)
            .bind(&self.config.collection)
            .bind(&now_str)
            .execute(&mut *tx)
            .await
            .map_err(HierarchyError::Database)?;

            level1_ids.push(result.last_insert_rowid());
        }

        // Insert level 2 (mid) tags, linking to level 1 parents where possible
        let mut level2_ids: Vec<i64> = Vec::with_capacity(hierarchy.level2.len());
        for tag in &hierarchy.level2 {
            let parent_id = tag.parent_index.and_then(|idx| level1_ids.get(idx).copied());

            let result = sqlx::query(
                "INSERT INTO canonical_tags (canonical_name, level, parent_id, tenant_id, collection, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
            )
            .bind(&tag.label)
            .bind(2_i32)
            .bind(parent_id)
            .bind(tenant_id)
            .bind(&self.config.collection)
            .bind(&now_str)
            .execute(&mut *tx)
            .await
            .map_err(HierarchyError::Database)?;

            let l2_id = result.last_insert_rowid();
            level2_ids.push(l2_id);

            // Create edge from level 1 parent to this level 2 tag
            if let Some(pid) = parent_id {
                sqlx::query(
                    "INSERT OR IGNORE INTO tag_hierarchy_edges (parent_tag_id, child_tag_id, similarity_score, tenant_id) \
                     VALUES (?1, ?2, ?3, ?4)"
                )
                .bind(pid)
                .bind(l2_id)
                .bind(0.0) // similarity computed during clustering
                .bind(tenant_id)
                .execute(&mut *tx)
                .await
                .map_err(HierarchyError::Database)?;

                edges_created += 1;
            }
        }

        // Insert level 3 (fine) tags, linking to level 2 parents where possible
        for tag in &hierarchy.level3 {
            let parent_id = tag.parent_index.and_then(|idx| level2_ids.get(idx).copied());

            let result = sqlx::query(
                "INSERT INTO canonical_tags (canonical_name, level, parent_id, tenant_id, collection, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
            )
            .bind(&tag.label)
            .bind(3_i32)
            .bind(parent_id)
            .bind(tenant_id)
            .bind(&self.config.collection)
            .bind(&now_str)
            .execute(&mut *tx)
            .await
            .map_err(HierarchyError::Database)?;

            let l3_id = result.last_insert_rowid();

            // Create edge from level 2 parent to this level 3 tag
            if let Some(pid) = parent_id {
                sqlx::query(
                    "INSERT OR IGNORE INTO tag_hierarchy_edges (parent_tag_id, child_tag_id, similarity_score, tenant_id) \
                     VALUES (?1, ?2, ?3, ?4)"
                )
                .bind(pid)
                .bind(l3_id)
                .bind(0.0)
                .bind(tenant_id)
                .execute(&mut *tx)
                .await
                .map_err(HierarchyError::Database)?;

                edges_created += 1;
            }
        }

        tx.commit().await.map_err(HierarchyError::Database)?;

        Ok(edges_created)
    }
}

/// Calculate the delay until the next 2:00 AM local time.
fn delay_until_next_2am() -> std::time::Duration {
    let now = chrono::Local::now();
    let today_2am = now
        .date_naive()
        .and_hms_opt(2, 0, 0)
        .expect("valid time");

    let target = if now.naive_local() < today_2am {
        // 2 AM hasn't passed today yet
        today_2am
    } else {
        // 2 AM already passed, schedule for tomorrow
        today_2am + chrono::Duration::days(1)
    };

    let target_local = target
        .and_local_timezone(now.timezone())
        .single()
        .unwrap_or_else(|| {
            // DST ambiguity fallback: use the latest option
            target
                .and_local_timezone(now.timezone())
                .latest()
                .expect("at least one valid local time")
        });

    let diff = target_local - now;
    diff.to_std().unwrap_or(std::time::Duration::from_secs(3600))
}

/// Errors from hierarchy builder operations.
#[derive(Debug, thiserror::Error)]
pub enum HierarchyError {
    #[error("Database error: {0}")]
    Database(sqlx::Error),

    #[error("Embedding error: {0}")]
    Embedding(crate::embedding::EmbeddingError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HierarchyRebuildConfig::default();
        assert_eq!(config.min_tags_threshold, 10);
        assert_eq!(config.collection, "projects");
        assert!((config.canonical.merge_threshold - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_rebuild_result_fields() {
        let result = RebuildResult {
            tenant_id: "test-tenant".to_string(),
            tags_collected: 50,
            level3_count: 30,
            level2_count: 10,
            level1_count: 3,
            edges_created: 40,
        };
        assert_eq!(result.tenant_id, "test-tenant");
        assert_eq!(result.tags_collected, 50);
    }

    #[test]
    fn test_full_rebuild_result_defaults() {
        let result = FullRebuildResult {
            tenants_processed: 0,
            tenants_skipped: 0,
            total_canonical_tags: 0,
            total_edges: 0,
            tenant_results: Vec::new(),
        };
        assert!(result.tenant_results.is_empty());
    }

    #[test]
    fn test_delay_until_next_2am_is_positive() {
        let delay = delay_until_next_2am();
        assert!(delay.as_secs() > 0, "Delay should be positive");
        // Should never be more than 24 hours
        assert!(
            delay.as_secs() < 86400 + 3600, // 25 hours max (DST edge)
            "Delay should be less than ~25 hours, got {}s",
            delay.as_secs()
        );
    }

    #[test]
    fn test_delay_is_reasonable() {
        let delay = delay_until_next_2am();
        // At minimum a few seconds, at maximum ~24h
        assert!(delay.as_secs() >= 1);
        assert!(delay.as_secs() <= 90000); // 25 hours
    }

    #[test]
    fn test_hierarchy_error_display() {
        let err = HierarchyError::Database(sqlx::Error::RowNotFound);
        assert!(err.to_string().contains("Database error"));
    }
}

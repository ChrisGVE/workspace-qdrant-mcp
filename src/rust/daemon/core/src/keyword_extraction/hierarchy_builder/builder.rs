//! HierarchyBuilder: orchestrates the nightly canonical tag hierarchy rebuild.

use std::sync::Arc;

use sqlx::{Row, SqlitePool};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use crate::embedding::EmbeddingGenerator;
use wqm_common::timestamps::now_utc;

use crate::keyword_extraction::canonical_tags::{self, CanonicalHierarchy, TagWithVector};

use super::scheduler::delay_until_next_2am;
use super::types::{FullRebuildResult, HierarchyError, HierarchyRebuildConfig, RebuildResult};

/// Hierarchy builder for nightly canonical tag rebuilds.
pub struct HierarchyBuilder {
    pub(super) pool: SqlitePool,
    pub(super) embedding_generator: Arc<EmbeddingGenerator>,
    pub(super) config: HierarchyRebuildConfig,
}

impl std::fmt::Debug for HierarchyBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HierarchyBuilder")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
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
    pub async fn rebuild_tenant(
        &self,
        tenant_id: &str,
    ) -> Result<Option<RebuildResult>, HierarchyError> {
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
    pub fn start_scheduled(
        self: Arc<Self>,
        cancel: CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
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

        let has_canonical: bool =
            sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM canonical_tags LIMIT 1)")
                .fetch_one(&self.pool)
                .await
                .unwrap_or(true); // default true to skip rebuild on error

        !has_canonical
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Collect distinct concept tags with their doc counts for a tenant.
    async fn collect_concept_tags(
        &self,
        tenant_id: &str,
    ) -> Result<Vec<(String, u32)>, HierarchyError> {
        let rows = sqlx::query(
            "SELECT tag, COUNT(DISTINCT doc_id) as doc_count \
             FROM tags \
             WHERE tenant_id = ?1 AND collection = ?2 AND tag_type = 'concept' \
             GROUP BY tag \
             ORDER BY doc_count DESC",
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
             HAVING COUNT(DISTINCT tag) >= ?2",
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

        // Clear existing data (edges cascade via ON DELETE CASCADE)
        sqlx::query("DELETE FROM canonical_tags WHERE tenant_id = ?1 AND collection = ?2")
            .bind(tenant_id)
            .bind(&self.config.collection)
            .execute(&mut *tx)
            .await
            .map_err(HierarchyError::Database)?;

        let mut edges_created = 0usize;

        let level1_ids = insert_level1_tags(
            &mut tx,
            &hierarchy.level1,
            tenant_id,
            &self.config.collection,
            &now_str,
        )
        .await?;

        let level2_ids = insert_level_n_tags(
            &mut tx,
            &hierarchy.level2,
            2,
            &level1_ids,
            tenant_id,
            &self.config.collection,
            &now_str,
            &mut edges_created,
        )
        .await?;

        insert_level_n_tags(
            &mut tx,
            &hierarchy.level3,
            3,
            &level2_ids,
            tenant_id,
            &self.config.collection,
            &now_str,
            &mut edges_created,
        )
        .await?;

        tx.commit().await.map_err(HierarchyError::Database)?;
        Ok(edges_created)
    }
}

/// Insert level-1 (root) canonical tags and return their row IDs.
async fn insert_level1_tags(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    tags: &[crate::keyword_extraction::canonical_tags::CanonicalTag],
    tenant_id: &str,
    collection: &str,
    now_str: &str,
) -> Result<Vec<i64>, HierarchyError> {
    let mut ids = Vec::with_capacity(tags.len());
    for tag in tags {
        let result = sqlx::query(
            "INSERT INTO canonical_tags \
             (canonical_name, level, parent_id, tenant_id, collection, created_at) \
             VALUES (?1, ?2, NULL, ?3, ?4, ?5)",
        )
        .bind(&tag.label)
        .bind(1_i32)
        .bind(tenant_id)
        .bind(collection)
        .bind(now_str)
        .execute(&mut **tx)
        .await
        .map_err(HierarchyError::Database)?;
        ids.push(result.last_insert_rowid());
    }
    Ok(ids)
}

/// Insert level-N (N ≥ 2) canonical tags with parent links, returning their row IDs.
async fn insert_level_n_tags(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    tags: &[crate::keyword_extraction::canonical_tags::CanonicalTag],
    level: i32,
    parent_ids: &[i64],
    tenant_id: &str,
    collection: &str,
    now_str: &str,
    edges_created: &mut usize,
) -> Result<Vec<i64>, HierarchyError> {
    let mut ids = Vec::with_capacity(tags.len());
    for tag in tags {
        let parent_id = tag
            .parent_index
            .and_then(|idx| parent_ids.get(idx).copied());
        let result = sqlx::query(
            "INSERT INTO canonical_tags \
             (canonical_name, level, parent_id, tenant_id, collection, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        )
        .bind(&tag.label)
        .bind(level)
        .bind(parent_id)
        .bind(tenant_id)
        .bind(collection)
        .bind(now_str)
        .execute(&mut **tx)
        .await
        .map_err(HierarchyError::Database)?;
        let child_id = result.last_insert_rowid();
        ids.push(child_id);

        if let Some(pid) = parent_id {
            sqlx::query(
                "INSERT OR IGNORE INTO tag_hierarchy_edges \
                 (parent_tag_id, child_tag_id, similarity_score, tenant_id) \
                 VALUES (?1, ?2, ?3, ?4)",
            )
            .bind(pid)
            .bind(child_id)
            .bind(tag.parent_similarity.unwrap_or(0.0))
            .bind(tenant_id)
            .execute(&mut **tx)
            .await
            .map_err(HierarchyError::Database)?;
            *edges_created += 1;
        }
    }
    Ok(ids)
}

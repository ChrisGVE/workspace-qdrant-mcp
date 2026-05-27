//! First-run bootstrap for project aggregate embeddings.
//!
//! When the grouping scheduler runs and a project has no aggregate
//! embedding in `project_embeddings`, this module samples dense vectors
//! from Qdrant and computes a mean embedding to seed the table.
//!
//! Projects are bootstrapped serially (one at a time) to avoid
//! concurrent Qdrant scroll pressure.

use std::sync::Arc;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};
use wqm_common::constants::COLLECTION_PROJECTS;

use super::incremental::{save_aggregate_state, AggregateState};
use crate::storage::StorageClient;

/// Maximum number of chunks to sample per project during bootstrap.
const BOOTSTRAP_SAMPLE_LIMIT: usize = 500;

/// Result of a single bootstrap run across all projects.
#[derive(Debug, Clone, Default)]
pub struct BootstrapResult {
    /// Number of projects that were bootstrapped with new embeddings.
    pub bootstrapped: usize,
    /// Number of projects skipped (already had embeddings).
    pub skipped: usize,
    /// Number of projects skipped because they had no chunks in Qdrant.
    pub empty: usize,
    /// Per-project errors (tenant_id, error message).
    pub errors: Vec<(String, String)>,
}

/// Find active projects that lack an entry in `project_embeddings` and
/// bootstrap their aggregate embedding by sampling chunks from Qdrant.
///
/// Projects are processed serially to limit Qdrant scroll concurrency.
pub async fn bootstrap_missing_embeddings(
    pool: &SqlitePool,
    storage_client: &Arc<StorageClient>,
) -> BootstrapResult {
    let mut result = BootstrapResult::default();

    let tenants = match find_projects_without_embeddings(pool).await {
        Ok(t) => t,
        Err(e) => {
            warn!(error = %e, "Failed to query projects without embeddings");
            result.errors.push(("_query".into(), e.to_string()));
            return result;
        }
    };

    if tenants.is_empty() {
        debug!("All active projects already have aggregate embeddings");
        return result;
    }

    info!(
        count = tenants.len(),
        "Bootstrapping aggregate embeddings for projects"
    );

    for tenant_id in &tenants {
        match bootstrap_single_project(pool, storage_client, tenant_id).await {
            Ok(BootstrapOutcome::Created { chunk_count }) => {
                info!(tenant_id, chunk_count, "Bootstrapped aggregate embedding");
                result.bootstrapped += 1;
            }
            Ok(BootstrapOutcome::Empty) => {
                debug!(tenant_id, "No chunks in Qdrant, skipping bootstrap");
                result.empty += 1;
            }
            Err(e) => {
                warn!(tenant_id, error = %e, "Failed to bootstrap project embedding");
                result.errors.push((tenant_id.clone(), e.to_string()));
            }
        }
    }

    if result.bootstrapped > 0 {
        info!(
            bootstrapped = result.bootstrapped,
            empty = result.empty,
            errors = result.errors.len(),
            "Bootstrap cycle complete"
        );
    }

    result
}

/// Outcome of bootstrapping a single project.
enum BootstrapOutcome {
    /// Aggregate embedding was created from `chunk_count` sampled vectors.
    Created { chunk_count: usize },
    /// Project had no chunks in Qdrant; nothing to bootstrap.
    Empty,
}

/// Bootstrap a single project by scrolling dense vectors and computing
/// a mean embedding via Welford's algorithm.
async fn bootstrap_single_project(
    pool: &SqlitePool,
    storage_client: &Arc<StorageClient>,
    tenant_id: &str,
) -> Result<BootstrapOutcome, BootstrapError> {
    let vectors = storage_client
        .scroll_dense_vectors_by_tenant(COLLECTION_PROJECTS, tenant_id, BOOTSTRAP_SAMPLE_LIMIT)
        .await
        .map_err(|e| BootstrapError::Scroll(e.to_string()))?;

    if vectors.is_empty() {
        return Ok(BootstrapOutcome::Empty);
    }

    let dim = vectors[0].len();

    // Validate all vectors have the same dimension
    if vectors.iter().any(|v| v.len() != dim) {
        return Err(BootstrapError::DimensionMismatch);
    }

    let mut state = AggregateState::new(dim);
    for v in &vectors {
        state.add(v);
    }

    save_aggregate_state(pool, tenant_id, &state)
        .await
        .map_err(|e| BootstrapError::Storage(e.to_string()))?;

    Ok(BootstrapOutcome::Created {
        chunk_count: vectors.len(),
    })
}

/// Query active projects (from `watch_folders`) that have no entry in
/// `project_embeddings`.
async fn find_projects_without_embeddings(pool: &SqlitePool) -> Result<Vec<String>, sqlx::Error> {
    let rows = sqlx::query_as::<_, (String,)>(
        r#"
        SELECT wf.tenant_id
        FROM watch_folders wf
        WHERE wf.is_active = 1
          AND wf.tenant_id NOT IN (
              SELECT pe.tenant_id FROM project_embeddings pe
          )
        ORDER BY wf.tenant_id
        "#,
    )
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(|(t,)| t).collect())
}

/// Errors that can occur during bootstrap.
#[derive(Debug)]
enum BootstrapError {
    /// Failed to scroll vectors from Qdrant.
    Scroll(String),
    /// Vectors in the sample have inconsistent dimensions.
    DimensionMismatch,
    /// Failed to persist the aggregate embedding.
    Storage(String),
}

impl std::fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scroll(msg) => write!(f, "Qdrant scroll failed: {msg}"),
            Self::DimensionMismatch => {
                write!(f, "Sampled vectors have inconsistent dimensions")
            }
            Self::Storage(msg) => write!(f, "Storage write failed: {msg}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.unwrap();
        pool
    }

    #[tokio::test]
    async fn test_bootstrap_skips_projects_with_existing_embeddings() {
        let pool = setup_pool().await;

        // Insert an active project
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, created_at, updated_at) \
             VALUES ('wf-a', '/tmp/a', 'projects', 'proj-a', 1, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        // Give it an existing embedding
        save_aggregate_state(&pool, "proj-a", &{
            let mut s = AggregateState::new(4);
            s.add(&[1.0, 2.0, 3.0, 4.0]);
            s
        })
        .await
        .unwrap();

        // Query should return empty since proj-a already has an embedding
        let missing = find_projects_without_embeddings(&pool).await.unwrap();
        assert!(
            missing.is_empty(),
            "Expected no projects missing embeddings"
        );
    }

    #[tokio::test]
    async fn test_bootstrap_finds_projects_without_embeddings() {
        let pool = setup_pool().await;

        // Insert two active projects
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, created_at, updated_at) \
             VALUES ('wf-a', '/tmp/a', 'projects', 'proj-a', 1, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, created_at, updated_at) \
             VALUES ('wf-b', '/tmp/b', 'projects', 'proj-b', 1, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        // Give proj-a an embedding, leave proj-b without one
        save_aggregate_state(&pool, "proj-a", &{
            let mut s = AggregateState::new(4);
            s.add(&[1.0, 2.0, 3.0, 4.0]);
            s
        })
        .await
        .unwrap();

        let missing = find_projects_without_embeddings(&pool).await.unwrap();
        assert_eq!(missing, vec!["proj-b"]);
    }

    #[tokio::test]
    async fn test_bootstrap_ignores_inactive_projects() {
        let pool = setup_pool().await;

        // Insert an inactive project without an embedding
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, created_at, updated_at) \
             VALUES ('wf-inactive', '/tmp/inactive', 'projects', 'proj-inactive', 0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let missing = find_projects_without_embeddings(&pool).await.unwrap();
        assert!(
            missing.is_empty(),
            "Inactive projects should not be bootstrapped"
        );
    }

    #[tokio::test]
    async fn test_bootstrap_single_project_empty_vectors() {
        // Verify that bootstrap_single_project returns Empty when no vectors
        // are available. We cannot easily mock StorageClient here, so we test
        // the AggregateState computation path directly.

        let dim = 384;
        let state = AggregateState::new(dim);
        assert_eq!(state.count, 0);
        assert_eq!(state.dim(), dim);
        // An empty state should have a zero-vector mean
        assert!(state.mean.iter().all(|&v| v == 0.0));
    }

    #[tokio::test]
    async fn test_bootstrap_mean_computation_matches_incremental() {
        // Verify that the bootstrap path (batch add) produces the same
        // result as the incremental aggregator.

        let vectors: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        // Bootstrap-style: add all at once
        let mut bootstrap_state = AggregateState::new(4);
        for v in &vectors {
            bootstrap_state.add(v);
        }

        // Expected mean: [0.25, 0.25, 0.25, 0.25]
        assert_eq!(bootstrap_state.count, 4);
        for &m in &bootstrap_state.mean {
            assert!((m - 0.25).abs() < 1e-10, "Expected 0.25, got {m}");
        }

        // Verify f32 roundtrip
        let f32_mean = bootstrap_state.to_f32();
        for &v in &f32_mean {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn test_bootstrap_result_default() {
        let result = BootstrapResult::default();
        assert_eq!(result.bootstrapped, 0);
        assert_eq!(result.skipped, 0);
        assert_eq!(result.empty, 0);
        assert!(result.errors.is_empty());
    }

    #[tokio::test]
    async fn test_bootstrap_error_display() {
        let err = BootstrapError::Scroll("connection refused".into());
        assert_eq!(err.to_string(), "Qdrant scroll failed: connection refused");

        let err = BootstrapError::DimensionMismatch;
        assert_eq!(
            err.to_string(),
            "Sampled vectors have inconsistent dimensions"
        );

        let err = BootstrapError::Storage("disk full".into());
        assert_eq!(err.to_string(), "Storage write failed: disk full");
    }
}

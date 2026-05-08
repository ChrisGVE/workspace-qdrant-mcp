//! Startup-time consistency checks.
//!
//! Currently exposes [`check_dim_consistency`], the dim-mismatch guard the
//! daemon runs between queue-init (Phase 5) and gRPC start (Phase 6). It
//! refuses to bring the daemon up when the active embedding provider's output
//! dimensionality disagrees with the dim of the existing `projects` Qdrant
//! collection — unless the operator has explicitly passed
//! `--bootstrap-reembed` to migrate to a new dim.
//!
//! Per PRD §5.10 / §6.5: `active_dim` MUST come from
//! `settings.embedding.output_dim`, not from `provider.output_dim()` and not
//! from the background probe — the guard runs before the probe completes.

use tracing::{error, warn};
use wqm_common::constants::COLLECTION_PROJECTS;

use crate::embedding::EmbeddingError;
use crate::storage::StorageClient;

/// Verify the active embedding provider's output dim matches the dim of the
/// existing `projects` Qdrant collection (if any).
///
/// Returns `Ok(())` when:
/// - The `projects` collection does not exist (first-run path).
/// - The stored dim equals `active_dim`.
/// - `bootstrap_reembed` is set, regardless of stored dim.
///
/// Returns [`EmbeddingError::DimensionMismatch`] otherwise. The caller is
/// responsible for surfacing a human-readable error message and aborting
/// startup.
pub async fn check_dim_consistency(
    storage_client: &StorageClient,
    active_dim: usize,
    bootstrap_reembed: bool,
) -> Result<(), EmbeddingError> {
    check_dim_consistency_with_lookup(
        |name| async move {
            let exists = storage_client.collection_exists(name).await.map_err(|e| {
                EmbeddingError::InitializationError {
                    message: format!("collection_exists({}) failed: {}", name, e),
                }
            })?;
            if !exists {
                return Ok(None);
            }
            let info = storage_client
                .get_collection_info(name)
                .await
                .map_err(|e| EmbeddingError::InitializationError {
                    message: format!("get_collection_info({}) failed: {}", name, e),
                })?;
            Ok(info.vector_dimension)
        },
        active_dim,
        bootstrap_reembed,
    )
    .await
}

/// Test-friendly core of the guard — caller injects the dim lookup so the
/// production path uses `StorageClient` while unit tests stub it.
pub async fn check_dim_consistency_with_lookup<F, Fut>(
    lookup: F,
    active_dim: usize,
    bootstrap_reembed: bool,
) -> Result<(), EmbeddingError>
where
    F: FnOnce(&'static str) -> Fut,
    Fut: std::future::Future<Output = Result<Option<u64>, EmbeddingError>>,
{
    let stored = lookup(COLLECTION_PROJECTS).await?;

    if bootstrap_reembed {
        warn!("--bootstrap-reembed is active; dim-mismatch guard suppressed for this startup.");
        if let Some(stored_dim) = stored {
            if stored_dim as usize == active_dim {
                warn!(
                    "--bootstrap-reembed was set but dim check would have passed; \
                     remove the flag after migration is complete."
                );
            }
        }
        return Ok(());
    }

    let Some(stored_dim_u64) = stored else {
        // First-run path: collection does not yet exist; it will be created at
        // the active dim by the queue processor.
        return Ok(());
    };
    let stored_dim = stored_dim_u64 as usize;
    if stored_dim == active_dim {
        return Ok(());
    }

    error!(
        active_dim,
        stored_dim,
        "FATAL: active embedding provider outputs {}-dim vectors but the '{}' \
         Qdrant collection was created with {}-dim vectors",
        active_dim,
        COLLECTION_PROJECTS,
        stored_dim
    );
    Err(EmbeddingError::DimensionMismatch {
        actual_dim: active_dim,
        stored_dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lookup_returns(
        value: Result<Option<u64>, EmbeddingError>,
    ) -> impl FnOnce(
        &'static str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Option<u64>, EmbeddingError>>>,
    > {
        move |name: &'static str| {
            assert_eq!(name, COLLECTION_PROJECTS);
            Box::pin(async move { value })
        }
    }

    #[tokio::test]
    async fn test_dim_mismatch_guard_aborts_startup() {
        let result =
            check_dim_consistency_with_lookup(lookup_returns(Ok(Some(384))), 1536, false).await;
        match result {
            Err(EmbeddingError::DimensionMismatch {
                actual_dim,
                stored_dim,
            }) => {
                assert_eq!(actual_dim, 1536);
                assert_eq!(stored_dim, 384);
            }
            other => panic!("expected DimensionMismatch, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_dim_mismatch_guard_passes_on_match() {
        let result =
            check_dim_consistency_with_lookup(lookup_returns(Ok(Some(1536))), 1536, false).await;
        assert!(result.is_ok(), "expected Ok, got {:?}", result);
    }

    #[tokio::test]
    async fn test_dim_mismatch_guard_skips_when_no_collection() {
        let result = check_dim_consistency_with_lookup(lookup_returns(Ok(None)), 1536, false).await;
        assert!(
            result.is_ok(),
            "expected Ok on first-run path, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_dim_mismatch_guard_bypassed_with_bootstrap_flag() {
        let result =
            check_dim_consistency_with_lookup(lookup_returns(Ok(Some(384))), 1536, true).await;
        assert!(
            result.is_ok(),
            "bootstrap_reembed must suppress the guard, got {:?}",
            result
        );
    }

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_dim_mismatch_guard_bootstrap_warns_when_would_pass() {
        let result =
            check_dim_consistency_with_lookup(lookup_returns(Ok(Some(1536))), 1536, true).await;
        assert!(result.is_ok(), "expected Ok, got {:?}", result);
        assert!(
            logs_contain("would have passed"),
            "expected redundant-flag WARN to be emitted"
        );
    }
}

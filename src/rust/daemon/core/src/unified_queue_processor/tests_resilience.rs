//! Integration tests for queue resilience features (tasks 121-146).

#[cfg(test)]
mod tests {
    use crate::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    use crate::queue_operations::QueueManager;
    use crate::schema_version::SchemaManager;
    use crate::unified_queue_processor::config::UnifiedProcessorConfig;
    use crate::unified_queue_processor::error::UnifiedProcessorError;
    use crate::unified_queue_processor::UnifiedQueueProcessor;
    use sqlx::sqlite::SqlitePoolOptions;
    use sqlx::{Executor, Row, SqlitePool};

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(2)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.unwrap();
        pool
    }

    async fn insert_failed_item(pool: &SqlitePool, queue_id: &str, error_msg: &str) {
        sqlx::query(
            "INSERT INTO unified_queue \
                (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
                 branch, status, error_message, retry_count, payload_json) \
             VALUES (?1, ?2, 'doc', 'add', 'test-tenant', 'projects', \
                     'main', 'failed', ?3, 3, '{}')",
        )
        .bind(queue_id)
        .bind(format!("key-{}", queue_id))
        .bind(error_msg)
        .execute(pool)
        .await
        .unwrap();
    }

    // ── Error classification (7 categories) ──────────────────────

    #[test]
    fn test_all_7_error_categories() {
        let cases = vec![
            (
                UnifiedProcessorError::FileNotFound("/gone".into()),
                "permanent_gone",
            ),
            (
                UnifiedProcessorError::InvalidPayload("bad".into()),
                "permanent_data",
            ),
            (
                UnifiedProcessorError::Storage("conn refused".into()),
                "transient_infrastructure",
            ),
            (
                UnifiedProcessorError::Embedding("OOM".into()),
                "transient_resource",
            ),
            (
                UnifiedProcessorError::EmbeddingUnavailable("backoff".into()),
                "subsystem_unavailable",
            ),
            (
                UnifiedProcessorError::Embedding("rate limit exceeded".into()),
                "rate_limit",
            ),
            (
                UnifiedProcessorError::QueueOperation("database locked".into()),
                "transient_infrastructure",
            ),
        ];

        for (error, expected) in cases {
            let category = UnifiedQueueProcessor::classify_error(&error);
            assert_eq!(
                category, expected,
                "Error {:?} should classify as {}",
                error, expected
            );
        }
    }

    #[test]
    fn test_permanent_categories_are_permanent() {
        assert!(UnifiedQueueProcessor::is_permanent_category(
            "permanent_data"
        ));
        assert!(UnifiedQueueProcessor::is_permanent_category(
            "permanent_gone"
        ));
        assert!(!UnifiedQueueProcessor::is_permanent_category(
            "transient_infrastructure"
        ));
        assert!(!UnifiedQueueProcessor::is_permanent_category("rate_limit"));
        assert!(!UnifiedQueueProcessor::is_permanent_category(
            "subsystem_unavailable"
        ));
    }

    // ── DLQ full lifecycle ───────────────────────────────────────

    #[tokio::test]
    async fn test_dlq_full_lifecycle() {
        let pool = setup_pool().await;
        let qm = QueueManager::new(pool.clone());

        insert_failed_item(&pool, "lifecycle-1", "[permanent_data] bad json").await;

        let dlq_id = qm.move_to_dlq("lifecycle-1").await.unwrap();

        let in_queue: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'lifecycle-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(in_queue, 0);

        let entry = qm.get_dlq_entry(&dlq_id).await.unwrap();
        assert_eq!(entry.error_category, "permanent_data");

        let replay_result = qm.replay_from_dlq(&dlq_id, false).await;
        assert!(replay_result.is_err());

        let new_id = qm.replay_from_dlq(&dlq_id, true).await.unwrap();

        let status: String =
            sqlx::query_scalar("SELECT status FROM unified_queue WHERE queue_id = ?1")
                .bind(&new_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(status, "pending");

        let in_dlq: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM dead_letter_queue WHERE dlq_id = ?1")
                .bind(&dlq_id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(in_dlq, 0);
    }

    #[tokio::test]
    async fn test_dlq_purge_respects_retention() {
        let pool = setup_pool().await;
        let qm = QueueManager::new(pool.clone());

        insert_failed_item(&pool, "old-1", "[permanent_data] old").await;
        insert_failed_item(&pool, "new-1", "[permanent_data] new").await;

        qm.move_to_dlq("old-1").await.unwrap();
        qm.move_to_dlq("new-1").await.unwrap();

        pool.execute(
            "UPDATE dead_letter_queue SET moved_to_dlq_at = '2020-01-01T00:00:00.000Z' \
             WHERE original_queue_id = 'old-1'",
        )
        .await
        .unwrap();

        let (deleted, _) = qm.purge_dlq(30, 500).await.unwrap();
        assert_eq!(deleted, 1);

        let remaining: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM dead_letter_queue")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(remaining, 1);
    }

    // ── Circuit breaker state machine ────────────────────────────

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let mut cb = CircuitBreaker::new(
            "test-cb",
            CircuitBreakerConfig {
                failure_threshold: 3,
                failure_window: 60,
                recovery_timeout: 300,
                success_threshold: 2,
            },
        );

        assert!(cb.is_closed());
        assert_eq!(cb.state_str(), "closed");

        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_closed());

        let opened = cb.record_failure();
        assert!(opened);
        assert!(!cb.is_closed());
        assert_eq!(cb.state_str(), "open");

        let (can_proceed, reason) = cb.check();
        assert!(!can_proceed);
        assert_eq!(reason, "circuit_open");
    }

    #[test]
    fn test_circuit_breaker_success_in_closed_is_noop() {
        let mut cb = CircuitBreaker::new(
            "test-success",
            CircuitBreakerConfig {
                failure_threshold: 3,
                failure_window: 60,
                recovery_timeout: 300,
                success_threshold: 2,
            },
        );

        cb.record_failure();
        cb.record_success();
        assert!(cb.is_closed());
    }

    // ── Config defaults ──────────────────────────────────────────

    #[test]
    fn test_config_dlq_defaults() {
        let config = UnifiedProcessorConfig::default();
        assert_eq!(config.dlq_retention_days, 30);
        assert_eq!(config.dlq_purge_batch_size, 500);
    }

    #[test]
    fn test_config_sqlite_breaker_defaults() {
        let config = UnifiedProcessorConfig::default();
        assert_eq!(config.sqlite_failure_threshold, 3);
        assert_eq!(config.sqlite_failure_window_secs, 30);
        assert_eq!(config.sqlite_probe_interval_secs, 5);
        assert_eq!(config.recovery_ramp_cycles, 5);
    }

    // ── DLQ listing and filtering ────────────────────────────────

    #[tokio::test]
    async fn test_dlq_list_with_category_filter() {
        let pool = setup_pool().await;
        let qm = QueueManager::new(pool.clone());

        insert_failed_item(&pool, "cat-1", "[permanent_data] bad").await;
        insert_failed_item(&pool, "cat-2", "[permanent_gone] gone").await;

        qm.move_to_dlq("cat-1").await.unwrap();
        qm.move_to_dlq("cat-2").await.unwrap();

        let (all, total) = qm.list_dlq(None, None, 50, 0).await.unwrap();
        assert_eq!(total, 2);
        assert_eq!(all.len(), 2);

        let (data_only, data_total) = qm
            .list_dlq(None, Some("permanent_data"), 50, 0)
            .await
            .unwrap();
        assert_eq!(data_total, 1);
        assert_eq!(data_only[0].error_category, "permanent_data");
    }

    // ── Schema v42 DLQ table exists ──────────────────────────────

    #[tokio::test]
    async fn test_dlq_table_created_by_migration() {
        let pool = setup_pool().await;
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='dead_letter_queue')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(exists);
    }

    #[tokio::test]
    async fn test_schema_version_is_42() {
        let pool = setup_pool().await;
        let version: i32 = sqlx::query_scalar("SELECT MAX(version) FROM schema_version")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(version, 42);
    }
}

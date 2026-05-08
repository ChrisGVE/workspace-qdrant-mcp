//! Full re-embed pipeline executed under `AdminWriteService.TriggerReembed`.
//!
//! Implements PRD §6.6 (embedding-providers): pause → drain → flush →
//! recreate canonical Qdrant collections at the configured `output_dim` →
//! enqueue new ingestion items → resume queue. The handler stays
//! enqueue-only on the queue side: the `'reembed'` collection items are
//! recorded in `unified_queue` for traceability, and the actual file/rule
//! /scratchpad re-ingestion goes through normal `add` items so existing
//! queue strategies pick them up.

mod context;
mod enqueue;
mod recreator;

pub use context::{ReembedContext, CANONICAL_COLLECTIONS};
pub use recreator::{CollectionRecreator, StorageClientRecreator};

use enqueue::{enqueue_folder_scans, enqueue_rules_mirror, enqueue_scratchpad_mirror};
use recreator::collection_reembed_idempotency_key;

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use tonic::Status;
use tracing::info;
use uuid::Uuid;

use crate::proto::TriggerReembedResponse;

/// Execute the full reembed flow.
///
/// Returns the populated [`TriggerReembedResponse`] on success or a
/// pre-mapped `tonic::Status` describing the failure mode (typically
/// `failed_precondition` for dim mismatch / drain timeout).
pub async fn execute_reembed<R: CollectionRecreator + ?Sized>(
    ctx: &ReembedContext,
    recreator: &R,
    drain_timeout: Duration,
    poll_interval: Duration,
) -> Result<TriggerReembedResponse, Status> {
    // ── 1. Pre-flight dim check ──────────────────────────────────────────
    let cfg_dim = ctx.settings.output_dim;
    let provider_dim = ctx.provider.output_dim();
    if cfg_dim != provider_dim {
        return Err(Status::failed_precondition(format!(
            "provider output_dim mismatch: settings.output_dim={} but provider.output_dim()={}",
            cfg_dim, provider_dim
        )));
    }

    // ── 2. Pause queue workers via shared flag ───────────────────────────
    ctx.pause_flag.store(true, Ordering::SeqCst);
    info!("reembed: pause flag set; awaiting queue quiescence");

    // ── 3. Drain to quiescence (60s hard cap, no recreation on timeout) ──
    let drain_started = Instant::now();
    loop {
        let in_flight: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue \
             WHERE status = 'in_progress' \
             AND lease_until IS NOT NULL \
             AND lease_until > strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        )
        .fetch_one(&ctx.pool)
        .await
        .map_err(|e| {
            ctx.pause_flag.store(false, Ordering::SeqCst);
            Status::internal(format!("drain query failed: {e}"))
        })?;

        if in_flight == 0 {
            break;
        }

        if drain_started.elapsed() >= drain_timeout {
            ctx.pause_flag.store(false, Ordering::SeqCst);
            return Err(Status::failed_precondition(format!(
                "drain-to-quiescence timeout: {} items still in_progress after {}s; pause flag released",
                in_flight,
                drain_timeout.as_secs()
            )));
        }

        tokio::time::sleep(poll_interval).await;
    }

    // ── 4. Flush stale pending items so the reembed enqueue keys do not
    //      collide with prior pending duplicates ──────────────────────────
    let stale_deleted = sqlx::query(
        "DELETE FROM unified_queue \
         WHERE status = 'pending' \
         AND collection IN ('projects','libraries','rules','scratchpad')",
    )
    .execute(&ctx.pool)
    .await
    .map_err(|e| Status::internal(format!("flush stale pending failed: {e}")))?;
    info!(
        rows = stale_deleted.rows_affected(),
        "reembed: flushed stale pending"
    );

    // ── 5. Clear vector-derived SQLite state in one transaction ──────────
    let mut tx = ctx
        .pool
        .begin()
        .await
        .map_err(|e| Status::internal(format!("clear-state tx begin failed: {e}")))?;
    sqlx::query("DELETE FROM tag_hierarchy_edges")
        .execute(&mut *tx)
        .await
        .map_err(|e| Status::internal(format!("clear tag_hierarchy_edges: {e}")))?;
    sqlx::query("DELETE FROM canonical_tags")
        .execute(&mut *tx)
        .await
        .map_err(|e| Status::internal(format!("clear canonical_tags: {e}")))?;
    tx.commit()
        .await
        .map_err(|e| Status::internal(format!("clear-state tx commit failed: {e}")))?;

    // ── 6. Recreate the four canonical collections at settings.output_dim
    //      while workers are still paused ────────────────────────────────
    let recreate_dim = cfg_dim as u64;
    for name in CANONICAL_COLLECTIONS {
        recreator.recreate(name, recreate_dim).await?;
    }

    // ── 7. Enqueue 4 collection-reembed traceability items ───────────────
    let now = wqm_common::timestamps::now_utc();
    for collection in CANONICAL_COLLECTIONS {
        let queue_id = Uuid::new_v4().to_string();
        let idem_key = collection_reembed_idempotency_key(collection);
        sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'collection', 'reembed', '_system', ?3, 'pending', '{}', ?4, ?5)",
        )
        .bind(&queue_id)
        .bind(&idem_key)
        .bind(collection)
        .bind(&now)
        .bind(&now)
        .execute(&ctx.pool)
        .await
        .map_err(|e| Status::internal(format!("enqueue reembed/{collection}: {e}")))?;
    }

    // ── 8. Re-enqueue from watch_folders, rules_mirror, scratchpad_mirror
    let files_enqueued = enqueue_folder_scans(&ctx.pool, &now)
        .await
        .map_err(|e| Status::internal(format!("re-enqueue folder scans: {e}")))?;
    let rules_enqueued = enqueue_rules_mirror(&ctx.pool, &now)
        .await
        .map_err(|e| Status::internal(format!("re-enqueue rules_mirror: {e}")))?;
    let scratchpad_enqueued = enqueue_scratchpad_mirror(&ctx.pool, &now)
        .await
        .map_err(|e| Status::internal(format!("re-enqueue scratchpad_mirror: {e}")))?;

    // ── 9. Resume queue workers ──────────────────────────────────────────
    ctx.pause_flag.store(false, Ordering::SeqCst);
    info!(
        files = files_enqueued,
        rules = rules_enqueued,
        scratchpad = scratchpad_enqueued,
        "reembed: complete; pause flag cleared"
    );

    Ok(TriggerReembedResponse {
        files_enqueued,
        rules_enqueued,
        scratchpad_enqueued,
        message: format!(
            "reembed complete at output_dim={cfg_dim}: {files_enqueued} files, \
             {rules_enqueued} rules, {scratchpad_enqueued} scratchpad items re-enqueued"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use sqlx::SqlitePool;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Mutex};
    use workspace_qdrant_core::embedding::provider::DenseProvider;
    use workspace_qdrant_core::embedding::{DenseEmbedding, EmbeddingError};
    use workspace_qdrant_core::storage::StorageClient;

    /// In-memory mock recreator: records (name, dim) per call.
    #[derive(Default)]
    struct MockRecreator {
        calls: Mutex<Vec<(String, u64)>>,
        fail_with: Mutex<Option<Status>>,
    }

    #[async_trait]
    impl CollectionRecreator for MockRecreator {
        async fn recreate(&self, name: &str, dim: u64) -> Result<(), Status> {
            if let Some(s) = self.fail_with.lock().unwrap().take() {
                return Err(s);
            }
            self.calls.lock().unwrap().push((name.to_string(), dim));
            Ok(())
        }
    }

    /// Stub provider with a configurable output_dim. Probe/embed are not
    /// exercised by the reembed flow.
    #[derive(Debug)]
    struct StubProvider {
        dim: usize,
    }

    #[async_trait]
    impl DenseProvider for StubProvider {
        async fn embed(&self, _texts: &[&str]) -> Result<Vec<DenseEmbedding>, EmbeddingError> {
            Ok(Vec::new())
        }
        fn output_dim(&self) -> usize {
            self.dim
        }
        fn provider_label(&self) -> &str {
            "stub"
        }
        fn metrics_label(&self) -> &'static str {
            "fastembed"
        }
        async fn probe(&self) -> Result<(), EmbeddingError> {
            Ok(())
        }
    }

    async fn fresh_pool() -> SqlitePool {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();

        // unified_queue (subset of canonical schema sufficient for these tests)
        sqlx::query(
            "CREATE TABLE unified_queue (
                queue_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                item_type TEXT NOT NULL,
                op TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                payload_json TEXT NOT NULL DEFAULT '{}',
                lease_until TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE watch_folders (
                watch_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                collection TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE rules_mirror (
                rule_id TEXT PRIMARY KEY,
                rule_text TEXT NOT NULL,
                scope TEXT,
                tenant_id TEXT
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE scratchpad_mirror (
                scratchpad_id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT NOT NULL,
                tags TEXT,
                tenant_id TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE canonical_tags (
                canonical_id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "CREATE TABLE tag_hierarchy_edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_tag_id INTEGER NOT NULL,
                child_tag_id INTEGER NOT NULL,
                tenant_id TEXT NOT NULL
            )",
        )
        .execute(&pool)
        .await
        .unwrap();

        pool
    }

    fn ctx_with(pool: SqlitePool, dim: usize) -> ReembedContext {
        let mut settings = workspace_qdrant_core::config::EmbeddingSettings::default();
        settings.output_dim = dim;
        ReembedContext {
            settings: Arc::new(settings),
            provider: Arc::new(StubProvider { dim }),
            // Storage client unused in these tests because we go through the
            // mock recreator. Use a default-constructed dummy client; never
            // dereferenced.
            storage_client: Arc::new(StorageClient::with_config(
                workspace_qdrant_core::storage::StorageConfig::default(),
            )),
            pool,
            pause_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    #[tokio::test]
    async fn fails_when_probe_dim_disagrees_with_settings() {
        let pool = fresh_pool().await;
        let mut ctx = ctx_with(pool, 1536);
        // Mismatch: settings says 1536, provider reports 384.
        let mut new_settings = (*ctx.settings).clone();
        new_settings.output_dim = 1536;
        ctx.settings = Arc::new(new_settings);
        ctx.provider = Arc::new(StubProvider { dim: 384 });

        let recreator = MockRecreator::default();
        let err = execute_reembed(
            &ctx,
            &recreator,
            Duration::from_secs(1),
            Duration::from_millis(10),
        )
        .await
        .expect_err("must reject dim mismatch");
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        assert!(err.message().contains("output_dim mismatch"));
        assert!(recreator.calls.lock().unwrap().is_empty());
        // pause_flag must NOT be left set
        assert!(!ctx.pause_flag.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn proceeds_when_no_in_flight_items() {
        let pool = fresh_pool().await;
        let ctx = ctx_with(pool, 384);
        let recreator = MockRecreator::default();
        let resp = execute_reembed(
            &ctx,
            &recreator,
            Duration::from_secs(2),
            Duration::from_millis(10),
        )
        .await
        .expect("reembed must succeed when no in-flight items");
        assert!(resp.message.contains("reembed complete"));
        // pause_flag returned to false on success
        assert!(!ctx.pause_flag.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn fails_when_quiescence_timeout_exceeded() {
        let pool = fresh_pool().await;
        // Insert an in_progress item with a future lease.
        sqlx::query(
            "INSERT INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, lease_until, created_at, updated_at) \
             VALUES ('q1','k1','file','add','t','projects','in_progress', \
                     strftime('%Y-%m-%dT%H:%M:%fZ','now','+10 seconds'), \
                     '2024-01-01T00:00:00.000Z','2024-01-01T00:00:00.000Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let ctx = ctx_with(pool, 384);
        let recreator = MockRecreator::default();
        let err = execute_reembed(
            &ctx,
            &recreator,
            Duration::from_millis(150),
            Duration::from_millis(20),
        )
        .await
        .expect_err("must time out");
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        assert!(err.message().contains("drain-to-quiescence timeout"));
        // pause_flag released on timeout
        assert!(!ctx.pause_flag.load(Ordering::SeqCst));
        // No collection recreation on timeout
        assert!(recreator.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn flushes_stale_pending_items() {
        let pool = fresh_pool().await;
        // Pre-existing pending row in 'projects' that should be flushed.
        sqlx::query(
            "INSERT INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, created_at, updated_at) \
             VALUES ('stale','stale-key','file','add','t','projects','pending', \
                     '2024-01-01T00:00:00.000Z','2024-01-01T00:00:00.000Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let ctx = ctx_with(pool.clone(), 384);
        let recreator = MockRecreator::default();
        execute_reembed(
            &ctx,
            &recreator,
            Duration::from_secs(1),
            Duration::from_millis(10),
        )
        .await
        .unwrap();

        let stale_remaining: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM unified_queue WHERE queue_id = 'stale'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(stale_remaining, 0, "stale pending row must be flushed");
    }

    #[tokio::test]
    async fn enqueues_collection_reset_items() {
        let pool = fresh_pool().await;
        let ctx = ctx_with(pool.clone(), 384);
        let recreator = MockRecreator::default();
        execute_reembed(
            &ctx,
            &recreator,
            Duration::from_secs(1),
            Duration::from_millis(10),
        )
        .await
        .unwrap();

        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue \
             WHERE item_type='collection' AND op='reembed' AND tenant_id='_system'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(
            count, 4,
            "must enqueue one reembed item per canonical collection"
        );

        for c in CANONICAL_COLLECTIONS {
            let exists: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM unified_queue \
                 WHERE item_type='collection' AND op='reembed' AND collection = ?1",
            )
            .bind(*c)
            .fetch_one(&pool)
            .await
            .unwrap();
            assert_eq!(exists, 1, "missing reembed enqueue for {c}");
        }
    }

    #[tokio::test]
    async fn clears_canonical_tags() {
        let pool = fresh_pool().await;
        sqlx::query(
            "INSERT INTO canonical_tags (canonical_name, tenant_id, collection) \
             VALUES ('foo','t','projects'), ('bar','t','projects')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO tag_hierarchy_edges (parent_tag_id, child_tag_id, tenant_id) \
             VALUES (1,2,'t')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let ctx = ctx_with(pool.clone(), 384);
        let recreator = MockRecreator::default();
        execute_reembed(
            &ctx,
            &recreator,
            Duration::from_secs(1),
            Duration::from_millis(10),
        )
        .await
        .unwrap();

        let canon_remaining: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM canonical_tags")
            .fetch_one(&pool)
            .await
            .unwrap();
        let edges_remaining: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM tag_hierarchy_edges")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(canon_remaining, 0);
        assert_eq!(edges_remaining, 0);
    }

    #[tokio::test]
    async fn repopulates_rules_and_scratchpad() {
        let pool = fresh_pool().await;
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, enabled) \
             VALUES ('w1','/tmp/p1','projects','t1',1), \
                    ('w2','/tmp/lib','libraries','tlib',1), \
                    ('w3','/tmp/disabled','projects','t1',0)",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO rules_mirror (rule_id, rule_text, scope, tenant_id) \
             VALUES ('r1','do not foo','global','_system'), \
                    ('r2','do not bar',NULL,NULL)",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO scratchpad_mirror (scratchpad_id, title, content, tags, tenant_id) \
             VALUES ('s1','t1','content one','[]','tA'), \
                    ('s2',NULL,'content two',NULL,'tB')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let ctx = ctx_with(pool.clone(), 384);
        let recreator = MockRecreator::default();
        let resp = execute_reembed(
            &ctx,
            &recreator,
            Duration::from_secs(1),
            Duration::from_millis(10),
        )
        .await
        .unwrap();

        assert_eq!(
            resp.files_enqueued, 2,
            "only enabled watch_folders must scan"
        );
        assert_eq!(resp.rules_enqueued, 2);
        assert_eq!(resp.scratchpad_enqueued, 2);

        let folder_scans: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue WHERE item_type='folder' AND op='scan'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let rule_adds: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue \
             WHERE item_type='text' AND op='add' AND collection='rules'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let pad_adds: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue \
             WHERE item_type='text' AND op='add' AND collection='scratchpad'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(folder_scans, 2);
        assert_eq!(rule_adds, 2);
        assert_eq!(pad_adds, 2);
    }

    #[tokio::test]
    async fn uses_settings_output_dim_for_recreation() {
        let pool = fresh_pool().await;
        // Configured 1536; provider also 1536 so pre-flight passes. Verify
        // the recreator is invoked at exactly 1536 — i.e. settings.output_dim
        // is the authoritative dim for recreation.
        let mut settings = workspace_qdrant_core::config::EmbeddingSettings::default();
        settings.output_dim = 1536;
        let ctx = ReembedContext {
            settings: Arc::new(settings),
            provider: Arc::new(StubProvider { dim: 1536 }),
            storage_client: Arc::new(StorageClient::with_config(
                workspace_qdrant_core::storage::StorageConfig::default(),
            )),
            pool,
            pause_flag: Arc::new(AtomicBool::new(false)),
        };
        let recreator = MockRecreator::default();
        execute_reembed(
            &ctx,
            &recreator,
            Duration::from_secs(1),
            Duration::from_millis(10),
        )
        .await
        .unwrap();

        let calls = recreator.calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 4);
        for (name, dim) in &calls {
            assert!(CANONICAL_COLLECTIONS.contains(&name.as_str()));
            assert_eq!(*dim, 1536u64, "recreation must use settings.output_dim");
        }
    }

    #[test]
    fn idempotency_key_is_deterministic_and_32_chars() {
        let k1 = collection_reembed_idempotency_key("projects");
        let k2 = collection_reembed_idempotency_key("projects");
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 32);
        assert_ne!(k1, collection_reembed_idempotency_key("libraries"));
    }

    // Suppress "unused" warnings on test-only counter when not needed.
    const _ASSERT_AT: AtomicUsize = AtomicUsize::new(0);
}

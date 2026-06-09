use super::pipeline::flush_and_clear_state;
use super::recreator::collection_reembed_idempotency_key;
use super::*;
use async_trait::async_trait;
use sqlx::SqlitePool;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tonic::Status;
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

    sqlx::query(
        "CREATE TABLE tracked_files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                watch_folder_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                branch TEXT,
                file_hash TEXT NOT NULL,
                collection TEXT NOT NULL DEFAULT 'projects'
            )",
    )
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        "CREATE TABLE qdrant_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                point_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL
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

/// Regression for the "reembed completes but does not re-ingest" bug: the flush
/// must delete ALL canonical-collection queue rows (including `done`), not just
/// `pending`. Leaving `done` rows behind dedups the re-enqueued File/Add via
/// `INSERT OR IGNORE` on idempotency_key, so files were never re-ingested.
#[tokio::test]
async fn flush_clears_done_rows_and_spares_non_canonical() {
    let pool = fresh_pool().await;
    for (qid, ik, coll, status) in [
        ("q1", "k1", "projects", "done"),
        ("q2", "k2", "projects", "pending"),
        ("q3", "k3", "rules", "done"),
        ("q4", "k4", "_internal", "done"), // non-canonical: must survive
    ] {
        sqlx::query(
            "INSERT INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, created_at, updated_at) \
             VALUES (?1, ?2, 'File', 'Add', 't1', ?3, ?4, 'now', 'now')",
        )
        .bind(qid)
        .bind(ik)
        .bind(coll)
        .bind(status)
        .execute(&pool)
        .await
        .unwrap();
    }

    // Seed per-file hash tracking: a canonical-collection file (must be
    // cleared so re-ingestion doesn't skip on hash-match) with a dependent
    // qdrant_chunks row, plus a non-canonical file that must survive.
    sqlx::query(
        "INSERT INTO tracked_files (file_id, watch_folder_id, relative_path, branch, file_hash, collection) \
         VALUES (1, 'w1', 'a.rs', 'main', 'h1', 'projects'), \
                (2, 'w2', 'b.txt', 'main', 'h2', '_internal')",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO qdrant_chunks (chunk_id, file_id, point_id, chunk_index) \
         VALUES (1, 1, 'p1', 0), (2, 2, 'p2', 0)",
    )
    .execute(&pool)
    .await
    .unwrap();

    let ctx = ctx_with(pool.clone(), 384);
    let deleted = flush_and_clear_state(&ctx).await.unwrap();

    // 2 projects (done+pending) + 1 rules (done) removed.
    assert_eq!(deleted, 3, "must delete done rows too, not just pending");
    let surviving: Vec<String> =
        sqlx::query_scalar("SELECT collection FROM unified_queue ORDER BY collection")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(surviving, vec!["_internal".to_string()]);

    // Canonical file's hash tracking + its chunks are cleared; the
    // non-canonical file's tracking survives.
    let tracked: Vec<String> =
        sqlx::query_scalar("SELECT collection FROM tracked_files ORDER BY collection")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(tracked, vec!["_internal".to_string()]);
    let chunk_files: Vec<i64> =
        sqlx::query_scalar("SELECT file_id FROM qdrant_chunks ORDER BY file_id")
            .fetch_all(&pool)
            .await
            .unwrap();
    assert_eq!(chunk_files, vec![2], "canonical file's chunks must be cleared");
}

/// Regression for the "reembed completes but re-ingests nothing" bug: the
/// re-enqueued folder scan must carry `folder_path: null` (scan the watch-folder
/// root), NOT the absolute root path. The folder-scan strategy joins a non-null
/// `folder_path` onto the root it looks up, so passing the absolute root yielded
/// a doubled path (`/repo/x/repo/x`) that "is not a directory" — every reembed
/// scan enqueued zero files.
#[tokio::test]
async fn folder_scan_enqueue_uses_null_path_for_root() {
    let pool = fresh_pool().await;
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, enabled) \
         VALUES ('w1', '/home/u/repos/DOC-V2', 'projects', 't1', 1)",
    )
    .execute(&pool)
    .await
    .unwrap();

    let n = super::enqueue::enqueue_folder_scans(&pool, "2026-01-01T00:00:00Z")
        .await
        .unwrap();
    assert_eq!(n, 1);

    let payload: String =
        sqlx::query_scalar("SELECT payload_json FROM unified_queue WHERE item_type = 'folder'")
            .fetch_one(&pool)
            .await
            .unwrap();
    let v: serde_json::Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(
        v.get("folder_path"),
        Some(&serde_json::Value::Null),
        "folder_path must be null (root scan), not the absolute path; got payload {payload}"
    );
    assert!(
        !payload.contains("/home/u/repos/DOC-V2"),
        "payload must not embed the absolute root path: {payload}"
    );
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

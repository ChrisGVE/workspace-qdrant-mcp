use serial_test::serial;

use super::super::*;
use super::create_test_pool;

// ============================================================================
// Helper: build a pool at v34 state with a simulated "legacy" unified_queue
// that does not yet include 'reembed' in its op CHECK constraint.
// ============================================================================

/// Run migrations 1–34 on a fresh in-memory pool, then replace `unified_queue`
/// with a legacy DDL that lacks `'reembed'` in the op CHECK.  This simulates
/// a database that predates v35 and exercises the full migration path.
///
/// On fresh test databases `CREATE_UNIFIED_QUEUE_SQL` already contains
/// `'reembed'` (v35 updated that constant), so v35 would normally skip the
/// rebuild.  By swapping in the legacy DDL we force the real migration path
/// through the rename → recreate → copy → drop sequence.
async fn create_test_pool_v34() -> (SqlitePool, SchemaManager) {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager.initialize().await.expect("init");
    for v in 1..=34 {
        manager
            .run_migration(v)
            .await
            .unwrap_or_else(|e| panic!("v{} migration failed: {}", v, e));
        manager
            .record_migration(v)
            .await
            .unwrap_or_else(|e| panic!("record v{} failed: {}", v, e));
    }

    // Confirm watch_folders has no active_provider yet.
    let has_active_provider: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') \
         WHERE name = 'active_provider'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        !has_active_provider,
        "watch_folders must NOT have active_provider before v35"
    );

    // Swap unified_queue for a legacy version that lacks 'reembed'.
    // This is the state any database had before v35 was written.
    simulate_pre_v35_unified_queue(&pool).await;

    let queue_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        !queue_sql.contains("'reembed'"),
        "legacy unified_queue must NOT contain 'reembed' — simulation failed"
    );

    let version = manager.get_current_version().await.unwrap();
    assert_eq!(version, Some(34), "pool must be at v34");

    (pool, manager)
}

/// Replace `unified_queue` with a legacy DDL that omits `'reembed'` from the
/// `op` CHECK constraint, simulating the pre-v35 schema state.
///
/// Note: `max_retries` was added in v11 and dropped in v27, so at v34 the
/// `unified_queue` table does not have that column.  The legacy DDL mirrors
/// the v34 column set exactly but removes `'reembed'` from the `op` CHECK.
async fn simulate_pre_v35_unified_queue(pool: &SqlitePool) {
    // Legacy op CHECK: no 'reembed' (v35 adds it).
    // Column set matches v34 state (no max_retries — dropped in v27).
    let legacy_ddl = r#"
        CREATE TABLE unified_queue_legacy (
            queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
            item_type TEXT NOT NULL CHECK (item_type IN (
                'text', 'file', 'url', 'website', 'doc', 'folder', 'tenant', 'collection'
            )),
            op TEXT NOT NULL CHECK (op IN ('add', 'update', 'delete', 'scan', 'rename', 'uplift', 'reset')),
            tenant_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
                'pending', 'in_progress', 'done', 'failed'
            )),
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            lease_until TEXT,
            worker_id TEXT,
            idempotency_key TEXT NOT NULL UNIQUE,
            payload_json TEXT NOT NULL DEFAULT '{}',
            retry_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            last_error_at TEXT,
            branch TEXT DEFAULT 'main',
            metadata TEXT DEFAULT '{}',
            file_path TEXT UNIQUE,
            qdrant_status TEXT DEFAULT 'pending' CHECK (qdrant_status IN ('pending', 'in_progress', 'done', 'failed')),
            search_status TEXT DEFAULT 'pending' CHECK (search_status IN ('pending', 'in_progress', 'done', 'failed')),
            decision_json TEXT
        )
    "#;

    sqlx::query("PRAGMA foreign_keys = OFF")
        .execute(pool)
        .await
        .unwrap();

    // Create legacy table, copy rows (using * — both tables share the same
    // column set at this point since we just created unified_queue_legacy with
    // an identical schema minus 'reembed' in the CHECK), then swap.
    sqlx::query(legacy_ddl).execute(pool).await.unwrap();
    sqlx::query(
        "INSERT INTO unified_queue_legacy \
         SELECT queue_id, item_type, op, tenant_id, collection, status, \
                created_at, updated_at, lease_until, worker_id, idempotency_key, \
                payload_json, retry_count, error_message, last_error_at, \
                branch, metadata, file_path, qdrant_status, search_status, decision_json \
         FROM unified_queue",
    )
    .execute(pool)
    .await
    .unwrap();
    sqlx::query("DROP TABLE unified_queue")
        .execute(pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE unified_queue_legacy RENAME TO unified_queue")
        .execute(pool)
        .await
        .unwrap();

    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(pool)
        .await
        .unwrap();
}

#[tokio::test]
#[serial]
async fn test_run_migrations_from_scratch() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations");

    let version = manager
        .get_current_version()
        .await
        .expect("Failed to get version");
    assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));

    let watch_folders_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='watch_folders')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        watch_folders_exists,
        "watch_folders table should exist after migration"
    );

    let unified_queue_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='unified_queue')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        unified_queue_exists,
        "unified_queue table should exist after migration"
    );

    let tracked_files_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        tracked_files_exists,
        "tracked_files table should exist after migration v2"
    );

    let qdrant_chunks_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='qdrant_chunks')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        qdrant_chunks_exists,
        "qdrant_chunks table should exist after migration v2"
    );
}

#[tokio::test]
#[serial]
async fn test_run_migrations_idempotent() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool);

    manager
        .run_migrations()
        .await
        .expect("First migration failed");
    manager
        .run_migrations()
        .await
        .expect("Second migration should be idempotent");
}

#[tokio::test]
#[serial]
async fn test_incremental_migration_v1_to_current() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager.initialize().await.expect("Failed to initialize");
    manager.run_migration(1).await.expect("Failed to run v1");
    manager
        .record_migration(1)
        .await
        .expect("Failed to record v1");

    let tracked_exists_before: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        !tracked_exists_before,
        "tracked_files should NOT exist before v2 migration"
    );

    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations from v1");

    let tracked_exists_after: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        tracked_exists_after,
        "tracked_files should exist after v2 migration"
    );

    let has_reconcile: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'needs_reconcile'"
    )
    .fetch_one(&pool).await.unwrap();
    assert!(
        has_reconcile,
        "needs_reconcile column should exist after v3 migration"
    );

    let version = manager
        .get_current_version()
        .await
        .expect("Failed to get version");
    assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));
}

#[tokio::test]
#[serial]
async fn test_incremental_migration_v2_to_v3() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager.initialize().await.expect("Failed to initialize");
    manager.run_migration(1).await.expect("Failed to run v1");
    manager
        .record_migration(1)
        .await
        .expect("Failed to record v1");
    manager.run_migration(2).await.expect("Failed to run v2");
    manager
        .record_migration(2)
        .await
        .expect("Failed to record v2");

    // Run only through v3 to assert v3's reconcile index — the later v48
    // migration rebuilds tracked_files and drops the v3-era indexes, so the
    // check must happen at the version it targets.
    manager
        .run_migrations_through(3)
        .await
        .expect("Failed to run migrations from v2 through v3");

    let has_index: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_tracked_files_reconcile'"
    )
    .fetch_one(&pool).await.unwrap();
    assert!(has_index, "Reconcile index should exist after v3 migration");

    // Finish the chain and confirm it lands at the current version.
    manager
        .run_migrations()
        .await
        .expect("Failed to run remaining migrations");
    let version = manager.get_current_version().await.unwrap();
    assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));
}

#[tokio::test]
#[serial]
async fn test_incremental_migration_v3_to_v4() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager.initialize().await.expect("Failed to initialize");
    for v in 1..=3 {
        manager
            .run_migration(v)
            .await
            .unwrap_or_else(|e| panic!("Failed to run v{}: {}", v, e));
        manager
            .record_migration(v)
            .await
            .unwrap_or_else(|e| panic!("Failed to record v{}: {}", v, e));
    }

    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations from v3 to v4");

    let version = manager.get_current_version().await.unwrap();
    assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));

    let has_is_paused: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'is_paused'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        has_is_paused,
        "is_paused column should exist after v4 migration"
    );

    let has_pause_start_time: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'pause_start_time'"
    )
    .fetch_one(&pool).await.unwrap();
    assert!(
        has_pause_start_time,
        "pause_start_time column should exist after v4 migration"
    );
}

#[tokio::test]
#[serial]
async fn test_incremental_migration_v4_to_current() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager.initialize().await.expect("Failed to initialize");
    for v in 1..=4 {
        manager
            .run_migration(v)
            .await
            .unwrap_or_else(|e| panic!("Failed to run v{}: {}", v, e));
        manager
            .record_migration(v)
            .await
            .unwrap_or_else(|e| panic!("Failed to record v{}: {}", v, e));
    }

    let exists_before: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='metrics_history')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        !exists_before,
        "metrics_history should NOT exist before v5 migration"
    );

    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations from v4 to current");

    let version = manager.get_current_version().await.unwrap();
    assert_eq!(version, Some(CURRENT_SCHEMA_VERSION as i32));

    let exists_after: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='metrics_history')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        exists_after,
        "metrics_history table should exist after v5 migration"
    );

    let idx_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name LIKE 'idx_metrics_%'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(idx_count, 3, "Should have 3 metrics_history indexes");

    let has_collection: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'collection'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        has_collection,
        "collection column should exist after v6 migration"
    );
}

// ============================================================================
// §7.11 — Migration v35 (active_provider column + reembed op)
// ============================================================================

#[tokio::test]
#[serial]
async fn test_v35_adds_active_provider_column() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations to current version");

    let has_active_provider: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') \
         WHERE name = 'active_provider'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        has_active_provider,
        "watch_folders.active_provider must exist after v35"
    );

    // Default value: an INSERT with no explicit value yields 'openai_compatible'.
    let now = "2026-01-01T00:00:00.000Z";
    sqlx::query(
        "INSERT INTO watch_folders \
            (watch_id, path, collection, tenant_id, created_at, updated_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
    )
    .bind("wf-v35-default")
    .bind("/tmp/v35-default")
    .bind("projects")
    .bind("tenant-v35")
    .bind(now)
    .execute(&pool)
    .await
    .expect("Failed to insert watch_folders row");
    let provider: String = sqlx::query_scalar(
        "SELECT active_provider FROM watch_folders WHERE watch_id = 'wf-v35-default'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(provider, "openai_compatible");
}

#[tokio::test]
#[serial]
async fn test_v35_queue_op_accepts_reembed() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations to current version");

    sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key) \
         VALUES ('collection', 'reembed', 't-v35', 'projects', 'idem-v35-1')",
    )
    .execute(&pool)
    .await
    .expect("Inserting op='reembed' should succeed after v35");
}

#[tokio::test]
#[serial]
async fn test_v35_existing_in_progress_rows_survive_migration() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());

    manager.initialize().await.expect("init");
    for v in 1..=34 {
        manager
            .run_migration(v)
            .await
            .unwrap_or_else(|e| panic!("Failed to run v{}: {}", v, e));
        manager
            .record_migration(v)
            .await
            .unwrap_or_else(|e| panic!("Failed to record v{}: {}", v, e));
    }

    sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key, status) \
         VALUES ('file', 'add', 't-v35', 'projects', 'idem-v35-survive', 'in_progress')",
    )
    .execute(&pool)
    .await
    .expect("Failed to insert pre-v35 in_progress row");

    manager.run_migration(35).await.expect("v35 should run");
    manager.record_migration(35).await.expect("record v35");

    let status: String = sqlx::query_scalar(
        "SELECT status FROM unified_queue WHERE idempotency_key = 'idem-v35-survive'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        status, "in_progress",
        "pre-v35 in_progress row must survive the table rebuild with status preserved"
    );
}

#[tokio::test]
#[serial]
async fn test_v35_ddl_has_no_extra_item_types() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations to current version");

    let queue_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    // The 8 canonical item_type values, in DDL order. No 'library', 'memory',
    // or 'scratchpad' should have leaked in.
    for kind in [
        "'text'",
        "'file'",
        "'url'",
        "'website'",
        "'doc'",
        "'folder'",
        "'tenant'",
        "'collection'",
    ] {
        assert!(
            queue_sql.contains(kind),
            "expected item_type {} in DDL, full DDL: {}",
            kind,
            queue_sql
        );
    }
    for forbidden in ["'library'", "'memory'", "'scratchpad'"] {
        assert!(
            !queue_sql.contains(forbidden),
            "v35 DDL must not contain {}, full DDL: {}",
            forbidden,
            queue_sql
        );
    }
}

// ============================================================================
// §7.12 — Migration v36 (composite partial UNIQUE on file_path)
// ============================================================================

/// v36 rebuilds `unified_queue` so `file_path` is a plain TEXT column
/// (no UNIQUE) and creates a composite partial UNIQUE index. Closes F-009.
#[tokio::test]
#[serial]
async fn test_v36_drops_column_unique_and_creates_composite_index() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations to current version");

    let queue_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        !queue_sql.contains("file_path TEXT UNIQUE"),
        "v36 must drop the column UNIQUE on file_path; got: {}",
        queue_sql
    );

    let composite_index: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master \
         WHERE type='index' AND name='idx_unified_queue_file_path_composite')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        composite_index,
        "v36 must create idx_unified_queue_file_path_composite"
    );
}

/// v36 allows the same `file_path` for two tenants and rejects duplicate
/// inserts within the same `(tenant, branch, collection, item_type, op)`
/// scope. Closes F-009.
#[tokio::test]
#[serial]
async fn test_v36_composite_unique_enforced_after_migration() {
    let pool = create_test_pool().await;
    let manager = SchemaManager::new(pool.clone());
    manager
        .run_migrations()
        .await
        .expect("Failed to run migrations to current version");

    sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key, branch, file_path) \
         VALUES ('file', 'add', 't-A', 'projects', 'idem-v36-1', 'main', '/data/x.md')",
    )
    .execute(&pool)
    .await
    .expect("first row must insert");

    // Same file_path, different tenant: allowed.
    sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key, branch, file_path) \
         VALUES ('file', 'add', 't-B', 'projects', 'idem-v36-2', 'main', '/data/x.md')",
    )
    .execute(&pool)
    .await
    .expect("cross-tenant insert must succeed");

    // Same (tenant, branch, collection, item_type, op, file_path): rejected.
    let err = sqlx::query(
        "INSERT INTO unified_queue \
             (item_type, op, tenant_id, collection, idempotency_key, branch, file_path) \
         VALUES ('file', 'add', 't-A', 'projects', 'idem-v36-3', 'main', '/data/x.md')",
    )
    .execute(&pool)
    .await;
    assert!(
        err.is_err(),
        "duplicate within same scope must violate composite UNIQUE"
    );
}

// ============================================================================
// §F-046 — v35 atomicity: BEGIN IMMEDIATE / COMMIT / ROLLBACK +
//          ForeignKeysGuard panic safety
// ============================================================================

/// Happy path: v34 pool → run v35 → schema is at v35 with correct structure.
#[tokio::test]
#[serial]
async fn test_v35_migration_atomic_success() {
    let (pool, manager) = create_test_pool_v34().await;

    manager
        .run_migration(35)
        .await
        .expect("v35 migration must succeed");
    manager
        .record_migration(35)
        .await
        .expect("record v35 must succeed");

    let version = manager.get_current_version().await.unwrap();
    assert_eq!(version, Some(35), "schema must be at v35 after migration");

    // active_provider column must exist.
    let has_active_provider: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') \
         WHERE name = 'active_provider'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(has_active_provider, "active_provider must exist after v35");

    // unified_queue must accept 'reembed'.
    let queue_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        queue_sql.contains("'reembed'"),
        "unified_queue must accept 'reembed' after v35; DDL: {}",
        queue_sql
    );

    // Transaction must have committed: no outstanding write lock.
    sqlx::query("SELECT 1").fetch_one(&pool).await.unwrap();
}

/// Failure injection: the fail-point fires after watch_folders ALTER but before
/// unified_queue rebuild. The migration must roll back, leaving the schema in
/// v34 state (no active_provider, no 'reembed' in CHECK). F-046 acceptance test.
#[tokio::test]
#[serial]
async fn test_v35_migration_rollback_on_error() {
    use crate::schema_version::v35::INJECT_FAILURE_AFTER_WATCH_FOLDERS;
    use std::sync::atomic::Ordering;

    let (pool, manager) = create_test_pool_v34().await;

    INJECT_FAILURE_AFTER_WATCH_FOLDERS.store(true, Ordering::SeqCst);
    let result = manager.run_migration(35).await;
    INJECT_FAILURE_AFTER_WATCH_FOLDERS.store(false, Ordering::SeqCst);

    assert!(
        result.is_err(),
        "migration must fail when fail-point is active"
    );

    // Schema must still be at v34 — the transaction was rolled back.
    let version = manager.get_current_version().await.unwrap();
    assert_eq!(
        version,
        Some(34),
        "schema must remain at v34 after rollback"
    );

    // watch_folders must NOT have gained active_provider.
    let has_active_provider: bool = sqlx::query_scalar(
        "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') \
         WHERE name = 'active_provider'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        !has_active_provider,
        "active_provider must not exist after failed v35 migration (rolled back)"
    );

    // unified_queue must NOT accept 'reembed' (rebuild was rolled back).
    let queue_sql: String = sqlx::query_scalar(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        !queue_sql.contains("'reembed'"),
        "unified_queue must not accept 'reembed' after failed v35 migration; DDL: {}",
        queue_sql
    );

    // FK checks must be restored (PRAGMA foreign_keys = 1).
    let fk_on: i32 = sqlx::query_scalar("PRAGMA foreign_keys")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(
        fk_on, 1,
        "PRAGMA foreign_keys must be restored to ON after failed migration"
    );
}

/// ForeignKeysGuard explicit restore: verify that `restore()` on the success
/// path returns the connection with FK checks correctly reinstated.
///
/// This test exercises the non-panic path end-to-end: disable → verify OFF →
/// restore → verify ON — all on the same `PoolConnection`.
#[tokio::test]
#[serial]
async fn test_foreign_keys_guard_explicit_restore() {
    let pool = create_test_pool().await;

    // Acquire a connection, enable FK checks so there is a non-zero value to
    // restore, then hand it to the guard.
    let mut conn = pool.acquire().await.unwrap();
    conn.execute("PRAGMA foreign_keys = ON").await.unwrap();

    let fk_before: i32 = sqlx::query_scalar("PRAGMA foreign_keys")
        .fetch_one(&mut *conn)
        .await
        .unwrap();
    assert_eq!(fk_before, 1, "FK must be ON before guard is constructed");

    let mut guard = ForeignKeysGuard::disable(conn).await.expect("disable");

    // While the guard is held FK must be OFF on the same connection.
    let fk_off: i32 = sqlx::query_scalar("PRAGMA foreign_keys")
        .fetch_one(&mut **guard.conn_mut())
        .await
        .unwrap();
    assert_eq!(fk_off, 0, "FK must be OFF while guard is held");

    // Explicit restore returns the connection; FK must be ON again.
    let mut restored_conn = guard.restore().await.expect("restore");
    let fk_restored: i32 = sqlx::query_scalar("PRAGMA foreign_keys")
        .fetch_one(&mut *restored_conn)
        .await
        .unwrap();
    assert_eq!(
        fk_restored, 1,
        "FK must be restored to ON on the returned connection after explicit restore()"
    );
}

/// ForeignKeysGuard panic safety: if the code panics while the guard is held,
/// Drop must restore PRAGMA foreign_keys via block_in_place without a
/// secondary panic.
///
/// Requires multi_thread flavor because block_in_place is not available on
/// current-thread runtimes.
///
/// Note: SQLite `foreign_keys` is a per-connection pragma. The guard restores
/// FK on its owned `PoolConnection` inside Drop; once that connection is
/// returned to the pool it is a distinct object from any subsequent acquire.
/// Therefore the restoration cannot be observed post-Drop on a new connection —
/// we verify the weaker (but still meaningful) guarantee: Drop fires, executes
/// the restore path, and does NOT produce a secondary panic.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_foreign_keys_guard_panic_safety() {
    use sqlx::sqlite::SqlitePoolOptions;
    use std::time::Duration;

    let pool = SqlitePoolOptions::new()
        .max_connections(2)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("pool");

    // Acquire a connection for the guard and enable FK checks so the guard
    // has a non-zero original_value to restore.
    let mut guard_conn = pool.acquire().await.unwrap();
    guard_conn
        .execute("PRAGMA foreign_keys = ON")
        .await
        .unwrap();

    // Spawn a task that acquires a guard, then panics.  The guard's Drop must
    // restore FK = ON on the connection and must NOT produce a secondary panic.
    let result = tokio::task::spawn(async move {
        let mut guard = ForeignKeysGuard::disable(guard_conn)
            .await
            .expect("disable");

        // Confirm FK is OFF while the guard is held.
        let fk_off: i32 = sqlx::query_scalar("PRAGMA foreign_keys")
            .fetch_one(&mut **guard.conn_mut())
            .await
            .unwrap();
        assert_eq!(fk_off, 0, "FK must be OFF while guard is held");

        // Panic — Drop fires here and must not produce a secondary panic.
        panic!("intentional test panic inside ForeignKeysGuard scope");

        #[allow(unreachable_code)]
        0i32
    })
    .await;

    // The spawned task must have panicked exactly once (the intentional panic).
    // If Drop had panicked too the process would have aborted; we would not
    // reach this assertion.
    assert!(result.is_err(), "task must have panicked as expected");
}

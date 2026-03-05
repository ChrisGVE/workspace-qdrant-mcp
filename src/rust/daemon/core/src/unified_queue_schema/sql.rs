//! SQL constants for the unified queue schema.
//!
//! Contains DDL for the `unified_queue` table, its indexes, and migration
//! statements for schema version upgrades.

/// SQL to create the unified queue schema.
///
/// Priority is computed dynamically at dequeue time via a CASE expression that JOINs
/// with `watch_folders.is_active`. Two levels:
/// - High (1): active projects + memory collection
/// - Low (0): libraries + inactive projects
///
/// See `dequeue_unified()` in `queue_operations/dequeue.rs`.
pub const CREATE_UNIFIED_QUEUE_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS unified_queue (
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
    max_retries INTEGER NOT NULL DEFAULT 3,
    error_message TEXT,
    last_error_at TEXT,
    branch TEXT DEFAULT 'main',
    metadata TEXT DEFAULT '{}',
    -- Task 22: Per-file deduplication
    -- Only set for item_type='file', NULL for other types
    -- UNIQUE constraint prevents duplicate file ingestion
    file_path TEXT UNIQUE,
    -- State integrity: per-destination status tracking
    qdrant_status TEXT DEFAULT 'pending' CHECK (qdrant_status IN ('pending', 'in_progress', 'done', 'failed')),
    search_status TEXT DEFAULT 'pending' CHECK (search_status IN ('pending', 'in_progress', 'done', 'failed')),
    -- Pre-computed processing decision (JSON)
    decision_json TEXT
)
"#;

/// SQL to create indexes for the unified queue.
pub const CREATE_UNIFIED_QUEUE_INDEXES_SQL: &[&str] = &[
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_dequeue
       ON unified_queue(status, created_at ASC)
       WHERE status = 'pending'"#,
    r#"CREATE UNIQUE INDEX IF NOT EXISTS idx_unified_queue_idempotency
       ON unified_queue(idempotency_key)"#,
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_lease_expiry
       ON unified_queue(lease_until)
       WHERE status = 'in_progress'"#,
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_collection_tenant
       ON unified_queue(collection, tenant_id)"#,
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_item_type
       ON unified_queue(item_type, status)"#,
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_failed
       ON unified_queue(status, last_error_at DESC)
       WHERE status = 'failed'"#,
    // Task 22: Per-file deduplication index
    r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_file_path
       ON unified_queue(file_path)
       WHERE file_path IS NOT NULL"#,
];

// ---------------------------------------------------------------------------
// Migration SQL — v20: qdrant_status, search_status, decision_json columns
// ---------------------------------------------------------------------------

/// SQL statements for migration v20: add per-destination status and decision columns.
///
/// NOTE: SQLite ALTER TABLE ADD COLUMN supports CHECK constraints only if they
/// reference the added column alone. We omit CHECK here for maximum compatibility
/// and rely on daemon code to enforce valid values.
pub const MIGRATE_V20_ADD_COLUMNS_SQL: &[&str] = &[
    "ALTER TABLE unified_queue ADD COLUMN qdrant_status TEXT DEFAULT 'pending'",
    "ALTER TABLE unified_queue ADD COLUMN search_status TEXT DEFAULT 'pending'",
    "ALTER TABLE unified_queue ADD COLUMN decision_json TEXT",
];

/// Index for finding items with incomplete Qdrant writes.
pub const CREATE_QDRANT_STATUS_INDEX_SQL: &str = r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_qdrant_status
       ON unified_queue(qdrant_status) WHERE qdrant_status != 'done'"#;

/// Index for finding items with incomplete search DB writes.
pub const CREATE_SEARCH_STATUS_INDEX_SQL: &str = r#"CREATE INDEX IF NOT EXISTS idx_unified_queue_search_status
       ON unified_queue(search_status) WHERE search_status != 'done'"#;

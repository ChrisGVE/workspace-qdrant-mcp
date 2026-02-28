//! SQL schema constants for tracked_files and qdrant_chunks tables

// ---------------------------------------------------------------------------
// SQL constants — tracked_files
// ---------------------------------------------------------------------------

/// SQL to create the tracked_files table
pub const CREATE_TRACKED_FILES_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS tracked_files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_folder_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    branch TEXT,
    file_type TEXT,
    language TEXT,
    file_mtime TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    chunking_method TEXT,
    lsp_status TEXT DEFAULT 'none' CHECK (lsp_status IN ('none', 'done', 'failed', 'skipped')),
    treesitter_status TEXT DEFAULT 'none' CHECK (treesitter_status IN ('none', 'done', 'failed', 'skipped')),
    last_error TEXT,
    needs_reconcile INTEGER DEFAULT 0,
    reconcile_reason TEXT,
    extension TEXT,
    is_test INTEGER DEFAULT 0,
    collection TEXT NOT NULL DEFAULT 'projects',
    base_point TEXT,
    relative_path TEXT,
    incremental INTEGER DEFAULT 0,
    component TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
    UNIQUE(watch_folder_id, file_path, branch)
)
"#;

/// SQL to create indexes for the tracked_files table
pub const CREATE_TRACKED_FILES_INDEXES_SQL: &[&str] = &[
    // Index for recovery: walk all files for a project
    r#"CREATE INDEX IF NOT EXISTS idx_tracked_files_watch
       ON tracked_files(watch_folder_id)"#,
    // Index for finding files by path (e.g., file watcher events)
    r#"CREATE INDEX IF NOT EXISTS idx_tracked_files_path
       ON tracked_files(file_path)"#,
    // Index for branch operations
    r#"CREATE INDEX IF NOT EXISTS idx_tracked_files_branch
       ON tracked_files(watch_folder_id, branch)"#,
];

// ---------------------------------------------------------------------------
// SQL constants — qdrant_chunks
// ---------------------------------------------------------------------------

/// SQL to create the qdrant_chunks table
pub const CREATE_QDRANT_CHUNKS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS qdrant_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    point_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_type TEXT,
    symbol_name TEXT,
    start_line INTEGER,
    end_line INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (file_id) REFERENCES tracked_files(file_id) ON DELETE CASCADE,
    UNIQUE(file_id, chunk_index)
)
"#;

/// SQL to create indexes for the qdrant_chunks table
pub const CREATE_QDRANT_CHUNKS_INDEXES_SQL: &[&str] = &[
    // Index for looking up chunks by Qdrant point ID
    r#"CREATE INDEX IF NOT EXISTS idx_qdrant_chunks_point
       ON qdrant_chunks(point_id)"#,
    // Index for file's chunks
    r#"CREATE INDEX IF NOT EXISTS idx_qdrant_chunks_file
       ON qdrant_chunks(file_id)"#,
];

// ---------------------------------------------------------------------------
// Migration SQL — v3: needs_reconcile columns
// ---------------------------------------------------------------------------

/// SQL statements for migration v3: add needs_reconcile and reconcile_reason
/// to tracked_files table
pub const MIGRATE_V3_SQL: &[&str] = &[
    "ALTER TABLE tracked_files ADD COLUMN needs_reconcile INTEGER DEFAULT 0",
    "ALTER TABLE tracked_files ADD COLUMN reconcile_reason TEXT",
];

// ---------------------------------------------------------------------------
// Migration SQL — v6: collection column for format-based routing
// ---------------------------------------------------------------------------

/// SQL statement for migration v6: add collection column to tracked_files
pub const MIGRATE_V6_SQL: &str =
    "ALTER TABLE tracked_files ADD COLUMN collection TEXT NOT NULL DEFAULT 'projects'";

// ---------------------------------------------------------------------------
// Migration SQL — v8: extension and is_test columns
// ---------------------------------------------------------------------------

/// SQL statements for migration v8: add extension and is_test columns to tracked_files
pub const MIGRATE_V8_ADD_COLUMNS_SQL: &[&str] = &[
    "ALTER TABLE tracked_files ADD COLUMN extension TEXT",
    "ALTER TABLE tracked_files ADD COLUMN is_test INTEGER DEFAULT 0",
];

// ---------------------------------------------------------------------------
// Migration SQL — v19: base_point, relative_path, incremental columns
// ---------------------------------------------------------------------------

/// SQL statements for migration v19: add base_point, relative_path, incremental
/// columns to tracked_files for content-addressed identity
/// SQL statement for migration v28: add component column to tracked_files
pub const MIGRATE_V28_ADD_COMPONENT_SQL: &str =
    "ALTER TABLE tracked_files ADD COLUMN component TEXT";

/// SQL statements for migration v19: add base_point, relative_path, incremental
/// columns to tracked_files for content-addressed identity
pub const MIGRATE_V19_ADD_COLUMNS_SQL: &[&str] = &[
    "ALTER TABLE tracked_files ADD COLUMN base_point TEXT",
    "ALTER TABLE tracked_files ADD COLUMN relative_path TEXT",
    "ALTER TABLE tracked_files ADD COLUMN incremental INTEGER DEFAULT 0",
];

/// Index for base_point lookups (e.g., reference-counted deletion)
pub const CREATE_BASE_POINT_INDEX_SQL: &str =
    r#"CREATE INDEX IF NOT EXISTS idx_tracked_files_base_point
       ON tracked_files(base_point) WHERE base_point IS NOT NULL"#;

/// Index for reference counting: find all files sharing a base_point across tenants
pub const CREATE_REFCOUNT_INDEX_SQL: &str =
    r#"CREATE INDEX IF NOT EXISTS idx_tracked_files_refcount
       ON tracked_files(base_point, watch_folder_id)"#;

/// Index for quickly finding files needing reconciliation
pub const CREATE_RECONCILE_INDEX_SQL: &str =
    r#"CREATE INDEX IF NOT EXISTS idx_tracked_files_reconcile
       ON tracked_files(needs_reconcile) WHERE needs_reconcile = 1"#;

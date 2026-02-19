//! Schema for the `code_lines` table in search.db.
//!
//! Stores line-level content for code files using gap-based `seq` ordering
//! (REAL type) to allow efficient insertions without cascading updates.
//! Line numbers are derived at query time via `ROW_NUMBER() OVER (ORDER BY seq)`.
//!
//! `file_id` references `tracked_files.file_id` in state.db. Cross-database
//! foreign keys are enforced at the application level (SQLite ATTACH does not
//! support cross-database FK constraints natively).
//!
//! Created in search.db schema version 2.

/// SQL to create the code_lines table.
pub const CREATE_CODE_LINES_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS code_lines (
    line_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    seq REAL NOT NULL,
    content TEXT NOT NULL,
    UNIQUE(file_id, seq)
)
"#;

/// Indexes for the code_lines table.
pub const CREATE_CODE_LINES_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_code_lines_file ON code_lines(file_id, seq)",
];

/// Initial gap between consecutive seq values.
///
/// Lines are inserted with seq values at multiples of this gap (1000.0, 2000.0, ...).
/// New lines inserted between existing lines get the midpoint (e.g., 1500.0).
pub const INITIAL_SEQ_GAP: f64 = 1000.0;

/// Minimum gap before rebalancing is needed.
///
/// When the gap between adjacent seq values drops below this threshold,
/// a local or full rebalance should be triggered (handled by Task 50).
pub const MIN_SEQ_GAP: f64 = 0.001;

/// Compute the initial seq value for a given 0-based line index.
///
/// Returns `(index + 1) * INITIAL_SEQ_GAP` (i.e., 1000.0, 2000.0, ...).
pub fn initial_seq(line_index: usize) -> f64 {
    (line_index as f64 + 1.0) * INITIAL_SEQ_GAP
}

/// Compute the midpoint seq between two adjacent seq values.
///
/// Used when inserting a new line between existing lines.
pub fn midpoint_seq(before: f64, after: f64) -> f64 {
    (before + after) / 2.0
}

// ============================================================================
// FTS5 Trigram Virtual Table (Task 47)
// ============================================================================

/// SQL to create the FTS5 trigram virtual table.
///
/// Uses external content mode: the FTS index references `code_lines.line_id`
/// via `content_rowid` and reads `content` from the `code_lines` table.
/// The trigram tokenizer generates all 3-character substrings for fast
/// substring matching (e.g., searching "println" matches via trigrams "pri",
/// "rin", "int", "ntl", "tln").
///
/// Since this uses external content mode, the FTS index must be rebuilt
/// after batch inserts/updates to `code_lines`. Use `rebuild_fts()` after
/// bulk operations rather than maintaining triggers (more efficient for
/// our batch-oriented workload).
pub const CREATE_CODE_LINES_FTS_SQL: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS code_lines_fts USING fts5(
    content,
    content='code_lines',
    content_rowid='line_id',
    tokenize='trigram'
)
"#;

/// SQL to rebuild the FTS5 index from the external content table.
///
/// Must be called after batch inserts/updates/deletes to `code_lines`.
pub const FTS5_REBUILD_SQL: &str =
    "INSERT INTO code_lines_fts(code_lines_fts) VALUES('rebuild')";

/// SQL to optimize the FTS5 index.
///
/// Merges internal b-tree segments for faster queries. Call after large
/// batch operations (>1000 lines) or periodically during idle time.
pub const FTS5_OPTIMIZE_SQL: &str =
    "INSERT INTO code_lines_fts(code_lines_fts) VALUES('optimize')";

/// Threshold for triggering FTS5 optimization after batch operations.
pub const FTS5_OPTIMIZE_THRESHOLD: usize = 1000;

/// SQL to search code_lines via FTS5 trigram MATCH.
///
/// Returns line_id, file_id, seq, content for matching lines.
/// The `?1` parameter is the search pattern (substring match via trigrams).
/// Use with `ROW_NUMBER()` for line numbers if needed.
pub const FTS5_SEARCH_SQL: &str = r#"
SELECT cl.line_id, cl.file_id, cl.seq, cl.content
FROM code_lines cl
JOIN code_lines_fts fts ON cl.line_id = fts.rowid
WHERE fts.content MATCH ?1
ORDER BY cl.file_id, cl.seq
"#;

/// SQL to search code_lines via FTS5 with file_id filter.
///
/// Returns matching lines within a specific file.
/// `?1` = search pattern, `?2` = file_id.
pub const FTS5_SEARCH_BY_FILE_SQL: &str = r#"
SELECT cl.line_id, cl.file_id, cl.seq, cl.content
FROM code_lines cl
JOIN code_lines_fts fts ON cl.line_id = fts.rowid
WHERE fts.content MATCH ?1 AND cl.file_id = ?2
ORDER BY cl.seq
"#;

// ============================================================================
// File Metadata Table (search.db v4 — project/branch/path scoping)
// ============================================================================

/// SQL to create the file_metadata table in search.db.
///
/// Denormalizes `tenant_id`, `branch`, and `file_path` from state.db's
/// `tracked_files` into search.db so FTS5 queries can be scoped by
/// project, branch, or path prefix without cross-database JOINs.
///
/// `file_id` is the same value as `tracked_files.file_id` in state.db.
/// Application-level consistency is maintained: when code_lines are
/// inserted/deleted for a file, its file_metadata row is upserted/removed.
pub const CREATE_FILE_METADATA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS file_metadata (
    file_id INTEGER PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    branch TEXT,
    file_path TEXT NOT NULL
)
"#;

/// SQL to add base_point columns to file_metadata (search.db v5).
///
/// Adds base_point, relative_path, and file_hash to align with the
/// base_point identity model used by Qdrant and tracked_files.
pub const ALTER_FILE_METADATA_V5_SQL: &[&str] = &[
    "ALTER TABLE file_metadata ADD COLUMN base_point TEXT",
    "ALTER TABLE file_metadata ADD COLUMN relative_path TEXT",
    "ALTER TABLE file_metadata ADD COLUMN file_hash TEXT",
];

/// Index for base_point lookups (search.db v5).
pub const CREATE_FILE_METADATA_BASE_POINT_INDEX_SQL: &str =
    "CREATE INDEX IF NOT EXISTS idx_file_metadata_base_point ON file_metadata(base_point)";

/// Indexes for the file_metadata table.
pub const CREATE_FILE_METADATA_INDEXES_SQL: &[&str] = &[
    "CREATE INDEX IF NOT EXISTS idx_file_metadata_tenant ON file_metadata(tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_file_metadata_tenant_branch ON file_metadata(tenant_id, branch)",
];

/// SQL to upsert a file_metadata row.
///
/// `?1` = file_id, `?2` = tenant_id, `?3` = branch (nullable), `?4` = file_path,
/// `?5` = base_point (nullable), `?6` = relative_path (nullable), `?7` = file_hash (nullable).
pub const UPSERT_FILE_METADATA_SQL: &str = r#"
INSERT INTO file_metadata (file_id, tenant_id, branch, file_path, base_point, relative_path, file_hash)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
ON CONFLICT(file_id) DO UPDATE SET
    tenant_id = excluded.tenant_id,
    branch = excluded.branch,
    file_path = excluded.file_path,
    base_point = excluded.base_point,
    relative_path = excluded.relative_path,
    file_hash = excluded.file_hash
"#;

/// SQL to delete a file_metadata row when its code_lines are removed.
pub const DELETE_FILE_METADATA_SQL: &str =
    "DELETE FROM file_metadata WHERE file_id = ?1";

/// SQL to delete all file_metadata rows for a tenant (project deletion).
pub const DELETE_FILE_METADATA_BY_TENANT_SQL: &str =
    "DELETE FROM file_metadata WHERE tenant_id = ?1";

/// SQL to delete file_metadata and cascade to code_lines by base_point.
///
/// Returns the file_id so the caller can also delete from code_lines.
pub const SELECT_FILE_ID_BY_BASE_POINT_SQL: &str =
    "SELECT file_id FROM file_metadata WHERE base_point = ?1";

// ============================================================================
// Scoped FTS5 Search Queries
// ============================================================================

/// FTS5 search scoped to a project (tenant_id).
///
/// `?1` = search pattern, `?2` = tenant_id.
pub const FTS5_SEARCH_BY_PROJECT_SQL: &str = r#"
SELECT cl.line_id, cl.file_id, cl.seq, cl.content, fm.tenant_id, fm.file_path, fm.branch
FROM code_lines cl
JOIN code_lines_fts fts ON cl.line_id = fts.rowid
JOIN file_metadata fm ON cl.file_id = fm.file_id
WHERE fts.content MATCH ?1 AND fm.tenant_id = ?2
ORDER BY cl.file_id, cl.seq
"#;

/// FTS5 search scoped to a project and branch.
///
/// `?1` = search pattern, `?2` = tenant_id, `?3` = branch.
pub const FTS5_SEARCH_BY_PROJECT_BRANCH_SQL: &str = r#"
SELECT cl.line_id, cl.file_id, cl.seq, cl.content, fm.tenant_id, fm.file_path, fm.branch
FROM code_lines cl
JOIN code_lines_fts fts ON cl.line_id = fts.rowid
JOIN file_metadata fm ON cl.file_id = fm.file_id
WHERE fts.content MATCH ?1 AND fm.tenant_id = ?2 AND fm.branch = ?3
ORDER BY cl.file_id, cl.seq
"#;

/// FTS5 search scoped by file path prefix.
///
/// `?1` = search pattern, `?2` = path prefix (use `prefix%` with LIKE).
pub const FTS5_SEARCH_BY_PATH_PREFIX_SQL: &str = r#"
SELECT cl.line_id, cl.file_id, cl.seq, cl.content, fm.tenant_id, fm.file_path, fm.branch
FROM code_lines cl
JOIN code_lines_fts fts ON cl.line_id = fts.rowid
JOIN file_metadata fm ON cl.file_id = fm.file_id
WHERE fts.content MATCH ?1 AND fm.file_path LIKE ?2
ORDER BY cl.file_id, cl.seq
"#;

/// FTS5 search scoped to a project with path prefix filter.
///
/// `?1` = search pattern, `?2` = tenant_id, `?3` = path prefix (use `prefix%` with LIKE).
pub const FTS5_SEARCH_BY_PROJECT_PATH_SQL: &str = r#"
SELECT cl.line_id, cl.file_id, cl.seq, cl.content, fm.tenant_id, fm.file_path, fm.branch
FROM code_lines cl
JOIN code_lines_fts fts ON cl.line_id = fts.rowid
JOIN file_metadata fm ON cl.file_id = fm.file_id
WHERE fts.content MATCH ?1 AND fm.tenant_id = ?2 AND fm.file_path LIKE ?3
ORDER BY cl.file_id, cl.seq
"#;

// ============================================================================
// Line Number Queries
// ============================================================================

/// SQL to derive line numbers from seq ordering.
///
/// Use as a subquery or CTE. Returns `line_id`, `line_number` (1-based),
/// `seq`, and `content` for a given `file_id`.
pub const LINE_NUMBER_QUERY: &str = r#"
SELECT
    line_id,
    ROW_NUMBER() OVER (ORDER BY seq) AS line_number,
    seq,
    content
FROM code_lines
WHERE file_id = ?1
ORDER BY seq
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_code_lines_sql_valid() {
        assert!(CREATE_CODE_LINES_SQL.contains("CREATE TABLE"));
        assert!(CREATE_CODE_LINES_SQL.contains("code_lines"));
        assert!(CREATE_CODE_LINES_SQL.contains("file_id INTEGER NOT NULL"));
        assert!(CREATE_CODE_LINES_SQL.contains("seq REAL NOT NULL"));
        assert!(CREATE_CODE_LINES_SQL.contains("content TEXT NOT NULL"));
        assert!(CREATE_CODE_LINES_SQL.contains("UNIQUE(file_id, seq)"));
    }

    #[test]
    fn test_indexes_idempotent() {
        for sql in CREATE_CODE_LINES_INDEXES_SQL {
            assert!(sql.contains("IF NOT EXISTS"), "Missing IF NOT EXISTS: {}", sql);
        }
    }

    #[test]
    fn test_index_count() {
        assert_eq!(CREATE_CODE_LINES_INDEXES_SQL.len(), 1);
    }

    #[test]
    fn test_initial_seq_values() {
        assert_eq!(initial_seq(0), 1000.0);
        assert_eq!(initial_seq(1), 2000.0);
        assert_eq!(initial_seq(99), 100_000.0);
    }

    #[test]
    fn test_midpoint_seq() {
        assert_eq!(midpoint_seq(1000.0, 2000.0), 1500.0);
        assert_eq!(midpoint_seq(1000.0, 1500.0), 1250.0);
        assert_eq!(midpoint_seq(0.0, 1000.0), 500.0);
    }

    #[test]
    fn test_min_seq_gap_is_small() {
        // Must be small enough for many insertions between integer gaps
        assert!(MIN_SEQ_GAP < 1.0);
        assert!(MIN_SEQ_GAP > 0.0);
    }

    #[test]
    fn test_initial_gap_allows_many_insertions() {
        // With INITIAL_SEQ_GAP = 1000.0 and MIN_SEQ_GAP = 0.001,
        // we can do ~20 successive midpoint insertions before rebalancing
        let mut gap = INITIAL_SEQ_GAP;
        let mut insertions = 0;
        while gap > MIN_SEQ_GAP {
            gap /= 2.0;
            insertions += 1;
        }
        assert!(
            insertions >= 19,
            "Should support at least 19 midpoint insertions, got {}",
            insertions
        );
    }

    #[test]
    fn test_line_number_query_valid() {
        assert!(LINE_NUMBER_QUERY.contains("ROW_NUMBER()"));
        assert!(LINE_NUMBER_QUERY.contains("ORDER BY seq"));
        assert!(LINE_NUMBER_QUERY.contains("file_id = ?1"));
    }

    // ── FTS5 SQL constant tests (Task 47) ──

    #[test]
    fn test_fts5_create_sql_valid() {
        assert!(CREATE_CODE_LINES_FTS_SQL.contains("CREATE VIRTUAL TABLE"));
        assert!(CREATE_CODE_LINES_FTS_SQL.contains("code_lines_fts"));
        assert!(CREATE_CODE_LINES_FTS_SQL.contains("fts5"));
        assert!(CREATE_CODE_LINES_FTS_SQL.contains("content='code_lines'"));
        assert!(CREATE_CODE_LINES_FTS_SQL.contains("content_rowid='line_id'"));
        assert!(CREATE_CODE_LINES_FTS_SQL.contains("tokenize='trigram'"));
    }

    #[test]
    fn test_fts5_rebuild_sql_valid() {
        assert!(FTS5_REBUILD_SQL.contains("code_lines_fts"));
        assert!(FTS5_REBUILD_SQL.contains("rebuild"));
    }

    #[test]
    fn test_fts5_optimize_sql_valid() {
        assert!(FTS5_OPTIMIZE_SQL.contains("code_lines_fts"));
        assert!(FTS5_OPTIMIZE_SQL.contains("optimize"));
    }

    #[test]
    fn test_fts5_search_sql_valid() {
        assert!(FTS5_SEARCH_SQL.contains("code_lines_fts"));
        assert!(FTS5_SEARCH_SQL.contains("MATCH ?1"));
        assert!(FTS5_SEARCH_SQL.contains("ORDER BY"));
    }

    #[test]
    fn test_fts5_search_by_file_sql_valid() {
        assert!(FTS5_SEARCH_BY_FILE_SQL.contains("MATCH ?1"));
        assert!(FTS5_SEARCH_BY_FILE_SQL.contains("file_id = ?2"));
    }

    #[test]
    fn test_fts5_optimize_threshold() {
        assert_eq!(FTS5_OPTIMIZE_THRESHOLD, 1000);
    }

    // ── file_metadata SQL constant tests (Task 6) ──

    #[test]
    fn test_file_metadata_create_sql_valid() {
        assert!(CREATE_FILE_METADATA_SQL.contains("CREATE TABLE"));
        assert!(CREATE_FILE_METADATA_SQL.contains("file_metadata"));
        assert!(CREATE_FILE_METADATA_SQL.contains("file_id INTEGER PRIMARY KEY"));
        assert!(CREATE_FILE_METADATA_SQL.contains("tenant_id TEXT NOT NULL"));
        assert!(CREATE_FILE_METADATA_SQL.contains("branch TEXT"));
        assert!(CREATE_FILE_METADATA_SQL.contains("file_path TEXT NOT NULL"));
    }

    #[test]
    fn test_file_metadata_indexes_idempotent() {
        for sql in CREATE_FILE_METADATA_INDEXES_SQL {
            assert!(sql.contains("IF NOT EXISTS"), "Missing IF NOT EXISTS: {}", sql);
        }
    }

    #[test]
    fn test_file_metadata_index_count() {
        assert_eq!(CREATE_FILE_METADATA_INDEXES_SQL.len(), 2);
    }

    #[test]
    fn test_upsert_file_metadata_sql_valid() {
        assert!(UPSERT_FILE_METADATA_SQL.contains("INSERT INTO file_metadata"));
        assert!(UPSERT_FILE_METADATA_SQL.contains("ON CONFLICT(file_id)"));
        assert!(UPSERT_FILE_METADATA_SQL.contains("DO UPDATE SET"));
        // v5 columns
        assert!(UPSERT_FILE_METADATA_SQL.contains("base_point"));
        assert!(UPSERT_FILE_METADATA_SQL.contains("relative_path"));
        assert!(UPSERT_FILE_METADATA_SQL.contains("file_hash"));
    }

    #[test]
    fn test_alter_file_metadata_v5_sql_valid() {
        assert_eq!(ALTER_FILE_METADATA_V5_SQL.len(), 3);
        assert!(ALTER_FILE_METADATA_V5_SQL[0].contains("base_point"));
        assert!(ALTER_FILE_METADATA_V5_SQL[1].contains("relative_path"));
        assert!(ALTER_FILE_METADATA_V5_SQL[2].contains("file_hash"));
    }

    #[test]
    fn test_file_metadata_base_point_index_sql_valid() {
        assert!(CREATE_FILE_METADATA_BASE_POINT_INDEX_SQL.contains("IF NOT EXISTS"));
        assert!(CREATE_FILE_METADATA_BASE_POINT_INDEX_SQL.contains("base_point"));
    }

    #[test]
    fn test_delete_file_metadata_sql_valid() {
        assert!(DELETE_FILE_METADATA_SQL.contains("DELETE FROM file_metadata"));
        assert!(DELETE_FILE_METADATA_SQL.contains("file_id = ?1"));
    }

    #[test]
    fn test_delete_file_metadata_by_tenant_sql_valid() {
        assert!(DELETE_FILE_METADATA_BY_TENANT_SQL.contains("DELETE FROM file_metadata"));
        assert!(DELETE_FILE_METADATA_BY_TENANT_SQL.contains("tenant_id = ?1"));
    }

    #[test]
    fn test_fts5_search_by_project_sql_valid() {
        assert!(FTS5_SEARCH_BY_PROJECT_SQL.contains("MATCH ?1"));
        assert!(FTS5_SEARCH_BY_PROJECT_SQL.contains("fm.tenant_id = ?2"));
        assert!(FTS5_SEARCH_BY_PROJECT_SQL.contains("file_metadata fm"));
    }

    #[test]
    fn test_fts5_search_by_project_branch_sql_valid() {
        assert!(FTS5_SEARCH_BY_PROJECT_BRANCH_SQL.contains("MATCH ?1"));
        assert!(FTS5_SEARCH_BY_PROJECT_BRANCH_SQL.contains("fm.tenant_id = ?2"));
        assert!(FTS5_SEARCH_BY_PROJECT_BRANCH_SQL.contains("fm.branch = ?3"));
    }

    #[test]
    fn test_fts5_search_by_path_prefix_sql_valid() {
        assert!(FTS5_SEARCH_BY_PATH_PREFIX_SQL.contains("MATCH ?1"));
        assert!(FTS5_SEARCH_BY_PATH_PREFIX_SQL.contains("fm.file_path LIKE ?2"));
    }

    #[test]
    fn test_fts5_search_by_project_path_sql_valid() {
        assert!(FTS5_SEARCH_BY_PROJECT_PATH_SQL.contains("MATCH ?1"));
        assert!(FTS5_SEARCH_BY_PROJECT_PATH_SQL.contains("fm.tenant_id = ?2"));
        assert!(FTS5_SEARCH_BY_PROJECT_PATH_SQL.contains("fm.file_path LIKE ?3"));
    }
}

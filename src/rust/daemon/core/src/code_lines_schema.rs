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
}

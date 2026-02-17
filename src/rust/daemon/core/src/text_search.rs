//! Exact String Search on FTS5 (Task 53)
//!
//! Provides exact substring search over code_lines using FTS5 trigram index
//! for candidate selection and LIKE verification for exact match.
//!
//! ## Architecture
//!
//! 1. **FTS5 trigram MATCH** — pre-filter using the trigram index (fast, ~O(1) per term)
//! 2. **LIKE verification** — exact match filter on `code_lines.content` (necessary
//!    because trigram matching is necessary-but-not-sufficient for exact substring)
//! 3. **ROW_NUMBER()** — derives 1-based line numbers from gap-based `seq` ordering
//! 4. **file_metadata JOIN** — scopes results by project/branch/path without
//!    cross-database JOINs (file_metadata is denormalized in search.db)
//!
//! ## FTS5 Trigram Pattern Escaping
//!
//! FTS5 trigram tokenizer treats `"` as special. All patterns must be double-quote
//! wrapped for exact phrase matching. Internal double quotes are escaped as `""`.
//! Patterns shorter than 3 characters cannot use the trigram index and fall back
//! to a full table scan with LIKE only.

use sqlx::Row;
use tracing::debug;

use crate::search_db::{SearchDbManager, SearchDbError};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// A single search match in a code file.
#[derive(Debug, Clone)]
pub struct SearchMatch {
    /// line_id from code_lines (primary key).
    pub line_id: i64,
    /// file_id reference to tracked_files.
    pub file_id: i64,
    /// 1-based line number within the file.
    pub line_number: i64,
    /// Full content of the matching line.
    pub content: String,
    /// File path from file_metadata.
    pub file_path: String,
    /// Tenant ID from file_metadata.
    pub tenant_id: String,
    /// Branch from file_metadata (may be empty).
    pub branch: Option<String>,
}

/// Search options for scoping and filtering.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Scope to a specific project (tenant_id).
    pub tenant_id: Option<String>,
    /// Scope to a specific branch.
    pub branch: Option<String>,
    /// Filter by file path prefix (e.g., "src/").
    pub path_prefix: Option<String>,
    /// Case-insensitive search (default: false = case-sensitive).
    pub case_insensitive: bool,
    /// Maximum number of results to return (0 = unlimited).
    pub max_results: usize,
}

/// Aggregated search results.
#[derive(Debug, Clone)]
pub struct SearchResults {
    /// The search pattern used.
    pub pattern: String,
    /// Matching lines.
    pub matches: Vec<SearchMatch>,
    /// Whether results were truncated by max_results.
    pub truncated: bool,
    /// Time spent in the FTS5 query (milliseconds).
    pub query_time_ms: u64,
}

// ---------------------------------------------------------------------------
// FTS5 pattern escaping
// ---------------------------------------------------------------------------

/// Escape a search pattern for FTS5 trigram MATCH.
///
/// FTS5 trigram tokenizer requires patterns to be double-quote wrapped.
/// Internal double quotes are escaped as `""`.
///
/// Returns `None` if the pattern is shorter than 3 characters (trigram minimum).
pub fn escape_fts5_pattern(pattern: &str) -> Option<String> {
    if pattern.len() < 3 {
        return None;
    }
    let escaped = pattern.replace('"', "\"\"");
    Some(format!("\"{}\"", escaped))
}

/// Escape a LIKE pattern — escape `%`, `_`, and `\` for exact substring match.
pub fn escape_like_pattern(pattern: &str) -> String {
    pattern
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

// ---------------------------------------------------------------------------
// Search implementation
// ---------------------------------------------------------------------------

/// Search code_lines for an exact substring pattern.
///
/// Uses a two-phase approach:
/// 1. FTS5 trigram MATCH for fast candidate selection
/// 2. LIKE verification for exact substring match
///
/// For patterns shorter than 3 characters, falls back to LIKE-only scan.
pub async fn search_exact(
    search_db: &SearchDbManager,
    pattern: &str,
    options: &SearchOptions,
) -> Result<SearchResults, SearchDbError> {
    let start = std::time::Instant::now();

    if pattern.is_empty() {
        return Ok(SearchResults {
            pattern: pattern.to_string(),
            matches: vec![],
            truncated: false,
            query_time_ms: 0,
        });
    }

    let fts5_pattern = escape_fts5_pattern(pattern);

    // Build the SQL query dynamically based on options
    let (sql, use_fts) = build_search_query(&fts5_pattern, options);

    debug!(
        "FTS5 search: pattern={:?}, fts5={:?}, use_fts={}, tenant={:?}, branch={:?}, path_prefix={:?}",
        pattern, fts5_pattern, use_fts,
        options.tenant_id, options.branch, options.path_prefix,
    );

    let pool = search_db.pool();
    let mut query = sqlx::query(&sql);

    // Bind parameters in order
    if use_fts {
        query = query.bind(fts5_pattern.as_ref().unwrap());
    }

    // INSTR pattern — case insensitive uses LOWER() in SQL so bind lowercase
    if options.case_insensitive {
        query = query.bind(pattern.to_lowercase());
    } else {
        query = query.bind(pattern);
    }

    // Bind optional scope filters
    if let Some(ref tid) = options.tenant_id {
        query = query.bind(tid);
    }
    if let Some(ref branch) = options.branch {
        query = query.bind(branch);
    }
    if let Some(ref prefix) = options.path_prefix {
        query = query.bind(format!("{}%", prefix));
    }

    let rows = query.fetch_all(pool).await?;

    let max_results = if options.max_results > 0 {
        options.max_results
    } else {
        usize::MAX
    };

    let truncated = rows.len() > max_results;
    let matches: Vec<SearchMatch> = rows
        .iter()
        .take(max_results)
        .map(|row| SearchMatch {
            line_id: row.get("line_id"),
            file_id: row.get("file_id"),
            line_number: row.get("line_number"),
            content: row.get("content"),
            file_path: row.get("file_path"),
            tenant_id: row.get("tenant_id"),
            branch: row.get("branch"),
        })
        .collect();

    let query_time_ms = start.elapsed().as_millis() as u64;

    debug!(
        "FTS5 search complete: {} matches in {}ms (pattern={:?}, truncated={})",
        matches.len(), query_time_ms, pattern, truncated
    );

    Ok(SearchResults {
        pattern: pattern.to_string(),
        matches,
        truncated,
        query_time_ms,
    })
}

/// Build the search SQL query based on options.
///
/// Returns (sql_string, uses_fts5). When patterns are < 3 chars, FTS5
/// trigram index cannot be used and we fall back to LIKE-only scan.
///
/// ## Query structure
///
/// Uses a two-CTE approach to correctly derive absolute line numbers:
/// 1. `all_lines` — numbers ALL lines in each file using `ROW_NUMBER()`
/// 2. `matching` — filters to lines matching the search pattern (FTS5 + LIKE)
///
/// This ensures line numbers reflect the true position in the file, not the
/// position among matching lines.
fn build_search_query(
    fts5_pattern: &Option<String>,
    options: &SearchOptions,
) -> (String, bool) {
    let use_fts = fts5_pattern.is_some();

    // CTE 1: Number ALL lines in each file
    let all_lines_cte = r#"
WITH all_lines AS (
    SELECT
        cl.line_id,
        cl.file_id,
        ROW_NUMBER() OVER (PARTITION BY cl.file_id ORDER BY cl.seq) AS line_number,
        cl.content
    FROM code_lines cl
)"#;

    // CTE 2: Filter to matching lines using FTS5 + exact match verification
    //
    // Case-sensitive: use INSTR() for exact substring (SQLite LIKE is case-insensitive by default)
    // Case-insensitive: use INSTR(LOWER(), LOWER()) for case-folded comparison
    let matching_cte = if use_fts {
        if options.case_insensitive {
            r#",
matching AS (
    SELECT al.line_id, al.file_id, al.line_number, al.content
    FROM all_lines al
    JOIN code_lines_fts fts ON al.line_id = fts.rowid
    WHERE fts.content MATCH ?1 AND INSTR(LOWER(al.content), ?2) > 0
)"#
        } else {
            r#",
matching AS (
    SELECT al.line_id, al.file_id, al.line_number, al.content
    FROM all_lines al
    JOIN code_lines_fts fts ON al.line_id = fts.rowid
    WHERE fts.content MATCH ?1 AND INSTR(al.content, ?2) > 0
)"#
        }
    } else {
        // INSTR-only fallback for short patterns (< 3 chars)
        if options.case_insensitive {
            r#",
matching AS (
    SELECT al.line_id, al.file_id, al.line_number, al.content
    FROM all_lines al
    WHERE INSTR(LOWER(al.content), ?1) > 0
)"#
        } else {
            r#",
matching AS (
    SELECT al.line_id, al.file_id, al.line_number, al.content
    FROM all_lines al
    WHERE INSTR(al.content, ?1) > 0
)"#
        }
    };

    // Main query joins with file_metadata for scoping
    let mut sql = format!(
        "{}{}
SELECT m.line_id, m.file_id, m.line_number, m.content,
       fm.file_path, fm.tenant_id, fm.branch
FROM matching m
JOIN file_metadata fm ON m.file_id = fm.file_id
WHERE 1=1",
        all_lines_cte, matching_cte
    );

    // The bind index depends on whether FTS5 is used
    // FTS5 mode: ?1 = fts5_pattern, ?2 = like_pattern, ?3+ = scope filters
    // LIKE-only mode: ?1 = like_pattern, ?2+ = scope filters
    let mut next_param = if use_fts { 3 } else { 2 };

    if options.tenant_id.is_some() {
        sql.push_str(&format!(" AND fm.tenant_id = ?{}", next_param));
        next_param += 1;
    }
    if options.branch.is_some() {
        sql.push_str(&format!(" AND fm.branch = ?{}", next_param));
        next_param += 1;
    }
    if options.path_prefix.is_some() {
        sql.push_str(&format!(" AND fm.file_path LIKE ?{} ESCAPE '\\'", next_param));
        // next_param += 1; // not needed — last parameter
    }

    sql.push_str("\nORDER BY m.file_id, m.line_number");

    (sql, use_fts)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_lines_schema::{initial_seq, UPSERT_FILE_METADATA_SQL};

    async fn setup_search_db() -> (tempfile::TempDir, SearchDbManager) {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test_search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();
        (tmp, manager)
    }

    async fn insert_file_content(
        db: &SearchDbManager,
        file_id: i64,
        lines: &[&str],
        tenant_id: &str,
        branch: Option<&str>,
        file_path: &str,
    ) {
        let pool = db.pool();

        // Insert lines
        for (i, line) in lines.iter().enumerate() {
            let seq = initial_seq(i);
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (?1, ?2, ?3)")
                .bind(file_id)
                .bind(seq)
                .bind(*line)
                .execute(pool)
                .await
                .unwrap();
        }

        // Insert file_metadata
        sqlx::query(UPSERT_FILE_METADATA_SQL)
            .bind(file_id)
            .bind(tenant_id)
            .bind(branch)
            .bind(file_path)
            .execute(pool)
            .await
            .unwrap();

        // Rebuild FTS
        db.rebuild_fts().await.unwrap();
    }

    // ── Pattern escaping tests ──

    #[test]
    fn test_escape_fts5_pattern_basic() {
        assert_eq!(
            escape_fts5_pattern("println"),
            Some("\"println\"".to_string())
        );
    }

    #[test]
    fn test_escape_fts5_pattern_with_quotes() {
        assert_eq!(
            escape_fts5_pattern("say \"hello\""),
            Some("\"say \"\"hello\"\"\"".to_string())
        );
    }

    #[test]
    fn test_escape_fts5_pattern_short() {
        assert_eq!(escape_fts5_pattern("fn"), None);
        assert_eq!(escape_fts5_pattern("a"), None);
        assert_eq!(escape_fts5_pattern(""), None);
    }

    #[test]
    fn test_escape_fts5_pattern_exactly_3() {
        assert_eq!(
            escape_fts5_pattern("abc"),
            Some("\"abc\"".to_string())
        );
    }

    #[test]
    fn test_escape_like_pattern() {
        assert_eq!(escape_like_pattern("hello"), "hello");
        assert_eq!(escape_like_pattern("100%"), "100\\%");
        assert_eq!(escape_like_pattern("under_score"), "under\\_score");
        assert_eq!(escape_like_pattern("back\\slash"), "back\\\\slash");
    }

    // ── Search query building tests ──

    #[test]
    fn test_build_query_with_fts5() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions::default();
        let (sql, use_fts) = build_search_query(&pattern, &options);
        assert!(use_fts);
        assert!(sql.contains("MATCH ?1"));
        assert!(sql.contains("INSTR(al.content, ?2)"));
        assert!(sql.contains("all_lines"));
        assert!(sql.contains("matching"));
    }

    #[test]
    fn test_build_query_without_fts5() {
        let options = SearchOptions::default();
        let (sql, use_fts) = build_search_query(&None, &options);
        assert!(!use_fts);
        assert!(!sql.contains("MATCH"));
        assert!(sql.contains("INSTR(al.content, ?1)"));
    }

    #[test]
    fn test_build_query_with_tenant_filter() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions {
            tenant_id: Some("proj1".to_string()),
            ..Default::default()
        };
        let (sql, _) = build_search_query(&pattern, &options);
        assert!(sql.contains("fm.tenant_id = ?3"));
    }

    #[test]
    fn test_build_query_case_insensitive() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions {
            case_insensitive: true,
            ..Default::default()
        };
        let (sql, _) = build_search_query(&pattern, &options);
        assert!(sql.contains("INSTR(LOWER(al.content), ?2)"));
    }

    #[test]
    fn test_build_query_with_all_filters() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions {
            tenant_id: Some("proj1".to_string()),
            branch: Some("main".to_string()),
            path_prefix: Some("src/".to_string()),
            ..Default::default()
        };
        let (sql, _) = build_search_query(&pattern, &options);
        assert!(sql.contains("fm.tenant_id = ?3"));
        assert!(sql.contains("fm.branch = ?4"));
        assert!(sql.contains("fm.file_path LIKE ?5"));
    }

    // ── Integration tests (FTS5 with real SQLite) ──

    #[tokio::test]
    async fn test_search_exact_basic() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn main() {", "    println!(\"hello\");", "}"],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        let results = search_exact(&db, "println", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 2);
        assert_eq!(results.matches[0].file_path, "src/main.rs");
        assert!(results.matches[0].content.contains("println"));
    }

    #[tokio::test]
    async fn test_search_exact_multiple_matches() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "use std::io;",
                "fn read() { io::stdin() }",
                "fn write() { io::stdout() }",
                "fn other() {}",
            ],
            "proj1",
            Some("main"),
            "src/io.rs",
        )
        .await;

        let results = search_exact(&db, "io::", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 2);
        assert_eq!(results.matches[0].line_number, 2);
        assert_eq!(results.matches[1].line_number, 3);
    }

    #[tokio::test]
    async fn test_search_exact_no_match() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn main() {}", "let x = 42;"],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        let results = search_exact(&db, "nonexistent_function", &SearchOptions::default())
            .await
            .unwrap();

        assert!(results.matches.is_empty());
    }

    #[tokio::test]
    async fn test_search_exact_case_sensitive() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn Main() {}", "fn main() {}"],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        // Case-sensitive: only "Main" matches
        let results = search_exact(
            &db,
            "Main",
            &SearchOptions::default(),
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 1);
    }

    #[tokio::test]
    async fn test_search_exact_case_insensitive() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn Main() {}", "fn main() {}"],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        // Case-insensitive: both match
        let results = search_exact(
            &db,
            "main",
            &SearchOptions {
                case_insensitive: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 2);
    }

    #[tokio::test]
    async fn test_search_exact_scoped_by_tenant() {
        let (_tmp, db) = setup_search_db().await;

        // Two different projects
        insert_file_content(
            &db,
            1,
            &["fn hello() {}", "fn world() {}"],
            "proj1",
            Some("main"),
            "src/a.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["fn hello() {}", "fn goodbye() {}"],
            "proj2",
            Some("main"),
            "src/b.rs",
        )
        .await;

        // Search only in proj1
        let results = search_exact(
            &db,
            "hello",
            &SearchOptions {
                tenant_id: Some("proj1".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].tenant_id, "proj1");
    }

    #[tokio::test]
    async fn test_search_exact_scoped_by_path_prefix() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db,
            1,
            &["fn func_a() {}"],
            "proj1",
            Some("main"),
            "src/module/a.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["fn func_a() {}"],
            "proj1",
            Some("main"),
            "tests/test_a.rs",
        )
        .await;

        // Search only in src/
        let results = search_exact(
            &db,
            "func_a",
            &SearchOptions {
                path_prefix: Some("src/".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].file_path, "src/module/a.rs");
    }

    #[tokio::test]
    async fn test_search_exact_max_results() {
        let (_tmp, db) = setup_search_db().await;

        let lines: Vec<String> = (0..20)
            .map(|i| format!("let item_{} = process();", i))
            .collect();
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();

        insert_file_content(&db, 1, &line_refs, "proj1", Some("main"), "src/many.rs")
            .await;

        let results = search_exact(
            &db,
            "process",
            &SearchOptions {
                max_results: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 5);
        assert!(results.truncated);
    }

    #[tokio::test]
    async fn test_search_exact_empty_pattern() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn main() {}"],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        let results = search_exact(&db, "", &SearchOptions::default())
            .await
            .unwrap();

        assert!(results.matches.is_empty());
    }

    #[tokio::test]
    async fn test_search_exact_special_characters() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "let pct = 100%;",
                "let _under = true;",
                "let path = \"C:\\\\Windows\";",
            ],
            "proj1",
            Some("main"),
            "src/special.rs",
        )
        .await;

        // Search for literal %
        let results = search_exact(&db, "100%", &SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 1);
    }

    #[tokio::test]
    async fn test_search_exact_short_pattern_fallback() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn a() {}", "fn b() {}", "fn ab() {}"],
            "proj1",
            Some("main"),
            "src/short.rs",
        )
        .await;

        // "fn" is only 2 chars — falls back to LIKE-only scan
        let results = search_exact(&db, "fn", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 3);
    }

    #[tokio::test]
    async fn test_search_exact_across_files() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db,
            1,
            &["fn handler() {}", "  // process request"],
            "proj1",
            Some("main"),
            "src/api.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["fn worker() {}", "  // process job"],
            "proj1",
            Some("main"),
            "src/worker.rs",
        )
        .await;

        let results = search_exact(&db, "process", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 2);
        // Results should be ordered by file_id, then line_number
        assert_eq!(results.matches[0].file_path, "src/api.rs");
        assert_eq!(results.matches[1].file_path, "src/worker.rs");
    }

    #[tokio::test]
    async fn test_search_exact_line_numbers_correct() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "// line 1",
                "// line 2",
                "fn target() {}",
                "// line 4",
                "fn target_two() {}",
            ],
            "proj1",
            Some("main"),
            "src/lines.rs",
        )
        .await;

        let results = search_exact(&db, "target", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 2);
        assert_eq!(results.matches[0].line_number, 3);
        assert_eq!(results.matches[1].line_number, 5);
    }

    #[tokio::test]
    async fn test_search_exact_branch_filter() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db,
            1,
            &["fn feature() {}"],
            "proj1",
            Some("main"),
            "src/a.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["fn feature() {}"],
            "proj1",
            Some("dev"),
            "src/a.rs",
        )
        .await;

        let results = search_exact(
            &db,
            "feature",
            &SearchOptions {
                branch: Some("dev".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].branch, Some("dev".to_string()));
    }

    #[tokio::test]
    async fn test_search_exact_pattern_with_double_quotes() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "let msg = \"hello world\";",
                "let other = 42;",
            ],
            "proj1",
            Some("main"),
            "src/quotes.rs",
        )
        .await;

        // Search for a string containing double quotes
        let results = search_exact(&db, "\"hello world\"", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert!(results.matches[0].content.contains("\"hello world\""));
    }
}

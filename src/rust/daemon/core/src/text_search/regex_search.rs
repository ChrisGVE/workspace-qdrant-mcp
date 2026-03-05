//! Regex search over code_lines with trigram acceleration (Task 54).
//!
//! Extracts literal substrings from regex patterns, uses FTS5 trigram MATCH
//! for candidate pre-filtering, then verifies with Rust's regex engine.

use futures::TryStreamExt;
use sqlx::Row;
use tracing::debug;

use super::escaping::{compile_glob_matcher, resolve_path_filter};
use super::exact_search::attach_context_lines;
use super::regex_parser::{build_fts5_query, extract_literals_from_regex};
use super::types::{SearchMatch, SearchOptions, SearchResults};
use crate::search_db::{SearchDbError, SearchDbManager};

/// Search code_lines using a regex pattern with trigram acceleration.
///
/// ## Strategy
///
/// 1. Extract literal substrings from the regex for FTS5 pre-filtering
/// 2. Lightweight FTS5-only probe: if candidates exceed threshold, delegate
///    to `grep-searcher` for SIMD-accelerated file scanning
/// 3. Otherwise stream FTS5 candidates, verify with regex in Rust
/// 4. If no extractable literals: full table scan with regex in Rust
///
/// Case-insensitive mode uses `regex::RegexBuilder::case_insensitive(true)`.
/// When `path_glob` is set, applies glob filtering in Rust after SQL results.
pub async fn search_regex(
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
            search_engine: "fts5".to_string(),
        });
    }

    let literals = extract_literals_from_regex(pattern);
    let fts5_query = build_fts5_query(&literals);
    let (glob_pattern, effective_options) = resolve_path_filter(options);
    let glob_matcher = glob_pattern
        .as_deref()
        .map(compile_glob_matcher)
        .transpose()?;
    let re = regex::RegexBuilder::new(pattern)
        .case_insensitive(options.case_insensitive)
        .build()
        .map_err(|e| SearchDbError::InvalidPattern(format!("{}", e)))?;

    debug!(
        "Regex search: pattern={:?}, literals={:?}, fts5_query={:?}, tenant={:?}, path_glob={:?}",
        pattern, literals, fts5_query, effective_options.tenant_id, options.path_glob,
    );

    let pool = search_db.pool();

    // Grep fallback: if FTS5 candidate count exceeds threshold, delegate to
    // grep-searcher for SIMD-accelerated file scanning.
    if let Some(ref fts_q) = fts5_query {
        let threshold = crate::grep_search::GREP_FALLBACK_THRESHOLD;
        if fts5_exceeds_threshold(pool, fts_q, threshold).await? {
            debug!(
                "grep fallback: FTS5 candidates exceed threshold ({}), using grep",
                threshold
            );
            return crate::grep_search::search_regex_via_grep(search_db, pattern, options).await;
        }
    }

    let (matches, truncated, candidates_scanned) = collect_regex_matches(
        pool,
        &fts5_query,
        &effective_options,
        &re,
        glob_matcher.as_ref(),
        options.max_results,
    )
    .await?;

    let mut matches = matches;
    if options.context_lines > 0 {
        attach_context_lines(search_db, &mut matches, options.context_lines).await?;
    }

    let query_time_ms = start.elapsed().as_millis() as u64;
    debug!(
        "Regex search complete: {} matches in {}ms (pattern={:?}, fts_candidates={}, truncated={})",
        matches.len(),
        query_time_ms,
        pattern,
        candidates_scanned,
        truncated
    );

    Ok(SearchResults {
        pattern: pattern.to_string(),
        matches,
        truncated,
        query_time_ms,
        search_engine: "fts5".to_string(),
    })
}

/// Build, bind, and execute the SQL query, then stream rows applying regex
/// verification and optional glob filtering.
///
/// Returns `(matches, truncated, candidates_scanned)`.
async fn collect_regex_matches(
    pool: &sqlx::SqlitePool,
    fts5_query: &Option<String>,
    options: &SearchOptions,
    re: &regex::Regex,
    glob_matcher: Option<&impl Fn(&str) -> bool>,
    max_results_hint: usize,
) -> Result<(Vec<SearchMatch>, bool, usize), SearchDbError> {
    let (sql, use_fts) = build_regex_search_query(fts5_query, options);
    let mut query = sqlx::query(&sql);

    if use_fts {
        query = query.bind(fts5_query.as_ref().unwrap());
    }
    if let Some(ref tid) = options.tenant_id {
        query = query.bind(tid);
    }
    if let Some(ref branch) = options.branch {
        query = query.bind(branch);
    }
    if let Some(ref prefix) = options.path_prefix {
        query = query.bind(format!("{}%", prefix));
    }

    let max_results = if max_results_hint > 0 {
        max_results_hint
    } else {
        usize::MAX
    };
    let mut stream = query.fetch(pool);
    let mut matches = Vec::new();
    let mut truncated = false;
    let mut candidates_scanned: usize = 0;

    while let Some(row) = stream.try_next().await? {
        candidates_scanned += 1;
        let file_path: String = row.get("file_path");

        if let Some(matcher) = glob_matcher {
            if !matcher(&file_path) {
                continue;
            }
        }

        let content: String = row.get("content");
        if re.is_match(&content) {
            matches.push(SearchMatch {
                line_id: row.get("line_id"),
                file_id: row.get("file_id"),
                line_number: row.get("line_number"),
                content,
                file_path,
                tenant_id: row.get("tenant_id"),
                branch: row.get("branch"),
                context_before: vec![],
                context_after: vec![],
            });
            if matches.len() >= max_results {
                truncated = true;
                break;
            }
        }
    }
    drop(stream);

    Ok((matches, truncated, candidates_scanned))
}

/// Lightweight FTS5-only candidate count probe.
async fn fts5_exceeds_threshold(
    pool: &sqlx::SqlitePool,
    fts5_query: &str,
    threshold: i64,
) -> Result<bool, SearchDbError> {
    let result: Option<i64> = sqlx::query_scalar(
        "SELECT rowid FROM code_lines_fts WHERE content MATCH ?1 LIMIT 1 OFFSET ?2",
    )
    .bind(fts5_query)
    .bind(threshold)
    .fetch_optional(pool)
    .await?;
    Ok(result.is_some())
}

/// Build SQL query for regex search.
fn build_regex_search_query(
    fts5_query: &Option<String>,
    options: &SearchOptions,
) -> (String, bool) {
    let use_fts = fts5_query.is_some();

    let candidates_cte = if use_fts {
        r#"
WITH candidates AS (
    SELECT cl.line_id, cl.file_id, cl.line_number, cl.content
    FROM code_lines cl
    JOIN code_lines_fts fts ON cl.line_id = fts.rowid
    WHERE fts.content MATCH ?1
)"#
        .to_string()
    } else {
        r#"
WITH candidates AS (
    SELECT cl.line_id, cl.file_id, cl.line_number, cl.content
    FROM code_lines cl
)"#
        .to_string()
    };

    let mut sql = format!(
        "{}
SELECT c.line_id, c.file_id, c.line_number, c.content,
       fm.file_path, fm.tenant_id, fm.branch
FROM candidates c
JOIN file_metadata fm ON c.file_id = fm.file_id
WHERE 1=1",
        candidates_cte
    );

    let mut next_param = if use_fts { 2 } else { 1 };

    if options.tenant_id.is_some() {
        sql.push_str(&format!(" AND fm.tenant_id = ?{}", next_param));
        next_param += 1;
    }
    if options.branch.is_some() {
        sql.push_str(&format!(" AND fm.branch = ?{}", next_param));
        next_param += 1;
    }
    if options.path_prefix.is_some() {
        sql.push_str(&format!(
            " AND fm.file_path LIKE ?{} ESCAPE '\\'",
            next_param
        ));
    }

    sql.push_str("\nORDER BY c.file_id, c.line_number");

    (sql, use_fts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_lines_schema::{initial_seq, UPSERT_FILE_METADATA_SQL};
    use crate::search_db::SearchDbManager;

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
        for (i, line) in lines.iter().enumerate() {
            let seq = initial_seq(i);
            let line_number = (i + 1) as i64;
            sqlx::query("INSERT INTO code_lines (file_id, seq, content, line_number) VALUES (?1, ?2, ?3, ?4)")
                .bind(file_id)
                .bind(seq)
                .bind(*line)
                .bind(line_number)
                .execute(pool)
                .await
                .unwrap();
        }
        sqlx::query(UPSERT_FILE_METADATA_SQL)
            .bind(file_id)
            .bind(tenant_id)
            .bind(branch)
            .bind(file_path)
            .bind(None::<&str>)
            .bind(None::<&str>)
            .bind(None::<&str>)
            .execute(pool)
            .await
            .unwrap();
        db.rebuild_fts().await.unwrap();
    }

    #[test]
    fn test_build_regex_query_with_fts() {
        let fts_query = Some("\"async\" OR \"await\"".to_string());
        let options = SearchOptions::default();
        let (sql, use_fts) = build_regex_search_query(&fts_query, &options);
        assert!(use_fts);
        assert!(sql.contains("MATCH ?1"));
        assert!(sql.contains("candidates"));
        assert!(sql.contains("cl.line_number"));
        assert!(!sql.contains("SELECT COUNT(*)"));
    }

    #[test]
    fn test_build_regex_query_without_fts() {
        let options = SearchOptions::default();
        let (sql, use_fts) = build_regex_search_query(&None, &options);
        assert!(!use_fts);
        assert!(!sql.contains("MATCH"));
        assert!(sql.contains("candidates"));
        assert!(!sql.contains("all_lines"));
    }

    #[test]
    fn test_build_regex_query_with_scope_filters() {
        let fts_query = Some("\"test\"".to_string());
        let options = SearchOptions {
            tenant_id: Some("proj1".to_string()),
            branch: Some("main".to_string()),
            path_prefix: Some("src/".to_string()),
            ..Default::default()
        };
        let (sql, _) = build_regex_search_query(&fts_query, &options);
        assert!(sql.contains("fm.tenant_id = ?2"));
        assert!(sql.contains("fm.branch = ?3"));
        assert!(sql.contains("fm.file_path LIKE ?4"));
    }

    #[tokio::test]
    async fn test_search_regex_basic() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "fn main() {",
                "    let x = 42;",
                "    println!(\"hello\");",
                "}",
            ],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;
        let results = search_regex(&db, "fn\\s+main", &SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 1);
        assert!(results.matches[0].content.contains("fn main"));
    }

    #[tokio::test]
    async fn test_search_regex_wildcard() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "async fn process_request() {}",
                "fn process_response() {}",
                "fn handle_error() {}",
            ],
            "proj1",
            Some("main"),
            "src/handler.rs",
        )
        .await;
        let results = search_regex(&db, "fn process_\\w+", &SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(results.matches.len(), 2);
        assert!(results.matches[0].content.contains("process_request"));
        assert!(results.matches[1].content.contains("process_response"));
    }

    #[tokio::test]
    async fn test_search_regex_no_trigrams_fallback() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["a", "ab", "abc", "abcd", "x"],
            "proj1",
            Some("main"),
            "src/short.rs",
        )
        .await;
        let results = search_regex(&db, "^.{3}$", &SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].content, "abc");
    }

    #[tokio::test]
    async fn test_search_regex_case_insensitive() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn Main() {}", "fn main() {}", "fn MAIN() {}"],
            "proj1",
            Some("main"),
            "src/case.rs",
        )
        .await;
        let results = search_regex(
            &db,
            "fn main",
            &SearchOptions {
                case_insensitive: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(results.matches.len(), 3);
    }

    #[tokio::test]
    async fn test_search_regex_scoped_by_tenant() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["pub fn hello() {}"],
            "proj1",
            Some("main"),
            "src/a.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["pub fn hello() {}"],
            "proj2",
            Some("main"),
            "src/b.rs",
        )
        .await;
        let results = search_regex(
            &db,
            "pub fn \\w+",
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
    async fn test_search_regex_alternation() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "let future = async { 42 };",
                "let result = await!(future);",
                "let sync_val = 10;",
            ],
            "proj1",
            Some("main"),
            "src/async.rs",
        )
        .await;
        let results = search_regex(&db, "async|await", &SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(results.matches.len(), 2);
    }

    #[tokio::test]
    async fn test_search_regex_max_results() {
        let (_tmp, db) = setup_search_db().await;
        let lines: Vec<String> = (0..20)
            .map(|i| format!("let item_{} = process();", i))
            .collect();
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        insert_file_content(&db, 1, &line_refs, "proj1", Some("main"), "src/many.rs").await;
        let results = search_regex(
            &db,
            "item_\\d+",
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
    async fn test_search_regex_empty_pattern() {
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
        let results = search_regex(&db, "", &SearchOptions::default())
            .await
            .unwrap();
        assert!(results.matches.is_empty());
    }

    #[tokio::test]
    async fn test_search_regex_invalid_pattern() {
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
        let result = search_regex(&db, "[invalid", &SearchOptions::default()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_regex_line_numbers_correct() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "// comment",
                "use std::io;",
                "// comment",
                "fn read() -> io::Result<()> {",
                "// comment",
            ],
            "proj1",
            Some("main"),
            "src/io.rs",
        )
        .await;
        let results = search_regex(&db, "fn \\w+\\(\\)", &SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 4);
    }

    #[tokio::test]
    async fn test_search_regex_word_boundary() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "class MyClass {}",
                "subclass OtherClass {}",
                "let classified = true;",
            ],
            "proj1",
            Some("main"),
            "src/class.rs",
        )
        .await;
        let results = search_regex(&db, "\\bclass\\b", &SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 1);
    }

    #[tokio::test]
    async fn test_search_regex_across_files() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["pub struct Config {}", "impl Config {}"],
            "proj1",
            Some("main"),
            "src/config.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["pub struct Handler {}", "impl Handler {}"],
            "proj1",
            Some("main"),
            "src/handler.rs",
        )
        .await;
        let results = search_regex(&db, "pub struct \\w+", &SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(results.matches.len(), 2);
        assert_eq!(results.matches[0].file_path, "src/config.rs");
        assert_eq!(results.matches[1].file_path, "src/handler.rs");
    }

    #[tokio::test]
    async fn test_search_regex_path_prefix_filter() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn test_func() {}"],
            "proj1",
            Some("main"),
            "src/lib.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["fn test_func() {}"],
            "proj1",
            Some("main"),
            "tests/test.rs",
        )
        .await;
        let results = search_regex(
            &db,
            "fn test_\\w+",
            &SearchOptions {
                path_prefix: Some("src/".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].file_path, "src/lib.rs");
    }

    #[tokio::test]
    async fn test_search_regex_with_path_glob() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["pub fn handler() {}"],
            "proj1",
            Some("main"),
            "src/api.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["pub fn handler() {}"],
            "proj1",
            Some("main"),
            "src/api.ts",
        )
        .await;
        insert_file_content(
            &db,
            3,
            &["pub fn handler() {}"],
            "proj1",
            Some("main"),
            "tests/test_api.rs",
        )
        .await;
        let results = search_regex(
            &db,
            "pub fn \\w+",
            &SearchOptions {
                path_glob: Some("src/**/*.rs".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].file_path, "src/api.rs");
    }

    #[tokio::test]
    async fn test_context_lines_with_regex() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "use std::io;",
                "fn read_file() {",
                "    let data = read();",
                "}",
            ],
            "proj1",
            Some("main"),
            "src/io.rs",
        )
        .await;
        let results = search_regex(
            &db,
            "fn \\w+\\(",
            &SearchOptions {
                context_lines: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(results.matches.len(), 1);
        let m = &results.matches[0];
        assert_eq!(m.line_number, 2);
        assert_eq!(m.context_before, vec!["use std::io;"]);
        assert_eq!(m.context_after, vec!["    let data = read();"]);
    }
}

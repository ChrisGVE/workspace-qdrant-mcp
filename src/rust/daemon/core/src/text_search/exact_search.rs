//! Exact substring search over code_lines using FTS5 trigram index.
//!
//! Uses a two-phase approach:
//! 1. FTS5 trigram MATCH for fast candidate selection
//! 2. INSTR verification for exact substring match

use std::collections::HashMap;

use futures::TryStreamExt;
use sqlx::Row;
use tracing::debug;

use crate::search_db::{SearchDbManager, SearchDbError};
use super::escaping::{escape_fts5_pattern, compile_glob_matcher, resolve_path_filter};
use super::types::{SearchMatch, SearchOptions, SearchResults};

/// Search code_lines for an exact substring pattern.
///
/// Uses a two-phase approach:
/// 1. FTS5 trigram MATCH for fast candidate selection
/// 2. INSTR verification for exact substring match
///
/// For patterns shorter than 3 characters, falls back to INSTR-only scan.
/// When `path_glob` is set, applies glob filtering in Rust after SQL results.
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
            search_engine: "fts5".to_string(),
        });
    }

    // Resolve path_glob -> SQL prefix + glob matcher
    let (glob_pattern, effective_options) = resolve_path_filter(options);
    let glob_matcher = glob_pattern
        .as_deref()
        .map(compile_glob_matcher)
        .transpose()?;

    let fts5_pattern = escape_fts5_pattern(pattern);

    // Build the SQL query dynamically based on options
    let (sql, use_fts) = build_search_query(&fts5_pattern, &effective_options);

    debug!(
        "FTS5 search: pattern={:?}, fts5={:?}, use_fts={}, tenant={:?}, branch={:?}, path_prefix={:?}, path_glob={:?}",
        pattern, fts5_pattern, use_fts,
        effective_options.tenant_id, effective_options.branch, effective_options.path_prefix,
        options.path_glob,
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
    if let Some(ref tid) = effective_options.tenant_id {
        query = query.bind(tid);
    }
    if let Some(ref branch) = effective_options.branch {
        query = query.bind(branch);
    }
    if let Some(ref prefix) = effective_options.path_prefix {
        query = query.bind(format!("{}%", prefix));
    }

    let max_results = if options.max_results > 0 {
        options.max_results
    } else {
        usize::MAX
    };

    let mut stream = query.fetch(pool);
    let mut matches = Vec::new();
    let mut truncated = false;

    while let Some(row) = stream.try_next().await? {
        let file_path: String = row.get("file_path");

        // Apply glob filter if set
        if let Some(ref matcher) = glob_matcher {
            if !matcher(&file_path) {
                continue;
            }
        }

        matches.push(SearchMatch {
            line_id: row.get("line_id"),
            file_id: row.get("file_id"),
            line_number: row.get("line_number"),
            content: row.get("content"),
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
    // Drop the stream to release the connection before context queries
    drop(stream);

    // Attach context lines if requested
    if options.context_lines > 0 {
        attach_context_lines(search_db, &mut matches, options.context_lines).await?;
    }

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
        search_engine: "fts5".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Context line retrieval (Task 56)
// ---------------------------------------------------------------------------

/// Attach context lines (before/after) to search matches.
///
/// Groups matches by file_id, fetches the needed line ranges from code_lines,
/// then distributes context to each match. Uses a single query per file for
/// efficiency.
pub(crate) async fn attach_context_lines(
    search_db: &SearchDbManager,
    matches: &mut [SearchMatch],
    context_lines: usize,
) -> Result<(), SearchDbError> {
    if matches.is_empty() || context_lines == 0 {
        return Ok(());
    }

    let context_n = context_lines as i64;

    // Group match indices by file_id
    let mut file_matches: HashMap<i64, Vec<usize>> = HashMap::new();
    for (idx, m) in matches.iter().enumerate() {
        file_matches.entry(m.file_id).or_default().push(idx);
    }

    let pool = search_db.pool();

    for (file_id, match_indices) in &file_matches {
        let min_line = match_indices
            .iter()
            .map(|&i| matches[i].line_number)
            .min()
            .unwrap();
        let max_line = match_indices
            .iter()
            .map(|&i| matches[i].line_number)
            .max()
            .unwrap();

        let range_start = (min_line - context_n).max(1);
        let range_end = max_line + context_n;

        let rows = sqlx::query(
            r#"
SELECT line_number, content
FROM code_lines
WHERE file_id = ?1 AND line_number BETWEEN ?2 AND ?3
ORDER BY line_number"#,
        )
        .bind(file_id)
        .bind(range_start)
        .bind(range_end)
        .fetch_all(pool)
        .await?;

        let line_map: HashMap<i64, String> = rows
            .iter()
            .map(|row| (row.get::<i64, _>("line_number"), row.get::<String, _>("content")))
            .collect();

        for &idx in match_indices {
            let match_line = matches[idx].line_number;

            let before_start = (match_line - context_n).max(1);
            for ln in before_start..match_line {
                if let Some(content) = line_map.get(&ln) {
                    matches[idx].context_before.push(content.clone());
                }
            }

            for ln in (match_line + 1)..=(match_line + context_n) {
                if let Some(content) = line_map.get(&ln) {
                    matches[idx].context_after.push(content.clone());
                }
            }
        }
    }

    Ok(())
}

/// Build the search SQL query based on options.
///
/// Returns (sql_string, uses_fts5). When patterns are < 3 chars, FTS5
/// trigram index cannot be used and we fall back to LIKE-only scan.
fn build_search_query(
    fts5_pattern: &Option<String>,
    options: &SearchOptions,
) -> (String, bool) {
    let use_fts = fts5_pattern.is_some();

    let matching_cte = if use_fts {
        if options.case_insensitive {
            r#"
WITH matching AS (
    SELECT cl.line_id, cl.file_id, cl.line_number, cl.content
    FROM code_lines cl
    JOIN code_lines_fts fts ON cl.line_id = fts.rowid
    WHERE fts.content MATCH ?1 AND INSTR(LOWER(cl.content), ?2) > 0
)"#
            .to_string()
        } else {
            r#"
WITH matching AS (
    SELECT cl.line_id, cl.file_id, cl.line_number, cl.content
    FROM code_lines cl
    JOIN code_lines_fts fts ON cl.line_id = fts.rowid
    WHERE fts.content MATCH ?1 AND INSTR(cl.content, ?2) > 0
)"#
            .to_string()
        }
    } else if options.case_insensitive {
        r#"
WITH matching AS (
    SELECT cl.line_id, cl.file_id, cl.line_number, cl.content
    FROM code_lines cl
    WHERE INSTR(LOWER(cl.content), ?1) > 0
)"#
        .to_string()
    } else {
        r#"
WITH matching AS (
    SELECT cl.line_id, cl.file_id, cl.line_number, cl.content
    FROM code_lines cl
    WHERE INSTR(cl.content, ?1) > 0
)"#
        .to_string()
    };

    let mut sql = format!(
        "{}
SELECT m.line_id, m.file_id, m.line_number, m.content,
       fm.file_path, fm.tenant_id, fm.branch
FROM matching m
JOIN file_metadata fm ON m.file_id = fm.file_id
WHERE 1=1",
        matching_cte
    );

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
    }

    sql.push_str("\nORDER BY m.file_id, m.line_number");

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
    fn test_build_query_with_fts5() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions::default();
        let (sql, use_fts) = build_search_query(&pattern, &options);
        assert!(use_fts);
        assert!(sql.contains("MATCH ?1"));
        assert!(sql.contains("INSTR(cl.content, ?2)"));
        assert!(sql.contains("matching"));
        assert!(sql.contains("cl.line_number"));
        assert!(!sql.contains("SELECT COUNT(*)"));
    }

    #[test]
    fn test_build_query_without_fts5() {
        let options = SearchOptions::default();
        let (sql, use_fts) = build_search_query(&None, &options);
        assert!(!use_fts);
        assert!(!sql.contains("MATCH"));
        assert!(sql.contains("INSTR(cl.content, ?1)"));
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
        assert!(sql.contains("INSTR(LOWER(cl.content), ?2)"));
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

    #[tokio::test]
    async fn test_search_exact_basic() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn main() {", "    println!(\"hello\");", "}"], "proj1", Some("main"), "src/main.rs").await;
        let results = search_exact(&db, "println", &SearchOptions::default()).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 2);
        assert_eq!(results.matches[0].file_path, "src/main.rs");
        assert!(results.matches[0].content.contains("println"));
    }

    #[tokio::test]
    async fn test_search_exact_multiple_matches() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["use std::io;", "fn read() { io::stdin() }", "fn write() { io::stdout() }", "fn other() {}"], "proj1", Some("main"), "src/io.rs").await;
        let results = search_exact(&db, "io::", &SearchOptions::default()).await.unwrap();
        assert_eq!(results.matches.len(), 2);
        assert_eq!(results.matches[0].line_number, 2);
        assert_eq!(results.matches[1].line_number, 3);
    }

    #[tokio::test]
    async fn test_search_exact_no_match() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn main() {}", "let x = 42;"], "proj1", Some("main"), "src/main.rs").await;
        let results = search_exact(&db, "nonexistent_function", &SearchOptions::default()).await.unwrap();
        assert!(results.matches.is_empty());
    }

    #[tokio::test]
    async fn test_search_exact_case_sensitive() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn Main() {}", "fn main() {}"], "proj1", Some("main"), "src/main.rs").await;
        let results = search_exact(&db, "Main", &SearchOptions::default()).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 1);
    }

    #[tokio::test]
    async fn test_search_exact_case_insensitive() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn Main() {}", "fn main() {}"], "proj1", Some("main"), "src/main.rs").await;
        let results = search_exact(&db, "main", &SearchOptions { case_insensitive: true, ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 2);
    }

    #[tokio::test]
    async fn test_search_exact_scoped_by_tenant() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn hello() {}", "fn world() {}"], "proj1", Some("main"), "src/a.rs").await;
        insert_file_content(&db, 2, &["fn hello() {}", "fn goodbye() {}"], "proj2", Some("main"), "src/b.rs").await;
        let results = search_exact(&db, "hello", &SearchOptions { tenant_id: Some("proj1".to_string()), ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].tenant_id, "proj1");
    }

    #[tokio::test]
    async fn test_search_exact_scoped_by_path_prefix() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn func_a() {}"], "proj1", Some("main"), "src/module/a.rs").await;
        insert_file_content(&db, 2, &["fn func_a() {}"], "proj1", Some("main"), "tests/test_a.rs").await;
        let results = search_exact(&db, "func_a", &SearchOptions { path_prefix: Some("src/".to_string()), ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].file_path, "src/module/a.rs");
    }

    #[tokio::test]
    async fn test_search_exact_max_results() {
        let (_tmp, db) = setup_search_db().await;
        let lines: Vec<String> = (0..20).map(|i| format!("let item_{} = process();", i)).collect();
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        insert_file_content(&db, 1, &line_refs, "proj1", Some("main"), "src/many.rs").await;
        let results = search_exact(&db, "process", &SearchOptions { max_results: 5, ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 5);
        assert!(results.truncated);
    }

    #[tokio::test]
    async fn test_search_exact_empty_pattern() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn main() {}"], "proj1", Some("main"), "src/main.rs").await;
        let results = search_exact(&db, "", &SearchOptions::default()).await.unwrap();
        assert!(results.matches.is_empty());
    }

    #[tokio::test]
    async fn test_search_exact_special_characters() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["let pct = 100%;", "let _under = true;", "let path = \"C:\\\\Windows\";"], "proj1", Some("main"), "src/special.rs").await;
        let results = search_exact(&db, "100%", &SearchOptions::default()).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 1);
    }

    #[tokio::test]
    async fn test_search_exact_short_pattern_fallback() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn a() {}", "fn b() {}", "fn ab() {}"], "proj1", Some("main"), "src/short.rs").await;
        let results = search_exact(&db, "fn", &SearchOptions::default()).await.unwrap();
        assert_eq!(results.matches.len(), 3);
    }

    #[tokio::test]
    async fn test_search_exact_across_files() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn handler() {}", "  // process request"], "proj1", Some("main"), "src/api.rs").await;
        insert_file_content(&db, 2, &["fn worker() {}", "  // process job"], "proj1", Some("main"), "src/worker.rs").await;
        let results = search_exact(&db, "process", &SearchOptions::default()).await.unwrap();
        assert_eq!(results.matches.len(), 2);
        assert_eq!(results.matches[0].file_path, "src/api.rs");
        assert_eq!(results.matches[1].file_path, "src/worker.rs");
    }

    #[tokio::test]
    async fn test_search_exact_line_numbers_correct() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["// line 1", "// line 2", "fn target() {}", "// line 4", "fn target_two() {}"], "proj1", Some("main"), "src/lines.rs").await;
        let results = search_exact(&db, "target", &SearchOptions::default()).await.unwrap();
        assert_eq!(results.matches.len(), 2);
        assert_eq!(results.matches[0].line_number, 3);
        assert_eq!(results.matches[1].line_number, 5);
    }

    #[tokio::test]
    async fn test_search_exact_branch_filter() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn feature() {}"], "proj1", Some("main"), "src/a.rs").await;
        insert_file_content(&db, 2, &["fn feature() {}"], "proj1", Some("dev"), "src/a.rs").await;
        let results = search_exact(&db, "feature", &SearchOptions { branch: Some("dev".to_string()), ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].branch, Some("dev".to_string()));
    }

    #[tokio::test]
    async fn test_search_exact_pattern_with_double_quotes() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["let msg = \"hello world\";", "let other = 42;"], "proj1", Some("main"), "src/quotes.rs").await;
        let results = search_exact(&db, "\"hello world\"", &SearchOptions::default()).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        assert!(results.matches[0].content.contains("\"hello world\""));
    }

    #[tokio::test]
    async fn test_search_exact_with_path_glob() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn hello() {}"], "proj1", Some("main"), "src/main.rs").await;
        insert_file_content(&db, 2, &["fn hello() {}"], "proj1", Some("main"), "src/utils.ts").await;
        insert_file_content(&db, 3, &["fn hello() {}"], "proj1", Some("main"), "tests/test_main.rs").await;
        let results = search_exact(&db, "hello", &SearchOptions { path_glob: Some("src/**/*.rs".to_string()), ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].file_path, "src/main.rs");
    }

    #[tokio::test]
    async fn test_search_exact_with_glob_star_star() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn target() {}"], "proj1", Some("main"), "src/lib.rs").await;
        insert_file_content(&db, 2, &["fn target() {}"], "proj1", Some("main"), "src/deep/nested/mod.rs").await;
        insert_file_content(&db, 3, &["fn target() {}"], "proj1", Some("main"), "docs/guide.md").await;
        let results = search_exact(&db, "target", &SearchOptions { path_glob: Some("**/*.rs".to_string()), ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 2);
    }

    #[tokio::test]
    async fn test_search_exact_with_glob_braces() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn target() {}"], "proj1", Some("main"), "src/main.rs").await;
        insert_file_content(&db, 2, &["fn target() {}"], "proj1", Some("main"), "Cargo.toml").await;
        insert_file_content(&db, 3, &["fn target() {}"], "proj1", Some("main"), "src/script.js").await;
        let results = search_exact(&db, "target", &SearchOptions { path_glob: Some("**/*.{rs,toml}".to_string()), ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 2);
    }

    #[tokio::test]
    async fn test_search_exact_glob_overrides_path_prefix() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn target() {}"], "proj1", Some("main"), "src/main.rs").await;
        insert_file_content(&db, 2, &["fn target() {}"], "proj1", Some("main"), "tests/test.rs").await;
        let results = search_exact(&db, "target", &SearchOptions { path_prefix: Some("tests/".to_string()), path_glob: Some("src/**/*.rs".to_string()), ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].file_path, "src/main.rs");
    }

    #[tokio::test]
    async fn test_search_exact_glob_no_matches() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn hello() {}"], "proj1", Some("main"), "src/main.rs").await;
        let results = search_exact(&db, "hello", &SearchOptions { path_glob: Some("**/*.py".to_string()), ..Default::default() }).await.unwrap();
        assert!(results.matches.is_empty());
    }

    #[tokio::test]
    async fn test_search_exact_glob_with_tenant() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn shared() {}"], "proj1", Some("main"), "src/lib.rs").await;
        insert_file_content(&db, 2, &["fn shared() {}"], "proj2", Some("main"), "src/lib.rs").await;
        let results = search_exact(&db, "shared", &SearchOptions { tenant_id: Some("proj1".to_string()), path_glob: Some("**/*.rs".to_string()), ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].tenant_id, "proj1");
    }

    #[tokio::test]
    async fn test_context_lines_basic() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["// line 1", "// line 2", "fn target() {}", "// line 4", "// line 5"], "proj1", Some("main"), "src/main.rs").await;
        let results = search_exact(&db, "target", &SearchOptions { context_lines: 2, ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        let m = &results.matches[0];
        assert_eq!(m.line_number, 3);
        assert_eq!(m.context_before, vec!["// line 1", "// line 2"]);
        assert_eq!(m.context_after, vec!["// line 4", "// line 5"]);
    }

    #[tokio::test]
    async fn test_context_lines_at_file_start() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["fn first_line() {}", "// line 2", "// line 3"], "proj1", Some("main"), "src/start.rs").await;
        let results = search_exact(&db, "first_line", &SearchOptions { context_lines: 3, ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        let m = &results.matches[0];
        assert_eq!(m.line_number, 1);
        assert!(m.context_before.is_empty());
        assert_eq!(m.context_after, vec!["// line 2", "// line 3"]);
    }

    #[tokio::test]
    async fn test_context_lines_at_file_end() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["// line 1", "// line 2", "fn last_line() {}"], "proj1", Some("main"), "src/end.rs").await;
        let results = search_exact(&db, "last_line", &SearchOptions { context_lines: 3, ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        let m = &results.matches[0];
        assert_eq!(m.line_number, 3);
        assert_eq!(m.context_before, vec!["// line 1", "// line 2"]);
        assert!(m.context_after.is_empty());
    }

    #[tokio::test]
    async fn test_context_lines_zero() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["// before", "fn target() {}", "// after"], "proj1", Some("main"), "src/zero.rs").await;
        let results = search_exact(&db, "target", &SearchOptions { context_lines: 0, ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        assert!(results.matches[0].context_before.is_empty());
        assert!(results.matches[0].context_after.is_empty());
    }

    #[tokio::test]
    async fn test_context_lines_multiple_matches_same_file() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["// line 1", "fn target_a() {}", "// line 3", "// line 4", "fn target_b() {}", "// line 6"], "proj1", Some("main"), "src/multi.rs").await;
        let results = search_exact(&db, "target", &SearchOptions { context_lines: 1, ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 2);
        let m0 = &results.matches[0];
        assert_eq!(m0.line_number, 2);
        assert_eq!(m0.context_before, vec!["// line 1"]);
        assert_eq!(m0.context_after, vec!["// line 3"]);
        let m1 = &results.matches[1];
        assert_eq!(m1.line_number, 5);
        assert_eq!(m1.context_before, vec!["// line 4"]);
        assert_eq!(m1.context_after, vec!["// line 6"]);
    }

    #[tokio::test]
    async fn test_context_lines_across_files() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["// before", "fn target() {}", "// after"], "proj1", Some("main"), "src/a.rs").await;
        insert_file_content(&db, 2, &["// pre", "fn target() {}", "// post"], "proj1", Some("main"), "src/b.rs").await;
        let results = search_exact(&db, "target", &SearchOptions { context_lines: 1, ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 2);
        assert_eq!(results.matches[0].context_before, vec!["// before"]);
        assert_eq!(results.matches[0].context_after, vec!["// after"]);
        assert_eq!(results.matches[1].context_before, vec!["// pre"]);
        assert_eq!(results.matches[1].context_after, vec!["// post"]);
    }

    #[tokio::test]
    async fn test_context_lines_large_context() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(&db, 1, &["// 1", "// 2", "fn target() {}", "// 4", "// 5"], "proj1", Some("main"), "src/large.rs").await;
        let results = search_exact(&db, "target", &SearchOptions { context_lines: 10, ..Default::default() }).await.unwrap();
        assert_eq!(results.matches.len(), 1);
        let m = &results.matches[0];
        assert_eq!(m.context_before.len(), 2);
        assert_eq!(m.context_after.len(), 2);
    }
}

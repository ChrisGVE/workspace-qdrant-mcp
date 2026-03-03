//! Integration tests for exact substring search.

use crate::code_lines_schema::{initial_seq, UPSERT_FILE_METADATA_SQL};
use crate::search_db::SearchDbManager;
use super::super::types::SearchOptions;
use super::search::search_exact;

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

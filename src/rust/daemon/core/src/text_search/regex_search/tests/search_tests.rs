//! Integration tests for `search_regex`.

use crate::code_lines_schema::{initial_seq, UPSERT_FILE_METADATA_SQL};
use crate::search_db::SearchDbManager;
use crate::text_search::regex_search::search_regex;
use crate::text_search::types::SearchOptions;

pub(super) async fn setup_search_db() -> (tempfile::TempDir, SearchDbManager) {
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("test_search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();
    (tmp, manager)
}

pub(super) async fn insert_file_content(
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
        sqlx::query(
            "INSERT INTO code_lines (file_id, seq, content, line_number) VALUES (?1, ?2, ?3, ?4)",
        )
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

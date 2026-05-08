//! Core exact search integration tests.

use super::super::super::types::SearchOptions;
use super::super::search::search_exact;
use super::{insert_file_content, setup_search_db};

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
    let results = search_exact(&db, "Main", &SearchOptions::default())
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
    insert_file_content(&db, 1, &line_refs, "proj1", Some("main"), "src/many.rs").await;
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
        &["let msg = \"hello world\";", "let other = 42;"],
        "proj1",
        Some("main"),
        "src/quotes.rs",
    )
    .await;
    let results = search_exact(&db, "\"hello world\"", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
    assert!(results.matches[0].content.contains("\"hello world\""));
}

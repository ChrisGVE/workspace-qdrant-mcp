//! Context line attachment tests for exact search.

use super::super::super::types::SearchOptions;
use super::super::search::search_exact;
use super::{insert_file_content, setup_search_db};

#[tokio::test]
async fn test_context_lines_basic() {
    let (_tmp, db) = setup_search_db().await;
    insert_file_content(
        &db,
        1,
        &[
            "// line 1",
            "// line 2",
            "fn target() {}",
            "// line 4",
            "// line 5",
        ],
        "proj1",
        Some("main"),
        "src/main.rs",
    )
    .await;
    let results = search_exact(
        &db,
        "target",
        &SearchOptions {
            context_lines: 2,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);
    let m = &results.matches[0];
    assert_eq!(m.line_number, 3);
    assert_eq!(m.context_before, vec!["// line 1", "// line 2"]);
    assert_eq!(m.context_after, vec!["// line 4", "// line 5"]);
}

#[tokio::test]
async fn test_context_lines_at_file_start() {
    let (_tmp, db) = setup_search_db().await;
    insert_file_content(
        &db,
        1,
        &["fn first_line() {}", "// line 2", "// line 3"],
        "proj1",
        Some("main"),
        "src/start.rs",
    )
    .await;
    let results = search_exact(
        &db,
        "first_line",
        &SearchOptions {
            context_lines: 3,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);
    let m = &results.matches[0];
    assert_eq!(m.line_number, 1);
    assert!(m.context_before.is_empty());
    assert_eq!(m.context_after, vec!["// line 2", "// line 3"]);
}

#[tokio::test]
async fn test_context_lines_at_file_end() {
    let (_tmp, db) = setup_search_db().await;
    insert_file_content(
        &db,
        1,
        &["// line 1", "// line 2", "fn last_line() {}"],
        "proj1",
        Some("main"),
        "src/end.rs",
    )
    .await;
    let results = search_exact(
        &db,
        "last_line",
        &SearchOptions {
            context_lines: 3,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);
    let m = &results.matches[0];
    assert_eq!(m.line_number, 3);
    assert_eq!(m.context_before, vec!["// line 1", "// line 2"]);
    assert!(m.context_after.is_empty());
}

#[tokio::test]
async fn test_context_lines_zero() {
    let (_tmp, db) = setup_search_db().await;
    insert_file_content(
        &db,
        1,
        &["// before", "fn target() {}", "// after"],
        "proj1",
        Some("main"),
        "src/zero.rs",
    )
    .await;
    let results = search_exact(
        &db,
        "target",
        &SearchOptions {
            context_lines: 0,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);
    assert!(results.matches[0].context_before.is_empty());
    assert!(results.matches[0].context_after.is_empty());
}

#[tokio::test]
async fn test_context_lines_multiple_matches_same_file() {
    let (_tmp, db) = setup_search_db().await;
    insert_file_content(
        &db,
        1,
        &[
            "// line 1",
            "fn target_a() {}",
            "// line 3",
            "// line 4",
            "fn target_b() {}",
            "// line 6",
        ],
        "proj1",
        Some("main"),
        "src/multi.rs",
    )
    .await;
    let results = search_exact(
        &db,
        "target",
        &SearchOptions {
            context_lines: 1,
            ..Default::default()
        },
    )
    .await
    .unwrap();
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
    insert_file_content(
        &db,
        1,
        &["// before", "fn target() {}", "// after"],
        "proj1",
        Some("main"),
        "src/a.rs",
    )
    .await;
    insert_file_content(
        &db,
        2,
        &["// pre", "fn target() {}", "// post"],
        "proj1",
        Some("main"),
        "src/b.rs",
    )
    .await;
    let results = search_exact(
        &db,
        "target",
        &SearchOptions {
            context_lines: 1,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 2);
    assert_eq!(results.matches[0].context_before, vec!["// before"]);
    assert_eq!(results.matches[0].context_after, vec!["// after"]);
    assert_eq!(results.matches[1].context_before, vec!["// pre"]);
    assert_eq!(results.matches[1].context_after, vec!["// post"]);
}

#[tokio::test]
async fn test_context_lines_large_context() {
    let (_tmp, db) = setup_search_db().await;
    insert_file_content(
        &db,
        1,
        &["// 1", "// 2", "fn target() {}", "// 4", "// 5"],
        "proj1",
        Some("main"),
        "src/large.rs",
    )
    .await;
    let results = search_exact(
        &db,
        "target",
        &SearchOptions {
            context_lines: 10,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);
    let m = &results.matches[0];
    assert_eq!(m.context_before.len(), 2);
    assert_eq!(m.context_after.len(), 2);
}

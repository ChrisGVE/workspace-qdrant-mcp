//! Path glob filtering integration tests for exact search.

use super::super::super::types::SearchOptions;
use super::super::search::search_exact;
use super::{insert_file_content, setup_search_db};

#[tokio::test]
async fn test_search_exact_with_path_glob() {
    let (_tmp, db) = setup_search_db().await;
    insert_file_content(
        &db,
        1,
        &["fn hello() {}"],
        "proj1",
        Some("main"),
        "src/main.rs",
    )
    .await;
    insert_file_content(
        &db,
        2,
        &["fn hello() {}"],
        "proj1",
        Some("main"),
        "src/utils.ts",
    )
    .await;
    insert_file_content(
        &db,
        3,
        &["fn hello() {}"],
        "proj1",
        Some("main"),
        "tests/test_main.rs",
    )
    .await;
    let results = search_exact(
        &db,
        "hello",
        &SearchOptions {
            path_glob: Some("src/**/*.rs".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);
    assert_eq!(results.matches[0].file_path, "src/main.rs");
}

#[tokio::test]
async fn test_search_exact_with_glob_star_star() {
    let (_tmp, db) = setup_search_db().await;
    insert_file_content(
        &db,
        1,
        &["fn target() {}"],
        "proj1",
        Some("main"),
        "src/lib.rs",
    )
    .await;
    insert_file_content(
        &db,
        2,
        &["fn target() {}"],
        "proj1",
        Some("main"),
        "src/deep/nested/mod.rs",
    )
    .await;
    insert_file_content(
        &db,
        3,
        &["fn target() {}"],
        "proj1",
        Some("main"),
        "docs/guide.md",
    )
    .await;
    let results = search_exact(
        &db,
        "target",
        &SearchOptions {
            path_glob: Some("**/*.rs".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 2);
}

#[tokio::test]
async fn test_search_exact_with_glob_braces() {
    let (_tmp, db) = setup_search_db().await;
    insert_file_content(
        &db,
        1,
        &["fn target() {}"],
        "proj1",
        Some("main"),
        "src/main.rs",
    )
    .await;
    insert_file_content(
        &db,
        2,
        &["fn target() {}"],
        "proj1",
        Some("main"),
        "Cargo.toml",
    )
    .await;
    insert_file_content(
        &db,
        3,
        &["fn target() {}"],
        "proj1",
        Some("main"),
        "src/script.js",
    )
    .await;
    let results = search_exact(
        &db,
        "target",
        &SearchOptions {
            path_glob: Some("**/*.{rs,toml}".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 2);
}

#[tokio::test]
async fn test_search_exact_glob_overrides_path_prefix() {
    let (_tmp, db) = setup_search_db().await;
    insert_file_content(
        &db,
        1,
        &["fn target() {}"],
        "proj1",
        Some("main"),
        "src/main.rs",
    )
    .await;
    insert_file_content(
        &db,
        2,
        &["fn target() {}"],
        "proj1",
        Some("main"),
        "tests/test.rs",
    )
    .await;
    let results = search_exact(
        &db,
        "target",
        &SearchOptions {
            path_prefix: Some("tests/".to_string()),
            path_glob: Some("src/**/*.rs".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);
    assert_eq!(results.matches[0].file_path, "src/main.rs");
}

#[tokio::test]
async fn test_search_exact_glob_no_matches() {
    let (_tmp, db) = setup_search_db().await;
    insert_file_content(
        &db,
        1,
        &["fn hello() {}"],
        "proj1",
        Some("main"),
        "src/main.rs",
    )
    .await;
    let results = search_exact(
        &db,
        "hello",
        &SearchOptions {
            path_glob: Some("**/*.py".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert!(results.matches.is_empty());
}

#[tokio::test]
async fn test_search_exact_glob_with_tenant() {
    let (_tmp, db) = setup_search_db().await;
    insert_file_content(
        &db,
        1,
        &["fn shared() {}"],
        "proj1",
        Some("main"),
        "src/lib.rs",
    )
    .await;
    insert_file_content(
        &db,
        2,
        &["fn shared() {}"],
        "proj2",
        Some("main"),
        "src/lib.rs",
    )
    .await;
    let results = search_exact(
        &db,
        "shared",
        &SearchOptions {
            tenant_id: Some("proj1".to_string()),
            path_glob: Some("**/*.rs".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);
    assert_eq!(results.matches[0].tenant_id, "proj1");
}

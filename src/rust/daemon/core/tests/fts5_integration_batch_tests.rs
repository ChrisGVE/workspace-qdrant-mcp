//! FTS5 Integration Tests — Batch Processing and Search Accuracy
//!
//! Tests covering batch-mode ingestion and search precision/recall:
//! - Batch mode processing -> search works across many files
//! - Search accuracy with 100 files and known patterns
//! - Regex search accuracy across files
//! - Context lines with pipeline integration
//! - Path glob filtering with pipeline integration

use tempfile::TempDir;

use workspace_qdrant_core::fts_batch_processor::{FileChange, FtsBatchConfig, FtsBatchProcessor};
use workspace_qdrant_core::search_db::SearchDbManager;
use workspace_qdrant_core::text_search::{search_exact, search_regex, SearchOptions};

async fn setup_db() -> (TempDir, SearchDbManager) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();
    (tmp, manager)
}

// ---------------------------------------------------------------------------
// Batch mode processing -> search works across many files
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_batch_mode_20_files_all_searchable() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Add 20 files
    for i in 0..20 {
        processor.add_change(FileChange {
            file_id: i + 1,
            old_content: String::new(),
            new_content: format!(
                "/// Module {}\nfn batch_item_{}() {{\n    process();\n}}",
                i, i
            ),
            tenant_id: "batch-proj".to_string(),
            branch: Some("main".to_string()),
            file_path: format!("src/mod_{}.rs", i),
            base_point: None,
            relative_path: None,
            file_hash: None,
        });
    }

    // Queue depth = 20 => batch mode (threshold = 10)
    let stats = processor.flush(20).await.unwrap();
    assert_eq!(stats.files_processed, 20);
    assert!(stats.batch_mode);

    // All 20 should be searchable
    let results = search_exact(
        &db,
        "process",
        &SearchOptions {
            tenant_id: Some("batch-proj".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(
        results.matches.len(),
        20,
        "All 20 files should have 'process' match"
    );

    // Each specific item should be findable (use trailing "()" to avoid
    // substring matches like batch_item_1 matching batch_item_10)
    for i in 0..20 {
        let results = search_exact(
            &db,
            &format!("batch_item_{}()", i),
            &SearchOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            results.matches.len(),
            1,
            "batch_item_{}() should have exactly 1 match",
            i
        );
    }
}

// ---------------------------------------------------------------------------
// Search accuracy with many files and known patterns
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_search_accuracy_100_files() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Create 100 files with deterministic content
    for i in 0..100 {
        let content = format!(
            "// File {i}\nfn function_{i}() {{\n    let value = {val};\n    compute_{group}(value);\n}}",
            i = i,
            val = i * 7,
            group = i % 5,  // 5 groups: compute_0..compute_4
        );
        processor.add_change(FileChange {
            file_id: i + 1,
            old_content: String::new(),
            new_content: content,
            tenant_id: "accuracy-proj".to_string(),
            branch: Some("main".to_string()),
            file_path: format!("src/module_{}.rs", i),
            base_point: None,
            relative_path: None,
            file_hash: None,
        });
    }

    // Use batch mode for 100 files
    processor.flush(100).await.unwrap();

    // Verify 100% recall: every file has a unique function name
    // Use trailing "()" to avoid substring matches (function_1 matching function_10)
    for i in 0..100 {
        let results = search_exact(
            &db,
            &format!("function_{}()", i),
            &SearchOptions {
                tenant_id: Some("accuracy-proj".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            results.matches.len(),
            1,
            "function_{}() should have exactly 1 match (recall)",
            i
        );
    }

    // Verify precision: search for a group function name
    // compute_0( should appear in files 0, 5, 10, ..., 95 = 20 files
    let group_results = search_exact(
        &db,
        "compute_0(",
        &SearchOptions {
            tenant_id: Some("accuracy-proj".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(
        group_results.matches.len(),
        20,
        "compute_0( should appear in exactly 20 files"
    );

    // Verify no false positives: search for non-existent pattern
    let no_results = search_exact(
        &db,
        "nonexistent_xyz_pattern",
        &SearchOptions {
            tenant_id: Some("accuracy-proj".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert!(
        no_results.matches.is_empty(),
        "Should have 0 false positives"
    );
}

// ---------------------------------------------------------------------------
// Regex search accuracy across files
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_regex_search_accuracy_across_files() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Create files with pattern variations
    let files = vec![
        (1, "fn async_handler() {}", "src/async.rs"),
        (2, "fn sync_handler() {}", "src/sync.rs"),
        (3, "async fn process() {}", "src/process.rs"),
        (4, "pub fn helper() {}", "src/helper.rs"),
        (5, "fn handle_request() {}", "src/request.rs"),
    ];

    for (id, content, path) in &files {
        processor.add_change(FileChange {
            file_id: *id,
            old_content: String::new(),
            new_content: content.to_string(),
            tenant_id: "regex-proj".to_string(),
            branch: Some("main".to_string()),
            file_path: path.to_string(),
            base_point: None,
            relative_path: None,
            file_hash: None,
        });
    }
    processor.flush(0).await.unwrap();

    // Regex: find all _handler functions
    let results = search_regex(
        &db,
        "fn \\w+_handler",
        &SearchOptions {
            tenant_id: Some("regex-proj".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(
        results.matches.len(),
        2,
        "Should find async_handler and sync_handler"
    );

    // Regex: find all async functions (either "async fn" or "fn async_")
    let results = search_regex(
        &db,
        "async",
        &SearchOptions {
            tenant_id: Some("regex-proj".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(
        results.matches.len(),
        2,
        "Should find 2 async-related lines"
    );
}

// ---------------------------------------------------------------------------
// Context lines with pipeline integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_context_lines_after_batch_ingest() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let content = "use std::io;\n\nfn main() {\n    let x = 42;\n    println!(\"{}\", x);\n}\n\nfn helper() {}";

    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: content.to_string(),
        tenant_id: "ctx-proj".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/main.rs".to_string(),
        base_point: None,
        relative_path: None,
        file_hash: None,
    });
    processor.flush(0).await.unwrap();

    let results = search_exact(
        &db,
        "println",
        &SearchOptions {
            context_lines: 2,
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(results.matches.len(), 1);
    let m = &results.matches[0];
    assert_eq!(m.line_number, 5);
    // 2 lines before: line 3 and line 4
    assert_eq!(m.context_before.len(), 2);
    assert!(m.context_before[0].contains("fn main()"));
    assert!(m.context_before[1].contains("let x = 42"));
    // 2 lines after: line 6 and line 7
    assert_eq!(m.context_after.len(), 2);
}

// ---------------------------------------------------------------------------
// Path glob filtering with pipeline integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_path_glob_across_ingested_files() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let files = vec![
        (1, "fn search_target() {}", "src/lib.rs"),
        (2, "fn search_target() {}", "src/deep/nested.rs"),
        (3, "fn search_target() {}", "tests/test_lib.rs"),
        (4, "fn search_target() {}", "docs/guide.md"),
    ];

    for (id, content, path) in &files {
        processor.add_change(FileChange {
            file_id: *id,
            old_content: String::new(),
            new_content: content.to_string(),
            tenant_id: "glob-proj".to_string(),
            branch: Some("main".to_string()),
            file_path: path.to_string(),
            base_point: None,
            relative_path: None,
            file_hash: None,
        });
    }
    processor.flush(0).await.unwrap();

    // Glob: only .rs files under src/
    let results = search_exact(
        &db,
        "search_target",
        &SearchOptions {
            path_glob: Some("src/**/*.rs".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(
        results.matches.len(),
        2,
        "Should find 2 .rs files under src/"
    );

    // Glob: any .rs file
    let results = search_exact(
        &db,
        "search_target",
        &SearchOptions {
            path_glob: Some("**/*.rs".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 3, "Should find 3 .rs files total");
}

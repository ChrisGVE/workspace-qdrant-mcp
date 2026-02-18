//! FTS5 Search Pipeline Integration Tests (Task 61)
//!
//! End-to-end tests covering the full FTS5 pipeline:
//! - File add → FtsBatchProcessor → search finds content
//! - File update (diff) → search finds new, not old content
//! - File delete → search no longer finds deleted content
//! - Batch mode processing → search works across many files
//! - Search accuracy with 100 files and known patterns
//! - Concurrent reads and writes

use std::sync::Arc;
use tempfile::TempDir;

use workspace_qdrant_core::fts_batch_processor::{
    FileChange, FtsBatchConfig, FtsBatchProcessor,
};
use workspace_qdrant_core::search_db::SearchDbManager;
use workspace_qdrant_core::text_search::{search_exact, search_regex, SearchOptions};

async fn setup_db() -> (TempDir, SearchDbManager) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();
    (tmp, manager)
}

// ---------------------------------------------------------------------------
// 1. File add → search finds content
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_file_add_then_search_exact() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    processor
        .full_rewrite(
            1,
            "use std::io;\nfn read_file() -> io::Result<String> {\n    Ok(String::new())\n}",
            "proj1",
            Some("main"),
            "src/reader.rs",
        )
        .await
        .unwrap();

    // Search for content that was just ingested
    let results = search_exact(&db, "read_file", &SearchOptions::default())
        .await
        .unwrap();

    assert_eq!(results.matches.len(), 1);
    assert_eq!(results.matches[0].line_number, 2);
    assert_eq!(results.matches[0].file_path, "src/reader.rs");
    assert_eq!(results.matches[0].tenant_id, "proj1");
}

#[tokio::test]
async fn test_file_add_via_flush_then_search() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: "pub struct Config {\n    pub port: u16,\n    pub host: String,\n}"
            .to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/config.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    let results = search_exact(
        &db,
        "Config",
        &SearchOptions {
            tenant_id: Some("proj1".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(results.matches.len(), 1);
    assert!(results.matches[0].content.contains("pub struct Config"));
}

// ---------------------------------------------------------------------------
// 2. File update (content changed) → search finds new content
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_file_update_diff_search_finds_new_content() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let original = "fn handler() {\n    println!(\"hello\");\n}";

    // First ingestion
    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: original.to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/handler.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    // Verify original content found
    let results = search_exact(&db, "hello", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);

    // Update: change "hello" to "goodbye"
    let updated = "fn handler() {\n    println!(\"goodbye\");\n}";
    processor.add_change(FileChange {
        file_id: 1,
        old_content: original.to_string(),
        new_content: updated.to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/handler.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    // Old content should NOT be found
    let results_old = search_exact(&db, "hello", &SearchOptions::default())
        .await
        .unwrap();
    assert!(
        results_old.matches.is_empty(),
        "Old content 'hello' should not be found after update"
    );

    // New content SHOULD be found
    let results_new = search_exact(&db, "goodbye", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results_new.matches.len(), 1);
    assert!(results_new.matches[0].content.contains("goodbye"));
}

#[tokio::test]
async fn test_file_update_adds_new_lines() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let original = "fn main() {\n    run();\n}";

    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: original.to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/main.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    // Add a new function
    let updated = "fn main() {\n    run();\n}\n\nfn helper_function() {\n    // added\n}";
    processor.add_change(FileChange {
        file_id: 1,
        old_content: original.to_string(),
        new_content: updated.to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/main.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    // New function should be searchable
    let results = search_exact(&db, "helper_function", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
    // Line number is derived from ROW_NUMBER() over seq ordering after diff;
    // verify it's in the expected range (lines 4-7 of a 7-line file)
    assert!(
        results.matches[0].line_number >= 4 && results.matches[0].line_number <= 7,
        "helper_function should be near the end of the file, got line {}",
        results.matches[0].line_number
    );
}

// ---------------------------------------------------------------------------
// 3. File delete → search no longer finds deleted content
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_file_delete_removes_from_search() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    processor
        .full_rewrite(
            1,
            "fn unique_function_name() {}",
            "proj1",
            Some("main"),
            "src/to_delete.rs",
        )
        .await
        .unwrap();

    // Verify found before deletion
    let results = search_exact(&db, "unique_function_name", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);

    // Delete the file
    let deleted = processor.delete_file(1).await.unwrap();
    assert_eq!(deleted, 1);

    // Should no longer be found
    let results_after = search_exact(&db, "unique_function_name", &SearchOptions::default())
        .await
        .unwrap();
    assert!(
        results_after.matches.is_empty(),
        "Deleted file content should not appear in search results"
    );
}

#[tokio::test]
async fn test_tenant_delete_removes_all_project_files() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Add files for two projects
    processor
        .full_rewrite(1, "fn proj_a_func() {}", "proj-a", Some("main"), "src/a.rs")
        .await
        .unwrap();
    processor
        .full_rewrite(2, "fn proj_a_other() {}", "proj-a", Some("main"), "src/b.rs")
        .await
        .unwrap();
    processor
        .full_rewrite(3, "fn proj_b_func() {}", "proj-b", Some("main"), "src/c.rs")
        .await
        .unwrap();

    // Delete proj-a
    processor.delete_tenant("proj-a").await.unwrap();

    // proj-a content should be gone
    let results_a = search_exact(
        &db,
        "proj_a",
        &SearchOptions {
            tenant_id: Some("proj-a".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert!(results_a.matches.is_empty());

    // proj-b content should remain
    let results_b = search_exact(
        &db,
        "proj_b_func",
        &SearchOptions {
            tenant_id: Some("proj-b".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results_b.matches.len(), 1);
}

// ---------------------------------------------------------------------------
// 4. Batch mode processing → search works across many files
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
// 5. Search accuracy with many files and known patterns
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
    assert!(no_results.matches.is_empty(), "Should have 0 false positives");
}

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
    assert_eq!(results.matches.len(), 2, "Should find async_handler and sync_handler");

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
    assert_eq!(results.matches.len(), 2, "Should find 2 async-related lines");
}

// ---------------------------------------------------------------------------
// 6. Context lines with pipeline integration
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
// 7. Path glob filtering with pipeline integration
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
    assert_eq!(results.matches.len(), 2, "Should find 2 .rs files under src/");

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

// ---------------------------------------------------------------------------
// 8. Concurrent reads and writes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_concurrent_search_during_writes() {
    let (_tmp, db) = setup_db().await;
    let db = Arc::new(db);

    // Pre-populate with some data
    {
        let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
        processor
            .full_rewrite(
                1,
                "fn existing_function() {}",
                "conc-proj",
                Some("main"),
                "src/existing.rs",
            )
            .await
            .unwrap();
    }

    // Spawn concurrent search tasks
    let mut handles = Vec::new();
    for i in 0..5 {
        let db_clone = Arc::clone(&db);
        let handle = tokio::spawn(async move {
            let results = search_exact(
                &db_clone,
                "existing_function",
                &SearchOptions {
                    tenant_id: Some("conc-proj".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

            // Should always find the pre-existing content
            assert!(
                !results.matches.is_empty(),
                "Search {} should find existing content",
                i
            );
        });
        handles.push(handle);
    }

    // Spawn a concurrent write
    let db_write = Arc::clone(&db);
    let write_handle = tokio::spawn(async move {
        let processor = FtsBatchProcessor::new(&db_write, FtsBatchConfig::default());
        processor
            .full_rewrite(
                2,
                "fn new_concurrent_fn() {}",
                "conc-proj",
                Some("main"),
                "src/concurrent.rs",
            )
            .await
            .unwrap();
    });

    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }
    write_handle.await.unwrap();

    // After all tasks complete, both functions should be searchable
    let results = search_exact(
        &db,
        "existing_function",
        &SearchOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);

    let results = search_exact(
        &db,
        "new_concurrent_fn",
        &SearchOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);
}

#[tokio::test]
async fn test_concurrent_writes_no_corruption() {
    let (_tmp, db) = setup_db().await;
    let db = Arc::new(db);

    // Spawn 10 concurrent write tasks
    let mut handles = Vec::new();
    for i in 0..10 {
        let db_clone = Arc::clone(&db);
        let handle = tokio::spawn(async move {
            let processor = FtsBatchProcessor::new(&db_clone, FtsBatchConfig::default());
            processor
                .full_rewrite(
                    i + 1,
                    &format!("fn concurrent_writer_{}() {{}}", i),
                    "conc-proj",
                    Some("main"),
                    &format!("src/writer_{}.rs", i),
                )
                .await
                .unwrap();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await.unwrap();
    }

    // After all writes, verify no data corruption — each file should be searchable
    // Use trailing "()" to avoid substring matches
    for i in 0..10 {
        let results = search_exact(
            &db,
            &format!("concurrent_writer_{}()", i),
            &SearchOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            results.matches.len(),
            1,
            "concurrent_writer_{}() should have exactly 1 match",
            i
        );
    }
}

// ---------------------------------------------------------------------------
// 9. Full pipeline round-trip: add → search → update → search → delete → search
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_full_lifecycle_round_trip() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Step 1: Add file
    let v1 = "fn lifecycle() {\n    step_one();\n}";
    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: v1.to_string(),
        tenant_id: "lifecycle-proj".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/lifecycle.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    let r1 = search_exact(&db, "step_one", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(r1.matches.len(), 1, "Step 1: should find step_one");

    // Step 2: Update file (change content)
    let v2 = "fn lifecycle() {\n    step_two();\n    extra_line();\n}";
    processor.add_change(FileChange {
        file_id: 1,
        old_content: v1.to_string(),
        new_content: v2.to_string(),
        tenant_id: "lifecycle-proj".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/lifecycle.rs".to_string(),
    });
    processor.flush(0).await.unwrap();

    let r2_old = search_exact(&db, "step_one", &SearchOptions::default())
        .await
        .unwrap();
    assert!(r2_old.matches.is_empty(), "Step 2: step_one should be gone");

    let r2_new = search_exact(&db, "step_two", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(r2_new.matches.len(), 1, "Step 2: should find step_two");

    let r2_extra = search_exact(&db, "extra_line", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(r2_extra.matches.len(), 1, "Step 2: should find extra_line");

    // Step 3: Delete file
    processor.delete_file(1).await.unwrap();

    let r3 = search_exact(&db, "step_two", &SearchOptions::default())
        .await
        .unwrap();
    assert!(r3.matches.is_empty(), "Step 3: all content should be gone after delete");

    let r3_extra = search_exact(&db, "extra_line", &SearchOptions::default())
        .await
        .unwrap();
    assert!(
        r3_extra.matches.is_empty(),
        "Step 3: extra_line should also be gone"
    );
}

// ---------------------------------------------------------------------------
// 10. Edge cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_empty_file_searchable() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    processor
        .full_rewrite(1, "", "proj1", Some("main"), "src/empty.rs")
        .await
        .unwrap();

    // Should not crash, should return empty results
    let results = search_exact(&db, "anything", &SearchOptions::default())
        .await
        .unwrap();
    assert!(results.matches.is_empty());
}

#[tokio::test]
async fn test_unicode_content_searchable() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    processor
        .full_rewrite(
            1,
            "// 日本語コメント\nfn greet() -> &'static str {\n    \"こんにちは世界\"\n}",
            "proj1",
            Some("main"),
            "src/unicode.rs",
        )
        .await
        .unwrap();

    // Search for Japanese text (trigram needs ≥3 chars)
    let results = search_exact(&db, "日本語", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
    assert_eq!(results.matches[0].line_number, 1);

    // Search for function name works normally
    let results = search_exact(&db, "greet", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
}

#[tokio::test]
async fn test_large_file_searchable() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Generate a 500-line file
    let mut lines = Vec::with_capacity(500);
    for i in 0..500 {
        lines.push(format!("let variable_{} = compute({});", i, i));
    }
    let content = lines.join("\n");

    processor
        .full_rewrite(1, &content, "proj1", Some("main"), "src/large.rs")
        .await
        .unwrap();

    // Search for specific line (first, middle, last)
    // Use " = " suffix to avoid substring matches (variable_0 matching variable_0X)
    let first = search_exact(&db, "variable_0 =", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(first.matches.len(), 1);
    assert_eq!(first.matches[0].line_number, 1);

    let middle = search_exact(&db, "variable_250 =", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(middle.matches.len(), 1);
    assert_eq!(middle.matches[0].line_number, 251);

    let last = search_exact(&db, "variable_499 =", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(last.matches.len(), 1);
    assert_eq!(last.matches[0].line_number, 500);
}

#[tokio::test]
async fn test_multi_tenant_isolation() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Three projects with similar content
    let projects = vec![
        ("alpha", 1, "fn shared_name() { alpha(); }"),
        ("beta", 2, "fn shared_name() { beta(); }"),
        ("gamma", 3, "fn shared_name() { gamma(); }"),
    ];

    for (tenant, id, content) in &projects {
        processor.add_change(FileChange {
            file_id: *id,
            old_content: String::new(),
            new_content: content.to_string(),
            tenant_id: tenant.to_string(),
            branch: Some("main".to_string()),
            file_path: "src/main.rs".to_string(),
        });
    }
    processor.flush(0).await.unwrap();

    // Without tenant filter: all 3 matches
    let all = search_exact(&db, "shared_name", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(all.matches.len(), 3);

    // With tenant filter: exactly 1 match per tenant
    for (tenant, _, content) in &projects {
        let results = search_exact(
            &db,
            "shared_name",
            &SearchOptions {
                tenant_id: Some(tenant.to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(results.matches.len(), 1, "Tenant {} should have 1 match", tenant);
        assert!(
            results.matches[0].content.contains(content),
            "Tenant {} match should contain the correct content",
            tenant
        );
    }
}

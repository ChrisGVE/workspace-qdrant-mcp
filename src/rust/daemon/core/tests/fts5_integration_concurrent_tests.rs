//! FTS5 Integration Tests — Concurrency, Lifecycle, and Edge Cases
//!
//! Tests covering concurrent access patterns, full lifecycle round-trips,
//! and edge-case inputs:
//! - Concurrent reads and writes
//! - Full pipeline round-trip: add -> search -> update -> search -> delete -> search
//! - Empty files, Unicode content, large files
//! - Multi-tenant isolation

use std::sync::Arc;
use tempfile::TempDir;

use workspace_qdrant_core::fts_batch_processor::{FileChange, FtsBatchConfig, FtsBatchProcessor};
use workspace_qdrant_core::search_db::SearchDbManager;
use workspace_qdrant_core::text_search::{search_exact, SearchOptions};

async fn setup_db() -> (TempDir, SearchDbManager) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();
    (tmp, manager)
}

// ---------------------------------------------------------------------------
// Concurrent reads and writes
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
                None,
                None,
                None,
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
                None,
                None,
                None,
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
    let results = search_exact(&db, "existing_function", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);

    let results = search_exact(&db, "new_concurrent_fn", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
}

#[tokio::test]
async fn test_sequential_writes_no_corruption() {
    // In production, the unified queue processor serializes all writes.
    // This test verifies that sequential writes to many files don't corrupt the FTS5 index.
    let (_tmp, db) = setup_db().await;

    for i in 0..10_i64 {
        let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
        processor
            .full_rewrite(
                i + 1,
                &format!("fn sequential_writer_{}() {{}}", i),
                "conc-proj",
                Some("main"),
                &format!("src/writer_{}.rs", i),
                None,
                None,
                None,
            )
            .await
            .unwrap();
    }

    // After all writes, verify no data corruption — each file should be searchable
    for i in 0..10 {
        let results = search_exact(
            &db,
            &format!("sequential_writer_{}()", i),
            &SearchOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            results.matches.len(),
            1,
            "sequential_writer_{}() should have exactly 1 match",
            i
        );
    }
}

// ---------------------------------------------------------------------------
// Full pipeline round-trip: add -> search -> update -> search -> delete -> search
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
        base_point: None,
        relative_path: None,
        file_hash: None,
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
        base_point: None,
        relative_path: None,
        file_hash: None,
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
    assert!(
        r3.matches.is_empty(),
        "Step 3: all content should be gone after delete"
    );

    let r3_extra = search_exact(&db, "extra_line", &SearchOptions::default())
        .await
        .unwrap();
    assert!(
        r3_extra.matches.is_empty(),
        "Step 3: extra_line should also be gone"
    );
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_empty_file_searchable() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    processor
        .full_rewrite(
            1,
            "",
            "proj1",
            Some("main"),
            "src/empty.rs",
            None,
            None,
            None,
        )
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
            None,
            None,
            None,
        )
        .await
        .unwrap();

    // Search for Japanese text (trigram needs >= 3 chars)
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
        .full_rewrite(
            1,
            &content,
            "proj1",
            Some("main"),
            "src/large.rs",
            None,
            None,
            None,
        )
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
            base_point: None,
            relative_path: None,
            file_hash: None,
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
        assert_eq!(
            results.matches.len(),
            1,
            "Tenant {} should have 1 match",
            tenant
        );
        assert!(
            results.matches[0].content.contains(content),
            "Tenant {} match should contain the correct content",
            tenant
        );
    }
}

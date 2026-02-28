//! FTS5 Edge Case Tests — Concurrent Access, Diff Edge Cases, Regex, and Batch
//!
//! Covers:
//! - Concurrent update + search on same file
//! - Diff with completely different content
//! - Diff content to empty / empty to content
//! - Regex with special quantifiers
//! - Case-insensitive search
//! - Batch with heterogeneous edge cases

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
// Concurrent update + search on same file
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_concurrent_update_and_search_same_file() {
    let (_tmp, db) = setup_db().await;
    let db = Arc::new(db);

    // Initial content
    {
        let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
        processor
            .full_rewrite(
                1,
                "fn stable_function() {}",
                "proj1",
                Some("main"),
                "src/concurrent.rs",
                None,
                None,
                None,
            )
            .await
            .unwrap();
    }

    // Spawn search tasks while updating
    let mut handles = Vec::new();

    // 5 concurrent searches
    for _ in 0..5 {
        let db_clone = Arc::clone(&db);
        handles.push(tokio::spawn(async move {
            // The function should be found at some point during the test
            let results = search_exact(
                &db_clone,
                "stable_function",
                &SearchOptions::default(),
            )
            .await;
            assert!(results.is_ok(), "Search should not error during concurrent access");
        }));
    }

    // Concurrent update
    {
        let db_clone = Arc::clone(&db);
        handles.push(tokio::spawn(async move {
            let processor = FtsBatchProcessor::new(&db_clone, FtsBatchConfig::default());
            let result = processor
                .full_rewrite(
                    1,
                    "fn updated_function() {}",
                    "proj1",
                    Some("main"),
                    "src/concurrent.rs",
                    None,
                    None,
                    None,
                )
                .await;
            assert!(result.is_ok(), "Update should not error during concurrent access");
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }
}

// ---------------------------------------------------------------------------
// Diff edge cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_diff_completely_different_content() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let v1 = "fn alpha() {\n    one();\n    two();\n    three();\n}";
    let v2 = "struct Beta {\n    x: i32,\n    y: f64,\n}";

    // Ingest v1
    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: v1.to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/file.rs".to_string(),
        base_point: None,
        relative_path: None,
        file_hash: None,
    });
    processor.flush(0).await.unwrap();

    // Replace with completely different v2
    processor.add_change(FileChange {
        file_id: 1,
        old_content: v1.to_string(),
        new_content: v2.to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/file.rs".to_string(),
        base_point: None,
        relative_path: None,
        file_hash: None,
    });
    processor.flush(0).await.unwrap();

    // v1 content fully gone
    for term in &["alpha", "one()", "two()", "three()"] {
        let results = search_exact(&db, term, &SearchOptions::default())
            .await
            .unwrap();
        assert!(results.matches.is_empty(), "{} from v1 should be gone", term);
    }

    // v2 content present
    let results = search_exact(&db, "struct Beta", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1);
}

#[tokio::test]
async fn test_diff_content_to_empty() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let v1 = "fn nonempty() {\n    code();\n}";

    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: v1.to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/file.rs".to_string(),
        base_point: None,
        relative_path: None,
        file_hash: None,
    });
    processor.flush(0).await.unwrap();

    // "Update" to empty content
    processor.add_change(FileChange {
        file_id: 1,
        old_content: v1.to_string(),
        new_content: String::new(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/file.rs".to_string(),
        base_point: None,
        relative_path: None,
        file_hash: None,
    });
    processor.flush(0).await.unwrap();

    let results = search_exact(&db, "nonempty", &SearchOptions::default())
        .await
        .unwrap();
    assert!(results.matches.is_empty(), "Content should be gone after emptying file");
}

#[tokio::test]
async fn test_diff_empty_to_content() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Start with empty file
    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: String::new(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/file.rs".to_string(),
        base_point: None,
        relative_path: None,
        file_hash: None,
    });
    processor.flush(0).await.unwrap();

    // Update empty to content
    processor.add_change(FileChange {
        file_id: 1,
        old_content: String::new(),
        new_content: "fn appeared() {}".to_string(),
        tenant_id: "proj1".to_string(),
        branch: Some("main".to_string()),
        file_path: "src/file.rs".to_string(),
        base_point: None,
        relative_path: None,
        file_hash: None,
    });
    processor.flush(0).await.unwrap();

    let results = search_exact(&db, "appeared", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 1, "Content should appear after empty->content update");
}

// ---------------------------------------------------------------------------
// Regex edge cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_regex_with_special_quantifiers() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let content = "let aaa = 1;\nlet aaab = 2;\nlet bbb = 3;\nlet abc = 4;";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/regex.rs", None, None, None)
        .await
        .unwrap();

    // Regex: a+ followed by b
    let results = search_regex(&db, "a+b", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(results.matches.len(), 2, "Should match aaab and abc");

    // Regex: word boundary pattern (Rust regex doesn't support backreferences)
    // Matches: aaa, bbb, abc (all are 3 lowercase letters)
    let results = search_regex(&db, "let [a-z]{3} =", &SearchOptions::default())
        .await
        .unwrap();
    assert_eq!(
        results.matches.len(),
        3,
        "Should find 'let aaa =', 'let bbb =', and 'let abc ='"
    );
}

#[tokio::test]
async fn test_case_insensitive_search() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let content = "fn MyFunction() {}\nfn myfunction() {}\nfn MYFUNCTION() {}";

    processor
        .full_rewrite(1, content, "proj1", Some("main"), "src/case.rs", None, None, None)
        .await
        .unwrap();

    // Case-sensitive (default): only one match
    let results = search_exact(
        &db,
        "MyFunction",
        &SearchOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1, "Case-sensitive should find exactly one");

    // Case-insensitive: all three
    let results = search_exact(
        &db,
        "myfunction",
        &SearchOptions {
            case_insensitive: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(
        results.matches.len(),
        3,
        "Case-insensitive should find all three variants"
    );
}

// ---------------------------------------------------------------------------
// Batch with heterogeneous edge cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_batch_with_mixed_edge_cases() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let long_line = "x".repeat(15_000);
    let files: Vec<(i64, &str, &str)> = vec![
        (1, "", "src/empty.rs"),                         // empty
        (2, "   \n\n\t\n", "src/whitespace.rs"),         // whitespace only
        (3, "fn normal_code() {}", "src/normal.rs"),     // normal
        (4, "\u{1F680} emoji_file", "src/emoji.rs"),     // emoji
        (5, "line\r\nwith\r\ncrlf", "src/crlf.rs"),      // CRLF
        (6, &long_line, "src/longline.rs"),              // long single line
    ];

    for (id, content, path) in &files {
        processor.add_change(FileChange {
            file_id: *id,
            old_content: String::new(),
            new_content: content.to_string(),
            tenant_id: "batch-edge".to_string(),
            branch: Some("main".to_string()),
            file_path: path.to_string(),
            base_point: None,
            relative_path: None,
            file_hash: None,
        });
    }

    // Should not crash on any edge case
    let stats = processor.flush(6).await.unwrap();
    assert_eq!(stats.files_processed, 6);

    // Normal file should be searchable
    let results = search_exact(
        &db,
        "normal_code",
        &SearchOptions {
            tenant_id: Some("batch-edge".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);

    // Emoji file should be searchable
    let results = search_exact(
        &db,
        "emoji_file",
        &SearchOptions {
            tenant_id: Some("batch-edge".to_string()),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(results.matches.len(), 1);
}

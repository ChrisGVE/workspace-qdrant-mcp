//! FTS5 Integration Tests — Basic Operations
//!
//! Tests covering fundamental FTS5 pipeline operations:
//! - File add -> FtsBatchProcessor -> search finds content
//! - File update (diff) -> search finds new, not old content
//! - File delete -> search no longer finds deleted content

use tempfile::TempDir;

use workspace_qdrant_core::fts_batch_processor::{
    FileChange, FtsBatchConfig, FtsBatchProcessor,
};
use workspace_qdrant_core::search_db::SearchDbManager;
use workspace_qdrant_core::text_search::{search_exact, SearchOptions};

async fn setup_db() -> (TempDir, SearchDbManager) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();
    (tmp, manager)
}

// ---------------------------------------------------------------------------
// 1. File add -> search finds content
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
            None,
            None,
            None,
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
        base_point: None,
        relative_path: None,
        file_hash: None,
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
// 2. File update (content changed) -> search finds new content
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
        base_point: None,
        relative_path: None,
        file_hash: None,
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
        base_point: None,
        relative_path: None,
        file_hash: None,
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
        base_point: None,
        relative_path: None,
        file_hash: None,
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
        base_point: None,
        relative_path: None,
        file_hash: None,
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
// 3. File delete -> search no longer finds deleted content
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
            None,
            None,
            None,
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
        .full_rewrite(1, "fn proj_a_func() {}", "proj-a", Some("main"), "src/a.rs", None, None, None)
        .await
        .unwrap();
    processor
        .full_rewrite(2, "fn proj_a_other() {}", "proj-a", Some("main"), "src/b.rs", None, None, None)
        .await
        .unwrap();
    processor
        .full_rewrite(3, "fn proj_b_func() {}", "proj-b", Some("main"), "src/c.rs", None, None, None)
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

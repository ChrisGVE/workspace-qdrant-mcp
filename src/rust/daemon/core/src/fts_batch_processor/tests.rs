use super::*;
use crate::search_db::SearchDbManager;
use tempfile::TempDir;

/// Helper to construct a FileChange for tests (new fields default to None).
fn test_change(
    file_id: i64,
    old_content: &str,
    new_content: &str,
    tenant_id: &str,
    branch: Option<&str>,
    file_path: &str,
) -> FileChange {
    FileChange {
        file_id,
        old_content: old_content.to_string(),
        new_content: new_content.to_string(),
        tenant_id: tenant_id.to_string(),
        branch: branch.map(|s| s.to_string()),
        file_path: file_path.to_string(),
        base_point: None,
        relative_path: None,
        file_hash: None,
    }
}

async fn setup_db() -> (TempDir, SearchDbManager) {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();
    (tmp, manager)
}

#[tokio::test]
async fn test_default_config() {
    let config = FtsBatchConfig::default();
    assert_eq!(config.burst_threshold, DEFAULT_BURST_THRESHOLD);
    assert_eq!(config.burst_threshold, 10);
}

#[tokio::test]
async fn test_should_use_batch_mode() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    assert!(!processor.should_use_batch_mode(0));
    assert!(!processor.should_use_batch_mode(5));
    assert!(!processor.should_use_batch_mode(10)); // <= threshold
    assert!(processor.should_use_batch_mode(11));
    assert!(processor.should_use_batch_mode(100));
}

#[tokio::test]
async fn test_flush_empty() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let stats = processor.flush(0).await.unwrap();
    assert_eq!(stats.files_processed, 0);
    assert_eq!(stats.total_affected(), 0);
}

#[tokio::test]
async fn test_single_file_new_ingestion() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    processor.add_change(test_change(
        1,
        "",
        "line 1\nline 2\nline 3",
        "proj-a",
        Some("main"),
        "/src/main.rs",
    ));

    // Queue depth = 1, below threshold => single-file mode
    let stats = processor.flush(1).await.unwrap();
    assert_eq!(stats.files_processed, 1);
    assert_eq!(stats.lines_inserted, 3);
    assert!(!stats.batch_mode);

    // Verify lines in DB
    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count, 3);

    // Verify file_metadata
    let tenant: String =
        sqlx::query_scalar("SELECT tenant_id FROM file_metadata WHERE file_id = 1")
            .fetch_one(db.pool())
            .await
            .unwrap();
    assert_eq!(tenant, "proj-a");

    db.close().await;
}

#[tokio::test]
async fn test_batch_mode_multiple_files() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Add 3 file changes
    for i in 1..=3 {
        processor.add_change(test_change(
            i,
            "",
            &format!("fn file{}() {{}}\nfn helper{}() {{}}", i, i),
            "proj-a",
            Some("main"),
            &format!("/src/file{}.rs", i),
        ));
    }

    assert_eq!(processor.pending_count(), 3);

    // Queue depth = 20, above threshold => batch mode
    let stats = processor.flush(20).await.unwrap();
    assert_eq!(stats.files_processed, 3);
    assert_eq!(stats.lines_inserted, 6); // 2 lines per file x 3 files
    assert!(stats.batch_mode);

    // Verify all files have lines
    for i in 1..=3_i64 {
        let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = ?1")
            .bind(i)
            .fetch_one(db.pool())
            .await
            .unwrap();
        assert_eq!(count, 2, "File {} should have 2 lines", i);
    }

    db.close().await;
}

#[tokio::test]
async fn test_update_with_diff() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // First: ingest original content
    processor.add_change(test_change(
        1,
        "",
        "line 1\nline 2\nline 3",
        "proj-a",
        Some("main"),
        "/src/main.rs",
    ));
    processor.flush(0).await.unwrap();

    // Second: update with modified content
    processor.add_change(test_change(
        1,
        "line 1\nline 2\nline 3",
        "line 1\nline 2 modified\nline 3\nline 4",
        "proj-a",
        Some("main"),
        "/src/main.rs",
    ));
    let stats = processor.flush(0).await.unwrap();

    assert_eq!(stats.files_processed, 1);
    // Should have some combination of unchanged/updated/inserted
    assert!(stats.lines_unchanged > 0 || stats.lines_updated > 0);

    // Verify final line count = 4
    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count, 4);

    db.close().await;
}

#[tokio::test]
async fn test_full_rewrite() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let stats = processor
        .full_rewrite(
            1,
            "alpha\nbeta\ngamma",
            "proj-a",
            Some("main"),
            "/src/lib.rs",
            None,
            None,
            None,
        )
        .await
        .unwrap();

    assert_eq!(stats.files_processed, 1);
    assert_eq!(stats.lines_inserted, 3);

    // Verify lines exist
    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count, 3);

    // Full rewrite again with different content
    let stats2 = processor
        .full_rewrite(
            1,
            "one\ntwo",
            "proj-a",
            Some("main"),
            "/src/lib.rs",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(stats2.lines_inserted, 2);

    let count2: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count2, 2);

    db.close().await;
}

#[tokio::test]
async fn test_delete_file() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Insert some lines first
    processor
        .full_rewrite(
            1,
            "a\nb\nc",
            "proj",
            Some("main"),
            "/file.rs",
            None,
            None,
            None,
        )
        .await
        .unwrap();

    let deleted = processor.delete_file(1).await.unwrap();
    assert_eq!(deleted, 3);

    // Verify no lines remain
    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count, 0);

    // Verify file_metadata also deleted
    let md_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE file_id = 1")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(md_count, 0);

    db.close().await;
}

#[tokio::test]
async fn test_delete_tenant() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Insert files for two tenants
    processor
        .full_rewrite(
            1,
            "a\nb",
            "proj-a",
            Some("main"),
            "/f1.rs",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    processor
        .full_rewrite(
            2,
            "c\nd\ne",
            "proj-a",
            Some("main"),
            "/f2.rs",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    processor
        .full_rewrite(
            3,
            "x\ny",
            "proj-b",
            Some("main"),
            "/f3.rs",
            None,
            None,
            None,
        )
        .await
        .unwrap();

    // Delete proj-a
    let deleted = processor.delete_tenant("proj-a").await.unwrap();
    assert_eq!(deleted, 5); // 2 + 3 lines

    // proj-b should be untouched
    let count_b: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 3")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count_b, 2);

    db.close().await;
}

#[tokio::test]
async fn test_fts5_searchable_after_flush() {
    use sqlx::Row;
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    processor.add_change(test_change(
        1,
        "",
        "fn search_target() {}\nfn other_function() {}",
        "proj-a",
        Some("main"),
        "/src/main.rs",
    ));
    processor.flush(0).await.unwrap();

    // FTS5 should be searchable after flush
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("search_target")
        .fetch_all(db.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert!(rows[0]
        .get::<String, _>("content")
        .contains("search_target"));

    db.close().await;
}

#[tokio::test]
async fn test_batch_mode_fts5_searchable() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Multiple files in batch mode
    processor.add_change(test_change(
        1,
        "",
        "fn batch_alpha() {}",
        "proj-a",
        Some("main"),
        "/src/a.rs",
    ));
    processor.add_change(test_change(
        2,
        "",
        "fn batch_beta() {}",
        "proj-a",
        Some("main"),
        "/src/b.rs",
    ));

    // Force batch mode with high queue depth
    processor.flush(50).await.unwrap();

    // Both should be searchable
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("batch_alpha")
        .fetch_all(db.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);

    let rows2 = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("batch_beta")
        .fetch_all(db.pool())
        .await
        .unwrap();
    assert_eq!(rows2.len(), 1);

    db.close().await;
}

#[tokio::test]
async fn test_scoped_search_after_flush() {
    use sqlx::Row;
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Two files in different projects
    processor.add_change(test_change(
        1,
        "",
        "fn shared_name() {}",
        "proj-x",
        Some("main"),
        "/src/x.rs",
    ));
    processor.add_change(test_change(
        2,
        "",
        "fn shared_name_v2() {}",
        "proj-y",
        Some("main"),
        "/src/y.rs",
    ));
    processor.flush(0).await.unwrap();

    // Scoped search for proj-x only
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_PROJECT_SQL)
        .bind("shared_name")
        .bind("proj-x")
        .fetch_all(db.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<String, _>("tenant_id"), "proj-x");

    db.close().await;
}

#[tokio::test]
async fn test_large_batch_throughput() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    // Simulate 50 files x 300 lines each = 15,000 lines
    for i in 1..=50 {
        let content: String = (0..300)
            .map(|j| format!("fn file{}_line{}() {{}}", i, j))
            .collect::<Vec<_>>()
            .join("\n");

        processor.add_change(test_change(
            i,
            "",
            &content,
            "proj-perf",
            Some("main"),
            &format!("/src/file{}.rs", i),
        ));
    }

    // Batch mode
    let stats = processor.flush(100).await.unwrap();
    assert_eq!(stats.files_processed, 50);
    assert_eq!(stats.lines_inserted, 15_000);
    assert!(stats.batch_mode);

    // Should complete in reasonable time (< 10s for 15K lines)
    assert!(
        stats.processing_time_ms < 10_000,
        "Batch processing took {}ms, expected < 10000ms",
        stats.processing_time_ms
    );

    // Verify total count
    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count, 15_000);

    db.close().await;
}

#[tokio::test]
async fn test_custom_burst_threshold() {
    let (_tmp, db) = setup_db().await;
    let config = FtsBatchConfig { burst_threshold: 5 };
    let processor = FtsBatchProcessor::new(&db, config);

    assert!(!processor.should_use_batch_mode(4));
    assert!(!processor.should_use_batch_mode(5));
    assert!(processor.should_use_batch_mode(6));
}

#[tokio::test]
async fn test_delete_nonexistent_file() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let deleted = processor.delete_file(999).await.unwrap();
    assert_eq!(deleted, 0);

    db.close().await;
}

#[tokio::test]
async fn test_delete_nonexistent_tenant() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());

    let deleted = processor.delete_tenant("nonexistent").await.unwrap();
    assert_eq!(deleted, 0);

    db.close().await;
}

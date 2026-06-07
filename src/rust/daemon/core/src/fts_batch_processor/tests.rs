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
    // Env overrides are absent in tests → compiled defaults apply
    assert_eq!(
        config.single_mode_threshold_bytes,
        DEFAULT_SINGLE_MODE_THRESHOLD_BYTES
    );
    assert_eq!(config.hard_cap_bytes, DEFAULT_HARD_CAP_BYTES);
}

/// Files whose old+new content exceeds `hard_cap_bytes` must be skipped
/// entirely — no code_lines writes, counted in `files_skipped_too_large` (#103).
#[tokio::test]
async fn test_hard_cap_skips_oversized_file() {
    let (_tmp, db) = setup_db().await;
    let config = FtsBatchConfig {
        hard_cap_bytes: 64,
        ..FtsBatchConfig::default()
    };
    let mut processor = FtsBatchProcessor::new(&db, config);

    let big = "x".repeat(100);
    processor.add_change(test_change(1, "", &big, "proj-a", Some("main"), "/big.js"));
    processor.add_change(test_change(
        2,
        "",
        "small line",
        "proj-a",
        Some("main"),
        "/small.rs",
    ));

    let stats = processor.flush(0).await.unwrap();
    assert_eq!(stats.files_skipped_too_large, 1);
    assert_eq!(stats.files_processed, 1, "small file still processed");

    let big_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(big_count, 0, "oversized file must not be indexed");
    db.close().await;
}

/// When accumulated content exceeds `single_mode_threshold_bytes`, flush
/// must fall back to single-file mode even if queue depth favors batch (#103).
#[tokio::test]
async fn test_bytes_budget_forces_single_file_mode() {
    let (_tmp, db) = setup_db().await;
    let config = FtsBatchConfig {
        single_mode_threshold_bytes: 32,
        ..FtsBatchConfig::default()
    };
    let mut processor = FtsBatchProcessor::new(&db, config);

    for i in 1..=3 {
        processor.add_change(test_change(
            i,
            "",
            "this line alone is over budget",
            "proj-a",
            Some("main"),
            &format!("/f{}.rs", i),
        ));
    }

    // queue_depth 20 > burst_threshold 10 → batch mode WOULD apply,
    // but the bytes budget forces single-file mode.
    let stats = processor.flush(20).await.unwrap();
    assert!(!stats.batch_mode, "bytes budget must override batch mode");
    assert_eq!(stats.files_processed, 3);
    db.close().await;
}

/// `full_rewrite` (no-diff path) must honor the hard cap too.
#[tokio::test]
async fn test_full_rewrite_honors_hard_cap() {
    let (_tmp, db) = setup_db().await;
    let config = FtsBatchConfig {
        hard_cap_bytes: 64,
        ..FtsBatchConfig::default()
    };
    let processor = FtsBatchProcessor::new(&db, config);

    let big = "y".repeat(200);
    let stats = processor
        .full_rewrite(
            7,
            &big,
            "proj-a",
            Some("main"),
            "/big.min.js",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert_eq!(stats.files_skipped_too_large, 1);
    assert_eq!(stats.lines_inserted, 0);

    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 7")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count, 0);
    db.close().await;
}

#[tokio::test]
async fn test_should_use_batch_mode() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
    assert!(!processor.should_use_batch_mode(0));
    assert!(!processor.should_use_batch_mode(5));
    assert!(!processor.should_use_batch_mode(10));
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
    let stats = processor.flush(1).await.unwrap();
    assert_eq!(stats.files_processed, 1);
    assert_eq!(stats.lines_inserted, 3);
    assert!(!stats.batch_mode);
    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count, 3);
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
    let stats = processor.flush(20).await.unwrap();
    assert_eq!(stats.files_processed, 3);
    assert_eq!(stats.lines_inserted, 6);
    assert!(stats.batch_mode);
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
    processor.add_change(test_change(
        1,
        "",
        "line 1\nline 2\nline 3",
        "proj-a",
        Some("main"),
        "/src/main.rs",
    ));
    processor.flush(0).await.unwrap();
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
    assert!(stats.lines_unchanged > 0 || stats.lines_updated > 0);
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
    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count, 3);
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
    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count, 0);
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
    let deleted = processor.delete_tenant("proj-a").await.unwrap();
    assert_eq!(deleted, 5);
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
    processor.flush(50).await.unwrap();
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
    let stats = processor.flush(100).await.unwrap();
    assert_eq!(stats.files_processed, 50);
    assert_eq!(stats.lines_inserted, 15_000);
    assert!(stats.batch_mode);
    assert!(
        stats.processing_time_ms < 30_000,
        "Batch processing took {}ms, expected < 30000ms",
        stats.processing_time_ms
    );
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
    let config = FtsBatchConfig {
        burst_threshold: 5,
        ..FtsBatchConfig::default()
    };
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

// ============================================================================
// Regression tests for TOCTOU fix and flush requeue
// ============================================================================

/// Regression: delete_file must leave no orphaned FTS5 entries.
#[tokio::test]
async fn test_delete_file_no_orphaned_fts5_entries() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
    processor
        .full_rewrite(
            1,
            "fn unique_marker_alpha() {}\nfn unique_marker_beta() {}",
            "proj",
            Some("main"),
            "/file.rs",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    let pre_rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("unique_marker_alpha")
        .fetch_all(db.pool())
        .await
        .unwrap();
    assert_eq!(pre_rows.len(), 1, "FTS5 entry should exist before delete");
    let deleted = processor.delete_file(1).await.unwrap();
    assert_eq!(deleted, 2);
    let post_alpha = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("unique_marker_alpha")
        .fetch_all(db.pool())
        .await
        .unwrap();
    assert_eq!(
        post_alpha.len(),
        0,
        "FTS5 should have zero entries for deleted file (alpha)"
    );
    let post_beta = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("unique_marker_beta")
        .fetch_all(db.pool())
        .await
        .unwrap();
    assert_eq!(
        post_beta.len(),
        0,
        "FTS5 should have zero entries for deleted file (beta)"
    );
    let code_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(code_count, 0);
    db.close().await;
}

/// Regression: delete_file after content replacement leaves no stale FTS5 entries.
#[tokio::test]
async fn test_delete_file_after_update_no_stale_fts5() {
    let (_tmp, db) = setup_db().await;
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
    processor
        .full_rewrite(
            1,
            "line_a\nline_b",
            "proj",
            Some("main"),
            "/file.rs",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    // Replace content via full_rewrite (simulates file content changing)
    processor
        .full_rewrite(
            1,
            "line_a\nline_b\nline_c_appended\nline_d_appended",
            "proj",
            Some("main"),
            "/file.rs",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(count, 4);
    let deleted = processor.delete_file(1).await.unwrap();
    assert_eq!(deleted, 4);
    for term in &["line_a", "line_b", "line_c_appended", "line_d_appended"] {
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind(*term)
            .fetch_all(db.pool())
            .await
            .unwrap();
        assert_eq!(
            rows.len(),
            0,
            "FTS5 should have no entries for '{}' after delete",
            term
        );
    }
    db.close().await;
}

/// Regression: flush requeues pending changes on single-file mode failure.
#[tokio::test]
async fn test_flush_requeues_on_single_file_failure() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
    processor.add_change(test_change(
        1,
        "",
        "fn requeue_test() {}",
        "proj",
        Some("main"),
        "/file.rs",
    ));
    processor.add_change(test_change(
        2,
        "",
        "fn requeue_test2() {}",
        "proj",
        Some("main"),
        "/file2.rs",
    ));
    assert_eq!(processor.pending_count(), 2);
    db.close().await;
    let result = processor.flush(0).await;
    assert!(result.is_err(), "flush should fail with closed pool");
    assert!(
        processor.pending_count() > 0,
        "pending should be non-empty after failed flush"
    );
    assert_eq!(
        processor.pending_count(),
        2,
        "all 2 changes should be requeued"
    );
}

/// Regression: flush requeues pending changes on batch mode failure.
#[tokio::test]
async fn test_flush_requeues_on_batch_mode_failure() {
    let (_tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
    for i in 1..=3 {
        processor.add_change(test_change(
            i,
            "",
            &format!("fn batch_requeue_{}() {{}}", i),
            "proj",
            Some("main"),
            &format!("/file{}.rs", i),
        ));
    }
    assert_eq!(processor.pending_count(), 3);
    db.close().await;
    let result = processor.flush(100).await;
    assert!(result.is_err(), "flush should fail with closed pool");
    assert_eq!(
        processor.pending_count(),
        3,
        "all 3 changes should be requeued after batch failure"
    );
}

/// Regression: successful flush after a failed flush with requeue.
#[tokio::test]
async fn test_flush_retry_after_requeue_succeeds() {
    let (tmp, db) = setup_db().await;
    let mut processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
    processor.add_change(test_change(
        1,
        "",
        "fn retry_content() {}",
        "proj",
        Some("main"),
        "/file.rs",
    ));
    db.close().await;
    let result = processor.flush(0).await;
    assert!(result.is_err());
    assert_eq!(processor.pending_count(), 1);
    let db_path = tmp.path().join("search.db");
    let db2 = SearchDbManager::new(&db_path).await.unwrap();
    let mut processor2 = FtsBatchProcessor::new(&db2, FtsBatchConfig::default());
    processor2.add_change(test_change(
        1,
        "",
        "fn retry_content() {}",
        "proj",
        Some("main"),
        "/file.rs",
    ));
    let stats = processor2.flush(0).await.unwrap();
    assert_eq!(stats.files_processed, 1);
    assert_eq!(stats.lines_inserted, 1);
    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(db2.pool())
        .await
        .unwrap();
    assert_eq!(count, 1);
    db2.close().await;
}

/// Regression: delete_file cleans up file_metadata even when no code_lines exist.
#[tokio::test]
async fn test_delete_file_metadata_only_cleanup() {
    let (_tmp, db) = setup_db().await;
    sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
        .bind(42_i64)
        .bind("proj")
        .bind(Some("main"))
        .bind("/orphaned.rs")
        .bind(None::<String>)
        .bind(None::<String>)
        .bind(None::<String>)
        .execute(db.pool())
        .await
        .unwrap();
    let md_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE file_id = 42")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(md_count, 1);
    let processor = FtsBatchProcessor::new(&db, FtsBatchConfig::default());
    let deleted = processor.delete_file(42).await.unwrap();
    assert_eq!(deleted, 0, "no code_lines to delete");
    let md_after: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE file_id = 42")
        .fetch_one(db.pool())
        .await
        .unwrap();
    assert_eq!(md_after, 0, "file_metadata should be deleted");
    db.close().await;
}

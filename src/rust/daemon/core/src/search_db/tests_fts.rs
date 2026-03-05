//! Tests for FTS5 trigram search, file_metadata scoping, and related operations.

#![cfg(test)]

use super::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_fts5_table_exists() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='code_lines_fts')",
    )
    .fetch_one(manager.pool())
    .await
    .unwrap();

    assert!(
        exists,
        "code_lines_fts virtual table should exist after migration v3"
    );
    manager.close().await;
}

#[tokio::test]
async fn test_fts5_match_basic() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert test code lines
    let lines = vec![
        "fn main() {",
        "    println!(\"hello world\");",
        "    let x = 42;",
        "}",
    ];
    for (i, line) in lines.iter().enumerate() {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(crate::code_lines_schema::initial_seq(i))
            .bind(*line)
            .execute(manager.pool())
            .await
            .unwrap();
    }

    // Rebuild FTS index
    manager.rebuild_fts().await.unwrap();

    // Search for "println" -- trigram matching
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("println")
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 1, "Should find exactly 1 line with 'println'");
    assert_eq!(
        rows[0].get::<String, _>("content"),
        "    println!(\"hello world\");"
    );

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_match_multiple_results() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let lines = vec![
        "fn foo() -> i32 {",
        "    return 1;",
        "}",
        "fn bar() -> i32 {",
        "    return 2;",
        "}",
    ];
    for (i, line) in lines.iter().enumerate() {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(crate::code_lines_schema::initial_seq(i))
            .bind(*line)
            .execute(manager.pool())
            .await
            .unwrap();
    }

    manager.rebuild_fts().await.unwrap();

    // Search for "return" -- should match 2 lines
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("return")
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 2, "Should find 2 lines with 'return'");
    assert!(rows[0].get::<String, _>("content").contains("return 1"));
    assert!(rows[1].get::<String, _>("content").contains("return 2"));

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_search_by_file() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // File 1
    sqlx::query(
        "INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn hello() {}')",
    )
    .execute(manager.pool())
    .await
    .unwrap();
    // File 2
    sqlx::query(
        "INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn hello_world() {}')",
    )
    .execute(manager.pool())
    .await
    .unwrap();

    manager.rebuild_fts().await.unwrap();

    // Search "hello" scoped to file 2
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_FILE_SQL)
        .bind("hello")
        .bind(2_i64)
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<i64, _>("file_id"), 2);
    assert!(rows[0].get::<String, _>("content").contains("hello_world"));

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_no_results() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    sqlx::query(
        "INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn main() {}')",
    )
    .execute(manager.pool())
    .await
    .unwrap();

    manager.rebuild_fts().await.unwrap();

    // Search for something that doesn't exist
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("nonexistent_xyz")
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 0, "Should find no results for nonexistent term");
    manager.close().await;
}

#[tokio::test]
async fn test_fts5_rebuild_after_insert() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert and rebuild
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'first line')")
        .execute(manager.pool())
        .await
        .unwrap();
    manager.rebuild_fts().await.unwrap();

    // Verify first line is findable
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("first")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);

    // Insert more without rebuild -- FTS won't see it yet
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 2000.0, 'second line')")
        .execute(manager.pool())
        .await
        .unwrap();

    let rows_before = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("second")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(
        rows_before.len(),
        0,
        "New content not visible before rebuild"
    );

    // Rebuild and verify
    manager.rebuild_fts().await.unwrap();
    let rows_after = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("second")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows_after.len(), 1, "New content visible after rebuild");
    assert_eq!(rows_after[0].get::<String, _>("content"), "second line");

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_rebuild_after_delete() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    sqlx::query(
        "INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'deletable line')",
    )
    .execute(manager.pool())
    .await
    .unwrap();
    manager.rebuild_fts().await.unwrap();

    // Verify it's findable
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("deletable")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);

    // Delete and rebuild
    sqlx::query("DELETE FROM code_lines WHERE file_id = 1")
        .execute(manager.pool())
        .await
        .unwrap();
    manager.rebuild_fts().await.unwrap();

    let rows_after = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("deletable")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(
        rows_after.len(),
        0,
        "Deleted content should not appear after rebuild"
    );

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_optimize() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert data and rebuild
    for i in 0..100 {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(crate::code_lines_schema::initial_seq(i))
            .bind(format!("line {} content", i))
            .execute(manager.pool())
            .await
            .unwrap();
    }
    manager.rebuild_fts().await.unwrap();

    // Optimize should not fail
    manager.optimize_fts().await.unwrap();

    // Verify search still works after optimize
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("content")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 100, "All 100 lines should match 'content'");

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_rebuild_and_maybe_optimize() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert 1500 lines (above threshold)
    let mut tx = manager.pool().begin().await.unwrap();
    for i in 0..1500 {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(crate::code_lines_schema::initial_seq(i))
            .bind(format!("code line {}", i))
            .execute(&mut *tx)
            .await
            .unwrap();
    }
    tx.commit().await.unwrap();

    // Should rebuild + optimize (1500 > 1000 threshold)
    manager.rebuild_and_maybe_optimize_fts(1500).await.unwrap();

    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("code line")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1500);

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_rebuild_below_threshold_no_optimize() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert 50 lines (below threshold)
    for i in 0..50 {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(crate::code_lines_schema::initial_seq(i))
            .bind(format!("small batch {}", i))
            .execute(manager.pool())
            .await
            .unwrap();
    }

    // Should rebuild only (50 < 1000 threshold)
    manager.rebuild_and_maybe_optimize_fts(50).await.unwrap();

    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("small batch")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 50);

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_unicode_search() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let lines = vec![
        "// \u{65e5}\u{672c}\u{8a9e}\u{30b3}\u{30e1}\u{30f3}\u{30c8}",
        "let emoji = \"\u{1f980}\";",
        "fn process_data() {}",
    ];
    for (i, line) in lines.iter().enumerate() {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(crate::code_lines_schema::initial_seq(i))
            .bind(*line)
            .execute(manager.pool())
            .await
            .unwrap();
    }
    manager.rebuild_fts().await.unwrap();

    // Search for ASCII substring in mixed content
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("process_data")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<String, _>("content"), "fn process_data() {}");

    manager.close().await;
}

#[tokio::test]
async fn test_fts5_external_content_linkage() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert a line and get its line_id
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'external content test')")
        .execute(manager.pool()).await.unwrap();

    let line_id: i64 =
        sqlx::query_scalar("SELECT line_id FROM code_lines WHERE file_id = 1 AND seq = 1000.0")
            .fetch_one(manager.pool())
            .await
            .unwrap();

    manager.rebuild_fts().await.unwrap();

    // FTS5 match should return the correct line_id
    let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("external content")
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get::<i64, _>("line_id"),
        line_id,
        "FTS5 should return the correct line_id from external content table"
    );

    manager.close().await;
}

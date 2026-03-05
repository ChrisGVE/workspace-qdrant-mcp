//! Tests for code_lines table operations, gap insertion, and rebalancing.

#![cfg(test)]

use super::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_code_lines_table_exists() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='code_lines')",
    )
    .fetch_one(manager.pool())
    .await
    .unwrap();

    assert!(exists, "code_lines table should exist after migration v2");
    manager.close().await;
}

#[tokio::test]
async fn test_code_lines_index_exists() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_code_lines_file')",
    )
    .fetch_one(manager.pool())
    .await
    .unwrap();

    assert!(exists, "idx_code_lines_file index should exist");
    manager.close().await;
}

#[tokio::test]
async fn test_code_lines_insert_and_query() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert lines for a hypothetical file_id=1
    let lines = vec!["fn main() {", "    println!(\"hello\");", "}"];
    for (i, line) in lines.iter().enumerate() {
        let seq = crate::code_lines_schema::initial_seq(i);
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (?1, ?2, ?3)")
            .bind(1_i64)
            .bind(seq)
            .bind(*line)
            .execute(manager.pool())
            .await
            .unwrap();
    }

    // Query with line numbers
    let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
        .bind(1_i64)
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].get::<i64, _>("line_number"), 1);
    assert_eq!(rows[0].get::<String, _>("content"), "fn main() {");
    assert_eq!(rows[1].get::<i64, _>("line_number"), 2);
    assert_eq!(rows[2].get::<i64, _>("line_number"), 3);
    assert_eq!(rows[2].get::<String, _>("content"), "}");

    manager.close().await;
}

#[tokio::test]
async fn test_code_lines_gap_insertion() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert 3 lines with standard gaps
    for i in 0..3 {
        let seq = crate::code_lines_schema::initial_seq(i);
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(seq)
            .bind(format!("line {}", i + 1))
            .execute(manager.pool())
            .await
            .unwrap();
    }

    // Insert a new line between line 1 (seq=1000) and line 2 (seq=2000)
    let mid = crate::code_lines_schema::midpoint_seq(1000.0, 2000.0);
    assert_eq!(mid, 1500.0);
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, 'inserted line')")
        .bind(mid)
        .execute(manager.pool())
        .await
        .unwrap();

    // Query and verify ordering
    let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
        .bind(1_i64)
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0].get::<String, _>("content"), "line 1");
    assert_eq!(rows[1].get::<String, _>("content"), "inserted line");
    assert_eq!(rows[1].get::<i64, _>("line_number"), 2);
    assert_eq!(rows[2].get::<String, _>("content"), "line 2");
    assert_eq!(rows[2].get::<i64, _>("line_number"), 3);
    assert_eq!(rows[3].get::<String, _>("content"), "line 3");

    manager.close().await;
}

#[tokio::test]
async fn test_code_lines_unique_constraint() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'first')")
        .execute(manager.pool())
        .await
        .unwrap();

    // Inserting same (file_id, seq) should fail
    let result = sqlx::query(
        "INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'duplicate')",
    )
    .execute(manager.pool())
    .await;

    assert!(result.is_err(), "Should reject duplicate (file_id, seq)");

    manager.close().await;
}

#[tokio::test]
async fn test_code_lines_delete_by_file() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert lines for two files
    for i in 0..3 {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (?1, ?2, ?3)")
            .bind(1_i64)
            .bind(crate::code_lines_schema::initial_seq(i))
            .bind(format!("file1 line {}", i))
            .execute(manager.pool())
            .await
            .unwrap();
    }
    for i in 0..2 {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (?1, ?2, ?3)")
            .bind(2_i64)
            .bind(crate::code_lines_schema::initial_seq(i))
            .bind(format!("file2 line {}", i))
            .execute(manager.pool())
            .await
            .unwrap();
    }

    // Delete all lines for file 1
    sqlx::query("DELETE FROM code_lines WHERE file_id = 1")
        .execute(manager.pool())
        .await
        .unwrap();

    // File 1 should have no lines
    let count1: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(count1, 0);

    // File 2 should still have its lines
    let count2: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 2")
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(count2, 2);

    manager.close().await;
}

#[tokio::test]
async fn test_code_lines_unicode_content() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let unicode_lines = vec![
        "// \u{65e5}\u{672c}\u{8a9e}\u{30b3}\u{30e1}\u{30f3}\u{30c8}",
        "let \u{03c0} = 3.14159;",
        "println!(\"\u{1f980} Rust!\");",
        "// Box-drawing: \u{250c}\u{2500}\u{2510}\u{2502}\u{2514}\u{2500}\u{2518}",
    ];

    for (i, line) in unicode_lines.iter().enumerate() {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(crate::code_lines_schema::initial_seq(i))
            .bind(*line)
            .execute(manager.pool())
            .await
            .unwrap();
    }

    let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
        .bind(1_i64)
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0].get::<String, _>("content"), unicode_lines[0]);
    assert_eq!(rows[1].get::<String, _>("content"), unicode_lines[1]);
    assert_eq!(rows[2].get::<String, _>("content"), unicode_lines[2]);

    manager.close().await;
}

#[tokio::test]
async fn test_code_lines_empty_content() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Empty lines are valid (blank lines in source code)
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, '')")
        .execute(manager.pool())
        .await
        .unwrap();

    let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
        .bind(1_i64)
        .fetch_all(manager.pool())
        .await
        .unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get::<String, _>("content"), "");

    manager.close().await;
}

#[tokio::test]
async fn test_code_lines_1000_lines() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Batch insert 1000 lines
    let mut tx = manager.pool().begin().await.unwrap();
    for i in 0..1000 {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(crate::code_lines_schema::initial_seq(i))
            .bind(format!("line {}", i + 1))
            .execute(&mut *tx)
            .await
            .unwrap();
    }
    tx.commit().await.unwrap();

    // Verify count
    let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
        .fetch_one(manager.pool())
        .await
        .unwrap();
    assert_eq!(count, 1000);

    // Verify line numbers 1-1000
    let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
        .bind(1_i64)
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 1000);
    assert_eq!(rows[0].get::<i64, _>("line_number"), 1);
    assert_eq!(rows[0].get::<String, _>("content"), "line 1");
    assert_eq!(rows[999].get::<i64, _>("line_number"), 1000);
    assert_eq!(rows[999].get::<String, _>("content"), "line 1000");

    manager.close().await;
}

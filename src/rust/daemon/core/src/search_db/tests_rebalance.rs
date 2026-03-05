//! Tests for seq gap management, rebalancing, and line insertion operations.

#![cfg(test)]

use super::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_insert_line_at_end_empty_file() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let result = manager.insert_line_at_end(1, "first line").await.unwrap();
    assert_eq!(result.seq, 1000.0);
    assert!(result.line_id > 0);

    manager.close().await;
}

#[tokio::test]
async fn test_insert_line_at_end_existing_lines() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    manager.insert_line_at_end(1, "line 1").await.unwrap();
    let r2 = manager.insert_line_at_end(1, "line 2").await.unwrap();
    assert_eq!(r2.seq, 2000.0);

    let r3 = manager.insert_line_at_end(1, "line 3").await.unwrap();
    assert_eq!(r3.seq, 3000.0);

    manager.close().await;
}

#[tokio::test]
async fn test_insert_line_at_start_empty_file() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let (result, rebalanced) = manager.insert_line_at_start(1, "first line").await.unwrap();
    assert_eq!(result.seq, 1000.0);
    assert!(!rebalanced);

    manager.close().await;
}

#[tokio::test]
async fn test_insert_line_at_start_existing_lines() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert initial line at seq=1000.0
    manager
        .insert_line_at_end(1, "original first")
        .await
        .unwrap();

    // Insert at start: should get seq=500.0
    let (result, rebalanced) = manager.insert_line_at_start(1, "new first").await.unwrap();
    assert_eq!(result.seq, 500.0);
    assert!(!rebalanced);

    // Verify ordering
    let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
        .bind(1_i64)
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].get::<String, _>("content"), "new first");
    assert_eq!(rows[1].get::<String, _>("content"), "original first");

    manager.close().await;
}

#[tokio::test]
async fn test_insert_line_between_basic() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert two lines with standard gap
    manager.insert_line_at_end(1, "line 1").await.unwrap();
    manager.insert_line_at_end(1, "line 2").await.unwrap();

    // Insert between them: midpoint of 1000.0 and 2000.0 = 1500.0
    let (result, rebalanced) = manager
        .insert_line_between(1, 1000.0, 2000.0, "inserted")
        .await
        .unwrap();
    assert_eq!(result.seq, 1500.0);
    assert!(!rebalanced);

    // Verify ordering
    let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
        .bind(1_i64)
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].get::<String, _>("content"), "line 1");
    assert_eq!(rows[1].get::<String, _>("content"), "inserted");
    assert_eq!(rows[2].get::<String, _>("content"), "line 2");

    manager.close().await;
}

#[tokio::test]
async fn test_needs_rebalance() {
    assert!(SearchDbManager::needs_rebalance(0.0005));
    assert!(SearchDbManager::needs_rebalance(0.0001));
    assert!(!SearchDbManager::needs_rebalance(0.001));
    assert!(!SearchDbManager::needs_rebalance(1.0));
    assert!(!SearchDbManager::needs_rebalance(1000.0));
}

#[tokio::test]
async fn test_rebalance_basic() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert lines with cramped seq values
    for i in 0..5 {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(1.0 + i as f64 * 0.0001)
            .bind(format!("line {}", i))
            .execute(manager.pool())
            .await
            .unwrap();
    }

    // Rebalance
    let result = manager.rebalance_file_seqs(1).await.unwrap();
    assert_eq!(result.lines_rebalanced, 5);
    assert_eq!(result.new_gap, 1000.0);

    // Verify new seq values
    let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
        .bind(1_i64)
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0].get::<f64, _>("seq"), 1000.0);
    assert_eq!(rows[1].get::<f64, _>("seq"), 2000.0);
    assert_eq!(rows[2].get::<f64, _>("seq"), 3000.0);
    assert_eq!(rows[3].get::<f64, _>("seq"), 4000.0);
    assert_eq!(rows[4].get::<f64, _>("seq"), 5000.0);

    // Verify content order preserved
    assert_eq!(rows[0].get::<String, _>("content"), "line 0");
    assert_eq!(rows[4].get::<String, _>("content"), "line 4");

    manager.close().await;
}

#[tokio::test]
async fn test_rebalance_empty_file() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    let result = manager.rebalance_file_seqs(999).await.unwrap();
    assert_eq!(result.lines_rebalanced, 0);

    manager.close().await;
}

#[tokio::test]
async fn test_rebalance_file_local() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // File 1: cramped
    for i in 0..3 {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(0.5 + i as f64 * 0.0001)
            .bind(format!("f1 line {}", i))
            .execute(manager.pool())
            .await
            .unwrap();
    }

    // File 2: normal gaps
    for i in 0..3 {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (2, ?1, ?2)")
            .bind(crate::code_lines_schema::initial_seq(i))
            .bind(format!("f2 line {}", i))
            .execute(manager.pool())
            .await
            .unwrap();
    }

    // Rebalance file 1 only
    manager.rebalance_file_seqs(1).await.unwrap();

    // File 2 should be untouched
    let rows2 = sqlx::query("SELECT seq FROM code_lines WHERE file_id = 2 ORDER BY seq")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows2[0].get::<f64, _>("seq"), 1000.0);
    assert_eq!(rows2[1].get::<f64, _>("seq"), 2000.0);
    assert_eq!(rows2[2].get::<f64, _>("seq"), 3000.0);

    manager.close().await;
}

#[tokio::test]
async fn test_insert_between_triggers_rebalance() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert two lines very close together (gap = 0.0002)
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
        .bind(1.0)
        .bind("line a")
        .execute(manager.pool())
        .await
        .unwrap();
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
        .bind(1.0002)
        .bind("line b")
        .execute(manager.pool())
        .await
        .unwrap();

    // Insert between: gap = 0.0001, below MIN_SEQ_GAP => rebalance
    let (result, rebalanced) = manager
        .insert_line_between(1, 1.0, 1.0002, "line mid")
        .await
        .unwrap();
    assert!(rebalanced, "Should have triggered rebalance");

    // After rebalance, seqs should be 1000.0, 2000.0, 3000.0
    let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
        .bind(1_i64)
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].get::<f64, _>("seq"), 1000.0);
    assert_eq!(rows[1].get::<f64, _>("seq"), 2000.0);
    assert_eq!(rows[2].get::<f64, _>("seq"), 3000.0);

    // Content order preserved: a, mid, b
    assert_eq!(rows[0].get::<String, _>("content"), "line a");
    assert_eq!(rows[1].get::<String, _>("content"), "line mid");
    assert_eq!(rows[2].get::<String, _>("content"), "line b");

    // The returned seq should match the post-rebalance value
    assert_eq!(result.seq, 2000.0);

    manager.close().await;
}

#[tokio::test]
async fn test_get_adjacent_seqs() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert 3 lines
    for i in 0..3 {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(crate::code_lines_schema::initial_seq(i))
            .bind(format!("line {}", i))
            .execute(manager.pool())
            .await
            .unwrap();
    }

    // Middle line (seq=2000.0)
    let (before, after) = manager.get_adjacent_seqs(1, 2000.0).await.unwrap();
    assert_eq!(before, Some(1000.0));
    assert_eq!(after, Some(3000.0));

    // First line (seq=1000.0)
    let (before, after) = manager.get_adjacent_seqs(1, 1000.0).await.unwrap();
    assert_eq!(before, None);
    assert_eq!(after, Some(2000.0));

    // Last line (seq=3000.0)
    let (before, after) = manager.get_adjacent_seqs(1, 3000.0).await.unwrap();
    assert_eq!(before, Some(2000.0));
    assert_eq!(after, None);

    manager.close().await;
}

#[tokio::test]
async fn test_min_seq_gap() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // No lines => None
    let gap = manager.min_seq_gap(1).await.unwrap();
    assert_eq!(gap, None);

    // One line => None (need at least 2)
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'a')")
        .execute(manager.pool())
        .await
        .unwrap();
    let gap = manager.min_seq_gap(1).await.unwrap();
    assert_eq!(gap, None);

    // Two lines with gap 500.0
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1500.0, 'b')")
        .execute(manager.pool())
        .await
        .unwrap();
    let gap = manager.min_seq_gap(1).await.unwrap();
    assert_eq!(gap, Some(500.0));

    // Add line with smaller gap
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1501.0, 'c')")
        .execute(manager.pool())
        .await
        .unwrap();
    let gap = manager.min_seq_gap(1).await.unwrap();
    assert_eq!(gap, Some(1.0));

    manager.close().await;
}

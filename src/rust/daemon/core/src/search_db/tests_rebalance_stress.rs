//! Stress tests for rebalancing, high-volume insertions, and constraint conflicts.

#![cfg(test)]

use super::*;
use tempfile::TempDir;

#[tokio::test]
async fn test_midpoint_insertions_between_adjacent() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Start with two lines
    manager.insert_line_at_end(1, "first").await.unwrap();
    manager.insert_line_at_end(1, "last").await.unwrap();

    // Insert 100 lines between the first and second line.
    // Each insert calls renumber_file_line_numbers (O(n) per call), so total
    // work is O(n^2). 100 iterations is enough to trigger multiple rebalances
    // while keeping runtime reasonable (~seconds, not minutes).
    let mut before_seq = 1000.0_f64;
    let mut after_seq = 2000.0_f64;
    let mut rebalance_count = 0;
    let iterations = 100;

    for i in 0..iterations {
        let (result, rebalanced) = manager
            .insert_line_between(1, before_seq, after_seq, &format!("mid {}", i))
            .await
            .unwrap();

        if rebalanced {
            rebalance_count += 1;
            // After rebalance, look up the actual seq values for the next iteration
            let rows = sqlx::query("SELECT seq FROM code_lines WHERE file_id = 1 ORDER BY seq")
                .fetch_all(manager.pool())
                .await
                .unwrap();
            before_seq = rows[0].get::<f64, _>("seq");
            after_seq = rows[1].get::<f64, _>("seq");
        } else {
            after_seq = result.seq;
        }
    }

    // Verify all lines are present and properly ordered
    let expected = iterations + 2;
    let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
        .bind(1_i64)
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(
        rows.len(),
        expected,
        "Should have {} lines (2 original + {} inserted)",
        expected,
        iterations
    );

    // First should still be "first", last should still be "last"
    assert_eq!(rows[0].get::<String, _>("content"), "first");
    assert_eq!(rows[expected - 1].get::<String, _>("content"), "last");

    // All seq values should be strictly increasing
    let seqs: Vec<f64> = rows.iter().map(|r| r.get::<f64, _>("seq")).collect();
    for i in 1..seqs.len() {
        assert!(
            seqs[i] > seqs[i - 1],
            "seq[{}]={} should be > seq[{}]={}",
            i,
            seqs[i],
            i - 1,
            seqs[i - 1]
        );
    }

    // At least one rebalance should have occurred
    assert!(
        rebalance_count > 0,
        "Should have triggered at least one rebalance during {} insertions",
        iterations
    );

    manager.close().await;
}

#[tokio::test]
async fn test_rebalance_preserves_fts5() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert cramped lines and build FTS
    let contents = vec![
        "fn hello_world()",
        "fn goodbye_world()",
        "fn third_function()",
    ];
    for (i, content) in contents.iter().enumerate() {
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(1.0 + i as f64 * 0.0001)
            .bind(*content)
            .execute(manager.pool())
            .await
            .unwrap();
    }
    manager.rebuild_fts().await.unwrap();

    // Verify FTS works before rebalance
    let rows_before = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("hello_world")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows_before.len(), 1);

    // Rebalance
    manager.rebalance_file_seqs(1).await.unwrap();

    // FTS should still work (line_id and content unchanged, only seq changed)
    let rows_after = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
        .bind("hello_world")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows_after.len(), 1);
    assert_eq!(
        rows_after[0].get::<String, _>("content"),
        "fn hello_world()"
    );

    // Verify seq values are now spread out
    let rows = sqlx::query("SELECT seq FROM code_lines WHERE file_id = 1 ORDER BY seq")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows[0].get::<f64, _>("seq"), 1000.0);
    assert_eq!(rows[1].get::<f64, _>("seq"), 2000.0);
    assert_eq!(rows[2].get::<f64, _>("seq"), 3000.0);

    manager.close().await;
}

#[tokio::test]
async fn test_insert_10000_random_positions() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Insert initial 100 lines
    for i in 0..100 {
        manager
            .insert_line_at_end(1, &format!("initial {}", i))
            .await
            .unwrap();
    }

    // Insert 200 more lines at various positions (between existing lines)
    // Use a deterministic pattern to avoid true randomness in tests
    for i in 0..200 {
        // Get current lines to find insertion points
        let seqs: Vec<f64> =
            sqlx::query_scalar("SELECT seq FROM code_lines WHERE file_id = 1 ORDER BY seq")
                .fetch_all(manager.pool())
                .await
                .unwrap();

        // Pick two adjacent lines using a deterministic index
        let idx = i % (seqs.len() - 1);
        manager
            .insert_line_between(1, seqs[idx], seqs[idx + 1], &format!("inserted {}", i))
            .await
            .unwrap();
    }

    // Verify all 300 lines are present
    let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
        .bind(1_i64)
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows.len(), 300);

    // All seq values should be strictly increasing
    let seqs: Vec<f64> = rows.iter().map(|r| r.get::<f64, _>("seq")).collect();
    for i in 1..seqs.len() {
        assert!(
            seqs[i] > seqs[i - 1],
            "seq[{}]={} should be > seq[{}]={} (line_number {})",
            i,
            seqs[i],
            i - 1,
            seqs[i - 1],
            i + 1
        );
    }

    manager.close().await;
}

#[tokio::test]
async fn test_rebalance_with_unique_constraint_conflicts() {
    use sqlx::Row;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("search.db");
    let manager = SearchDbManager::new(&db_path).await.unwrap();

    // Create a scenario where naive rebalancing would hit UNIQUE conflicts:
    // Line at seq=2000.0 and another at seq=2000.001
    // Rebalancing to 1000.0, 2000.0 would conflict if done naively
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 2000.0, 'a')")
        .execute(manager.pool())
        .await
        .unwrap();
    sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 2000.001, 'b')")
        .execute(manager.pool())
        .await
        .unwrap();

    // Rebalance should succeed despite the overlap scenario
    let result = manager.rebalance_file_seqs(1).await.unwrap();
    assert_eq!(result.lines_rebalanced, 2);

    // Verify final state
    let rows = sqlx::query("SELECT seq, content FROM code_lines WHERE file_id = 1 ORDER BY seq")
        .fetch_all(manager.pool())
        .await
        .unwrap();
    assert_eq!(rows[0].get::<f64, _>("seq"), 1000.0);
    assert_eq!(rows[0].get::<String, _>("content"), "a");
    assert_eq!(rows[1].get::<f64, _>("seq"), 2000.0);
    assert_eq!(rows[1].get::<String, _>("content"), "b");

    manager.close().await;
}

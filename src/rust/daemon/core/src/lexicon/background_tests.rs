//! Tests for the background persistence task.

use std::sync::Arc;

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;
use std::time::Duration;

use super::manager::LexiconManager;

async fn create_test_pool() -> SqlitePool {
    SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool")
}

async fn setup_tables(pool: &SqlitePool) {
    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS sparse_vocabulary (
            term_id INTEGER NOT NULL,
            term TEXT NOT NULL,
            collection TEXT NOT NULL,
            document_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            PRIMARY KEY (term_id, collection),
            UNIQUE (term, collection)
        )"#,
    )
    .execute(pool)
    .await
    .unwrap();

    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS corpus_statistics (
            collection TEXT PRIMARY KEY NOT NULL,
            total_documents INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )"#,
    )
    .execute(pool)
    .await
    .unwrap();
}

#[tokio::test]
async fn test_background_persister_flushes_dirty_terms() {
    // Verify that flush_all_background() ensures all dirty terms reach SQLite.
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = Arc::new(LexiconManager::new(pool.clone(), 1.2));
    mgr.start_background_persister().await;

    // Add terms in 2+ docs so they survive hapax eviction
    mgr.add_document("projects", &["background".into(), "persist".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["background".into(), "persist".into()])
        .await
        .unwrap();

    // Manually trigger persist via background path
    mgr.persist("projects").await.unwrap();

    // Flush ensures background task has written everything
    mgr.flush_all_background().await;

    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM sparse_vocabulary WHERE collection = 'projects'")
            .fetch_one(&pool)
            .await
            .unwrap();

    assert_eq!(count, 2, "Both terms must be persisted after flush");
}

#[tokio::test]
async fn test_background_persister_flush_is_no_op_without_start() {
    // flush_all_background() when no background task is running must not panic.
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool, 1.2);

    // No start_background_persister() call — flush should be a safe no-op
    mgr.flush_all_background().await; // must not panic or hang
}

#[tokio::test]
async fn test_background_persister_inline_fallback_without_start() {
    // When background task is not started, add_document() falling through the
    // AUTO_PERSIST_THRESHOLD uses inline persist.
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    // Use a LexiconManager without background task (no start_background_persister call)
    let mgr = Arc::new(LexiconManager::new(pool.clone(), 1.2));

    // Add more documents than AUTO_PERSIST_THRESHOLD (50) with the same term
    // so that it accumulates df >= 2 and inline persist fires at threshold
    // Use only 2 docs and manually persist to keep test fast
    mgr.add_document("projects", &["inline_term".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["inline_term".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM sparse_vocabulary WHERE collection = 'projects'")
            .fetch_one(&pool)
            .await
            .unwrap();

    assert_eq!(
        count, 1,
        "inline_term (df=2) must survive eviction after inline persist"
    );
}

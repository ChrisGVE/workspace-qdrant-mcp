//! Unit tests for the LexiconManager.

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
async fn test_new_collection_starts_empty() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool, 1.2);
    assert_eq!(mgr.corpus_size("projects").await, 0);
    assert_eq!(mgr.document_frequency("projects", "test").await, 0);
}

#[tokio::test]
async fn test_add_document_tracks_terms() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool, 1.2);
    mgr.add_document("projects", &["hello".into(), "world".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["hello".into(), "rust".into()])
        .await
        .unwrap();

    assert_eq!(mgr.corpus_size("projects").await, 2);
    assert_eq!(mgr.document_frequency("projects", "hello").await, 2);
    assert_eq!(mgr.document_frequency("projects", "world").await, 1);
    assert_eq!(mgr.document_frequency("projects", "rust").await, 1);
    assert_eq!(mgr.document_frequency("projects", "missing").await, 0);
}

#[tokio::test]
async fn test_persist_and_reload() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool.clone(), 1.2);
    // Three documents so fn/main/test all appear in 2+ docs (hapax terms are evicted)
    mgr.add_document("projects", &["fn".into(), "main".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["fn".into(), "test".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["main".into(), "test".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    // Create a new manager to test loading from SQLite
    let mgr2 = LexiconManager::new(pool, 1.2);
    mgr2.load_collection("projects").await.unwrap();

    assert_eq!(mgr2.corpus_size("projects").await, 3);
    assert_eq!(mgr2.document_frequency("projects", "fn").await, 2);
    assert_eq!(mgr2.document_frequency("projects", "main").await, 2);
    assert_eq!(mgr2.document_frequency("projects", "test").await, 2);
}

#[tokio::test]
async fn test_collections_are_isolated() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool, 1.2);
    mgr.add_document("projects", &["hello".into()])
        .await
        .unwrap();
    mgr.add_document("libraries", &["world".into()])
        .await
        .unwrap();

    assert_eq!(mgr.corpus_size("projects").await, 1);
    assert_eq!(mgr.corpus_size("libraries").await, 1);
    assert_eq!(mgr.document_frequency("projects", "hello").await, 1);
    assert_eq!(mgr.document_frequency("projects", "world").await, 0);
    assert_eq!(mgr.document_frequency("libraries", "world").await, 1);
    assert_eq!(mgr.document_frequency("libraries", "hello").await, 0);
}

#[tokio::test]
async fn test_persist_all() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool.clone(), 1.2);
    // Add each term twice so they survive hapax eviction (df >= 2)
    mgr.add_document("projects", &["a".into()]).await.unwrap();
    mgr.add_document("projects", &["a".into()]).await.unwrap();
    mgr.add_document("libraries", &["b".into()]).await.unwrap();
    mgr.add_document("libraries", &["b".into()]).await.unwrap();
    mgr.persist_all().await.unwrap();

    // Verify both persisted (terms with df >= 2 survive eviction)
    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM sparse_vocabulary WHERE collection = 'projects'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 1);

    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM sparse_vocabulary WHERE collection = 'libraries'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 1);
}

#[tokio::test]
async fn test_persist_is_idempotent() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool.clone(), 1.2);
    // Add "hello" twice so it survives hapax eviction (df=2)
    mgr.add_document("projects", &["hello".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["hello".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();
    mgr.persist("projects").await.unwrap(); // Second persist should not error

    let count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM sparse_vocabulary WHERE collection = 'projects'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(count, 1);
}

#[tokio::test]
async fn test_incremental_persist_updates_df() {
    // Verify that document_count is correctly updated when terms accumulate
    // across multiple persist cycles.  Terms must reach df>=2 within a single
    // persist cycle to survive hapax eviction.
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool.clone(), 1.2);

    // First cycle: "hello" in 2 docs → df=2, survives
    mgr.add_document("projects", &["hello".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["hello".into(), "world".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    // "hello" df=2 persisted, "world" df=1 evicted
    let df: i64 = sqlx::query_scalar(
        "SELECT document_count FROM sparse_vocabulary WHERE term = 'hello' AND collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(df, 2);

    // Second cycle: "hello" in 2 more docs → in-memory df increments to 4; "world" in 2 docs → survives
    mgr.add_document("projects", &["hello".into(), "world".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["hello".into(), "world".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    // "hello" should now show df=4 (accumulated across cycles)
    let df: i64 = sqlx::query_scalar(
        "SELECT document_count FROM sparse_vocabulary WHERE term = 'hello' AND collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(df, 4);

    let total: i64 = sqlx::query_scalar(
        "SELECT total_documents FROM corpus_statistics WHERE collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(total, 4);
}

#[tokio::test]
async fn test_generate_sparse_vector_with_idf() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool, 1.2);

    // Add documents: "function" appears in all 3, "qdrant" in 1
    mgr.add_document(
        "projects",
        &["function".into(), "return".into(), "test".into()],
    )
    .await
    .unwrap();
    mgr.add_document(
        "projects",
        &["function".into(), "return".into(), "qdrant".into()],
    )
    .await
    .unwrap();
    mgr.add_document(
        "projects",
        &["function".into(), "hello".into(), "world".into()],
    )
    .await
    .unwrap();

    // Generate sparse vector for a query with both common and rare terms
    let sparse = mgr
        .generate_sparse_vector("projects", &["function".into(), "qdrant".into()])
        .await;

    assert!(
        !sparse.indices.is_empty(),
        "Should produce non-empty sparse vector"
    );
    assert_eq!(sparse.indices.len(), sparse.values.len());

    // "qdrant" (df=1) should have a higher BM25 score than "function" (df=3)
    // Find their scores
    let instances = mgr.instances.read().await;
    let bm25 = instances.get("projects").unwrap();
    let qdrant_id = *bm25.vocab().get("qdrant").unwrap();
    let function_id = *bm25.vocab().get("function").unwrap();

    let qdrant_score = sparse
        .indices
        .iter()
        .zip(sparse.values.iter())
        .find(|(&idx, _)| idx == qdrant_id)
        .map(|(_, &val)| val)
        .unwrap_or(0.0);
    let function_score = sparse
        .indices
        .iter()
        .zip(sparse.values.iter())
        .find(|(&idx, _)| idx == function_id)
        .map(|(_, &val)| val)
        .unwrap_or(0.0);

    assert!(
        qdrant_score > function_score,
        "Rare term 'qdrant' (score={}) should have higher BM25 score than common 'function' (score={})",
        qdrant_score, function_score
    );
}

#[tokio::test]
async fn test_generate_sparse_vector_empty_collection() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool, 1.2);

    // Empty collection should return empty sparse vector (fallback to TF-only in caller)
    let sparse = mgr.generate_sparse_vector("empty", &["hello".into()]).await;

    assert!(
        sparse.indices.is_empty(),
        "Empty collection should produce empty sparse vector"
    );
}

#[tokio::test]
async fn test_persist_only_writes_dirty_terms() {
    // Verify that a second persist (after no new add_document calls) writes nothing.
    // Terms appear in 2 docs each so they survive hapax eviction (df >= 2).
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool.clone(), 1.2);
    mgr.add_document("projects", &["alpha".into(), "beta".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["alpha".into(), "beta".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    // Snapshot row count after first persist (alpha and beta survive with df=2)
    let count_after_first: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sparse_vocabulary WHERE collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(count_after_first, 2);

    // Second persist with no new documents — dirty set is empty, nothing is written.
    mgr.persist("projects").await.unwrap();

    let count_after_second: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sparse_vocabulary WHERE collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        count_after_second, count_after_first,
        "Second persist with no new docs must not alter vocabulary row count"
    );

    // Add "gamma" twice and persist — gamma survives with df=2.
    mgr.add_document("projects", &["gamma".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["gamma".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    let count_after_third: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sparse_vocabulary WHERE collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        count_after_third, 3,
        "Third persist should add exactly one new term 'gamma'"
    );
}


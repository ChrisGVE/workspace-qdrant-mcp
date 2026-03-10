//! Unit tests for the LexiconManager.

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;
use std::time::Duration;
use wqm_common::timestamps;

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
    mgr.add_document("projects", &["fn".into(), "main".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["fn".into(), "test".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    // Create a new manager to test loading from SQLite
    let mgr2 = LexiconManager::new(pool, 1.2);
    mgr2.load_collection("projects").await.unwrap();

    assert_eq!(mgr2.corpus_size("projects").await, 2);
    assert_eq!(mgr2.document_frequency("projects", "fn").await, 2);
    assert_eq!(mgr2.document_frequency("projects", "main").await, 1);
    assert_eq!(mgr2.document_frequency("projects", "test").await, 1);
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
    mgr.add_document("projects", &["a".into()]).await.unwrap();
    mgr.add_document("libraries", &["b".into()]).await.unwrap();
    mgr.persist_all().await.unwrap();

    // Verify both persisted
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
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool.clone(), 1.2);
    mgr.add_document("projects", &["hello".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    // Add more documents
    mgr.add_document("projects", &["hello".into(), "world".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    // Verify updated DF
    let df: i64 = sqlx::query_scalar(
        "SELECT document_count FROM sparse_vocabulary WHERE term = 'hello' AND collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(df, 2);

    let total: i64 = sqlx::query_scalar(
        "SELECT total_documents FROM corpus_statistics WHERE collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(total, 2);
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
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool.clone(), 1.2);
    mgr.add_document("projects", &["alpha".into(), "beta".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    // Snapshot row count after first persist
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

    // Add one new term and persist — only that term should be new.
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

#[tokio::test]
async fn test_cleanup_junk_terms() {
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let now = timestamps::now_utc();

    // Insert a mix of valid and junk terms
    let terms = vec![
        (0, "function", "projects"),     // valid
        (1, "120", "projects"),          // junk: pure digits
        (2, "2.0.0", "projects"),        // junk: version string
        (3, "abc123def456", "projects"), // junk: hex hash (12 chars)
        (4, "0xff", "projects"),         // junk: hex literal
        (5, "usr/bin", "projects"),      // junk: contains path separator
        (6, "a", "projects"),            // junk: single char
        (7, "hello", "projects"),        // valid
        (8, "v1.2.3", "projects"),       // junk: version with v prefix
    ];

    for (id, term, collection) in &terms {
        sqlx::query(
            "INSERT INTO sparse_vocabulary (term_id, term, collection, document_count, created_at) VALUES (?1, ?2, ?3, 1, ?4)"
        )
        .bind(*id as i64)
        .bind(*term)
        .bind(*collection)
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();
    }

    let mgr = LexiconManager::new(pool.clone(), 1.2);
    let removed = mgr.cleanup_junk_terms().await.unwrap();

    // Should have removed 7 junk terms (120, 2.0.0, abc123def456, 0xff, usr/bin, a, v1.2.3)
    assert_eq!(
        removed, 7,
        "Should remove 7 junk terms, removed {}",
        removed
    );

    // Verify valid terms remain
    let remaining: Vec<(String,)> =
        sqlx::query_as("SELECT term FROM sparse_vocabulary ORDER BY term")
            .fetch_all(&pool)
            .await
            .unwrap();

    let remaining_terms: Vec<&str> = remaining.iter().map(|(t,)| t.as_str()).collect();
    assert_eq!(remaining_terms, vec!["function", "hello"]);
}

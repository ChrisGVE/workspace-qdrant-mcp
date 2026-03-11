//! Tests for hapax legomena eviction and vocabulary cleanup.

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
async fn test_hapax_eviction_removes_single_occurrence_terms() {
    // Terms that appear in only one document are evicted from SQLite on persist.
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool.clone(), 1.2);
    // "common" appears in 2 docs (df=2, survives), "rare" in 1 doc (df=1, evicted)
    mgr.add_document("projects", &["common".into(), "rare".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["common".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    let common_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sparse_vocabulary WHERE term = 'common' AND collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let rare_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sparse_vocabulary WHERE term = 'rare' AND collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    assert_eq!(common_count, 1, "'common' (df=2) must survive eviction");
    assert_eq!(rare_count, 0, "'rare' (df=1) must be evicted");

    // corpus_size must be unaffected by eviction (total_docs tracks docs, not terms)
    assert_eq!(mgr.corpus_size("projects").await, 2);
}

#[tokio::test]
async fn test_hapax_eviction_term_readded_after_eviction() {
    // A term evicted at df=1 is re-added if it appears in 2+ docs in the next cycle.
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool.clone(), 1.2);

    // Cycle 1: "phoenix" appears once → evicted
    mgr.add_document("projects", &["phoenix".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    let after_eviction: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sparse_vocabulary WHERE term = 'phoenix' AND collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        after_eviction, 0,
        "'phoenix' should be evicted after first persist"
    );

    // Cycle 2: "phoenix" appears in 2 docs → df=2, survives second persist
    mgr.add_document("projects", &["phoenix".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["phoenix".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    let after_readd: i64 = sqlx::query_scalar(
        "SELECT document_count FROM sparse_vocabulary WHERE term = 'phoenix' AND collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        after_readd, 2,
        "'phoenix' (df=2) should survive the second persist"
    );
}

#[tokio::test]
async fn test_hapax_eviction_does_not_affect_corpus_size() {
    // corpus_size (total_documents) is independent of vocabulary eviction.
    let pool = create_test_pool().await;
    setup_tables(&pool).await;

    let mgr = LexiconManager::new(pool.clone(), 1.2);

    // Add 5 documents, all with unique terms (all hapax after persist)
    for i in 0..5u32 {
        mgr.add_document("projects", &[format!("unique_term_{}", i)])
            .await
            .unwrap();
    }
    mgr.persist("projects").await.unwrap();

    // All 5 terms evicted (each df=1), but corpus_size is 5
    let vocab_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sparse_vocabulary WHERE collection = 'projects'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(vocab_count, 0, "All hapax terms must be evicted");
    assert_eq!(
        mgr.corpus_size("projects").await,
        5,
        "corpus_size must equal total documents added"
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

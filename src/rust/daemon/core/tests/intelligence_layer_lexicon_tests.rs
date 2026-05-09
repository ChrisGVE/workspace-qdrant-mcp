//! Integration tests for the dynamic lexicon and sparse-vector sub-systems.
//!
//! Tests:
//! - Dynamic lexicon: per-collection vocabulary, document frequency, persistence
//! - Lexicon sparse vector generation via persisted BM25 IDF
//! - Cross-feature: lexicon + BM25 consistency and incremental corpus growth

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;
use std::time::Duration;
use workspace_qdrant_core::lexicon::LexiconManager;

/// Create in-memory SQLite pool with lexicon tables.
async fn create_lexicon_pool() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create pool");

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
    .execute(&pool)
    .await
    .unwrap();

    sqlx::query(
        r#"CREATE TABLE IF NOT EXISTS corpus_statistics (
            collection TEXT PRIMARY KEY NOT NULL,
            total_documents INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )"#,
    )
    .execute(&pool)
    .await
    .unwrap();

    pool
}

// ── Dynamic Lexicon Tests ──

#[tokio::test]
async fn test_lexicon_tracks_code_identifiers() {
    let pool = create_lexicon_pool().await;
    let mgr = LexiconManager::new(pool, 1.2);

    // Simulate ingesting Rust source files
    let doc1_tokens: Vec<String> = vec![
        "fn", "main", "let", "result", "qdrant", "client", "search", "fn", "test", "assert",
        "result",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let doc2_tokens: Vec<String> = vec![
        "fn", "process", "let", "data", "qdrant", "insert", "batch", "fn", "validate", "assert",
        "data",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    let doc3_tokens: Vec<String> = vec![
        "fn", "cleanup", "let", "config", "tokio", "spawn", "async", "fn", "monitor", "tracing",
        "info",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    mgr.add_document("projects", &doc1_tokens).await.unwrap();
    mgr.add_document("projects", &doc2_tokens).await.unwrap();
    mgr.add_document("projects", &doc3_tokens).await.unwrap();

    assert_eq!(mgr.corpus_size("projects").await, 3);

    // "fn" and "let" appear in all 3 docs — high DF
    assert_eq!(mgr.document_frequency("projects", "fn").await, 3);
    assert_eq!(mgr.document_frequency("projects", "let").await, 3);

    // "qdrant" appears in 2 docs — moderate DF
    assert_eq!(mgr.document_frequency("projects", "qdrant").await, 2);

    // "tokio" appears in 1 doc — low DF (distinctive)
    assert_eq!(mgr.document_frequency("projects", "tokio").await, 1);

    // Unknown terms have DF=0
    assert_eq!(mgr.document_frequency("projects", "nonexistent").await, 0);
}

#[tokio::test]
async fn test_lexicon_persist_survives_reload() {
    let pool = create_lexicon_pool().await;

    // First session: build vocabulary
    {
        let mgr = LexiconManager::new(pool.clone(), 1.2);
        mgr.add_document(
            "projects",
            &["vector".into(), "search".into(), "qdrant".into()],
        )
        .await
        .unwrap();
        mgr.add_document(
            "projects",
            &["vector".into(), "embed".into(), "model".into()],
        )
        .await
        .unwrap();
        mgr.persist("projects").await.unwrap();
    }

    // Second session: load and verify
    //
    // Note: hapax legomena (df=1) are evicted from the vocabulary on persist.
    // "search" and "qdrant" each appear in only 1 document, so they are not
    // present after reload. Only "vector" (df=2) survives eviction.
    {
        let mgr = LexiconManager::new(pool.clone(), 1.2);
        mgr.load_collection("projects").await.unwrap();

        assert_eq!(mgr.corpus_size("projects").await, 2);
        assert_eq!(
            mgr.document_frequency("projects", "vector").await,
            2,
            "'vector' (df=2) must survive hapax eviction"
        );
        assert_eq!(
            mgr.document_frequency("projects", "search").await,
            0,
            "'search' (df=1) is evicted as a hapax legomenon"
        );
    }
}

#[tokio::test]
async fn test_lexicon_multi_collection_isolation() {
    let pool = create_lexicon_pool().await;
    let mgr = LexiconManager::new(pool, 1.2);

    mgr.add_document("projects", &["rust".into(), "tokio".into()])
        .await
        .unwrap();
    mgr.add_document("libraries", &["python".into(), "django".into()])
        .await
        .unwrap();

    // Each collection is isolated
    assert_eq!(mgr.document_frequency("projects", "rust").await, 1);
    assert_eq!(mgr.document_frequency("projects", "python").await, 0);
    assert_eq!(mgr.document_frequency("libraries", "python").await, 1);
    assert_eq!(mgr.document_frequency("libraries", "rust").await, 0);
}

// ── Lexicon Sparse Vector Tests ──

#[tokio::test]
async fn test_lexicon_sparse_vector_uses_persisted_idf() {
    let pool = create_lexicon_pool().await;
    let mgr = LexiconManager::new(pool, 1.2);

    // Build corpus: N=20
    // "return" in 5 docs (df=5, IDF = ln((20-5+0.5)/(5+0.5)) = ln(15.5/5.5) ≈ 1.04)
    // "custom_handler" in 2 docs (df=2, IDF = ln((20-2+0.5)/(2+0.5)) = ln(18.5/2.5) ≈ 2.0)
    // Both df < N/2 so both have positive IDF
    for _ in 0..15 {
        mgr.add_document("projects", &["other".into(), "terms".into()])
            .await
            .unwrap();
    }
    for _ in 0..3 {
        mgr.add_document("projects", &["return".into(), "more".into()])
            .await
            .unwrap();
    }
    mgr.add_document("projects", &["return".into(), "custom_handler".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["return".into(), "custom_handler".into()])
        .await
        .unwrap();

    let df_custom = mgr.document_frequency("projects", "custom_handler").await;
    let df_return = mgr.document_frequency("projects", "return").await;
    assert_eq!(df_custom, 2, "custom_handler in 2 docs");
    assert_eq!(df_return, 5, "return in 5 docs");

    // Generate sparse vector for "return" and "custom_handler"
    // Both should have positive IDF since both df < N/2
    let sparse = mgr
        .generate_sparse_vector("projects", &["return".into(), "custom_handler".into()])
        .await;

    assert!(
        !sparse.indices.is_empty(),
        "Should produce non-empty sparse vector"
    );
    assert_eq!(
        sparse.indices.len(),
        2,
        "Both terms should have positive weight"
    );

    // custom_handler (df=2) should have higher weight than return (df=5)
    let mut weights: Vec<f32> = sparse.values.clone();
    weights.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!(
        weights[0] > weights[1],
        "Rarer term should have higher weight: max={}, min={}",
        weights[0],
        weights[1]
    );
}

#[tokio::test]
async fn test_lexicon_sparse_vector_empty_for_unknown_collection() {
    let pool = create_lexicon_pool().await;
    let mgr = LexiconManager::new(pool, 1.2);

    let sparse = mgr
        .generate_sparse_vector("unknown", &["test".into()])
        .await;
    assert!(
        sparse.indices.is_empty(),
        "Unknown collection should return empty sparse vector"
    );
}

// ── Cross-Feature Integration Tests ──

#[tokio::test]
async fn test_lexicon_and_bm25_consistency() {
    let pool = create_lexicon_pool().await;
    let mgr = LexiconManager::new(pool.clone(), 1.2);

    // Build corpus: N=20, "async" df=4 (<N/2), "tokio" df=2 (<N/2)
    // Both will have positive IDF
    for _ in 0..16 {
        mgr.add_document("projects", &["other".into(), "stuff".into()])
            .await
            .unwrap();
    }
    mgr.add_document("projects", &["async".into(), "await".into()])
        .await
        .unwrap();
    mgr.add_document("projects", &["async".into(), "fn".into()])
        .await
        .unwrap();
    mgr.add_document(
        "projects",
        &["async".into(), "tokio".into(), "runtime".into()],
    )
    .await
    .unwrap();
    mgr.add_document("projects", &["async".into(), "tokio".into()])
        .await
        .unwrap();
    mgr.persist("projects").await.unwrap();

    // Verify DF
    let df_async = mgr.document_frequency("projects", "async").await;
    let df_tokio = mgr.document_frequency("projects", "tokio").await;
    assert_eq!(df_async, 4, "async in 4 documents");
    assert_eq!(df_tokio, 2, "tokio in 2 documents");

    // Both terms should have positive IDF (df < N/2 = 10)
    let sparse = mgr
        .generate_sparse_vector("projects", &["async".into(), "tokio".into()])
        .await;
    assert!(
        !sparse.indices.is_empty(),
        "Sparse vector should have entries"
    );
    assert_eq!(
        sparse.indices.len(),
        2,
        "Both terms should have positive IDF weight"
    );

    // tokio (df=2) should have higher weight than async (df=4)
    let mut weights: Vec<f32> = sparse.values.clone();
    weights.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!(
        weights[0] > weights[1],
        "Rarer term should have higher weight: max={}, min={}",
        weights[0],
        weights[1]
    );
}

#[tokio::test]
async fn test_lexicon_incremental_corpus_growth() {
    let pool = create_lexicon_pool().await;
    let mgr = LexiconManager::new(pool, 1.2);

    // Phase 1: small corpus
    mgr.add_document("projects", &["hello".into(), "world".into()])
        .await
        .unwrap();
    let size1 = mgr.corpus_size("projects").await;
    assert_eq!(size1, 1);

    // Phase 2: corpus grows
    for i in 0..10 {
        mgr.add_document("projects", &[format!("term_{}", i), "hello".into()])
            .await
            .unwrap();
    }
    let size2 = mgr.corpus_size("projects").await;
    assert_eq!(size2, 11);

    // "hello" should now have df=11 (in all docs)
    assert_eq!(mgr.document_frequency("projects", "hello").await, 11);

    // Each unique term_N should have df=1
    assert_eq!(mgr.document_frequency("projects", "term_0").await, 1);
    assert_eq!(mgr.document_frequency("projects", "term_9").await, 1);
}

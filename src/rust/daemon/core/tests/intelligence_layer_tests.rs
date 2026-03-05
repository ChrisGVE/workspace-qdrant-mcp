//! Integration tests for intelligence layer features.
//!
//! Tests:
//! - Dynamic lexicon: per-collection vocabulary, document frequency, persistence
//! - BM25 IDF weighting: common vs distinctive term scoring
//! - Sparse vector generation via lexicon (persisted BM25)
//! - Metadata uplift: UpliftConfig, UpliftStats, candidate identification
//! - Keyword extraction pipeline configuration

use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;
use std::time::Duration;
use workspace_qdrant_core::{
    embedding::BM25,
    keyword_extraction::keyword_selector::SelectedKeyword,
    keyword_extraction::pipeline::{ExtractionResult, PipelineConfig},
    keyword_extraction::tag_selector::{SelectedTag, TagType},
    lexicon::LexiconManager,
    metadata_uplift::{UpliftConfig, UpliftStats},
};

/// Create in-memory SQLite pool with lexicon tables
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
    {
        let mgr = LexiconManager::new(pool.clone(), 1.2);
        mgr.load_collection("projects").await.unwrap();

        assert_eq!(mgr.corpus_size("projects").await, 2);
        assert_eq!(mgr.document_frequency("projects", "vector").await, 2);
        assert_eq!(mgr.document_frequency("projects", "search").await, 1);
        assert_eq!(mgr.document_frequency("projects", "qdrant").await, 1);
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

// ── BM25 IDF Weighting Tests ──

#[test]
fn test_bm25_idf_downweights_common_terms() {
    let mut bm25 = BM25::new(1.2);

    // Build corpus of 20 docs: "function" in all 20, "qdrant_search" in 2
    for _ in 0..20 {
        bm25.add_document(&["function".into(), "return".into(), "let".into()]);
    }
    // Add 2 more docs with "qdrant_search" (total N=22, function df=22, qdrant_search df=2)
    bm25.add_document(&["qdrant_search".into(), "rare_api".into()]);
    bm25.add_document(&["qdrant_search".into(), "rare_api".into()]);

    let sparse = bm25.generate_sparse_vector(&["function".into(), "qdrant_search".into()]);

    // "function" (df=20, N=22): IDF = ln((22-20+0.5)/(20+0.5)) = ln(2.5/20.5) < 0 → floored to 0
    // So "function" should NOT be in the sparse vector (zero weight)
    // "qdrant_search" (df=2, N=22): IDF = ln((22-2+0.5)/(2+0.5)) = ln(20.5/2.5) = ln(8.2) ≈ 2.1
    // So "qdrant_search" should have positive weight

    let vocab = bm25.vocab();
    let function_id = *vocab.get("function").unwrap();
    let qdrant_id = *vocab.get("qdrant_search").unwrap();

    let function_weight = sparse
        .indices
        .iter()
        .zip(sparse.values.iter())
        .find(|(&idx, _)| idx == function_id)
        .map(|(_, &val)| val);
    let qdrant_weight = sparse
        .indices
        .iter()
        .zip(sparse.values.iter())
        .find(|(&idx, _)| idx == qdrant_id)
        .map(|(_, &val)| val);

    // Common term "function" gets zero IDF (floored), so excluded from sparse vector
    assert!(
        function_weight.is_none() || function_weight.unwrap() == 0.0,
        "Ubiquitous term should have zero or no weight: {:?}",
        function_weight
    );

    // Rare term "qdrant_search" should have positive weight
    assert!(
        qdrant_weight.is_some() && qdrant_weight.unwrap() > 0.0,
        "Rare term should have positive weight: {:?}",
        qdrant_weight
    );
}

#[test]
fn test_bm25_empty_corpus_uses_tf_only() {
    let bm25 = BM25::new(1.2);

    // No documents added — should still generate sparse vector (TF-only fallback)
    let sparse = bm25.generate_sparse_vector(&["hello".into(), "world".into()]);

    // With empty corpus, unknown terms won't have vocab IDs, so vector is empty
    assert!(
        sparse.indices.is_empty(),
        "Terms not in vocab should produce empty vector"
    );
}

#[test]
fn test_bm25_tf_saturation() {
    let mut bm25 = BM25::new(1.2);

    // Need corpus where "test" has positive IDF: N=10, test df=2
    for _ in 0..8 {
        bm25.add_document(&["other".into()]);
    }
    bm25.add_document(&["test".into(), "other".into()]);
    bm25.add_document(&["test".into(), "other".into()]);

    // Repeated term: tf=5
    let sparse_repeated = bm25.generate_sparse_vector(&[
        "test".into(),
        "test".into(),
        "test".into(),
        "test".into(),
        "test".into(),
    ]);
    // Single occurrence: tf=1
    let sparse_single = bm25.generate_sparse_vector(&["test".into()]);

    // Find "test" weight in each vector
    let test_id = *bm25.vocab().get("test").unwrap();
    let repeated_weight = sparse_repeated
        .indices
        .iter()
        .zip(sparse_repeated.values.iter())
        .find(|(&i, _)| i == test_id)
        .map(|(_, &v)| v)
        .unwrap_or(0.0);
    let single_weight = sparse_single
        .indices
        .iter()
        .zip(sparse_single.values.iter())
        .find(|(&i, _)| i == test_id)
        .map(|(_, &v)| v)
        .unwrap_or(0.0);

    assert!(
        single_weight > 0.0,
        "Single occurrence should have positive weight"
    );
    assert!(
        repeated_weight > single_weight,
        "Higher TF should give higher weight"
    );
    assert!(
        repeated_weight < single_weight * 3.0,
        "BM25 k1 saturation should prevent linear scaling (repeated={}, single={})",
        repeated_weight,
        single_weight
    );
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

// ── Metadata Uplift Tests ──

#[test]
fn test_uplift_config_respects_generation_tracking() {
    let config = UpliftConfig::default();
    assert_eq!(config.current_generation, 1, "Should start at generation 1");
    assert_eq!(config.batch_size, 10);
    assert_eq!(config.min_interval_secs, 300, "5 minute minimum interval");
}

#[test]
fn test_uplift_stats_aggregation() {
    let mut stats = UpliftStats::default();
    stats.scanned = 10;
    stats.updated = 5;
    stats.skipped = 3;
    stats.errors = 2;

    assert_eq!(stats.scanned, stats.updated + stats.skipped + stats.errors);
}

// ── Pipeline Configuration Tests ──

#[test]
fn test_pipeline_config_defaults_are_sane() {
    let config = PipelineConfig::default();

    assert!(config.keyword.max_keywords > 0, "Should have keyword limit");
    assert!(config.tag.max_tags > 0, "Should have tag limit");
    assert!(
        config.basket.min_similarity > 0.0,
        "Should have minimum similarity"
    );
    assert!(
        config.basket.min_similarity < 1.0,
        "Minimum similarity should be < 1"
    );
}

#[test]
fn test_extraction_result_accessors() {
    let result = ExtractionResult {
        summary_vector: Some(vec![0.1, 0.2, 0.3]),
        gist_indices: vec![0, 2, 4],
        keywords: vec![
            SelectedKeyword {
                phrase: "vector_search".to_string(),
                score: 0.95,
                semantic_score: 0.9,
                lexical_score: 1.0,
                stability_count: 5,
                ngram_size: 1,
            },
            SelectedKeyword {
                phrase: "embedding_model".to_string(),
                score: 0.85,
                semantic_score: 0.8,
                lexical_score: 0.9,
                stability_count: 3,
                ngram_size: 1,
            },
        ],
        tags: vec![SelectedTag {
            phrase: "search".to_string(),
            tag_type: TagType::Concept,
            score: 0.9,
            diversity_score: 1.0,
            semantic_score: 0.85,
            ngram_size: 1,
        }],
        structural_tags: vec![
            SelectedTag {
                phrase: "language:rust".to_string(),
                tag_type: TagType::Structural,
                score: 1.0,
                diversity_score: 1.0,
                semantic_score: 1.0,
                ngram_size: 1,
            },
            SelectedTag {
                phrase: "framework:tokio".to_string(),
                tag_type: TagType::Structural,
                score: 1.0,
                diversity_score: 1.0,
                semantic_score: 1.0,
                ngram_size: 1,
            },
        ],
        baskets: vec![],
    };

    let kw_phrases = result.keyword_phrases();
    assert_eq!(kw_phrases.len(), 2);
    assert!(kw_phrases.contains(&"vector_search".to_string()));
    assert!(kw_phrases.contains(&"embedding_model".to_string()));

    let tag_phrases = result.tag_phrases();
    assert_eq!(tag_phrases, vec!["search"]);

    let struct_map = result.structural_tags_map();
    assert_eq!(
        struct_map.get("language").unwrap(),
        &vec!["rust".to_string()]
    );
    assert_eq!(
        struct_map.get("framework").unwrap(),
        &vec!["tokio".to_string()]
    );

    assert_eq!(result.gist_indices.len(), 3);
    assert!(result.summary_vector.is_some());
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

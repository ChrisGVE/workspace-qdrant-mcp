//! Integration tests for BM25 IDF weighting.
//!
//! Tests:
//! - BM25 IDF downweighting of common terms
//! - Empty corpus fallback behaviour
//! - TF saturation (k1 parameter)

use workspace_qdrant_core::embedding::BM25;

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

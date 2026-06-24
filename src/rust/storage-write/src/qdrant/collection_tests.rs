//! Tests for `wqm-storage-write/src/qdrant/collection.rs` (AC-F11.1).
//!
//! All tests are OFFLINE -- no live Qdrant required. They assert the exact
//! collection spec by inspecting the builder outputs directly.

use super::*;
use qdrant_client::qdrant::{vectors_config, SparseVectorConfig, VectorsConfig};

// ---------------------------------------------------------------------------
// AC-F11.1: dense vector spec
// ---------------------------------------------------------------------------

// AC-F11.1 (dense): build_vectors_config produces exactly one named dense
// vector ("dense") with 768 dimensions and Cosine distance.
#[test]
fn vectors_config_has_single_dense_768_cosine() {
    let cfg = build_vectors_config();

    let params_map = match cfg.config {
        Some(vectors_config::Config::ParamsMap(ref m)) => &m.map,
        other => panic!("expected ParamsMap, got {:?}", other),
    };

    assert_eq!(
        params_map.len(),
        1,
        "exactly one named dense vector (AC-F11.1)"
    );

    let dense = params_map
        .get(DENSE_VECTOR_NAME)
        .expect("key 'dense' must be present (AC-F11.1)");

    assert_eq!(
        dense.size, DENSE_DIM,
        "dense vector must be 768-dim (AC-F11.1)"
    );

    // Distance::Cosine = 1 in the qdrant protobuf enum.
    assert_eq!(
        dense.distance,
        qdrant_client::qdrant::Distance::Cosine as i32,
        "dense vector must use Cosine distance (AC-F11.1)"
    );
}

// AC-F11.1 (dense key): the dense vector is named DENSE_VECTOR_NAME ("dense"),
// not placed under a generic or default key.
#[test]
fn dense_vector_name_is_correct() {
    assert_eq!(DENSE_VECTOR_NAME, "dense");
    let cfg = build_vectors_config();
    let params_map = match cfg.config {
        Some(vectors_config::Config::ParamsMap(ref m)) => &m.map,
        other => panic!("expected ParamsMap, got {:?}", other),
    };
    assert!(
        params_map.contains_key("dense"),
        "dense vector must be named 'dense' (AC-F11.1)"
    );
}

// ---------------------------------------------------------------------------
// AC-F11.1: sparse vector spec (dot-product under top-level sparse_vectors)
// ---------------------------------------------------------------------------

// AC-F11.1 (sparse placement): build_sparse_config declares the sparse vector
// under a SparseVectorConfig, NOT inside VectorsConfig. This is the top-level
// `sparse_vectors` map (correct), not inside `vectors` (wrong -- would require
// a distance field, which SparseVectorParams does not have).
#[test]
fn sparse_config_is_separate_from_vectors_config() {
    let vectors_cfg: VectorsConfig = build_vectors_config();
    let sparse_cfg: SparseVectorConfig = build_sparse_config();

    // The vectors config must NOT contain "sparse".
    let params_map = match vectors_cfg.config {
        Some(vectors_config::Config::ParamsMap(ref m)) => &m.map,
        other => panic!("expected ParamsMap, got {:?}", other),
    };
    assert!(
        !params_map.contains_key(SPARSE_VECTOR_NAME),
        "sparse vector must NOT be in VectorsConfig (AC-F11.1): \
         it goes under sparse_vectors, not vectors"
    );

    // The sparse config must contain "sparse".
    assert!(
        sparse_cfg.map.contains_key(SPARSE_VECTOR_NAME),
        "sparse vector must be in SparseVectorConfig under key 'sparse' (AC-F11.1)"
    );
}

// AC-F11.1 (sparse key): the sparse vector is named SPARSE_VECTOR_NAME ("sparse").
#[test]
fn sparse_vector_name_is_correct() {
    assert_eq!(SPARSE_VECTOR_NAME, "sparse");
    let cfg = build_sparse_config();
    assert!(
        cfg.map.contains_key("sparse"),
        "sparse vector must be named 'sparse' (AC-F11.1)"
    );
}

// AC-F11.1 (sparse params): SparseVectorParams has no distance field and both
// optional index and modifier are None (the default -- correct for implicit
// dot-product, arch §5.3).
#[test]
fn sparse_params_have_no_distance_field() {
    let cfg = build_sparse_config();
    let params = cfg.map.get("sparse").expect("sparse key must be present");

    assert!(
        params.index.is_none(),
        "SparseVectorParams.index must be None (AC-F11.1)"
    );
    assert!(
        params.modifier.is_none(),
        "SparseVectorParams.modifier must be None (AC-F11.1)"
    );
}

// ---------------------------------------------------------------------------
// AC-F11.1: constants sanity
// ---------------------------------------------------------------------------

// AC-F11.1: DENSE_DIM is exactly 768 (the model output dimension).
#[test]
fn dense_dim_is_768() {
    assert_eq!(DENSE_DIM, 768, "dense dim must be 768 (AC-F11.1)");
}

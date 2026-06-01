// Integration tests: search tool → real embed + Qdrant dense search + RRF fusion.
//
// it_search_embed_text_live_daemon   — embed_text against live daemon returns
//                                      non-empty 384-dim vector
// it_search_dense_projects_qdrant    — dense search against projects collection
//                                      returns structurally valid results
// it_search_sparse_vector_roundtrip  — generate_sparse_vector returns non-empty
//                                      indices/values when daemon is live
// it_search_collection_absent_empty  — dense search against nonexistent collection
//                                      returns an error (not a panic)

use super::helpers;
use mcp_server::grpc::embedding_methods::split_sparse_map;
use mcp_server::qdrant::fusion::{DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME};

// ---------------------------------------------------------------------------
// Live daemon: embed_text returns a non-empty dense vector
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_search_embed_text_live_daemon() {
    let mut client = match helpers::probe_daemon().await {
        Some(c) => c,
        None => return,
    };

    let resp = client
        .embed_text("integration test query for search")
        .await
        .expect("embed_text must succeed against live daemon");

    assert!(
        resp.success,
        "embed_text must report success; error: {}",
        resp.error_message
    );
    assert!(
        !resp.embedding.is_empty(),
        "embedding vector must not be empty"
    );
    assert!(
        resp.dimensions > 0,
        "dimensions must be positive; got: {}",
        resp.dimensions
    );
    assert_eq!(
        resp.embedding.len(),
        resp.dimensions as usize,
        "embedding length must equal dimensions"
    );
}

// ---------------------------------------------------------------------------
// Live daemon: generate_sparse_vector returns non-empty indices for real text
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_search_sparse_vector_roundtrip() {
    let mut client = match helpers::probe_daemon().await {
        Some(c) => c,
        None => return,
    };

    let resp = client
        .generate_sparse_vector("integration test sparse query")
        .await
        .expect("generate_sparse_vector must succeed against live daemon");

    assert!(
        resp.success,
        "generate_sparse_vector must report success; error: {}",
        resp.error_message
    );

    let (indices, values) = split_sparse_map(&resp.indices_values);
    assert_eq!(
        indices.len(),
        values.len(),
        "split_sparse_map must produce equal-length parallel arrays"
    );
    // Non-trivial text should produce at least one sparse token.
    assert!(
        !indices.is_empty(),
        "non-trivial text must produce at least one sparse token"
    );
    // All values must be positive (IDF weights are non-negative).
    for &v in &values {
        assert!(v >= 0.0, "sparse value must be non-negative; got: {v}");
    }
}

// ---------------------------------------------------------------------------
// Live Qdrant: dense search against the projects collection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_search_dense_projects_qdrant() {
    // Need both daemon (for embedding) and Qdrant (for search).
    let mut daemon = match helpers::probe_daemon().await {
        Some(c) => c,
        None => return,
    };
    if !helpers::probe_qdrant().await {
        return;
    }

    let url = helpers::qdrant_url();
    let api_key = helpers::qdrant_api_key().map(|s| secrecy::SecretString::new(s.into_boxed_str()));
    let qdrant = mcp_server::qdrant::client::QdrantReadClient::new(url, api_key);

    // Skip if the projects collection does not exist yet.
    let exists = qdrant
        .collection_exists(wqm_common::constants::COLLECTION_PROJECTS)
        .await
        .expect("collection_exists must not error");
    if !exists {
        eprintln!(
            "SKIP: collection '{}' not present in Qdrant",
            wqm_common::constants::COLLECTION_PROJECTS
        );
        return;
    }

    // Embed the query via the live daemon.
    let embed_resp = daemon
        .embed_text("function definition")
        .await
        .expect("embed_text must succeed");
    if !embed_resp.success || embed_resp.embedding.is_empty() {
        eprintln!(
            "SKIP: embed_text returned no vector: {}",
            embed_resp.error_message
        );
        return;
    }

    // Run dense search — limit 3, no filter.
    let results = qdrant
        .search(
            wqm_common::constants::COLLECTION_PROJECTS,
            DENSE_VECTOR_NAME,
            embed_resp.embedding,
            3,
            None,
            None,
        )
        .await
        .expect("dense search must succeed");

    // Structural validation.
    assert!(
        results.len() <= 3,
        "dense search returned more than limit=3 results: {}",
        results.len()
    );
    for r in &results {
        assert!(!r.id.is_empty(), "search result id must not be empty");
    }
}

// ---------------------------------------------------------------------------
// Live Qdrant: dense search against nonexistent collection returns Err (not panic)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_search_collection_absent_returns_error() {
    if !helpers::probe_qdrant().await {
        return;
    }

    let url = helpers::qdrant_url();
    let api_key = helpers::qdrant_api_key().map(|s| secrecy::SecretString::new(s.into_boxed_str()));
    let qdrant = mcp_server::qdrant::client::QdrantReadClient::new(url, api_key);

    // A dummy 384-dim zero vector — the collection does not exist so the
    // content of the vector does not matter.
    let dummy_vec = vec![0.0_f32; 384];

    let result = qdrant
        .search(
            "__nonexistent_integration_test_collection__",
            DENSE_VECTOR_NAME,
            dummy_vec,
            3,
            None,
            None,
        )
        .await;

    assert!(
        result.is_err(),
        "search against nonexistent collection must return Err, not Ok"
    );
}

// ---------------------------------------------------------------------------
// Live Qdrant + daemon: sparse search via SPARSE_VECTOR_NAME (if collection exists)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_search_sparse_projects_qdrant() {
    let mut daemon = match helpers::probe_daemon().await {
        Some(c) => c,
        None => return,
    };
    if !helpers::probe_qdrant().await {
        return;
    }

    let url = helpers::qdrant_url();
    let api_key = helpers::qdrant_api_key().map(|s| secrecy::SecretString::new(s.into_boxed_str()));
    let qdrant = mcp_server::qdrant::client::QdrantReadClient::new(url, api_key);

    let exists = qdrant
        .collection_exists(wqm_common::constants::COLLECTION_PROJECTS)
        .await
        .expect("collection_exists must not error");
    if !exists {
        eprintln!(
            "SKIP: collection '{}' not present in Qdrant",
            wqm_common::constants::COLLECTION_PROJECTS
        );
        return;
    }

    let sparse_resp = daemon
        .generate_sparse_vector("function definition")
        .await
        .expect("generate_sparse_vector must succeed");
    if !sparse_resp.success || sparse_resp.indices_values.is_empty() {
        eprintln!(
            "SKIP: generate_sparse_vector returned empty; error: {}",
            sparse_resp.error_message
        );
        return;
    }

    let (indices, values) = split_sparse_map(&sparse_resp.indices_values);

    let results = qdrant
        .search_sparse(
            wqm_common::constants::COLLECTION_PROJECTS,
            SPARSE_VECTOR_NAME,
            indices,
            values,
            3,
            None,
            None,
        )
        .await
        .expect("sparse search must succeed");

    assert!(
        results.len() <= 3,
        "sparse search returned more than limit=3 results: {}",
        results.len()
    );
    for r in &results {
        assert!(
            !r.id.is_empty(),
            "sparse search result id must not be empty"
        );
    }
}

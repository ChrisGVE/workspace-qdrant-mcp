// Integration tests: retrieve tool → real Qdrant scroll/retrieve.
//
// it_retrieve_scroll_projects  — scroll the projects collection (AC-Q1)
// it_retrieve_scroll_rules     — scroll the rules collection

use super::helpers;

// ---------------------------------------------------------------------------
// Live Qdrant: scroll the projects collection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_retrieve_scroll_projects() {
    if !helpers::probe_qdrant().await {
        return;
    }

    let url = helpers::qdrant_url();
    let api_key = helpers::qdrant_api_key().map(|s| secrecy::SecretString::new(s.into_boxed_str()));

    let client = mcp_server::qdrant::client::QdrantReadClient::new(url, api_key);

    // Check whether the projects collection exists before scrolling.
    let exists = client
        .collection_exists(wqm_common::constants::COLLECTION_PROJECTS)
        .await
        .expect("collection_exists must not error");

    if !exists {
        eprintln!(
            "SKIP: collection '{}' does not exist in Qdrant",
            wqm_common::constants::COLLECTION_PROJECTS
        );
        return;
    }

    // Scroll up to 3 points — structural validation only.
    let (points, _next) = client
        .scroll(
            wqm_common::constants::COLLECTION_PROJECTS,
            None, // no filter
            3,
            None, // no offset
        )
        .await
        .expect("scroll must succeed");

    // Each retrieved point must have a non-empty id.
    for p in &points {
        assert!(!p.id.is_empty(), "retrieved point id must not be empty");
    }
    assert!(
        points.len() <= 3,
        "scroll returned more than limit=3 points: {}",
        points.len()
    );
}

// ---------------------------------------------------------------------------
// Live Qdrant: scroll the rules collection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_retrieve_scroll_rules() {
    if !helpers::probe_qdrant().await {
        return;
    }

    let url = helpers::qdrant_url();
    let api_key = helpers::qdrant_api_key().map(|s| secrecy::SecretString::new(s.into_boxed_str()));

    let client = mcp_server::qdrant::client::QdrantReadClient::new(url, api_key);

    let exists = client
        .collection_exists(wqm_common::constants::COLLECTION_RULES)
        .await
        .expect("collection_exists must not error");

    if !exists {
        eprintln!(
            "SKIP: collection '{}' does not exist",
            wqm_common::constants::COLLECTION_RULES
        );
        return;
    }

    let (points, _next) = client
        .scroll(wqm_common::constants::COLLECTION_RULES, None, 5, None)
        .await
        .expect("scroll rules must succeed");

    for p in &points {
        assert!(!p.id.is_empty(), "point id must not be empty");
    }
}

// ---------------------------------------------------------------------------
// Live Qdrant: collection_exists returns false for a nonexistent collection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_collection_exists_false_for_nonexistent() {
    if !helpers::probe_qdrant().await {
        return;
    }

    let url = helpers::qdrant_url();
    let api_key = helpers::qdrant_api_key().map(|s| secrecy::SecretString::new(s.into_boxed_str()));

    let client = mcp_server::qdrant::client::QdrantReadClient::new(url, api_key);

    let exists = client
        .collection_exists("__nonexistent_integration_test_collection__")
        .await
        .expect("collection_exists must not error");

    assert!(!exists, "nonexistent collection must return false");
}

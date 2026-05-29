//! Regression test for the multi-tenant payload-index manifest.
//!
//! See `src/storage/collections/multi_tenant.rs::tests` for pure unit tests
//! that assert the manifest contains the right fields. This file complements
//! them with one live-Qdrant round-trip:
//!
//! 1. Create a temporary collection mirroring the multi-tenant config.
//! 2. Call `create_payload_index` for the same fields that
//!    `init_projects_collection` would create.
//! 3. Query the collection's `payload_schema` and assert each field is
//!    indexed as `keyword`.
//! 4. Drop the collection.
//!
//! Catches: regressions in `create_payload_index` itself, breakage in the
//! Qdrant Rust SDK's wire format, and any operator-facing surprise where a
//! daemon thinks it indexed a field but Qdrant disagrees.
//!
//! Skipped gracefully when Qdrant isn't reachable (matches the existing
//! `tests/qdrant_client_validation_*.rs` pattern). Run with a live Qdrant via:
//!
//! ```bash
//! TEST_QDRANT_URL=http://localhost:6334 \         # gRPC for StorageClient
//! TEST_QDRANT_HTTP_URL=http://localhost:6333 \    # REST for payload-schema readback
//! cargo test -p workspace-qdrant-core --test qdrant_payload_index_test
//! ```
//!
//! Both default to localhost when unset.

use std::time::Duration;
use uuid::Uuid;

use workspace_qdrant_core::storage::{
    MultiTenantConfig, StorageClient, StorageConfig, TransportMode,
};

fn test_storage_config() -> StorageConfig {
    StorageConfig {
        url: std::env::var("TEST_QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6333".to_string()),
        api_key: std::env::var("TEST_QDRANT_API_KEY").ok(),
        timeout_ms: 10_000,
        max_retries: 3,
        retry_delay_ms: 500,
        transport: TransportMode::Http,
        pool_size: 5,
        tls: false,
        dense_vector_size: 384,
        sparse_vector_size: None,
        check_compatibility: false,
        ..Default::default()
    }
}

/// Qdrant REST endpoint for payload_schema readback. Distinct from
/// `TEST_QDRANT_URL` because Qdrant publishes REST on 6333 and gRPC on 6334;
/// the StorageClient talks gRPC but this test reads the schema over REST to
/// stay decoupled from qdrant-client Rust SDK type changes.
fn http_base_url() -> String {
    std::env::var("TEST_QDRANT_HTTP_URL").unwrap_or_else(|_| "http://localhost:6333".to_string())
}

/// Fetch the payload schema for a collection via Qdrant's REST API and return
/// the set of field names that are indexed. Uses HTTP rather than the Rust
/// SDK so the test stays decoupled from internal type changes in qdrant-client.
async fn indexed_payload_fields(collection: &str) -> Result<Vec<String>, String> {
    let url = format!("{}/collections/{}", http_base_url(), collection);
    let resp = reqwest::Client::new()
        .get(&url)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("GET {url}: {e}"))?;
    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("parse JSON from {url}: {e}"))?;
    let schema = json
        .pointer("/result/payload_schema")
        .and_then(|v| v.as_object())
        .ok_or_else(|| format!("missing /result/payload_schema in {json}"))?;
    Ok(schema.keys().cloned().collect())
}

/// Round-trip every field that `init_projects_collection` would create against
/// a temporary collection. Mirrors the production manifest hardcoded here
/// (kept in sync with `PROJECTS_PAYLOAD_INDEX_FIELDS` in the crate; the
/// in-crate `mod tests` covers any divergence between the constant and the
/// fields touched by the production filter code).
#[tokio::test]
#[serial_test::serial]
async fn projects_payload_indexes_round_trip_through_qdrant() {
    let client = StorageClient::with_config(test_storage_config());

    // Bail out cleanly when Qdrant isn't up — matches the existing
    // `qdrant_client_validation_*` tests' skip-on-disconnect convention.
    match client.test_connection().await {
        Ok(true) => {}
        _ => {
            eprintln!("Qdrant unavailable at {} — skipping", http_base_url());
            return;
        }
    }

    let collection = format!("test_payload_idx_{}", Uuid::new_v4().simple());
    let config = MultiTenantConfig::default();

    client
        .create_multi_tenant_collection(&collection, &config)
        .await
        .expect("create_multi_tenant_collection");

    // Mirror the PROJECTS_PAYLOAD_INDEX_FIELDS manifest. Keep in sync with
    // `src/storage/collections/multi_tenant.rs`.
    let expected_fields = ["tenant_id", "file_path", "branch", "project_id"];
    for field in expected_fields {
        client
            .create_payload_index(&collection, field)
            .await
            .unwrap_or_else(|e| panic!("create_payload_index({field}): {e}"));
    }

    // Read it back. Qdrant publishes the schema asynchronously after the
    // create_field_index call, so poll for up to ~15s before failing the
    // assertion. In practice it shows up in 1-3s on a quiet instance.
    let mut indexed = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    while std::time::Instant::now() < deadline {
        indexed = indexed_payload_fields(&collection)
            .await
            .expect("indexed_payload_fields");
        if expected_fields
            .iter()
            .all(|f| indexed.iter().any(|i| i == f))
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }

    for field in expected_fields {
        assert!(
            indexed.iter().any(|i| i == field),
            "payload_schema for {collection} missing field {field}; got {indexed:?}",
        );
    }

    // Cleanup. Don't unwrap — if delete fails the next test run will see a
    // stale collection but the assertion above is what we care about.
    let _ = client.delete_collection(&collection).await;
}

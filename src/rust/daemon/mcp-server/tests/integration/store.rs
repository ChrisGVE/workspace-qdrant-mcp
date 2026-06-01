// Integration tests: store tool → real enqueue_item + idempotency_key cross-check.
//
// it_store_enqueue_item_idempotency_key_matches — enqueue_item against live daemon,
//     verify resp.idempotency_key matches wqm_common::hashing::generate_idempotency_key
// it_store_idempotency_key_stable_on_resubmit  — same payload submitted twice →
//     same idempotency_key returned both times (daemon dedup contract)

use super::helpers;
use mcp_server::canonicalize::payload_builders::build_store_payload;
use serde_json::Map;
use wqm_common::hashing::generate_idempotency_key;
use wqm_common::queue_types::{ItemType, QueueOperation};

// ---------------------------------------------------------------------------
// Live daemon: enqueue_item + idempotency_key cross-check
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_store_enqueue_item_idempotency_key_matches() {
    let mut client = match helpers::probe_daemon().await {
        Some(c) => c,
        None => return,
    };

    let tenant_id = "integration_test_tenant";
    let collection = wqm_common::constants::COLLECTION_LIBRARIES;
    let library_name = "integration_test_lib";
    let content = "integration test content — safe to discard";
    let document_id = format!("inttest_{}", nanos_hex());

    // Build the canonical payload_json the same way the MCP store tool does.
    let payload_json =
        build_store_payload(content, &document_id, "library", &Map::new(), library_name);

    // Compute the expected idempotency key locally.
    let expected_key = generate_idempotency_key(
        ItemType::Text,
        QueueOperation::Add,
        tenant_id,
        collection,
        &payload_json,
    )
    .expect("generate_idempotency_key must succeed for valid inputs");

    // Enqueue via gRPC.
    let resp = client
        .enqueue_item(
            ItemType::Text.to_string(),
            QueueOperation::Add.to_string(),
            tenant_id.to_string(),
            collection.to_string(),
            payload_json.clone(),
            "main".to_string(),
            None,
        )
        .await
        .expect("enqueue_item must succeed against live daemon");

    assert!(
        !resp.queue_id.is_empty(),
        "response queue_id must not be empty"
    );

    // Cross-check: daemon key == locally computed key.
    assert_eq!(
        resp.idempotency_key, expected_key,
        "daemon idempotency_key must match wqm_common::hashing::generate_idempotency_key"
    );
}

// ---------------------------------------------------------------------------
// Live daemon: same payload submitted twice → same idempotency_key (idempotent)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn it_store_idempotency_key_stable_on_resubmit() {
    let mut client = match helpers::probe_daemon().await {
        Some(c) => c,
        None => return,
    };

    let tenant_id = "integration_test_tenant";
    let collection = wqm_common::constants::COLLECTION_LIBRARIES;
    let library_name = "integration_test_lib_idem";
    let content = "idempotency stability test — safe to discard";
    // Fixed document_id so both submissions are truly identical.
    let document_id = "inttest_stable_idem_doc";

    let payload_json =
        build_store_payload(content, document_id, "library", &Map::new(), library_name);

    let resp1 = client
        .enqueue_item(
            ItemType::Text.to_string(),
            QueueOperation::Add.to_string(),
            tenant_id.to_string(),
            collection.to_string(),
            payload_json.clone(),
            "main".to_string(),
            None,
        )
        .await
        .expect("first enqueue_item must succeed");

    let resp2 = client
        .enqueue_item(
            ItemType::Text.to_string(),
            QueueOperation::Add.to_string(),
            tenant_id.to_string(),
            collection.to_string(),
            payload_json.clone(),
            "main".to_string(),
            None,
        )
        .await
        .expect("second enqueue_item must succeed");

    assert_eq!(
        resp1.idempotency_key, resp2.idempotency_key,
        "identical payloads must produce identical idempotency_key on both submissions"
    );
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn nanos_hex() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{n:08x}")
}

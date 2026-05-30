//! Assertion helpers for the TS↔Rust parity corpus.
//!
//! Each helper loads a committed corpus file under `tests/golden/parity/`
//! (captured from the TypeScript MCP server / its verbatim canonicalizer — see
//! `tests/golden/parity/README.md`), looks up a single case by `name`, drives
//! the corresponding Rust function, and asserts byte-for-byte parity with the
//! TS-sourced expected value.
//!
//! The per-case `#[test]` wrappers that call these helpers live in the
//! `gen_*.rs` files (one test per corpus row, so `cargo test` reports each
//! case individually).

use std::path::Path;

use serde_json::Value;

use mcp_server::canonicalize::stable_stringify::stable_stringify;
use wqm_common::hashing::generate_idempotency_key;
use wqm_common::queue_types::{ItemType, QueueOperation};

/// Load a parity corpus array by file stem (e.g. `"idempotency"`).
fn load_corpus(stem: &str) -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden/parity")
        .join(stem)
        .with_extension("json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read corpus {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse corpus {}: {e}", path.display()))
}

/// Find a single case by its `name` field.
fn case(stem: &str, name: &str) -> Value {
    load_corpus(stem)
        .into_iter()
        .find(|c| c["name"].as_str() == Some(name))
        .unwrap_or_else(|| panic!("corpus {stem}: no case named {name:?}"))
}

fn expect_str<'a>(c: &'a Value, field: &str) -> &'a str {
    c[field]
        .as_str()
        .unwrap_or_else(|| panic!("case field {field:?} must be a string"))
}

/// `stable_stringify(value) == <TS stableStringify output>`.
pub fn assert_stable_stringify(name: &str) {
    let c = case("stable_stringify", name);
    let actual = stable_stringify(&c["value"]);
    let expected = expect_str(&c, "expected");
    assert_eq!(
        actual, expected,
        "stable_stringify[{name}] mismatch\n  actual:   {actual}\n  expected: {expected}"
    );
}

/// Two assertions per idempotency case:
/// 1. `stable_stringify(payload)` matches the TS canonical `payload_json`.
/// 2. `generate_idempotency_key(...)` matches the TS-sourced 32-hex key.
pub fn assert_idempotency(name: &str) {
    let c = case("idempotency", name);

    let payload_json = stable_stringify(&c["payload"]);
    let expected_payload = expect_str(&c, "expectedPayloadJson");
    assert_eq!(
        payload_json, expected_payload,
        "idempotency[{name}] payload_json mismatch\n  actual:   {payload_json}\n  expected: {expected_payload}"
    );

    let item_type = ItemType::parse_str(expect_str(&c, "itemType"))
        .unwrap_or_else(|| panic!("idempotency[{name}]: bad itemType"));
    let op = QueueOperation::parse_str(expect_str(&c, "op"))
        .unwrap_or_else(|| panic!("idempotency[{name}]: bad op"));
    let key = generate_idempotency_key(
        item_type,
        op,
        expect_str(&c, "tenantId"),
        expect_str(&c, "collection"),
        &payload_json,
    )
    .unwrap_or_else(|e| panic!("idempotency[{name}]: key error: {e:?}"));
    let expected_key = expect_str(&c, "expectedKey");
    assert_eq!(
        key, expected_key,
        "idempotency[{name}] key mismatch\n  actual:   {key}\n  expected: {expected_key}"
    );
}

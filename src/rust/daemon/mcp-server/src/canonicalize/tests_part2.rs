//! Canonicalization tests part 2: AC-CANON6 (escape goldens) and
//! stable_stringify primitive unit tests.
//!
//! Included from `tests.rs` via `#[path = "tests_part2.rs"] mod part2;`.

use super::super::payload_builders::{build_rule_payload, build_store_payload, RulePayloadInput};
use super::super::stable_stringify::stable_stringify;
use serde_json::{json, Value};
use wqm_common::hashing::generate_idempotency_key;
use wqm_common::queue_types::{ItemType, QueueOperation};

// ============================================================
// AC-CANON6: TS-captured goldens for escape-triggering string VALUES
//
// Goldens were produced by running the Node.js script
// `tmp/20260529-1500_escape_goldens.mjs` against the live
// TypeScript stableStringify (queue-operations.ts:36-47) and
// Node.js 24.15.0.  The script was deleted after capture.
// ============================================================

/// AC-CANON6a: rule `content` containing \n, ", and \\  (multi-line rule text)
///
/// TS golden:
///   payload_json = {"action":"add","content":"First line\nSecond \"quoted\" line\nWith backslash: \\end","label":"escape-test-a","scope":"global","source_type":"rule"}
///   key          = 42dbeed3563e40198697485a600cbbe0
#[test]
fn ac_canon6a_content_with_newline_quote_backslash() {
    let content = "First line\nSecond \"quoted\" line\nWith backslash: \\end";
    let payload = build_rule_payload(RulePayloadInput {
        action: "add",
        label: "escape-test-a",
        content: Some(content),
        scope: Some("global"),
        project_id: None,
        title: None,
        tags: None,
        priority: None,
    });
    assert_eq!(
        payload,
        r#"{"action":"add","content":"First line\nSecond \"quoted\" line\nWith backslash: \\end","label":"escape-test-a","scope":"global","source_type":"rule"}"#,
        "AC-CANON6a: payload_json must match TS golden for content with \\n, \", \\\\"
    );
    let key = generate_idempotency_key(
        ItemType::Text,
        QueueOperation::Add,
        "global",
        "rules",
        &payload,
    )
    .expect("AC-CANON6a: key generation must succeed");
    assert_eq!(
        key, "42dbeed3563e40198697485a600cbbe0",
        "AC-CANON6a: idempotency key must match TS-computed golden"
    );
}

/// AC-CANON6b: rule `title` containing a tab (\t)
///
/// TS golden:
///   payload_json = {"action":"add","content":"Some rule content","label":"escape-test-b","scope":"global","source_type":"rule","title":"Tabbed\tTitle"}
///   key          = 968f63a912ad9636969bc03cd11394ee
#[test]
fn ac_canon6b_title_with_tab() {
    let payload = build_rule_payload(RulePayloadInput {
        action: "add",
        label: "escape-test-b",
        content: Some("Some rule content"),
        scope: Some("global"),
        project_id: None,
        title: Some("Tabbed\tTitle"),
        tags: None,
        priority: None,
    });
    assert_eq!(
        payload,
        r#"{"action":"add","content":"Some rule content","label":"escape-test-b","scope":"global","source_type":"rule","title":"Tabbed\tTitle"}"#,
        "AC-CANON6b: payload_json must match TS golden for title with tab"
    );
    let key = generate_idempotency_key(
        ItemType::Text,
        QueueOperation::Add,
        "global",
        "rules",
        &payload,
    )
    .expect("AC-CANON6b: key generation must succeed");
    assert_eq!(
        key, "968f63a912ad9636969bc03cd11394ee",
        "AC-CANON6b: idempotency key must match TS-computed golden"
    );
}

/// AC-CANON6c: store metadata value containing U+000B (vertical tab) and U+007F (DEL)
///
/// JS `JSON.stringify` behaviour (confirmed via Node 24.15.0):
///   - U+000B (0x0B, < 0x20, not a named escape like \n/\t)
///     -> JS emits the 6-char escape sequence ``
///   - U+007F (0x7F, > 0x1F, so not a C0 control)
///     -> emitted RAW as the 0x7F byte (not escaped)
///
/// TS golden (from `tmp/20260529-1500_escape_goldens.mjs`):
///   payload_json contains `..."beforeafter<0x7F>end"...`
///   key = 4a429c9ea3bc627c7994c41d6d9e130f
///   (where <0x7F> is the literal DEL byte, not an escape sequence)
#[test]
fn ac_canon6c_metadata_vtab_and_del() {
    let mut meta = serde_json::Map::new();
    // U+000B (vertical tab) followed by U+007F (DEL) embedded in ASCII
    meta.insert(
        "special".to_string(),
        Value::String("before\u{000B}after\u{007F}end".to_string()),
    );
    let payload = build_store_payload(
        "Store content",
        "docid-escape-c",
        "user_input",
        &meta,
        "test-lib",
    );

    // Build expected byte-by-byte to avoid ambiguity around embedded
    // control chars in Rust string literals.
    // The JSON-encoded value must be:
    //   "beforeafter<0x7F>end"
    // i.e. U+000B -> the 6-char escape ``; U+007F -> raw 0x7F byte.
    let expected = {
        let mut s = String::new();
        s.push_str(
            r#"{"content":"Store content","document_id":"docid-escape-c","library_name":"test-lib","metadata":{"special":""#,
        );
        s.push_str("before");
        // U+000B in JSON -> the 6-char escape sequence (backslash u 0 0 0 b)
        s.push_str("\\u000b");
        s.push_str("after");
        // U+007F in JSON -> emitted raw (0x7F byte) because 0x7F > 0x1F
        s.push('\u{007F}');
        s.push_str("end");
        s.push_str(r#""},"source_type":"user_input"}"#);
        s
    };
    assert_eq!(
        payload, expected,
        "AC-CANON6c: U+000B must be encoded as \\u000b; U+007F must be emitted raw"
    );

    let key = generate_idempotency_key(
        ItemType::Tenant,
        QueueOperation::Add,
        "test-lib",
        "libraries",
        &payload,
    )
    .expect("AC-CANON6c: key generation must succeed");
    assert_eq!(
        key, "4a429c9ea3bc627c7994c41d6d9e130f",
        "AC-CANON6c: idempotency key must match TS-computed golden"
    );
}

/// AC-CANON6d: value containing forward slash — must NOT be escaped end-to-end
///
/// JS `JSON.stringify` does not escape `/`; this test verifies the Rust
/// builder + canonicalizer preserves that behaviour through to the key.
///
/// TS golden:
///   payload_json = {"action":"add","content":"path/to/resource","label":"escape-test-d","scope":"global","source_type":"rule"}
///   key          = 5cb7eb4a8aba1b7ee773d813e4243b18
#[test]
fn ac_canon6d_forward_slash_not_escaped() {
    let payload = build_rule_payload(RulePayloadInput {
        action: "add",
        label: "escape-test-d",
        content: Some("path/to/resource"),
        scope: Some("global"),
        project_id: None,
        title: None,
        tags: None,
        priority: None,
    });
    assert_eq!(
        payload,
        r#"{"action":"add","content":"path/to/resource","label":"escape-test-d","scope":"global","source_type":"rule"}"#,
        "AC-CANON6d: forward slash must NOT be escaped in the payload JSON"
    );
    assert!(
        !payload.contains("\\/"),
        "AC-CANON6d: payload must not contain escaped slash"
    );
    let key = generate_idempotency_key(
        ItemType::Text,
        QueueOperation::Add,
        "global",
        "rules",
        &payload,
    )
    .expect("AC-CANON6d: key generation must succeed");
    assert_eq!(
        key, "5cb7eb4a8aba1b7ee773d813e4243b18",
        "AC-CANON6d: idempotency key must match TS-computed golden"
    );
}

// ============================================================
// Additional unit tests for stable_stringify primitives
// ============================================================

#[test]
fn stable_stringify_null() {
    assert_eq!(stable_stringify(&Value::Null), "null");
}

#[test]
fn stable_stringify_bool() {
    assert_eq!(stable_stringify(&Value::Bool(true)), "true");
    assert_eq!(stable_stringify(&Value::Bool(false)), "false");
}

#[test]
fn stable_stringify_string_basic() {
    assert_eq!(
        stable_stringify(&Value::String("hello".to_string())),
        "\"hello\""
    );
}

#[test]
fn stable_stringify_string_escapes() {
    // Verify JS JSON.stringify-compatible escaping
    assert_eq!(
        stable_stringify(&Value::String("line1\nline2".to_string())),
        "\"line1\\nline2\""
    );
    assert_eq!(
        stable_stringify(&Value::String("tab\there".to_string())),
        "\"tab\\there\""
    );
    assert_eq!(
        stable_stringify(&Value::String("quote\"here".to_string())),
        "\"quote\\\"here\""
    );
    assert_eq!(
        stable_stringify(&Value::String("back\\slash".to_string())),
        "\"back\\\\slash\""
    );
    // Forward slash NOT escaped (JS JSON.stringify does NOT escape /)
    assert_eq!(
        stable_stringify(&Value::String("path/to/file".to_string())),
        "\"path/to/file\""
    );
}

#[test]
fn stable_stringify_number_integer() {
    // i64 stored as JSON Number must not get .0
    let val: Value = serde_json::from_str("42").unwrap();
    assert_eq!(stable_stringify(&val), "42");
    let val: Value = serde_json::from_str("0").unwrap();
    assert_eq!(stable_stringify(&val), "0");
    let val: Value = serde_json::from_str("-7").unwrap();
    assert_eq!(stable_stringify(&val), "-7");
}

#[test]
fn stable_stringify_number_float() {
    let val: Value = serde_json::from_str("3.14").unwrap();
    assert_eq!(stable_stringify(&val), "3.14");
}

#[test]
fn stable_stringify_empty_array() {
    assert_eq!(stable_stringify(&json!([])), "[]");
}

#[test]
fn stable_stringify_array_order_preserved() {
    // Arrays preserve insertion order, NOT sorted
    assert_eq!(
        stable_stringify(&json!(["c", "a", "b"])),
        r#"["c","a","b"]"#
    );
}

#[test]
fn stable_stringify_empty_object() {
    assert_eq!(stable_stringify(&json!({})), "{}");
}

#[test]
fn stable_stringify_object_keys_sorted_ascii() {
    let val = json!({"z": 1, "a": 2, "m": 3});
    assert_eq!(stable_stringify(&val), r#"{"a":2,"m":3,"z":1}"#);
}

#[test]
fn stable_stringify_nested() {
    let val = json!({"b": {"y": 1, "x": 2}, "a": [3, 1, 2]});
    assert_eq!(stable_stringify(&val), r#"{"a":[3,1,2],"b":{"x":2,"y":1}}"#);
}

#[test]
fn stable_stringify_metadata_always_present_empty() {
    // metadata:{} must appear even when empty
    let json = build_store_payload(
        "content",
        "docid",
        "user_input",
        &serde_json::Map::new(),
        "lib",
    );
    assert!(
        json.contains("\"metadata\":{}"),
        "metadata must always be present, got: {json}"
    );
}

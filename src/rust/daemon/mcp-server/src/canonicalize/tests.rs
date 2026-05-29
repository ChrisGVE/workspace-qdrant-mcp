//! Canonicalization corpus tests — golden values captured from the live
//! TypeScript `stableStringify` in queue-operations.ts (lines 36-47).
//!
//! Every expected string was produced by running the Node script
//! `tmp/20260529-2134_golden_canonicalize.mjs` and is the authoritative
//! byte-for-byte reference for the Rust implementation.

use super::payload_builders::{build_rule_payload, build_store_payload, RulePayloadInput};
use super::stable_stringify::stable_stringify;
use serde_json::{json, Value};
use wqm_common::hashing::generate_idempotency_key;
use wqm_common::queue_types::{ItemType, QueueOperation};

// ============================================================
// AC-CANON1: byte-match with TS stableStringify golden output
// ============================================================

/// AC-CANON1a: store-library with empty metadata
#[test]
fn ac_canon1a_store_library_empty_metadata() {
    let json = build_store_payload(
        "Hello world documentation",
        "deadbeef00112233445566778899aabb",
        "user_input",
        &serde_json::Map::new(),
        "my-library",
    );
    assert_eq!(
        json,
        r#"{"content":"Hello world documentation","document_id":"deadbeef00112233445566778899aabb","library_name":"my-library","metadata":{},"source_type":"user_input"}"#,
        "AC-CANON1a: store payload with empty metadata must match TS golden"
    );
}

/// AC-CANON1b: store-library with non-empty metadata including astral-char keys
#[test]
fn ac_canon1b_store_library_astral_metadata_keys() {
    let mut meta = serde_json::Map::new();
    meta.insert("source_type".to_string(), Value::String("web".to_string()));
    // U+10000 𐀀 — surrogate pair D800 DC00 in UTF-16
    meta.insert(
        "\u{10000}linear".to_string(),
        Value::String("astral-a".to_string()),
    );
    // U+1F600 😀 — surrogate pair D83D DE00 in UTF-16
    meta.insert(
        "\u{1F600}emoji".to_string(),
        Value::String("astral-b".to_string()),
    );
    // U+2603 ☃ — BMP snowman, sorts before surrogates
    meta.insert(
        "snowman\u{2603}".to_string(),
        Value::String("bmp".to_string()),
    );
    meta.insert("alpha".to_string(), Value::String("first".to_string()));

    let json = build_store_payload(
        "Library content with unicode",
        "aabb001122334455deadbeef00001111",
        "web",
        &meta,
        "unicode-lib",
    );
    assert_eq!(
        json,
        "{\"content\":\"Library content with unicode\",\"document_id\":\"aabb001122334455deadbeef00001111\",\"library_name\":\"unicode-lib\",\"metadata\":{\"alpha\":\"first\",\"snowman\u{2603}\":\"bmp\",\"source_type\":\"web\",\"\u{10000}linear\":\"astral-a\",\"\u{1F600}emoji\":\"astral-b\"},\"source_type\":\"web\"}",
        "AC-CANON1b: astral-char metadata keys must sort by UTF-16 code units"
    );
}

/// AC-CANON1c: rules add with priority:0 and empty tags array
#[test]
fn ac_canon1c_rules_add_priority_zero_empty_tags() {
    let json = build_rule_payload(RulePayloadInput {
        action: "add",
        label: "prefer-uv",
        content: Some("Always use uv for Python dependency management"),
        scope: Some("global"),
        project_id: None,
        title: Some("Prefer uv"),
        tags: Some(vec![]), // empty array → INCLUDED ([] is truthy in JS)
        priority: Some(0),  // priority:0 → INCLUDED (not undefined)
    });
    assert_eq!(
        json,
        r#"{"action":"add","content":"Always use uv for Python dependency management","label":"prefer-uv","priority":0,"scope":"global","source_type":"rule","tags":[],"title":"Prefer uv"}"#,
        "AC-CANON1c: priority:0 must be included, empty tags array must be included"
    );
}

/// AC-CANON1d: rules add with non-empty tags array and project scope
#[test]
fn ac_canon1d_rules_add_truthy_tags() {
    let json = build_rule_payload(RulePayloadInput {
        action: "add",
        label: "prefer-bun",
        content: Some("Use bun instead of npm"),
        scope: Some("project"),
        project_id: Some("proj_abc123"),
        title: Some("Prefer bun"),
        tags: Some(vec!["tooling", "workflow"]),
        priority: Some(5),
    });
    assert_eq!(
        json,
        r#"{"action":"add","content":"Use bun instead of npm","label":"prefer-bun","priority":5,"project_id":"proj_abc123","scope":"project","source_type":"rule","tags":["tooling","workflow"],"title":"Prefer bun"}"#,
        "AC-CANON1d: rules add with project scope and tags"
    );
}

/// AC-CANON1e: rules add — empty-string title DROPPED, no tags, no priority
#[test]
fn ac_canon1e_rules_add_empty_title_dropped() {
    let json = build_rule_payload(RulePayloadInput {
        action: "add",
        label: "no-stubs",
        content: Some("Never use stubs or placeholder code"),
        scope: Some("global"),
        project_id: None,
        title: Some(""), // empty string → falsy in JS → DROPPED
        tags: None,      // None → omitted
        priority: None,  // None → omitted
    });
    assert_eq!(
        json,
        r#"{"action":"add","content":"Never use stubs or placeholder code","label":"no-stubs","scope":"global","source_type":"rule"}"#,
        "AC-CANON1e: empty-string title must be dropped; absent tags/priority omitted"
    );
}

/// AC-CANON1f: rules update with priority
#[test]
fn ac_canon1f_rules_update_with_priority() {
    let json = build_rule_payload(RulePayloadInput {
        action: "update",
        label: "prefer-uv",
        content: Some("Updated: always use uv for Python"),
        scope: Some("global"),
        project_id: None,
        title: None,
        tags: None,
        priority: Some(8),
    });
    assert_eq!(
        json,
        r#"{"action":"update","content":"Updated: always use uv for Python","label":"prefer-uv","priority":8,"scope":"global","source_type":"rule"}"#,
        "AC-CANON1f: rules update must include priority"
    );
}

/// AC-CANON1g: rules remove (minimal payload — no content, scope, etc.)
#[test]
fn ac_canon1g_rules_remove_minimal() {
    let json = build_rule_payload(RulePayloadInput {
        action: "remove",
        label: "old-rule",
        content: None,
        scope: None,
        project_id: None,
        title: None,
        tags: None,
        priority: None,
    });
    assert_eq!(
        json, r#"{"action":"remove","label":"old-rule","source_type":"rule"}"#,
        "AC-CANON1g: rules remove must produce minimal payload"
    );
}

// ============================================================
// AC-CANON2: integers render without .0
// ============================================================

#[test]
fn ac_canon2_integer_no_dot_zero() {
    let val = json!({
        "count": 42,
        "float_val": 3.14,
        "neg": -7,
        "priority": 0,
        "zero": 0,
    });
    let json = stable_stringify(&val);
    assert_eq!(
        json, r#"{"count":42,"float_val":3.14,"neg":-7,"priority":0,"zero":0}"#,
        "AC-CANON2: integers must not have .0 suffix; floats preserve decimal"
    );
}

// ============================================================
// AC-CANON3: astral-char key sort by UTF-16 code units
// ============================================================

#[test]
fn ac_canon3_utf16_sort_order() {
    // Keys and their UTF-16 representations:
    //   "Akey"   → [0x0041, 0x006B, 0x0065, 0x0079]
    //   "☃key"   → [0x2603, 0x006B, 0x0065, 0x0079]
    //   "𐀀key"   → [0xD800, 0xDC00, 0x006B, 0x0065, 0x0079]  (U+10000)
    //   "😀key"   → [0xD83D, 0xDE00, 0x006B, 0x0065, 0x0079]  (U+1F600)
    // Expected JS sort order: Akey < ☃key < 𐀀key < 😀key
    let val = json!({
        "\u{1F600}key": "emoji",
        "\u{10000}key": "astral-a",
        "\u{2603}key": "snowman",
        "Akey": "latin",
    });
    let json = stable_stringify(&val);
    assert_eq!(
        json,
        "{\"Akey\":\"latin\",\"\u{2603}key\":\"snowman\",\"\u{10000}key\":\"astral-a\",\"\u{1F600}key\":\"emoji\"}",
        "AC-CANON3: astral chars must sort by UTF-16 code units (surrogates)"
    );
}

// ============================================================
// AC-CANON4: tags:Some(vec![]) INCLUDED; None OMITTED
// ============================================================

#[test]
fn ac_canon4a_empty_tags_array_included() {
    let json = build_rule_payload(RulePayloadInput {
        action: "add",
        label: "test-tags-empty",
        content: Some("Test with empty tags array"),
        scope: Some("global"),
        project_id: None,
        title: None,
        tags: Some(vec![]), // empty array → truthy in JS → INCLUDED
        priority: None,
    });
    assert_eq!(
        json,
        r#"{"action":"add","content":"Test with empty tags array","label":"test-tags-empty","scope":"global","source_type":"rule","tags":[]}"#,
        "AC-CANON4a: Some(vec![]) must produce \"tags\":[] — empty array is truthy in JS"
    );
}

#[test]
fn ac_canon4b_none_tags_omitted() {
    let json = build_rule_payload(RulePayloadInput {
        action: "add",
        label: "test-tags-null",
        content: Some("Test with null/undefined tags"),
        scope: Some("global"),
        project_id: None,
        title: None,
        tags: None, // None → undefined in JS → omitted
        priority: None,
    });
    assert_eq!(
        json,
        r#"{"action":"add","content":"Test with null/undefined tags","label":"test-tags-null","scope":"global","source_type":"rule"}"#,
        "AC-CANON4b: None tags must be omitted entirely"
    );
}

// ============================================================
// AC-CANON5: idempotency key matches wqm_common::hashing output
// ============================================================

#[test]
fn ac_canon5_idempotency_key_round_trip() {
    let meta = {
        let mut m = serde_json::Map::new();
        m.insert(
            "source".to_string(),
            Value::String("mcp_store_tool".to_string()),
        );
        m
    };
    let payload_json = build_store_payload(
        "Test content for key verification",
        "00112233445566778899aabbccddeeff",
        "user_input",
        &meta,
        "test-library",
    );

    // Verify payload_json matches golden
    assert_eq!(
        payload_json,
        r#"{"content":"Test content for key verification","document_id":"00112233445566778899aabbccddeeff","library_name":"test-library","metadata":{"source":"mcp_store_tool"},"source_type":"user_input"}"#,
        "AC-CANON5: store payload must match golden before key computation"
    );

    // Compute key via wqm_common::hashing (same formula as daemon)
    let key = generate_idempotency_key(
        ItemType::Tenant,
        QueueOperation::Add,
        "test-library",
        "libraries",
        &payload_json,
    )
    .expect("idempotency key generation must succeed");

    // Key must match the value computed by the Node golden script
    assert_eq!(
        key, "10f052a970b67f7fabe1f54ab2e2aecc",
        "AC-CANON5: Rust key must match Node-computed key (same SHA256 formula)"
    );
}

/// Verify all 7 rule corpus cases produce keys matching TS node script
#[test]
fn ac_canon5_rule_keys_match_ts() {
    let cases: &[(&str, &str, &str, &str)] = &[
        (
            "ac_canon1c",
            r#"{"action":"add","content":"Always use uv for Python dependency management","label":"prefer-uv","priority":0,"scope":"global","source_type":"rule","tags":[],"title":"Prefer uv"}"#,
            "global",
            "71516ada387ffa178100bb4d8f965e8b",
        ),
        (
            "ac_canon1d",
            r#"{"action":"add","content":"Use bun instead of npm","label":"prefer-bun","priority":5,"project_id":"proj_abc123","scope":"project","source_type":"rule","tags":["tooling","workflow"],"title":"Prefer bun"}"#,
            "proj_abc123",
            "27db1f37ff6d735503a6145b1c0d8b96",
        ),
        (
            "ac_canon1e",
            r#"{"action":"add","content":"Never use stubs or placeholder code","label":"no-stubs","scope":"global","source_type":"rule"}"#,
            "global",
            "0a13dcacc2d1d440c64caaa469d644d3",
        ),
        (
            "ac_canon1f",
            r#"{"action":"update","content":"Updated: always use uv for Python","label":"prefer-uv","priority":8,"scope":"global","source_type":"rule"}"#,
            "global",
            "469a0c36280008eacd015f5e0fb3fc3a",
        ),
        (
            "ac_canon1g",
            r#"{"action":"remove","label":"old-rule","source_type":"rule"}"#,
            "global",
            "14357d08b523b229eeaa03c2bc720a3b",
        ),
    ];

    for (name, payload_json, tenant_id, expected_key) in cases {
        // Determine op from payload JSON
        let op = if payload_json.contains("\"action\":\"update\"") {
            QueueOperation::Update
        } else if payload_json.contains("\"action\":\"remove\"") {
            QueueOperation::Delete
        } else {
            QueueOperation::Add
        };
        let key = generate_idempotency_key(ItemType::Text, op, tenant_id, "rules", payload_json)
            .unwrap_or_else(|e| panic!("{name}: key gen failed: {e}"));
        assert_eq!(&key, expected_key, "{name}: Rust key must match TS golden");
    }
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

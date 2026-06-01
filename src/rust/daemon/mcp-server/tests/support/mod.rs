//! Shared helpers for the golden conformance tests.
//!
//! Lives under `tests/support/` (a subdirectory) so Cargo does NOT compile it
//! as its own test binary — it is included via `mod support;` from
//! `conformance.rs`.

use std::path::Path;

use serde_json::Value;

/// Extract the inner result text from a `CallToolResult` (the single text
/// content block produced by the envelope helpers).
pub fn content_text(r: &rmcp::model::CallToolResult) -> &str {
    r.content
        .first()
        .expect("content must not be empty")
        .raw
        .as_text()
        .expect("content must be text")
        .text
        .as_str()
}

/// Load a golden fixture by path relative to `tests/golden/`.
pub fn load_golden(rel: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(rel)
        .with_extension("json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read golden {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse golden {}: {e}", path.display()))
}

/// Assert error-envelope parity.
///
/// Error-envelope goldens have shape `{ "is_error": true, "text": "Error: ..." }`
/// (or `"Unknown tool: ..."`). Verifies (a) `is_error == Some(true)` and (b) the
/// content text matches the golden's `text` field byte-for-byte.
pub fn assert_error_envelope(result: &rmcp::model::CallToolResult, golden: &Value) {
    assert_eq!(
        result.is_error,
        Some(true),
        "error envelope: expected is_error=true"
    );
    let golden_text = golden["text"].as_str().expect("golden.text must be string");
    let actual_text = content_text(result);
    assert_eq!(
        actual_text, golden_text,
        "error envelope text mismatch\n  actual:   {actual_text:?}\n  expected: {golden_text:?}"
    );
}

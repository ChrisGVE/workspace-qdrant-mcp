//! Canonical JSON serializer matching the TypeScript `stableStringify` from
//! `src/typescript/mcp-server/src/clients/queue-operations.ts` (lines 36-47).
//!
//! # Semantics
//! - `null` / scalars → `JSON.stringify` equivalent
//! - strings → JSON-escaped with double quotes; forward-slash NOT escaped
//! - numbers → integer representation without trailing `.0` (RISK-2)
//! - arrays → `[e0,e1,…]` with order PRESERVED (not sorted)
//! - objects → `{"k0":v0,"k1":v1,…}` with keys sorted by **UTF-16 code
//!   unit sequence** (`str::encode_utf16()`), matching JavaScript's
//!   `Array.prototype.sort` behaviour for astral characters (≥ U+10000).
//!   No whitespace anywhere.

use serde_json::Value;

/// Sort comparator: compare two string keys by their UTF-16 code unit sequences.
///
/// JavaScript's `String.prototype.localeCompare`-free `Array.prototype.sort`
/// compares strings by UTF-16 code unit values.  For astral code points
/// (≥ U+10000) this means the surrogate pair values (0xD800–0xDFFF) are used,
/// not the Unicode scalar value.  Rust's default `str` `Ord` uses Unicode
/// scalar ordering, which differs for astral characters.
pub fn compare_keys_utf16(a: &str, b: &str) -> std::cmp::Ordering {
    let a_utf16: Vec<u16> = a.encode_utf16().collect();
    let b_utf16: Vec<u16> = b.encode_utf16().collect();
    a_utf16.cmp(&b_utf16)
}

/// Produce a canonical JSON string from a `serde_json::Value`.
///
/// Mirrors `stableStringify` in `queue-operations.ts`:
/// ```text
/// function stableStringify(value: unknown): string {
///   if (value === null || typeof value !== 'object') return JSON.stringify(value);
///   if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
///   const obj = value as Record<string, unknown>;
///   const sortedKeys = Object.keys(obj).sort();
///   const entries = sortedKeys.map(k => `${JSON.stringify(k)}:${stableStringify(obj[k])}`);
///   return `{${entries.join(',')}}`;
/// }
/// ```
pub fn stable_stringify(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => stringify_number(n),
        Value::String(s) => json_escape_string(s),
        Value::Array(arr) => {
            let elements: Vec<String> = arr.iter().map(stable_stringify).collect();
            format!("[{}]", elements.join(","))
        }
        Value::Object(map) => {
            // Collect and sort keys by UTF-16 code unit sequence
            let mut keys: Vec<&str> = map.keys().map(String::as_str).collect();
            keys.sort_by(|a, b| compare_keys_utf16(a, b));

            let entries: Vec<String> = keys
                .iter()
                .map(|k| format!("{}:{}", json_escape_string(k), stable_stringify(&map[*k])))
                .collect();
            format!("{{{}}}", entries.join(","))
        }
    }
}

/// Serialize a JSON number without trailing `.0` for integers.
///
/// `serde_json::Number` may represent i64, u64, or f64.  JavaScript's single
/// number type means `priority: 0` serializes as `0`, not `0.0`.  When the
/// number is representable as an integer we emit it without a decimal point.
///
/// # Float fallback — latent divergence hazard
///
/// The MCP write-path payloads produced by this crate contain **no float
/// values**: `priority` is always `i64`, and every `metadata` value that
/// reaches this function is a JSON string emitted by the TypeScript MCP
/// server.  Therefore the `f64` branch below is **not expected to be
/// reached** during normal operation.
///
/// If it ever is reached (e.g. a caller passes a manually-constructed
/// `serde_json::Value::Number` that is not representable as i64 or u64),
/// `serde_json::Number::to_string()` is used as a best-effort fallback.
/// **This diverges from JavaScript's `JSON.stringify` in two magnitude
/// bands:**
/// - `[1e20, 1e21)` — JS switches to exponential notation (`1e+20`);
///   serde_json emits the full decimal mantissa.
/// - Absolute values below `5e-7` — JS uses exponential notation (`5e-7`);
///   serde_json may emit `0.0000005`.
/// - Integers in `(i64::MAX, u64::MAX]` — the u64 branch above handles
///   these correctly; they never reach the float path.
///
/// Porting JS's exact `Number.prototype.toString` float formatting is
/// out-of-scope for the current task; if float payloads ever appear on
/// the write-path, that parity work should be tracked as a follow-up.
fn stringify_number(n: &serde_json::Number) -> String {
    // Try i64 first (negative integers), then u64 (large positive), then f64.
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    // Floating-point fallback: use serde_json's Display.
    // See doc-comment above for known divergence bands vs JS JSON.stringify.
    //
    // LATENT HAZARD: if this branch is reached on the MCP write-path (which
    // is not expected — all current payload numeric fields are i64/u64),
    // the output MAY diverge from JS JSON.stringify in the bands documented
    // above.  A debug_assert! here was considered but omitted because the
    // stable_stringify function is also used in unit tests that exercise float
    // formatting explicitly, and the assert would break those tests without
    // providing correctness value on paths that never see production floats.
    n.to_string()
}

/// Produce a JSON-encoded string literal with double quotes.
///
/// Matches `JSON.stringify(s)` escaping rules:
/// - `"` → `\"`
/// - `\` → `\\`
/// - `\n` → `\n`  (U+000A)
/// - `\r` → `\r`  (U+000D)
/// - `\t` → `\t`  (U+0009)
/// - U+0008 → `\b`
/// - U+000C → `\f`
/// - U+0000–U+001F (other) → `\uXXXX` (4 lowercase hex digits)
/// - `/` is NOT escaped (JS `JSON.stringify` does not escape forward slash)
/// - Non-ASCII above U+001F are emitted as-is (UTF-8 bytes), not `\uXXXX`
fn json_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0C' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                // Other C0 control characters → \uXXXX
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

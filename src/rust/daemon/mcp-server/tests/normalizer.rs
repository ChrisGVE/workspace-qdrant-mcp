//! S10.4 normalizer for golden conformance comparisons.
//!
//! Applies deterministic mutations to a `serde_json::Value` before comparison
//! so that volatile fields (timing, random IDs, floats) do not cause spurious
//! failures.
//!
//! ## Rules
//!
//! 1. **Volatile field masking** — fields named `latency_ms`, `*_ms`, `queue_id`,
//!    `session_id`, `event_id`, `createdAt`, `updatedAt` are replaced with the
//!    sentinel string `"MASKED"`. Presence is asserted but value is not compared.
//!
//! 2. **Float rounding (OQ-8)** — `score`, `similarity`, and `diversity_score`
//!    float values are rounded to 6 decimal places. This precision was chosen
//!    because Qdrant returns scores as 32-bit floats promoted to f64 by the Rust
//!    client, while the TS client uses the JS Number (f64 native). The 6th decimal
//!    digit is where rounding error first appears in practice; clamping there
//!    makes both sides compare equal.
//!
//! 3. **Equal-score ordering** — when a `results` array contains items with equal
//!    (post-rounding) `score` values, those items are sorted by `id` ascending.
//!    Items with distinct scores retain their original relative order.
//!
//! 4. **Health presence** — a `health` key is structural-only: the normalizer
//!    replaces its value with `true` when present so only presence/absence is
//!    compared, not the inner health object.
//!
//! 5. **Parsed JSON equality** — normalization produces values suitable for
//!    `assert_eq!(Value, Value)` (i.e., whitespace-insensitive).
//!
//! 6. **Canonical format** — `round_trip_canonical(text)` serializes the parsed
//!    value back to `serde_json::to_string_pretty` (2-space indent) for byte-exact
//!    format tests.

use serde_json::Value;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Apply all S10.4 normalization rules in-place.
///
/// Recursively processes `value`, mutating it:
/// - Objects: mask volatile keys, normalize health, recurse into values.
/// - Arrays: recurse into elements, then apply equal-score sort if the array
///   looks like a `results` list.
/// - Numbers: round if encountered at a score key (handled from the parent Object).
///
/// Call this on both the Rust output and the golden before comparing.
pub fn normalize(value: &mut Value) {
    normalize_inner(value, None);
}

/// Re-serialize a JSON text as canonical 2-space pretty-print.
///
/// Used by byte-exact format tests (rule 6).
/// Returns the canonical string, or the original text on parse failure.
pub fn round_trip_canonical(text: &str) -> String {
    match serde_json::from_str::<Value>(text) {
        Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| text.to_string()),
        Err(_) => text.to_string(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal
// ─────────────────────────────────────────────────────────────────────────────

/// Field names that are volatile (timing, random IDs, timestamps).
const VOLATILE_FIELDS: &[&str] = &[
    "latency_ms",
    "queue_id",
    "session_id",
    "event_id",
    "createdAt",
    "updatedAt",
];

/// Float fields to round to 6 decimal places.
const FLOAT_FIELDS: &[&str] = &["score", "similarity", "diversity_score"];

const FLOAT_PRECISION: f64 = 1_000_000.0; // 6 decimal places
const MASKED: &str = "MASKED";

fn normalize_inner(value: &mut Value, parent_key: Option<&str>) {
    match value {
        Value::Object(map) => {
            // 1. Mask volatile fields.
            for key in VOLATILE_FIELDS {
                if map.contains_key(*key) {
                    map.insert(key.to_string(), Value::String(MASKED.to_string()));
                }
            }
            // Also mask any key that ends with "_ms" (catches latency_ms, duration_ms etc.)
            let ms_keys: Vec<String> = map.keys().filter(|k| k.ends_with("_ms")).cloned().collect();
            for key in ms_keys {
                map.insert(key, Value::String(MASKED.to_string()));
            }

            // 4. Health presence: replace health value with boolean sentinel.
            if map.contains_key("health") {
                map.insert("health".to_string(), Value::Bool(true));
            }

            // 2. Round float fields, then recurse.
            for (k, v) in map.iter_mut() {
                if FLOAT_FIELDS.contains(&k.as_str()) {
                    round_float(v);
                } else {
                    normalize_inner(v, Some(k.as_str()));
                }
            }

            // 3. Equal-score sort on `results` arrays — handled after recursion.
            if let Some(Value::Array(arr)) = map.get_mut("results") {
                stable_sort_by_score_then_id(arr);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                normalize_inner(item, parent_key);
            }
        }
        _ => {}
    }
}

/// Round a JSON number to 6 decimal places. Non-numbers are left unchanged.
fn round_float(value: &mut Value) {
    if let Value::Number(n) = value {
        if let Some(f) = n.as_f64() {
            let rounded = (f * FLOAT_PRECISION).round() / FLOAT_PRECISION;
            // Convert back to JSON number (use serde_json::Number).
            // If the rounded value has no fractional part, serde_json may
            // serialize as integer — force float by including .0 if needed.
            if let Some(new_n) = serde_json::Number::from_f64(rounded) {
                *value = Value::Number(new_n);
            }
        }
    }
}

/// Stable sort of a `results` array by (rounded_score desc, id asc).
///
/// Items with different scores retain their original relative order (stable
/// sort on id only within tied-score groups). This mirrors the determinism
/// requirement in S10.4 rule 3.
fn stable_sort_by_score_then_id(arr: &mut Vec<Value>) {
    arr.sort_by(|a, b| {
        let score_a = extract_rounded_score(a);
        let score_b = extract_rounded_score(b);
        // Descending score first.
        match score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Equal => {
                // Ascending id for tie-breaking.
                let id_a = a.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let id_b = b.get("id").and_then(|v| v.as_str()).unwrap_or("");
                id_a.cmp(id_b)
            }
            other => other,
        }
    });
}

fn extract_rounded_score(item: &Value) -> f64 {
    item.get("score")
        .and_then(|v| v.as_f64())
        .map(|f| (f * FLOAT_PRECISION).round() / FLOAT_PRECISION)
        .unwrap_or(0.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests for the normalizer itself
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    // ── volatile field masking ────────────────────────────────────────────────

    #[test]
    fn masks_latency_ms() {
        let mut v = json!({ "latency_ms": 42 });
        normalize(&mut v);
        assert_eq!(v["latency_ms"], json!("MASKED"));
    }

    #[test]
    fn masks_queue_id() {
        let mut v = json!({ "queue_id": "abc-123" });
        normalize(&mut v);
        assert_eq!(v["queue_id"], json!("MASKED"));
    }

    #[test]
    fn masks_created_at() {
        let mut v = json!({ "createdAt": "2026-01-01T00:00:00Z" });
        normalize(&mut v);
        assert_eq!(v["createdAt"], json!("MASKED"));
    }

    #[test]
    fn masks_updated_at() {
        let mut v = json!({ "updatedAt": "2026-01-01T00:00:00Z" });
        normalize(&mut v);
        assert_eq!(v["updatedAt"], json!("MASKED"));
    }

    #[test]
    fn masks_arbitrary_ms_suffix() {
        let mut v = json!({ "duration_ms": 100, "response_ms": 50 });
        normalize(&mut v);
        assert_eq!(v["duration_ms"], json!("MASKED"));
        assert_eq!(v["response_ms"], json!("MASKED"));
    }

    #[test]
    fn non_volatile_field_preserved() {
        let mut v = json!({ "success": true, "total": 5 });
        normalize(&mut v);
        assert_eq!(v["success"], json!(true));
        assert_eq!(v["total"], json!(5));
    }

    // ── float rounding ────────────────────────────────────────────────────────

    #[test]
    fn rounds_score_to_six_places() {
        let mut v = json!({ "score": 0.123456789 });
        normalize(&mut v);
        // 0.123456789 rounded to 6 decimals = 0.123457
        let rounded = v["score"].as_f64().expect("numeric");
        assert!(
            (rounded - 0.123457_f64).abs() < 1e-10,
            "expected ~0.123457 got {rounded}"
        );
    }

    #[test]
    fn rounds_similarity_to_six_places() {
        let mut v = json!({ "similarity": 0.9999999 });
        normalize(&mut v);
        let rounded = v["similarity"].as_f64().expect("numeric");
        assert!(
            (rounded - 1.0_f64).abs() < 1e-10,
            "expected ~1.0 got {rounded}"
        );
    }

    #[test]
    fn does_not_round_non_float_fields() {
        let mut v = json!({ "total": 42, "files": 10 });
        normalize(&mut v);
        assert_eq!(v["total"], json!(42));
    }

    // ── equal-score sort ──────────────────────────────────────────────────────

    #[test]
    fn sorts_tied_scores_by_id() {
        let mut v = json!({
            "results": [
                { "id": "z", "score": 0.5 },
                { "id": "a", "score": 0.5 },
                { "id": "m", "score": 0.5 }
            ]
        });
        normalize(&mut v);
        let ids: Vec<&str> = v["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            vec!["a", "m", "z"],
            "tied scores should sort by id asc"
        );
    }

    #[test]
    fn keeps_distinct_scores_in_desc_order() {
        let mut v = json!({
            "results": [
                { "id": "a", "score": 0.9 },
                { "id": "b", "score": 0.7 },
                { "id": "c", "score": 0.5 }
            ]
        });
        normalize(&mut v);
        let scores: Vec<f64> = v["results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["score"].as_f64().unwrap())
            .collect();
        assert!(
            scores[0] > scores[1] && scores[1] > scores[2],
            "distinct scores must be descending"
        );
    }

    // ── health presence ───────────────────────────────────────────────────────

    #[test]
    fn health_present_normalized_to_bool_true() {
        let mut v = json!({ "health": { "status": "uncertain", "reason": "timeout" } });
        normalize(&mut v);
        assert_eq!(
            v["health"],
            json!(true),
            "health value replaced with sentinel"
        );
    }

    #[test]
    fn no_health_key_stays_absent() {
        let mut v = json!({ "results": [] });
        normalize(&mut v);
        assert!(v.get("health").is_none(), "health key must remain absent");
    }

    // ── canonical format ──────────────────────────────────────────────────────

    #[test]
    fn round_trip_canonical_produces_two_space_indent() {
        let input = r#"{"a":1,"b":2}"#;
        let canonical = round_trip_canonical(input);
        assert!(canonical.contains("  \"a\""), "should be 2-space indented");
    }

    #[test]
    fn round_trip_canonical_is_stable() {
        let input = r#"{"a":1}"#;
        let first = round_trip_canonical(input);
        let second = round_trip_canonical(&first);
        assert_eq!(first, second, "canonical should be idempotent");
    }

    // ── nested recursion ─────────────────────────────────────────────────────

    #[test]
    fn normalizes_nested_objects() {
        let mut v = json!({
            "outer": {
                "latency_ms": 10,
                "inner": { "queue_id": "x" }
            }
        });
        normalize(&mut v);
        assert_eq!(v["outer"]["latency_ms"], json!("MASKED"));
        assert_eq!(v["outer"]["inner"]["queue_id"], json!("MASKED"));
    }

    #[test]
    fn normalizes_arrays_of_objects() {
        let mut v = json!([
            { "score": 0.123456789 },
            { "score": 0.987654321 }
        ]);
        normalize(&mut v);
        // Both should be rounded
        let s0 = v[0]["score"].as_f64().unwrap();
        let s1 = v[1]["score"].as_f64().unwrap();
        assert!((s0 - 0.123457).abs() < 1e-10);
        assert!((s1 - 0.987654).abs() < 1e-10);
    }
}

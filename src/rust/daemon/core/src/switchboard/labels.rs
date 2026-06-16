//! Label canonicalization — stable JSON keys for the `control_baseline` PK.
//!
//! Labels are persisted as a canonical JSON object whose keys are sorted
//! alphabetically. A `BTreeMap` guarantees that order across daemon versions; a
//! plain `serde_json::Map` preserves insertion order, which would orphan PK
//! lookups after a code change reorders labels (arch §4a, data-F1). Used only at
//! persist time — never on the hot path.

use std::collections::BTreeMap;

/// Serialize labels to canonical JSON with alphabetically-sorted keys.
/// Empty map → `"{}"`.
pub fn canonicalize_labels(labels: &BTreeMap<&str, &str>) -> String {
    serde_json::to_string(labels).expect("BTreeMap<&str,&str> serialization cannot fail")
}

/// Parse canonical JSON labels back into an owned map.
pub fn parse_labels(json: &str) -> Result<BTreeMap<String, String>, serde_json::Error> {
    serde_json::from_str(json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keys_sorted_alphabetically() {
        let mut labels = BTreeMap::new();
        labels.insert("zone", "embedding");
        labels.insert("model", "fastembed");
        // BTreeMap orders keys: model < zone, regardless of insertion order.
        assert_eq!(
            canonicalize_labels(&labels),
            r#"{"model":"fastembed","zone":"embedding"}"#
        );
    }

    #[test]
    fn test_empty_labels() {
        let labels: BTreeMap<&str, &str> = BTreeMap::new();
        assert_eq!(canonicalize_labels(&labels), "{}");
    }

    #[test]
    fn test_roundtrip() {
        let mut labels = BTreeMap::new();
        labels.insert("model", "openai");
        let json = canonicalize_labels(&labels);
        let parsed = parse_labels(&json).unwrap();
        assert_eq!(parsed.get("model"), Some(&"openai".to_string()));
    }
}

//! Merge an override config document over typed defaults.
//!
//! The merge operates on a neutral [`serde_json::Value`]: objects merge
//! recursively (key-by-key); arrays and scalars **replace** wholesale. This
//! matches the TypeScript `mergeConfigs` spread semantics
//! (`{ ...base.section, ...override.section }`) for both flat sections and
//! nested ones (e.g. `rules.limits`).

use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;

use super::ConfigError;

/// Recursively merge `override_value` onto `base`, in place.
///
/// - Both objects → merge key-by-key (recursing into nested objects).
/// - Anything else (array, scalar, type mismatch) → `override_value` replaces
///   the base slot. Arrays therefore replace entirely; they are never appended.
pub fn merge_value(base: &mut Value, override_value: &Value) {
    match (base, override_value) {
        (Value::Object(base_map), Value::Object(ovr_map)) => {
            for (key, ovr) in ovr_map {
                match base_map.get_mut(key) {
                    Some(slot) => merge_value(slot, ovr),
                    None => {
                        base_map.insert(key.clone(), ovr.clone());
                    }
                }
            }
        }
        (slot, ovr) => *slot = ovr.clone(),
    }
}

/// Merge a parsed override document over typed `base` defaults and deserialize
/// back into the typed config `T`.
///
/// `T` must round-trip through [`serde_json::Value`]; fields absent from the
/// override keep their `base` value, which preserves compiled-in defaults.
pub fn merge_over_defaults<T>(base: T, override_value: &Value) -> Result<T, ConfigError>
where
    T: Serialize + DeserializeOwned,
{
    let mut base_value =
        serde_json::to_value(&base).map_err(|e| ConfigError::Merge(e.to_string()))?;
    merge_value(&mut base_value, override_value);
    serde_json::from_value(base_value).map_err(|e| ConfigError::Merge(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[test]
    fn objects_merge_recursively() {
        let mut base = json!({"a": {"x": 1, "y": 2}, "b": 3});
        merge_value(&mut base, &json!({"a": {"y": 20, "z": 30}}));
        assert_eq!(base, json!({"a": {"x": 1, "y": 20, "z": 30}, "b": 3}));
    }

    #[test]
    fn arrays_replace_not_append() {
        let mut base = json!({"patterns": ["*.py", "*.rs"]});
        merge_value(&mut base, &json!({"patterns": ["*.go"]}));
        assert_eq!(base, json!({"patterns": ["*.go"]}));
    }

    #[test]
    fn scalar_replaces() {
        let mut base = json!({"timeout": 30});
        merge_value(&mut base, &json!({"timeout": 5}));
        assert_eq!(base["timeout"], 5);
    }

    #[test]
    fn type_mismatch_override_wins() {
        let mut base = json!({"v": {"nested": true}});
        merge_value(&mut base, &json!({"v": "now-a-string"}));
        assert_eq!(base["v"], "now-a-string");
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "camelCase")]
    struct Sample {
        url: String,
        timeout: u64,
        nested: Nested,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "camelCase")]
    struct Nested {
        max_len: u32,
        keep: String,
    }

    fn sample_default() -> Sample {
        Sample {
            url: "http://default:6333".into(),
            timeout: 30,
            nested: Nested {
                max_len: 15,
                keep: "default".into(),
            },
        }
    }

    #[test]
    fn merge_over_defaults_overrides_present_keeps_absent() {
        // Override only url and nested.maxLen; timeout + nested.keep stay default.
        let ovr = json!({"url": "http://custom:6334", "nested": {"maxLen": 99}});
        let merged = merge_over_defaults(sample_default(), &ovr).expect("merge");
        assert_eq!(merged.url, "http://custom:6334");
        assert_eq!(merged.timeout, 30);
        assert_eq!(merged.nested.max_len, 99);
        assert_eq!(merged.nested.keep, "default");
    }

    #[test]
    fn merge_over_defaults_empty_override_is_identity() {
        let merged = merge_over_defaults(sample_default(), &json!({})).expect("merge");
        assert_eq!(merged, sample_default());
    }
}

//! YAML config merge logic.
//!
//! Mirrors `mergeConfigs()` from `src/typescript/mcp-server/src/config.ts`.
//!
//! Semantics:
//! - Each top-level section (database, qdrant, daemon, …) is a **shallow merge**:
//!   override fields win, missing fields fall back to the base value.
//! - Arrays **replace entirely** (not appended).  This matches the TS spread
//!   (`{ ...base.watching, ...override.watching }`).
//! - The optional `rules` block uses the same shallow merge for its fields,
//!   plus a nested shallow merge for `rules.limits`.

use anyhow::{Context, Result};
use serde_json::Value;

use crate::config::types::{
    CollectionsConfig, DaemonConfig, DatabaseConfig, EnvironmentConfig, QdrantConfig, RuleConfig,
    RuleLimitsConfig, ServerConfig, WatchingConfig,
};

/// Parse a YAML string into a partial `ServerConfig` represented as
/// `serde_json::Value` for flexible merging.
pub fn parse_yaml_partial(yaml_content: &str) -> Result<Value> {
    let value: Value =
        serde_yaml_ng::from_str(yaml_content).context("failed to parse YAML config")?;
    Ok(value)
}

/// Merge a parsed YAML `Value` (override) onto top of a base `ServerConfig`.
///
/// Only fields present in the override are changed; absent fields keep the
/// base value.
pub fn merge_yaml_over_defaults(base: ServerConfig, override_yaml: &Value) -> ServerConfig {
    let obj = match override_yaml.as_object() {
        Some(o) => o,
        None => return base, // non-object YAML → ignore
    };

    ServerConfig {
        database: merge_database(base.database, obj.get("database")),
        qdrant: merge_qdrant(base.qdrant, obj.get("qdrant")),
        daemon: merge_daemon(base.daemon, obj.get("daemon")),
        watching: merge_watching(base.watching, obj.get("watching")),
        collections: merge_collections(base.collections, obj.get("collections")),
        environment: merge_environment(base.environment, obj.get("environment")),
        rules: merge_rules(base.rules, obj.get("rules")),
    }
}

// ---------------------------------------------------------------------------
// Per-section mergers
// ---------------------------------------------------------------------------

fn merge_database(base: DatabaseConfig, ovr: Option<&Value>) -> DatabaseConfig {
    let Some(obj) = ovr.and_then(Value::as_object) else {
        return base;
    };
    DatabaseConfig {
        path: string_field(obj, "path").unwrap_or(base.path),
    }
}

fn merge_qdrant(base: QdrantConfig, ovr: Option<&Value>) -> QdrantConfig {
    let Some(obj) = ovr.and_then(Value::as_object) else {
        return base;
    };
    QdrantConfig {
        url: string_field(obj, "url").unwrap_or(base.url),
        api_key: string_field(obj, "apiKey").or(base.api_key),
        timeout: u64_field(obj, "timeout").unwrap_or(base.timeout),
    }
}

fn merge_daemon(base: DaemonConfig, ovr: Option<&Value>) -> DaemonConfig {
    let Some(obj) = ovr.and_then(Value::as_object) else {
        return base;
    };
    DaemonConfig {
        grpc_host: string_field(obj, "grpcHost").unwrap_or(base.grpc_host),
        grpc_port: u16_field(obj, "grpcPort").unwrap_or(base.grpc_port),
        queue_poll_interval_ms: u64_field(obj, "queuePollIntervalMs")
            .unwrap_or(base.queue_poll_interval_ms),
        queue_batch_size: u32_field(obj, "queueBatchSize").unwrap_or(base.queue_batch_size),
    }
}

fn merge_watching(base: WatchingConfig, ovr: Option<&Value>) -> WatchingConfig {
    let Some(obj) = ovr.and_then(Value::as_object) else {
        return base;
    };
    // Arrays replace entirely (TS spread semantics).
    WatchingConfig {
        patterns: string_array_field(obj, "patterns").unwrap_or(base.patterns),
        ignore_patterns: string_array_field(obj, "ignorePatterns").unwrap_or(base.ignore_patterns),
    }
}

fn merge_collections(base: CollectionsConfig, ovr: Option<&Value>) -> CollectionsConfig {
    let Some(obj) = ovr.and_then(Value::as_object) else {
        return base;
    };
    CollectionsConfig {
        rules_collection_name: string_field(obj, "rulesCollectionName")
            .unwrap_or(base.rules_collection_name),
    }
}

fn merge_environment(base: EnvironmentConfig, ovr: Option<&Value>) -> EnvironmentConfig {
    let Some(obj) = ovr.and_then(Value::as_object) else {
        return base;
    };
    EnvironmentConfig {
        user_path: string_field(obj, "userPath").or(base.user_path),
    }
}

fn merge_rules(base: Option<RuleConfig>, ovr: Option<&Value>) -> Option<RuleConfig> {
    // If neither side has a rules block, keep None.
    if base.is_none() && ovr.is_none() {
        return None;
    }

    let base_rule = base.unwrap_or_default();
    let Some(obj) = ovr.and_then(Value::as_object) else {
        return Some(base_rule);
    };

    let limits = if let Some(lim_val) = obj.get("limits").and_then(Value::as_object) {
        RuleLimitsConfig {
            max_label_length: u32_field(lim_val, "maxLabelLength")
                .unwrap_or(base_rule.limits.max_label_length),
            max_title_length: u32_field(lim_val, "maxTitleLength")
                .unwrap_or(base_rule.limits.max_title_length),
            max_tag_length: u32_field(lim_val, "maxTagLength")
                .unwrap_or(base_rule.limits.max_tag_length),
            max_tags_per_rule: u32_field(lim_val, "maxTagsPerRule")
                .unwrap_or(base_rule.limits.max_tags_per_rule),
        }
    } else {
        base_rule.limits
    };

    let duplication_threshold = if let Some(v) = obj.get("duplicationThreshold") {
        v.as_f64().or(base_rule.duplication_threshold)
    } else {
        base_rule.duplication_threshold
    };

    Some(RuleConfig {
        limits,
        duplication_threshold,
    })
}

// ---------------------------------------------------------------------------
// Value extraction helpers
// ---------------------------------------------------------------------------

fn string_field(obj: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    obj.get(key)?.as_str().map(str::to_owned)
}

fn u64_field(obj: &serde_json::Map<String, Value>, key: &str) -> Option<u64> {
    obj.get(key)?.as_u64()
}

fn u32_field(obj: &serde_json::Map<String, Value>, key: &str) -> Option<u32> {
    obj.get(key)?.as_u64().map(|v| v as u32)
}

fn u16_field(obj: &serde_json::Map<String, Value>, key: &str) -> Option<u16> {
    obj.get(key)?.as_u64().map(|v| v as u16)
}

fn string_array_field(obj: &serde_json::Map<String, Value>, key: &str) -> Option<Vec<String>> {
    let arr = obj.get(key)?.as_array()?;
    Some(
        arr.iter()
            .filter_map(|v| v.as_str())
            .map(str::to_owned)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_YAML: &str = r#"
qdrant:
  url: "http://custom-qdrant:6333"
"#;

    const FULL_OVERRIDE_YAML: &str = r#"
database:
  path: "/custom/db.sqlite"
qdrant:
  url: "http://myqdrant:6333"
  apiKey: "secret-key"
  timeout: 5000
daemon:
  grpcHost: "myhost"
  grpcPort: 9999
  queuePollIntervalMs: 1000
  queueBatchSize: 20
watching:
  patterns:
    - "*.go"
  ignorePatterns:
    - ".custom/*"
collections:
  rulesCollectionName: "custom-rules"
environment:
  userPath: "/usr/local/bin"
rules:
  limits:
    maxLabelLength: 30
    maxTitleLength: 100
    maxTagLength: 40
    maxTagsPerRule: 10
  duplicationThreshold: 0.85
"#;

    #[test]
    fn minimal_yaml_overrides_only_qdrant_url() {
        let base = ServerConfig::default();
        let yaml_val = parse_yaml_partial(MINIMAL_YAML).unwrap();
        let merged = merge_yaml_over_defaults(base.clone(), &yaml_val);

        // Overridden
        assert_eq!(merged.qdrant.url, "http://custom-qdrant:6333");
        // Unchanged
        assert_eq!(merged.qdrant.timeout, base.qdrant.timeout);
        assert_eq!(merged.daemon.grpc_port, base.daemon.grpc_port);
        assert_eq!(merged.database.path, base.database.path);
    }

    #[test]
    fn full_override_yaml_sets_all_fields() {
        let base = ServerConfig::default();
        let yaml_val = parse_yaml_partial(FULL_OVERRIDE_YAML).unwrap();
        let merged = merge_yaml_over_defaults(base, &yaml_val);

        assert_eq!(merged.database.path, "/custom/db.sqlite");
        assert_eq!(merged.qdrant.url, "http://myqdrant:6333");
        assert_eq!(merged.qdrant.api_key, Some("secret-key".to_string()));
        assert_eq!(merged.qdrant.timeout, 5000);
        assert_eq!(merged.daemon.grpc_host, "myhost");
        assert_eq!(merged.daemon.grpc_port, 9999);
        assert_eq!(merged.daemon.queue_poll_interval_ms, 1000);
        assert_eq!(merged.daemon.queue_batch_size, 20);
        assert_eq!(merged.watching.patterns, vec!["*.go"]);
        assert_eq!(merged.watching.ignore_patterns, vec![".custom/*"]);
        assert_eq!(merged.collections.rules_collection_name, "custom-rules");
        assert_eq!(
            merged.environment.user_path,
            Some("/usr/local/bin".to_string())
        );
        let rules = merged.rules.unwrap();
        assert_eq!(rules.limits.max_label_length, 30);
        assert_eq!(rules.limits.max_title_length, 100);
        assert_eq!(rules.limits.max_tag_length, 40);
        assert_eq!(rules.limits.max_tags_per_rule, 10);
        assert_eq!(rules.duplication_threshold, Some(0.85));
    }

    #[test]
    fn array_replace_not_append() {
        // Override provides a single pattern — should replace defaults entirely,
        // not append to them.
        let yaml = r#"
watching:
  patterns:
    - "*.go"
"#;
        let base = ServerConfig::default();
        let default_ignore_count = base.watching.ignore_patterns.len();
        let yaml_val = parse_yaml_partial(yaml).unwrap();
        let merged = merge_yaml_over_defaults(base, &yaml_val);

        assert_eq!(merged.watching.patterns, vec!["*.go"]);
        // ignorePatterns not present in override → keeps default
        assert_eq!(merged.watching.ignore_patterns.len(), default_ignore_count);
    }

    #[test]
    fn rules_limits_partial_override() {
        let yaml = r#"
rules:
  limits:
    maxTitleLength: 80
"#;
        let base = ServerConfig::default();
        let yaml_val = parse_yaml_partial(yaml).unwrap();
        let merged = merge_yaml_over_defaults(base.clone(), &yaml_val);

        let rules = merged.rules.unwrap();
        // Overridden
        assert_eq!(rules.limits.max_title_length, 80);
        // Unchanged defaults
        assert_eq!(
            rules.limits.max_label_length,
            base.rules.unwrap().limits.max_label_length
        );
    }

    #[test]
    fn empty_yaml_object_leaves_defaults_intact() {
        let yaml = "{}";
        let base = ServerConfig::default();
        let yaml_val = parse_yaml_partial(yaml).unwrap();
        let merged = merge_yaml_over_defaults(base.clone(), &yaml_val);
        assert_eq!(merged.qdrant.url, base.qdrant.url);
        assert_eq!(merged.daemon.grpc_port, base.daemon.grpc_port);
    }

    #[test]
    fn invalid_yaml_returns_error() {
        let result = parse_yaml_partial(": invalid: yaml: {{{");
        assert!(result.is_err());
    }
}

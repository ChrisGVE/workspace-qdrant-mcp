//! Tests for the daemon config view, validation, and serde round-trips.

use super::*;
use crate::storage::TransportMode;
use wqm_common::yaml_defaults::{self, YamlConfig};

#[test]
fn daemon_endpoint_auth_token_never_serialized() {
    // WI-g1 / AC-g1.1: auth_token must not be written back (skip_serializing,
    // not skip_serializing_if which still emitted it when present).
    let cfg = DaemonEndpointConfig {
        auth_token: Some("secret-token-value".to_string()),
        ..DaemonEndpointConfig::default()
    };
    let json = serde_json::to_string(&cfg).expect("serialize");
    assert!(
        !json.contains("secret-token-value"),
        "auth_token leaked into serialized config: {json}"
    );
    // AC-g1.2: an operator-provided token still loads into memory.
    let mut value = serde_json::to_value(DaemonEndpointConfig::default()).unwrap();
    value["auth_token"] = serde_json::Value::String("loaded-token".to_string());
    let loaded: DaemonEndpointConfig = serde_json::from_value(value).expect("load");
    assert_eq!(loaded.auth_token.as_deref(), Some("loaded-token"));
}

#[test]
fn test_daemon_config_defaults() {
    let config = DaemonConfig::default();
    assert_eq!(config.queue_processor.batch_size, 10);
    assert_eq!(config.monitoring.check_interval_hours, 24);
    assert!(config.monitoring.enable_monitoring);
    assert!(config.git.enable_branch_detection);
    assert_eq!(config.git.cache_ttl_seconds, 5); // from YAML branch_scan_interval_seconds
                                                 // Embedding settings defaults
    assert_eq!(config.embedding.cache_max_entries, 1000);
    assert!(config.embedding.model_cache_dir.is_none());
}

#[test]
fn test_default_config_creation() {
    let config = DaemonConfig::default();
    assert_eq!(config.max_concurrent_tasks, Some(4));
    assert_eq!(config.chunk_size, 1000);
    assert_eq!(config.log_level, "info");
}

#[test]
fn test_daemon_config_includes_grammars() {
    let config = DaemonConfig::default();
    // YAML default: required is empty (grammars downloaded on first use)
    assert!(config.grammars.required.is_empty());
    assert!(config.grammars.auto_download);
}

#[test]
fn test_daemon_config_includes_resource_limits() {
    let config = DaemonConfig::default();
    assert_eq!(config.resource_limits.nice_level, 10);
    assert_eq!(
        config.resource_limits.max_concurrent_embeddings, 0,
        "default is 0 (auto-detect)"
    );
    assert_eq!(config.resource_limits.max_memory_percent, 70);
    assert_eq!(
        config.resource_limits.onnx_intra_threads, 0,
        "default is 0 (auto-detect)"
    );
}

#[test]
fn test_daemon_config_includes_startup() {
    let config = DaemonConfig::default();
    assert_eq!(config.startup.warmup_delay_secs, 5);
    assert_eq!(config.startup.warmup_window_secs, 30);
}

// ── DaemonEndpointConfig::validate() tests ──────────────────────────────

#[test]
fn test_daemon_endpoint_config_validate_default_ok() {
    let config = DaemonEndpointConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_daemon_endpoint_config_validate_rejects_empty_host() {
    let empty = DaemonEndpointConfig {
        host: "".to_string(),
        ..DaemonEndpointConfig::default()
    };
    assert!(empty.validate().is_err());

    let whitespace = DaemonEndpointConfig {
        host: "   ".to_string(),
        ..DaemonEndpointConfig::default()
    };
    assert!(whitespace.validate().is_err());
}

#[test]
fn test_daemon_endpoint_config_validate_rejects_zero_grpc_port() {
    let config = DaemonEndpointConfig {
        grpc_port: 0,
        ..DaemonEndpointConfig::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("grpc_port"));
}

#[test]
fn test_daemon_endpoint_config_validate_rejects_bad_health_endpoint() {
    let config = DaemonEndpointConfig {
        health_endpoint: "health".to_string(),
        ..DaemonEndpointConfig::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("health_endpoint"));
}

#[test]
fn test_daemon_endpoint_config_validate_accepts_empty_health_endpoint() {
    let config = DaemonEndpointConfig {
        health_endpoint: "".to_string(),
        ..DaemonEndpointConfig::default()
    };
    assert!(config.validate().is_ok());
}

// ── DaemonConfig::validate() tests ──────────────────────────────────────

#[test]
fn test_daemon_config_validate_default_ok() {
    assert!(DaemonConfig::default().validate().is_ok());
}

#[test]
fn test_daemon_config_validate_propagates_queue_error() {
    let mut config = DaemonConfig::default();
    config.queue_processor.batch_size = 0;
    let result = config.validate();
    assert!(result.is_err());
    let msg = result.unwrap_err();
    assert!(
        msg.contains("queue_processor:"),
        "expected 'queue_processor:' in '{msg}'"
    );
}

#[test]
fn test_daemon_config_validate_propagates_observability_error() {
    let mut config = DaemonConfig::default();
    config.observability.collection_interval = 0;
    let result = config.validate();
    assert!(result.is_err());
    let msg = result.unwrap_err();
    assert!(
        msg.contains("observability:"),
        "expected 'observability:' in '{msg}'"
    );
}

#[test]
fn test_daemon_config_validate_short_circuits_on_first_error() {
    let mut config = DaemonConfig::default();
    // queue_processor is first in the section chain
    config.queue_processor.batch_size = 0;
    config.observability.collection_interval = 0;
    let result = config.validate();
    assert!(result.is_err());
    let msg = result.unwrap_err();
    assert!(
        msg.contains("queue_processor:"),
        "expected 'queue_processor:' in '{msg}'"
    );
}

// ── Top-level field validation (WI-a2: consolidated from the loader) ─────

#[test]
fn test_validate_rejects_zero_chunk_size() {
    let mut config = DaemonConfig::default();
    config.chunk_size = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_rejects_zero_max_concurrent_tasks() {
    let mut config = DaemonConfig::default();
    config.max_concurrent_tasks = Some(0);
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_rejects_invalid_log_level() {
    let mut config = DaemonConfig::default();
    config.log_level = "invalid".to_string();
    assert!(config.validate().is_err());
}

#[test]
fn test_validate_rejects_schemeless_qdrant_url() {
    let mut config = DaemonConfig::default();
    config.qdrant.url = "localhost:6333".to_string();
    let err = config.validate().expect_err("schemeless URL must fail");
    assert!(
        err.contains("qdrant.url"),
        "expected 'qdrant.url' in '{err}'"
    );
}

// ── Config-from-YAML drift regression tests ─────────────────────────────

#[test]
fn partial_yaml_fills_defaults() {
    let partial = r#"
qdrant:
  url: "http://custom-host:6333"
"#;
    let yaml: YamlConfig = serde_yaml_ng::from_str(partial).expect("partial YAML must parse");
    let config = DaemonConfig::from(&yaml);
    assert_eq!(config.qdrant.url, "http://custom-host:6333");
    assert_eq!(
        config.queue_processor.batch_size, 10,
        "unspecified sections must use defaults"
    );
    assert!(config.validate().is_ok());
}

#[test]
fn transport_mode_deserializes_lowercase_alias() {
    let grpc: TransportMode = serde_yaml_ng::from_str("grpc").expect("lowercase grpc must parse");
    assert!(matches!(grpc, TransportMode::Grpc));
    let http: TransportMode = serde_yaml_ng::from_str("http").expect("lowercase http must parse");
    assert!(matches!(http, TransportMode::Http));
}

#[test]
fn daemon_config_yaml_round_trip() {
    // F-051: DaemonConfig must survive a full YAML round-trip without losing
    // representative fields from every top-level section.
    let original = DaemonConfig::default();
    let yaml = serde_yaml_ng::to_string(&original).expect("serialise to YAML");
    let restored: DaemonConfig = serde_yaml_ng::from_str(&yaml).expect("deserialise back");

    assert_eq!(restored.chunk_size, original.chunk_size, "chunk_size");
    assert_eq!(
        restored.max_concurrent_tasks, original.max_concurrent_tasks,
        "max_concurrent_tasks"
    );
    assert_eq!(restored.log_level, original.log_level, "log_level");
    assert_eq!(
        restored.enable_preemption, original.enable_preemption,
        "enable_preemption"
    );
    assert_eq!(
        restored.embedding.cache_max_entries, original.embedding.cache_max_entries,
        "embedding.cache_max_entries"
    );
    assert_eq!(
        restored.daemon_endpoint.grpc_port, original.daemon_endpoint.grpc_port,
        "daemon_endpoint.grpc_port"
    );
    assert_eq!(
        restored.resource_limits.nice_level, original.resource_limits.nice_level,
        "resource_limits.nice_level"
    );
    assert_eq!(
        restored.url_ingestion.max_body_bytes, original.url_ingestion.max_body_bytes,
        "url_ingestion.max_body_bytes"
    );
}

#[test]
fn default_yaml_const_matches_default_config() {
    // The DEFAULT_YAML string and the in-memory default must agree.
    let yaml: YamlConfig =
        serde_yaml_ng::from_str(yaml_defaults::DEFAULT_YAML).expect("DEFAULT_YAML parses");
    let from_yaml = DaemonConfig::from(&yaml);
    let from_default = DaemonConfig::default();
    assert_eq!(
        from_yaml.queue_processor.batch_size,
        from_default.queue_processor.batch_size
    );
    assert_eq!(
        from_yaml.embedding.cache_max_entries,
        from_default.embedding.cache_max_entries
    );
}

// ── WI-g2: secret Debug/log redaction ──────────────────────────────────

#[test]
fn daemon_endpoint_auth_token_redacted_in_debug() {
    // AC-g2.1: auth_token must never appear in Debug output (plain or alternate).
    let cfg = DaemonEndpointConfig {
        auth_token: Some("tok-debug-secret".to_string()),
        ..DaemonEndpointConfig::default()
    };
    let plain = format!("{cfg:?}");
    let alt = format!("{cfg:#?}");
    assert!(
        !plain.contains("tok-debug-secret"),
        "auth_token leaked into {{:?}}: {plain}"
    );
    assert!(
        !alt.contains("tok-debug-secret"),
        "auth_token leaked into {{:#?}}: {alt}"
    );
    assert!(plain.contains("[REDACTED]"), "expected redaction marker");
    let none_dbg = format!("{:?}", DaemonEndpointConfig::default());
    assert!(none_dbg.contains("auth_token: None"), "got: {none_dbg}");
}

#[test]
fn daemon_config_debug_redacts_nested_secrets() {
    // AC-g2.1: DaemonConfig's derived Debug recurses into the manual Debug
    // impls of StorageConfig (qdrant.api_key) and DaemonEndpointConfig
    // (daemon_endpoint.auth_token) — neither secret may surface.
    let mut cfg = DaemonConfig::default();
    cfg.qdrant.api_key = Some("sk-nested-secret".to_string());
    cfg.daemon_endpoint.auth_token = Some("tok-nested-secret".to_string());
    for rendered in [format!("{cfg:?}"), format!("{cfg:#?}")] {
        assert!(
            !rendered.contains("sk-nested-secret"),
            "qdrant.api_key leaked: {rendered}"
        );
        assert!(
            !rendered.contains("tok-nested-secret"),
            "daemon_endpoint.auth_token leaked: {rendered}"
        );
    }
}

#[test]
#[tracing_test::traced_test]
fn loader_log_lines_never_contain_secret_values() {
    // AC-g2.2: a loader that debug-prints the resolved config (the realistic
    // leak vector) must emit no log line containing a secret VALUE.
    let mut cfg = DaemonConfig::default();
    cfg.qdrant.api_key = Some("sk-loader-secret".to_string());
    cfg.daemon_endpoint.auth_token = Some("tok-loader-secret".to_string());
    tracing::info!("Configuration loaded: {:?}", cfg);
    tracing::debug!("Configuration loaded (pretty): {:#?}", cfg);
    assert!(
        !logs_contain("sk-loader-secret"),
        "qdrant.api_key value leaked into a log line"
    );
    assert!(
        !logs_contain("tok-loader-secret"),
        "daemon_endpoint.auth_token value leaked into a log line"
    );
}

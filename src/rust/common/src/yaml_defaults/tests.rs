use super::*;

#[test]
fn test_default_yaml_parses() {
    let config: YamlConfig = serde_yaml_ng::from_str(DEFAULT_YAML).expect("YAML should parse");
    // Spot-check key values
    assert_eq!(config.qdrant.url, "http://localhost:6333");
    assert_eq!(config.grpc.port, 50051);
    assert_eq!(config.performance.chunk_size, 1000);
    assert_eq!(config.performance.max_concurrent_tasks, 4);
    assert!(config.performance.enable_preemption);
}

#[test]
fn test_lazy_lock_config() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert_eq!(config.qdrant.url, "http://localhost:6333");
    assert_eq!(config.grpc.port, 50051);
    assert_eq!(config.grpc.host, "127.0.0.1");
    assert!(config.grpc.enabled);
}

#[test]
fn test_qdrant_timeout_parsed() {
    let config = &*DEFAULT_YAML_CONFIG;
    // YAML says "30s" which should parse to 30000ms
    assert_eq!(config.qdrant.timeout, 30_000);
}

#[test]
fn test_qdrant_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert!(config.qdrant.prefer_grpc);
    assert_eq!(config.qdrant.transport, "grpc");
    assert_eq!(config.qdrant.pool.max_connections, 10);
    assert_eq!(config.qdrant.default_collection.vector_size, 384);
    assert_eq!(config.qdrant.default_collection.hnsw.m, 16);
    assert_eq!(config.qdrant.default_collection.hnsw.ef_construct, 100);
}

#[test]
fn test_grpc_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert_eq!(config.grpc.host, "127.0.0.1");
    assert_eq!(config.grpc.port, 50051);
    assert!(config.grpc.enabled);
    assert!(config.grpc.fallback_to_direct);
    assert_eq!(config.grpc.max_retries, 3);
}

#[test]
fn test_auto_ingestion_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert!(config.auto_ingestion.enabled);
    assert!(config.auto_ingestion.auto_create_watches);
    assert_eq!(config.auto_ingestion.max_files_per_batch, 5);
    assert_eq!(config.auto_ingestion.debounce_seconds(), 10);
}

#[test]
fn test_queue_processor_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert_eq!(config.queue_processor.batch_size, 10);
    assert_eq!(config.queue_processor.poll_interval_ms, 500);
    assert_eq!(config.queue_processor.max_retries, 5);
    assert_eq!(config.queue_processor.target_throughput, 1000);
    assert!(config.queue_processor.enable_metrics);
    assert_eq!(config.queue_processor.worker_count, 4);
    assert_eq!(config.queue_processor.backpressure_threshold, 1000);
}

#[test]
fn test_git_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert!(config.git.track_branch_lifecycle);
    assert!(config.git.auto_delete_branch_documents);
    assert_eq!(config.git.branch_scan_interval_seconds, 5);
}

#[test]
fn test_embedding_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert_eq!(config.embedding.model, "text-embedding-3-small");
    assert!(config.embedding.enable_sparse_vectors);
    assert_eq!(config.embedding.cache_max_entries, 1000);
}

#[test]
fn test_lsp_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert_eq!(config.lsp.max_servers_per_project, 3);
    assert!(config.lsp.auto_start_on_activation);
    assert_eq!(config.lsp.deactivation_delay_secs, 60);
    assert_eq!(config.lsp.cache_ttl_secs, 300);
    assert_eq!(config.lsp.startup_timeout_secs, 30);
    assert_eq!(config.lsp.request_timeout_secs, 10);
    assert_eq!(config.lsp.max_restart_attempts, 3);
}

#[test]
fn test_grammars_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert_eq!(
        config.grammars.cache_dir,
        "~/.cache/workspace-qdrant/grammars"
    );
    assert!(
        config.grammars.required.is_empty(),
        "Default required should be empty"
    );
    assert!(config.grammars.auto_download);
    assert_eq!(
        config.grammars.tree_sitter_version,
        env!("TREE_SITTER_VERSION_MAJOR_MINOR")
    );
}

#[test]
fn test_updates_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert!(config.updates.auto_check);
    assert_eq!(config.updates.channel, "stable");
    assert!(config.updates.notify_only);
    assert_eq!(config.updates.check_interval_hours, 24);
}

#[test]
fn test_performance_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert_eq!(config.performance.max_concurrent_tasks, 4);
    assert_eq!(config.performance.default_timeout_ms(), 30_000);
    assert!(config.performance.enable_preemption);
    assert_eq!(config.performance.chunk_size, 1000);
}

#[test]
fn test_observability_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert_eq!(config.observability.collection_interval_secs(), 60);
    assert!(!config.observability.metrics.enabled);
    assert!(!config.observability.telemetry.enabled);
    assert_eq!(config.observability.telemetry.history_retention, 120);
}

#[test]
fn test_telemetry_export_defaults_parse_from_yaml() {
    let t = &DEFAULT_YAML_CONFIG.observability.telemetry;
    assert_eq!(t.service_name, "memexd");
    assert!(!t.prometheus.enabled);
    assert_eq!(t.prometheus.port, 6337);
    assert_eq!(t.prometheus.bind, "0.0.0.0");
    assert!(!t.otlp.enabled);
    assert!(!t.otlp.metrics_enabled);
    assert_eq!(t.otlp.endpoint, "http://localhost:4318");
    assert_eq!(t.otlp.protocol, "http/protobuf");
    assert!((t.otlp.sample_rate - 1.0).abs() < f64::EPSILON);
    assert!(t.otlp.headers.is_empty());
}

#[test]
fn test_parse_duration_to_ms() {
    assert_eq!(parse_duration_to_ms("30s"), Some(30_000));
    assert_eq!(parse_duration_to_ms("5m"), Some(300_000));
    assert_eq!(parse_duration_to_ms("1h"), Some(3_600_000));
    assert_eq!(parse_duration_to_ms("500ms"), Some(500));
    assert_eq!(parse_duration_to_ms("10"), Some(10_000)); // bare number = seconds
    assert_eq!(parse_duration_to_ms(""), None);
}

#[test]
fn test_parse_size_to_bytes() {
    assert_eq!(parse_size_to_bytes("50MB"), Some(50 * 1_048_576));
    assert_eq!(parse_size_to_bytes("100KB"), Some(100 * 1024));
    assert_eq!(parse_size_to_bytes("1GB"), Some(1_073_741_824));
    assert_eq!(parse_size_to_bytes("1024B"), Some(1024));
    assert_eq!(parse_size_to_bytes(""), None);
}

#[test]
fn test_resource_limits_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert_eq!(config.resource_limits.nice_level, 10);
    assert_eq!(config.resource_limits.max_concurrent_embeddings, 0);
    assert_eq!(config.resource_limits.max_memory_percent, 70);
    assert_eq!(config.resource_limits.onnx_intra_threads, 0);
    assert_eq!(config.resource_limits.idle_threshold_secs, 120);
    assert_eq!(config.resource_limits.idle_confirmation_secs, 300);
    assert_eq!(config.resource_limits.ramp_up_step_secs, 120);
    assert_eq!(config.resource_limits.ramp_down_step_secs, 300);
    assert_eq!(config.resource_limits.burst_hold_secs, 600);
    assert!((config.resource_limits.burst_concurrency_multiplier - 2.0).abs() < f64::EPSILON);
    assert!((config.resource_limits.cpu_pressure_threshold - 0.6).abs() < f64::EPSILON);
    assert_eq!(config.resource_limits.idle_poll_interval_secs, 5);
}

#[test]
fn test_grammar_download_url_is_full_template() {
    // Verify YAML defaults contain the complete URL template with all placeholders,
    // not just a base URL prefix. This prevents grammar downloads from producing
    // incomplete artifact URLs.
    let defaults = YamlGrammarsConfig::default();
    assert!(
        defaults.download_base_url.contains("{language}"),
        "Missing {{language}} placeholder"
    );
    assert!(
        defaults.download_base_url.contains("{version}"),
        "Missing {{version}} placeholder"
    );
    assert!(
        defaults.download_base_url.contains("{platform}"),
        "Missing {{platform}} placeholder"
    );
    assert!(
        defaults.download_base_url.contains("{ext}"),
        "Missing {{ext}} placeholder"
    );
}

#[test]
fn test_tagging_tier3_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert!(!config.tagging.tier3.enabled);
    assert_eq!(config.tagging.tier3.primary.provider, "anthropic");
    assert_eq!(config.tagging.tier3.primary.access_mode, "cli");
    assert_eq!(
        config.tagging.tier3.primary.model,
        "claude-haiku-4-5-20251001"
    );
    assert_eq!(
        config.tagging.tier3.primary.api_key_env,
        "ANTHROPIC_API_KEY"
    );
    assert!(config.tagging.tier3.primary.base_url.is_none());
    assert!(config.tagging.tier3.fallback.is_none());
    assert_eq!(config.tagging.tier3.max_chunks_per_doc, 10);
    assert_eq!(config.tagging.tier3.max_tags_per_chunk, 5);
    assert_eq!(config.tagging.tier3.timeout_secs, 15);
    assert_eq!(config.tagging.tier3.max_retries, 2);
    assert_eq!(config.tagging.tier3.rate_limit_rps, 10);
    assert!((config.tagging.tier3.temperature - 0.3).abs() < 1e-6);
    assert_eq!(config.tagging.tier3.total_budget_secs, 60);
    assert_eq!(config.tagging.tier3.max_consecutive_failures, 2);
}

/// Round-trip: parse a minimal YAML snippet into YamlConfig twice and confirm
/// both parses yield identical key values. Validates that serde deserialization
/// is stable and deterministic across repeated calls.
#[test]
fn test_yaml_config_round_trip() {
    let snippet = r#"
qdrant:
  url: "https://example.qdrant.io:6334"
  timeout: "10s"
grpc:
  port: 12345
queue_processor:
  batch_size: 99
embedding:
  model: "my-custom-model"
grammars:
  cache_dir: "~/.cache/workspace-qdrant/grammars"
"#;

    let first: YamlConfig = serde_yaml_ng::from_str(snippet).expect("first parse must succeed");
    let second: YamlConfig = serde_yaml_ng::from_str(snippet).expect("second parse must succeed");

    assert_eq!(first.qdrant.url, "https://example.qdrant.io:6334");
    assert_eq!(first.grpc.port, 12345);
    assert_eq!(first.queue_processor.batch_size, 99);
    assert_eq!(first.embedding.model, "my-custom-model");
    assert_eq!(
        first.grammars.cache_dir,
        "~/.cache/workspace-qdrant/grammars"
    );
    assert_eq!(first.qdrant.url, second.qdrant.url);
    assert_eq!(first.grpc.port, second.grpc.port);
    assert_eq!(
        first.queue_processor.batch_size,
        second.queue_processor.batch_size
    );
}

/// Empty YAML `{}` must parse into YamlConfig with all defaults.
/// This is the critical partial-config scenario: a user config file
/// that specifies nothing must still produce a valid config.
#[test]
fn test_empty_yaml_parses_to_defaults() {
    let config: YamlConfig = serde_yaml_ng::from_str("{}").expect("empty YAML must parse");
    let defaults = &*DEFAULT_YAML_CONFIG;
    assert_eq!(config.qdrant.url, "http://localhost:6333");
    assert_eq!(config.grpc.port, defaults.grpc.port);
    assert_eq!(
        config.performance.chunk_size,
        defaults.performance.chunk_size
    );
    assert!(config.mounts.is_empty());
}

/// A YAML with only one section must still produce valid defaults for all others.
#[test]
fn test_partial_yaml_fills_missing_sections() {
    let partial = r#"
qdrant:
  url: "http://custom:6333"
"#;
    let config: YamlConfig = serde_yaml_ng::from_str(partial).expect("partial YAML must parse");
    assert_eq!(config.qdrant.url, "http://custom:6333");
    assert_eq!(config.grpc.port, 50051);
    assert_eq!(config.performance.max_concurrent_tasks, 4);
    assert!(config.performance.enable_preemption);
}

/// Malformed YAML must return an Err — not panic or produce a default value.
#[test]
fn test_malformed_yaml_returns_err() {
    let bad_yaml = "qdrant: {url: [unclosed";
    let result: Result<YamlConfig, _> = serde_yaml_ng::from_str(bad_yaml);
    assert!(result.is_err(), "Malformed YAML should return Err, not Ok");
}

/// Completely invalid (non-YAML) input must also return an Err.
#[test]
fn test_invalid_input_returns_err() {
    let not_yaml = "\x00\x01\x02binary garbage \u{FFFD}";
    let result: Result<YamlConfig, _> = serde_yaml_ng::from_str(not_yaml);
    assert!(result.is_err(), "Non-YAML input should return Err, not Ok");
}

/// The grammars cache_dir default must use the XDG cache path, not the legacy
/// ~/.workspace-qdrant root directory.
#[test]
fn test_grammars_cache_dir_is_xdg() {
    let config: YamlConfig = serde_yaml_ng::from_str(DEFAULT_YAML).expect("YAML should parse");
    assert!(
        config.grammars.cache_dir.contains(".cache"),
        "cache_dir must be under ~/.cache (XDG): {}",
        config.grammars.cache_dir
    );
    assert!(
        !config.grammars.cache_dir.starts_with("~/.workspace-qdrant"),
        "cache_dir must not use legacy root: {}",
        config.grammars.cache_dir
    );
}

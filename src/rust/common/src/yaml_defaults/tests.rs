use super::*;

#[test]
fn test_default_yaml_parses() {
    let config: YamlConfig =
        serde_yml::from_str(DEFAULT_YAML).expect("YAML should parse");
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
    assert_eq!(config.embedding.model, "sentence-transformers/all-MiniLM-L6-v2");
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
    assert_eq!(config.grammars.cache_dir, "~/.workspace-qdrant/grammars");
    assert!(config.grammars.required.contains(&"rust".to_string()));
    assert!(config.grammars.required.contains(&"python".to_string()));
    assert!(config.grammars.auto_download);
    assert_eq!(config.grammars.tree_sitter_version, "0.24");
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
    assert_eq!(config.resource_limits.inter_item_delay_ms, 50);
    assert_eq!(config.resource_limits.max_concurrent_embeddings, 0);
    assert_eq!(config.resource_limits.max_memory_percent, 70);
    assert_eq!(config.resource_limits.onnx_intra_threads, 0);
    assert_eq!(config.resource_limits.idle_threshold_secs, 120);
    assert_eq!(config.resource_limits.idle_confirmation_secs, 300);
    assert_eq!(config.resource_limits.ramp_up_step_secs, 120);
    assert_eq!(config.resource_limits.ramp_down_step_secs, 300);
    assert_eq!(config.resource_limits.burst_hold_secs, 600);
    assert!((config.resource_limits.burst_concurrency_multiplier - 2.0).abs() < f64::EPSILON);
    assert_eq!(config.resource_limits.burst_inter_item_delay_ms, 0);
    assert!((config.resource_limits.cpu_pressure_threshold - 0.6).abs() < f64::EPSILON);
    assert_eq!(config.resource_limits.idle_poll_interval_secs, 5);
}

#[test]
fn test_grammar_download_url_is_full_template() {
    // Verify YAML defaults contain the complete URL template with all placeholders,
    // not just a base URL prefix. This prevents grammar downloads from producing
    // incomplete artifact URLs.
    let defaults = YamlGrammarsConfig::default();
    assert!(defaults.download_base_url.contains("{language}"), "Missing {{language}} placeholder");
    assert!(defaults.download_base_url.contains("{version}"), "Missing {{version}} placeholder");
    assert!(defaults.download_base_url.contains("{platform}"), "Missing {{platform}} placeholder");
    assert!(defaults.download_base_url.contains("{ext}"), "Missing {{ext}} placeholder");
}

#[test]
fn test_tagging_tier3_defaults() {
    let config = &*DEFAULT_YAML_CONFIG;
    assert!(!config.tagging.tier3.enabled);
    assert_eq!(config.tagging.tier3.primary.provider, "anthropic");
    assert_eq!(config.tagging.tier3.primary.access_mode, "cli");
    assert_eq!(config.tagging.tier3.primary.model, "claude-haiku-4-5-20251001");
    assert_eq!(config.tagging.tier3.primary.api_key_env, "ANTHROPIC_API_KEY");
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

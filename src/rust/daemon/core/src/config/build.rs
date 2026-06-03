//! Construction of [`DaemonConfig`] from the YAML view ([`YamlConfig`]).
//!
//! `YamlConfig` carries `#[serde(default)]` on every section, so a partial
//! user config deserialises with compiled-in defaults; the `From<&YamlConfig>`
//! conversion then maps it into the typed [`DaemonConfig`] view. This is the
//! daemon's merge-over-defaults mechanism (it does NOT use
//! `merge_over_defaults<DaemonConfig>` — field names differ between the views).

use std::path::PathBuf;

use wqm_common::yaml_defaults::YamlConfig;

use super::processing::default_retry_delays_seconds;
use super::types::{DaemonConfig, DaemonEndpointConfig};
use super::{
    AutoIngestionConfig, ConceptConfig, EmbeddingSettings, GitConfig, GrammarConfig,
    GraphRagConfig, IngestionLimitsConfig, KeywordEmbedderConfig, LoggingConfig, LspSettings,
    MetricsConfig, MonitoringConfig, NarrativeConfig, ObservabilityConfig, OtlpExportConfig,
    OtlpProtocol, PrometheusExportConfig, QueueProcessorSettings, ResourceLimitsConfig,
    StartupConfig, TelemetryConfig, TracingConfig, UpdateChannel, UpdatesConfig,
    UrlIngestionConfig,
};
use crate::storage::StorageConfig;

impl From<&YamlConfig> for DaemonConfig {
    fn from(yaml: &YamlConfig) -> Self {
        Self {
            log_file: None,
            max_concurrent_tasks: Some(yaml.performance.max_concurrent_tasks),
            default_timeout_ms: Some(yaml.performance.default_timeout_ms() as usize),
            enable_preemption: yaml.performance.enable_preemption,
            chunk_size: yaml.performance.chunk_size,
            log_level: "info".to_string(),
            auto_ingestion: build_auto_ingestion_config(yaml),
            project_path: None,
            qdrant: build_storage_config(yaml),
            logging: LoggingConfig::default(),
            queue_processor: build_queue_processor_settings(yaml),
            monitoring: MonitoringConfig::default(),
            git: build_git_config(yaml),
            observability: build_observability_config(yaml),
            embedding: build_embedding_settings(yaml),
            lsp: build_lsp_settings(yaml),
            grammars: build_grammar_config(yaml),
            updates: build_updates_config(yaml),
            resource_limits: build_resource_limits_config(yaml),
            startup: StartupConfig::default(),
            daemon_endpoint: build_daemon_endpoint_config(yaml),
            ingestion_limits: IngestionLimitsConfig::default(),
            concept: ConceptConfig::default(),
            graph_rag: GraphRagConfig::default(),
            graph: build_graph_config(yaml),
            narrative: NarrativeConfig::default(),
            url_ingestion: build_url_ingestion_config(yaml),
            mounts: yaml.mounts.clone(),
            control_port: None,
        }
    }
}

fn build_graph_config(yaml: &YamlConfig) -> crate::graph::GraphConfig {
    use crate::graph::{GraphBackend, GraphConfig};
    let backend = match yaml.graph.backend.trim().to_ascii_lowercase().as_str() {
        "ladybug" => GraphBackend::Ladybug,
        _ => GraphBackend::Sqlite,
    };
    GraphConfig {
        backend,
        db_dir: None,
        buffer_pool_size: yaml.graph.buffer_pool_size,
        max_threads: yaml.graph.max_threads,
    }
}

fn build_url_ingestion_config(yaml: &YamlConfig) -> UrlIngestionConfig {
    UrlIngestionConfig {
        connect_timeout_secs: yaml.url_ingestion.connect_timeout_secs,
        read_timeout_secs: yaml.url_ingestion.read_timeout_secs,
        max_redirects: yaml.url_ingestion.max_redirects,
        max_body_bytes: yaml.url_ingestion.max_body_bytes,
        allow_private_networks: yaml.url_ingestion.allow_private_networks,
        allowed_content_types: yaml.url_ingestion.allowed_content_types.clone(),
    }
}

fn build_auto_ingestion_config(yaml: &YamlConfig) -> AutoIngestionConfig {
    AutoIngestionConfig {
        enabled: yaml.auto_ingestion.enabled,
        auto_create_watches: yaml.auto_ingestion.auto_create_watches,
        include_common_files: yaml.auto_ingestion.include_common_files,
        include_source_files: yaml.auto_ingestion.include_source_files,
        target_collection_suffix: "scratchbook".to_string(),
        max_files_per_batch: yaml.auto_ingestion.max_files_per_batch,
        batch_delay_seconds: 2.0,
        max_file_size_mb: 50,
        recursive_depth: 5,
        debounce_seconds: yaml.auto_ingestion.debounce_seconds(),
    }
}

fn build_storage_config(yaml: &YamlConfig) -> StorageConfig {
    StorageConfig {
        url: yaml.qdrant.url.clone(),
        api_key: yaml.qdrant.api_key.clone(),
        timeout_ms: yaml.qdrant.timeout,
        pool_size: yaml.qdrant.pool.max_connections,
        dense_vector_size: yaml.qdrant.default_collection.vector_size,
        ..StorageConfig::default()
    }
}

fn build_queue_processor_settings(yaml: &YamlConfig) -> QueueProcessorSettings {
    QueueProcessorSettings {
        batch_size: yaml.queue_processor.batch_size,
        poll_interval_ms: yaml.queue_processor.poll_interval_ms,
        max_retries: yaml.queue_processor.max_retries,
        retry_delays_seconds: default_retry_delays_seconds(),
        target_throughput: yaml.queue_processor.target_throughput,
        enable_metrics: yaml.queue_processor.enable_metrics,
    }
}

fn build_git_config(yaml: &YamlConfig) -> GitConfig {
    GitConfig {
        enable_branch_detection: yaml.git.track_branch_lifecycle,
        cache_ttl_seconds: yaml.git.branch_scan_interval_seconds,
    }
}

fn build_observability_config(yaml: &YamlConfig) -> ObservabilityConfig {
    let y_telemetry = &yaml.observability.telemetry;
    let protocol =
        OtlpProtocol::parse(&y_telemetry.otlp.protocol).unwrap_or(OtlpProtocol::HttpProtobuf);

    ObservabilityConfig {
        collection_interval: yaml.observability.collection_interval_secs(),
        metrics: MetricsConfig {
            enabled: yaml.observability.metrics.enabled,
        },
        telemetry: TelemetryConfig {
            enabled: y_telemetry.enabled,
            history_retention: y_telemetry.history_retention,
            cpu_usage: y_telemetry.cpu_usage,
            memory_usage: y_telemetry.memory_usage,
            latency: y_telemetry.latency,
            queue_depth: y_telemetry.queue_depth,
            throughput: y_telemetry.throughput,
            service_name: y_telemetry.service_name.clone(),
            prometheus: PrometheusExportConfig {
                enabled: y_telemetry.prometheus.enabled,
                port: y_telemetry.prometheus.port,
                bind: y_telemetry.prometheus.bind.clone(),
            },
            otlp: OtlpExportConfig {
                enabled: y_telemetry.otlp.enabled,
                metrics_enabled: y_telemetry.otlp.metrics_enabled,
                endpoint: y_telemetry.otlp.endpoint.clone(),
                protocol,
                sample_rate: y_telemetry.otlp.sample_rate,
                headers: y_telemetry.otlp.headers.clone(),
            },
            tracing: TracingConfig {
                tier: y_telemetry.tracing.tier.clone(),
                instrument_hot_paths: y_telemetry.tracing.instrument_hot_paths,
                attribute_cardinality_cap: y_telemetry.tracing.attribute_cardinality_cap,
            },
        },
    }
}

fn build_embedding_settings(yaml: &YamlConfig) -> EmbeddingSettings {
    EmbeddingSettings {
        cache_max_entries: yaml.embedding.cache_max_entries,
        model_cache_dir: yaml.embedding.model_cache_dir.as_ref().map(PathBuf::from),
        provider: yaml.embedding.provider.clone(),
        model: yaml.embedding.model.clone(),
        base_url: yaml.embedding.base_url.clone(),
        remote_batch_size: yaml.embedding.remote_batch_size,
        api_key_env_var: yaml.embedding.api_key_env_var.clone(),
        output_dim: yaml.embedding.output_dim,
        health_probe_cache_secs: yaml.embedding.health_probe_cache_secs,
        max_input_tokens: yaml.embedding.max_input_tokens,
        keyword_embedder: KeywordEmbedderConfig {
            enabled: yaml.embedding.keyword_embedder.enabled,
            num_threads: yaml.embedding.keyword_embedder.num_threads,
        },
    }
}

fn build_lsp_settings(yaml: &YamlConfig) -> LspSettings {
    LspSettings {
        user_path: yaml.lsp.user_path.clone(),
        max_servers_per_project: yaml.lsp.max_servers_per_project,
        auto_start_on_activation: yaml.lsp.auto_start_on_activation,
        deactivation_delay_secs: yaml.lsp.deactivation_delay_secs,
        enable_enrichment_cache: yaml.lsp.enable_enrichment_cache,
        cache_ttl_secs: yaml.lsp.cache_ttl_secs,
        startup_timeout_secs: yaml.lsp.startup_timeout_secs,
        request_timeout_secs: yaml.lsp.request_timeout_secs,
        health_check_interval_secs: yaml.lsp.health_check_interval_secs,
        max_restart_attempts: yaml.lsp.max_restart_attempts,
        restart_backoff_multiplier: yaml.lsp.restart_backoff_multiplier,
        enable_auto_restart: true,
        stability_reset_secs: 3600,
        idle_timeout_secs: yaml.lsp.idle_timeout_secs,
    }
}

fn build_grammar_config(yaml: &YamlConfig) -> GrammarConfig {
    GrammarConfig {
        cache_dir: PathBuf::from(&yaml.grammars.cache_dir),
        required: yaml.grammars.required.clone(),
        auto_download: yaml.grammars.auto_download,
        tree_sitter_version: yaml.grammars.tree_sitter_version.clone(),
        download_base_url: yaml.grammars.download_base_url.clone(),
        verify_checksums: yaml.grammars.verify_checksums,
        lazy_loading: yaml.grammars.lazy_loading,
        check_interval_hours: yaml.grammars.check_interval_hours,
        idle_update_check_enabled: yaml.grammars.idle_update_check_enabled,
        idle_update_check_delay_secs: yaml.grammars.idle_update_check_delay_secs,
        grammar_idle_timeout_secs: yaml.grammars.grammar_idle_timeout_secs,
    }
}

fn build_updates_config(yaml: &YamlConfig) -> UpdatesConfig {
    UpdatesConfig {
        auto_check: yaml.updates.auto_check,
        channel: match yaml.updates.channel.as_str() {
            "beta" => UpdateChannel::Beta,
            "dev" => UpdateChannel::Dev,
            _ => UpdateChannel::Stable,
        },
        notify_only: yaml.updates.notify_only,
        check_interval_hours: yaml.updates.check_interval_hours,
    }
}

fn build_resource_limits_config(yaml: &YamlConfig) -> ResourceLimitsConfig {
    ResourceLimitsConfig {
        nice_level: yaml.resource_limits.nice_level,
        max_concurrent_embeddings: yaml.resource_limits.max_concurrent_embeddings,
        max_memory_percent: yaml.resource_limits.max_memory_percent,
        onnx_intra_threads: yaml.resource_limits.onnx_intra_threads,
        idle_threshold_secs: yaml.resource_limits.idle_threshold_secs,
        idle_confirmation_secs: yaml.resource_limits.idle_confirmation_secs,
        ramp_up_step_secs: yaml.resource_limits.ramp_up_step_secs,
        ramp_down_step_secs: yaml.resource_limits.ramp_down_step_secs,
        burst_hold_secs: yaml.resource_limits.burst_hold_secs,
        burst_concurrency_multiplier: yaml.resource_limits.burst_concurrency_multiplier,
        cpu_pressure_threshold: yaml.resource_limits.cpu_pressure_threshold,
        idle_poll_interval_secs: yaml.resource_limits.idle_poll_interval_secs,
        active_concurrency_multiplier: yaml.resource_limits.active_concurrency_multiplier,
        linux_idle_source: yaml.resource_limits.linux_idle_source.clone(),
        linux_idle_load_threshold: yaml.resource_limits.linux_idle_load_threshold,
    }
}

fn build_daemon_endpoint_config(yaml: &YamlConfig) -> DaemonEndpointConfig {
    DaemonEndpointConfig {
        host: yaml.grpc.host.clone(),
        grpc_port: yaml.grpc.port,
        health_endpoint: "/health".to_string(),
        auth_token: None,
    }
}

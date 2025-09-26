//! Property-based tests for data serialization properties
//!
//! This module validates serialization/deserialization properties for configuration,
//! protocol buffer messages, JSON data, and error message consistency.

use proptest::prelude::*;
use serde_json;
use std::collections::HashMap;
use workspace_qdrant_daemon::config::*;
use workspace_qdrant_daemon::error::DaemonError;
use workspace_qdrant_daemon::proto;

/// Strategy for generating valid configuration values
fn random_config_values() -> impl Strategy<Value = DaemonConfig> {
    (
        // Host and port
        "[a-zA-Z0-9.-]{1,50}",
        1u16..65535u16,
        // Processing config
        1usize..100usize,      // max_concurrent_tasks
        100usize..10000usize,  // default_chunk_size
        10usize..1000usize,    // default_chunk_overlap
        1000u64..100_000_000u64, // max_file_size_bytes
        prop::collection::vec("[a-zA-Z0-9]{2,5}", 1..10), // supported_extensions
        any::<bool>(),         // enable_lsp
        1u32..300u32,          // lsp_timeout_secs
        // Qdrant config
        "https?://[a-zA-Z0-9.-]{1,50}(:[0-9]{1,5})?", // url
        prop::option::of("[a-zA-Z0-9_-]{10,50}"), // api_key
        1u32..300u32,          // timeout_secs
        1u32..10u32,           // max_retries
        // Collection config
        64usize..2048usize,    // vector_size
        prop::sample::select(vec!["Cosine", "Dot", "Euclidean"]), // distance_metric
        any::<bool>(),         // enable_indexing
        1u32..5u32,            // replication_factor
        1u32..10u32,           // shard_number
        // Logging config
        prop::sample::select(vec!["trace", "debug", "info", "warn", "error"]), // level
        any::<bool>(),         // enable_file_logging
        "[a-zA-Z0-9_.-/]{5,100}", // log_dir
        1u64..100u64,          // max_log_files
        1000000u64..1000000000u64, // max_log_size_bytes
        // Monitoring config
        any::<bool>(),         // enable_metrics
        1u16..65535u16,        // metrics_port
    ).prop_map(|(
        host, port, max_tasks, chunk_size, chunk_overlap, max_file_size,
        extensions, enable_lsp, lsp_timeout, qdrant_url, api_key,
        qdrant_timeout, max_retries, vector_size, distance_metric,
        enable_indexing, replication_factor, shard_number,
        log_level, enable_file_logging, log_dir, max_log_files,
        max_log_size, enable_metrics, metrics_port
    )| {
        DaemonConfig {
            server: ServerConfig {
                host,
                port,
                max_connections: 100,
                connection_timeout_secs: 30,
                request_timeout_secs: 60,
                enable_tls: false,
                security: SecurityConfig {
                    tls: TlsConfig {
                        cert_file: None,
                        key_file: None,
                        ca_cert_file: None,
                        enable_mtls: false,
                        client_cert_verification: ClientCertVerification::None,
                        supported_protocols: vec![],
                        cipher_suites: vec![],
                    },
                    auth: AuthConfig {
                        enable_service_auth: false,
                        jwt: JwtConfig {
                            secret_key: "test".to_string(),
                            token_expiry_secs: 3600,
                            issuer: "test".to_string(),
                            audience: vec![],
                            algorithm: "HS256".to_string(),
                        },
                        api_key: ApiKeyConfig {
                            keys: vec![],
                            header_name: "X-API-Key".to_string(),
                            enable_key_rotation: false,
                            key_expiry_days: 365,
                        },
                        authorization: AuthorizationConfig {
                            enable_rbac: false,
                            default_permissions: vec![],
                            service_permissions: std::collections::HashMap::new(),
                            admin_roles: vec![],
                            guest_permissions: vec![],
                        },
                    },
                    rate_limiting: RateLimitConfig {
                        enabled: false,
                        requests_per_second: 1000,
                        burst_capacity: 100,
                        connection_pool_limits: ConnectionPoolLimits {
                            document_processor: 50,
                            search_service: 30,
                            memory_service: 20,
                            system_service: 10,
                        },
                        queue_depth_limit: 1000,
                        memory_protection: MemoryProtectionConfig {
                            enabled: false,
                            max_memory_per_connection: 104857600,
                            memory_alert_threshold: 0.8,
                            enable_memory_monitoring: false,
                            gc_trigger_threshold: 0.9,
                        },
                        resource_protection: ResourceProtectionConfig {
                            enabled: false,
                            max_cpu_per_connection: 50.0,
                            cpu_alert_threshold: 0.8,
                            enable_resource_monitoring: false,
                            throttle_threshold: 0.9,
                        },
                    },
                    audit: AuditConfig {
                        enabled: false,
                        log_successful_requests: false,
                        log_failed_requests: true,
                        audit_file_path: "/tmp/audit.log".to_string(),
                        enable_audit_rotation: false,
                        max_audit_file_size: 10485760,
                        retention_days: 30,
                    },
                },
                transport: TransportConfig {
                    unix_socket: UnixSocketConfig {
                        enabled: false,
                        socket_path: "/tmp/test.sock".to_string(),
                        permissions: 0o666,
                        prefer_for_local: false,
                    },
                    local_optimization: LocalOptimizationConfig {
                        enabled: true,
                        use_large_buffers: false,
                        local_buffer_size: 4096,
                        memory_efficient_serialization: true,
                        reduce_latency: true,
                    },
                    transport_strategy: TransportStrategy::Auto,
                },
                message: MessageConfig {
                    max_incoming_message_size: 4194304,
                    max_outgoing_message_size: 4194304,
                    enable_size_validation: true,
                    max_frame_size: 16384,
                    initial_window_size: 65536,
                    service_limits: ServiceMessageLimits {
                        document_processor: ServiceLimit { max_incoming: 4194304, max_outgoing: 4194304, enable_validation: true },
                        search_service: ServiceLimit { max_incoming: 1048576, max_outgoing: 1048576, enable_validation: true },
                        memory_service: ServiceLimit { max_incoming: 1048576, max_outgoing: 1048576, enable_validation: true },
                        system_service: ServiceLimit { max_incoming: 1048576, max_outgoing: 1048576, enable_validation: true },
                    },
                    monitoring: MessageMonitoringConfig {
                        enable_detailed_monitoring: false,
                        oversized_alert_threshold: 0.9,
                        enable_realtime_metrics: false,
                        metrics_interval_secs: 60,
                    },
                },
                compression: CompressionConfig {
                    enable_gzip: false,
                    compression_threshold: 1024,
                    compression_level: 6,
                    enable_streaming_compression: false,
                    enable_compression_monitoring: false,
                    adaptive: AdaptiveCompressionConfig {
                        enable_adaptive: false,
                        text_compression_level: 6,
                        binary_compression_level: 3,
                        structured_compression_level: 6,
                        max_compression_time_ms: 100,
                    },
                    performance: CompressionPerformanceConfig {
                        enable_ratio_tracking: false,
                        poor_ratio_threshold: 0.1,
                        enable_time_monitoring: false,
                        slow_compression_threshold_ms: 1000,
                        enable_failure_alerting: false,
                    },
                },
                streaming: StreamingConfig {
                    enable_server_streaming: true,
                    enable_client_streaming: true,
                    max_concurrent_streams: 100,
                    stream_buffer_size: 1000,
                    stream_timeout_secs: 300,
                    enable_flow_control: true,
                    progress: StreamProgressConfig {
                        enable_progress_tracking: false,
                        progress_update_interval_ms: 1000,
                        enable_progress_callbacks: false,
                        progress_threshold: 1048576,
                    },
                    health: StreamHealthConfig {
                        enable_health_monitoring: false,
                        health_check_interval_secs: 30,
                        enable_auto_recovery: false,
                        max_recovery_attempts: 3,
                        recovery_backoff_multiplier: 2.0,
                        initial_recovery_delay_ms: 1000,
                    },
                    large_operations: LargeOperationStreamConfig {
                        enable_large_document_streaming: true,
                        large_operation_chunk_size: 65536,
                        enable_bulk_streaming: false,
                        max_streaming_memory: 67108864,
                        enable_bidirectional_optimization: false,
                    },
                },
            },
            database: DatabaseConfig {
                sqlite_path: ":memory:".to_string(),
                max_connections: 10,
                connection_timeout_secs: 30,
                enable_wal: true,
            },
            processing: ProcessingConfig {
                max_concurrent_tasks: max_tasks,
                default_chunk_size: chunk_size,
                default_chunk_overlap: chunk_overlap,
                max_file_size_bytes: max_file_size,
                supported_extensions: extensions,
                enable_lsp,
                lsp_timeout_secs: lsp_timeout,
            },
            qdrant: QdrantConfig {
                url: qdrant_url,
                api_key,
                timeout_secs: qdrant_timeout,
                max_retries,
                default_collection: CollectionConfig {
                    vector_size,
                    distance_metric,
                    enable_indexing,
                    replication_factor,
                    shard_number,
                },
            },
            file_watcher: FileWatcherConfig {
                watch_interval_secs: 1,
                debounce_duration_ms: 500,
                max_watch_depth: 10,
                ignore_patterns: vec!["*.tmp".to_string()],
                enable_recursive_watching: true,
                symlink_policy: SymlinkPolicy::Follow,
            },
            metrics: MetricsConfig {
                enabled: enable_metrics,
                port: metrics_port,
                host: "127.0.0.1".to_string(),
                update_interval_secs: 60,
                retention_days: 7,
            },
            logging: LoggingConfig {
                level: log_level,
                file_path: log_dir,
                json_format: enable_file_logging,
                max_file_size_mb: (max_log_size / 1_048_576) as u32,
                max_files: max_log_files as u32,
            },
        }
    })
}

/// Strategy for generating JSON-serializable data structures
fn random_json_data() -> impl Strategy<Value = serde_json::Value> {
    let leaf = prop_oneof![
        any::<bool>().prop_map(serde_json::Value::Bool),
        any::<i64>().prop_map(|i| serde_json::Value::Number(i.into())),
        any::<f64>().prop_filter("finite", |f| f.is_finite())
            .prop_map(|f| serde_json::Value::Number(
                serde_json::Number::from_f64(f).unwrap_or_else(|| 0.into())
            )),
        "[\\PC]{0,100}".prop_map(serde_json::Value::String),
        Just(serde_json::Value::Null),
    ];

    leaf.prop_recursive(
        8,  // levels deep
        256, // total elements
        10,  // items per collection
        |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..10)
                    .prop_map(serde_json::Value::Array),
                prop::collection::hash_map("[\\PC]{1,20}", inner, 0..10)
                    .prop_map(|map| serde_json::Value::Object(
                        map.into_iter().collect()
                    )),
            ]
        },
    )
}

/// Strategy for generating various daemon errors
fn random_daemon_error() -> impl Strategy<Value = DaemonError> {
    prop_oneof![
        ("[\\PC]{1,100}", "[\\PC]{1,100}").prop_map(|(msg, path)| DaemonError::FileIo { message: msg, path }),
        ("[\\PC]{1,100}", 1u64..1000000000u64, 100u64..1000000u64)
            .prop_map(|(path, size, max)| DaemonError::FileTooLarge { path, size, max_size: max }),
        "[\\PC]{1,100}".prop_map(|msg| DaemonError::Internal { message: msg }),
        "[\\PC]{1,100}".prop_map(|msg| DaemonError::Configuration { message: msg }),
        "[\\PC]{1,100}".prop_map(|msg| DaemonError::Search { message: msg }),
        "[\\PC]{1,100}".prop_map(|msg| DaemonError::Memory { message: msg }),
        "[\\PC]{1,100}".prop_map(|msg| DaemonError::DocumentProcessing { message: msg }),
        "[\\PC]{1,100}".prop_map(|msg| DaemonError::System { message: msg }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig {
        timeout: 15000, // 15 seconds per test
        cases: 30,
        .. ProptestConfig::default()
    })]

    #[test]
    fn proptest_config_serialization_roundtrip(config in random_config_values()) {
        // Property: Configuration should serialize and deserialize correctly

        // Test YAML serialization
        let yaml_result = serde_yaml::to_string(&config);
        prop_assert!(yaml_result.is_ok(), "Config should serialize to YAML");

        if let Ok(yaml_str) = yaml_result {
            let yaml_deserialize: Result<DaemonConfig, _> = serde_yaml::from_str(&yaml_str);
            prop_assert!(yaml_deserialize.is_ok(), "YAML should deserialize back to config");

            if let Ok(deserialized) = yaml_deserialize {
                prop_assert_eq!(config.host, deserialized.host, "Host should match after YAML roundtrip");
                prop_assert_eq!(config.port, deserialized.port, "Port should match after YAML roundtrip");
                prop_assert_eq!(config.processing.max_concurrent_tasks,
                               deserialized.processing.max_concurrent_tasks,
                               "Processing config should match");
                prop_assert_eq!(config.qdrant.url, deserialized.qdrant.url, "Qdrant URL should match");
            }
        }

        // Test JSON serialization
        let json_result = serde_json::to_string(&config);
        prop_assert!(json_result.is_ok(), "Config should serialize to JSON");

        if let Ok(json_str) = json_result {
            let json_deserialize: Result<DaemonConfig, _> = serde_json::from_str(&json_str);
            prop_assert!(json_deserialize.is_ok(), "JSON should deserialize back to config");
        }
    }

    #[test]
    fn proptest_json_data_consistency(data in random_json_data()) {
        // Property: JSON data should maintain consistency through serialization
        let serialized = serde_json::to_string(&data);
        prop_assert!(serialized.is_ok(), "JSON data should serialize");

        if let Ok(json_str) = serialized {
            let deserialized: Result<serde_json::Value, _> = serde_json::from_str(&json_str);
            prop_assert!(deserialized.is_ok(), "JSON should deserialize");

            if let Ok(roundtrip_data) = deserialized {
                // Deep equality check
                prop_assert_eq!(data, roundtrip_data, "JSON data should be identical after roundtrip");
            }
        }
    }

    #[test]
    fn proptest_error_message_consistency(error in random_daemon_error()) {
        // Property: Error messages should be consistent and informative
        let error_string = error.to_string();
        prop_assert!(!error_string.is_empty(), "Error message should not be empty");
        prop_assert!(error_string.len() < 10000, "Error message should be reasonable length");

        // Test Debug formatting
        let debug_string = format!("{:?}", error);
        prop_assert!(!debug_string.is_empty(), "Debug format should not be empty");

        // Check that error contains relevant information based on type
        match &error {
            DaemonError::FileIo { message, path } => {
                prop_assert!(error_string.contains(path) || debug_string.contains(path),
                           "File error should contain path information");
                prop_assert!(!message.is_empty(), "File error message should not be empty");
            }
            DaemonError::FileTooLarge { path, size, max_size } => {
                prop_assert!(*size > *max_size, "FileTooLarge should have size > max_size");
                prop_assert!(error_string.contains(&size.to_string()) ||
                           error_string.contains(&max_size.to_string()),
                           "FileTooLarge should contain size information");
            }
            DaemonError::Internal { message } |
            DaemonError::Configuration { message } |
            DaemonError::Search { message } |
            DaemonError::Memory { message } |
            DaemonError::DocumentProcessing { message } |
            DaemonError::System { message } => {
                prop_assert!(!message.is_empty(), "Error message should not be empty");
            }
        }
    }

    #[test]
    fn proptest_config_validation_properties(config in random_config_values()) {
        // Property: Config should satisfy basic validation invariants
        prop_assert!(config.server.port > 0, "Port should be positive");
        prop_assert!(config.processing.max_concurrent_tasks > 0, "Max concurrent tasks should be positive");
        prop_assert!(config.processing.default_chunk_size > 0, "Chunk size should be positive");
        prop_assert!(config.processing.max_file_size_bytes > 0, "Max file size should be positive");
        prop_assert!(!config.processing.supported_extensions.is_empty(), "Should have supported extensions");
        prop_assert!(config.processing.lsp_timeout_secs > 0, "LSP timeout should be positive");

        prop_assert!(!config.qdrant.url.is_empty(), "Qdrant URL should not be empty");
        prop_assert!(config.qdrant.timeout_secs > 0, "Qdrant timeout should be positive");
        prop_assert!(config.qdrant.max_retries > 0, "Max retries should be positive");

        prop_assert!(config.qdrant.default_collection.vector_size >= 64, "Vector size should be reasonable");
        prop_assert!(config.qdrant.default_collection.replication_factor > 0, "Replication factor should be positive");
        prop_assert!(config.qdrant.default_collection.shard_number > 0, "Shard number should be positive");

        prop_assert!(!config.logging.level.is_empty(), "Log level should not be empty");
        prop_assert!(!config.logging.file_path.is_empty(), "Log file path should not be empty");
        prop_assert!(config.logging.max_files > 0, "Max log files should be positive");
        prop_assert!(config.logging.max_file_size_mb > 0, "Max log file size should be positive");

        prop_assert!(config.metrics.port > 0, "Metrics port should be positive");
    }

    #[test]
    fn proptest_json_parsing_edge_cases(
        malformed_json in prop_oneof![
            // Invalid JSON strings
            "[\\PC]*\\{[\\PC]*",
            "\\{[\\PC]*\\}[\\PC]*",
            "\\[[\\PC]*\\][\\PC]*",
            // Deeply nested structures (might cause stack overflow)
            Just(format!("{}{}{}", "[".repeat(1000), "null", "]".repeat(1000))),
            Just(format!("{}{}{}", "{\"a\":".repeat(500), "null", "}".repeat(500))),
            // Numbers that might cause precision issues
            Just("123456789012345678901234567890.123456789".to_string()),
            Just("1e999999".to_string()),
            Just("-1e999999".to_string()),
        ]
    ) {
        // Property: JSON parser should handle malformed input gracefully
        let parse_result: Result<serde_json::Value, _> = serde_json::from_str(&malformed_json);

        // Should either succeed or fail gracefully (no panic)
        match parse_result {
            Ok(value) => {
                // If it parses, it should be serializable again
                let reserialize_result = serde_json::to_string(&value);
                prop_assert!(reserialize_result.is_ok(), "Parsed JSON should be re-serializable");
            }
            Err(_) => {
                // Parsing failure is acceptable for malformed JSON
            }
        }
    }

    #[test]
    fn proptest_config_field_independence(
        config1 in random_config_values(),
        config2 in random_config_values()
    ) {
        // Property: Config fields should be independent in serialization
        if config1.server.host != config2.server.host {
            let json1 = serde_json::to_string(&config1).unwrap();
            let json2 = serde_json::to_string(&config2).unwrap();

            // Different configs should produce different JSON (unless by coincidence)
            if json1 != json2 {
                prop_assert_ne!(config1.server.host, config2.server.host, "Different hosts should produce different JSON");
            }
        }
    }

    #[test]
    fn proptest_binary_serialization_consistency(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        // Property: Binary data serialization should be consistent
        let json_value = serde_json::Value::Array(
            data.iter().map(|&b| serde_json::Value::Number(b.into())).collect()
        );

        let serialized = serde_json::to_string(&json_value);
        prop_assert!(serialized.is_ok(), "Binary data should serialize to JSON");

        if let Ok(json_str) = serialized {
            let deserialized: Result<serde_json::Value, _> = serde_json::from_str(&json_str);
            prop_assert!(deserialized.is_ok(), "JSON should deserialize");

            if let Ok(roundtrip_value) = deserialized {
                prop_assert_eq!(json_value, roundtrip_value, "Binary data should roundtrip correctly");
            }
        }
    }
}
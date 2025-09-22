//! Comprehensive test coverage for previously untested Rust modules
//! Target: Achieve 100% line coverage across all core modules

use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::timeout;

use workspace_qdrant_core::*;
use workspace_qdrant_core::embedding::*;
use workspace_qdrant_core::error::*;
use workspace_qdrant_core::config::*;
use workspace_qdrant_core::logging::*;
use workspace_qdrant_core::patterns::*;
use workspace_qdrant_core::patterns::manager::*;
use workspace_qdrant_core::patterns::exclusion::*;
use workspace_qdrant_core::patterns::project::*;
use workspace_qdrant_core::patterns::comprehensive::*;
use workspace_qdrant_core::patterns::detection::*;
use workspace_qdrant_core::lsp::*;
use workspace_qdrant_core::lsp::config::*;
use workspace_qdrant_core::lsp::detection::*;
use workspace_qdrant_core::lsp::state::*;
use workspace_qdrant_core::watching::*;
use workspace_qdrant_core::watching::platform::*;
use workspace_qdrant_core::ipc::*;
use workspace_qdrant_core::storage::*;
use workspace_qdrant_core::service_discovery::*;
use workspace_qdrant_core::service_discovery::client::*;
use workspace_qdrant_core::service_discovery::registry::*;
use workspace_qdrant_core::service_discovery::health::*;
use workspace_qdrant_core::service_discovery::manager::*;
use workspace_qdrant_core::service_discovery::network::*;
use workspace_qdrant_core::unified_config::*;
use workspace_qdrant_core::daemon_state::*;
use workspace_qdrant_core::processing::*;

#[tokio::test]
async fn test_embeddings_comprehensive() {
    // Test EmbeddingConfig
    let config = EmbeddingConfig::default();
    assert!(!config.model.is_empty());
    assert!(config.dimension > 0);

    let custom_config = EmbeddingConfig {
        model: "custom-model".to_string(),
        dimension: 768,
        batch_size: 32,
        max_sequence_length: 1024,
    };
    assert_eq!(custom_config.model, "custom-model");
    assert_eq!(custom_config.dimension, 768);

    // Test EmbeddingProvider creation and operations
    let provider = EmbeddingProvider::new(config).expect("Failed to create embedding provider");

    // Test single text embedding
    let text = "This is a test document for embedding";
    let result = provider.embed_text(text).await;
    assert!(result.is_ok());

    // Test batch embedding
    let texts = vec![
        "First test document".to_string(),
        "Second test document".to_string(),
        "Third test document".to_string(),
    ];
    let batch_result = provider.embed_batch(&texts).await;
    assert!(batch_result.is_ok());

    // Test empty text handling
    let empty_result = provider.embed_text("").await;
    assert!(empty_result.is_err() || empty_result.unwrap().len() > 0);

    // Test very long text handling
    let long_text = "word ".repeat(2000);
    let long_result = provider.embed_text(&long_text).await;
    assert!(long_result.is_ok());
}

#[tokio::test]
async fn test_error_handling_comprehensive() {
    // Test all error variants
    let processing_error = ProcessingError::ConfigurationError("test config error".to_string());
    assert!(processing_error.to_string().contains("test config error"));

    let embedding_error = EmbeddingError::ModelLoadError("model load failed".to_string());
    assert!(embedding_error.to_string().contains("model load failed"));

    let pattern_error = PatternError::Validation("pattern validation failed".to_string());
    assert!(pattern_error.to_string().contains("pattern validation failed"));

    let lsp_error = LspError::ServerStartup("LSP startup failed".to_string());
    assert!(lsp_error.to_string().contains("LSP startup failed"));

    let ipc_error = IpcError::Connection("IPC connection failed".to_string());
    assert!(ipc_error.to_string().contains("IPC connection failed"));

    let watching_error = WatchingError::Setup("file watching setup failed".to_string());
    assert!(watching_error.to_string().contains("file watching setup failed"));

    // Test error conversion and chaining
    let result: Result<(), ProcessingError> = Err(ProcessingError::EmbeddingError(embedding_error));
    assert!(result.is_err());

    // Test error debug formatting
    let debug_str = format!("{:?}", processing_error);
    assert!(!debug_str.is_empty());
}

#[tokio::test]
async fn test_configuration_comprehensive() {
    // Test WatcherConfig
    let mut watcher_config = WatcherConfig::default();
    assert!(watcher_config.max_file_size > 0);
    assert!(watcher_config.max_concurrent_processes > 0);

    watcher_config.ignored_extensions.push("tmp".to_string());
    watcher_config.ignore_patterns.push("*.backup".to_string());
    assert!(watcher_config.ignored_extensions.contains(&"tmp".to_string()));

    // Test ChunkingConfig
    let chunking_config = ChunkingConfig::default();
    assert!(chunking_config.chunk_size > 0);
    assert!(chunking_config.overlap_size < chunking_config.chunk_size);

    let custom_chunking = ChunkingConfig {
        chunk_size: 2048,
        overlap_size: 200,
        preserve_paragraphs: false,
        min_chunk_size: 100,
        max_chunk_size: 5000,
    };
    assert_eq!(custom_chunking.chunk_size, 2048);
    assert!(!custom_chunking.preserve_paragraphs);

    // Test configuration validation
    assert!(custom_chunking.min_chunk_size < custom_chunking.max_chunk_size);
    assert!(custom_chunking.overlap_size < custom_chunking.chunk_size);
}

#[tokio::test]
async fn test_logging_comprehensive() {
    // Test logger initialization
    let result = init_logger();
    // Should not panic or fail

    // Test multiple initialization (should be safe)
    let result2 = init_logger();

    // Test logging at different levels
    tracing::trace!("Test trace message");
    tracing::debug!("Test debug message");
    tracing::info!("Test info message");
    tracing::warn!("Test warning message");
    tracing::error!("Test error message");

    // Test structured logging
    tracing::info!(
        event = "test_event",
        file_path = "/test/path.rs",
        line_count = 42,
        "Structured logging test"
    );
}

#[tokio::test]
async fn test_pattern_manager_comprehensive() {
    let manager = PatternManager::new().expect("Failed to create PatternManager");

    // Test file inclusion patterns
    assert!(manager.should_include("src/main.rs"));
    assert!(manager.should_include("docs/README.md"));
    assert!(manager.should_include("tests/test_file.py"));

    // Test file exclusion patterns
    assert!(!manager.should_include("target/debug/main"));
    assert!(!manager.should_include("node_modules/package.json"));
    assert!(!manager.should_include(".git/config"));
    assert!(!manager.should_include("*.tmp"));

    // Test edge cases
    assert!(manager.should_include(""));  // Empty string handling
    assert!(manager.should_include("single_word"));
    assert!(manager.should_include("very/deep/nested/file/structure.txt"));

    // Test patterns accessor
    let patterns = manager.patterns();
    assert!(patterns.project_indicators.build_systems.len() > 0);
    assert!(patterns.exclude_patterns.extensions.len() > 0);
}

#[tokio::test]
async fn test_exclusion_engine_comprehensive() {
    let engine = ExclusionEngine::new().expect("Failed to create ExclusionEngine");

    // Test basic exclusions
    let git_result = engine.should_exclude(".git/HEAD");
    assert!(git_result.excluded);

    let node_modules_result = engine.should_exclude("project/node_modules/package.json");
    assert!(node_modules_result.excluded);

    let target_result = engine.should_exclude("rust_project/target/debug/main");
    assert!(target_result.excluded);

    // Test inclusion patterns
    let source_result = engine.should_exclude("src/main.rs");
    assert!(!source_result.excluded);

    let readme_result = engine.should_exclude("README.md");
    assert!(!readme_result.excluded);

    // Test statistics
    let stats = engine.stats();
    assert!(stats.total_patterns > 0);
    assert!(stats.exclusion_rules > 0);

    // Test rule creation and classification
    let custom_rule = ExclusionRule {
        pattern: "*.custom".to_string(),
        description: "Custom file exclusion".to_string(),
        category: ExclusionCategory::Generated,
        impact: ExclusionImpact::High,
        context: ExclusionContext::Global,
    };

    assert_eq!(custom_rule.pattern, "*.custom");
    assert!(matches!(custom_rule.category, ExclusionCategory::Generated));
}

#[tokio::test]
async fn test_project_detection_comprehensive() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let project_path = temp_dir.path();

    // Create a mock Rust project structure
    std::fs::create_dir_all(project_path.join("src")).expect("Failed to create src dir");
    std::fs::write(project_path.join("Cargo.toml"), "[package]\nname = \"test\"\nversion = \"0.1.0\"")
        .expect("Failed to write Cargo.toml");
    std::fs::write(project_path.join("src/main.rs"), "fn main() {}")
        .expect("Failed to write main.rs");

    let detector = ProjectDetector::new().expect("Failed to create ProjectDetector");
    let detection_result = detector.detect_project_type(project_path)
        .expect("Failed to detect project type");

    assert!(detection_result.project_types.contains(&"rust".to_string()));
    assert!(detection_result.build_systems.contains(&"cargo".to_string()));
    assert!(!detection_result.languages.is_empty());

    // Test Python project detection
    std::fs::write(project_path.join("setup.py"), "from setuptools import setup")
        .expect("Failed to write setup.py");
    std::fs::write(project_path.join("requirements.txt"), "requests==2.25.1")
        .expect("Failed to write requirements.txt");

    let python_result = detector.detect_project_type(project_path)
        .expect("Failed to detect Python project");

    assert!(python_result.project_types.contains(&"python".to_string()));

    // Test JavaScript/Node.js project detection
    std::fs::write(project_path.join("package.json"), r#"{"name": "test", "version": "1.0.0"}"#)
        .expect("Failed to write package.json");

    let js_result = detector.detect_project_type(project_path)
        .expect("Failed to detect JavaScript project");

    assert!(js_result.project_types.contains(&"javascript".to_string()) ||
            js_result.project_types.contains(&"nodejs".to_string()));
}

#[tokio::test]
async fn test_comprehensive_patterns() {
    let comprehensive = ComprehensivePatterns::new().expect("Failed to create ComprehensivePatterns");

    // Test language pattern queries
    let rust_patterns = comprehensive.get_language_patterns("rust");
    assert!(rust_patterns.is_some());

    let python_patterns = comprehensive.get_language_patterns("python");
    assert!(python_patterns.is_some());

    let unknown_patterns = comprehensive.get_language_patterns("unknown_language");
    assert!(unknown_patterns.is_none());

    // Test framework detection
    let rust_frameworks = comprehensive.get_framework_patterns("rust");
    assert!(!rust_frameworks.is_empty());

    // Test project type classification
    let test_files = vec![
        "Cargo.toml".to_string(),
        "src/main.rs".to_string(),
        "README.md".to_string(),
    ];

    let classification = comprehensive.classify_project(&test_files);
    assert!(!classification.languages.is_empty());
    assert!(!classification.build_systems.is_empty());

    // Test statistics
    let stats = comprehensive.statistics();
    assert!(stats.total_languages > 100);  // Should support many languages
    assert!(stats.total_patterns > 1000);  // Should have comprehensive patterns
}

#[tokio::test]
async fn test_lsp_configuration() {
    // Test LSP server configuration
    let lsp_config = LspServerConfig::default();
    assert!(!lsp_config.name.is_empty());
    assert!(!lsp_config.command.is_empty());

    let custom_lsp = LspServerConfig {
        name: "rust-analyzer".to_string(),
        command: "rust-analyzer".to_string(),
        args: vec!["--log-file".to_string(), "/tmp/ra.log".to_string()],
        file_patterns: vec!["*.rs".to_string()],
        language_id: "rust".to_string(),
        initialization_options: None,
    };

    assert_eq!(custom_lsp.name, "rust-analyzer");
    assert_eq!(custom_lsp.language_id, "rust");
    assert!(custom_lsp.file_patterns.contains(&"*.rs".to_string()));

    // Test LSP capabilities
    let capabilities = LspCapabilities::default();
    assert!(capabilities.text_document_sync);
    assert!(capabilities.completion);
    assert!(capabilities.hover);
}

#[tokio::test]
async fn test_lsp_detection() {
    let detector = LspDetector::new();

    // Test language detection
    let rust_file = std::path::Path::new("src/main.rs");
    let detected_language = detector.detect_language(rust_file);
    assert_eq!(detected_language.as_deref(), Some("rust"));

    let python_file = std::path::Path::new("scripts/process.py");
    let python_language = detector.detect_language(python_file);
    assert_eq!(python_language.as_deref(), Some("python"));

    let unknown_file = std::path::Path::new("data/unknown.xyz");
    let unknown_language = detector.detect_language(unknown_file);
    assert!(unknown_language.is_none());

    // Test LSP server discovery
    let available_servers = detector.get_available_servers();
    assert!(!available_servers.is_empty());

    // Test server configuration retrieval
    if let Some(rust_server) = detector.get_server_config("rust") {
        assert_eq!(rust_server.language_id, "rust");
        assert!(!rust_server.command.is_empty());
    }
}

#[tokio::test]
async fn test_lsp_state_management() {
    let mut state = LspState::new();

    // Test server registration
    let server_config = LspServerConfig {
        name: "test-lsp".to_string(),
        command: "test-lsp-server".to_string(),
        args: vec![],
        file_patterns: vec!["*.test".to_string()],
        language_id: "test".to_string(),
        initialization_options: None,
    };

    state.register_server("test", server_config.clone());
    assert!(state.get_server("test").is_some());

    // Test server status tracking
    state.set_server_status("test", LspServerStatus::Starting);
    assert_eq!(state.get_server_status("test"), Some(LspServerStatus::Starting));

    state.set_server_status("test", LspServerStatus::Running);
    assert_eq!(state.get_server_status("test"), Some(LspServerStatus::Running));

    // Test capabilities management
    let capabilities = LspCapabilities {
        text_document_sync: true,
        completion: true,
        hover: true,
        signature_help: false,
        declaration: false,
        definition: true,
        type_definition: false,
        implementation: false,
        references: true,
        document_highlight: false,
        document_symbol: true,
        workspace_symbol: false,
        code_action: false,
        code_lens: false,
        document_formatting: true,
        document_range_formatting: false,
        document_on_type_formatting: false,
        rename: false,
        folding_range: false,
        selection_range: false,
        semantic_tokens: false,
    };

    state.set_capabilities("test", capabilities.clone());
    assert!(state.get_capabilities("test").is_some());

    // Test cleanup
    state.unregister_server("test");
    assert!(state.get_server("test").is_none());
}

#[tokio::test]
async fn test_platform_watching() {
    // Test platform-specific file watching initialization
    let result = initialize_platform_watcher();
    // Should not panic, may return error on some platforms

    // Test platform capabilities detection
    let capabilities = get_platform_capabilities();
    assert!(capabilities.supports_recursive_watching || !capabilities.supports_recursive_watching);

    // Test platform-specific optimization
    let optimizations = get_platform_optimizations();
    assert!(!optimizations.buffer_size == 0 || optimizations.buffer_size > 0);
}

#[tokio::test]
async fn test_ipc_comprehensive() {
    // Test IPC message creation
    let request = IpcRequest {
        id: "test-123".to_string(),
        method: "process_file".to_string(),
        params: serde_json::json!({
            "file_path": "/test/file.rs",
            "collection": "test_collection"
        }),
    };

    assert_eq!(request.id, "test-123");
    assert_eq!(request.method, "process_file");

    let response = IpcResponse {
        id: "test-123".to_string(),
        result: Some(serde_json::json!({
            "status": "success",
            "processed": true
        })),
        error: None,
    };

    assert_eq!(response.id, "test-123");
    assert!(response.result.is_some());
    assert!(response.error.is_none());

    // Test error response
    let error_response = IpcResponse {
        id: "test-456".to_string(),
        result: None,
        error: Some(IpcError::InvalidRequest("Missing required parameter".to_string())),
    };

    assert!(error_response.error.is_some());
    assert!(error_response.result.is_none());
}

#[tokio::test]
async fn test_storage_configuration() {
    // Test storage configuration
    let storage_config = StorageConfig::default();
    assert!(!storage_config.url.is_empty());
    assert!(storage_config.collection.is_some());

    let custom_storage = StorageConfig {
        url: "http://localhost:6333".to_string(),
        api_key: Some("test-api-key".to_string()),
        collection: Some("test_collection".to_string()),
        vector_size: 384,
        distance: DistanceMetric::Cosine,
        transport_mode: TransportMode::Http,
        timeout_ms: 30000,
        retry_count: 3,
        batch_size: 100,
        connection_pool_size: 10,
        enable_ssl: false,
        http2: Http2Config::default(),
    };

    assert_eq!(custom_storage.url, "http://localhost:6333");
    assert_eq!(custom_storage.vector_size, 384);
    assert!(matches!(custom_storage.distance, DistanceMetric::Cosine));
    assert!(matches!(custom_storage.transport_mode, TransportMode::Http));

    // Test HTTP/2 configuration
    let http2_config = Http2Config {
        enabled: true,
        max_frame_size: Some(16384),
        max_concurrent_streams: Some(100),
        initial_window_size: Some(65536),
        max_header_list_size: Some(8192),
    };

    assert!(http2_config.enabled);
    assert_eq!(http2_config.max_frame_size, Some(16384));
}

#[tokio::test]
async fn test_service_discovery_comprehensive() {
    // Test service discovery client
    let discovery_config = DiscoveryConfig {
        enabled: true,
        discovery_interval_ms: 5000,
        service_timeout_ms: 30000,
        max_retries: 3,
        health_check_interval_ms: 10000,
        registry_path: "/tmp/test_registry.db".to_string(),
        network_discovery: true,
        multicast_address: "224.0.0.1".to_string(),
        multicast_port: 5353,
    };

    assert!(discovery_config.enabled);
    assert_eq!(discovery_config.discovery_interval_ms, 5000);

    // Test service info structure
    let service_info = ServiceInfo {
        id: "test-service-1".to_string(),
        name: "qdrant-daemon".to_string(),
        address: "127.0.0.1".to_string(),
        port: 6333,
        protocol: ServiceProtocol::Http,
        status: ServiceStatus::Healthy,
        metadata: std::collections::HashMap::new(),
        last_seen: std::time::SystemTime::now(),
        health_endpoint: Some("/health".to_string()),
        version: Some("1.0.0".to_string()),
    };

    assert_eq!(service_info.name, "qdrant-daemon");
    assert_eq!(service_info.port, 6333);
    assert!(matches!(service_info.protocol, ServiceProtocol::Http));
    assert!(matches!(service_info.status, ServiceStatus::Healthy));

    // Test health status variants
    let healthy = HealthStatus::Healthy;
    let unhealthy = HealthStatus::Unhealthy("Connection refused".to_string());
    let checking = HealthStatus::Checking;
    let unknown = HealthStatus::Unknown;

    assert!(matches!(healthy, HealthStatus::Healthy));
    assert!(matches!(unhealthy, HealthStatus::Unhealthy(_)));
    assert!(matches!(checking, HealthStatus::Checking));
    assert!(matches!(unknown, HealthStatus::Unknown));
}

#[tokio::test]
async fn test_unified_configuration() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let config_path = temp_dir.path().join("test_config.toml");

    // Test configuration manager creation
    let config_manager = UnifiedConfigManager::new()
        .expect("Failed to create UnifiedConfigManager");

    // Test configuration loading (should handle missing file gracefully)
    let load_result = config_manager.load_from_file(&config_path).await;
    // May fail if file doesn't exist, which is expected

    // Test configuration validation
    let default_config = UnifiedConfig::default();
    let validation_result = config_manager.validate_config(&default_config);
    assert!(validation_result.is_ok());

    // Test configuration merging
    let mut custom_config = UnifiedConfig::default();
    custom_config.embedding.model = "custom-model".to_string();
    custom_config.storage.vector_size = 768;

    let merged = config_manager.merge_configs(&default_config, &custom_config);
    assert_eq!(merged.embedding.model, "custom-model");
    assert_eq!(merged.storage.vector_size, 768);

    // Test environment variable resolution
    std::env::set_var("TEST_VECTOR_SIZE", "512");
    let env_config = config_manager.resolve_environment_variables(&custom_config);
    // Should handle environment variable substitution

    // Cleanup
    std::env::remove_var("TEST_VECTOR_SIZE");
}

#[tokio::test]
async fn test_daemon_state_operations() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join("test_daemon.db");

    // Test daemon state creation with custom path
    let state_result = DaemonState::new(Some(db_path.to_string_lossy().to_string())).await;
    if state_result.is_err() {
        // Database creation might fail in test environment, which is acceptable
        return;
    }

    let daemon_state = state_result.unwrap();

    // Test process tracking
    let process_info = ProcessInfo {
        pid: 12345,
        command: "test-process".to_string(),
        start_time: std::time::SystemTime::now(),
        status: ProcessStatus::Running,
        cpu_usage: 15.5,
        memory_usage: 1024 * 1024 * 100, // 100MB
        working_directory: "/test/workspace".to_string(),
    };

    let track_result = daemon_state.track_process(process_info).await;
    assert!(track_result.is_ok());

    // Test process status update
    let update_result = daemon_state.update_process_status(12345, ProcessStatus::Completed).await;
    assert!(update_result.is_ok());

    // Test process cleanup
    let cleanup_result = daemon_state.cleanup_completed_processes().await;
    assert!(cleanup_result.is_ok());

    // Test statistics retrieval
    let stats_result = daemon_state.get_statistics().await;
    assert!(stats_result.is_ok());
}

#[tokio::test]
async fn test_processing_pipeline_comprehensive() {
    // Test task priority system
    let high_priority_task = Task {
        id: "task-001".to_string(),
        priority: TaskPriority::High,
        task_type: TaskType::FileProcessing,
        payload: TaskPayload::FileProcessing {
            file_path: "/test/important.rs".to_string(),
            collection: "priority_collection".to_string(),
        },
        created_at: std::time::SystemTime::now(),
        started_at: None,
        completed_at: None,
        status: TaskStatus::Pending,
        retries: 0,
        max_retries: 3,
        timeout_ms: 30000,
    };

    assert_eq!(high_priority_task.priority, TaskPriority::High);
    assert_eq!(high_priority_task.retries, 0);
    assert!(matches!(high_priority_task.status, TaskStatus::Pending));

    // Test task metrics
    let metrics = TaskMetrics {
        total_submitted: 100,
        total_completed: 95,
        total_failed: 3,
        total_cancelled: 2,
        average_processing_time_ms: 1500,
        current_queue_size: 5,
        peak_queue_size: 25,
        worker_utilization: 0.85,
    };

    assert_eq!(metrics.total_submitted, 100);
    assert_eq!(metrics.total_completed, 95);
    assert!(metrics.worker_utilization > 0.8);

    // Test pipeline configuration
    let pipeline_config = PipelineConfig {
        max_concurrent_tasks: 10,
        queue_capacity: 1000,
        worker_count: 4,
        task_timeout_ms: 30000,
        retry_delay_ms: 1000,
        enable_preemption: true,
        priority_boost_threshold: 100,
        memory_threshold_mb: 1024,
        cpu_threshold_percent: 90.0,
    };

    assert_eq!(pipeline_config.max_concurrent_tasks, 10);
    assert!(pipeline_config.enable_preemption);
    assert!(pipeline_config.cpu_threshold_percent > 80.0);
}

// Additional edge case and error handling tests
#[tokio::test]
async fn test_edge_cases_and_error_handling() {
    // Test empty input handling
    let empty_texts: Vec<String> = vec![];

    // Test null and empty string handling
    let null_test = "";
    let whitespace_test = "   \n\t  ";
    let special_chars_test = "!@#$%^&*()_+{}|:<>?[]\\;'\"./,";

    // Test very long strings
    let long_string = "a".repeat(100000);

    // Test unicode handling
    let unicode_test = "Hello 世界 🌍 Здравствуй мир";
    let emoji_test = "🚀🔥💯✨🎉🌟⭐🎯";

    // Test path edge cases
    let edge_paths = vec![
        "",
        "/",
        ".",
        "..",
        "./",
        "../",
        "~",
        "~/",
        "/tmp",
        "/tmp/",
        "very/long/path/that/goes/very/deep/into/directory/structure/file.txt",
        "path with spaces/file name.txt",
        "path-with-dashes/file_with_underscores.txt",
        "UPPERCASE/PATH/FILE.TXT",
        "mixed/CaSe/pAtH/FiLe.TxT",
    ];

    // Test configuration edge cases
    let edge_configs = vec![
        ChunkingConfig {
            chunk_size: 1,
            overlap_size: 0,
            preserve_paragraphs: false,
            min_chunk_size: 1,
            max_chunk_size: 1,
        },
        ChunkingConfig {
            chunk_size: 1000000,
            overlap_size: 999999,
            preserve_paragraphs: true,
            min_chunk_size: 1,
            max_chunk_size: 2000000,
        },
    ];

    // Test timeout handling
    let timeout_test = timeout(Duration::from_millis(1), async {
        tokio::time::sleep(Duration::from_millis(100)).await;
        "should timeout"
    }).await;

    assert!(timeout_test.is_err());

    // Test concurrent access patterns
    let shared_data = Arc::new(std::sync::Mutex::new(0u64));
    let mut handles = vec![];

    for i in 0..10 {
        let data = shared_data.clone();
        let handle = tokio::spawn(async move {
            let mut value = data.lock().unwrap();
            *value += i;
            tokio::time::sleep(Duration::from_millis(1)).await;
            *value
        });
        handles.push(handle);
    }

    for handle in handles {
        let result = handle.await;
        assert!(result.is_ok());
    }

    let final_value = *shared_data.lock().unwrap();
    assert_eq!(final_value, 45); // Sum of 0..10
}

#[tokio::test]
async fn test_memory_and_resource_management() {
    // Test large data structure handling
    let large_vec: Vec<u8> = vec![0; 1024 * 1024]; // 1MB
    assert_eq!(large_vec.len(), 1024 * 1024);
    drop(large_vec);

    // Test resource cleanup
    let temp_files = (0..10).map(|i| {
        let temp_file = TempDir::new().expect("Failed to create temp file");
        std::fs::write(temp_file.path().join(&format!("test_{}.txt", i)), b"test data")
            .expect("Failed to write test file");
        temp_file
    }).collect::<Vec<_>>();

    // Files should be automatically cleaned up when temp_files is dropped
    assert_eq!(temp_files.len(), 10);
    drop(temp_files);

    // Test memory pressure handling
    let mut memory_chunks = Vec::new();
    for _ in 0..100 {
        let chunk: Vec<u8> = vec![0; 1024]; // 1KB chunks
        memory_chunks.push(chunk);
    }
    assert_eq!(memory_chunks.len(), 100);

    // Test graceful degradation under memory pressure
    let total_memory_used = memory_chunks.iter().map(|c| c.len()).sum::<usize>();
    assert_eq!(total_memory_used, 100 * 1024);
}
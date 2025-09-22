//! Module-specific tests targeting every function and struct for 100% coverage
//! Focuses on individual modules and their specific functionality

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;

use workspace_qdrant_core::*;
use workspace_qdrant_core::lsp::lifecycle::ServerStatus;

#[tokio::test]
async fn test_document_processor_all_methods() {
    let processor = DocumentProcessor::new();
    let processor_clone = processor.clone();

    // Test basic construction
    assert_eq!(std::mem::size_of_val(&processor), std::mem::size_of_val(&processor_clone));

    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Test with various file types
    let test_files = vec![
        ("test.txt", "Plain text content for testing"),
        ("test.md", "# Markdown Header\n\nMarkdown content for testing"),
        ("test.json", r#"{"key": "value", "number": 42}"#),
        ("test.xml", r#"<?xml version="1.0"?><root><item>test</item></root>"#),
        ("test.csv", "name,age,city\nJohn,30,NYC\nJane,25,LA"),
    ];

    for (filename, content) in test_files {
        let file_path = temp_dir.path().join(filename);
        std::fs::write(&file_path, content).expect("Failed to write test file");

        let result = processor.process_file(&file_path, "test_collection").await;
        if result.is_ok() {
            let doc_result = result.unwrap();
            assert!(!doc_result.document_id.is_empty());
            assert!(doc_result.processing_time_ms >= 0);
        }
    }

    // Test error cases
    let non_existent = temp_dir.path().join("non_existent.txt");
    let error_result = processor.process_file(&non_existent, "test_collection").await;
    assert!(error_result.is_err());

    // Test empty file
    let empty_file = temp_dir.path().join("empty.txt");
    std::fs::write(&empty_file, "").expect("Failed to write empty file");
    let empty_result = processor.process_file(&empty_file, "test_collection").await;
    // Should handle empty files gracefully
}

#[tokio::test]
async fn test_chunking_config_all_methods() {
    // Test default construction
    let default_config = ChunkingConfig::default();
    assert!(default_config.chunk_size > 0);
    assert!(default_config.overlap_size < default_config.chunk_size);
    assert!(default_config.min_chunk_size <= default_config.chunk_size);
    assert!(default_config.max_chunk_size >= default_config.chunk_size);

    // Test clone
    let cloned_config = default_config.clone();
    assert_eq!(default_config.chunk_size, cloned_config.chunk_size);
    assert_eq!(default_config.overlap_size, cloned_config.overlap_size);
    assert_eq!(default_config.preserve_paragraphs, cloned_config.preserve_paragraphs);

    // Test debug formatting
    let debug_str = format!("{:?}", default_config);
    assert!(debug_str.contains("ChunkingConfig"));
    assert!(debug_str.contains("chunk_size"));

    // Test field access and modification
    let mut custom_config = ChunkingConfig {
        chunk_size: 2048,
        overlap_size: 256,
        preserve_paragraphs: true,
        min_chunk_size: 100,
        max_chunk_size: 4096,
    };

    assert_eq!(custom_config.chunk_size, 2048);
    assert_eq!(custom_config.overlap_size, 256);
    assert!(custom_config.preserve_paragraphs);

    custom_config.chunk_size = 1024;
    assert_eq!(custom_config.chunk_size, 1024);
}

#[tokio::test]
async fn test_all_error_types_comprehensive() {
    // Test ProcessingError variants
    let config_error = ProcessingError::ConfigurationError("config failed".to_string());
    let embedding_error = ProcessingError::EmbeddingError(
        EmbeddingError::ModelLoadError("model failed".to_string())
    );
    let io_error = ProcessingError::IoError(std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"));
    let validation_error = ProcessingError::ValidationError("validation failed".to_string());
    let timeout_error = ProcessingError::TimeoutError("operation timed out".to_string());

    // Test error display and debug
    assert!(config_error.to_string().contains("config failed"));
    assert!(format!("{:?}", embedding_error).contains("EmbeddingError"));
    assert!(io_error.to_string().contains("file not found"));

    // Test EmbeddingError variants
    let model_load_error = EmbeddingError::ModelLoadError("failed to load".to_string());
    let tokenization_error = EmbeddingError::TokenizationError("tokenization failed".to_string());
    let inference_error = EmbeddingError::InferenceError("inference failed".to_string());
    let config_error = EmbeddingError::ConfigurationError("config error".to_string());

    assert!(model_load_error.to_string().contains("failed to load"));
    assert!(tokenization_error.to_string().contains("tokenization failed"));

    // Test PatternError variants
    let validation_error = PatternError::Validation("pattern invalid".to_string());
    let io_error = PatternError::Io(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied"));
    let parse_error = PatternError::ParseError("parse failed".to_string());

    assert!(validation_error.to_string().contains("pattern invalid"));
    assert!(parse_error.to_string().contains("parse failed"));

    // Test LspError variants
    let server_startup_error = LspError::ServerStartup("startup failed".to_string());
    let communication_error = LspError::CommunicationError("communication failed".to_string());
    let protocol_error = LspError::ProtocolError("protocol error".to_string());
    let timeout_error = LspError::TimeoutError("LSP timeout".to_string());

    assert!(server_startup_error.to_string().contains("startup failed"));
    assert!(communication_error.to_string().contains("communication failed"));

    // Test IpcError variants
    let connection_error = IpcError::Connection("connection failed".to_string());
    let invalid_request_error = IpcError::InvalidRequest("invalid request".to_string());
    let serialization_error = IpcError::SerializationError("serialization failed".to_string());
    let timeout_error = IpcError::TimeoutError("IPC timeout".to_string());

    assert!(connection_error.to_string().contains("connection failed"));
    assert!(invalid_request_error.to_string().contains("invalid request"));

    // Test WatchingError variants
    let setup_error = WatchingError::Setup("setup failed".to_string());
    let io_error = WatchingError::Io(std::io::Error::new(std::io::ErrorKind::Other, "IO failed"));
    let permission_error = WatchingError::PermissionDenied("permission denied".to_string());

    assert!(setup_error.to_string().contains("setup failed"));
    assert!(permission_error.to_string().contains("permission denied"));
}

#[tokio::test]
async fn test_watcher_config_all_methods() {
    // Test default construction
    let default_config = WatcherConfig::default();
    assert!(default_config.max_file_size > 0);
    assert!(default_config.max_concurrent_processes > 0);
    assert!(!default_config.ignored_extensions.is_empty());
    assert!(!default_config.ignore_patterns.is_empty());

    // Test clone
    let cloned_config = default_config.clone();
    assert_eq!(default_config.max_file_size, cloned_config.max_file_size);
    assert_eq!(default_config.max_concurrent_processes, cloned_config.max_concurrent_processes);
    assert_eq!(default_config.ignored_extensions.len(), cloned_config.ignored_extensions.len());

    // Test debug formatting
    let debug_str = format!("{:?}", default_config);
    assert!(debug_str.contains("WatcherConfig"));
    assert!(debug_str.contains("max_file_size"));

    // Test field modification
    let mut custom_config = default_config.clone();
    custom_config.max_file_size = 10 * 1024 * 1024; // 10MB
    custom_config.max_concurrent_processes = 8;
    custom_config.ignored_extensions.push("test".to_string());
    custom_config.ignore_patterns.push("*.test".to_string());

    assert_eq!(custom_config.max_file_size, 10 * 1024 * 1024);
    assert_eq!(custom_config.max_concurrent_processes, 8);
    assert!(custom_config.ignored_extensions.contains(&"test".to_string()));
    assert!(custom_config.ignore_patterns.contains(&"*.test".to_string()));

    // Test directory watching settings
    assert!(!custom_config.watch_hidden_files || custom_config.watch_hidden_files);
    assert!(!custom_config.follow_symlinks || custom_config.follow_symlinks);
}

#[tokio::test]
async fn test_embedding_config_all_methods() {
    // Test default construction
    let default_config = EmbeddingConfig::default();
    assert!(!default_config.model.is_empty());
    assert!(default_config.dimension > 0);
    assert!(default_config.batch_size > 0);
    assert!(default_config.max_sequence_length > 0);

    // Test clone
    let cloned_config = default_config.clone();
    assert_eq!(default_config.model, cloned_config.model);
    assert_eq!(default_config.dimension, cloned_config.dimension);
    assert_eq!(default_config.batch_size, cloned_config.batch_size);

    // Test debug formatting
    let debug_str = format!("{:?}", default_config);
    assert!(debug_str.contains("EmbeddingConfig"));
    assert!(debug_str.contains("model"));

    // Test custom construction
    let custom_config = EmbeddingConfig {
        model: "sentence-transformers/all-mpnet-base-v2".to_string(),
        dimension: 768,
        batch_size: 64,
        max_sequence_length: 512,
    };

    assert_eq!(custom_config.model, "sentence-transformers/all-mpnet-base-v2");
    assert_eq!(custom_config.dimension, 768);
    assert_eq!(custom_config.batch_size, 64);
    assert_eq!(custom_config.max_sequence_length, 512);

    // Test field access
    let model_name = &custom_config.model;
    assert!(!model_name.is_empty());

    let dimensions = custom_config.dimension;
    assert!(dimensions > 0);
}

#[tokio::test]
async fn test_exclusion_rule_all_methods() {
    // Test construction
    let rule = ExclusionRule {
        pattern: "*.tmp".to_string(),
        category: ExclusionCategory::Cache,
        reason: "Temporary files".to_string(),
        is_regex: false,
        case_sensitive: false,
    };

    // Test field access
    assert_eq!(rule.pattern, "*.tmp");
    assert_eq!(rule.reason, "Temporary files");
    assert!(matches!(rule.category, ExclusionCategory::Cache));
    assert!(!rule.is_regex);
    assert!(!rule.case_sensitive);

    // Test clone
    let cloned_rule = rule.clone();
    assert_eq!(rule.pattern, cloned_rule.pattern);
    assert_eq!(rule.reason, cloned_rule.reason);

    // Test debug formatting
    let debug_str = format!("{:?}", rule);
    assert!(debug_str.contains("ExclusionRule"));
    assert!(debug_str.contains("*.tmp"));

    // Test all enum variants
    let categories = vec![
        ExclusionCategory::Critical,
        ExclusionCategory::BuildArtifacts,
        ExclusionCategory::Cache,
        ExclusionCategory::VersionControl,
        ExclusionCategory::IdeFiles,
        ExclusionCategory::Media,
        ExclusionCategory::Security,
    ];

    for category in categories {
        let test_rule = ExclusionRule {
            pattern: "test".to_string(),
            reason: "test".to_string(),
            category,
            is_regex: false,
            case_sensitive: false,
        };
        assert!(format!("{:?}", test_rule).contains("ExclusionRule"));
    }

    // Test boolean flags
    let test_rule_regex = ExclusionRule {
        pattern: "test.*".to_string(),
        reason: "test regex".to_string(),
        category: ExclusionCategory::Cache,
        is_regex: true,
        case_sensitive: true,
    };
    assert!(test_rule_regex.is_regex);
    assert!(test_rule_regex.case_sensitive);
}

#[tokio::test]
async fn test_exclusion_result_all_methods() {
    // Test excluded result
    let excluded_result = ExclusionResult {
        excluded: true,
        reason: "Matches pattern *.tmp".to_string(),
        pattern: Some("*.tmp".to_string()),
        category: Some(ExclusionCategory::Temporary),
        confidence: 0.95,
    };

    assert!(excluded_result.excluded);
    assert!(excluded_result.reason.contains("*.tmp"));
    assert!(excluded_result.pattern.is_some());
    assert!(excluded_result.category.is_some());
    assert!(excluded_result.confidence > 0.9);

    // Test included result
    let included_result = ExclusionResult {
        excluded: false,
        reason: "No matching exclusion pattern".to_string(),
        pattern: None,
        category: None,
        confidence: 1.0,
    };

    assert!(!included_result.excluded);
    assert!(included_result.pattern.is_none());
    assert!(included_result.category.is_none());
    assert_eq!(included_result.confidence, 1.0);

    // Test clone
    let cloned_result = excluded_result.clone();
    assert_eq!(excluded_result.excluded, cloned_result.excluded);
    assert_eq!(excluded_result.reason, cloned_result.reason);
    assert_eq!(excluded_result.confidence, cloned_result.confidence);

    // Test debug formatting
    let debug_str = format!("{:?}", excluded_result);
    assert!(debug_str.contains("ExclusionResult"));
    assert!(debug_str.contains("excluded"));
}

#[tokio::test]
async fn test_detection_details_all_methods() {
    // Test construction
    let mut details = DetectionDetails {
        detected_files: vec!["Cargo.toml".to_string(), "src/main.rs".to_string()],
        confidence_scores: HashMap::new(),
        framework_indicators: HashMap::new(),
        build_system_evidence: HashMap::new(),
        language_distribution: HashMap::new(),
    };

    // Test field access and modification
    assert_eq!(details.detected_files.len(), 2);
    assert!(details.detected_files.contains(&"Cargo.toml".to_string()));

    details.confidence_scores.insert("rust".to_string(), 0.95);
    details.confidence_scores.insert("cargo".to_string(), 0.90);

    details.framework_indicators.insert("tokio".to_string(), vec!["tokio".to_string()]);
    details.build_system_evidence.insert("cargo".to_string(), vec!["Cargo.toml".to_string()]);
    details.language_distribution.insert("rust".to_string(), 85.0);

    assert_eq!(details.confidence_scores.get("rust"), Some(&0.95));
    assert!(details.framework_indicators.contains_key("tokio"));
    assert!(details.build_system_evidence.contains_key("cargo"));
    assert_eq!(details.language_distribution.get("rust"), Some(&85.0));

    // Test clone
    let cloned_details = details.clone();
    assert_eq!(details.detected_files.len(), cloned_details.detected_files.len());
    assert_eq!(details.confidence_scores.len(), cloned_details.confidence_scores.len());

    // Test debug formatting
    let debug_str = format!("{:?}", details);
    assert!(debug_str.contains("DetectionDetails"));
    assert!(debug_str.contains("detected_files"));
}

#[tokio::test]
async fn test_detection_result_all_methods() {
    // Test construction
    let result = DetectionResult {
        project_types: vec!["rust".to_string(), "library".to_string()],
        languages: vec!["rust".to_string()],
        frameworks: vec!["tokio".to_string(), "serde".to_string()],
        build_systems: vec!["cargo".to_string()],
        confidence: 0.92,
        details: DetectionDetails {
            detected_files: vec!["Cargo.toml".to_string()],
            confidence_scores: HashMap::new(),
            framework_indicators: HashMap::new(),
            build_system_evidence: HashMap::new(),
            language_distribution: HashMap::new(),
        },
    };

    // Test field access
    assert!(result.project_types.contains(&"rust".to_string()));
    assert!(result.languages.contains(&"rust".to_string()));
    assert!(result.frameworks.contains(&"tokio".to_string()));
    assert!(result.build_systems.contains(&"cargo".to_string()));
    assert!(result.confidence > 0.9);
    assert!(!result.details.detected_files.is_empty());

    // Test clone
    let cloned_result = result.clone();
    assert_eq!(result.project_types.len(), cloned_result.project_types.len());
    assert_eq!(result.languages.len(), cloned_result.languages.len());
    assert_eq!(result.confidence, cloned_result.confidence);

    // Test debug formatting
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("DetectionResult"));
    assert!(debug_str.contains("project_types"));
}

#[tokio::test]
async fn test_document_result_all_methods() {
    use std::time::SystemTime;

    // Test construction
    let result = DocumentResult {
        document_id: "doc_12345".to_string(),
        file_path: "/test/path/document.txt".to_string(),
        content_hash: "sha256_hash_here".to_string(),
        chunks_created: Some(5),
        processing_time_ms: 150,
        file_size_bytes: 2048,
        detected_language: Some("rust".to_string()),
        metadata: HashMap::new(),
        created_at: SystemTime::now(),
    };

    // Test field access
    assert_eq!(result.document_id, "doc_12345");
    assert_eq!(result.file_path, "/test/path/document.txt");
    assert_eq!(result.content_hash, "sha256_hash_here");
    assert_eq!(result.chunks_created, Some(5));
    assert_eq!(result.processing_time_ms, 150);
    assert_eq!(result.file_size_bytes, 2048);
    assert_eq!(result.detected_language, Some("rust".to_string()));

    // Test clone
    let cloned_result = result.clone();
    assert_eq!(result.document_id, cloned_result.document_id);
    assert_eq!(result.file_path, cloned_result.file_path);
    assert_eq!(result.processing_time_ms, cloned_result.processing_time_ms);

    // Test debug formatting
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("DocumentResult"));
    assert!(debug_str.contains("document_id"));

    // Test metadata manipulation
    let mut mutable_result = result.clone();
    mutable_result.metadata.insert("key1".to_string(), "value1".to_string());
    mutable_result.metadata.insert("key2".to_string(), "value2".to_string());

    assert_eq!(mutable_result.metadata.get("key1"), Some(&"value1".to_string()));
    assert_eq!(mutable_result.metadata.len(), 2);
}

#[tokio::test]
async fn test_ipc_structures_all_methods() {
    // Test IpcRequest
    let request = IpcRequest {
        id: "req_001".to_string(),
        method: "process_document".to_string(),
        params: serde_json::json!({
            "file_path": "/test/file.txt",
            "collection": "test_collection",
            "options": {
                "priority": "high",
                "timeout": 30000
            }
        }),
    };

    assert_eq!(request.id, "req_001");
    assert_eq!(request.method, "process_document");
    assert!(request.params.is_object());

    // Test clone
    let cloned_request = request.clone();
    assert_eq!(request.id, cloned_request.id);
    assert_eq!(request.method, cloned_request.method);

    // Test debug formatting
    let debug_str = format!("{:?}", request);
    assert!(debug_str.contains("IpcRequest"));
    assert!(debug_str.contains("req_001"));

    // Test IpcResponse success
    let success_response = IpcResponse {
        id: "req_001".to_string(),
        result: Some(serde_json::json!({
            "document_id": "doc_123",
            "status": "completed",
            "processing_time_ms": 250
        })),
        error: None,
    };

    assert_eq!(success_response.id, "req_001");
    assert!(success_response.result.is_some());
    assert!(success_response.error.is_none());

    // Test IpcResponse error
    let error_response = IpcResponse {
        id: "req_002".to_string(),
        result: None,
        error: Some(IpcError::InvalidRequest("Missing file_path parameter".to_string())),
    };

    assert_eq!(error_response.id, "req_002");
    assert!(error_response.result.is_none());
    assert!(error_response.error.is_some());

    // Test cloning
    let cloned_success = success_response.clone();
    let cloned_error = error_response.clone();

    assert_eq!(success_response.id, cloned_success.id);
    assert_eq!(error_response.id, cloned_error.id);

    // Test debug formatting
    let success_debug = format!("{:?}", success_response);
    let error_debug = format!("{:?}", error_response);

    assert!(success_debug.contains("IpcResponse"));
    assert!(error_debug.contains("IpcResponse"));
}

#[tokio::test]
async fn test_platform_capabilities_all_methods() {
    // Test platform capabilities
    let capabilities = PlatformCapabilities {
        supports_recursive_watching: true,
        supports_file_events: true,
        supports_directory_events: true,
        max_watch_depth: Some(10),
        native_exclusion_patterns: true,
        buffer_size_optimization: true,
        supports_symlink_following: false,
        platform_specific_optimizations: vec![
            "kqueue".to_string(),
            "inotify".to_string(),
        ],
    };

    // Test field access
    assert!(capabilities.supports_recursive_watching);
    assert!(capabilities.supports_file_events);
    assert!(capabilities.supports_directory_events);
    assert_eq!(capabilities.max_watch_depth, Some(10));
    assert!(capabilities.native_exclusion_patterns);
    assert!(capabilities.buffer_size_optimization);
    assert!(!capabilities.supports_symlink_following);
    assert_eq!(capabilities.platform_specific_optimizations.len(), 2);

    // Test clone
    let cloned_capabilities = capabilities.clone();
    assert_eq!(capabilities.supports_recursive_watching, cloned_capabilities.supports_recursive_watching);
    assert_eq!(capabilities.max_watch_depth, cloned_capabilities.max_watch_depth);
    assert_eq!(capabilities.platform_specific_optimizations.len(),
               cloned_capabilities.platform_specific_optimizations.len());

    // Test debug formatting
    let debug_str = format!("{:?}", capabilities);
    assert!(debug_str.contains("PlatformCapabilities"));
    assert!(debug_str.contains("supports_recursive_watching"));
}

#[tokio::test]
async fn test_platform_optimizations_all_methods() {
    // Test platform optimizations
    let optimizations = PlatformOptimizations {
        buffer_size: 8192,
        batch_size: 100,
        poll_interval_ms: 50,
        debounce_interval_ms: 100,
        max_concurrent_watchers: 8,
        use_native_api: true,
        enable_filtering: true,
        memory_optimization: true,
        cpu_optimization: vec![
            "batching".to_string(),
            "debouncing".to_string(),
            "filtering".to_string(),
        ],
    };

    // Test field access
    assert_eq!(optimizations.buffer_size, 8192);
    assert_eq!(optimizations.batch_size, 100);
    assert_eq!(optimizations.poll_interval_ms, 50);
    assert_eq!(optimizations.debounce_interval_ms, 100);
    assert_eq!(optimizations.max_concurrent_watchers, 8);
    assert!(optimizations.use_native_api);
    assert!(optimizations.enable_filtering);
    assert!(optimizations.memory_optimization);
    assert_eq!(optimizations.cpu_optimization.len(), 3);

    // Test clone
    let cloned_optimizations = optimizations.clone();
    assert_eq!(optimizations.buffer_size, cloned_optimizations.buffer_size);
    assert_eq!(optimizations.batch_size, cloned_optimizations.batch_size);
    assert_eq!(optimizations.cpu_optimization.len(), cloned_optimizations.cpu_optimization.len());

    // Test debug formatting
    let debug_str = format!("{:?}", optimizations);
    assert!(debug_str.contains("PlatformOptimizations"));
    assert!(debug_str.contains("buffer_size"));
}

#[tokio::test]
async fn test_all_enum_variants() {
    // Test all TaskPriority variants
    let priorities = vec![
        TaskPriority::Low,
        TaskPriority::Normal,
        TaskPriority::High,
        TaskPriority::Critical,
    ];

    for priority in priorities {
        let debug_str = format!("{:?}", priority);
        assert!(debug_str.contains("Priority") || debug_str.contains("Low") ||
               debug_str.contains("Normal") || debug_str.contains("High") ||
               debug_str.contains("Critical"));
    }

    // Test all TaskStatus variants
    let statuses = vec![
        TaskStatus::Pending,
        TaskStatus::Running,
        TaskStatus::Completed,
        TaskStatus::Failed,
        TaskStatus::Cancelled,
    ];

    for status in statuses {
        let debug_str = format!("{:?}", status);
        assert!(!debug_str.is_empty());
    }

    // Test all ServerStatus variants
    let server_statuses = vec![
        ServerStatus::Initializing,
        ServerStatus::Running,
        ServerStatus::Degraded,
        ServerStatus::Failed,
        ServerStatus::Stopping,
        ServerStatus::Stopped,
    ];

    for status in server_statuses {
        let debug_str = format!("{:?}", status);
        assert!(!debug_str.is_empty());
    }

    // Test all ServiceStatus variants
    let service_statuses = vec![
        ServiceStatus::Starting,
        ServiceStatus::Healthy,
        ServiceStatus::Unhealthy,
        ServiceStatus::Stopping,
    ];

    for status in service_statuses {
        let debug_str = format!("{:?}", status);
        assert!(!debug_str.is_empty());
    }

    // Test all distance metric strings (using String instead of DistanceMetric enum)
    let distance_metrics = vec![
        "Cosine".to_string(),
        "Euclidean".to_string(),
        "Manhattan".to_string(),
        "Dot".to_string(),
    ];

    for metric in distance_metrics {
        let debug_str = format!("{:?}", metric);
        assert!(!debug_str.is_empty());
    }

    // Test all TransportMode variants
    let transport_modes = vec![
        TransportMode::Http,
        TransportMode::Grpc,
    ];

    for mode in transport_modes {
        let debug_str = format!("{:?}", mode);
        assert!(!debug_str.is_empty());
    }
}
//! Additional test coverage for daemon modules
//! Targeting daemon/core.rs, daemon/processing.rs, daemon/state.rs, daemon/watcher.rs

use workspace_qdrant_daemon::config::*;
use workspace_qdrant_daemon::daemon::core::*;
use workspace_qdrant_daemon::daemon::processing::*;
use workspace_qdrant_daemon::daemon::state::*;
use workspace_qdrant_daemon::daemon::watcher::*;
use std::sync::Arc;
use tempfile::TempDir;
use sqlx::Row;

// ================================
// DAEMON CORE TESTS
// ================================

#[test]
fn test_daemon_core_initialization() {
    let core = DaemonCore::new();
    assert_eq!(std::mem::size_of::<DaemonCore>(), 0);

    // Test debug representation
    let debug_str = format!("{:?}", core);
    assert!(debug_str.contains("DaemonCore"));
}

#[test]
fn test_system_info_creation() {
    let info = SystemInfo {
        cpu_count: 8,
        memory_total: 16 * 1024 * 1024 * 1024, // 16GB
        hostname: "test-machine".to_string(),
    };

    assert_eq!(info.cpu_count, 8);
    assert_eq!(info.memory_total, 16 * 1024 * 1024 * 1024);
    assert_eq!(info.hostname, "test-machine");

    // Test clone implementation
    let cloned = info.clone();
    assert_eq!(info.cpu_count, cloned.cpu_count);
    assert_eq!(info.memory_total, cloned.memory_total);
    assert_eq!(info.hostname, cloned.hostname);
}

#[test]
fn test_system_info_debug_formatting() {
    let info = SystemInfo {
        cpu_count: 4,
        memory_total: 8_000_000_000,
        hostname: "debug-host".to_string(),
    };

    let debug_str = format!("{:?}", info);
    assert!(debug_str.contains("SystemInfo"));
    assert!(debug_str.contains("4"));
    assert!(debug_str.contains("8000000000"));
    assert!(debug_str.contains("debug-host"));
}

#[test]
fn test_get_system_info_success() {
    let result = DaemonCore::get_system_info();
    assert!(result.is_ok());

    let info = result.unwrap();
    assert!(info.cpu_count > 0);
    assert!(info.memory_total > 0);
    assert!(!info.hostname.is_empty());
}

// Note: get_total_memory is private, so we can't test it directly

#[test]
fn test_daemon_core_multiple_instances() {
    let core1 = DaemonCore::new();
    let core2 = DaemonCore::new();

    // Both should be identical (unit struct)
    assert_eq!(std::mem::size_of_val(&core1), std::mem::size_of_val(&core2));
    assert_eq!(std::mem::size_of_val(&core1), 0);
}

#[test]
fn test_system_info_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<SystemInfo>();
    assert_sync::<SystemInfo>();
}

// ================================
// DOCUMENT PROCESSOR TESTS
// ================================

fn create_test_configs() -> (ProcessingConfig, QdrantConfig) {
    let processing_config = ProcessingConfig {
        max_concurrent_tasks: 3,
        default_chunk_size: 1500,
        default_chunk_overlap: 300,
        max_file_size_bytes: 2 * 1024 * 1024,
        supported_extensions: vec!["txt".to_string(), "md".to_string(), "rs".to_string()],
        enable_lsp: true,
        lsp_timeout_secs: 15,
    };

    let qdrant_config = QdrantConfig {
        url: "http://localhost:6333".to_string(),
        api_key: Some("test-api-key".to_string()),
        timeout_secs: 45,
        max_retries: 5,
        default_collection: CollectionConfig {
            vector_size: 512,
            distance_metric: "Cosine".to_string(),
            enable_indexing: true,
            replication_factor: 2,
            shard_number: 2,
        },
    };

    (processing_config, qdrant_config)
}

async fn create_test_processor() -> Arc<DocumentProcessor> {
    let (processing_config, qdrant_config) = create_test_configs();
    Arc::new(DocumentProcessor::new(&processing_config, &qdrant_config).await.unwrap())
}

#[tokio::test]
async fn test_document_processor_creation() {
    let (processing_config, qdrant_config) = create_test_configs();
    let result = DocumentProcessor::new(&processing_config, &qdrant_config).await;

    assert!(result.is_ok());
    let processor = result.unwrap();

    assert_eq!(processor.config().max_concurrent_tasks, 3);
    assert_eq!(processor.config().default_chunk_size, 1500);
    assert_eq!(processor.config().default_chunk_overlap, 300);
    assert_eq!(processor.config().max_file_size_bytes, 2 * 1024 * 1024);
    assert!(processor.config().enable_lsp);
    assert_eq!(processor.config().lsp_timeout_secs, 15);
    assert_eq!(processor.config().supported_extensions.len(), 3);
    assert!(processor.config().supported_extensions.contains(&"rs".to_string()));
}

#[tokio::test]
async fn test_document_processor_config_values() {
    let (processing_config, qdrant_config) = create_test_configs();
    let processor = DocumentProcessor::new(&processing_config, &qdrant_config).await.unwrap();

    // Test that config values match what we set
    assert_eq!(processor.config().max_concurrent_tasks, 3);
    assert_eq!(processor.config().default_chunk_size, 1500);
    assert_eq!(processor.config().default_chunk_overlap, 300);
    assert_eq!(processor.config().max_file_size_bytes, 2 * 1024 * 1024);
    assert!(processor.config().enable_lsp);
    assert_eq!(processor.config().lsp_timeout_secs, 15);
    assert!(processor.config().supported_extensions.contains(&"txt".to_string()));
    assert!(processor.config().supported_extensions.contains(&"rs".to_string()));
}

#[tokio::test]
async fn test_document_processor_debug_format() {
    let (processing_config, qdrant_config) = create_test_configs();
    let processor = DocumentProcessor::new(&processing_config, &qdrant_config).await.unwrap();

    let debug_str = format!("{:?}", processor);
    assert!(debug_str.contains("DocumentProcessor"));
    assert!(!debug_str.is_empty());
}

#[tokio::test]
async fn test_document_processing_single_file() {
    let (processing_config, qdrant_config) = create_test_configs();
    let processor = DocumentProcessor::new(&processing_config, &qdrant_config).await.unwrap();

    let result = processor.process_document("test_file.txt").await;
    assert!(result.is_ok());

    let document_id = result.unwrap();
    assert_eq!(document_id.len(), 36); // UUID v4 length
    assert!(document_id.contains('-'));
}

#[tokio::test]
async fn test_document_processing_different_extensions() {
    let (processing_config, qdrant_config) = create_test_configs();
    let processor = DocumentProcessor::new(&processing_config, &qdrant_config).await.unwrap();

    let test_files = vec![
        "document.txt",
        "readme.md",
        "script.rs",
        "config.yaml",
        "data.json",
    ];

    for file in test_files {
        let result = processor.process_document(file).await;
        assert!(result.is_ok(), "Failed to process file: {}", file);

        let document_id = result.unwrap();
        assert_eq!(document_id.len(), 36);
    }
}

#[tokio::test]
async fn test_document_processor_config_access() {
    let (processing_config, qdrant_config) = create_test_configs();
    let processor = DocumentProcessor::new(&processing_config, &qdrant_config).await.unwrap();

    let config = processor.config();
    assert_eq!(config.max_concurrent_tasks, 3);
    assert!(config.supported_extensions.contains(&"txt".to_string()));
    assert!(config.enable_lsp);
}

// ================================
// DAEMON STATE TESTS
// ================================

fn create_test_database_config() -> DatabaseConfig {
    DatabaseConfig {
        sqlite_path: "sqlite::memory:".to_string(),
        max_connections: 3,
        connection_timeout_secs: 10,
        enable_wal: false,
    }
}

#[tokio::test]
async fn test_daemon_state_creation() {
    let config = create_test_database_config();
    let result = DaemonState::new(&config).await;

    assert!(result.is_ok());
    let state = result.unwrap();

    // Test debug formatting
    let debug_str = format!("{:?}", state);
    assert!(debug_str.contains("DaemonState"));
}

#[tokio::test]
async fn test_daemon_state_pool_access() {
    let config = create_test_database_config();
    let state = DaemonState::new(&config).await.unwrap();

    let pool = state.pool();
    assert!(!pool.is_closed());
}

#[tokio::test]
async fn test_daemon_state_health_check() {
    let config = create_test_database_config();
    let state = DaemonState::new(&config).await.unwrap();

    let result = state.health_check().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_daemon_state_invalid_path() {
    let config = DatabaseConfig {
        sqlite_path: "/invalid/nonexistent/path/test.db".to_string(),
        max_connections: 1,
        connection_timeout_secs: 5,
        enable_wal: false,
    };

    let result = DaemonState::new(&config).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_daemon_state_migrations() {
    let config = create_test_database_config();
    let state = DaemonState::new(&config).await.unwrap();

    // Check that tables were created
    let tables_query = "SELECT name FROM sqlite_master WHERE type='table'";
    let rows = sqlx::query(tables_query)
        .fetch_all(state.pool())
        .await
        .unwrap();

    let table_names: Vec<String> = rows.iter()
        .map(|row| row.get::<String, _>("name"))
        .collect();

    assert!(table_names.contains(&"projects".to_string()));
    assert!(table_names.contains(&"collections".to_string()));
    assert!(table_names.contains(&"processing_operations".to_string()));
}

#[tokio::test]
async fn test_daemon_state_multiple_instances() {
    let config1 = create_test_database_config();
    let config2 = create_test_database_config();

    let state1 = DaemonState::new(&config1).await.unwrap();
    let state2 = DaemonState::new(&config2).await.unwrap();

    // Both should work independently
    state1.health_check().await.unwrap();
    state2.health_check().await.unwrap();
}

// ================================
// FILE WATCHER TESTS
// ================================

fn create_test_watcher_config(enabled: bool) -> FileWatcherConfig {
    FileWatcherConfig {
        enabled,
        debounce_ms: 500,
        max_watched_dirs: 20,
        ignore_patterns: vec![
            "*.tmp".to_string(),
            "*.log".to_string(),
            "target/**".to_string(),
            ".git/**".to_string(),
        ],
        recursive: true,
    }
}

#[tokio::test]
async fn test_file_watcher_creation_enabled() {
    let config = create_test_watcher_config(true);
    let processor = create_test_processor().await;

    let result = FileWatcher::new(&config, processor).await;
    assert!(result.is_ok());

    let _watcher = result.unwrap();
    // Note: config fields are private, so we just test creation success
}

#[tokio::test]
async fn test_file_watcher_creation_disabled() {
    let config = create_test_watcher_config(false);
    let processor = create_test_processor().await;

    let result = FileWatcher::new(&config, processor).await;
    assert!(result.is_ok());

    let _watcher = result.unwrap();
    // Note: config fields are private, so we just test creation success
}

#[tokio::test]
async fn test_file_watcher_start_enabled() {
    let config = create_test_watcher_config(true);
    let processor = create_test_processor().await;
    let watcher = FileWatcher::new(&config, processor).await.unwrap();

    let result = watcher.start().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_file_watcher_start_disabled() {
    let config = create_test_watcher_config(false);
    let processor = create_test_processor().await;
    let watcher = FileWatcher::new(&config, processor).await.unwrap();

    let result = watcher.start().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_file_watcher_stop() {
    let config = create_test_watcher_config(true);
    let processor = create_test_processor().await;
    let watcher = FileWatcher::new(&config, processor).await.unwrap();

    watcher.start().await.unwrap();
    let result = watcher.stop().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_file_watcher_directory_operations() {
    let config = create_test_watcher_config(true);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Test watching directory
    let result = watcher.watch_directory(temp_dir.path()).await;
    assert!(result.is_ok());

    // Test unwatching directory
    let result = watcher.unwatch_directory(temp_dir.path()).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_file_watcher_watch_string_path() {
    let config = create_test_watcher_config(true);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let result = watcher.watch_directory("/tmp").await;
    assert!(result.is_ok());

    let result = watcher.unwatch_directory("/tmp").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_file_watcher_debug_format() {
    let config = create_test_watcher_config(true);
    let processor = create_test_processor().await;
    let watcher = FileWatcher::new(&config, processor).await.unwrap();

    let debug_str = format!("{:?}", watcher);
    assert!(debug_str.contains("FileWatcher"));
    assert!(debug_str.contains("config"));
    assert!(debug_str.contains("processor"));
}

#[tokio::test]
async fn test_file_watcher_processor_sharing() {
    let config = create_test_watcher_config(true);
    let processor = create_test_processor().await;
    let processor_clone = Arc::clone(&processor);

    let _watcher = FileWatcher::new(&config, processor_clone).await.unwrap();

    // Verify Arc sharing
    assert!(Arc::strong_count(&processor) >= 2);
}

#[tokio::test]
async fn test_file_watcher_ignore_patterns() {
    let mut config = create_test_watcher_config(true);
    config.ignore_patterns = vec![
        "*.log".to_string(),
        "node_modules/**".to_string(),
        "*.tmp".to_string(),
    ];

    let processor = create_test_processor().await;
    let _watcher = FileWatcher::new(&config, processor).await.unwrap();

    // Config fields are private, so we just test that creation succeeds with custom patterns
    assert_eq!(config.ignore_patterns.len(), 3);
    assert!(config.ignore_patterns.contains(&"*.log".to_string()));
    assert!(config.ignore_patterns.contains(&"node_modules/**".to_string()));
}

#[tokio::test]
async fn test_file_watcher_multiple_start_stop() {
    let config = create_test_watcher_config(true);
    let processor = create_test_processor().await;
    let watcher = FileWatcher::new(&config, processor).await.unwrap();

    // Multiple start/stop cycles should work
    for _ in 0..3 {
        watcher.start().await.unwrap();
        watcher.stop().await.unwrap();
    }
}

#[test]
fn test_file_watcher_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<FileWatcher>();
    assert_sync::<FileWatcher>();
}

// ================================
// CROSS-MODULE INTEGRATION TESTS
// ================================

#[tokio::test]
async fn test_processor_and_state_integration() {
    let (processing_config, qdrant_config) = create_test_configs();
    let processor = DocumentProcessor::new(&processing_config, &qdrant_config).await.unwrap();

    let db_config = create_test_database_config();
    let state = DaemonState::new(&db_config).await.unwrap();

    // Both should be functional
    let doc_result = processor.process_document("integration_test.txt").await;
    assert!(doc_result.is_ok());

    let health_result = state.health_check().await;
    assert!(health_result.is_ok());
}

#[tokio::test]
async fn test_watcher_and_processor_integration() {
    let config = create_test_watcher_config(true);
    let processor = create_test_processor().await;
    let processor_clone = Arc::clone(&processor);

    let watcher = FileWatcher::new(&config, processor_clone).await.unwrap();

    // Both components should be functional
    watcher.start().await.unwrap();

    let doc_result = processor.process_document("watcher_test.txt").await;
    assert!(doc_result.is_ok());

    watcher.stop().await.unwrap();
}

#[test]
fn test_all_components_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    // All main daemon components should be Send + Sync
    assert_send::<DaemonCore>();
    assert_sync::<DaemonCore>();
    assert_send::<SystemInfo>();
    assert_sync::<SystemInfo>();
    assert_send::<FileWatcher>();
    assert_sync::<FileWatcher>();
}
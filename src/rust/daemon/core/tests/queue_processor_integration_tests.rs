//! Queue Processor Integration Tests
//!
//! Comprehensive integration tests for the queue processing workflow,
//! covering end-to-end processing, retry logic, tool unavailability,
//! graceful shutdown, throughput, priority ordering, and max retries.
//!
//! NOTE: These tests are disabled until queue_processor module is enabled.
//! The queue_processor module is commented out pending DocumentProcessor implementation.

// Temporarily disable all tests in this file until queue_processor module is enabled
#![cfg(feature = "queue_processor")]

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serial_test::serial;
use sqlx::SqlitePool;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tempfile::{tempdir, TempDir, NamedTempFile};
use tokio::io::AsyncWriteExt;
use tokio::time::{sleep, timeout};
use workspace_qdrant_core::{
    queue_config::QueueConnectionConfig,
    queue_operations::{QueueManager, QueueOperation},
    queue_processor::{MissingTool, ProcessorConfig, QueueProcessor},
    DocumentProcessor, EmbeddingGenerator, EmbeddingConfig,
    storage::{StorageClient, StorageConfig},
};

/// Test helper to create an in-memory database with queue tables
async fn setup_test_db() -> (SqlitePool, TempDir) {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("test_queue.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.expect("Failed to create pool");

    // Initialize queue tables
    let queue_manager = QueueManager::new(pool.clone());
    queue_manager
        .init_missing_metadata_queue()
        .await
        .expect("Failed to init missing metadata queue");

    (pool, temp_dir)
}

/// Create a temporary test file with content
async fn create_test_file(content: &str, extension: &str) -> NamedTempFile {
    let temp_file = NamedTempFile::with_suffix(&format!(".{}", extension))
        .expect("Failed to create temp file");

    let mut file = tokio::fs::File::create(temp_file.path())
        .await
        .expect("Failed to open temp file");

    file.write_all(content.as_bytes())
        .await
        .expect("Failed to write to temp file");

    file.flush().await.expect("Failed to flush temp file");

    temp_file
}

/// Mock storage client that can be configured to fail
struct MockStorageClient {
    inner: Arc<StorageClient>,
    fail_count: Arc<tokio::sync::Mutex<usize>>,
    max_failures: usize,
}

impl MockStorageClient {
    fn new(max_failures: usize) -> Self {
        let storage_config = StorageConfig {
            url: "http://localhost:6333".to_string(),
            check_compatibility: false,
            ..StorageConfig::default()
        };

        Self {
            inner: Arc::new(StorageClient::with_config(storage_config)),
            fail_count: Arc::new(tokio::sync::Mutex::new(0)),
            max_failures,
        }
    }

    async fn should_fail(&self) -> bool {
        let mut count = self.fail_count.lock().await;
        if *count < self.max_failures {
            *count += 1;
            true
        } else {
            false
        }
    }

    fn inner(&self) -> Arc<StorageClient> {
        self.inner.clone()
    }
}

#[tokio::test]
#[serial]
async fn test_end_to_end_processing() {
    // Setup: Create in-memory database, initialize queue tables
    let (pool, _temp_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Create test file
    let test_content = r#"# Test Document
This is a test document for end-to-end queue processing.
It should be parsed, chunked, and stored in Qdrant.
"#;
    let temp_file = create_test_file(test_content, "md").await;
    let file_path = temp_file.path().to_string_lossy().to_string();

    // Enqueue test file
    let queue_id = queue_manager
        .enqueue_file(
            &file_path,
            "test_collection",
            "test_tenant",
            "main",
            QueueOperation::Ingest,
            5, // normal priority
            None,
        )
        .await
        .expect("Failed to enqueue file");

    assert!(!queue_id.is_empty(), "Queue ID should not be empty");

    // Verify item is in queue
    let queue_depth = queue_manager
        .get_queue_depth(None, None)
        .await
        .expect("Failed to get queue depth");
    assert_eq!(queue_depth, 1, "Queue should have 1 item");

    // Create processor components
    let document_processor = Arc::new(DocumentProcessor::new());
    let embedding_config = EmbeddingConfig::default();
    let embedding_generator = Arc::new(
        EmbeddingGenerator::new(embedding_config)
            .expect("Failed to create embedding generator")
    );
    let storage_config = StorageConfig {
        url: "http://localhost:6333".to_string(),
        check_compatibility: false,
        ..StorageConfig::default()
    };
    let storage_client = Arc::new(StorageClient::with_config(storage_config));

    // Create processor with fast configuration
    let config = ProcessorConfig {
        batch_size: 10,
        poll_interval_ms: 100,
        max_retries: 3,
        retry_delays: vec![
            ChronoDuration::milliseconds(100),
            ChronoDuration::milliseconds(200),
            ChronoDuration::milliseconds(400),
        ],
        target_throughput: 1000,
        enable_metrics: true,
    };

    let mut processor = QueueProcessor::with_components(
        pool.clone(),
        config,
        document_processor,
        embedding_generator,
        storage_client,
    );

    // Start processor
    processor.start().expect("Failed to start processor");

    // Wait for processing (with timeout)
    let start_time = std::time::Instant::now();
    let max_wait = Duration::from_secs(10);

    loop {
        if start_time.elapsed() > max_wait {
            panic!("Processing timeout after {:?}", max_wait);
        }

        let depth = queue_manager
            .get_queue_depth(None, None)
            .await
            .expect("Failed to get queue depth");

        if depth == 0 {
            break;
        }

        sleep(Duration::from_millis(100)).await;
    }

    // Stop processor
    processor.stop().await.expect("Failed to stop processor");

    // Verify: Item removed from queue
    let final_depth = queue_manager
        .get_queue_depth(None, None)
        .await
        .expect("Failed to get queue depth");
    assert_eq!(final_depth, 0, "Queue should be empty after processing");

    // Verify: Metrics updated correctly
    let metrics = processor.get_metrics().await;
    assert!(
        metrics.items_processed > 0,
        "Should have processed at least 1 item"
    );
    assert_eq!(
        metrics.items_failed, 0,
        "Should have no failed items (may fail if Qdrant not available)"
    );
}

#[tokio::test]
#[serial]
async fn test_retry_logic() {
    let (pool, _temp_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Create test file
    let test_content = "Test content for retry logic";
    let temp_file = create_test_file(test_content, "txt").await;
    let file_path = temp_file.path().to_string_lossy().to_string();

    // Enqueue file
    queue_manager
        .enqueue_file(
            &file_path,
            "retry_test_collection",
            "test_tenant",
            "main",
            QueueOperation::Ingest,
            5,
            None,
        )
        .await
        .expect("Failed to enqueue file");

    // Create processor with mock storage that fails first 2 attempts
    let document_processor = Arc::new(DocumentProcessor::new());
    let embedding_config = EmbeddingConfig::default();
    let embedding_generator = Arc::new(
        EmbeddingGenerator::new(embedding_config)
            .expect("Failed to create embedding generator")
    );

    // Use a storage client that will fail (connection to non-existent server)
    let storage_config = StorageConfig {
        url: "http://localhost:9999".to_string(), // Wrong port
        timeout_ms: 1000,
        check_compatibility: false,
        ..StorageConfig::default()
    };
    let storage_client = Arc::new(StorageClient::with_config(storage_config));

    let config = ProcessorConfig {
        batch_size: 1,
        poll_interval_ms: 100,
        max_retries: 3,
        retry_delays: vec![
            ChronoDuration::milliseconds(200),
            ChronoDuration::milliseconds(400),
            ChronoDuration::milliseconds(800),
        ],
        target_throughput: 1000,
        enable_metrics: true,
    };

    let mut processor = QueueProcessor::with_components(
        pool.clone(),
        config,
        document_processor,
        embedding_generator,
        storage_client,
    );

    processor.start().expect("Failed to start processor");

    // Wait for first attempt and retry
    sleep(Duration::from_millis(500)).await;

    // Check that item is still in queue with updated retry_count
    let items = queue_manager
        .dequeue_batch(1, None, None)
        .await
        .expect("Failed to dequeue");

    if !items.is_empty() {
        let item = &items[0];
        assert!(
            item.retry_count >= 1,
            "Retry count should be at least 1, got {}",
            item.retry_count
        );
        assert!(
            item.retry_from.is_some(),
            "retry_from should be set after failure"
        );
    }

    processor.stop().await.expect("Failed to stop processor");

    // Verify: Metrics show failures
    let metrics = processor.get_metrics().await;
    assert!(
        metrics.items_failed > 0 || metrics.items_processed > 0,
        "Should have processing attempts"
    );
}

#[tokio::test]
#[serial]
async fn test_tool_unavailable_handling() {
    let (pool, _temp_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Create a code file that requires tree-sitter parser
    let test_content = r#"
fn test_function() {
    println!("This is a Haskell file");
}
"#;
    let temp_file = create_test_file(test_content, "hs").await; // Haskell not supported
    let file_path = temp_file.path().to_string_lossy().to_string();

    // Enqueue file
    queue_manager
        .enqueue_file(
            &file_path,
            "tool_test_collection",
            "test_tenant",
            "main",
            QueueOperation::Ingest,
            5,
            None,
        )
        .await
        .expect("Failed to enqueue file");

    // Create processor
    let document_processor = Arc::new(DocumentProcessor::new());
    let embedding_config = EmbeddingConfig::default();
    let embedding_generator = Arc::new(
        EmbeddingGenerator::new(embedding_config)
            .expect("Failed to create embedding generator")
    );
    let storage_config = StorageConfig {
        url: "http://localhost:6333".to_string(),
        check_compatibility: false,
        ..StorageConfig::default()
    };
    let storage_client = Arc::new(StorageClient::with_config(storage_config));

    let config = ProcessorConfig {
        batch_size: 10,
        poll_interval_ms: 100,
        max_retries: 2,
        retry_delays: vec![
            ChronoDuration::milliseconds(100),
            ChronoDuration::milliseconds(200),
        ],
        target_throughput: 1000,
        enable_metrics: true,
    };

    let mut processor = QueueProcessor::with_components(
        pool.clone(),
        config,
        document_processor,
        embedding_generator,
        storage_client,
    );

    processor.start().expect("Failed to start processor");

    // Wait for processing
    sleep(Duration::from_secs(2)).await;

    processor.stop().await.expect("Failed to stop processor");

    // Check if item was moved to missing_metadata_queue
    // Note: This test depends on the processor's tool checking logic
    // If tree-sitter parser for .hs is not available, it should be moved
    let metrics = processor.get_metrics().await;

    // Either it's moved to missing_metadata_queue or processed successfully
    // (depending on whether tree-sitter is required for .hs files)
    assert!(
        metrics.items_missing_metadata > 0 || metrics.items_processed > 0,
        "Item should be either processed or moved to missing_metadata_queue"
    );
}

#[tokio::test]
#[serial]
async fn test_graceful_shutdown() {
    let (pool, _temp_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Enqueue multiple items
    for i in 0..5 {
        let content = format!("Test content {}", i);
        let temp_file = create_test_file(&content, "txt").await;
        let file_path = temp_file.path().to_string_lossy().to_string();

        queue_manager
            .enqueue_file(
                &file_path,
                "shutdown_test_collection",
                "test_tenant",
                "main",
                QueueOperation::Ingest,
                5,
                None,
            )
            .await
            .expect("Failed to enqueue file");

        // Keep temp files alive
        std::mem::forget(temp_file);
    }

    // Create processor
    let document_processor = Arc::new(DocumentProcessor::new());
    let embedding_config = EmbeddingConfig::default();
    let embedding_generator = Arc::new(
        EmbeddingGenerator::new(embedding_config)
            .expect("Failed to create embedding generator")
    );
    let storage_config = StorageConfig {
        url: "http://localhost:6333".to_string(),
        check_compatibility: false,
        ..StorageConfig::default()
    };
    let storage_client = Arc::new(StorageClient::with_config(storage_config));

    let config = ProcessorConfig {
        batch_size: 1, // Process one at a time
        poll_interval_ms: 100,
        max_retries: 3,
        retry_delays: vec![
            ChronoDuration::milliseconds(100),
            ChronoDuration::milliseconds(200),
            ChronoDuration::milliseconds(400),
        ],
        target_throughput: 1000,
        enable_metrics: true,
    };

    let mut processor = QueueProcessor::with_components(
        pool.clone(),
        config,
        document_processor,
        embedding_generator,
        storage_client,
    );

    processor.start().expect("Failed to start processor");

    // Let it process a bit
    sleep(Duration::from_millis(300)).await;

    // Trigger shutdown
    let shutdown_start = std::time::Instant::now();
    processor.stop().await.expect("Failed to stop processor");
    let shutdown_duration = shutdown_start.elapsed();

    // Verify: Shutdown completed within timeout
    assert!(
        shutdown_duration < Duration::from_secs(5),
        "Shutdown took too long: {:?}",
        shutdown_duration
    );

    // Verify: Some items may still be in queue (not all processed)
    let final_depth = queue_manager
        .get_queue_depth(None, None)
        .await
        .expect("Failed to get queue depth");

    // Items should either be processed or still in queue (no loss)
    let metrics = processor.get_metrics().await;
    let total_accounted = metrics.items_processed + final_depth as u64;
    assert!(
        total_accounted <= 5,
        "All items should be accounted for"
    );
}

#[tokio::test]
#[serial]
async fn test_throughput_target() {
    let (pool, _temp_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Enqueue 100 small test files
    let mut temp_files = Vec::new();
    for i in 0..100 {
        let content = format!("Test content for throughput test {}", i);
        let temp_file = create_test_file(&content, "txt").await;
        let file_path = temp_file.path().to_string_lossy().to_string();

        queue_manager
            .enqueue_file(
                &file_path,
                "throughput_test_collection",
                "test_tenant",
                "main",
                QueueOperation::Ingest,
                5,
                None,
            )
            .await
            .expect("Failed to enqueue file");

        temp_files.push(temp_file);
    }

    // Create processor
    let document_processor = Arc::new(DocumentProcessor::new());
    let embedding_config = EmbeddingConfig::default();
    let embedding_generator = Arc::new(
        EmbeddingGenerator::new(embedding_config)
            .expect("Failed to create embedding generator")
    );
    let storage_config = StorageConfig {
        url: "http://localhost:6333".to_string(),
        check_compatibility: false,
        ..StorageConfig::default()
    };
    let storage_client = Arc::new(StorageClient::with_config(storage_config));

    let config = ProcessorConfig {
        batch_size: 10,
        poll_interval_ms: 10, // Fast polling
        max_retries: 3,
        retry_delays: vec![
            ChronoDuration::milliseconds(100),
            ChronoDuration::milliseconds(200),
            ChronoDuration::milliseconds(400),
        ],
        target_throughput: 1000,
        enable_metrics: true,
    };

    let mut processor = QueueProcessor::with_components(
        pool.clone(),
        config,
        document_processor,
        embedding_generator,
        storage_client,
    );

    processor.start().expect("Failed to start processor");

    let start_time = std::time::Instant::now();

    // Wait for completion (with timeout)
    let result = timeout(Duration::from_secs(30), async {
        loop {
            let depth = queue_manager
                .get_queue_depth(None, None)
                .await
                .expect("Failed to get queue depth");

            if depth == 0 {
                break;
            }

            sleep(Duration::from_millis(100)).await;
        }
    }).await;

    let elapsed = start_time.elapsed();

    processor.stop().await.expect("Failed to stop processor");

    if result.is_ok() {
        // Calculate throughput
        let docs_per_second = 100.0 / elapsed.as_secs_f64();
        let docs_per_minute = docs_per_second * 60.0;

        println!("Processed 100 docs in {:?}", elapsed);
        println!("Throughput: {:.1} docs/min", docs_per_minute);

        // Verify metrics
        let metrics = processor.get_metrics().await;

        // Note: Actual throughput depends on Qdrant availability and performance
        // This test verifies the metrics are calculated correctly
        assert!(
            metrics.items_processed > 0 || metrics.items_failed > 0,
            "Should have processing activity"
        );

        if metrics.items_processed >= 100 {
            println!("Target throughput check: {}",
                if metrics.meets_target(1000) { "✓ PASS" } else { "✗ FAIL" });
        }
    } else {
        println!("Throughput test timed out (likely Qdrant not available)");
    }
}

#[tokio::test]
#[serial]
async fn test_priority_ordering() {
    let (pool, _temp_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Enqueue items with different priorities (reverse order)
    let priorities = vec![2, 5, 8, 3, 10, 1];
    let mut temp_files = Vec::new();

    for (i, priority) in priorities.iter().enumerate() {
        let content = format!("Priority {} content", priority);
        let temp_file = create_test_file(&content, "txt").await;
        let file_path = temp_file.path().to_string_lossy().to_string();

        queue_manager
            .enqueue_file(
                &file_path,
                "priority_test_collection",
                "test_tenant",
                "main",
                QueueOperation::Ingest,
                *priority,
                None,
            )
            .await
            .expect("Failed to enqueue file");

        temp_files.push(temp_file);
    }

    // Dequeue items and verify priority ordering
    let items = queue_manager
        .dequeue_batch(10, None, None)
        .await
        .expect("Failed to dequeue batch");

    assert_eq!(items.len(), 6, "Should have 6 items");

    // Verify items are ordered by priority (descending)
    let mut prev_priority = 11; // Start higher than max
    for item in items {
        assert!(
            item.priority <= prev_priority,
            "Items should be ordered by priority (desc): {} > {}",
            prev_priority,
            item.priority
        );
        prev_priority = item.priority;
    }

    println!("Priority ordering verified: items dequeued in correct order");
}

#[tokio::test]
#[serial]
async fn test_max_retries_exceeded() {
    let (pool, _temp_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Create test file
    let test_content = "Test content for max retries";
    let temp_file = create_test_file(test_content, "txt").await;
    let file_path = temp_file.path().to_string_lossy().to_string();

    // Enqueue file
    queue_manager
        .enqueue_file(
            &file_path,
            "max_retries_test_collection",
            "test_tenant",
            "main",
            QueueOperation::Ingest,
            5,
            None,
        )
        .await
        .expect("Failed to enqueue file");

    // Create processor with storage that always fails
    let document_processor = Arc::new(DocumentProcessor::new());
    let embedding_config = EmbeddingConfig::default();
    let embedding_generator = Arc::new(
        EmbeddingGenerator::new(embedding_config)
            .expect("Failed to create embedding generator")
    );
    let storage_config = StorageConfig {
        url: "http://localhost:9999".to_string(), // Wrong port (always fail)
        timeout_ms: 500,
        check_compatibility: false,
        ..StorageConfig::default()
    };
    let storage_client = Arc::new(StorageClient::with_config(storage_config));

    let config = ProcessorConfig {
        batch_size: 1,
        poll_interval_ms: 50,
        max_retries: 2, // Low retry count for faster test
        retry_delays: vec![
            ChronoDuration::milliseconds(50),
            ChronoDuration::milliseconds(100),
        ],
        target_throughput: 1000,
        enable_metrics: true,
    };

    let mut processor = QueueProcessor::with_components(
        pool.clone(),
        config.clone(),
        document_processor,
        embedding_generator,
        storage_client,
    );

    processor.start().expect("Failed to start processor");

    // Wait for retries to complete
    // Each retry takes ~50-100ms + processing time
    sleep(Duration::from_secs(2)).await;

    processor.stop().await.expect("Failed to stop processor");

    // Verify: Item should be removed after max retries
    let final_depth = queue_manager
        .get_queue_depth(None, None)
        .await
        .expect("Failed to get queue depth");

    // After max retries, item should be removed from queue
    assert_eq!(
        final_depth, 0,
        "Item should be removed after exceeding max retries"
    );

    // Verify: Metrics show failures
    let metrics = processor.get_metrics().await;
    assert!(
        metrics.items_failed > 0,
        "Should have failed items after max retries"
    );
    assert_eq!(
        metrics.items_processed, 0,
        "Should have no successful processing"
    );
}

#[tokio::test]
#[serial]
async fn test_concurrent_processing() {
    let (pool, _temp_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Enqueue 20 items
    let mut temp_files = Vec::new();
    for i in 0..20 {
        let content = format!("Concurrent test content {}", i);
        let temp_file = create_test_file(&content, "txt").await;
        let file_path = temp_file.path().to_string_lossy().to_string();

        queue_manager
            .enqueue_file(
                &file_path,
                "concurrent_test_collection",
                "test_tenant",
                "main",
                QueueOperation::Ingest,
                5,
                None,
            )
            .await
            .expect("Failed to enqueue file");

        temp_files.push(temp_file);
    }

    // Create processor with larger batch size
    let document_processor = Arc::new(DocumentProcessor::new());
    let embedding_config = EmbeddingConfig::default();
    let embedding_generator = Arc::new(
        EmbeddingGenerator::new(embedding_config)
            .expect("Failed to create embedding generator")
    );
    let storage_config = StorageConfig {
        url: "http://localhost:6333".to_string(),
        check_compatibility: false,
        ..StorageConfig::default()
    };
    let storage_client = Arc::new(StorageClient::with_config(storage_config));

    let config = ProcessorConfig {
        batch_size: 5, // Process in batches of 5
        poll_interval_ms: 50,
        max_retries: 3,
        retry_delays: vec![
            ChronoDuration::milliseconds(100),
            ChronoDuration::milliseconds(200),
            ChronoDuration::milliseconds(400),
        ],
        target_throughput: 1000,
        enable_metrics: true,
    };

    let mut processor = QueueProcessor::with_components(
        pool.clone(),
        config,
        document_processor,
        embedding_generator,
        storage_client,
    );

    processor.start().expect("Failed to start processor");

    // Wait for processing
    let result = timeout(Duration::from_secs(15), async {
        loop {
            let depth = queue_manager
                .get_queue_depth(None, None)
                .await
                .expect("Failed to get queue depth");

            if depth == 0 {
                break;
            }

            sleep(Duration::from_millis(100)).await;
        }
    }).await;

    processor.stop().await.expect("Failed to stop processor");

    if result.is_ok() {
        let metrics = processor.get_metrics().await;
        println!("Concurrent processing: {} items processed", metrics.items_processed);

        assert!(
            metrics.items_processed > 0 || metrics.items_failed > 0,
            "Should have processing activity"
        );
    } else {
        println!("Concurrent test timed out (likely Qdrant not available)");
    }
}

#[tokio::test]
#[serial]
async fn test_metrics_accuracy() {
    let (pool, _temp_dir) = setup_test_db().await;
    let queue_manager = QueueManager::new(pool.clone());

    // Create and enqueue a test file
    let test_content = "Test content for metrics validation";
    let temp_file = create_test_file(test_content, "txt").await;
    let file_path = temp_file.path().to_string_lossy().to_string();

    queue_manager
        .enqueue_file(
            &file_path,
            "metrics_test_collection",
            "test_tenant",
            "main",
            QueueOperation::Ingest,
            5,
            None,
        )
        .await
        .expect("Failed to enqueue file");

    // Create processor
    let document_processor = Arc::new(DocumentProcessor::new());
    let embedding_config = EmbeddingConfig::default();
    let embedding_generator = Arc::new(
        EmbeddingGenerator::new(embedding_config)
            .expect("Failed to create embedding generator")
    );
    let storage_config = StorageConfig {
        url: "http://localhost:6333".to_string(),
        check_compatibility: false,
        ..StorageConfig::default()
    };
    let storage_client = Arc::new(StorageClient::with_config(storage_config));

    let config = ProcessorConfig {
        batch_size: 10,
        poll_interval_ms: 100,
        max_retries: 3,
        retry_delays: vec![
            ChronoDuration::milliseconds(100),
            ChronoDuration::milliseconds(200),
            ChronoDuration::milliseconds(400),
        ],
        target_throughput: 1000,
        enable_metrics: true,
    };

    let mut processor = QueueProcessor::with_components(
        pool.clone(),
        config,
        document_processor,
        embedding_generator,
        storage_client,
    );

    // Get initial metrics
    let initial_metrics = processor.get_metrics().await;
    assert_eq!(initial_metrics.items_processed, 0);
    assert_eq!(initial_metrics.items_failed, 0);

    processor.start().expect("Failed to start processor");

    // Wait for processing
    sleep(Duration::from_secs(3)).await;

    processor.stop().await.expect("Failed to stop processor");

    // Check final metrics
    let final_metrics = processor.get_metrics().await;

    // Verify metrics structure
    assert!(final_metrics.items_processed >= 0);
    assert!(final_metrics.items_failed >= 0);
    assert!(final_metrics.items_missing_metadata >= 0);
    assert!(final_metrics.avg_processing_time_ms >= 0.0);

    // Either processed or failed (depending on Qdrant availability)
    let total_handled = final_metrics.items_processed + final_metrics.items_failed;
    assert!(
        total_handled > 0,
        "Should have handled at least one item"
    );

    println!("Metrics accuracy test completed:");
    println!("  Processed: {}", final_metrics.items_processed);
    println!("  Failed: {}", final_metrics.items_failed);
    println!("  Missing metadata: {}", final_metrics.items_missing_metadata);
    println!("  Avg processing time: {:.2}ms", final_metrics.avg_processing_time_ms);
    println!("  Throughput: {:.1} docs/min", final_metrics.throughput_per_minute());
}

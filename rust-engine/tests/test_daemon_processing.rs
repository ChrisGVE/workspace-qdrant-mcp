//! Comprehensive unit tests for daemon/processing.rs module
//! Tests DocumentProcessor and related functions with proven file-by-file methodology

use std::sync::Arc;
use std::time::{Duration, Instant};
use workspace_qdrant_daemon::config::{ProcessingConfig, QdrantConfig, CollectionConfig};
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;

/// Configuration module for test setup
#[cfg(test)]
mod test_config {
    use super::*;

    /// Create a standard test processing configuration
    pub fn create_test_processing_config() -> ProcessingConfig {
        ProcessingConfig {
            max_concurrent_tasks: 2,
            default_chunk_size: 1000,
            default_chunk_overlap: 200,
            max_file_size_bytes: 1024 * 1024, // 1MB
            supported_extensions: vec![
                "rs".to_string(),
                "py".to_string(),
                "md".to_string(),
                "txt".to_string(),
            ],
            enable_lsp: true,
            lsp_timeout_secs: 10,
        }
    }

    /// Create a variant processing configuration for testing different values
    pub fn create_variant_processing_config() -> ProcessingConfig {
        ProcessingConfig {
            max_concurrent_tasks: 4,
            default_chunk_size: 2000,
            default_chunk_overlap: 400,
            max_file_size_bytes: 10 * 1024 * 1024, // 10MB
            supported_extensions: vec![
                "js".to_string(),
                "ts".to_string(),
                "json".to_string(),
            ],
            enable_lsp: false,
            lsp_timeout_secs: 20,
        }
    }

    /// Create a minimal processing configuration for edge testing
    pub fn create_minimal_processing_config() -> ProcessingConfig {
        ProcessingConfig {
            max_concurrent_tasks: 1,
            default_chunk_size: 100,
            default_chunk_overlap: 20,
            max_file_size_bytes: 1024, // 1KB
            supported_extensions: vec!["txt".to_string()],
            enable_lsp: false,
            lsp_timeout_secs: 5,
        }
    }

    /// Create a standard test qdrant configuration
    pub fn create_test_qdrant_config() -> QdrantConfig {
        QdrantConfig {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            max_retries: 3,
            default_collection: CollectionConfig {
                vector_size: 384,
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                replication_factor: 1,
                shard_number: 1,
            },
        }
    }

    /// Create a qdrant configuration with API key for testing
    pub fn create_qdrant_config_with_key() -> QdrantConfig {
        QdrantConfig {
            url: "https://remote-qdrant.example.com:6333".to_string(),
            api_key: Some("test-api-key-12345".to_string()),
            max_retries: 5,
            default_collection: CollectionConfig {
                vector_size: 768,
                distance_metric: "Euclidean".to_string(),
                enable_indexing: false,
                replication_factor: 2,
                shard_number: 3,
            },
        }
    }

    /// Create an alternative qdrant configuration
    pub fn create_alternative_qdrant_config() -> QdrantConfig {
        QdrantConfig {
            url: "http://test-qdrant:6334".to_string(),
            api_key: Some("alternative-key".to_string()),
            max_retries: 2,
            default_collection: CollectionConfig {
                vector_size: 512,
                distance_metric: "Dot".to_string(),
                enable_indexing: true,
                replication_factor: 3,
                shard_number: 2,
            },
        }
    }
}

/// Basic DocumentProcessor creation and configuration tests
#[cfg(test)]
mod basic_creation_tests {
    use super::*;
    use super::test_config::*;

    #[tokio::test]
    async fn test_document_processor_creation_standard_config() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor");

        // Verify configuration is properly stored
        assert_eq!(processor.config().max_concurrent_tasks, 2);
        assert_eq!(processor.config().default_chunk_size, 1000);
        assert_eq!(processor.config().default_chunk_overlap, 200);
        assert_eq!(processor.config().max_file_size_bytes, 1024 * 1024);
        assert_eq!(processor.config().supported_extensions.len(), 4);
        assert!(processor.config().supported_extensions.contains(&"rs".to_string()));
        assert!(processor.config().supported_extensions.contains(&"py".to_string()));
        assert!(processor.config().enable_lsp);
        assert_eq!(processor.config().lsp_timeout_secs, 10);
    }

    #[tokio::test]
    async fn test_document_processor_creation_variant_config() {
        let processing_config = create_variant_processing_config();
        let qdrant_config = create_qdrant_config_with_key();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor with variant config");

        // Verify variant configuration values
        assert_eq!(processor.config().max_concurrent_tasks, 4);
        assert_eq!(processor.config().default_chunk_size, 2000);
        assert_eq!(processor.config().default_chunk_overlap, 400);
        assert_eq!(processor.config().max_file_size_bytes, 10 * 1024 * 1024);
        assert_eq!(processor.config().supported_extensions.len(), 3);
        assert!(processor.config().supported_extensions.contains(&"js".to_string()));
        assert!(processor.config().supported_extensions.contains(&"ts".to_string()));
        assert!(processor.config().supported_extensions.contains(&"json".to_string()));
        assert!(!processor.config().enable_lsp);
        assert_eq!(processor.config().lsp_timeout_secs, 20);
    }

    #[tokio::test]
    async fn test_document_processor_creation_minimal_config() {
        let processing_config = create_minimal_processing_config();
        let qdrant_config = create_alternative_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor with minimal config");

        // Verify minimal configuration values
        assert_eq!(processor.config().max_concurrent_tasks, 1);
        assert_eq!(processor.config().default_chunk_size, 100);
        assert_eq!(processor.config().default_chunk_overlap, 20);
        assert_eq!(processor.config().max_file_size_bytes, 1024);
        assert_eq!(processor.config().supported_extensions.len(), 1);
        assert!(processor.config().supported_extensions.contains(&"txt".to_string()));
        assert!(!processor.config().enable_lsp);
        assert_eq!(processor.config().lsp_timeout_secs, 5);
    }

    #[tokio::test]
    async fn test_document_processor_debug_format() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor for debug test");

        // Test debug formatting
        let debug_str = format!("{:?}", processor);
        assert!(debug_str.contains("DocumentProcessor"));
        assert!(!debug_str.is_empty());
    }

    #[tokio::test]
    async fn test_document_processor_config_access() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor for config access test");

        // Test config access returns correct reference
        let config = processor.config();
        assert_eq!(config.max_concurrent_tasks, processing_config.max_concurrent_tasks);
        assert_eq!(config.default_chunk_size, processing_config.default_chunk_size);
        assert_eq!(config.default_chunk_overlap, processing_config.default_chunk_overlap);
        assert_eq!(config.max_file_size_bytes, processing_config.max_file_size_bytes);
        assert_eq!(config.supported_extensions, processing_config.supported_extensions);
        assert_eq!(config.enable_lsp, processing_config.enable_lsp);
        assert_eq!(config.lsp_timeout_secs, processing_config.lsp_timeout_secs);
    }

    #[tokio::test]
    async fn test_document_processor_multiple_configs() {
        let config1 = create_test_processing_config();
        let config2 = create_variant_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor1 = DocumentProcessor::new(&config1, &qdrant_config)
            .await
            .expect("Failed to create first DocumentProcessor");

        let processor2 = DocumentProcessor::new(&config2, &qdrant_config)
            .await
            .expect("Failed to create second DocumentProcessor");

        // Verify processors have different configurations
        assert_ne!(processor1.config().max_concurrent_tasks, processor2.config().max_concurrent_tasks);
        assert_ne!(processor1.config().default_chunk_size, processor2.config().default_chunk_size);
        assert_ne!(processor1.config().supported_extensions, processor2.config().supported_extensions);
    }
}

/// Document processing workflow tests
#[cfg(test)]
mod document_processing_tests {
    use super::*;
    use super::test_config::*;

    #[tokio::test]
    async fn test_process_single_document_success() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor");

        let result = processor.process_document("test_file.rs").await;
        assert!(result.is_ok(), "Document processing should succeed");

        let document_id = result.unwrap();
        // UUID v4 is 36 characters with 4 hyphens
        assert_eq!(document_id.len(), 36);
        assert_eq!(document_id.matches('-').count(), 4);
        
        // Verify it's a valid UUID format
        assert!(uuid::Uuid::parse_str(&document_id).is_ok());
    }

    #[tokio::test]
    async fn test_process_multiple_different_files() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor");

        let test_files = vec![
            "main.rs",
            "script.py", 
            "README.md",
            "config.txt",
            "data.json",
            "index.html",
            "style.css",
        ];

        let mut results = Vec::new();
        for file in &test_files {
            let result = processor.process_document(file).await;
            assert!(result.is_ok(), "Failed to process file: {}", file);
            results.push(result.unwrap());
        }

        // All results should be valid UUIDs
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.len(), 36, "Invalid UUID length for file: {}", test_files[i]);
            assert!(uuid::Uuid::parse_str(result).is_ok(), "Invalid UUID format for file: {}", test_files[i]);
        }

        // All UUIDs should be unique
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                assert_ne!(results[i], results[j], "UUIDs should be unique");
            }
        }
    }

    #[tokio::test]
    async fn test_process_document_with_special_characters() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor");

        let special_files = vec![
            "file with spaces.txt",
            "file-with-dashes.md",
            "file_with_underscores.py",
            "file.with.dots.rs",
            "file@with@symbols.txt",
            "файл-с-unicode.txt", // Cyrillic characters
            "文件.txt", // Chinese characters
        ];

        for file in &special_files {
            let result = processor.process_document(file).await;
            assert!(result.is_ok(), "Failed to process file with special characters: {}", file);
            
            let document_id = result.unwrap();
            assert_eq!(document_id.len(), 36);
            assert!(uuid::Uuid::parse_str(&document_id).is_ok());
        }
    }

    #[tokio::test]
    async fn test_process_document_with_long_path() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor");

        // Create a very long file path
        let long_path = format!(
            "{}/test_file.rs",
            "very/deep/nested/directory/structure/with/many/levels/of/subdirectories/that/goes/on/and/on/and/on/for/a/very/long/time/to/test/edge/cases"
        );

        let result = processor.process_document(&long_path).await;
        assert!(result.is_ok(), "Should handle long file paths");
        
        let document_id = result.unwrap();
        assert_eq!(document_id.len(), 36);
        assert!(uuid::Uuid::parse_str(&document_id).is_ok());
    }

    #[tokio::test]
    async fn test_process_document_empty_filename() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor");

        let result = processor.process_document("").await;
        assert!(result.is_ok(), "Should handle empty filename");
        
        let document_id = result.unwrap();
        assert_eq!(document_id.len(), 36);
        assert!(uuid::Uuid::parse_str(&document_id).is_ok());
    }
}

/// Concurrent processing and semaphore tests
#[cfg(test)]
mod concurrency_tests {
    use super::*;
    use super::test_config::*;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_concurrent_processing_within_limits() {
        let processing_config = create_test_processing_config(); // max_concurrent_tasks = 2
        let qdrant_config = create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .expect("Failed to create DocumentProcessor")
        );

        let mut handles = vec![];
        let start_time = Instant::now();

        // Spawn 4 tasks (more than the limit of 2)
        for i in 0..4 {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                processor_clone
                    .process_document(&format!("concurrent_test_{}.rs", i))
                    .await
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results = futures_util::future::join_all(handles).await;
        let elapsed = start_time.elapsed();

        // All tasks should complete successfully
        for (i, result) in results.iter().enumerate() {
            let task_result = result.as_ref().expect(&format!("Task {} panicked", i));
            assert!(task_result.is_ok(), "Task {} failed: {:?}", i, task_result);
            
            let document_id = task_result.as_ref().unwrap();
            assert_eq!(document_id.len(), 36);
            assert!(uuid::Uuid::parse_str(document_id).is_ok());
        }

        // With max_concurrent_tasks = 2 and 4 tasks taking ~100ms each,
        // total time should be at least 200ms (two batches)
        assert!(elapsed >= Duration::from_millis(180), "Tasks should be limited by semaphore, elapsed: {:?}", elapsed);
    }

    #[tokio::test]
    async fn test_sequential_processing_with_single_limit() {
        let mut processing_config = create_minimal_processing_config();
        processing_config.max_concurrent_tasks = 1; // Force sequential processing
        let qdrant_config = create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .expect("Failed to create DocumentProcessor")
        );

        let start_time = Instant::now();
        let task_count = 3;
        let mut handles = vec![];

        // Spawn multiple tasks that should run sequentially
        for i in 0..task_count {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                processor_clone
                    .process_document(&format!("sequential_test_{}.rs", i))
                    .await
            });
            handles.push(handle);
        }

        let results = futures_util::future::join_all(handles).await;
        let elapsed = start_time.elapsed();

        // All tasks should succeed
        for (i, result) in results.iter().enumerate() {
            let task_result = result.as_ref().expect(&format!("Task {} panicked", i));
            assert!(task_result.is_ok(), "Task {} failed", i);
        }

        // With 3 tasks running sequentially, each taking ~100ms, total should be ~300ms
        assert!(elapsed >= Duration::from_millis(250), "Sequential tasks should take longer, elapsed: {:?}", elapsed);
    }

    #[tokio::test]
    async fn test_high_concurrency_processing() {
        let mut processing_config = create_variant_processing_config();
        processing_config.max_concurrent_tasks = 8; // High concurrency
        let qdrant_config = create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .expect("Failed to create DocumentProcessor")
        );

        let task_count = 16;
        let mut handles = vec![];
        let start_time = Instant::now();

        // Spawn many concurrent tasks
        for i in 0..task_count {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                processor_clone
                    .process_document(&format!("high_concurrency_test_{}.rs", i))
                    .await
            });
            handles.push(handle);
        }

        let results = futures_util::future::join_all(handles).await;
        let elapsed = start_time.elapsed();

        // All tasks should succeed
        for (i, result) in results.iter().enumerate() {
            let task_result = result.as_ref().expect(&format!("Task {} panicked", i));
            assert!(task_result.is_ok(), "Task {} failed", i);
        }

        // With high concurrency (8), 16 tasks should complete faster than sequential
        // Should complete in roughly 2 batches: ~200ms
        assert!(elapsed <= Duration::from_millis(400), "High concurrency should be faster, elapsed: {:?}", elapsed);
    }

    #[tokio::test]
    async fn test_semaphore_limits_concurrent_execution() {
        let processing_config = create_test_processing_config(); // max_concurrent_tasks = 2
        let qdrant_config = create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .expect("Failed to create DocumentProcessor")
        );

        let start_time = Instant::now();
        let task_count = 4;
        let mut handles = vec![];

        // Spawn multiple tasks that should be limited by the semaphore
        for i in 0..task_count {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                let start = Instant::now();
                let result = processor_clone
                    .process_document(&format!("semaphore_limit_test_{}.rs", i))
                    .await;
                let elapsed = start.elapsed();
                (i, result, elapsed)
            });
            handles.push(handle);
        }

        let results = futures_util::future::join_all(handles).await;
        let total_elapsed = start_time.elapsed();

        // All tasks should succeed
        for result in &results {
            let (task_id, process_result, _) = result.as_ref().expect("Task should not panic");
            assert!(process_result.is_ok(), "Task {} failed", task_id);
        }

        // With max_concurrent_tasks = 2 and 4 tasks, execution should take longer
        // than if all tasks ran concurrently (which would be ~100ms)
        // It should take at least 200ms (two batches of concurrent execution)
        assert!(total_elapsed >= Duration::from_millis(180),
            "Semaphore should limit concurrency, total time: {:?}", total_elapsed);

        // Verify all UUIDs are valid and unique
        let mut document_ids = vec![];
        for result in results {
            let (_, process_result, _) = result.expect("Task should not panic");
            let document_id = process_result.unwrap();
            assert_eq!(document_id.len(), 36);
            assert!(uuid::Uuid::parse_str(&document_id).is_ok());
            document_ids.push(document_id);
        }

        // All UUIDs should be unique
        for i in 0..document_ids.len() {
            for j in (i + 1)..document_ids.len() {
                assert_ne!(document_ids[i], document_ids[j], "All UUIDs should be unique");
            }
        }
    }
}

/// Error handling and edge case tests
#[cfg(test)]
mod error_handling_tests {
    use super::*;
    use super::test_config::*;

    #[tokio::test]
    async fn test_process_document_timing_consistency() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor");

        let mut durations = vec![];
        
        // Process the same document multiple times to check timing consistency
        for i in 0..5 {
            let start = Instant::now();
            let result = processor.process_document(&format!("timing_test_{}.rs", i)).await;
            let duration = start.elapsed();
            
            assert!(result.is_ok(), "Document processing should succeed");
            durations.push(duration);
        }

        // All processing times should be reasonably consistent (within a range)
        // Current implementation sleeps for 100ms, so durations should be around that
        for (i, duration) in durations.iter().enumerate() {
            assert!(
                duration >= &Duration::from_millis(95) && duration <= &Duration::from_millis(150),
                "Duration {} ({:?}) is outside expected range", i, duration
            );
        }
    }

    #[tokio::test]
    async fn test_document_processor_stress_test() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .expect("Failed to create DocumentProcessor")
        );

        let stress_test_count = 20;
        let mut handles = vec![];

        // Create many concurrent tasks to stress test the processor
        for i in 0..stress_test_count {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                // Process multiple documents per task
                let mut results = vec![];
                for j in 0..3 {
                    let file_path = format!("stress_test_{}_{}.rs", i, j);
                    let result = processor_clone.process_document(&file_path).await;
                    results.push((file_path, result));
                }
                results
            });
            handles.push(handle);
        }

        let task_results = futures_util::future::join_all(handles).await;

        // Verify all stress test tasks completed successfully
        let mut total_processed = 0;
        for (task_id, task_result) in task_results.iter().enumerate() {
            let document_results = task_result.as_ref().expect(&format!("Stress test task {} panicked", task_id));
            
            for (file_path, result) in document_results {
                assert!(result.is_ok(), "Failed to process {} in stress test", file_path);
                total_processed += 1;
                
                let document_id = result.as_ref().unwrap();
                assert_eq!(document_id.len(), 36);
                assert!(uuid::Uuid::parse_str(document_id).is_ok());
            }
        }

        assert_eq!(total_processed, stress_test_count * 3, "Not all documents were processed");
    }

    #[tokio::test]
    async fn test_processor_with_extreme_configurations() {
        // Test with extremely small configuration
        let mut tiny_config = create_minimal_processing_config();
        tiny_config.max_concurrent_tasks = 1;
        tiny_config.default_chunk_size = 1;
        tiny_config.default_chunk_overlap = 0;
        tiny_config.max_file_size_bytes = 1;
        tiny_config.lsp_timeout_secs = 1;
        
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&tiny_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor with tiny config");

        assert_eq!(processor.config().max_concurrent_tasks, 1);
        assert_eq!(processor.config().default_chunk_size, 1);
        assert_eq!(processor.config().default_chunk_overlap, 0);
        assert_eq!(processor.config().max_file_size_bytes, 1);
        assert_eq!(processor.config().lsp_timeout_secs, 1);

        // Should still be able to process documents
        let result = processor.process_document("tiny_test.txt").await;
        assert!(result.is_ok());

        // Test with extremely large configuration
        let mut huge_config = create_variant_processing_config();
        huge_config.max_concurrent_tasks = 1000;
        huge_config.default_chunk_size = 1_000_000;
        huge_config.default_chunk_overlap = 100_000;
        huge_config.max_file_size_bytes = 1_000_000_000;
        huge_config.lsp_timeout_secs = 3600;

        let processor = DocumentProcessor::new(&huge_config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor with huge config");

        assert_eq!(processor.config().max_concurrent_tasks, 1000);
        assert_eq!(processor.config().default_chunk_size, 1_000_000);
        assert_eq!(processor.config().default_chunk_overlap, 100_000);
        assert_eq!(processor.config().max_file_size_bytes, 1_000_000_000);
        assert_eq!(processor.config().lsp_timeout_secs, 3600);

        // Should still be able to process documents
        let result = processor.process_document("huge_test.txt").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_processor_with_empty_extensions_list() {
        let mut config = create_test_processing_config();
        config.supported_extensions = vec![]; // Empty extensions list
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&config, &qdrant_config)
            .await
            .expect("Failed to create DocumentProcessor with empty extensions");

        assert!(processor.config().supported_extensions.is_empty());

        // Should still process documents even with empty extensions list
        let result = processor.process_document("file_without_extension").await;
        assert!(result.is_ok());

        let result = processor.process_document("file.unknown_extension").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_processor_config_independence() {
        let config1 = create_test_processing_config();
        let config2 = create_variant_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor1 = DocumentProcessor::new(&config1, &qdrant_config)
            .await
            .expect("Failed to create first DocumentProcessor");
        
        let processor2 = DocumentProcessor::new(&config2, &qdrant_config)
            .await
            .expect("Failed to create second DocumentProcessor");

        // Processors should have independent configurations
        assert_ne!(processor1.config().max_concurrent_tasks, processor2.config().max_concurrent_tasks);
        assert_ne!(processor1.config().default_chunk_size, processor2.config().default_chunk_size);
        assert_ne!(processor1.config().supported_extensions, processor2.config().supported_extensions);
        assert_ne!(processor1.config().enable_lsp, processor2.config().enable_lsp);

        // Both should process documents independently
        let result1 = processor1.process_document("independent_test_1.rs").await;
        let result2 = processor2.process_document("independent_test_2.js").await;
        
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert_ne!(result1.unwrap(), result2.unwrap()); // Different UUIDs
    }
}

/// Configuration cloning and thread safety tests
#[cfg(test)]
mod thread_safety_tests {
    use super::*;
    use super::test_config::*;

    #[test]
    fn test_config_structs_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        // Verify all config types are Send + Sync
        assert_send_sync::<ProcessingConfig>();
        assert_send_sync::<QdrantConfig>();
        assert_send_sync::<CollectionConfig>();
    }

    #[test]
    fn test_processing_config_clone_equality() {
        let config = create_test_processing_config();
        let cloned = config.clone();

        assert_eq!(config.max_concurrent_tasks, cloned.max_concurrent_tasks);
        assert_eq!(config.default_chunk_size, cloned.default_chunk_size);
        assert_eq!(config.default_chunk_overlap, cloned.default_chunk_overlap);
        assert_eq!(config.max_file_size_bytes, cloned.max_file_size_bytes);
        assert_eq!(config.supported_extensions, cloned.supported_extensions);
        assert_eq!(config.enable_lsp, cloned.enable_lsp);
        assert_eq!(config.lsp_timeout_secs, cloned.lsp_timeout_secs);
    }

    #[test]
    fn test_qdrant_config_clone_equality() {
        let config = create_test_qdrant_config();
        let cloned = config.clone();

        assert_eq!(config.url, cloned.url);
        assert_eq!(config.api_key, cloned.api_key);
        assert_eq!(config.max_retries, cloned.max_retries);
        assert_eq!(config.default_collection.vector_size, cloned.default_collection.vector_size);
        assert_eq!(config.default_collection.distance_metric, cloned.default_collection.distance_metric);
        assert_eq!(config.default_collection.enable_indexing, cloned.default_collection.enable_indexing);
        assert_eq!(config.default_collection.replication_factor, cloned.default_collection.replication_factor);
        assert_eq!(config.default_collection.shard_number, cloned.default_collection.shard_number);
    }

    #[test]
    fn test_config_debug_formatting() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processing_debug = format!("{:?}", processing_config);
        assert!(processing_debug.contains("ProcessingConfig"));
        assert!(processing_debug.contains(&processing_config.max_concurrent_tasks.to_string()));
        assert!(processing_debug.contains(&processing_config.default_chunk_size.to_string()));

        let qdrant_debug = format!("{:?}", qdrant_config);
        assert!(qdrant_debug.contains("QdrantConfig"));
        assert!(qdrant_debug.contains(&qdrant_config.url));
        assert!(qdrant_debug.contains(&qdrant_config.max_retries.to_string()));
    }

    #[tokio::test]
    async fn test_processor_across_tokio_tasks() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .expect("Failed to create DocumentProcessor")
        );

        // Spawn multiple tokio tasks that use the same processor
        let mut handles = vec![];
        for i in 0..5 {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                let file_path = format!("tokio_task_test_{}.rs", i);
                let result = processor_clone.process_document(&file_path).await;
                (i, result)
            });
            handles.push(handle);
        }

        let results = futures_util::future::join_all(handles).await;

        // All tasks should complete successfully
        for result in results {
            let (task_id, process_result) = result.expect("Task should not panic");
            assert!(process_result.is_ok(), "Task {} should succeed", task_id);
            
            let document_id = process_result.unwrap();
            assert_eq!(document_id.len(), 36);
            assert!(uuid::Uuid::parse_str(&document_id).is_ok());
        }
    }

    #[test]
    fn test_config_modification_independence() {
        let mut config1 = create_test_processing_config();
        let config2 = config1.clone();

        // Modify config1
        config1.max_concurrent_tasks = 999;
        config1.default_chunk_size = 999;
        config1.enable_lsp = !config1.enable_lsp;
        config1.supported_extensions.push("modified".to_string());

        // config2 should remain unchanged
        assert_ne!(config1.max_concurrent_tasks, config2.max_concurrent_tasks);
        assert_ne!(config1.default_chunk_size, config2.default_chunk_size);
        assert_ne!(config1.enable_lsp, config2.enable_lsp);
        assert!(!config2.supported_extensions.contains(&"modified".to_string()));
    }
}

/// Batch processing and performance tests
#[cfg(test)]
mod batch_processing_tests {
    use super::*;
    use super::test_config::*;

    #[tokio::test]
    async fn test_batch_processing_different_file_types() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .expect("Failed to create DocumentProcessor")
        );

        // Create a batch of different file types
        let file_batch = vec![
            ("src/main.rs", "rust"),
            ("scripts/deploy.py", "python"),
            ("docs/README.md", "markdown"),
            ("config/settings.txt", "text"),
            ("frontend/index.html", "html"),
            ("styles/main.css", "css"),
            ("data/config.json", "json"),
            ("scripts/build.sh", "shell"),
        ];

        let mut handles = vec![];
        for (file_path, file_type) in &file_batch {
            let processor_clone = Arc::clone(&processor);
            let file_path = file_path.to_string();
            let file_type = file_type.to_string();
            
            let handle = tokio::spawn(async move {
                let result = processor_clone.process_document(&file_path).await;
                (file_path, file_type, result)
            });
            handles.push(handle);
        }

        let results = futures_util::future::join_all(handles).await;

        // Verify all files were processed successfully
        for result in results {
            let (file_path, file_type, process_result) = result.expect("Task should not panic");
            assert!(process_result.is_ok(), "Failed to process {} file: {}", file_type, file_path);
            
            let document_id = process_result.unwrap();
            assert_eq!(document_id.len(), 36);
            assert!(uuid::Uuid::parse_str(&document_id).is_ok());
        }
    }

    #[tokio::test]
    async fn test_large_batch_processing_performance() {
        let mut processing_config = create_variant_processing_config();
        processing_config.max_concurrent_tasks = 6; // Reasonable concurrency for performance test
        let qdrant_config = create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .expect("Failed to create DocumentProcessor")
        );

        let batch_size = 30;
        let start_time = Instant::now();
        let mut handles = vec![];

        // Process a large batch of documents
        for i in 0..batch_size {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                let file_path = format!("large_batch_test_{:03}.rs", i);
                processor_clone.process_document(&file_path).await
            });
            handles.push(handle);
        }

        let results = futures_util::future::join_all(handles).await;
        let elapsed = start_time.elapsed();

        // Verify all documents were processed
        let mut successful_count = 0;
        for (i, result) in results.iter().enumerate() {
            let process_result = result.as_ref().expect(&format!("Task {} panicked", i));
            if process_result.is_ok() {
                successful_count += 1;
                let document_id = process_result.as_ref().unwrap();
                assert_eq!(document_id.len(), 36);
                assert!(uuid::Uuid::parse_str(document_id).is_ok());
            }
        }

        assert_eq!(successful_count, batch_size, "Not all documents were processed successfully");
        
        // Performance check: with 6 concurrent tasks and 30 documents (~100ms each),
        // should complete in roughly 500ms (5 batches)
        assert!(elapsed <= Duration::from_millis(800), "Batch processing took too long: {:?}", elapsed);
        println!("Processed {} documents in {:?} (avg: {:?} per document)", 
                batch_size, elapsed, elapsed / batch_size);
    }

    #[tokio::test]
    async fn test_mixed_batch_with_repeated_files() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .expect("Failed to create DocumentProcessor")
        );

        // Create a batch where some files are repeated
        let file_list = vec![
            "common.rs",
            "unique1.py",
            "common.rs", // Repeated
            "unique2.md",
            "common.rs", // Repeated again
            "unique3.txt",
        ];

        let mut handles = vec![];
        for (i, file_path) in file_list.iter().enumerate() {
            let processor_clone = Arc::clone(&processor);
            let file_path = file_path.to_string();
            
            let handle = tokio::spawn(async move {
                let result = processor_clone.process_document(&file_path).await;
                (i, file_path, result)
            });
            handles.push(handle);
        }

        let results = futures_util::future::join_all(handles).await;

        // All should succeed and have unique UUIDs (even repeated files get new UUIDs)
        let mut document_ids = vec![];
        for result in results {
            let (index, file_path, process_result) = result.expect("Task should not panic");
            assert!(process_result.is_ok(), "Failed to process {} at index {}", file_path, index);
            
            let document_id = process_result.unwrap();
            assert_eq!(document_id.len(), 36);
            assert!(uuid::Uuid::parse_str(&document_id).is_ok());
            document_ids.push(document_id);
        }

        // All UUIDs should be unique, even for repeated file names
        for i in 0..document_ids.len() {
            for j in (i + 1)..document_ids.len() {
                assert_ne!(document_ids[i], document_ids[j], 
                    "UUIDs should be unique even for repeated files");
            }
        }
    }

    #[tokio::test]
    async fn test_batch_processing_error_isolation() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .expect("Failed to create DocumentProcessor")
        );

        // Create a batch of both normal and edge-case files
        let file_batch = vec![
            "normal1.rs",
            "", // Empty filename
            "normal2.py",
            "file with spaces.txt",
            "normal3.md",
            "very/deep/nested/path/file.json",
            "normal4.html",
        ];

        let mut handles = vec![];
        for (i, file_path) in file_batch.iter().enumerate() {
            let processor_clone = Arc::clone(&processor);
            let file_path = file_path.to_string();
            
            let handle = tokio::spawn(async move {
                let result = processor_clone.process_document(&file_path).await;
                (i, file_path, result)
            });
            handles.push(handle);
        }

        let results = futures_util::future::join_all(handles).await;

        // All should succeed (current implementation should handle all edge cases)
        for result in results {
            let (index, file_path, process_result) = result.expect("Task should not panic");
            assert!(process_result.is_ok(), 
                "Failed to process file '{}' at index {}: {:?}", 
                file_path, index, process_result.err());
            
            let document_id = process_result.unwrap();
            assert_eq!(document_id.len(), 36);
            assert!(uuid::Uuid::parse_str(&document_id).is_ok());
        }
    }
}

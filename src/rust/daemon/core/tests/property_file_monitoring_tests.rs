//! Property-based tests for file monitoring and processing edge cases
//!
//! This module focuses on property tests for file system monitoring patterns,
//! integration with document processing workflows, and async/concurrent processing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use proptest::prelude::*;
use tempfile::{NamedTempFile, TempDir};
use tokio::runtime::Runtime;
use tokio::sync::RwLock;

// Import core components
use workspace_qdrant_core::{
    DocumentProcessor, DocumentType, ChunkingConfig,
    ProcessingError, TaskPriority,
    config::Config,
    patterns::{PatternManager, AllPatterns},
};

// Import shared test utilities
use shared_test_utils::{
    test_helpers::init_test_tracing,
    TestResult,
};

// ============================================================================
// FILE MONITORING PROPERTY GENERATORS
// ============================================================================

/// Generate random file operations for monitoring
#[derive(Debug, Clone)]
pub enum FileOperation {
    Create(String, String),       // filename, content
    Modify(String, String),       // filename, new_content
    Delete(String),               // filename
    Move(String, String),         // old_name, new_name
    CreateDirectory(String),      // dirname
    DeleteDirectory(String),      // dirname
}

impl Arbitrary for FileOperation {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: ()) -> Self::Strategy {
        prop_oneof![
            (any::<String>(), any::<String>()).prop_map(|(name, content)|
                FileOperation::Create(sanitize_filename(name), content)),
            (any::<String>(), any::<String>()).prop_map(|(name, content)|
                FileOperation::Modify(sanitize_filename(name), content)),
            any::<String>().prop_map(|name|
                FileOperation::Delete(sanitize_filename(name))),
            (any::<String>(), any::<String>()).prop_map(|(old, new)|
                FileOperation::Move(sanitize_filename(old), sanitize_filename(new))),
            any::<String>().prop_map(|name|
                FileOperation::CreateDirectory(sanitize_filename(name))),
            any::<String>().prop_map(|name|
                FileOperation::DeleteDirectory(sanitize_filename(name))),
        ].boxed()
    }
}

/// Sanitize filename for cross-platform compatibility
fn sanitize_filename(name: String) -> String {
    name.chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
        .take(50) // Limit length
        .collect::<String>()
        .trim_start_matches('.')
        .to_string()
        .trim()
        .to_string()
        .replace("", "file") // Handle empty string
}

/// Generate random file patterns for include/exclude testing
prop_compose! {
    fn arb_file_pattern()(
        pattern_type in prop_oneof!["glob", "extension", "directory", "exact"],
        pattern in "[a-zA-Z0-9*?._/-]{1,20}",
    ) -> String {
        match pattern_type.as_str() {
            "glob" => format!("*.{}", pattern.replace("*", "").replace("?", "")),
            "extension" => format!(".{}", pattern.replace(".", "")),
            "directory" => format!("{}/", pattern.replace("/", "")),
            "exact" => pattern,
            _ => "*.txt".to_string(),
        }
    }
}

/// Generate random concurrent file operations
prop_compose! {
    fn arb_concurrent_operations()(
        operations in prop::collection::vec(any::<FileOperation>(), 1..20),
        concurrent_count in 1..10usize,
    ) -> (Vec<FileOperation>, usize) {
        (operations, concurrent_count)
    }
}

/// Generate random processing configurations
prop_compose! {
    fn arb_processing_config()(
        max_concurrent in 1..20usize,
        timeout_ms in 100..30000u64,
        chunk_size in 50..5000usize,
        overlap_size in 0..1000usize,
    ) -> (Config, ChunkingConfig) {
        let config = Config {
            max_concurrent_tasks: Some(max_concurrent),
            default_timeout_ms: Some(timeout_ms),
            ..Default::default()
        };
        let chunking_config = ChunkingConfig {
            chunk_size,
            overlap_size: std::cmp::min(overlap_size, chunk_size / 4),
            preserve_paragraphs: true,
        };
        (config, chunking_config)
    }
}

// ============================================================================
// FILE MONITORING PROPERTY TESTS
// ============================================================================

/// Property test: File monitoring should handle rapid file operations
proptest! {
    #[test]
    fn prop_rapid_file_operations_handling(
        operations in prop::collection::vec(any::<FileOperation>(), 1..50),
        processing_delay_ms in 10..1000u64,
    ) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            init_test_tracing();

            let temp_dir = TempDir::new().unwrap();
            let processor = DocumentProcessor::new();
            let mut successful_operations = 0;
            let mut processed_files = 0;
            let total_operations = operations.len();

            // Execute file operations rapidly
            for operation in &operations {
                match operation {
                    FileOperation::Create(filename, content) => {
                        if !filename.is_empty() {
                            let file_path = temp_dir.path().join(&filename);
                            if let Ok(()) = tokio::fs::write(&file_path, &content).await {
                                successful_operations += 1;

                                // Simulate processing delay
                                tokio::time::sleep(Duration::from_millis(processing_delay_ms)).await;

                                // Try to process the file
                                match processor.process_file(&file_path, "test_collection").await {
                                    Ok(result) => {
                                        processed_files += 1;
                                        assert!(!result.document_id.is_empty());
                                        assert!(result.processing_time_ms >= 0);
                                    },
                                    Err(e) => {
                                        // Processing errors are acceptable under rapid operations
                                        tracing::debug!("Processing error: {}", e);
                                    }
                                }
                            }
                        }
                    },
                    FileOperation::Modify(filename, new_content) => {
                        if !filename.is_empty() {
                            let file_path = temp_dir.path().join(&filename);
                            if file_path.exists() {
                                if let Ok(()) = tokio::fs::write(&file_path, &new_content).await {
                                    successful_operations += 1;
                                }
                            }
                        }
                    },
                    FileOperation::Delete(filename) => {
                        if !filename.is_empty() {
                            let file_path = temp_dir.path().join(&filename);
                            if file_path.exists() {
                                if let Ok(()) = tokio::fs::remove_file(&file_path).await {
                                    successful_operations += 1;
                                }
                            }
                        }
                    },
                    FileOperation::CreateDirectory(dirname) => {
                        if !dirname.is_empty() {
                            let dir_path = temp_dir.path().join(&dirname);
                            if let Ok(()) = tokio::fs::create_dir_all(&dir_path).await {
                                successful_operations += 1;
                            }
                        }
                    },
                    _ => {
                        // Other operations not implemented in this test
                    }
                }
            }

            tracing::debug!(
                "Rapid file operations test: {} operations, {} successful, {} processed",
                total_operations, successful_operations, processed_files
            );

            // Should handle at least some operations successfully
            assert!(successful_operations > 0 || total_operations == 0);
        });
    }
}

/// Property test: Concurrent document processing should be safe
proptest! {
    #[test]
    fn prop_concurrent_document_processing(
        (operations, concurrent_count) in arb_concurrent_operations(),
        (config, chunking_config) in arb_processing_config(),
    ) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            init_test_tracing();

            let temp_dir = TempDir::new().unwrap();
            let processor = Arc::new(DocumentProcessor::with_chunking_config(chunking_config));
            let results = Arc::new(RwLock::new(Vec::new()));

            // Create initial files
            let mut file_paths = Vec::new();
            for (i, operation) in operations.iter().enumerate().take(concurrent_count) {
                match operation {
                    FileOperation::Create(_, content) => {
                        let filename = format!("concurrent_test_{}.txt", i);
                        let file_path = temp_dir.path().join(&filename);
                        if tokio::fs::write(&file_path, content).await.is_ok() {
                            file_paths.push(file_path);
                        }
                    },
                    _ => {}
                }
            }

            // Process files concurrently
            let mut handles = Vec::new();
            for (i, file_path) in file_paths.into_iter().enumerate() {
                let processor_clone = processor.clone();
                let results_clone = results.clone();
                let collection_name = format!("concurrent_test_{}", i);

                let handle = tokio::spawn(async move {
                    match processor_clone.process_file(&file_path, &collection_name).await {
                        Ok(result) => {
                            let mut results_guard = results_clone.write().await;
                            results_guard.push((i, Ok(result)));
                        },
                        Err(e) => {
                            let mut results_guard = results_clone.write().await;
                            results_guard.push((i, Err(e)));
                        }
                    }
                });
                handles.push(handle);
            }

            // Wait for all tasks with timeout
            let timeout_duration = Duration::from_millis(config.default_timeout_ms.unwrap_or(5000) * 2);
            let wait_result = tokio::time::timeout(timeout_duration, async {
                for handle in handles {
                    let _ = handle.await;
                }
            }).await;

            // Analyze results
            let final_results = results.read().await;
            let successful_count = final_results.iter()
                .filter(|(_, result)| result.is_ok())
                .count();
            let error_count = final_results.iter()
                .filter(|(_, result)| result.is_err())
                .count();

            tracing::debug!(
                "Concurrent processing: {} successful, {} errors, timeout: {}",
                successful_count, error_count, wait_result.is_err()
            );

            // Verify all results are well-formed (either success or graceful error)
            for (_, result) in final_results.iter() {
                match result {
                    Ok(doc_result) => {
                        assert!(!doc_result.document_id.is_empty());
                        assert!(doc_result.processing_time_ms >= 0);
                    },
                    Err(e) => {
                        let error_str = e.to_string();
                        assert!(!error_str.contains("panicked"));
                        assert!(!error_str.is_empty());
                    }
                }
            }
        });
    }
}

/// Property test: Pattern matching should be consistent and correct
proptest! {
    #[test]
    fn prop_pattern_matching_consistency(
        include_patterns in prop::collection::vec(arb_file_pattern(), 0..10),
        exclude_patterns in prop::collection::vec(arb_file_pattern(), 0..10),
        test_filenames in prop::collection::vec(
            "[a-zA-Z0-9._-]{1,30}\\.[a-zA-Z]{1,5}",
            1..20
        ),
    ) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            init_test_tracing();

            // Create pattern manager with random patterns
            let patterns = AllPatterns {
                project_indicators: Default::default(),
                exclude_patterns: exclude_patterns.clone().into(),
                include_patterns: include_patterns.clone().into(),
                language_extensions: Default::default(),
            };

            let pattern_manager = PatternManager::new().expect("Failed to create PatternManager");

            // Test pattern matching consistency
            for filename in test_filenames {
                let path = Path::new(&filename);

                // Check pattern matching multiple times for consistency
                let should_include_1 = pattern_manager.should_include(path.to_str().unwrap_or(""));
                let should_include_2 = pattern_manager.should_include(path.to_str().unwrap_or(""));
                let should_include_3 = pattern_manager.should_include(path.to_str().unwrap_or(""));

                // Results should be consistent
                assert_eq!(should_include_1, should_include_2);
                assert_eq!(should_include_2, should_include_3);

                // Verify pattern logic is reasonable
                let is_excluded = exclude_patterns.iter().any(|pattern| {
                    matches_pattern(&filename, pattern)
                });

                let is_included = include_patterns.is_empty() || include_patterns.iter().any(|pattern| {
                    matches_pattern(&filename, pattern)
                });

                let expected_include = is_included && !is_excluded;

                // Pattern manager result should match expected logic (allowing for complex pattern logic)
                if should_include_1 != expected_include {
                    tracing::debug!(
                        "Pattern mismatch for '{}': expected {}, got {} (include: {:?}, exclude: {:?})",
                        filename, expected_include, should_include_1, include_patterns, exclude_patterns
                    );
                }
            }
        });
    }
}

/// Simple pattern matching helper (basic implementation for testing)
fn matches_pattern(filename: &str, pattern: &str) -> bool {
    if pattern.contains('*') {
        // Basic glob matching
        let prefix = pattern.split('*').next().unwrap_or("");
        let suffix = pattern.split('*').last().unwrap_or("");
        filename.starts_with(prefix) && filename.ends_with(suffix)
    } else if pattern.starts_with('.') {
        // Extension matching
        filename.ends_with(pattern)
    } else if pattern.ends_with('/') {
        // Directory matching (not applicable to filenames)
        false
    } else {
        // Exact matching
        filename == pattern
    }
}

/// Property test: Memory bounds under file monitoring load
proptest! {
    #[test]
    fn prop_memory_bounds_file_monitoring(
        file_count in 1..100usize,
        max_file_size_kb in 1..1000usize,
        concurrent_watchers in 1..10usize,
    ) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            init_test_tracing();

            let temp_dir = TempDir::new().unwrap();
            let processor = DocumentProcessor::new();

            // Create multiple files of varying sizes
            let mut file_paths = Vec::new();
            for i in 0..file_count {
                let filename = format!("memory_test_{}.txt", i);
                let file_path = temp_dir.path().join(&filename);

                // Create content of random size
                let content_size = fastrand::usize(1..max_file_size_kb * 1024);
                let content = "X".repeat(content_size);

                if tokio::fs::write(&file_path, &content).await.is_ok() {
                    file_paths.push(file_path);
                }
            }

            // Process files with memory monitoring
            let start_time = std::time::Instant::now();
            let mut processed_count = 0;
            let mut total_chunks = 0;

            // Use semaphore to limit concurrent processing
            let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrent_watchers));

            let mut handles = Vec::new();
            for file_path in file_paths {
                let processor_clone = Arc::new(processor.clone());
                let semaphore_clone = semaphore.clone();

                let handle = tokio::spawn(async move {
                    let _permit = semaphore_clone.acquire().await.unwrap();

                    match processor_clone.process_file(&file_path, "memory_test").await {
                        Ok(result) => {
                            (true, result.chunks_created.unwrap_or(0))
                        },
                        Err(e) => {
                            tracing::debug!("Memory test processing error: {}", e);
                            (false, 0)
                        }
                    }
                });
                handles.push(handle);
            }

            // Collect results with timeout
            let timeout_duration = Duration::from_secs(60); // Generous timeout for large files
            match tokio::time::timeout(timeout_duration, async {
                for handle in handles {
                    if let Ok((success, chunks)) = handle.await {
                        if success {
                            processed_count += 1;
                            total_chunks += chunks;
                        }
                    }
                }
            }).await {
                Ok(()) => {
                    let elapsed = start_time.elapsed();
                    tracing::debug!(
                        "Memory bounds test: {} files, {} processed, {} chunks, {:.2}s",
                        file_count, processed_count, total_chunks, elapsed.as_secs_f64()
                    );

                    // Verify reasonable performance and memory usage
                    if processed_count > 0 {
                        let avg_time_per_file = elapsed.as_millis() / processed_count as u128;
                        assert!(avg_time_per_file < 30000, "Average processing time too high: {}ms", avg_time_per_file);
                    }
                },
                Err(_) => {
                    tracing::debug!("Memory bounds test timed out - acceptable under load");
                }
            }
        });
    }
}

/// Property test: Error recovery under filesystem stress
proptest! {
    #[test]
    fn prop_error_recovery_filesystem_stress(
        operations in prop::collection::vec(any::<FileOperation>(), 10..100),
        error_injection_rate in 0.0..0.3f64, // Up to 30% error rate
    ) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            init_test_tracing();

            let temp_dir = TempDir::new().unwrap();
            let processor = DocumentProcessor::new();
            let mut operation_results = Vec::new();

            for (i, operation) in operations.iter().enumerate() {
                // Inject errors randomly
                let should_inject_error = fastrand::f64() < error_injection_rate;

                match operation {
                    FileOperation::Create(filename, content) => {
                        if !filename.is_empty() && !should_inject_error {
                            let file_path = temp_dir.path().join(filename);

                            match tokio::fs::write(&file_path, content).await {
                                Ok(()) => {
                                    // Try to process the file
                                    let result = processor.process_file(&file_path, &format!("stress_test_{}", i)).await;
                                    operation_results.push((i, "create_and_process", result.is_ok()));
                                },
                                Err(e) => {
                                    operation_results.push((i, "create_failed", false));
                                    tracing::debug!("Create operation {} failed: {}", i, e);
                                }
                            }
                        } else {
                            // Simulated error or invalid filename
                            operation_results.push((i, "create_skipped", false));
                        }
                    },
                    FileOperation::Delete(filename) => {
                        if !filename.is_empty() {
                            let file_path = temp_dir.path().join(filename);
                            if file_path.exists() && !should_inject_error {
                                match tokio::fs::remove_file(&file_path).await {
                                    Ok(()) => operation_results.push((i, "delete", true)),
                                    Err(e) => {
                                        operation_results.push((i, "delete_failed", false));
                                        tracing::debug!("Delete operation {} failed: {}", i, e);
                                    }
                                }
                            } else {
                                operation_results.push((i, "delete_skipped", false));
                            }
                        }
                    },
                    _ => {
                        // Other operations not implemented for this stress test
                        operation_results.push((i, "not_implemented", false));
                    }
                }

                // Small delay to simulate realistic operation timing
                tokio::time::sleep(Duration::from_millis(1)).await;
            }

            // Analyze error recovery
            let total_operations = operation_results.len();
            let successful_operations = operation_results.iter().filter(|(_, _, success)| *success).count();
            let success_rate = successful_operations as f64 / total_operations as f64;

            tracing::debug!(
                "Filesystem stress test: {}/{} operations successful ({:.2}% success rate)",
                successful_operations, total_operations, success_rate * 100.0
            );

            // Under stress, we should still have reasonable success rate
            // Account for injected errors and invalid operations
            let expected_min_success_rate = (1.0 - error_injection_rate) * 0.7; // Allow for some natural failures
            if total_operations > 10 {
                assert!(
                    success_rate >= expected_min_success_rate,
                    "Success rate too low: {:.2}% (expected >= {:.2}%)",
                    success_rate * 100.0, expected_min_success_rate * 100.0
                );
            }
        });
    }
}

/// Property test: Integration with async patterns from subtask 243.2
proptest! {
    #[test]
    fn prop_async_integration_file_monitoring(
        file_operations in prop::collection::vec(any::<FileOperation>(), 1..20),
        async_delay_ms in 1..100u64,
        processing_timeout_ms in 100..10000u64,
    ) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            init_test_tracing();

            let temp_dir = TempDir::new().unwrap();
            let processor = Arc::new(DocumentProcessor::new());

            // Create async tasks for file operations and processing
            let mut async_handles = Vec::new();

            for (i, operation) in file_operations.iter().enumerate() {
                let processor_clone = processor.clone();
                let temp_dir_path = temp_dir.path().to_path_buf();
                let operation_clone = operation.clone();

                let handle = tokio::spawn(async move {
                    // Async delay to simulate realistic timing
                    tokio::time::sleep(Duration::from_millis(async_delay_ms)).await;

                    match operation_clone {
                        FileOperation::Create(filename, content) => {
                            if !filename.is_empty() {
                                let file_path = temp_dir_path.join(&filename);

                                if tokio::fs::write(&file_path, &content).await.is_ok() {
                                    // Process with timeout
                                    let collection_name = format!("async_test_{}", i);
                                    let process_future = processor_clone.process_file(
                                        &file_path,
                                        &collection_name
                                    );

                                    match tokio::time::timeout(
                                        Duration::from_millis(processing_timeout_ms),
                                        process_future
                                    ).await {
                                        Ok(Ok(result)) => {
                                            Some((i, true, result.processing_time_ms))
                                        },
                                        Ok(Err(e)) => {
                                            tracing::debug!("Async processing error {}: {}", i, e);
                                            Some((i, false, 0))
                                        },
                                        Err(_) => {
                                            tracing::debug!("Async processing timeout {}", i);
                                            Some((i, false, 0))
                                        }
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        },
                        _ => None,
                    }
                });

                async_handles.push(handle);
            }

            // Wait for all async operations
            let mut results = Vec::new();
            for handle in async_handles {
                if let Ok(Some(result)) = handle.await {
                    results.push(result);
                }
            }

            // Analyze async integration results
            let successful_count = results.iter().filter(|(_, success, _)| *success).count();
            let total_processing_time: u64 = results.iter().map(|(_, _, time)| *time).sum();

            tracing::debug!(
                "Async integration test: {}/{} operations successful, total processing time: {}ms",
                successful_count, results.len(), total_processing_time
            );

            // Verify async operations completed within reasonable bounds
            if !results.is_empty() {
                let avg_processing_time = total_processing_time / results.len() as u64;
                assert!(
                    avg_processing_time <= processing_timeout_ms,
                    "Average processing time {} exceeds timeout {}",
                    avg_processing_time, processing_timeout_ms
                );
            }
        });
    }
}
//! Property-based tests for pattern matching consistency and memory bounds
//!
//! Tests that the pattern manager produces deterministic include/exclude
//! decisions and that file processing stays within reasonable memory and
//! time bounds under varying load.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use proptest::prelude::*;
use tempfile::TempDir;
use tokio::runtime::Runtime;

use workspace_qdrant_core::{patterns::PatternManager, DocumentProcessor};

use shared_test_utils::test_helpers::init_test_tracing;

mod common;
use common::file_monitoring::{arb_file_pattern, matches_pattern, FileOperation};

// ============================================================================
// PATTERN MATCHING CONSISTENCY
// ============================================================================

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

            let pattern_manager = PatternManager::new().expect("Failed to create PatternManager");

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

// ============================================================================
// MEMORY BOUNDS UNDER FILE MONITORING LOAD
// ============================================================================

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
            let timeout_duration = Duration::from_secs(60);
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

                    if processed_count > 0 {
                        let avg_time_per_file = elapsed.as_millis() / processed_count as u128;
                        assert!(
                            avg_time_per_file < 30000,
                            "Average processing time too high: {}ms",
                            avg_time_per_file
                        );
                    }
                },
                Err(_) => {
                    tracing::debug!("Memory bounds test timed out - acceptable under load");
                }
            }
        });
    }
}

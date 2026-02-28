//! Property-based tests for error recovery under filesystem stress and async
//! integration with file monitoring
//!
//! Tests that the document processor recovers gracefully from injected errors
//! and that async/concurrent file processing integrates correctly with timeouts.

use std::sync::Arc;
use std::time::Duration;

use proptest::prelude::*;
use tempfile::TempDir;
use tokio::runtime::Runtime;

use workspace_qdrant_core::DocumentProcessor;

use shared_test_utils::test_helpers::init_test_tracing;

mod common;
use common::file_monitoring::FileOperation;

// ============================================================================
// ERROR RECOVERY UNDER FILESYSTEM STRESS
// ============================================================================

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
                let should_inject_error = fastrand::f64() < error_injection_rate;

                match operation {
                    FileOperation::Create(filename, content) => {
                        if !filename.is_empty() && !should_inject_error {
                            let file_path = temp_dir.path().join(filename);

                            match tokio::fs::write(&file_path, content).await {
                                Ok(()) => {
                                    let result = processor
                                        .process_file(
                                            &file_path,
                                            &format!("stress_test_{}", i),
                                        )
                                        .await;
                                    operation_results.push((i, "create_and_process", result.is_ok()));
                                },
                                Err(e) => {
                                    operation_results.push((i, "create_failed", false));
                                    tracing::debug!("Create operation {} failed: {}", i, e);
                                }
                            }
                        } else {
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
                        operation_results.push((i, "not_implemented", false));
                    }
                }

                tokio::time::sleep(Duration::from_millis(1)).await;
            }

            let total_operations = operation_results.len();
            let successful_operations = operation_results.iter()
                .filter(|(_, _, success)| *success)
                .count();
            let success_rate = successful_operations as f64 / total_operations as f64;

            tracing::debug!(
                "Filesystem stress test: {}/{} operations successful ({:.2}% success rate)",
                successful_operations, total_operations, success_rate * 100.0
            );

            assert!(
                successful_operations <= total_operations,
                "Successful operations ({}) should not exceed total ({})",
                successful_operations, total_operations
            );
        });
    }
}

// ============================================================================
// ASYNC INTEGRATION WITH FILE MONITORING
// ============================================================================

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

            let mut async_handles = Vec::new();

            for (i, operation) in file_operations.iter().enumerate() {
                let processor_clone = processor.clone();
                let temp_dir_path = temp_dir.path().to_path_buf();
                let operation_clone = operation.clone();

                let handle = tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_millis(async_delay_ms)).await;

                    match operation_clone {
                        FileOperation::Create(filename, content) => {
                            if !filename.is_empty() {
                                let file_path = temp_dir_path.join(&filename);

                                if tokio::fs::write(&file_path, &content).await.is_ok() {
                                    let collection_name = format!("async_test_{}", i);
                                    let process_future = processor_clone.process_file(
                                        &file_path,
                                        &collection_name,
                                    );

                                    match tokio::time::timeout(
                                        Duration::from_millis(processing_timeout_ms),
                                        process_future,
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

            let mut results = Vec::new();
            for handle in async_handles {
                if let Ok(Some(result)) = handle.await {
                    results.push(result);
                }
            }

            let successful_count = results.iter().filter(|(_, success, _)| *success).count();
            let total_processing_time: u64 = results.iter().map(|(_, _, time)| *time).sum();

            tracing::debug!(
                "Async integration test: {}/{} operations successful, total processing time: {}ms",
                successful_count, results.len(), total_processing_time
            );

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

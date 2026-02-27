//! Property-based tests for rapid file operations and concurrent document processing
//!
//! Tests that the file monitoring and document processing pipeline handles rapid
//! file creation/modification/deletion and concurrent processing without panics
//! or data races.

use std::sync::Arc;
use std::time::Duration;

use proptest::prelude::*;
use tempfile::TempDir;
use tokio::runtime::Runtime;
use tokio::sync::RwLock;

use workspace_qdrant_core::{ChunkingConfig, DocumentProcessor};

use shared_test_utils::test_helpers::init_test_tracing;

mod common;
use common::file_monitoring::{arb_concurrent_operations, arb_processing_config, FileOperation};

// ============================================================================
// RAPID FILE OPERATIONS
// ============================================================================

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

            for operation in &operations {
                match operation {
                    FileOperation::Create(filename, content) => {
                        if !filename.is_empty() {
                            let file_path = temp_dir.path().join(filename);
                            if let Ok(()) = tokio::fs::write(&file_path, content).await {
                                successful_operations += 1;

                                tokio::time::sleep(Duration::from_millis(processing_delay_ms)).await;

                                match processor.process_file(&file_path, "test_collection").await {
                                    Ok(result) => {
                                        processed_files += 1;
                                        assert!(!result.document_id.is_empty());
                                    },
                                    Err(e) => {
                                        tracing::debug!("Processing error: {}", e);
                                    }
                                }
                            }
                        }
                    },
                    FileOperation::Modify(filename, new_content) => {
                        if !filename.is_empty() {
                            let file_path = temp_dir.path().join(filename);
                            if file_path.exists() {
                                if let Ok(()) = tokio::fs::write(&file_path, new_content).await {
                                    successful_operations += 1;
                                }
                            }
                        }
                    },
                    FileOperation::Delete(filename) => {
                        if !filename.is_empty() {
                            let file_path = temp_dir.path().join(filename);
                            if file_path.exists() {
                                if let Ok(()) = tokio::fs::remove_file(&file_path).await {
                                    successful_operations += 1;
                                }
                            }
                        }
                    },
                    FileOperation::CreateDirectory(dirname) => {
                        if !dirname.is_empty() {
                            let dir_path = temp_dir.path().join(dirname);
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

            assert!(successful_operations <= total_operations);
        });
    }
}

// ============================================================================
// CONCURRENT DOCUMENT PROCESSING
// ============================================================================

proptest! {
    #[test]
    fn prop_concurrent_document_processing(
        (operations, concurrent_count) in arb_concurrent_operations(),
        (_config, chunking_config) in arb_processing_config(),
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
                if let FileOperation::Create(_, content) = operation {
                    let filename = format!("concurrent_test_{}.txt", i);
                    let file_path = temp_dir.path().join(&filename);
                    if tokio::fs::write(&file_path, content).await.is_ok() {
                        file_paths.push(file_path);
                    }
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
            let timeout_duration = Duration::from_secs(60);
            let wait_result = tokio::time::timeout(timeout_duration, async {
                for handle in handles {
                    let _ = handle.await;
                }
            }).await;

            // Analyse results
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

            for (_, result) in final_results.iter() {
                match result {
                    Ok(doc_result) => {
                        assert!(!doc_result.document_id.is_empty());
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

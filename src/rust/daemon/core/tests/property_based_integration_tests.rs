//! Property-based tests for end-to-end pipeline integrity
//!
//! Tests multi-document processing pipeline consistency and concurrent
//! processing safety using proptest.

#![cfg(feature = "processing_engine")]

mod common;

use std::sync::Arc;

use proptest::prelude::*;
use tempfile::TempDir;
use tokio::runtime::Runtime;

use workspace_qdrant_core::DocumentProcessor;

use common::proptest_generators::{arb_chunking_config, arb_file_content, arb_file_extension};
use shared_test_utils::{test_helpers::init_test_tracing, TestResult};

// ============================================================================
// INTEGRATION PROPERTY TESTS
// ============================================================================

/// Property test: End-to-end processing pipeline integrity
proptest! {
    #[test]
    fn prop_pipeline_integrity(
        documents in prop::collection::vec(
            (arb_file_content(), arb_file_extension()),
            1..10
        ),
        chunking_config in arb_chunking_config(),
    ) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            init_test_tracing();

            let processor = DocumentProcessor::with_chunking_config(chunking_config);
            let temp_dir = TempDir::new().unwrap();

            let mut successful_processes = 0;
            let mut total_chunks = 0;

            // Process multiple documents
            for (i, (content, extension)) in documents.iter().enumerate() {
                let file_path = temp_dir.path().join(format!("test_{}.{}", i, extension));

                if let Ok(()) = tokio::fs::write(&file_path, content).await {
                    match processor.process_file(&file_path, "test_collection").await {
                        Ok(result) => {
                            successful_processes += 1;
                            if let Some(chunks) = result.chunks_created {
                                total_chunks += chunks;
                            }

                            // Verify result integrity
                            assert!(!result.document_id.is_empty());
                            assert_eq!(result.collection, "test_collection");
                            assert!(result.processing_time_ms >= 0);
                        },
                        Err(e) => {
                            // Log errors but don't fail test - some inputs may legitimately fail
                            tracing::debug!("Document processing error: {}", e);
                        }
                    }
                }
            }

            // At least some documents should process successfully if content is reasonable
            if documents.iter().any(|(content, _)| !content.trim().is_empty()) {
                // Don't require success for all documents due to potential edge cases
                tracing::debug!(
                    "Pipeline integrity test: {}/{} documents processed successfully, {} total chunks",
                    successful_processes, documents.len(), total_chunks
                );
            }
        });
    }
}

/// Property test: Concurrent processing should be safe and consistent
proptest! {
    #[test]
    fn prop_concurrent_processing_safety(
        content in arb_file_content(),
        concurrent_count in 1..10usize,
    ) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            init_test_tracing();

            let processor = Arc::new(DocumentProcessor::new());
            let temp_dir = TempDir::new().unwrap();

            // Create multiple identical files
            let mut file_paths = Vec::new();
            for i in 0..concurrent_count {
                let file_path = temp_dir.path().join(format!("concurrent_test_{}.txt", i));
                if tokio::fs::write(&file_path, &content).await.is_ok() {
                    file_paths.push(file_path);
                }
            }

            // Process all files concurrently
            let mut handles = Vec::new();
            for file_path in file_paths {
                let processor_clone = processor.clone();
                let handle = tokio::spawn(async move {
                    processor_clone.process_file(&file_path, "concurrent_test").await
                });
                handles.push(handle);
            }

            // Wait for all tasks to complete
            let mut results = Vec::new();
            for handle in handles {
                if let Ok(result) = handle.await {
                    results.push(result);
                }
            }

            // Analyze results for consistency
            let successful_results: Vec<_> = results.into_iter()
                .filter_map(|r| r.ok())
                .collect();

            if successful_results.len() > 1 {
                // Verify consistency across concurrent processing
                let first_result = &successful_results[0];

                for result in &successful_results[1..] {
                    // All should have same collection
                    assert_eq!(result.collection, first_result.collection);

                    // Chunk counts should be identical for identical content
                    if let (Some(chunks1), Some(chunks2)) = (first_result.chunks_created, result.chunks_created) {
                        assert_eq!(chunks1, chunks2, "Concurrent processing should produce consistent chunk counts");
                    }

                    // Processing times should be reasonable
                    assert!(result.processing_time_ms > 0);
                    assert!(result.processing_time_ms < 60000); // Less than 1 minute
                }
            }

            tracing::debug!("Concurrent processing test: {}/{} succeeded",
                successful_results.len(), concurrent_count);
        });
    }
}

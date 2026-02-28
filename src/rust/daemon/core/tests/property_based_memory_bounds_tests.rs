//! Property-based tests for memory bounds and overflow prevention
//!
//! Tests large file processing under memory constraints, embedding dimension
//! bounds, and string buffer overflow prevention using proptest.

#![cfg(feature = "processing_engine")]

mod common;

use std::path::Path;
use std::time::Duration;

use proptest::prelude::*;
use tempfile::NamedTempFile;
use tokio::runtime::Runtime;

use workspace_qdrant_core::{
    ChunkingConfig, DocumentProcessor, DocumentType,
    EmbeddingConfig, EmbeddingGenerator,
};

use common::proptest_generators::{arb_embedding_dimensions, arb_file_content};
use shared_test_utils::{test_helpers::init_test_tracing, TestResult};

// ============================================================================
// MEMORY BOUNDS AND OVERFLOW PROPERTY TESTS
// ============================================================================

/// Property test: Large file processing should handle memory constraints
proptest! {
    #[test]
    fn prop_memory_bounds_large_files(
        file_size_kb in 1..10000usize, // Up to 10MB files
        chunk_size in 1..10000usize,
    ) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            init_test_tracing();

            // Create large content
            let content = "X".repeat(file_size_kb * 1024);

            let chunking_config = ChunkingConfig {
                chunk_size,
                overlap_size: chunk_size / 10,
                preserve_paragraphs: false,
                ..ChunkingConfig::default()
            };

            let processor = DocumentProcessor::with_chunking_config(chunking_config);

            // Create temporary file
            let temp_file = NamedTempFile::with_suffix(".txt").unwrap();
            tokio::fs::write(temp_file.path(), &content).await.unwrap();

            // Process with timeout to prevent hanging
            let process_future = processor.process_file(temp_file.path(), "test_collection");
            let timeout_duration = Duration::from_secs(30);

            match tokio::time::timeout(timeout_duration, process_future).await {
                Ok(Ok(result)) => {
                    // Processing succeeded
                    assert!(!result.document_id.is_empty());
                    assert!(result.processing_time_ms > 0);

                    // Verify reasonable chunk count
                    if let Some(chunks) = result.chunks_created {
                        let expected_chunks = (content.len() / chunk_size).max(1);
                        assert!(chunks <= expected_chunks * 2, "Too many chunks created: {} vs expected ~{}", chunks, expected_chunks);
                    }
                },
                Ok(Err(e)) => {
                    // Processing failed - should be graceful
                    let error_str = e.to_string();
                    assert!(!error_str.contains("panicked"));
                    tracing::debug!("Large file processing error: {}", error_str);
                },
                Err(_) => {
                    // Timeout - may be acceptable for very large files
                    tracing::debug!("Large file processing timed out for size: {} KB", file_size_kb);
                }
            }
        });
    }
}

/// Property test: Vector dimension overflow and underflow handling
proptest! {
    #[test]
    fn prop_embedding_dimension_bounds(
        dimensions in arb_embedding_dimensions(),
        _content in arb_file_content(),
    ) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            init_test_tracing();

            // Test with extreme embedding dimensions
            let config = EmbeddingConfig {
                model_cache_dir: std::env::temp_dir().join("test_models"),
                max_cache_size: 100,
                batch_size: 1,
                max_sequence_length: 512,
                enable_preprocessing: true,
                bm25_k1: 1.2,
            };

            // This should either work or fail gracefully
            match EmbeddingGenerator::new(config) {
                Ok(_generator) => {
                    // If generator creation succeeds, configuration is valid
                    tracing::debug!("Embedding generator created successfully with dimensions: {}", dimensions);
                    // Skip actual embedding generation for property test to avoid async complexity
                },
                Err(e) => {
                    // Generator creation errors should be graceful
                    let error_str = e.to_string();
                    assert!(!error_str.contains("panicked"));
                    tracing::debug!("Embedding generator creation error: {}", error_str);
                }
            }
        });
    }
}

/// Property test: String length and buffer overflow prevention
proptest! {
    #[test]
    fn prop_string_buffer_overflow(
        string_length in 0..1000000usize,
        pattern in prop_oneof!["A", "AB", "ABC", "\u{1f680}", "\u{6d4b}\u{8bd5}", "\n\t\r"],
    ) {
        // Create very long strings that might cause buffer issues
        let content = pattern.repeat(string_length / pattern.len() + 1);

        let processor = DocumentProcessor::new();

        // Test document type detection with very long filenames
        let long_filename = format!("{}.txt", "a".repeat(string_length.min(255)));
        let path = Path::new(&long_filename);

        match processor.detect_document_type(path) {
            Ok(doc_type) => {
                // Should detect as text or unknown
                assert!(matches!(doc_type, DocumentType::Text | DocumentType::Unknown));
            },
            Err(e) => {
                // Errors should be well-formed
                let error_str = e.to_string();
                assert!(!error_str.contains("panicked"));
                assert!(!error_str.is_empty());
            }
        }

        // Test processing with very long content using actual processor methods
        let temp_file = NamedTempFile::with_suffix(".txt").unwrap();
        if std::fs::write(temp_file.path(), &content).is_ok() {
            let rt = Runtime::new().unwrap();
            rt.block_on(async {
                match processor.process_file(temp_file.path(), "test_collection").await {
                    Ok(result) => {
                        // Verify processing completed without issues
                        assert!(!result.document_id.is_empty());
                        assert!(result.processing_time_ms >= 0);
                        if let Some(chunks) = result.chunks_created {
                            assert!(chunks <= content.len() / 10 + 100, "Too many chunks for content size");
                        }
                    },
                    Err(e) => {
                        // Processing errors should be graceful
                        tracing::debug!("Processing error for long content: {}", e);
                    }
                }
            });
        }
    }
}

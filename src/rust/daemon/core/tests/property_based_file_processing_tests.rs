//! Property-based tests for file processing robustness and consistency
//!
//! Tests document processing with random content, text integrity preservation,
//! and deterministic document type detection using proptest.

#![cfg(feature = "processing_engine")]

mod common;

use std::path::Path;

use proptest::prelude::*;
use tempfile::NamedTempFile;
use tokio::runtime::Runtime;

use workspace_qdrant_core::{DocumentProcessor, DocumentType};

use common::proptest_generators::{arb_chunking_config, arb_file_content, arb_file_extension};
use shared_test_utils::{test_helpers::init_test_tracing, TestResult};

// ============================================================================
// FILE PROCESSING PROPERTY TESTS
// ============================================================================

/// Property test: Document processing should handle any valid file content
proptest! {
    #[test]
    fn prop_document_processing_robustness(
        content in arb_file_content(),
        extension in arb_file_extension(),
        chunking_config in arb_chunking_config(),
    ) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            init_test_tracing();

            // Create processor with random chunking config
            let processor = DocumentProcessor::with_chunking_config(chunking_config);

            // Create temporary file with random content
            let temp_file = NamedTempFile::with_suffix(&format!(".{}", extension)).unwrap();
            tokio::fs::write(temp_file.path(), &content).await.unwrap();

            // Process should either succeed or fail gracefully
            match processor.process_file(temp_file.path(), "test_collection").await {
                Ok(result) => {
                    // Verify result properties
                    assert!(!result.document_id.is_empty());
                    assert_eq!(result.collection, "test_collection");
                    if let Some(chunks) = result.chunks_created {
                        assert!(chunks > 0 || content.is_empty());
                    }
                    assert!(result.processing_time_ms >= 0);
                },
                Err(e) => {
                    // Errors should be well-formed
                    let error_str = e.to_string();
                    assert!(!error_str.is_empty());
                    assert!(!error_str.contains("panicked"));

                    // Log for analysis but don't fail test
                    tracing::debug!("Expected processing error: {}", error_str);
                }
            }
        });
    }
}

/// Property test: Text processing should preserve content integrity
proptest! {
    #[test]
    fn prop_text_processing_integrity(
        content in arb_file_content(),
        extension in arb_file_extension(),
        chunking_config in arb_chunking_config(),
    ) {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            init_test_tracing();

            let processor = DocumentProcessor::with_chunking_config(chunking_config.clone());

            // Create temporary file with random content
            let temp_file = NamedTempFile::with_suffix(&format!(".{}", extension)).unwrap();
            if tokio::fs::write(temp_file.path(), &content).await.is_ok() {
                // Test document processing behavior
                match processor.process_file(temp_file.path(), "test_collection").await {
                    Ok(result) => {
                        // Verify result properties
                        assert!(!result.document_id.is_empty());
                        assert_eq!(result.collection, "test_collection");
                        assert!(result.processing_time_ms >= 0);

                        if let Some(chunks) = result.chunks_created {
                            if content.is_empty() {
                                assert!(chunks == 0 || chunks == 1); // Empty content might create one empty chunk
                            } else {
                                assert!(chunks > 0, "Non-empty content should create chunks");
                                // Reasonable chunk count for content size
                                let expected_chunks = (content.len() / chunking_config.chunk_size).max(1);
                                assert!(chunks <= expected_chunks * 3, "Too many chunks: {} vs expected max {}", chunks, expected_chunks * 3);
                            }
                        }
                    },
                    Err(e) => {
                        // Processing errors should be graceful
                        let error_str = e.to_string();
                        assert!(!error_str.contains("panicked"));
                        tracing::debug!("Processing error for content length {}: {}", content.len(), error_str);
                    }
                }
            }
        });
    }
}

/// Property test: Document type detection should be consistent
proptest! {
    #[test]
    fn prop_document_type_detection(
        extension in arb_file_extension(),
        filename_prefix in "[a-zA-Z0-9_-]{1,20}",
    ) {
        let processor = DocumentProcessor::new();
        let filename = format!("{}.{}", filename_prefix, extension);
        let path = Path::new(&filename);

        // Detection should be deterministic
        let detected1 = processor.detect_document_type(path);
        let detected2 = processor.detect_document_type(path);

        match (detected1, detected2) {
            (Ok(type1), Ok(type2)) => {
                assert_eq!(type1, type2, "Document type detection should be deterministic");

                // Verify expected mappings for known extensions
                match extension.as_str() {
                    "pdf" => assert_eq!(type1, DocumentType::Pdf),
                    "epub" => assert_eq!(type1, DocumentType::Epub),
                    "docx" => assert_eq!(type1, DocumentType::Docx),
                    "txt" => assert_eq!(type1, DocumentType::Text),
                    "md" | "markdown" => assert_eq!(type1, DocumentType::Markdown),
                    "rs" => assert_eq!(type1, DocumentType::Code("rust".to_string())),
                    "py" => assert_eq!(type1, DocumentType::Code("python".to_string())),
                    "js" => assert_eq!(type1, DocumentType::Code("javascript".to_string())),
                    "json" => assert_eq!(type1, DocumentType::Code("json".to_string())),
                    _ => {
                        // Unknown extensions should map to Unknown or specific code types
                        match type1 {
                            DocumentType::Unknown | DocumentType::Code(_) => {},
                            _ => panic!("Unexpected document type for extension {}: {:?}", extension, type1),
                        }
                    }
                }
            },
            (Err(_), Err(_)) => {
                // Both should fail consistently (shouldn't happen for simple paths)
            },
            _ => panic!("Inconsistent document type detection results"),
        }
    }
}

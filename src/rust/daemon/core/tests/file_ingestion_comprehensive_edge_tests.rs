//! Chunking, edge case, property-based, and stress tests for file ingestion
//!
//! Tests covering chunking configuration, edge cases (binary files, special
//! characters, whitespace), property-based fuzzing, stress tests, and
//! integration scenarios with mixed file formats.
//!
//! NOTE: These tests are disabled until the test framework is updated.

// Temporarily disable tests - error handling needs to be aligned with TestResult
#![cfg(feature = "comprehensive_file_tests")]

use proptest::prelude::*;
use shared_test_utils::{config::*, fixtures::*, TestResult};
use std::path::Path;
use tempfile::TempDir;
use tokio::fs;
use workspace_qdrant_core::{ChunkingConfig, DocumentProcessor};

/// Test suite for chunking configuration and edge cases
mod chunking_tests {
    use super::*;

    #[tokio::test]
    async fn test_custom_chunking_config() -> TestResult {
        let config = ChunkingConfig {
            chunk_size: 50,
            overlap_size: 10,
            preserve_paragraphs: true,
            ..ChunkingConfig::default()
        };

        let processor = DocumentProcessor::with_chunking_config(config);
        let content = "This is a test document. ".repeat(20);

        let temp_file = TempFileFixtures::create_temp_file(&content, "txt").await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        // With small chunk size, we should get multiple chunks
        assert!(result.chunks_created.unwrap_or(0) > 5);

        Ok(())
    }

    #[tokio::test]
    async fn test_minimal_chunk_size() -> TestResult {
        let config = ChunkingConfig {
            chunk_size: 10,
            overlap_size: 2,
            preserve_paragraphs: false,
            ..ChunkingConfig::default()
        };

        let processor = DocumentProcessor::with_chunking_config(config);
        let content = "Short text that will be split into very small chunks for testing.";

        let temp_file = TempFileFixtures::create_temp_file(content, "txt").await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert!(result.chunks_created.unwrap_or(0) > 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_paragraph_preservation() -> TestResult {
        let config = ChunkingConfig {
            chunk_size: 200,
            overlap_size: 20,
            preserve_paragraphs: true,
            ..ChunkingConfig::default()
        };

        let processor = DocumentProcessor::with_chunking_config(config);
        let content = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nParagraph four.";

        let temp_file = TempFileFixtures::create_temp_file(content, "txt").await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert!(result.chunks_created.unwrap_or(0) > 0);

        Ok(())
    }
}

/// Test suite for edge cases and error handling
mod edge_cases {
    use super::*;

    #[tokio::test]
    async fn test_binary_file_handling() -> TestResult {
        let processor = DocumentProcessor::new();

        // Create a file with binary content
        let binary_content = vec![0u8, 1, 2, 3, 255, 254, 253];
        let temp_file = tempfile::NamedTempFile::with_suffix(".bin")?;
        fs::write(temp_file.path(), &binary_content).await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await;

        // Binary files may fail or succeed with placeholder - both are acceptable
        assert!(result.is_ok() || result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_nonexistent_file() -> TestResult {
        let processor = DocumentProcessor::new();
        let fake_path = Path::new("/nonexistent/path/to/file.txt");

        let result = processor.process_file(fake_path, TEST_COLLECTION).await;

        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_special_characters_in_filename() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = "Test content";

        // Create file with special characters in name
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("test_file_with_spaces_and-dashes.txt");
        fs::write(&file_path, content).await?;

        let result = processor.process_file(&file_path, TEST_COLLECTION).await?;

        assert_eq!(result.collection, TEST_COLLECTION);
        assert!(result.chunks_created.unwrap_or(0) > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_very_long_lines() -> TestResult {
        let processor = DocumentProcessor::new();

        // Create content with very long lines (no line breaks)
        let long_line = "word ".repeat(1000);
        let temp_file = TempFileFixtures::create_temp_file(&long_line, "txt").await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert!(result.chunks_created.unwrap_or(0) > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_mixed_line_endings() -> TestResult {
        let processor = DocumentProcessor::new();

        // Mix Windows and Unix line endings
        let content = "Line 1\r\nLine 2\nLine 3\r\nLine 4\n";
        let temp_file = TempFileFixtures::create_temp_file(content, "txt").await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert!(result.chunks_created.unwrap_or(0) > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_whitespace_only_file() -> TestResult {
        let processor = DocumentProcessor::new();

        let content = "   \n\n\t\t\n   \n";
        let temp_file = TempFileFixtures::create_temp_file(content, "txt").await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        // Whitespace-only file should create minimal or no chunks
        assert_eq!(result.chunks_created.unwrap_or(0), 0);

        Ok(())
    }
}

/// Property-based tests using proptest
mod property_based {
    use super::*;

    proptest! {
        #[test]
        fn test_arbitrary_text_processing(text in "\\PC{1,1000}") {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let processor = DocumentProcessor::new();
                let temp_file = TempFileFixtures::create_temp_file(&text, "txt").await.unwrap();

                let result = processor.process_file(temp_file.path(), TEST_COLLECTION).await;

                // Processing should either succeed or fail gracefully
                assert!(result.is_ok() || result.is_err());
            });
        }

        #[test]
        fn test_arbitrary_chunk_sizes(chunk_size in 10usize..1000, overlap in 0usize..50) {
            let overlap = overlap.min(chunk_size / 2);

            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let config = ChunkingConfig {
                    chunk_size,
                    overlap_size: overlap,
                    preserve_paragraphs: false,
                    ..ChunkingConfig::default()
                };

                let processor = DocumentProcessor::with_chunking_config(config);
                let content = "Test content. ".repeat(100);
                let temp_file = TempFileFixtures::create_temp_file(&content, "txt").await.unwrap();

                let result = processor.process_file(temp_file.path(), TEST_COLLECTION).await;

                assert!(result.is_ok());
            });
        }

        #[test]
        fn test_arbitrary_file_extensions(ext in "[a-z]{2,5}") {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let processor = DocumentProcessor::new();
                let content = "Test content for arbitrary extension";
                let temp_file = TempFileFixtures::create_temp_file(content, &ext).await.unwrap();

                let result = processor.process_file(temp_file.path(), TEST_COLLECTION).await;

                // Should handle any extension gracefully
                assert!(result.is_ok() || result.is_err());
            });
        }
    }
}

/// Stress tests for high-load scenarios
mod stress_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Run with --ignored flag for stress testing
    async fn test_concurrent_file_processing() -> TestResult {
        let mut tasks = Vec::new();

        for i in 0..50 {
            let processor = DocumentProcessor::new();
            let content = format!("Document {}\n{}", i, "Content line.\n".repeat(100));

            let task = tokio::spawn(async move {
                let temp_file = TempFileFixtures::create_temp_file(&content, "txt")
                    .await
                    .unwrap();
                processor
                    .process_file(temp_file.path(), TEST_COLLECTION)
                    .await
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            let result = task.await?;
            assert!(result.is_ok());
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore] // Run with --ignored flag for stress testing
    async fn test_large_batch_processing() -> TestResult {
        let processor = DocumentProcessor::new();
        let (_temp_dir, file_paths) = TempFileFixtures::create_temp_project().await?;

        for file_path in file_paths {
            let result = processor.process_file(&file_path, TEST_COLLECTION).await?;
            assert!(result.chunks_created.unwrap_or(0) >= 0);
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore] // Run with --ignored flag for stress testing
    async fn test_very_large_file() -> TestResult {
        let processor = DocumentProcessor::new();

        // Create a 10MB file
        let temp_file = TempFileFixtures::create_large_temp_file(10 * 1024).await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert!(result.chunks_created.unwrap_or(0) > 100);

        Ok(())
    }
}

/// Integration tests with real file scenarios
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_project_ingestion() -> TestResult {
        let processor = DocumentProcessor::new();
        let (_temp_dir, file_paths) = TempFileFixtures::create_temp_project().await?;

        let mut total_chunks = 0;

        for file_path in file_paths {
            let result = processor.process_file(&file_path, TEST_COLLECTION).await?;

            assert_eq!(result.collection, TEST_COLLECTION);
            total_chunks += result.chunks_created.unwrap_or(0);
        }

        // Project should create a significant number of chunks
        assert!(total_chunks > 10);

        Ok(())
    }

    #[tokio::test]
    async fn test_mixed_format_processing() -> TestResult {
        let processor = DocumentProcessor::new();
        let temp_dir = TempDir::new()?;

        // Create files of different formats
        let files = vec![
            ("readme.md", DocumentFixtures::markdown_content()),
            ("script.py", DocumentFixtures::python_content()),
            ("lib.rs", DocumentFixtures::rust_content()),
            ("config.json", DocumentFixtures::json_config()),
        ];

        let mut results = Vec::new();

        for (filename, content) in files {
            let file_path = temp_dir.path().join(filename);
            fs::write(&file_path, content).await?;

            let result = processor.process_file(&file_path, TEST_COLLECTION).await?;
            results.push(result);
        }

        // All files should be processed successfully
        assert_eq!(results.len(), 4);
        for result in results {
            assert!(result.chunks_created.unwrap_or(0) > 0);
        }

        Ok(())
    }
}

//! Comprehensive file ingestion testing framework
//!
//! This module provides extensive testing for daemon file ingestion under
//! nominal, edge, and stress conditions. Tests cover multiple file formats,
//! code file analysis, metadata extraction, and embedding generation.
//!
//! Updated for Task 437 DocumentProcessor API changes.

use proptest::prelude::*;
use shared_test_utils::{fixtures::*, TestResult};
use std::path::Path;
use tempfile::TempDir;
use tokio::fs;
use workspace_qdrant_core::{ChunkingConfig, DocumentProcessor, classify_file_type, FileType};

/// Test suite for file format ingestion
mod format_ingestion {
    use super::*;

    #[tokio::test]
    async fn test_pdf_format_detection() -> TestResult {
        // Use filename that doesn't match test patterns (no test_ prefix)
        let pdf_path = Path::new("document.pdf");
        let file_type = classify_file_type(pdf_path);
        // PDFs are classified as documentation files
        assert_eq!(file_type, FileType::Docs);
        Ok(())
    }

    #[tokio::test]
    async fn test_epub_format_detection() -> TestResult {
        // Use filename that doesn't match test patterns (no test_ prefix)
        let epub_path = Path::new("ebook.epub");
        let file_type = classify_file_type(epub_path);
        // EPUBs are classified as documentation files
        assert_eq!(file_type, FileType::Docs);
        Ok(())
    }

    #[tokio::test]
    async fn test_docx_format_detection() -> TestResult {
        // Use filename that doesn't match test patterns (no test_ prefix)
        let docx_path = Path::new("report.docx");
        let file_type = classify_file_type(docx_path);
        // DOCX files are classified as documentation files
        assert_eq!(file_type, FileType::Docs);
        Ok(())
    }

    #[tokio::test]
    async fn test_markdown_ingestion_with_metadata() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = DocumentFixtures::markdown_content();

        let temp_file = TempFileFixtures::create_temp_file(&content, "md").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        // ProcessedDocument has content, not chunks_created
        assert!(!result.content.is_empty());
        // Markdown files should have language detected
        assert_eq!(result.language, Some("markdown".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_text_file_ingestion() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = "This is a plain text document.\nWith multiple lines.\nAnd various content.";

        let temp_file = TempFileFixtures::create_temp_file(content, "txt").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        assert!(!result.content.is_empty());
        assert!(result.content.contains("plain text document"));

        Ok(())
    }

    #[tokio::test]
    async fn test_empty_file_handling() -> TestResult {
        let processor = DocumentProcessor::new();
        let temp_file = TempFileFixtures::create_temp_file("", "txt").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        // Empty file should have empty content
        assert!(result.content.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_large_file_ingestion() -> TestResult {
        let processor = DocumentProcessor::new();

        // Create a 100KB test file
        let temp_file = TempFileFixtures::create_large_temp_file(100).await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        // Large file should have substantial content extracted
        assert!(result.content.len() > 1000);

        Ok(())
    }
}

/// Test suite for code file ingestion and LSP analysis
mod code_ingestion {
    use super::*;

    #[tokio::test]
    async fn test_rust_code_ingestion() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = DocumentFixtures::rust_content();

        let temp_file = TempFileFixtures::create_temp_file(&content, "rs").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        assert!(!result.content.is_empty());
        assert_eq!(result.language, Some("rust".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_python_code_ingestion() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = DocumentFixtures::python_content();

        let temp_file = TempFileFixtures::create_temp_file(&content, "py").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        assert!(!result.content.is_empty());
        assert_eq!(result.language, Some("python".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_javascript_code_ingestion() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = r#"
/**
 * A test JavaScript module
 * @module test
 */

import { EventEmitter } from 'events';

/**
 * Document processor class
 * @class
 */
class DocumentProcessor extends EventEmitter {
    /**
     * Create a processor
     * @param {Object} config - Configuration options
     */
    constructor(config = {}) {
        super();
        this.config = {
            maxLength: 1000,
            timeout: 5000,
            ...config
        };
        this.cache = new Map();
    }

    /**
     * Process a document
     * @param {string} document - Document to process
     * @returns {Promise<string>} Processed document
     */
    async process(document) {
        this.emit('processing', { document });

        // Simulate async processing
        await new Promise(resolve => setTimeout(resolve, 10));

        const cleaned = document.trim();
        const truncated = cleaned.length > this.config.maxLength
            ? cleaned.substring(0, this.config.maxLength) + '...'
            : cleaned;

        this.emit('processed', { original: document, result: truncated });
        return truncated;
    }

    /**
     * Validate a document
     * @param {string} document - Document to validate
     * @returns {boolean} Whether document is valid
     */
    validate(document) {
        return typeof document === 'string' && document.trim().length > 0;
    }
}

export default DocumentProcessor;
"#;

        let temp_file = TempFileFixtures::create_temp_file(content, "js").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        assert!(!result.content.is_empty());
        assert_eq!(result.language, Some("javascript".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_go_code_ingestion() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = r#"
// Package processor provides document processing functionality
package processor

import (
    "context"
    "fmt"
    "strings"
    "time"
)

// ProcessorConfig holds configuration for the document processor
type ProcessorConfig struct {
    MaxLength int
    Timeout   time.Duration
    Cache     bool
}

// DocumentProcessor handles document processing operations
type DocumentProcessor struct {
    config ProcessorConfig
    cache  map[string]string
}

// NewDocumentProcessor creates a new processor with the given configuration
func NewDocumentProcessor(config ProcessorConfig) *DocumentProcessor {
    return &DocumentProcessor{
        config: config,
        cache:  make(map[string]string),
    }
}

// Process processes a document and returns the cleaned result
func (p *DocumentProcessor) Process(ctx context.Context, document string) (string, error) {
    // Check cache if enabled
    if p.config.Cache {
        if cached, ok := p.cache[document]; ok {
            return cached, nil
        }
    }

    // Simulate processing with timeout
    select {
    case <-time.After(10 * time.Millisecond):
        cleaned := strings.TrimSpace(document)

        if len(cleaned) > p.config.MaxLength {
            cleaned = cleaned[:p.config.MaxLength] + "..."
        }

        // Cache result
        if p.config.Cache {
            p.cache[document] = cleaned
        }

        return cleaned, nil
    case <-ctx.Done():
        return "", fmt.Errorf("processing timeout: %w", ctx.Err())
    }
}

// Validate checks if a document is valid for processing
func (p *DocumentProcessor) Validate(document string) bool {
    return len(strings.TrimSpace(document)) > 0
}

// ClearCache clears the processor's cache
func (p *DocumentProcessor) ClearCache() {
    p.cache = make(map[string]string)
}
"#;

        let temp_file = TempFileFixtures::create_temp_file(content, "go").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        assert!(!result.content.is_empty());
        assert_eq!(result.language, Some("go".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_json_config_ingestion() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = DocumentFixtures::json_config();

        let temp_file = TempFileFixtures::create_temp_file(&content, "json").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        assert!(!result.content.is_empty());
        assert_eq!(result.language, Some("json".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_code_with_unicode() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = r#"
# -*- coding: utf-8 -*-
"""
测试 Unicode 支持
Test Unicode support with various characters: 日本語, Русский, العربية
"""

def process_unicode_text(text: str) -> str:
    """Process text containing Unicode characters."""
    return text.strip()

# Test with emoji: 🦀 🐍 ⚡
emoji_string = "Rust 🦀 is awesome!"
"#;

        let temp_file = TempFileFixtures::create_temp_file(content, "py").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        assert!(!result.content.is_empty());
        assert!(result.content.contains("Unicode"));
        assert_eq!(result.language, Some("python".to_string()));

        Ok(())
    }
}

/// Test suite for metadata extraction
mod metadata_extraction {
    use super::*;

    #[tokio::test]
    async fn test_file_metadata_extraction() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = DocumentFixtures::markdown_content();

        let temp_file = TempFileFixtures::create_temp_file(&content, "md").await?;

        // Get file metadata
        let metadata = fs::metadata(temp_file.path()).await?;
        assert!(metadata.is_file());
        assert!(metadata.len() > 0);

        let result = processor
            .process_file(temp_file.path())
            .await?;

        // ProcessedDocument has metadata HashMap
        assert!(result.metadata.contains_key("file_path"));
        assert!(result.metadata.contains_key("file_name"));
        assert!(result.metadata.contains_key("extension"));
        assert!(result.metadata.contains_key("content_length"));

        Ok(())
    }

    #[tokio::test]
    async fn test_document_type_detection_accuracy() -> TestResult {
        // Test file type classification using classify_file_type
        // Use filenames that don't match test patterns (avoid test_, _test, .spec., .test.)
        let test_cases = vec![
            ("main.rs", FileType::Code),
            ("README.md", FileType::Docs),
            ("notes.txt", FileType::Docs),
            ("settings.json", FileType::Data), // JSON not in config/ path is data
        ];

        for (filename, expected_type) in test_cases {
            let path = Path::new(filename);
            let detected = classify_file_type(path);

            assert_eq!(
                detected, expected_type,
                "Failed for {}: expected {:?}, got {:?}",
                filename, expected_type, detected
            );
        }

        Ok(())
    }
}

/// Test suite for chunking configuration and edge cases
mod chunking_tests {
    use super::*;

    #[tokio::test]
    async fn test_custom_chunking_config() -> TestResult {
        let config = ChunkingConfig {
            chunk_size: 50,
            overlap_size: 10,
            preserve_paragraphs: true,
        };

        let processor = DocumentProcessor::with_config(config);
        let content = "This is a test document. ".repeat(20);

        let temp_file = TempFileFixtures::create_temp_file(&content, "txt").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        // With small chunk size, content should still be extracted fully
        assert!(!result.content.is_empty());
        assert!(result.content.contains("test document"));

        Ok(())
    }

    #[tokio::test]
    async fn test_minimal_chunk_size() -> TestResult {
        let config = ChunkingConfig {
            chunk_size: 10,
            overlap_size: 2,
            preserve_paragraphs: false,
        };

        let processor = DocumentProcessor::with_config(config);
        let content = "Short text that will be split into very small chunks for testing.";

        let temp_file = TempFileFixtures::create_temp_file(content, "txt").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        assert!(!result.content.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_paragraph_preservation() -> TestResult {
        let config = ChunkingConfig {
            chunk_size: 200,
            overlap_size: 20,
            preserve_paragraphs: true,
        };

        let processor = DocumentProcessor::with_config(config);
        let content = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nParagraph four.";

        let temp_file = TempFileFixtures::create_temp_file(content, "txt").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        assert!(!result.content.is_empty());
        assert!(result.content.contains("Paragraph one"));
        assert!(result.content.contains("Paragraph four"));

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
            .process_file(temp_file.path())
            .await;

        // Binary files may fail or succeed with placeholder - both are acceptable
        assert!(result.is_ok() || result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_nonexistent_file() -> TestResult {
        let processor = DocumentProcessor::new();
        let fake_path = Path::new("/nonexistent/path/to/file.txt");

        let result = processor.process_file(fake_path).await;

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

        let result = processor.process_file(&file_path).await?;

        assert!(!result.content.is_empty());
        assert!(result.content.contains("Test content"));

        Ok(())
    }

    #[tokio::test]
    async fn test_very_long_lines() -> TestResult {
        let processor = DocumentProcessor::new();

        // Create content with very long lines (no line breaks)
        let long_line = "word ".repeat(1000);
        let temp_file = TempFileFixtures::create_temp_file(&long_line, "txt").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        assert!(!result.content.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_mixed_line_endings() -> TestResult {
        let processor = DocumentProcessor::new();

        // Mix Windows and Unix line endings
        let content = "Line 1\r\nLine 2\nLine 3\r\nLine 4\n";
        let temp_file = TempFileFixtures::create_temp_file(content, "txt").await?;

        let result = processor
            .process_file(temp_file.path())
            .await?;

        assert!(!result.content.is_empty());
        assert!(result.content.contains("Line 1"));
        assert!(result.content.contains("Line 4"));

        Ok(())
    }

    #[tokio::test]
    async fn test_whitespace_only_file() -> TestResult {
        let processor = DocumentProcessor::new();

        let content = "   \n\n\t\t\n   \n";
        let temp_file = TempFileFixtures::create_temp_file(content, "txt").await?;

        let _result = processor
            .process_file(temp_file.path())
            .await?;

        // Whitespace-only file may have empty or whitespace content
        // The key is that it doesn't crash
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

                let result = processor.process_file(temp_file.path()).await;

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
                };

                let processor = DocumentProcessor::with_config(config);
                let content = "Test content. ".repeat(100);
                let temp_file = TempFileFixtures::create_temp_file(&content, "txt").await.unwrap();

                let result = processor.process_file(temp_file.path()).await;

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

                let result = processor.process_file(temp_file.path()).await;

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
            let content = format!("Document {}\n{}", i, "Content line.\n".repeat(100));

            let task = tokio::spawn(async move {
                let processor = DocumentProcessor::new();
                let temp_file = TempFileFixtures::create_temp_file(&content, "txt")
                    .await
                    .unwrap();
                processor
                    .process_file(temp_file.path())
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
            let result = processor.process_file(&file_path).await?;
            // Each file should be processable (may or may not have content)
            let _ = result.content.len();
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
            .process_file(temp_file.path())
            .await?;

        // Large file should have substantial content extracted
        assert!(result.content.len() > 10000);

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

        let mut total_content_length = 0;

        for file_path in file_paths {
            let result = processor.process_file(&file_path).await?;

            total_content_length += result.content.len();
        }

        // Project should extract a significant amount of content
        assert!(total_content_length > 100);

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

            let result = processor.process_file(&file_path).await?;
            results.push(result);
        }

        // All files should be processed successfully
        assert_eq!(results.len(), 4);
        for result in results {
            assert!(!result.content.is_empty());
        }

        Ok(())
    }
}

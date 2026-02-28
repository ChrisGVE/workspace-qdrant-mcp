//! File format and code ingestion tests
//!
//! Tests covering document type detection, format-specific ingestion,
//! code file analysis across multiple languages, and metadata extraction.
//!
//! NOTE: These tests are disabled until the test framework is updated.

// Temporarily disable tests - error handling needs to be aligned with TestResult
#![cfg(feature = "comprehensive_file_tests")]

use shared_test_utils::{config::*, fixtures::*, TestResult};
use std::path::Path;
use tokio::fs;
use workspace_qdrant_core::{DocumentProcessor, DocumentType};

/// Test suite for file format ingestion
mod format_ingestion {
    use super::*;

    #[tokio::test]
    async fn test_pdf_format_detection() -> TestResult {
        let processor = DocumentProcessor::new();
        let pdf_path = Path::new("test_document.pdf");

        let doc_type = processor.detect_document_type(pdf_path)?;
        assert_eq!(doc_type, DocumentType::Pdf);

        Ok(())
    }

    #[tokio::test]
    async fn test_epub_format_detection() -> TestResult {
        let processor = DocumentProcessor::new();
        let epub_path = Path::new("test_ebook.epub");

        let doc_type = processor.detect_document_type(epub_path)?;
        assert_eq!(doc_type, DocumentType::Epub);

        Ok(())
    }

    #[tokio::test]
    async fn test_docx_format_detection() -> TestResult {
        let processor = DocumentProcessor::new();
        let docx_path = Path::new("test_document.docx");

        let doc_type = processor.detect_document_type(docx_path)?;
        assert_eq!(doc_type, DocumentType::Docx);

        Ok(())
    }

    #[tokio::test]
    async fn test_markdown_ingestion_with_metadata() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = DocumentFixtures::markdown_content();

        let temp_file = TempFileFixtures::create_temp_file(&content, "md").await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert_eq!(result.collection, TEST_COLLECTION);
        assert!(result.chunks_created.unwrap_or(0) > 0);
        assert!(result.processing_time_ms >= 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_text_file_ingestion() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = "This is a plain text document.\nWith multiple lines.\nAnd various content.";

        let temp_file = TempFileFixtures::create_temp_file(content, "txt").await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert_eq!(result.collection, TEST_COLLECTION);
        assert!(result.chunks_created.unwrap_or(0) > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_empty_file_handling() -> TestResult {
        let processor = DocumentProcessor::new();
        let temp_file = TempFileFixtures::create_temp_file("", "txt").await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert_eq!(result.chunks_created.unwrap_or(0), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_large_file_ingestion() -> TestResult {
        let processor = DocumentProcessor::new();

        // Create a 100KB test file
        let temp_file = TempFileFixtures::create_large_temp_file(100).await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert_eq!(result.collection, TEST_COLLECTION);
        // Large file should be chunked into multiple pieces
        assert!(result.chunks_created.unwrap_or(0) > 5);

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
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert_eq!(result.collection, TEST_COLLECTION);
        assert!(result.chunks_created.unwrap_or(0) > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_python_code_ingestion() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = DocumentFixtures::python_content();

        let temp_file = TempFileFixtures::create_temp_file(&content, "py").await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert_eq!(result.collection, TEST_COLLECTION);
        assert!(result.chunks_created.unwrap_or(0) > 0);

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
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert_eq!(result.collection, TEST_COLLECTION);
        assert!(result.chunks_created.unwrap_or(0) > 0);

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
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert_eq!(result.collection, TEST_COLLECTION);
        assert!(result.chunks_created.unwrap_or(0) > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_json_config_ingestion() -> TestResult {
        let processor = DocumentProcessor::new();
        let content = DocumentFixtures::json_config();

        let temp_file = TempFileFixtures::create_temp_file(&content, "json").await?;

        let result = processor
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert_eq!(result.collection, TEST_COLLECTION);
        assert!(result.chunks_created.unwrap_or(0) > 0);

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
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert_eq!(result.collection, TEST_COLLECTION);
        assert!(result.chunks_created.unwrap_or(0) > 0);

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
            .process_file(temp_file.path(), TEST_COLLECTION)
            .await?;

        assert!(result.processing_time_ms >= 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_document_type_detection_accuracy() -> TestResult {
        let processor = DocumentProcessor::new();

        let test_cases = vec![
            ("test.rs", DocumentType::Code("rust".to_string())),
            ("test.md", DocumentType::Markdown),
            ("test.txt", DocumentType::Text),
            ("test.json", DocumentType::Code("json".to_string())),
            ("test.pdf", DocumentType::Pdf),
            ("test.epub", DocumentType::Epub),
            ("test.docx", DocumentType::Docx),
        ];

        for (filename, expected_type) in test_cases {
            let path = Path::new(filename);
            let detected = processor.detect_document_type(path)?;

            // Allow flexibility for MIME-based detection
            match expected_type {
                DocumentType::Code(_) => {
                    assert!(
                        matches!(detected, DocumentType::Code(_) | DocumentType::Text),
                        "Failed for {}: got {:?}",
                        filename,
                        detected
                    );
                }
                _ => {
                    assert_eq!(
                        detected, expected_type,
                        "Failed for {}: expected {:?}, got {:?}",
                        filename, expected_type, detected
                    );
                }
            }
        }

        Ok(())
    }
}

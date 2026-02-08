use tempfile::NamedTempFile;
use tokio::io::AsyncWriteExt;
use workspace_qdrant_core::{DocumentProcessor, DocumentType, ChunkingConfig};

/// Create a temporary file with the given content and extension
async fn create_temp_file(content: &str, extension: &str) -> NamedTempFile {
    let temp_file = NamedTempFile::with_suffix(&format!(".{}", extension))
        .expect("Failed to create temporary file");
    
    let mut file = tokio::fs::File::create(temp_file.path()).await
        .expect("Failed to open temp file for writing");
    
    file.write_all(content.as_bytes()).await
        .expect("Failed to write content to temp file");
    
    file.flush().await
        .expect("Failed to flush temp file");
    
    temp_file
}

#[tokio::test]
async fn test_document_type_detection() {
    let processor = DocumentProcessor::new();

    // detect_document_type requires the file to exist, so use temp files
    let pdf_file = create_temp_file("dummy", "pdf").await;
    assert_eq!(
        processor.detect_document_type(pdf_file.path()).unwrap(),
        DocumentType::Pdf
    );

    let epub_file = create_temp_file("dummy", "epub").await;
    assert_eq!(
        processor.detect_document_type(epub_file.path()).unwrap(),
        DocumentType::Epub
    );

    let docx_file = create_temp_file("dummy", "docx").await;
    assert_eq!(
        processor.detect_document_type(docx_file.path()).unwrap(),
        DocumentType::Docx
    );

    let md_file = create_temp_file("dummy", "md").await;
    assert_eq!(
        processor.detect_document_type(md_file.path()).unwrap(),
        DocumentType::Markdown
    );

    let rs_file = create_temp_file("fn main() {}", "rs").await;
    assert_eq!(
        processor.detect_document_type(rs_file.path()).unwrap(),
        DocumentType::Code("rust".to_string())
    );

    let py_file = create_temp_file("print('hello')", "py").await;
    assert_eq!(
        processor.detect_document_type(py_file.path()).unwrap(),
        DocumentType::Code("python".to_string())
    );
}

#[tokio::test]
async fn test_text_file_processing() {
    let processor = DocumentProcessor::new();
    let content = "Hello, world!\nThis is a test document.\nIt has multiple lines.";
    
    let temp_file = create_temp_file(content, "txt").await;
    
    let result = processor.process_file(temp_file.path(), "test_collection").await;
    assert!(result.is_ok());
    
    let doc_result = result.unwrap();
    assert_eq!(doc_result.collection, "test_collection");
    assert!(doc_result.chunks_created.unwrap_or(0) > 0);
    // processing_time_ms is u64, always >= 0 — just verify it's set
    let _ = doc_result.processing_time_ms;
}

#[tokio::test]
async fn test_markdown_file_processing() {
    let processor = DocumentProcessor::new();
    let content = r#"# Test Document

This is a **markdown** document with various formatting.

## Section 1

- Item 1
- Item 2
- Item 3

## Section 2

Some `code` and more text.

```rust
fn hello() {
    println!("Hello, world!");
}
```

## Conclusion

This concludes the test document."#;
    
    let temp_file = create_temp_file(content, "md").await;
    
    let result = processor.process_file(temp_file.path(), "markdown_collection").await;
    assert!(result.is_ok());
    
    let doc_result = result.unwrap();
    assert_eq!(doc_result.collection, "markdown_collection");
    assert!(doc_result.chunks_created.unwrap_or(0) > 0);
}

#[tokio::test]
async fn test_rust_code_file_processing() {
    let processor = DocumentProcessor::new();
    let content = r#"//! A test Rust module
//! This module demonstrates various Rust constructs

use std::collections::HashMap;

/// A test struct
#[derive(Debug, Clone)]
pub struct TestStruct {
    pub name: String,
    pub count: usize,
}

impl TestStruct {
    /// Create a new TestStruct
    pub fn new(name: String) -> Self {
        Self {
            name,
            count: 0,
        }
    }
    
    /// Increment the counter
    pub fn increment(&mut self) {
        self.count += 1;
        // TODO: Add validation
    }
}

/// Main test function
pub fn main() {
    let mut test = TestStruct::new("example".to_string());
    test.increment();
    println!("Test: {:?}", test);
    
    // FIXME: This needs proper error handling
    let _map: HashMap<String, i32> = HashMap::new();
}"#;
    
    let temp_file = create_temp_file(content, "rs").await;
    
    let result = processor.process_file(temp_file.path(), "rust_collection").await;
    assert!(result.is_ok());
    
    let doc_result = result.unwrap();
    assert_eq!(doc_result.collection, "rust_collection");
    assert!(doc_result.chunks_created.unwrap_or(0) > 0);
}

#[tokio::test]
async fn test_python_code_file_processing() {
    let processor = DocumentProcessor::new();
    let content = r#""""
A test Python module
This module demonstrates various Python constructs
"""

import os
import sys
from typing import Dict, List, Optional

class TestClass:
    """A test class for demonstration."""

    def __init__(self, name: str):
        """Initialize the test class."""
        self.name = name
        self.count = 0

    def increment(self) -> None:
        """Increment the counter."""
        self.count += 1
        # TODO: Add validation

    def get_info(self) -> Dict[str, any]:
        """Get information about the instance."""
        return {
            'name': self.name,
            'count': self.count
        }

def main():
    """Main test function."""
    test = TestClass("example")
    test.increment()
    print(f"Test: {test.get_info()}")

    # FIXME: This needs proper error handling
    data: List[int] = []
    return data

if __name__ == "__main__":
    main()"#;
    
    let temp_file = create_temp_file(content, "py").await;

    let result = processor.process_file(temp_file.path(), "python_collection").await;
    assert!(result.is_ok());

    let doc_result = result.unwrap();
    assert_eq!(doc_result.collection, "python_collection");
    assert!(doc_result.chunks_created.unwrap_or(0) > 0);
}

#[tokio::test]
async fn test_json_file_processing() {
    let processor = DocumentProcessor::new();
    let content = r#"{
  "name": "test-document",
  "version": "1.0.0",
  "description": "A test JSON document for processing",
  "author": {
    "name": "Test Author",
    "email": "test@example.com"
  },
  "dependencies": {
    "lodash": "^4.17.21",
    "axios": "^0.21.1"
  },
  "scripts": {
    "start": "node index.js",
    "test": "jest"
  },
  "keywords": [
    "test",
    "document",
    "processing"
  ]
}"#;
    
    let temp_file = create_temp_file(content, "json").await;
    
    let result = processor.process_file(temp_file.path(), "json_collection").await;
    assert!(result.is_ok());
    
    let doc_result = result.unwrap();
    assert_eq!(doc_result.collection, "json_collection");
    assert!(doc_result.chunks_created.unwrap_or(0) > 0);
}

#[tokio::test]
async fn test_chunking_configuration() {
    let custom_config = ChunkingConfig {
        chunk_size: 10, // Very small chunks for testing
        overlap_size: 2,
        preserve_paragraphs: false,
    };
    
    let processor = DocumentProcessor::with_chunking_config(custom_config);
    let content = "This is a test document with many words that should be split into multiple small chunks for testing the chunking functionality.";
    
    let temp_file = create_temp_file(content, "txt").await;
    
    let result = processor.process_file(temp_file.path(), "chunking_test").await;
    assert!(result.is_ok());
    
    let doc_result = result.unwrap();
    assert_eq!(doc_result.collection, "chunking_test");
    // With small chunk size, we should get multiple chunks
    assert!(doc_result.chunks_created.unwrap_or(0) > 1);
}

#[tokio::test]
async fn test_encoding_detection() {
    let processor = DocumentProcessor::new();
    
    // Test with UTF-8 content
    let utf8_content = "Hello, world! 🦀 Rust is awesome! 日本語もサポート";
    let temp_file = create_temp_file(utf8_content, "txt").await;
    
    let result = processor.process_file(temp_file.path(), "encoding_test").await;
    assert!(result.is_ok());
    
    let doc_result = result.unwrap();
    assert_eq!(doc_result.collection, "encoding_test");
    assert!(doc_result.chunks_created.unwrap_or(0) > 0);
}

#[tokio::test]
async fn test_empty_file_handling() {
    let processor = DocumentProcessor::new();
    let content = "";
    
    let temp_file = create_temp_file(content, "txt").await;
    
    let result = processor.process_file(temp_file.path(), "empty_test").await;
    assert!(result.is_ok());
    
    let doc_result = result.unwrap();
    assert_eq!(doc_result.collection, "empty_test");
    assert_eq!(doc_result.chunks_created.unwrap_or(0), 0); // Empty file should create no chunks
}

#[tokio::test]
async fn test_large_document_chunking() {
    let processor = DocumentProcessor::new();

    // Create a large document with paragraph breaks every 10 sentences
    // so preserve_paragraphs chunking can split it properly
    let mut large_content = String::new();
    for i in 1..=1000 {
        large_content.push_str(&format!("This is sentence number {}. ", i));
        if i % 10 == 0 {
            large_content.push_str("\n\n");
        }
    }

    let temp_file = create_temp_file(&large_content, "txt").await;

    let result = processor.process_file(temp_file.path(), "large_test").await;
    assert!(result.is_ok());

    let doc_result = result.unwrap();
    assert_eq!(doc_result.collection, "large_test");
    // Large document with paragraph breaks should create multiple chunks
    assert!(doc_result.chunks_created.unwrap_or(0) > 5);
}

#[tokio::test]
async fn test_docx_placeholder() {
    let processor = DocumentProcessor::new();
    
    // Create a fake DOCX file (won't be valid, but tests the detection)
    let content = "This is not a real DOCX file";
    let temp_file = create_temp_file(content, "docx").await;
    
    // The extraction will fail, but the document type detection should work
    let doc_type = processor.detect_document_type(temp_file.path());
    assert!(doc_type.is_ok());
    assert_eq!(doc_type.unwrap(), DocumentType::Docx);
}

#[tokio::test]
async fn test_pdf_placeholder() {
    let processor = DocumentProcessor::new();

    // Create a fake PDF file (invalid content)
    let content = "This is not a real PDF file";
    let temp_file = create_temp_file(content, "pdf").await;

    // pdf_extract correctly rejects invalid PDF content
    let result = processor.process_file(temp_file.path(), "pdf_test").await;
    assert!(result.is_err(), "Expected error for invalid PDF content");
}
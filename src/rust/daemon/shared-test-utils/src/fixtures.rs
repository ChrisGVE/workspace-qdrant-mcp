//! Test fixtures and sample data generators

use std::path::PathBuf;
use tempfile::{NamedTempFile, TempDir};
use tokio::fs;
use tokio::io::AsyncWriteExt;
// use uuid::Uuid; // Unused

use crate::config::TEST_EMBEDDING_DIM;
use crate::TestResult;

/// Test document content generator
pub struct DocumentFixtures;

impl DocumentFixtures {
    /// Generate a sample Markdown document
    pub fn markdown_content() -> String {
        r#"# Test Document

This is a comprehensive test document with various Markdown features.

## Section 1: Basic Text

This section contains regular paragraph text with **bold** and *italic* formatting.
It also includes `inline code` and [links](https://example.com).

## Section 2: Lists

### Unordered List
- First item
- Second item
  - Nested item
  - Another nested item
- Third item

### Ordered List
1. First numbered item
2. Second numbered item
3. Third numbered item

## Section 3: Code Blocks

Here's a Rust code example:

```rust
fn main() {
    println!("Hello, world!");
    let numbers = vec![1, 2, 3, 4, 5];
    let sum: i32 = numbers.iter().sum();
    println!("Sum: {}", sum);
}
```

## Section 4: Tables

| Name | Age | City |
|------|-----|------|
| Alice | 30 | New York |
| Bob | 25 | San Francisco |
| Charlie | 35 | Chicago |

## Section 5: Quotes

> This is a blockquote with multiple lines.
> It can span several lines and include *formatting*.

## Conclusion

This document provides a comprehensive test case for Markdown parsing
and chunking algorithms.
"#.to_string()
    }

    /// Generate a sample Python code file
    pub fn python_content() -> String {
        r#"#!/usr/bin/env python3
"""
A comprehensive Python module for testing code parsing and analysis.

This module demonstrates various Python constructs including:
- Classes and inheritance
- Async/await patterns
- Type hints
- Decorators
- Context managers
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Generic, TypeVar
from contextlib import asynccontextmanager


T = TypeVar('T')


@dataclass
class DataPoint:
    """Represents a single data point with metadata."""
    id: str
    value: float
    timestamp: int
    metadata: Dict[str, str]

    def __post_init__(self):
        if self.value < 0:
            raise ValueError("Value must be non-negative")


class BaseProcessor(ABC, Generic[T]):
    """Abstract base class for data processors."""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"processor.{name}")

    @abstractmethod
    async def process(self, data: T) -> T:
        """Process a single data item."""
        pass

    @abstractmethod
    def validate(self, data: T) -> bool:
        """Validate a data item."""
        pass


class DocumentProcessor(BaseProcessor[str]):
    """Concrete processor for text documents."""

    def __init__(self, name: str, max_length: int = 1000):
        super().__init__(name)
        self.max_length = max_length

    async def process(self, data: str) -> str:
        """Process a document by cleaning and truncating."""
        self.logger.info(f"Processing document of length {len(data)}")

        # Simulate async processing
        await asyncio.sleep(0.01)

        # Clean and truncate
        cleaned = data.strip().replace('\n\n', '\n')
        if len(cleaned) > self.max_length:
            cleaned = cleaned[:self.max_length] + "..."

        return cleaned

    def validate(self, data: str) -> bool:
        """Validate that the document is not empty."""
        return bool(data and data.strip())

    @asynccontextmanager
    async def batch_context(self):
        """Context manager for batch processing."""
        self.logger.info("Starting batch processing")
        try:
            yield self
        finally:
            self.logger.info("Finished batch processing")


async def process_documents(
    processor: DocumentProcessor,
    documents: List[str]
) -> List[str]:
    """Process multiple documents concurrently."""
    async with processor.batch_context():
        tasks = [processor.process(doc) for doc in documents]
        return await asyncio.gather(*tasks)


def main():
    """Main function demonstrating the processor."""
    logging.basicConfig(level=logging.INFO)

    processor = DocumentProcessor("test_processor")
    test_doc = "This is a test document with some content."

    async def run():
        result = await processor.process(test_doc)
        print(f"Processed: {result}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
"#.to_string()
    }

    /// Generate a sample Rust code file
    pub fn rust_content() -> String {
        r#"//! A comprehensive Rust module for testing code parsing and analysis.
//!
//! This module demonstrates various Rust constructs including:
//! - Structs and enums
//! - Traits and implementations
//! - Async/await patterns
//! - Error handling
//! - Generics and lifetimes

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use tokio::time::{sleep, Duration};

/// Custom error type for the processor
#[derive(Debug)]
pub enum ProcessorError {
    ValidationError(String),
    ProcessingError(String),
    IoError(std::io::Error),
}

impl fmt::Display for ProcessorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProcessorError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            ProcessorError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
            ProcessorError::IoError(err) => write!(f, "IO error: {}", err),
        }
    }
}

impl Error for ProcessorError {}

impl From<std::io::Error> for ProcessorError {
    fn from(err: std::io::Error) -> Self {
        ProcessorError::IoError(err)
    }
}

/// Configuration for the document processor
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    pub max_length: usize,
    pub timeout_ms: u64,
    pub enable_caching: bool,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            max_length: 1000,
            timeout_ms: 5000,
            enable_caching: true,
        }
    }
}

/// A trait for objects that can be processed
pub trait Processable {
    type Output;
    type Error;

    fn validate(&self) -> Result<(), Self::Error>;
    async fn process(&self) -> Result<Self::Output, Self::Error>;
}

/// Generic document processor with configurable behavior
pub struct DocumentProcessor<T> {
    config: ProcessorConfig,
    cache: HashMap<String, T>,
    stats: ProcessingStats,
}

#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub documents_processed: usize,
    pub cache_hits: usize,
    pub total_processing_time_ms: u64,
}

impl<T> DocumentProcessor<T>
where
    T: Clone + fmt::Debug,
{
    /// Create a new processor with default configuration
    pub fn new() -> Self {
        Self::with_config(ProcessorConfig::default())
    }

    /// Create a new processor with custom configuration
    pub fn with_config(config: ProcessorConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            stats: ProcessingStats::default(),
        }
    }

    /// Get processing statistics
    pub fn stats(&self) -> &ProcessingStats {
        &self.stats
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Process with caching
    pub async fn process_with_cache(
        &mut self,
        key: String,
        processor_fn: impl Fn() -> Result<T, ProcessorError>,
    ) -> Result<T, ProcessorError> {
        let start = std::time::Instant::now();

        let result = if self.config.enable_caching {
            if let Some(cached) = self.cache.get(&key).cloned() {
                self.stats.cache_hits += 1;
                Ok(cached)
            } else {
                let processed = processor_fn()?;
                self.cache.insert(key, processed.clone());
                Ok(processed)
            }
        } else {
            processor_fn()
        };

        self.stats.documents_processed += 1;
        self.stats.total_processing_time_ms += start.elapsed().as_millis() as u64;

        result
    }
}

impl<T> Default for DocumentProcessor<T>
where
    T: Clone + fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Async function demonstrating complex processing
pub async fn batch_process_documents<T>(
    processor: &mut DocumentProcessor<T>,
    documents: Vec<(String, T)>,
) -> Result<Vec<T>, ProcessorError>
where
    T: Clone + fmt::Debug + Send + 'static,
{
    let mut results = Vec::new();

    for (key, doc) in documents {
        // Simulate async processing with timeout
        let result = tokio::time::timeout(
            Duration::from_millis(processor.config.timeout_ms),
            async move {
                sleep(Duration::from_millis(10)).await;
                Ok::<T, ProcessorError>(doc)
            }
        ).await;

        match result {
            Ok(Ok(processed)) => results.push(processed),
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(ProcessorError::ProcessingError("Timeout".to_string())),
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_processor_creation() {
        let processor: DocumentProcessor<String> = DocumentProcessor::new();
        assert_eq!(processor.stats().documents_processed, 0);
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let mut processor = DocumentProcessor::new();
        let documents = vec![
            ("doc1".to_string(), "content1".to_string()),
            ("doc2".to_string(), "content2".to_string()),
        ];

        let results = batch_process_documents(&mut processor, documents).await;
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 2);
    }
}
"#.to_string()
    }

    /// Generate a sample JSON configuration file
    pub fn json_config() -> String {
        serde_json::json!({
            "app_name": "workspace-qdrant-mcp",
            "version": "0.2.1",
            "config": {
                "qdrant": {
                    "url": "http://localhost:6333",
                    "api_key": null,
                    "timeout_ms": 5000,
                    "max_retries": 3
                },
                "embedding": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimension": 384,
                    "batch_size": 32,
                    "cache_size": 1000
                },
                "processing": {
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "max_workers": 4,
                    "queue_size": 100
                },
                "logging": {
                    "level": "info",
                    "format": "json",
                    "file": null
                }
            },
            "collections": [
                "default",
                "documents",
                "code"
            ],
            "features": {
                "auto_embedding": true,
                "file_watching": true,
                "hybrid_search": true,
                "caching": true
            }
        }).to_string()
    }
}

/// File system test utilities
pub struct TempFileFixtures;

impl TempFileFixtures {
    /// Create a temporary file with given content and extension
    pub async fn create_temp_file(content: &str, extension: &str) -> TestResult<NamedTempFile> {
        let temp_file = NamedTempFile::with_suffix(format!(".{}", extension))?;

        let mut file = fs::File::create(temp_file.path()).await?;
        file.write_all(content.as_bytes()).await?;
        file.flush().await?;

        Ok(temp_file)
    }

    /// Create a temporary directory with test files
    pub async fn create_temp_project() -> TestResult<(TempDir, Vec<PathBuf>)> {
        let temp_dir = TempDir::new()?;
        let mut file_paths = Vec::new();

        // Create a basic project structure
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&src_dir).await?;

        let docs_dir = temp_dir.path().join("docs");
        fs::create_dir(&docs_dir).await?;

        // Create sample files
        let files = vec![
            ("README.md", DocumentFixtures::markdown_content()),
            ("src/main.py", DocumentFixtures::python_content()),
            ("src/lib.rs", DocumentFixtures::rust_content()),
            ("config.json", DocumentFixtures::json_config()),
        ];

        for (relative_path, content) in files {
            let file_path = temp_dir.path().join(relative_path);

            // Ensure parent directory exists
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent).await?;
            }

            fs::write(&file_path, content).await?;
            file_paths.push(file_path);
        }

        Ok((temp_dir, file_paths))
    }

    /// Create a temporary file with random content for stress testing
    pub async fn create_large_temp_file(size_kb: usize) -> TestResult<NamedTempFile> {
        let temp_file = NamedTempFile::with_suffix(".txt")?;

        let chunk = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(10);
        let chunks_needed = (size_kb * 1024) / chunk.len() + 1;

        let mut file = fs::File::create(temp_file.path()).await?;
        for _ in 0..chunks_needed {
            file.write_all(chunk.as_bytes()).await?;
        }
        file.flush().await?;

        Ok(temp_file)
    }
}

/// Vector and embedding test utilities
pub struct EmbeddingFixtures;

impl EmbeddingFixtures {
    /// Generate a random embedding vector
    pub fn random_embedding() -> Vec<f32> {
        use std::f32::consts::PI;

        (0..TEST_EMBEDDING_DIM)
            .map(|i| (i as f32 * PI / TEST_EMBEDDING_DIM as f32).sin() * 0.5)
            .collect()
    }

    /// Generate embeddings for test documents
    pub fn test_document_embeddings() -> Vec<(String, Vec<f32>)> {
        vec![
            ("Machine learning is a subset of artificial intelligence".to_string(), Self::random_embedding()),
            ("Natural language processing enables computers to understand text".to_string(), Self::random_embedding()),
            ("Vector databases store high-dimensional data efficiently".to_string(), Self::random_embedding()),
            ("Semantic search finds meaning rather than exact matches".to_string(), Self::random_embedding()),
            ("Document chunking splits text into manageable pieces".to_string(), Self::random_embedding()),
        ]
    }

    /// Generate sparse vector for BM25 testing
    pub fn test_sparse_vector() -> (Vec<u32>, Vec<f32>) {
        let indices = vec![0, 5, 12, 23, 45, 67, 89, 123];
        let values = vec![0.8, 0.6, 0.9, 0.4, 0.7, 0.3, 0.5, 0.2];
        (indices, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_temp_file_creation() -> TestResult {
        let content = "test content";
        let temp_file = TempFileFixtures::create_temp_file(content, "txt").await?;

        let read_content = fs::read_to_string(temp_file.path()).await?;
        assert_eq!(read_content, content);

        Ok(())
    }

    #[tokio::test]
    async fn test_temp_project_creation() -> TestResult {
        let (_temp_dir, file_paths) = TempFileFixtures::create_temp_project().await?;

        assert_eq!(file_paths.len(), 4);

        for file_path in &file_paths {
            assert!(file_path.exists());
        }

        Ok(())
    }

    #[test]
    fn test_embedding_generation() {
        let embedding = EmbeddingFixtures::random_embedding();
        assert_eq!(embedding.len(), TEST_EMBEDDING_DIM);
    }

    #[test]
    fn test_document_fixtures() {
        let markdown = DocumentFixtures::markdown_content();
        assert!(markdown.contains("# Test Document"));
        assert!(markdown.contains("```rust"));

        let python = DocumentFixtures::python_content();
        assert!(python.contains("import asyncio"));
        assert!(python.contains("async def"));

        let rust = DocumentFixtures::rust_content();
        assert!(rust.contains("pub struct"));
        assert!(rust.contains("impl"));
    }
}
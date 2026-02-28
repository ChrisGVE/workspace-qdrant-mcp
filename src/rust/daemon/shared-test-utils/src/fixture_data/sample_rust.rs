//! A comprehensive Rust module for testing code parsing and analysis.
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

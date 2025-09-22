//! Document processing engine

use crate::config::{ProcessingConfig, QdrantConfig};
use crate::error::{DaemonError, DaemonResult};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{info, debug};

/// Document processor
#[derive(Debug)]
pub struct DocumentProcessor {
    config: ProcessingConfig,
    qdrant_config: QdrantConfig,
    semaphore: Arc<Semaphore>,
}

impl DocumentProcessor {
    /// Create a new document processor
    pub async fn new(config: &ProcessingConfig, qdrant_config: &QdrantConfig) -> DaemonResult<Self> {
        info!("Initializing document processor with max concurrent tasks: {}", config.max_concurrent_tasks);

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_tasks));

        Ok(Self {
            config: config.clone(),
            qdrant_config: qdrant_config.clone(),
            semaphore,
        })
    }

    /// Process a single document
    pub async fn process_document(&self, file_path: &str) -> DaemonResult<String> {
        let _permit = self.semaphore.acquire().await
            .map_err(|e| DaemonError::Internal { message: format!("Semaphore error: {}", e) })?;

        debug!("Processing document: {}", file_path);

        // TODO: Implement actual document processing
        // This is a placeholder
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        Ok(uuid::Uuid::new_v4().to_string())
    }

    /// Get processing configuration
    pub fn config(&self) -> &ProcessingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ProcessingConfig, QdrantConfig, CollectionConfig};
    use std::sync::Arc;

    fn create_test_processing_config() -> ProcessingConfig {
        ProcessingConfig {
            max_concurrent_tasks: 2,
            default_chunk_size: 1000,
            default_chunk_overlap: 200,
            max_file_size_bytes: 1024 * 1024,
            supported_extensions: vec!["rs".to_string(), "py".to_string()],
            enable_lsp: true,
            lsp_timeout_secs: 10,
        }
    }

    fn create_test_qdrant_config() -> QdrantConfig {
        QdrantConfig {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            timeout_secs: 30,
            max_retries: 3,
            default_collection: CollectionConfig {
                vector_size: 384,
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                replication_factor: 1,
                shard_number: 1,
            },
        }
    }

    #[tokio::test]
    async fn test_document_processor_new() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        // Test that the processor was created successfully
        assert_eq!(processor.config().max_concurrent_tasks, 2);
        assert_eq!(processor.config().default_chunk_size, 1000);
        assert_eq!(processor.config().default_chunk_overlap, 200);
        assert_eq!(processor.config().max_file_size_bytes, 1024 * 1024);
        assert!(processor.config().enable_lsp);
        assert_eq!(processor.config().lsp_timeout_secs, 10);

        // Test debug formatting
        let debug_str = format!("{:?}", processor);
        assert!(debug_str.contains("DocumentProcessor"));
    }

    #[tokio::test]
    async fn test_process_document() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        let result = processor.process_document("test_file.rs").await.unwrap();

        // Should return a UUID string
        assert_eq!(result.len(), 36); // UUID v4 string length
        assert!(result.contains('-'));
    }

    #[tokio::test]
    async fn test_concurrent_processing() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .unwrap(),
        );

        let mut handles = vec![];

        // Spawn multiple concurrent tasks
        for i in 0..4 {
            let processor_clone = Arc::clone(&processor);
            let handle = tokio::spawn(async move {
                processor_clone
                    .process_document(&format!("test_file_{}.rs", i))
                    .await
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results: Vec<_> = futures_util::future::join_all(handles).await;

        // All tasks should complete successfully
        for result in results {
            let task_result = result.unwrap();
            assert!(task_result.is_ok());
            let uuid_str = task_result.unwrap();
            assert_eq!(uuid_str.len(), 36);
        }
    }

    #[tokio::test]
    async fn test_config_access() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        let config = processor.config();
        assert_eq!(config.max_concurrent_tasks, 2);
        assert_eq!(config.default_chunk_size, 1000);
        assert_eq!(config.supported_extensions, vec!["rs", "py"]);
    }

    #[tokio::test]
    async fn test_semaphore_limits() {
        let mut processing_config = create_test_processing_config();
        processing_config.max_concurrent_tasks = 1; // Allow only 1 concurrent task
        let qdrant_config = create_test_qdrant_config();

        let processor = Arc::new(
            DocumentProcessor::new(&processing_config, &qdrant_config)
                .await
                .unwrap(),
        );

        let start_time = std::time::Instant::now();

        let processor1 = Arc::clone(&processor);
        let processor2 = Arc::clone(&processor);

        let handle1 = tokio::spawn(async move {
            processor1.process_document("test1.rs").await
        });

        let handle2 = tokio::spawn(async move {
            processor2.process_document("test2.rs").await
        });

        let (result1, result2) = tokio::join!(handle1, handle2);

        // Both should succeed
        assert!(result1.unwrap().is_ok());
        assert!(result2.unwrap().is_ok());

        // Should take at least 200ms due to semaphore limiting concurrency
        let elapsed = start_time.elapsed();
        assert!(elapsed >= std::time::Duration::from_millis(150));
    }

    #[tokio::test]
    async fn test_processor_with_different_configs() {
        let mut config1 = create_test_processing_config();
        config1.max_concurrent_tasks = 1;
        config1.default_chunk_size = 500;

        let mut config2 = create_test_processing_config();
        config2.max_concurrent_tasks = 5;
        config2.default_chunk_size = 2000;

        let qdrant_config = create_test_qdrant_config();

        let processor1 = DocumentProcessor::new(&config1, &qdrant_config)
            .await
            .unwrap();
        let processor2 = DocumentProcessor::new(&config2, &qdrant_config)
            .await
            .unwrap();

        assert_eq!(processor1.config().max_concurrent_tasks, 1);
        assert_eq!(processor1.config().default_chunk_size, 500);

        assert_eq!(processor2.config().max_concurrent_tasks, 5);
        assert_eq!(processor2.config().default_chunk_size, 2000);
    }

    #[tokio::test]
    async fn test_various_file_extensions() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        let test_files = vec![
            "test.rs",
            "script.py",
            "document.md",
            "data.json",
            "page.html",
        ];

        for file in test_files {
            let result = processor.process_document(file).await;
            assert!(result.is_ok(), "Failed to process file: {}", file);
            let uuid_str = result.unwrap();
            assert_eq!(uuid_str.len(), 36);
        }
    }
}
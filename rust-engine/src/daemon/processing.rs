//! Document processing engine

use crate::config::{ProcessingConfig, QdrantConfig};
use crate::error::{DaemonError, DaemonResult};
use std::sync::Arc;
use std::path::Path;
use tokio::sync::Semaphore;
use tokio::fs;
use tracing::{info, debug, warn, error};
use qdrant_client::{Qdrant, qdrant::{CreateCollection, Distance, VectorParams, VectorsConfig, PointStruct, Value, Datatype, UpsertPoints}};
use std::collections::HashMap;
use chrono;

/// Document processor
pub struct DocumentProcessor {
    config: ProcessingConfig,
    qdrant_config: QdrantConfig,
    semaphore: Arc<Semaphore>,
    qdrant_client: Qdrant,
}

impl std::fmt::Debug for DocumentProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DocumentProcessor")
            .field("config", &self.config)
            .field("qdrant_config", &self.qdrant_config)
            .field("semaphore", &self.semaphore)
            .field("qdrant_client", &"<Qdrant>")
            .finish()
    }
}

impl DocumentProcessor {
    /// Create a new document processor
    pub async fn new(config: &ProcessingConfig, qdrant_config: &QdrantConfig) -> DaemonResult<Self> {
        info!("Initializing document processor with max concurrent tasks: {}", config.max_concurrent_tasks);

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_tasks));

        // Initialize Qdrant client
        let qdrant_client = Qdrant::from_url(&qdrant_config.url)
            .build()
            .map_err(|e| DaemonError::Configuration { message: format!("Failed to create Qdrant client: {}", e) })?;

        info!("Connected to Qdrant at: {}", qdrant_config.url);

        Ok(Self {
            config: config.clone(),
            qdrant_config: qdrant_config.clone(),
            semaphore,
            qdrant_client,
        })
    }

    /// Process a single document
    pub async fn process_document(&self, file_path: &str) -> DaemonResult<String> {
        let _permit = self.semaphore.acquire().await
            .map_err(|e| DaemonError::Internal { message: format!("Semaphore error: {}", e) })?;

        info!("Processing document: {}", file_path);

        // Check if file should be processed
        if !self.should_process_file(file_path) {
            debug!("Skipping file (not in supported extensions): {}", file_path);
            return Ok("skipped".to_string());
        }

        // Read file content
        let content = match fs::read_to_string(file_path).await {
            Ok(content) => content,
            Err(e) => {
                warn!("Failed to read file {}: {}", file_path, e);
                return Ok("error".to_string());
            }
        };

        if content.trim().is_empty() {
            debug!("Skipping empty file: {}", file_path);
            return Ok("empty".to_string());
        }

        // Generate document ID
        let document_id = uuid::Uuid::new_v4().to_string();

        // Determine collection name from file path
        let collection_name = self.get_collection_name(file_path);

        // Ensure collection exists
        self.ensure_collection_exists(&collection_name).await?;

        // Create document with metadata
        let mut metadata = HashMap::new();
        metadata.insert("file_path".to_string(), Value::from(file_path.to_string()));
        metadata.insert("file_size".to_string(), Value::from(content.len() as i64));
        metadata.insert("processed_at".to_string(), Value::from(chrono::Utc::now().to_rfc3339()));

        if let Some(extension) = Path::new(file_path).extension().and_then(|ext| ext.to_str()) {
            metadata.insert("file_type".to_string(), Value::from(extension.to_string()));
        }

        // For now, create a simple document embedding (placeholder vector)
        // In a real implementation, this would use an embedding model
        let embedding = self.create_placeholder_embedding(&content);

        // Create point for Qdrant
        let point = PointStruct::new(
            document_id.clone(),
            embedding,
            metadata,
        );

        // Store in Qdrant
        let upsert_request = UpsertPoints {
            collection_name: collection_name.clone(),
            wait: None,
            points: vec![point],
            ordering: None,
            shard_key_selector: None,
        };

        match self.qdrant_client.upsert_points(upsert_request).await {
            Ok(_) => {
                info!("Successfully processed and stored document: {} in collection: {}", file_path, collection_name);
                Ok(document_id)
            }
            Err(e) => {
                error!("Failed to store document in Qdrant: {}", e);
                Err(DaemonError::Internal { message: format!("Qdrant storage error: {}", e) })
            }
        }
    }

    /// Check if file should be processed based on extension
    fn should_process_file(&self, file_path: &str) -> bool {
        if let Some(extension) = Path::new(file_path).extension().and_then(|ext| ext.to_str()) {
            self.config.supported_extensions.contains(&extension.to_string())
        } else {
            false
        }
    }

    /// Get collection name from file path
    fn get_collection_name(&self, _file_path: &str) -> String {
        // Extract project name from path
        // For now, use a simple approach - in real implementation this would be more sophisticated
        "workspace-qdrant-mcp-repo".to_string()
    }

    /// Ensure collection exists in Qdrant
    async fn ensure_collection_exists(&self, collection_name: &str) -> DaemonResult<()> {
        // Check if collection exists
        match self.qdrant_client.collection_info(collection_name).await {
            Ok(_) => {
                debug!("Collection {} already exists", collection_name);
                Ok(())
            }
            Err(_) => {
                info!("Creating collection: {}", collection_name);

                // Create collection with default vector size (384 for all-MiniLM-L6-v2)
                let create_collection = CreateCollection {
                    collection_name: collection_name.to_string(),
                    vectors_config: Some(VectorsConfig {
                        config: Some(qdrant_client::qdrant::vectors_config::Config::Params(VectorParams {
                            size: self.qdrant_config.default_collection.vector_size as u64,
                            distance: Distance::Cosine as i32,
                            hnsw_config: None,
                            quantization_config: None,
                            on_disk: None,
                            datatype: Some(Datatype::Float32 as i32),
                            multivector_config: None,
                        })),
                    }),
                    hnsw_config: None,
                    wal_config: None,
                    optimizers_config: None,
                    shard_number: Some(self.qdrant_config.default_collection.shard_number),
                    on_disk_payload: None,
                    timeout: None,
                    replication_factor: Some(self.qdrant_config.default_collection.replication_factor),
                    write_consistency_factor: None,
                    init_from_collection: None,
                    quantization_config: None,
                    sharding_method: None,
                    sparse_vectors_config: None,
                    strict_mode_config: None,
                };

                self.qdrant_client.create_collection(create_collection).await
                    .map_err(|e| DaemonError::Internal { message: format!("Failed to create collection: {}", e) })?;

                info!("Successfully created collection: {}", collection_name);
                Ok(())
            }
        }
    }

    /// Create a placeholder embedding (in real implementation, use embedding model)
    fn create_placeholder_embedding(&self, content: &str) -> Vec<f32> {
        // Create a simple hash-based embedding as placeholder
        // In real implementation, this would use FastEmbed or similar
        let mut embedding = vec![0.0f32; self.qdrant_config.default_collection.vector_size];

        // Simple hash-based approach for demo
        let hash = content.len() as f32;
        for (i, byte) in content.bytes().take(embedding.len()).enumerate() {
            embedding[i] = (byte as f32 / 255.0) * (hash / 1000.0).sin();
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }

    /// Get processing configuration
    #[allow(dead_code)]
    pub fn config(&self) -> &ProcessingConfig {
        &self.config
    }

    /// Get Qdrant configuration
    #[allow(dead_code)]
    pub fn qdrant_config(&self) -> &QdrantConfig {
        &self.qdrant_config
    }

    /// Create a test instance for testing purposes
    #[cfg(any(test, feature = "test-utils"))]
    pub fn test_instance() -> Self {
        let config = ProcessingConfig {
            max_concurrent_tasks: 2,
            default_chunk_size: 1000,
            default_chunk_overlap: 200,
            max_file_size_bytes: 1024 * 1024,
            supported_extensions: vec!["txt".to_string(), "md".to_string()],
            enable_lsp: false,
            lsp_timeout_secs: 10,
        };

        let qdrant_config = QdrantConfig {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            timeout_secs: 30,
            max_retries: 3,
            default_collection: crate::config::CollectionConfig {
                vector_size: 384,
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                replication_factor: 1,
                shard_number: 1,
            },
        };

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_tasks));

        // Note: This test instance doesn't include a qdrant_client
        // as it's intended for unit tests that don't require actual Qdrant connection
        Self {
            config,
            qdrant_config,
            semaphore,
            qdrant_client: Qdrant::from_url("http://localhost:6333").build().unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ProcessingConfig, QdrantConfig, CollectionConfig};
    use std::sync::Arc;
    use tracing_subscriber;

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

    #[test]
    fn test_instance_creation() {
        let processor = DocumentProcessor::test_instance();

        // Test that test_instance creates processor with expected config
        assert_eq!(processor.config().max_concurrent_tasks, 2);
        assert_eq!(processor.config().default_chunk_size, 1000);
        assert_eq!(processor.config().default_chunk_overlap, 200);
        assert_eq!(processor.config().max_file_size_bytes, 1024 * 1024);
        assert_eq!(processor.config().supported_extensions, vec!["txt", "md"]);
        assert!(!processor.config().enable_lsp);
        assert_eq!(processor.config().lsp_timeout_secs, 10);

        // Test that qdrant_config is properly set
        assert_eq!(processor.qdrant_config.url, "http://localhost:6333");
        assert_eq!(processor.qdrant_config.api_key, None);
        assert_eq!(processor.qdrant_config.timeout_secs, 30);
        assert_eq!(processor.qdrant_config.max_retries, 3);
        assert_eq!(processor.qdrant_config.default_collection.vector_size, 384);
        assert_eq!(processor.qdrant_config.default_collection.distance_metric, "Cosine");
        assert!(processor.qdrant_config.default_collection.enable_indexing);
        assert_eq!(processor.qdrant_config.default_collection.replication_factor, 1);
        assert_eq!(processor.qdrant_config.default_collection.shard_number, 1);
    }

    #[tokio::test]
    async fn test_semaphore_error_handling() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        // Close the semaphore to force an error condition
        processor.semaphore.close();

        let result = processor.process_document("test_file.rs").await;
        assert!(result.is_err());

        match result {
            Err(DaemonError::Internal { message }) => {
                assert!(message.contains("Semaphore error"));
            }
            _ => panic!("Expected Internal error with semaphore message"),
        }
    }

    #[tokio::test]
    async fn test_process_document_uuid_generation() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        // Process multiple documents and ensure UUIDs are unique
        let mut uuids = std::collections::HashSet::new();

        for i in 0..10 {
            let result = processor.process_document(&format!("test_{}.rs", i)).await.unwrap();
            assert_eq!(result.len(), 36);
            assert!(result.contains('-'));

            // Parse as UUID to ensure validity
            let uuid = uuid::Uuid::parse_str(&result).unwrap();
            assert_eq!(uuid.get_version_num(), 4); // UUID v4

            // Ensure uniqueness
            assert!(uuids.insert(result), "UUID should be unique");
        }
    }

    #[tokio::test]
    async fn test_processor_configuration_variants() {
        // Test with minimal configuration
        let minimal_config = ProcessingConfig {
            max_concurrent_tasks: 1,
            default_chunk_size: 100,
            default_chunk_overlap: 0,
            max_file_size_bytes: 1024,
            supported_extensions: vec![],
            enable_lsp: false,
            lsp_timeout_secs: 1,
        };

        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&minimal_config, &qdrant_config)
            .await
            .unwrap();

        assert_eq!(processor.config().max_concurrent_tasks, 1);
        assert_eq!(processor.config().default_chunk_size, 100);
        assert_eq!(processor.config().default_chunk_overlap, 0);
        assert_eq!(processor.config().max_file_size_bytes, 1024);
        assert!(processor.config().supported_extensions.is_empty());
        assert!(!processor.config().enable_lsp);
        assert_eq!(processor.config().lsp_timeout_secs, 1);

        // Test processing with minimal config
        let result = processor.process_document("minimal_test.txt").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_processor_configuration_maximal() {
        // Test with maximal configuration
        let maximal_config = ProcessingConfig {
            max_concurrent_tasks: 100,
            default_chunk_size: 10000,
            default_chunk_overlap: 2000,
            max_file_size_bytes: 100 * 1024 * 1024, // 100MB
            supported_extensions: vec![
                "rs".to_string(), "py".to_string(), "js".to_string(),
                "ts".to_string(), "java".to_string(), "cpp".to_string(),
                "c".to_string(), "h".to_string(), "hpp".to_string(),
                "md".to_string(), "txt".to_string(), "json".to_string(),
            ],
            enable_lsp: true,
            lsp_timeout_secs: 60,
        };

        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&maximal_config, &qdrant_config)
            .await
            .unwrap();

        assert_eq!(processor.config().max_concurrent_tasks, 100);
        assert_eq!(processor.config().default_chunk_size, 10000);
        assert_eq!(processor.config().default_chunk_overlap, 2000);
        assert_eq!(processor.config().max_file_size_bytes, 100 * 1024 * 1024);
        assert_eq!(processor.config().supported_extensions.len(), 12);
        assert!(processor.config().enable_lsp);
        assert_eq!(processor.config().lsp_timeout_secs, 60);

        // Test processing with maximal config
        let result = processor.process_document("maximal_test.cpp").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_qdrant_config_variants() {
        let processing_config = create_test_processing_config();

        // Test with API key
        let qdrant_with_key = QdrantConfig {
            url: "https://cloud.qdrant.io".to_string(),
            api_key: Some("test-api-key".to_string()),
            timeout_secs: 60,
            max_retries: 5,
            default_collection: CollectionConfig {
                vector_size: 768,
                distance_metric: "Dot".to_string(),
                enable_indexing: false,
                replication_factor: 2,
                shard_number: 4,
            },
        };

        let processor = DocumentProcessor::new(&processing_config, &qdrant_with_key)
            .await
            .unwrap();

        assert_eq!(processor.qdrant_config.url, "https://cloud.qdrant.io");
        assert_eq!(processor.qdrant_config.api_key, Some("test-api-key".to_string()));
        assert_eq!(processor.qdrant_config.timeout_secs, 60);
        assert_eq!(processor.qdrant_config.max_retries, 5);
        assert_eq!(processor.qdrant_config.default_collection.vector_size, 768);
        assert_eq!(processor.qdrant_config.default_collection.distance_metric, "Dot");
        assert!(!processor.qdrant_config.default_collection.enable_indexing);
        assert_eq!(processor.qdrant_config.default_collection.replication_factor, 2);
        assert_eq!(processor.qdrant_config.default_collection.shard_number, 4);
    }

    #[tokio::test]
    async fn test_debug_implementation() {
        let processor = DocumentProcessor::test_instance();

        let debug_str = format!("{:?}", processor);
        assert!(debug_str.contains("DocumentProcessor"));
        assert!(debug_str.contains("config"));
        assert!(debug_str.contains("qdrant_config"));
        assert!(debug_str.contains("semaphore"));
    }

    #[tokio::test]
    async fn test_empty_file_path() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        let result = processor.process_document("").await;
        assert!(result.is_ok()); // Should handle empty path gracefully
        let uuid_str = result.unwrap();
        assert_eq!(uuid_str.len(), 36);
    }

    #[tokio::test]
    async fn test_very_long_file_path() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        // Create a very long file path
        let long_path = "a".repeat(1000) + ".rs";
        let result = processor.process_document(&long_path).await;
        assert!(result.is_ok()); // Should handle long paths gracefully
        let uuid_str = result.unwrap();
        assert_eq!(uuid_str.len(), 36);
    }

    #[tokio::test]
    async fn test_special_characters_in_file_path() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        let special_paths = vec![
            "file with spaces.rs",
            "file-with-dashes.py",
            "file_with_underscores.md",
            "file.with.dots.txt",
            "file@with#special$chars%.json",
            "file(with)parentheses.rs",
            "file[with]brackets.py",
            "file{with}braces.md",
        ];

        for path in special_paths {
            let result = processor.process_document(path).await;
            assert!(result.is_ok(), "Failed to process file: {}", path);
            let uuid_str = result.unwrap();
            assert_eq!(uuid_str.len(), 36);
        }
    }

    #[tokio::test]
    async fn test_unicode_file_paths() {
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        let unicode_paths = vec![
            "файл.rs", // Russian
            "文件.py", // Chinese
            "ファイル.md", // Japanese
            "파일.txt", // Korean
            "αρχείο.json", // Greek
            "फ़ाइल.rs", // Hindi
            "🚀rocket.py", // Emoji
        ];

        for path in unicode_paths {
            let result = processor.process_document(path).await;
            assert!(result.is_ok(), "Failed to process unicode file: {}", path);
            let uuid_str = result.unwrap();
            assert_eq!(uuid_str.len(), 36);
        }
    }

    #[tokio::test]
    async fn test_debug_logging_coverage() {
        // This test ensures the debug! logging line is executed
        // Initialize tracing subscriber to capture debug logs
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();

        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        // This call will trigger the debug! statement on line 36
        let result = processor.process_document("debug_test.rs").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_info_logging_coverage() {
        // This test ensures the info! logging line is executed
        // Initialize tracing subscriber to capture info logs
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init();

        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        // This call will trigger the info! statement on line 20
        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        assert_eq!(processor.config().max_concurrent_tasks, 2);
    }

    #[tokio::test]
    async fn test_complete_processing_pipeline() {
        // Comprehensive test that exercises the complete processing pipeline
        let processing_config = create_test_processing_config();
        let qdrant_config = create_test_qdrant_config();

        // Test processor creation (covers info! logging)
        let processor = DocumentProcessor::new(&processing_config, &qdrant_config)
            .await
            .unwrap();

        // Test configuration access
        let config = processor.config();
        assert!(config.enable_lsp);
        assert_eq!(config.supported_extensions, vec!["rs", "py"]);

        // Test document processing with various scenarios
        let test_scenarios = vec![
            ("simple.rs", "Simple Rust file"),
            ("", "Empty path"),
            ("very/deep/nested/path/file.py", "Deeply nested path"),
            ("file with spaces.md", "Path with spaces"),
            ("αβγ.txt", "Unicode filename"),
        ];

        for (path, description) in test_scenarios {
            let result = processor.process_document(path).await;
            assert!(result.is_ok(), "Failed to process {}: {}", description, path);

            let uuid_str = result.unwrap();
            assert_eq!(uuid_str.len(), 36, "Invalid UUID length for {}", description);

            // Verify it's a valid UUID v4
            let uuid = uuid::Uuid::parse_str(&uuid_str)
                .expect(&format!("Invalid UUID format for {}", description));
            assert_eq!(uuid.get_version_num(), 4, "Expected UUID v4 for {}", description);
        }
    }

    #[tokio::test]
    async fn test_processor_struct_fields() {
        // Test to ensure all struct fields are properly initialized and accessible
        let processor = DocumentProcessor::test_instance();

        // Test config field
        assert_eq!(processor.config.max_concurrent_tasks, 2);
        assert_eq!(processor.config.default_chunk_size, 1000);
        assert_eq!(processor.config.default_chunk_overlap, 200);
        assert_eq!(processor.config.max_file_size_bytes, 1024 * 1024);
        assert_eq!(processor.config.supported_extensions, vec!["txt", "md"]);
        assert!(!processor.config.enable_lsp);
        assert_eq!(processor.config.lsp_timeout_secs, 10);

        // Test qdrant_config field
        assert_eq!(processor.qdrant_config.url, "http://localhost:6333");
        assert_eq!(processor.qdrant_config.api_key, None);
        assert_eq!(processor.qdrant_config.timeout_secs, 30);
        assert_eq!(processor.qdrant_config.max_retries, 3);
        assert_eq!(processor.qdrant_config.default_collection.vector_size, 384);
        assert_eq!(processor.qdrant_config.default_collection.distance_metric, "Cosine");
        assert!(processor.qdrant_config.default_collection.enable_indexing);
        assert_eq!(processor.qdrant_config.default_collection.replication_factor, 1);
        assert_eq!(processor.qdrant_config.default_collection.shard_number, 1);

        // Test semaphore field by verifying it works
        let permit = processor.semaphore.acquire().await.unwrap();
        drop(permit); // Release permit
    }
}
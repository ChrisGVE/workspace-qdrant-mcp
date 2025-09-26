//! Shared test utilities for gRPC protocol integration testing
//!
//! This module provides common utilities and helpers for testing gRPC protocol
//! correctness, serialization, and service integration.

use crate::{
    config::*,
    daemon::WorkspaceDaemon,
    grpc::services::*,
    proto::*,
};

use std::{net::SocketAddr, sync::Arc, time::Duration, collections::HashMap};
use tokio::time::timeout;
use tonic::{
    transport::{Server, Channel},
    Request, Response, Status, Code,
    metadata::{MetadataValue, MetadataMap},
};
use prost_types::Timestamp;

/// Test configuration builder for integration tests
pub struct TestConfigBuilder {
    config: DaemonConfig,
}

impl TestConfigBuilder {
    /// Create a new test configuration builder with defaults
    pub fn new() -> Self {
        Self {
            config: DaemonConfig {
                server: ServerConfig {
                    host: "127.0.0.1".to_string(),
                    port: 0, // Use ephemeral port for testing
                    max_connections: 100,
                    connection_timeout_secs: 5,
                    request_timeout_secs: 10,
                    enable_tls: false,
                    security: crate::config::SecurityConfig::default(),
                    transport: crate::config::TransportConfig::default(),
                    message: crate::config::MessageConfig::default(),
                    compression: crate::config::CompressionConfig::default(),
                    streaming: crate::config::StreamingConfig::default(),
                },
                database: DatabaseConfig {
                    sqlite_path: ":memory:".to_string(),
                    max_connections: 5,
                    connection_timeout_secs: 5,
                    enable_wal: true,
                },
                qdrant: QdrantConfig {
                    url: "http://localhost:6333".to_string(),
                    api_key: None,
                    timeout_secs: 5,
                    max_retries: 2,
                    default_collection: crate::config::CollectionConfig {
                        vector_size: 384,
                        distance_metric: "Cosine".to_string(),
                        enable_indexing: true,
                        replication_factor: 1,
                        shard_number: 1,
                    },
                },
                processing: ProcessingConfig {
                    max_concurrent_tasks: 2,
                    default_chunk_size: 1000,
                    default_chunk_overlap: 200,
                    max_file_size_bytes: 1024 * 1024,
                    supported_extensions: vec!["txt".to_string(), "md".to_string()],
                    enable_lsp: false,
                    lsp_timeout_secs: 5,
                },
                file_watcher: FileWatcherConfig {
                    enabled: false,
                    debounce_ms: 100,
                    max_watched_dirs: 5,
                    ignore_patterns: vec![],
                    recursive: true,
                },
                metrics: MetricsConfig {
                    enabled: false,
                    collection_interval_secs: 30,
                    retention_days: 1,
                    enable_prometheus: false,
                    prometheus_port: 9090,
                },
                logging: LoggingConfig {
                    level: "info".to_string(),
                    file_path: None,
                    json_format: false,
                    max_file_size_mb: 10,
                    max_files: 2,
                },
            },
        }
    }

    /// Set the server port (use 0 for ephemeral)
    pub fn with_port(mut self, port: u16) -> Self {
        self.config.server.port = port;
        self
    }

    /// Set the database path
    pub fn with_database_path(mut self, path: String) -> Self {
        self.config.database.sqlite_path = path;
        self
    }

    /// Set the Qdrant URL
    pub fn with_qdrant_url(mut self, url: String) -> Self {
        self.config.qdrant.url = url;
        self
    }

    /// Set connection limits
    pub fn with_connection_limits(mut self, max_connections: usize, timeout_secs: u64) -> Self {
        self.config.server.max_connections = max_connections;
        self.config.server.connection_timeout_secs = timeout_secs;
        self
    }

    /// Enable TLS
    pub fn with_tls(mut self) -> Self {
        self.config.server.enable_tls = true;
        self
    }

    /// Build the configuration
    pub fn build(self) -> DaemonConfig {
        self.config
    }
}

impl Default for TestConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Test server wrapper for gRPC integration testing
pub struct TestGrpcServer {
    pub address: SocketAddr,
    pub daemon: Arc<WorkspaceDaemon>,
    server_handle: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
}

impl TestGrpcServer {
    /// Start a new test server with the given configuration
    pub async fn start_with_config(config: DaemonConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let daemon = Arc::new(WorkspaceDaemon::new(config).await?);
        let address = SocketAddr::from(([127, 0, 0, 1], 0)); // Ephemeral port

        // Create service implementations
        let document_processor = DocumentProcessorImpl::new(Arc::clone(&daemon));
        let search_service = SearchServiceImpl::new(Arc::clone(&daemon));
        let memory_service = MemoryServiceImpl::new(Arc::clone(&daemon));
        let system_service = SystemServiceImpl::new(Arc::clone(&daemon));

        // Build server with reflection for protocol validation
        let reflection_service = tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(
                include_bytes!(concat!(env!("OUT_DIR"), "/workspace_daemon_descriptor.bin"))
            )
            .build_v1()
            .expect("Failed to build reflection service");

        let server = Server::builder()
            .timeout(Duration::from_secs(10))
            .concurrency_limit_per_connection(256)
            .add_service(document_processor_server::DocumentProcessorServer::new(document_processor))
            .add_service(search_service_server::SearchServiceServer::new(search_service))
            .add_service(memory_service_server::MemoryServiceServer::new(memory_service))
            .add_service(system_service_server::SystemServiceServer::new(system_service))
            .add_service(reflection_service);

        // Start server and get actual bound address
        let listener = tokio::net::TcpListener::bind(address).await?;
        let actual_addr = listener.local_addr()?;

        let server_handle = tokio::spawn(async move {
            server.serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
        });

        // Wait a bit for server to be ready
        tokio::time::sleep(Duration::from_millis(50)).await;

        Ok(Self {
            address: actual_addr,
            daemon,
            server_handle,
        })
    }

    /// Start a test server with default configuration
    pub async fn start() -> Result<Self, Box<dyn std::error::Error>> {
        let config = TestConfigBuilder::new().build();
        Self::start_with_config(config).await
    }

    /// Get a client channel connected to this server
    pub async fn get_client_channel(&self) -> Result<Channel, Box<dyn std::error::Error>> {
        let uri = format!("http://{}", self.address);
        Ok(Channel::from_shared(uri)?
            .timeout(Duration::from_secs(10))
            .connect()
            .await?)
    }

    /// Get all available service clients
    pub async fn get_clients(&self) -> Result<GrpcClients, Box<dyn std::error::Error>> {
        let channel = self.get_client_channel().await?;
        Ok(GrpcClients::new(channel))
    }

    /// Shutdown the test server
    pub async fn shutdown(self) {
        self.server_handle.abort();
        let _ = self.server_handle.await;
    }
}

/// Collection of all gRPC service clients for testing
pub struct GrpcClients {
    pub document_processor: document_processor_client::DocumentProcessorClient<Channel>,
    pub search_service: search_service_client::SearchServiceClient<Channel>,
    pub memory_service: memory_service_client::MemoryServiceClient<Channel>,
    pub system_service: system_service_client::SystemServiceClient<Channel>,
}

impl GrpcClients {
    /// Create a new set of clients with the given channel
    pub fn new(channel: Channel) -> Self {
        Self {
            document_processor: document_processor_client::DocumentProcessorClient::new(channel.clone()),
            search_service: search_service_client::SearchServiceClient::new(channel.clone()),
            memory_service: memory_service_client::MemoryServiceClient::new(channel.clone()),
            system_service: system_service_client::SystemServiceClient::new(channel),
        }
    }
}

/// Protocol buffer test data factory
pub struct TestDataFactory;

impl TestDataFactory {
    /// Create a test timestamp
    pub fn create_timestamp() -> Timestamp {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        Timestamp {
            seconds: now.as_secs() as i64,
            nanos: (now.subsec_nanos() % 1_000_000_000) as i32,
        }
    }

    /// Create a test ProcessDocumentRequest
    pub fn create_process_document_request(
        file_path: &str,
        project_id: &str,
        collection_name: &str,
        doc_type: DocumentType,
    ) -> ProcessDocumentRequest {
        ProcessDocumentRequest {
            file_path: file_path.to_string(),
            project_id: project_id.to_string(),
            collection_name: collection_name.to_string(),
            document_type: doc_type as i32,
            metadata: HashMap::from([
                ("test".to_string(), "true".to_string()),
                ("created_by".to_string(), "test_factory".to_string()),
            ]),
            options: Some(ProcessingOptions {
                enable_lsp_analysis: true,
                chunk_size: 1024,
                chunk_overlap: 128,
                extract_metadata: true,
                detect_language: true,
                custom_parsers: vec![],
            }),
        }
    }

    /// Create a test HybridSearchRequest
    pub fn create_hybrid_search_request(
        query: &str,
        project_id: &str,
        collection_names: Vec<String>,
    ) -> HybridSearchRequest {
        HybridSearchRequest {
            query: query.to_string(),
            context: SearchContext::Project as i32,
            options: Some(SearchOptions {
                limit: 10,
                score_threshold: 0.5,
                include_metadata: true,
                include_content: true,
                ranking: Some(RankingOptions {
                    semantic_weight: 0.7,
                    keyword_weight: 0.3,
                    rrf_constant: 60.0,
                }),
            }),
            project_id: project_id.to_string(),
            collection_names,
        }
    }

    /// Create a test DocumentContent with chunks
    pub fn create_document_content(text: &str, chunk_count: usize) -> DocumentContent {
        let chunk_size = text.len() / chunk_count.max(1);
        let chunks: Vec<DocumentChunk> = (0..chunk_count)
            .map(|i| {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(text.len());
                DocumentChunk {
                    id: format!("chunk_{}", i),
                    content: text[start..end].to_string(),
                    start_offset: start as i32,
                    end_offset: end as i32,
                    metadata: HashMap::from([
                        ("chunk_index".to_string(), i.to_string()),
                        ("chunk_type".to_string(), "test".to_string()),
                    ]),
                }
            })
            .collect();

        DocumentContent {
            text: text.to_string(),
            chunks,
            extracted_metadata: HashMap::from([
                ("total_chunks".to_string(), chunk_count.to_string()),
                ("content_length".to_string(), text.len().to_string()),
                ("test_data".to_string(), "true".to_string()),
            ]),
        }
    }

    /// Create a test AddDocumentRequest
    pub fn create_add_document_request(
        file_path: &str,
        collection_name: &str,
        project_id: &str,
        content: Option<DocumentContent>,
    ) -> AddDocumentRequest {
        AddDocumentRequest {
            file_path: file_path.to_string(),
            collection_name: collection_name.to_string(),
            project_id: project_id.to_string(),
            content,
            metadata: HashMap::from([
                ("source".to_string(), "test_factory".to_string()),
                ("test_mode".to_string(), "true".to_string()),
            ]),
        }
    }
}

/// Test assertions and validation helpers
pub struct TestValidators;

impl TestValidators {
    /// Validate a timestamp is recent (within the last minute)
    pub fn validate_recent_timestamp(timestamp: &Option<Timestamp>) -> bool {
        if let Some(ts) = timestamp {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            let diff = (now.as_secs() as i64) - ts.seconds;
            diff >= 0 && diff < 60 // Within last minute
        } else {
            false
        }
    }

    /// Validate a ProcessDocumentResponse
    pub fn validate_process_document_response(response: &ProcessDocumentResponse) -> bool {
        !response.document_id.is_empty()
            && response.status == ProcessingStatus::Completed as i32
            && response.error_message.is_empty()
            && response.chunks_created >= 0
            && Self::validate_recent_timestamp(&response.processed_at)
    }

    /// Validate a HybridSearchResponse
    pub fn validate_hybrid_search_response(response: &HybridSearchResponse) -> bool {
        !response.query_id.is_empty()
            && response.metadata.is_some()
            && response.results.len() >= 0
    }

    /// Validate a HealthCheckResponse
    pub fn validate_health_check_response(response: &HealthCheckResponse) -> bool {
        response.status == ServiceStatus::Healthy as i32
            && !response.components.is_empty()
            && Self::validate_recent_timestamp(&response.timestamp)
    }

    /// Validate gRPC metadata contains expected keys
    pub fn validate_metadata_contains(metadata: &MetadataMap, expected_keys: &[&str]) -> bool {
        expected_keys.iter().all(|key| metadata.contains_key(*key))
    }
}

/// Concurrent testing utilities
pub struct ConcurrentTestRunner;

impl ConcurrentTestRunner {
    /// Run multiple concurrent gRPC requests and collect results
    pub async fn run_concurrent_requests<T, F, Fut>(
        request_count: usize,
        request_factory: F,
    ) -> Vec<Result<T, tonic::Status>>
    where
        F: Fn(usize) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<T, tonic::Status>> + Send + 'static,
        T: Send + 'static,
    {
        let mut handles = Vec::with_capacity(request_count);

        for i in 0..request_count {
            let request_fut = request_factory(i);
            let handle = tokio::spawn(async move { request_fut.await });
            handles.push(handle);
        }

        let mut results = Vec::with_capacity(request_count);
        for handle in handles {
            match handle.await {
                Ok(result) => results.push(result),
                Err(_) => results.push(Err(tonic::Status::internal("Task join error"))),
            }
        }

        results
    }

    /// Analyze concurrent test results
    pub fn analyze_results<T>(results: &[Result<T, tonic::Status>]) -> ConcurrentTestStats {
        let total = results.len();
        let successes = results.iter().filter(|r| r.is_ok()).count();
        let failures = total - successes;

        let error_codes: HashMap<tonic::Code, usize> = results
            .iter()
            .filter_map(|r| r.as_ref().err())
            .fold(HashMap::new(), |mut acc, status| {
                *acc.entry(status.code()).or_insert(0) += 1;
                acc
            });

        ConcurrentTestStats {
            total_requests: total,
            successful_requests: successes,
            failed_requests: failures,
            success_rate: successes as f64 / total as f64,
            error_codes,
        }
    }
}

/// Statistics from concurrent testing
#[derive(Debug)]
pub struct ConcurrentTestStats {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub success_rate: f64,
    pub error_codes: HashMap<tonic::Code, usize>,
}

impl ConcurrentTestStats {
    /// Check if the success rate meets a threshold
    pub fn meets_success_threshold(&self, threshold: f64) -> bool {
        self.success_rate >= threshold
    }

    /// Get the most common error code
    pub fn most_common_error(&self) -> Option<(tonic::Code, usize)> {
        self.error_codes
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(code, count)| (*code, *count))
    }
}

/// Protocol compliance test utilities
pub struct ProtocolTestSuite;

impl ProtocolTestSuite {
    /// Test all enum values are properly serialized
    pub async fn test_enum_serialization<F, Fut>(
        enum_values: &[i32],
        request_factory: F,
    ) -> Vec<Result<bool, Box<dyn std::error::Error>>>
    where
        F: Fn(i32) -> Fut,
        Fut: std::future::Future<Output = Result<bool, Box<dyn std::error::Error>>>,
    {
        let mut results = Vec::new();
        for &enum_value in enum_values {
            let result = request_factory(enum_value).await;
            results.push(result);
        }
        results
    }

    /// Test metadata propagation through the protocol stack
    pub async fn test_metadata_propagation(
        clients: &mut GrpcClients,
        metadata_pairs: Vec<(String, String)>,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let mut request = Request::new(());
        for (key, value) in metadata_pairs {
            request.metadata_mut().insert(
                tonic::metadata::MetadataKey::from_bytes(key.as_bytes()).unwrap(),
                MetadataValue::try_from(value).unwrap()
            );
        }

        let response = timeout(
            Duration::from_secs(5),
            clients.system_service.health_check(request)
        ).await??;

        // For now, just validate the request succeeded
        // Real implementation would check response metadata
        Ok(response.into_inner().status == ServiceStatus::Healthy as i32)
    }

    /// Test gRPC error code mapping
    pub async fn test_error_code_mapping(
        clients: &mut GrpcClients,
    ) -> Result<Vec<(String, tonic::Code)>, Box<dyn std::error::Error>> {
        let mut error_tests = Vec::new();

        // Test with invalid document request
        let invalid_request = GetDocumentRequest {
            document_id: "".to_string(),
            collection_name: "".to_string(),
        };

        match clients.memory_service.get_document(Request::new(invalid_request)).await {
            Ok(_) => error_tests.push(("empty_document_id".to_string(), tonic::Code::Ok)),
            Err(status) => error_tests.push(("empty_document_id".to_string(), status.code())),
        }

        // Test with non-existent document
        let not_found_request = GetDocumentRequest {
            document_id: "non_existent_id".to_string(),
            collection_name: "non_existent_collection".to_string(),
        };

        match clients.memory_service.get_document(Request::new(not_found_request)).await {
            Ok(_) => error_tests.push(("non_existent_document".to_string(), tonic::Code::Ok)),
            Err(status) => error_tests.push(("non_existent_document".to_string(), status.code())),
        }

        Ok(error_tests)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = TestConfigBuilder::new()
            .with_port(8080)
            .with_database_path("/tmp/test.db".to_string())
            .with_qdrant_url("http://localhost:6334".to_string())
            .with_connection_limits(50, 30)
            .with_tls()
            .build();

        assert_eq!(config.server.port, 8080);
        assert_eq!(config.database.sqlite_path, "/tmp/test.db");
        assert_eq!(config.qdrant.url, "http://localhost:6334");
        assert_eq!(config.server.max_connections, 50);
        assert_eq!(config.server.connection_timeout_secs, 30);
        assert!(config.server.enable_tls);
    }

    #[test]
    fn test_data_factory() {
        let timestamp = TestDataFactory::create_timestamp();
        assert!(timestamp.seconds > 0);

        let request = TestDataFactory::create_process_document_request(
            "/test/file.txt",
            "test_project",
            "test_collection",
            DocumentType::Text,
        );
        assert_eq!(request.file_path, "/test/file.txt");
        assert_eq!(request.document_type, DocumentType::Text as i32);
        assert!(request.options.is_some());

        let content = TestDataFactory::create_document_content("Hello world test content", 3);
        assert_eq!(content.chunks.len(), 3);
        assert!(!content.text.is_empty());
    }

    #[test]
    fn test_validators() {
        let recent_timestamp = Some(TestDataFactory::create_timestamp());
        assert!(TestValidators::validate_recent_timestamp(&recent_timestamp));

        let old_timestamp = Some(Timestamp {
            seconds: 1000, // Very old timestamp
            nanos: 0,
        });
        assert!(!TestValidators::validate_recent_timestamp(&old_timestamp));

        assert!(!TestValidators::validate_recent_timestamp(&None));
    }

    #[test]
    fn test_concurrent_stats() {
        let stats = ConcurrentTestStats {
            total_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            success_rate: 0.95,
            error_codes: HashMap::from([
                (tonic::Code::DeadlineExceeded, 3),
                (tonic::Code::Unavailable, 2),
            ]),
        };

        assert!(stats.meets_success_threshold(0.9));
        assert!(!stats.meets_success_threshold(0.98));

        let (code, count) = stats.most_common_error().unwrap();
        assert_eq!(code, tonic::Code::DeadlineExceeded);
        assert_eq!(count, 3);
    }
}
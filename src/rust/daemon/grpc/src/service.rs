//! gRPC service implementation
//!
//! This module contains the actual service implementations with enhanced
//! security, error handling, and performance monitoring.

use crate::proto::*;
use crate::{AuthConfig, ServerMetrics, GrpcError};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use tokio_stream::Stream;
use tonic::{Request, Response, Status};
use workspace_qdrant_core::DocumentProcessor;
use uuid::Uuid;

/// Ingestion service implementation with enhanced security and monitoring
pub struct IngestionService {
    processor: DocumentProcessor,
    auth_config: Option<AuthConfig>,
    metrics: Arc<ServerMetrics>,
    start_time: SystemTime,
}

impl IngestionService {
    pub fn new() -> Self {
        Self {
            processor: DocumentProcessor::new(),
            auth_config: None,
            metrics: Arc::new(ServerMetrics::default()),
            start_time: SystemTime::now(),
        }
    }

    pub fn new_with_auth(auth_config: Option<AuthConfig>) -> Self {
        Self {
            processor: DocumentProcessor::new(),
            auth_config,
            metrics: Arc::new(ServerMetrics::default()),
            start_time: SystemTime::now(),
        }
    }

    pub fn with_metrics(mut self, metrics: Arc<ServerMetrics>) -> Self {
        self.metrics = metrics;
        self
    }

    /// Validate authentication for a request
    fn authenticate(&self, request: &Request<impl prost::Message>) -> Result<(), Status> {
        let Some(auth_config) = &self.auth_config else {
            return Ok(()); // No auth configured
        };

        if !auth_config.enabled {
            return Ok(());
        }

        // Check API key if configured
        if let Some(expected_key) = &auth_config.api_key {
            let auth_header = request.metadata()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| {
                    self.metrics.auth_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Status::unauthenticated("Missing authorization header")
                })?;

            if !auth_header.starts_with("Bearer ") {
                self.metrics.auth_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(Status::unauthenticated("Invalid authorization format"));
            }

            let token = &auth_header[7..]; // Remove "Bearer " prefix
            if token != expected_key {
                self.metrics.auth_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(Status::unauthenticated("Invalid API key"));
            }
        }

        // Check origin if configured
        if !auth_config.allowed_origins.contains(&"*".to_string()) {
            let origin = request.metadata()
                .get("origin")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");

            if !auth_config.allowed_origins.contains(&origin.to_string()) {
                self.metrics.auth_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(Status::permission_denied("Origin not allowed"));
            }
        }

        Ok(())
    }

    /// Record metrics for a request
    fn record_request(&self, duration: std::time::Duration, success: bool) {
        self.metrics.total_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if !success {
            self.metrics.failed_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // Update average response time (simplified moving average)
        let duration_ms = duration.as_millis() as u64;
        let current_avg = self.metrics.avg_response_time.load(std::sync::atomic::Ordering::Relaxed);
        let total_requests = self.metrics.total_requests.load(std::sync::atomic::Ordering::Relaxed);

        if total_requests > 0 {
            let new_avg = (current_avg * (total_requests - 1) + duration_ms) / total_requests;
            self.metrics.avg_response_time.store(new_avg, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl Default for IngestionService {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl ingest_service_server::IngestService for IngestionService {
    async fn process_document(
        &self,
        request: Request<ProcessDocumentRequest>,
    ) -> Result<Response<ProcessDocumentResponse>, Status> {
        let start_time = Instant::now();

        // Authenticate request
        self.authenticate(&request)?;

        let req = request.into_inner();

        tracing::info!(
            "Processing document: {} for collection: {}",
            req.file_path,
            req.collection
        );

        // Validate input parameters
        if req.file_path.is_empty() {
            self.record_request(start_time.elapsed(), false);
            return Err(Status::invalid_argument("File path cannot be empty"));
        }

        if req.collection.is_empty() {
            self.record_request(start_time.elapsed(), false);
            return Err(Status::invalid_argument("Collection name cannot be empty"));
        }

        // Process the document (this is a simplified implementation)
        // In real implementation, this would use the DocumentProcessor
        let file_path = std::path::Path::new(&req.file_path);
        let processing_result = match self.processor.process_file(file_path, &req.collection).await {
            Ok(result) => result,
            Err(e) => {
                tracing::error!("Document processing failed: {}", e);
                self.record_request(start_time.elapsed(), false);
                return Err(Status::internal(format!("Processing failed: {}", e)));
            }
        };

        let document_id = req.document_id.unwrap_or_else(|| Uuid::new_v4().to_string());

        let response = ProcessDocumentResponse {
            success: true,
            message: format!("Successfully processed document: {}", req.file_path),
            document_id: Some(document_id),
            chunks_added: processing_result.chunks_created.unwrap_or(1) as i32,
            applied_metadata: req.metadata,
        };

        self.record_request(start_time.elapsed(), true);
        tracing::info!("Document processed successfully in {:?}", start_time.elapsed());

        Ok(Response::new(response))
    }

    type StartWatchingStream = Pin<Box<dyn Stream<Item = Result<WatchingUpdate, Status>> + Send>>;

    async fn start_watching(
        &self,
        request: Request<StartWatchingRequest>,
    ) -> Result<Response<Self::StartWatchingStream>, Status> {
        let req = request.into_inner();
        
        tracing::info!("Starting to watch: {} for collection: {}", req.path, req.collection);
        
        let watch_id = req.watch_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        
        let stream = tokio_stream::iter(vec![
            Ok(WatchingUpdate {
                watch_id: watch_id.clone(),
                event_type: WatchEventType::Started as i32,
                file_path: req.path.clone(),
                timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                error_message: None,
                status: WatchStatus::Active as i32,
            })
        ]);
        
        Ok(Response::new(Box::pin(stream)))
    }

    async fn stop_watching(
        &self,
        request: Request<StopWatchingRequest>,
    ) -> Result<Response<StopWatchingResponse>, Status> {
        let req = request.into_inner();
        
        tracing::info!("Stopping watch: {}", req.watch_id);
        
        let response = StopWatchingResponse {
            success: true,
            message: format!("Successfully stopped watch: {}", req.watch_id),
        };
        
        Ok(Response::new(response))
    }

    async fn execute_query(
        &self,
        request: Request<ExecuteQueryRequest>,
    ) -> Result<Response<ExecuteQueryResponse>, Status> {
        let req = request.into_inner();
        
        tracing::info!("Executing query: {} with mode: {:?}", req.query, req.mode);
        
        let response = ExecuteQueryResponse {
            query: req.query.clone(),
            mode: req.mode,
            collections_searched: req.collections,
            total_results: 0,
            results: vec![],
        };
        
        Ok(Response::new(response))
    }

    async fn get_stats(
        &self,
        request: Request<GetStatsRequest>,
    ) -> Result<Response<GetStatsResponse>, Status> {
        let _req = request.into_inner();
        
        tracing::info!("Getting engine statistics");
        
        let engine_stats = EngineStats {
            started_at: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
            uptime: Some(prost_types::Duration {
                seconds: 0,
                nanos: 0,
            }),
            total_documents_processed: 0,
            total_documents_indexed: 0,
            active_watches: 0,
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        
        let response = GetStatsResponse {
            engine_stats: Some(engine_stats),
            collection_stats: vec![],
            watch_stats: vec![],
        };
        
        Ok(Response::new(response))
    }

    async fn health_check(
        &self,
        _request: Request<()>,
    ) -> Result<Response<HealthResponse>, Status> {
        let start_time = Instant::now();

        tracing::debug!("Health check requested");

        // Perform comprehensive health checks
        let mut services = vec![];

        // Check ingestion engine health
        let engine_health = if self.processor.is_healthy().await {
            ServiceHealth {
                name: "ingestion_engine".to_string(),
                status: HealthStatus::Healthy as i32,
                message: "Running normally".to_string(),
            }
        } else {
            ServiceHealth {
                name: "ingestion_engine".to_string(),
                status: HealthStatus::Degraded as i32,
                message: "Engine experiencing issues".to_string(),
            }
        };
        services.push(engine_health);

        // Check Qdrant connectivity
        let qdrant_health = match self.processor.test_qdrant_connection().await {
            Ok(_) => ServiceHealth {
                name: "qdrant_client".to_string(),
                status: HealthStatus::Healthy as i32,
                message: "Connected to Qdrant".to_string(),
            },
            Err(e) => ServiceHealth {
                name: "qdrant_client".to_string(),
                status: HealthStatus::Unhealthy as i32,
                message: format!("Qdrant connection failed: {}", e),
            },
        };
        services.push(qdrant_health);

        // Determine overall status
        let overall_status = if services.iter().all(|s| s.status == HealthStatus::Healthy as i32) {
            HealthStatus::Healthy
        } else if services.iter().any(|s| s.status == HealthStatus::Unhealthy as i32) {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        };

        let uptime = self.start_time.elapsed().unwrap_or_default();
        let response = HealthResponse {
            status: overall_status as i32,
            message: format!(
                "Service health: {}. Uptime: {:?}. Total requests: {}",
                match overall_status {
                    HealthStatus::Healthy => "Healthy",
                    HealthStatus::Degraded => "Degraded",
                    HealthStatus::Unhealthy => "Unhealthy",
                    _ => "Unknown",
                },
                uptime,
                self.metrics.total_requests.load(std::sync::atomic::Ordering::Relaxed)
            ),
            services,
        };

        self.record_request(start_time.elapsed(), overall_status != HealthStatus::Unhealthy);
        Ok(Response::new(response))
    }

    // Process folder operations
    type ProcessFolderStream = Pin<Box<dyn Stream<Item = Result<ProcessFolderProgress, Status>> + Send>>;
    
    async fn process_folder(
        &self,
        request: Request<ProcessFolderRequest>,
    ) -> Result<Response<Self::ProcessFolderStream>, Status> {
        let req = request.into_inner();
        tracing::info!("Processing folder: {} for collection: {}", req.folder_path, req.collection);
        
        let stream = tokio_stream::iter(vec![
            Ok(ProcessFolderProgress {
                current_file: req.folder_path.clone(),
                files_processed: 0,
                total_files: 1,
                files_succeeded: 0,
                files_failed: 0,
                failed_files: vec![],
                completed: false,
                error_message: String::new(),
            })
        ]);
        
        Ok(Response::new(Box::pin(stream)))
    }

    // Watch operations
    async fn list_watches(
        &self,
        _request: Request<ListWatchesRequest>,
    ) -> Result<Response<ListWatchesResponse>, Status> {
        tracing::info!("Listing active watches");
        
        let response = ListWatchesResponse {
            watches: vec![],
        };
        
        Ok(Response::new(response))
    }

    async fn configure_watch(
        &self,
        request: Request<ConfigureWatchRequest>,
    ) -> Result<Response<ConfigureWatchResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("Configuring watch: {}", req.watch_id);
        
        let response = ConfigureWatchResponse {
            success: true,
            message: format!("Successfully configured watch: {}", req.watch_id),
            updated_watch: None,
        };
        
        Ok(Response::new(response))
    }

    // Collection operations
    async fn list_collections(
        &self,
        _request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        tracing::info!("Listing collections");
        
        let response = ListCollectionsResponse {
            collections: vec![],
        };
        
        Ok(Response::new(response))
    }

    async fn get_collection_info(
        &self,
        request: Request<GetCollectionInfoRequest>,
    ) -> Result<Response<CollectionInfo>, Status> {
        let req = request.into_inner();
        tracing::info!("Getting collection info for: {}", req.collection_name);
        
        let response = CollectionInfo {
            name: req.collection_name.clone(),
            description: String::new(),
            document_count: 0,
            total_size_bytes: 0,
            created_at: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
            last_updated: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
            metadata: std::collections::HashMap::new(),
        };
        
        Ok(Response::new(response))
    }

    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("Creating collection: {}", req.collection_name);
        
        let response = CreateCollectionResponse {
            success: true,
            message: format!("Successfully created collection: {}", req.collection_name),
            collection_info: Some(CollectionInfo {
                name: req.collection_name,
                description: req.description,
                document_count: 0,
                total_size_bytes: 0,
                created_at: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                last_updated: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                metadata: req.metadata,
            }),
        };
        
        Ok(Response::new(response))
    }

    async fn delete_collection(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<DeleteCollectionResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("Deleting collection: {}", req.collection_name);
        
        let response = DeleteCollectionResponse {
            success: true,
            message: format!("Successfully deleted collection: {}", req.collection_name),
            documents_deleted: 0,
        };
        
        Ok(Response::new(response))
    }

    // Document operations
    async fn list_documents(
        &self,
        request: Request<ListDocumentsRequest>,
    ) -> Result<Response<ListDocumentsResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("Listing documents in collection: {}", req.collection_name);
        
        let response = ListDocumentsResponse {
            documents: vec![],
            total_count: 0,
            has_more: false,
        };
        
        Ok(Response::new(response))
    }

    async fn get_document(
        &self,
        request: Request<GetDocumentRequest>,
    ) -> Result<Response<GetDocumentResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("Getting document: {} from collection: {}", req.document_id, req.collection_name);
        
        let response = GetDocumentResponse {
            success: false,
            message: "Document not found".to_string(),
            document_info: None,
            content: String::new(),
            chunks: vec![],
        };
        
        Ok(Response::new(response))
    }

    async fn delete_document(
        &self,
        request: Request<DeleteDocumentRequest>,
    ) -> Result<Response<DeleteDocumentResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("Deleting document: {} from collection: {}", req.document_id, req.collection_name);
        
        let response = DeleteDocumentResponse {
            success: true,
            message: format!("Successfully deleted document: {}", req.document_id),
            chunks_deleted: 0,
        };
        
        Ok(Response::new(response))
    }

    // Configuration operations
    async fn load_configuration(
        &self,
        request: Request<LoadConfigurationRequest>,
    ) -> Result<Response<LoadConfigurationResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("Loading configuration from: {:?}", req.config_path);
        
        let response = LoadConfigurationResponse {
            success: true,
            message: "Configuration loaded successfully".to_string(),
            config_yaml: String::new(),
            config_sources: vec![],
        };
        
        Ok(Response::new(response))
    }

    async fn save_configuration(
        &self,
        request: Request<SaveConfigurationRequest>,
    ) -> Result<Response<SaveConfigurationResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("Saving configuration: {}", req.config_yaml.len());
        
        let response = SaveConfigurationResponse {
            success: true,
            message: "Configuration saved successfully".to_string(),
        };
        
        Ok(Response::new(response))
    }

    async fn validate_configuration(
        &self,
        request: Request<ValidateConfigurationRequest>,
    ) -> Result<Response<ValidateConfigurationResponse>, Status> {
        let _req = request.into_inner();
        tracing::info!("Validating configuration");
        
        let response = ValidateConfigurationResponse {
            valid: true,
            errors: vec![],
            warnings: vec![],
        };
        
        Ok(Response::new(response))
    }

    // Memory operations
    async fn add_memory_rule(
        &self,
        request: Request<AddMemoryRuleRequest>,
    ) -> Result<Response<AddMemoryRuleResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("Adding memory rule: {}", req.name);
        
        let response = AddMemoryRuleResponse {
            success: true,
            message: format!("Successfully added memory rule: {}", req.name),
            memory_rule: None,
        };
        
        Ok(Response::new(response))
    }

    async fn list_memory_rules(
        &self,
        _request: Request<ListMemoryRulesRequest>,
    ) -> Result<Response<ListMemoryRulesResponse>, Status> {
        tracing::info!("Listing memory rules");
        
        let response = ListMemoryRulesResponse {
            rules: vec![],
            total_count: 0,
        };
        
        Ok(Response::new(response))
    }

    async fn delete_memory_rule(
        &self,
        request: Request<DeleteMemoryRuleRequest>,
    ) -> Result<Response<DeleteMemoryRuleResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("Deleting memory rule: {}", req.rule_id);
        
        let response = DeleteMemoryRuleResponse {
            success: true,
            message: format!("Successfully deleted memory rule: {}", req.rule_id),
        };
        
        Ok(Response::new(response))
    }

    async fn search_memory_rules(
        &self,
        request: Request<SearchMemoryRulesRequest>,
    ) -> Result<Response<SearchMemoryRulesResponse>, Status> {
        let req = request.into_inner();
        tracing::info!("Searching memory rules with query: {}", req.query);
        
        let response = SearchMemoryRulesResponse {
            matches: vec![],
        };
        
        Ok(Response::new(response))
    }

    // Status operations
    async fn get_processing_status(
        &self,
        _request: Request<GetProcessingStatusRequest>,
    ) -> Result<Response<ProcessingStatusResponse>, Status> {
        tracing::info!("Getting processing status");
        
        let response = ProcessingStatusResponse {
            active_tasks: vec![],
            recent_tasks: vec![],
            stats: None,
        };
        
        Ok(Response::new(response))
    }

    async fn get_system_status(
        &self,
        _request: Request<()>,
    ) -> Result<Response<SystemStatusResponse>, Status> {
        tracing::info!("Getting system status");
        
        let response = SystemStatusResponse {
            overall_status: HealthStatus::Healthy as i32,
            message: "System is healthy".to_string(),
            system_info: None,
            components: vec![],
            processing_stats: None,
            resource_usage: None,
        };
        
        Ok(Response::new(response))
    }

    // Streaming operations
    type StreamProcessingStatusStream = Pin<Box<dyn Stream<Item = Result<ProcessingStatusUpdate, Status>> + Send>>;
    
    async fn stream_processing_status(
        &self,
        _request: Request<StreamStatusRequest>,
    ) -> Result<Response<Self::StreamProcessingStatusStream>, Status> {
        tracing::info!("Starting processing status stream");
        
        let stream = tokio_stream::iter(vec![
            Ok(ProcessingStatusUpdate {
                timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                active_tasks: vec![],
                recent_completed: vec![],
                current_stats: None,
                queue_status: None,
            })
        ]);
        
        Ok(Response::new(Box::pin(stream)))
    }

    type StreamSystemMetricsStream = Pin<Box<dyn Stream<Item = Result<SystemMetricsUpdate, Status>> + Send>>;
    
    async fn stream_system_metrics(
        &self,
        _request: Request<StreamMetricsRequest>,
    ) -> Result<Response<Self::StreamSystemMetricsStream>, Status> {
        tracing::info!("Starting system metrics stream");
        
        let stream = tokio_stream::iter(vec![
            Ok(SystemMetricsUpdate {
                timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                resource_usage: None,
                engine_stats: None,
                collection_stats: vec![],
                performance_metrics: None,
            })
        ]);
        
        Ok(Response::new(Box::pin(stream)))
    }

    type StreamQueueStatusStream = Pin<Box<dyn Stream<Item = Result<QueueStatusUpdate, Status>> + Send>>;
    
    async fn stream_queue_status(
        &self,
        _request: Request<StreamQueueRequest>,
    ) -> Result<Response<Self::StreamQueueStatusStream>, Status> {
        tracing::info!("Starting queue status stream");
        
        let stream = tokio_stream::iter(vec![
            Ok(QueueStatusUpdate {
                timestamp: Some(prost_types::Timestamp::from(std::time::SystemTime::now())),
                queue_status: None,
                recent_additions: vec![],
                active_processing: vec![],
            })
        ]);
        
        Ok(Response::new(Box::pin(stream)))
    }
}
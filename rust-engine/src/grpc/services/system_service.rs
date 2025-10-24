//! System administration gRPC service implementation
//!
//! Implements the complete SystemService with 7 RPCs:
//! 1. HealthCheck - Quick health check for monitoring/alerting
//! 2. GetStatus - Comprehensive system state snapshot
//! 3. GetMetrics - Current performance metrics (snapshot only)
//! 4. SendRefreshSignal - Event-driven refresh signaling
//! 5. NotifyServerStatus - MCP/CLI lifecycle notifications
//! 6. PauseAllWatchers - Pause all file watchers (master switch)
//! 7. ResumeAllWatchers - Resume all file watchers with catch-up

use crate::daemon::WorkspaceDaemon;
use crate::proto::{
    system_service_server::SystemService,
    HealthCheckResponse, SystemStatusResponse, MetricsResponse,
    RefreshSignalRequest, ServerStatusNotification,
    ServiceStatus, ComponentHealth, SystemMetrics,
    QueueType, ServerState,
};
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn};

/// Refresh hint for batched memory refresh
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RefreshHint {
    queue_type: QueueType,
    lsp_languages: Vec<String>,
    grammar_languages: Vec<String>,
}

/// Active server session tracking
#[derive(Debug, Clone)]
pub struct ServerSession {
    project_name: Option<String>,
    project_root: Option<String>,
    connected_at: chrono::DateTime<chrono::Utc>,
}

/// System service implementation
#[derive(Debug)]
pub struct SystemServiceImpl {
    daemon: Arc<WorkspaceDaemon>,
    refresh_hints: Arc<RwLock<HashSet<RefreshHint>>>,
    active_sessions: Arc<RwLock<HashMap<String, ServerSession>>>,
    uptime_since: chrono::DateTime<chrono::Utc>,
}

impl SystemServiceImpl {
    pub fn new(daemon: Arc<WorkspaceDaemon>) -> Self {
        Self {
            daemon,
            refresh_hints: Arc::new(RwLock::new(HashSet::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            uptime_since: chrono::Utc::now(),
        }
    }

    /// Get current refresh hints (for testing/debugging)
    #[allow(dead_code)]
    pub async fn get_refresh_hints(&self) -> HashSet<RefreshHint> {
        self.refresh_hints.read().await.clone()
    }

    /// Get active sessions (for testing/debugging)
    #[allow(dead_code)]
    pub async fn get_active_sessions(&self) -> HashMap<String, ServerSession> {
        self.active_sessions.read().await.clone()
    }

    /// Clear refresh hints (for testing)
    #[allow(dead_code)]
    pub async fn clear_refresh_hints(&self) {
        self.refresh_hints.write().await.clear();
    }
}

#[tonic::async_trait]
impl SystemService for SystemServiceImpl {
    async fn health_check(
        &self,
        _request: Request<()>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        debug!("Health check requested");

        // TODO: Implement actual health checks for each component
        let response = HealthCheckResponse {
            status: ServiceStatus::Healthy as i32,
            components: vec![
                ComponentHealth {
                    component_name: "database".to_string(),
                    status: ServiceStatus::Healthy as i32,
                    message: "SQLite database operational".to_string(),
                    last_check: Some(prost_types::Timestamp {
                        seconds: chrono::Utc::now().timestamp(),
                        nanos: 0,
                    }),
                },
                ComponentHealth {
                    component_name: "qdrant".to_string(),
                    status: ServiceStatus::Healthy as i32,
                    message: "Qdrant connection active".to_string(),
                    last_check: Some(prost_types::Timestamp {
                        seconds: chrono::Utc::now().timestamp(),
                        nanos: 0,
                    }),
                },
                ComponentHealth {
                    component_name: "file_watcher".to_string(),
                    status: ServiceStatus::Healthy as i32,
                    message: "File watching active".to_string(),
                    last_check: Some(prost_types::Timestamp {
                        seconds: chrono::Utc::now().timestamp(),
                        nanos: 0,
                    }),
                },
            ],
            timestamp: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        };

        Ok(Response::new(response))
    }

    async fn get_status(
        &self,
        _request: Request<()>,
    ) -> Result<Response<SystemStatusResponse>, Status> {
        debug!("System status requested");

        // TODO: Implement actual system status collection
        let response = SystemStatusResponse {
            status: ServiceStatus::Healthy as i32,
            metrics: Some(SystemMetrics {
                cpu_usage_percent: 15.5,
                memory_usage_bytes: 128 * 1024 * 1024, // 128MB
                memory_total_bytes: 8 * 1024 * 1024 * 1024, // 8GB
                disk_usage_bytes: 2 * 1024 * 1024 * 1024, // 2GB
                disk_total_bytes: 500 * 1024 * 1024 * 1024, // 500GB
                active_connections: self.active_sessions.read().await.len() as i32,
                pending_operations: self.refresh_hints.read().await.len() as i32,
            }),
            active_projects: vec!["workspace-qdrant-mcp".to_string()],
            total_documents: 1000,
            total_collections: 5,
            uptime_since: Some(prost_types::Timestamp {
                seconds: self.uptime_since.timestamp(),
                nanos: 0,
            }),
        };

        Ok(Response::new(response))
    }

    async fn get_metrics(
        &self,
        _request: Request<()>,
    ) -> Result<Response<MetricsResponse>, Status> {
        debug!("Metrics requested");

        // TODO: Implement actual metrics collection
        let response = MetricsResponse {
            metrics: vec![
                crate::proto::Metric {
                    name: "grpc_requests_total".to_string(),
                    r#type: "counter".to_string(),
                    labels: [("method".to_string(), "ProcessDocument".to_string())].into_iter().collect(),
                    value: 150.0,
                    timestamp: Some(prost_types::Timestamp {
                        seconds: chrono::Utc::now().timestamp(),
                        nanos: 0,
                    }),
                },
                crate::proto::Metric {
                    name: "document_processing_duration_seconds".to_string(),
                    r#type: "histogram".to_string(),
                    labels: [("status".to_string(), "success".to_string())].into_iter().collect(),
                    value: 2.5,
                    timestamp: Some(prost_types::Timestamp {
                        seconds: chrono::Utc::now().timestamp(),
                        nanos: 0,
                    }),
                },
                crate::proto::Metric {
                    name: "refresh_hints_pending".to_string(),
                    r#type: "gauge".to_string(),
                    labels: HashMap::new(),
                    value: self.refresh_hints.read().await.len() as f64,
                    timestamp: Some(prost_types::Timestamp {
                        seconds: chrono::Utc::now().timestamp(),
                        nanos: 0,
                    }),
                },
                crate::proto::Metric {
                    name: "active_sessions".to_string(),
                    r#type: "gauge".to_string(),
                    labels: HashMap::new(),
                    value: self.active_sessions.read().await.len() as f64,
                    timestamp: Some(prost_types::Timestamp {
                        seconds: chrono::Utc::now().timestamp(),
                        nanos: 0,
                    }),
                },
            ],
            collected_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        };

        Ok(Response::new(response))
    }

    async fn send_refresh_signal(
        &self,
        request: Request<RefreshSignalRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        debug!("Refresh signal received: queue_type={:?}", req.queue_type);

        // Create refresh hint
        let hint = RefreshHint {
            queue_type: QueueType::try_from(req.queue_type)
                .map_err(|_| Status::invalid_argument("Invalid queue type"))?,
            lsp_languages: req.lsp_languages,
            grammar_languages: req.grammar_languages,
        };

        // Store hint (deduplicated by HashSet)
        let mut hints = self.refresh_hints.write().await;
        let was_new = hints.insert(hint.clone());

        if was_new {
            info!("Refresh hint added: {:?}", hint.queue_type);
        } else {
            debug!("Refresh hint already exists: {:?}", hint.queue_type);
        }

        // TODO: Trigger batched refresh when thresholds exceeded
        // For now, just store the hint for later processing

        Ok(Response::new(()))
    }

    async fn notify_server_status(
        &self,
        request: Request<ServerStatusNotification>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let state = ServerState::try_from(req.state)
            .map_err(|_| Status::invalid_argument("Invalid server state"))?;

        match state {
            ServerState::Up => {
                info!("Server UP notification received: project={:?}, root={:?}",
                    req.project_name, req.project_root);

                // Create session identifier
                let session_id = req.project_root.clone()
                    .or_else(|| req.project_name.clone())
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

                // Register active session
                let session = ServerSession {
                    project_name: req.project_name.clone(),
                    project_root: req.project_root.clone(),
                    connected_at: chrono::Utc::now(),
                };

                self.active_sessions.write().await.insert(session_id.clone(), session);

                // TODO: Start watchers for this project if project_root provided
                if let Some(_root) = req.project_root {
                    debug!("Would start watchers for session: {}", session_id);
                }
            }
            ServerState::Down => {
                info!("Server DOWN notification received");

                // Find and remove session
                let session_id = req.project_root.clone()
                    .or_else(|| req.project_name.clone());

                if let Some(id) = session_id {
                    if self.active_sessions.write().await.remove(&id).is_some() {
                        info!("Session removed: {}", id);
                    }
                }
            }
            ServerState::Unspecified => {
                warn!("Unspecified server state received");
            }
        }

        Ok(Response::new(()))
    }

    async fn pause_all_watchers(
        &self,
        _request: Request<()>,
    ) -> Result<Response<()>, Status> {
        info!("Pausing all file watchers");

        // TODO: Implement actual pause functionality
        // This should:
        // 1. Set paused flag in all watchers
        // 2. Stop file system event monitoring
        // 3. Continue queue processing
        // 4. Record pause timestamp for catch-up

        Ok(Response::new(()))
    }

    async fn resume_all_watchers(
        &self,
        _request: Request<()>,
    ) -> Result<Response<()>, Status> {
        info!("Resuming all file watchers");

        // TODO: Implement actual resume functionality
        // This should:
        // 1. Clear paused flag in all watchers
        // 2. Resume file system event monitoring
        // 3. Perform catch-up scan for changes during pause
        // 4. Process queued changes

        Ok(Response::new(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use tonic::Request;

    fn create_test_daemon_config() -> DaemonConfig {
        // Use in-memory SQLite database for tests
        let db_path = ":memory:";

        DaemonConfig {
            system: SystemConfig {
                project_name: "test-project".to_string(),
                database: DatabaseConfig {
                    sqlite_path: db_path.to_string(),
                    max_connections: 5,
                    connection_timeout_secs: 30,
                    enable_wal: true,
                },
                auto_ingestion: AutoIngestionConfig {
                    enabled: false,
                    project_collection: "test".to_string(),
                    auto_create_watches: false,
                    project_path: None,
                    include_source_files: true,
                    include_patterns: vec![],
                    exclude_patterns: vec![],
                    max_depth: 10,
                },
                processing: ProcessingConfig {
                    max_concurrent_tasks: 4,
                    supported_extensions: vec![],
                    default_chunk_size: 1000,
                    default_chunk_overlap: 200,
                    max_file_size_bytes: 10 * 1024 * 1024,
                    enable_lsp: false,
                    lsp_timeout_secs: 10,
                },
                file_watcher: FileWatcherConfig {
                    enabled: false,
                    ignore_patterns: vec![],
                    recursive: true,
                    max_watched_dirs: 100,
                    debounce_ms: 100,
                },
            },
            grpc: GrpcConfig {
                server: GrpcServerConfig {
                    enabled: true,
                    port: 50051,
                },
                client: GrpcClientConfig::default(),
                security: SecurityConfig::default(),
                transport: TransportConfig::default(),
                message: MessageConfig::default(),
            },
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 50051,
                max_connections: 100,
                connection_timeout_secs: 30,
                request_timeout_secs: 60,
                enable_tls: false,
                security: crate::config::SecurityConfig::default(),
                transport: crate::config::TransportConfig::default(),
                message: crate::config::MessageConfig::default(),
                compression: crate::config::CompressionConfig::default(),
                streaming: crate::config::StreamingConfig::default(),
            },
            external_services: ExternalServicesConfig {
                qdrant: QdrantConfig {
                    url: "http://localhost:6333".to_string(),
                    api_key: None,
                    max_retries: 3,
                    default_collection: crate::config::CollectionConfig {
                        vector_size: 384,
                        distance_metric: "Cosine".to_string(),
                        enable_indexing: true,
                        replication_factor: 1,
                        shard_number: 1,
                    },
                },
            },
            transport: crate::config::TransportConfig::default(),
            message: crate::config::MessageConfig::default(),
            security: crate::config::SecurityConfig::default(),
            streaming: crate::config::StreamingConfig::default(),
            compression: crate::config::CompressionConfig::default(),
            database: DatabaseConfig {
                sqlite_path: db_path.to_string(),
                max_connections: 5,
                connection_timeout_secs: 30,
                enable_wal: true,
            },
            qdrant: QdrantConfig {
                url: "http://localhost:6333".to_string(),
                api_key: None,
                max_retries: 3,
                default_collection: crate::config::CollectionConfig {
                    vector_size: 384,
                    distance_metric: "Cosine".to_string(),
                    enable_indexing: true,
                    replication_factor: 1,
                    shard_number: 1,
                },
            },
            workspace: WorkspaceConfig {
                collection_basename: Some("test-workspace".to_string()),
                collection_types: vec!["code".to_string(), "notes".to_string()],
                memory_collection_name: "_memory".to_string(),
                auto_create_collections: true,
            },
            processing: ProcessingConfig {
                max_concurrent_tasks: 4,
                default_chunk_size: 1000,
                default_chunk_overlap: 200,
                max_file_size_bytes: 1024 * 1024,
                supported_extensions: vec!["txt".to_string(), "md".to_string()],
                enable_lsp: false,
                lsp_timeout_secs: 10,
            },
            file_watcher: FileWatcherConfig {
                enabled: false,
                debounce_ms: 500,
                max_watched_dirs: 10,
                ignore_patterns: vec![],
                recursive: true,
            },
            auto_ingestion: crate::config::AutoIngestionConfig::default(),
            metrics: MetricsConfig {
                enabled: false,
                collection_interval_secs: 60,
            },
            logging: LoggingConfig {
                enabled: true,
                level: "info".to_string(),
                file_path: None,
                max_file_size: SizeUnit(100 * 1024 * 1024), // 100MB
                max_files: 5,
                enable_json: false,
                enable_structured: false,
                enable_console: true,
            },
        }
    }

    async fn create_test_daemon() -> Arc<WorkspaceDaemon> {
        let config = create_test_daemon_config();
        Arc::new(WorkspaceDaemon::new(config).await.expect("Failed to create daemon"))
    }

    #[tokio::test]
    async fn test_health_check() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(());
        let result = service.health_check(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.status, ServiceStatus::Healthy as i32);
        assert_eq!(response.components.len(), 3);
        assert!(response.timestamp.is_some());

        // Check component names
        let component_names: Vec<_> = response.components.iter()
            .map(|c| c.component_name.as_str())
            .collect();
        assert!(component_names.contains(&"database"));
        assert!(component_names.contains(&"qdrant"));
        assert!(component_names.contains(&"file_watcher"));
    }

    #[tokio::test]
    async fn test_get_status() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(());
        let result = service.get_status(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.status, ServiceStatus::Healthy as i32);
        assert!(response.metrics.is_some());
        assert!(response.uptime_since.is_some());
    }

    #[tokio::test]
    async fn test_get_metrics() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(());
        let result = service.get_metrics(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(!response.metrics.is_empty());
        assert!(response.collected_at.is_some());

        // Should include our new metrics
        let metric_names: Vec<_> = response.metrics.iter()
            .map(|m| m.name.as_str())
            .collect();
        assert!(metric_names.contains(&"refresh_hints_pending"));
        assert!(metric_names.contains(&"active_sessions"));
    }

    #[tokio::test]
    async fn test_send_refresh_signal() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        // Send first refresh signal
        let request = Request::new(RefreshSignalRequest {
            queue_type: QueueType::IngestQueue as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        });

        let result = service.send_refresh_signal(request).await;
        assert!(result.is_ok());

        // Check that hint was stored
        let hints = service.get_refresh_hints().await;
        assert_eq!(hints.len(), 1);

        // Send duplicate - should not increase count
        let request = Request::new(RefreshSignalRequest {
            queue_type: QueueType::IngestQueue as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        });

        let result = service.send_refresh_signal(request).await;
        assert!(result.is_ok());

        let hints = service.get_refresh_hints().await;
        assert_eq!(hints.len(), 1); // Still 1 due to deduplication
    }

    #[tokio::test]
    async fn test_send_refresh_signal_different_types() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        // Send different queue types
        let queue_types = vec![
            QueueType::IngestQueue,
            QueueType::WatchedProjects,
            QueueType::WatchedFolders,
            QueueType::ToolsAvailable,
        ];

        for queue_type in queue_types {
            let request = Request::new(RefreshSignalRequest {
                queue_type: queue_type as i32,
                lsp_languages: vec![],
                grammar_languages: vec![],
            });

            let result = service.send_refresh_signal(request).await;
            assert!(result.is_ok());
        }

        // Should have 4 different hints
        let hints = service.get_refresh_hints().await;
        assert_eq!(hints.len(), 4);
    }

    #[tokio::test]
    async fn test_notify_server_status_up() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(ServerStatusNotification {
            state: ServerState::Up as i32,
            project_name: Some("test-project".to_string()),
            project_root: Some("/path/to/project".to_string()),
        });

        let result = service.notify_server_status(request).await;
        assert!(result.is_ok());

        // Check session was registered
        let sessions = service.get_active_sessions().await;
        assert_eq!(sessions.len(), 1);
        assert!(sessions.contains_key("/path/to/project"));
    }

    #[tokio::test]
    async fn test_notify_server_status_down() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        // Register session first
        let request = Request::new(ServerStatusNotification {
            state: ServerState::Up as i32,
            project_name: Some("test-project".to_string()),
            project_root: Some("/path/to/project".to_string()),
        });
        service.notify_server_status(request).await.unwrap();

        // Now send DOWN
        let request = Request::new(ServerStatusNotification {
            state: ServerState::Down as i32,
            project_name: Some("test-project".to_string()),
            project_root: Some("/path/to/project".to_string()),
        });

        let result = service.notify_server_status(request).await;
        assert!(result.is_ok());

        // Session should be removed
        let sessions = service.get_active_sessions().await;
        assert_eq!(sessions.len(), 0);
    }

    #[tokio::test]
    async fn test_pause_all_watchers() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(());
        let result = service.pause_all_watchers(request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_resume_all_watchers() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(());
        let result = service.resume_all_watchers(request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_concurrent_refresh_signals() {
        let daemon = create_test_daemon().await;
        let service = Arc::new(SystemServiceImpl::new(daemon));

        let mut handles = vec![];

        // Send 100 concurrent refresh signals
        for i in 0..100 {
            let service_clone = Arc::clone(&service);
            let handle = tokio::spawn(async move {
                let queue_type = match i % 4 {
                    0 => QueueType::IngestQueue,
                    1 => QueueType::WatchedProjects,
                    2 => QueueType::WatchedFolders,
                    _ => QueueType::ToolsAvailable,
                };

                let request = Request::new(RefreshSignalRequest {
                    queue_type: queue_type as i32,
                    lsp_languages: vec![],
                    grammar_languages: vec![],
                });

                service_clone.send_refresh_signal(request).await
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            assert!(handle.await.unwrap().is_ok());
        }

        // Should have only 4 unique hints (one per queue type)
        let hints = service.get_refresh_hints().await;
        assert_eq!(hints.len(), 4);
    }

    #[tokio::test]
    async fn test_refresh_hint_equality() {
        let hint1 = RefreshHint {
            queue_type: QueueType::IngestQueue,
            lsp_languages: vec!["rust".to_string()],
            grammar_languages: vec!["python".to_string()],
        };

        let hint2 = RefreshHint {
            queue_type: QueueType::IngestQueue,
            lsp_languages: vec!["rust".to_string()],
            grammar_languages: vec!["python".to_string()],
        };

        let hint3 = RefreshHint {
            queue_type: QueueType::WatchedProjects,
            lsp_languages: vec!["rust".to_string()],
            grammar_languages: vec!["python".to_string()],
        };

        assert_eq!(hint1, hint2);
        assert_ne!(hint1, hint3);

        // Test HashSet deduplication
        let mut set = HashSet::new();
        set.insert(hint1.clone());
        set.insert(hint2);
        assert_eq!(set.len(), 1);

        set.insert(hint3);
        assert_eq!(set.len(), 2);
    }
}

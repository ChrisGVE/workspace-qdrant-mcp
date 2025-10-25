//! System administration gRPC service implementation

use crate::daemon::WorkspaceDaemon;
use crate::proto::{
    system_service_server::SystemService,
    HealthCheckResponse, SystemStatusResponse, MetricsRequest, MetricsResponse,
    ConfigResponse, UpdateConfigRequest, DetectProjectRequest, DetectProjectResponse,
    ListProjectsResponse, ServiceStatus, ComponentHealth, SystemMetrics, ProjectInfo,
};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{debug, info};

/// System service implementation
#[derive(Debug)]
pub struct SystemServiceImpl {
    daemon: Arc<WorkspaceDaemon>,
}

impl SystemServiceImpl {
    pub fn new(daemon: Arc<WorkspaceDaemon>) -> Self {
        Self { daemon }
    }
}

#[tonic::async_trait]
impl SystemService for SystemServiceImpl {
    async fn health_check(
        &self,
        _request: Request<()>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        debug!("Health check requested");

        // TODO: Implement actual health checks
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
                active_connections: 5,
                pending_operations: 0,
            }),
            active_projects: vec!["workspace-qdrant-mcp".to_string()],
            total_documents: 1000,
            total_collections: 5,
            uptime_since: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp() - 3600, // 1 hour ago
                nanos: 0,
            }),
        };

        Ok(Response::new(response))
    }

    async fn get_metrics(
        &self,
        request: Request<MetricsRequest>,
    ) -> Result<Response<MetricsResponse>, Status> {
        let req = request.into_inner();
        debug!("Metrics requested: {:?}", req.metric_names);

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
            ],
            collected_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        };

        Ok(Response::new(response))
    }

    async fn get_config(
        &self,
        _request: Request<()>,
    ) -> Result<Response<ConfigResponse>, Status> {
        debug!("Configuration requested");

        let config = self.daemon.config();

        // Convert configuration to string map
        let mut configuration = std::collections::HashMap::new();
        configuration.insert("server.host".to_string(), config.server.host.clone());
        configuration.insert("server.port".to_string(), config.server.port.to_string());
        configuration.insert("qdrant.url".to_string(), config.qdrant.url.clone());
        configuration.insert("database.sqlite_path".to_string(), config.database.sqlite_path.clone());
        configuration.insert("processing.max_concurrent_tasks".to_string(), config.processing.max_concurrent_tasks.to_string());

        let response = ConfigResponse {
            configuration,
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        Ok(Response::new(response))
    }

    async fn update_config(
        &self,
        request: Request<UpdateConfigRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        info!("Configuration update requested: {:?}", req.configuration);

        // TODO: Implement configuration updates
        // For now, just log the request

        if req.restart_required {
            info!("Configuration update requires restart");
        }

        Ok(Response::new(()))
    }

    async fn detect_project(
        &self,
        request: Request<DetectProjectRequest>,
    ) -> Result<Response<DetectProjectResponse>, Status> {
        let req = request.into_inner();
        debug!("Project detection requested for: {}", req.path);

        // TODO: Implement actual project detection
        let response = DetectProjectResponse {
            project: Some(ProjectInfo {
                project_id: uuid::Uuid::new_v4().to_string(),
                name: "example-project".to_string(),
                root_path: req.path.clone(),
                git_repository: "https://github.com/example/project.git".to_string(),
                git_branch: "main".to_string(),
                submodules: vec![],
                metadata: [("detected_at".to_string(), chrono::Utc::now().to_rfc3339())].into_iter().collect(),
                detected_at: Some(prost_types::Timestamp {
                    seconds: chrono::Utc::now().timestamp(),
                    nanos: 0,
                }),
            }),
            is_valid_project: true,
            reasons: vec!["Git repository detected".to_string()],
        };

        Ok(Response::new(response))
    }

    async fn list_projects(
        &self,
        _request: Request<()>,
    ) -> Result<Response<ListProjectsResponse>, Status> {
        debug!("Project list requested");

        // TODO: Implement actual project listing
        let response = ListProjectsResponse {
            projects: vec![
                ProjectInfo {
                    project_id: uuid::Uuid::new_v4().to_string(),
                    name: "workspace-qdrant-mcp".to_string(),
                    root_path: "/Users/example/workspace-qdrant-mcp".to_string(),
                    git_repository: "https://github.com/example/workspace-qdrant-mcp.git".to_string(),
                    git_branch: "main".to_string(),
                    submodules: vec![],
                    metadata: [("language".to_string(), "rust,python".to_string())].into_iter().collect(),
                    detected_at: Some(prost_types::Timestamp {
                        seconds: chrono::Utc::now().timestamp() - 86400, // 1 day ago
                        nanos: 0,
                    }),
                },
            ],
        };

        Ok(Response::new(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use tonic::Request;
    use std::collections::HashMap;

    fn create_test_daemon_config() -> DaemonConfig {
        // Use in-memory SQLite database for tests
        let db_path = ":memory:";

        DaemonConfig {
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
            database: DatabaseConfig {
                sqlite_path: db_path.to_string(),
                max_connections: 5,
                connection_timeout_secs: 30,
                enable_wal: true,
            },
            qdrant: QdrantConfig {
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
            metrics: MetricsConfig {
                enabled: false,
                collection_interval_secs: 60,
                retention_days: 30,
                enable_prometheus: false,
                prometheus_port: 9090,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                file_path: None,
                json_format: false,
                max_file_size_mb: 100,
                max_files: 5,
            },
        }
    }

    async fn create_test_daemon() -> Arc<WorkspaceDaemon> {
        let config = create_test_daemon_config();
        Arc::new(WorkspaceDaemon::new(config).await.expect("Failed to create daemon"))
    }

    #[tokio::test]
    async fn test_system_service_impl_new() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon.clone());

        assert!(Arc::ptr_eq(&service.daemon, &daemon));
    }

    #[tokio::test]
    async fn test_system_service_impl_debug() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("SystemServiceImpl"));
        assert!(debug_str.contains("daemon"));
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

        // Check individual components
        let component_names: Vec<_> = response.components.iter()
            .map(|c| c.component_name.as_str())
            .collect();
        assert!(component_names.contains(&"database"));
        assert!(component_names.contains(&"qdrant"));
        assert!(component_names.contains(&"file_watcher"));

        // All components should be healthy
        for component in &response.components {
            assert_eq!(component.status, ServiceStatus::Healthy as i32);
            assert!(!component.message.is_empty());
            assert!(component.last_check.is_some());
        }
    }

    #[tokio::test]
    async fn test_health_check_timestamps() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let before_check = chrono::Utc::now().timestamp();
        let request = Request::new(());
        let result = service.health_check(request).await;
        let after_check = chrono::Utc::now().timestamp();

        assert!(result.is_ok());
        let response = result.unwrap().into_inner();

        // Main timestamp
        assert!(response.timestamp.is_some());
        let timestamp = response.timestamp.unwrap().seconds;
        assert!(timestamp >= before_check && timestamp <= after_check);

        // Component timestamps
        for component in &response.components {
            assert!(component.last_check.is_some());
            let component_timestamp = component.last_check.as_ref().unwrap().seconds;
            assert!(component_timestamp >= before_check && component_timestamp <= after_check);
        }
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
        assert_eq!(response.active_projects.len(), 1);
        assert_eq!(response.active_projects[0], "workspace-qdrant-mcp");
        assert_eq!(response.total_documents, 1000);
        assert_eq!(response.total_collections, 5);
        assert!(response.uptime_since.is_some());

        // Check metrics
        let metrics = response.metrics.unwrap();
        assert_eq!(metrics.cpu_usage_percent, 15.5);
        assert_eq!(metrics.memory_usage_bytes, 128 * 1024 * 1024);
        assert_eq!(metrics.memory_total_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(metrics.disk_usage_bytes, 2 * 1024 * 1024 * 1024);
        assert_eq!(metrics.disk_total_bytes, 500 * 1024 * 1024 * 1024);
        assert_eq!(metrics.active_connections, 5);
        assert_eq!(metrics.pending_operations, 0);
    }

    #[tokio::test]
    async fn test_get_status_uptime() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(());
        let result = service.get_status(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(response.uptime_since.is_some());

        let uptime_timestamp = response.uptime_since.unwrap().seconds;
        let current_time = chrono::Utc::now().timestamp();

        // Uptime should be in the past (simulated as 1 hour ago)
        assert!(uptime_timestamp < current_time);
        assert!(current_time - uptime_timestamp >= 3500); // Allow some margin
    }

    #[tokio::test]
    async fn test_get_metrics_basic() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(MetricsRequest {
            since: None,
            metric_names: vec!["grpc_requests_total".to_string()],
        });

        let result = service.get_metrics(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.metrics.len(), 2);
        assert!(response.collected_at.is_some());

        // Check specific metrics
        let grpc_metric = response.metrics.iter()
            .find(|m| m.name == "grpc_requests_total")
            .expect("grpc_requests_total metric not found");
        assert_eq!(grpc_metric.r#type, "counter");
        assert_eq!(grpc_metric.value, 150.0);
        assert!(grpc_metric.labels.contains_key("method"));
        assert!(grpc_metric.timestamp.is_some());

        let duration_metric = response.metrics.iter()
            .find(|m| m.name == "document_processing_duration_seconds")
            .expect("duration metric not found");
        assert_eq!(duration_metric.r#type, "histogram");
        assert_eq!(duration_metric.value, 2.5);
        assert!(duration_metric.labels.contains_key("status"));
    }

    #[tokio::test]
    async fn test_get_metrics_different_requests() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let metric_requests = [
            vec!["grpc_requests_total".to_string()],
            vec!["document_processing_duration_seconds".to_string()],
            vec!["grpc_requests_total".to_string(), "document_processing_duration_seconds".to_string()],
            vec![], // Empty list
            vec!["nonexistent_metric".to_string()],
        ];

        for metrics in metric_requests {
            let request = Request::new(MetricsRequest {
                since: None,
                metric_names: metrics.clone(),
            });

            let result = service.get_metrics(request).await;
            assert!(result.is_ok(), "Failed for metrics: {:?}", metrics);

            let response = result.unwrap().into_inner();
            // Current implementation returns 2 metrics regardless of request
            assert_eq!(response.metrics.len(), 2);
        }
    }

    #[tokio::test]
    async fn test_get_metrics_timestamps() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let before_request = chrono::Utc::now().timestamp();
        let request = Request::new(MetricsRequest {
            since: Some(prost_types::Timestamp {
                seconds: before_request - 3600,
                nanos: 0,
            }),
            metric_names: vec!["test_metric".to_string()],
        });

        let result = service.get_metrics(request).await;
        let after_request = chrono::Utc::now().timestamp();
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(response.collected_at.is_some());

        let collected_timestamp = response.collected_at.unwrap().seconds;
        assert!(collected_timestamp >= before_request && collected_timestamp <= after_request);

        // Check individual metric timestamps
        for metric in &response.metrics {
            assert!(metric.timestamp.is_some());
            let metric_timestamp = metric.timestamp.as_ref().unwrap().seconds;
            assert!(metric_timestamp >= before_request && metric_timestamp <= after_request);
        }
    }

    #[tokio::test]
    async fn test_get_config() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(());
        let result = service.get_config(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(!response.configuration.is_empty());
        assert!(!response.version.is_empty());

        // Check specific configuration values
        assert_eq!(response.configuration.get("server.host").unwrap(), "127.0.0.1");
        assert_eq!(response.configuration.get("server.port").unwrap(), "50051");
        assert_eq!(response.configuration.get("qdrant.url").unwrap(), "http://localhost:6333");
        assert!(response.configuration.contains_key("database.sqlite_path"));
        assert_eq!(response.configuration.get("processing.max_concurrent_tasks").unwrap(), "4");

        // Check version format
        let version = &response.version;
        assert!(version.contains('.'), "Version should contain dots: {}", version);
    }

    #[tokio::test]
    async fn test_update_config_basic() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let mut config_updates = HashMap::new();
        config_updates.insert("server.port".to_string(), "8080".to_string());
        config_updates.insert("processing.max_concurrent_tasks".to_string(), "8".to_string());

        let request = Request::new(UpdateConfigRequest {
            configuration: config_updates,
            restart_required: false,
        });

        let result = service.update_config(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response, ());
    }

    #[tokio::test]
    async fn test_update_config_restart_required() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let mut config_updates = HashMap::new();
        config_updates.insert("qdrant.url".to_string(), "http://new-host:6333".to_string());

        let request = Request::new(UpdateConfigRequest {
            configuration: config_updates,
            restart_required: true,
        });

        let result = service.update_config(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response, ());
    }

    #[tokio::test]
    async fn test_detect_project() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(DetectProjectRequest {
            path: "/path/to/project".to_string(),
        });

        let result = service.detect_project(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(response.project.is_some());
        assert!(response.is_valid_project);
        assert_eq!(response.reasons.len(), 1);
        assert_eq!(response.reasons[0], "Git repository detected");

        let project = response.project.unwrap();
        assert!(!project.project_id.is_empty());
        assert_eq!(project.name, "example-project");
        assert_eq!(project.root_path, "/path/to/project");
        assert_eq!(project.git_repository, "https://github.com/example/project.git");
        assert_eq!(project.git_branch, "main");
        assert!(project.submodules.is_empty());
        assert!(project.metadata.contains_key("detected_at"));
        assert!(project.detected_at.is_some());
    }

    #[tokio::test]
    async fn test_detect_project_different_paths() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let test_paths = [
            "/home/user/project",
            "/Users/developer/workspace/my-app",
            "./relative/path",
            "../parent/project",
            "/very/deep/nested/project/structure",
        ];

        for path in test_paths {
            let request = Request::new(DetectProjectRequest {
                path: path.to_string(),
            });

            let result = service.detect_project(request).await;
            assert!(result.is_ok(), "Failed for path: {}", path);

            let response = result.unwrap().into_inner();
            assert!(response.project.is_some());
            assert_eq!(response.project.unwrap().root_path, path);
        }
    }

    #[tokio::test]
    async fn test_detect_project_timestamps() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let before_detection = chrono::Utc::now().timestamp();
        let request = Request::new(DetectProjectRequest {
            path: "/timestamp/test/project".to_string(),
        });

        let result = service.detect_project(request).await;
        let after_detection = chrono::Utc::now().timestamp();
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(response.project.is_some());

        let project = response.project.unwrap();
        assert!(project.detected_at.is_some());

        let detected_timestamp = project.detected_at.unwrap().seconds;
        assert!(detected_timestamp >= before_detection && detected_timestamp <= after_detection);
    }

    #[tokio::test]
    async fn test_list_projects() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(());
        let result = service.list_projects(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.projects.len(), 1);

        let project = &response.projects[0];
        assert!(!project.project_id.is_empty());
        assert_eq!(project.name, "workspace-qdrant-mcp");
        assert_eq!(project.root_path, "/Users/example/workspace-qdrant-mcp");
        assert_eq!(project.git_repository, "https://github.com/example/workspace-qdrant-mcp.git");
        assert_eq!(project.git_branch, "main");
        assert!(project.submodules.is_empty());
        assert!(project.metadata.contains_key("language"));
        assert_eq!(project.metadata.get("language").unwrap(), "rust,python");
        assert!(project.detected_at.is_some());
    }

    #[tokio::test]
    async fn test_list_projects_timestamps() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(());
        let result = service.list_projects(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.projects.len(), 1);

        let project = &response.projects[0];
        assert!(project.detected_at.is_some());

        let detected_timestamp = project.detected_at.as_ref().unwrap().seconds;
        let current_time = chrono::Utc::now().timestamp();

        // Should be detected in the past (simulated as 1 day ago)
        assert!(detected_timestamp < current_time);
        assert!(current_time - detected_timestamp >= 86300); // Allow some margin for 1 day
    }

    #[tokio::test]
    async fn test_concurrent_system_operations() {
        let daemon = create_test_daemon().await;
        let service = Arc::new(SystemServiceImpl::new(daemon));

        let mut handles = vec![];

        // Perform multiple concurrent system operations
        for i in 0..5 {
            let service_clone = Arc::clone(&service);
            let handle = tokio::spawn(async move {
                // Health check
                let health_request = Request::new(());
                let health_result = service_clone.health_check(health_request).await;

                // Get status
                let status_request = Request::new(());
                let status_result = service_clone.get_status(status_request).await;

                // Get config
                let config_request = Request::new(());
                let config_result = service_clone.get_config(config_request).await;

                // Project detection
                let detect_request = Request::new(DetectProjectRequest {
                    path: format!("/test/project_{}", i),
                });
                let detect_result = service_clone.detect_project(detect_request).await;

                (health_result, status_result, config_result, detect_result)
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        let results: Vec<_> = futures_util::future::join_all(handles).await;

        // All operations should complete successfully
        for (i, result) in results.into_iter().enumerate() {
            let (health_result, status_result, config_result, detect_result) = result.unwrap();
            assert!(health_result.is_ok(), "Health check {} failed", i);
            assert!(status_result.is_ok(), "Status check {} failed", i);
            assert!(config_result.is_ok(), "Config check {} failed", i);
            assert!(detect_result.is_ok(), "Project detection {} failed", i);
        }
    }

    #[tokio::test]
    async fn test_project_unique_ids() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let mut project_ids = std::collections::HashSet::new();

        // Detect multiple projects and verify unique IDs
        for i in 0..5 {
            let request = Request::new(DetectProjectRequest {
                path: format!("/test/unique_project_{}", i),
            });

            let result = service.detect_project(request).await;
            assert!(result.is_ok());

            let response = result.unwrap().into_inner();
            assert!(response.project.is_some());

            let project = response.project.unwrap();
            assert!(project_ids.insert(project.project_id.clone()),
                    "Duplicate project ID generated: {}", project.project_id);
        }

        assert_eq!(project_ids.len(), 5);
    }

    #[tokio::test]
    async fn test_metrics_labels_structure() {
        let daemon = create_test_daemon().await;
        let service = SystemServiceImpl::new(daemon);

        let request = Request::new(MetricsRequest {
            since: None,
            metric_names: vec!["test_metric".to_string()],
        });

        let result = service.get_metrics(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.metrics.len(), 2);

        for metric in &response.metrics {
            assert!(!metric.name.is_empty());
            assert!(!metric.r#type.is_empty());
            assert!(!metric.labels.is_empty());
            assert!(metric.value >= 0.0);
            assert!(metric.timestamp.is_some());
        }
    }

    #[test]
    fn test_system_service_impl_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<SystemServiceImpl>();
        assert_sync::<SystemServiceImpl>();
    }
}
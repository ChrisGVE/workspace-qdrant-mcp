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
use tracing::{debug, error, info};

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
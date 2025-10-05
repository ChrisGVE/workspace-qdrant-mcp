//! SystemService gRPC implementation
//!
//! Handles system health monitoring, status reporting, refresh signaling,
//! and lifecycle management operations.
//! Provides 7 RPCs: HealthCheck, GetStatus, GetMetrics, SendRefreshSignal,
//! NotifyServerStatus, PauseAllWatchers, ResumeAllWatchers

use std::sync::Arc;
use std::time::SystemTime;
use tonic::{Request, Response, Status};
use tracing::{debug, info};

use crate::proto::{
    system_service_server::SystemService,
    HealthCheckResponse, SystemStatusResponse, MetricsResponse,
    RefreshSignalRequest, ServerStatusNotification,
    ComponentHealth, SystemMetrics, Metric,
    ServiceStatus,
};

/// SystemService implementation
///
/// This is a basic implementation that provides minimal stub functionality
/// for testing purposes. Full integration with actual system monitoring
/// will be implemented in later phases.
#[derive(Debug)]
pub struct SystemServiceImpl {
    start_time: SystemTime,
}

impl SystemServiceImpl {
    /// Create a new SystemService
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
        }
    }
}

impl Default for SystemServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl SystemService for SystemServiceImpl {
    /// Quick health check for monitoring/alerting
    async fn health_check(
        &self,
        _request: Request<()>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        debug!("Health check requested");

        let response = HealthCheckResponse {
            status: ServiceStatus::Healthy as i32,
            components: vec![
                ComponentHealth {
                    component_name: "grpc_server".to_string(),
                    status: ServiceStatus::Healthy as i32,
                    message: "Running".to_string(),
                    last_check: Some(prost_types::Timestamp::from(SystemTime::now())),
                },
            ],
            timestamp: Some(prost_types::Timestamp::from(SystemTime::now())),
        };

        Ok(Response::new(response))
    }

    /// Comprehensive system state snapshot
    async fn get_status(
        &self,
        _request: Request<()>,
    ) -> Result<Response<SystemStatusResponse>, Status> {
        info!("System status requested");

        let response = SystemStatusResponse {
            status: ServiceStatus::Healthy as i32,
            metrics: Some(SystemMetrics {
                cpu_usage_percent: 0.0,
                memory_usage_bytes: 0,
                memory_total_bytes: 0,
                disk_usage_bytes: 0,
                disk_total_bytes: 0,
                active_connections: 1,
                pending_operations: 0,
            }),
            active_projects: vec![],
            total_documents: 0,
            total_collections: 0,
            uptime_since: Some(prost_types::Timestamp::from(self.start_time)),
        };

        Ok(Response::new(response))
    }

    /// Current performance metrics (no historical data)
    async fn get_metrics(
        &self,
        _request: Request<()>,
    ) -> Result<Response<MetricsResponse>, Status> {
        debug!("Metrics requested");

        let response = MetricsResponse {
            metrics: vec![
                Metric {
                    name: "requests_total".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: 0.0,
                    timestamp: Some(prost_types::Timestamp::from(SystemTime::now())),
                },
            ],
            collected_at: Some(prost_types::Timestamp::from(SystemTime::now())),
        };

        Ok(Response::new(response))
    }

    /// Signal database state changes for event-driven refresh
    async fn send_refresh_signal(
        &self,
        request: Request<RefreshSignalRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        debug!("Refresh signal received: queue_type={:?}", req.queue_type);

        // Stub implementation - just acknowledge receipt
        Ok(Response::new(()))
    }

    /// MCP/CLI server lifecycle notifications
    async fn notify_server_status(
        &self,
        request: Request<ServerStatusNotification>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        info!(
            "Server status notification: state={:?}, project={:?}",
            req.state, req.project_name
        );

        // Stub implementation - just acknowledge receipt
        Ok(Response::new(()))
    }

    /// Pause all file watchers (master switch)
    async fn pause_all_watchers(
        &self,
        _request: Request<()>,
    ) -> Result<Response<()>, Status> {
        info!("Pause all watchers requested");

        // Stub implementation - would pause watchers in full implementation
        Ok(Response::new(()))
    }

    /// Resume all file watchers (master switch)
    async fn resume_all_watchers(
        &self,
        _request: Request<()>,
    ) -> Result<Response<()>, Status> {
        info!("Resume all watchers requested");

        // Stub implementation - would resume watchers in full implementation
        Ok(Response::new(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_creation() {
        let service = SystemServiceImpl::new();
        assert!(service.start_time <= SystemTime::now());
    }

    #[tokio::test]
    async fn test_default_trait() {
        let _service = SystemServiceImpl::default();
        // Should not panic
    }
}

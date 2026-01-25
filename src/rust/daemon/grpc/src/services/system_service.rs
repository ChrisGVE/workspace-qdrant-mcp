//! SystemService gRPC implementation
//!
//! Handles system health monitoring, status reporting, refresh signaling,
//! and lifecycle management operations.
//! Provides 7 RPCs: HealthCheck, GetStatus, GetMetrics, SendRefreshSignal,
//! NotifyServerStatus, PauseAllWatchers, ResumeAllWatchers

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

/// Queue processor health state for monitoring
#[derive(Debug, Default)]
pub struct QueueProcessorHealth {
    /// Whether the processor is currently running
    pub is_running: AtomicBool,
    /// Last poll timestamp (Unix millis)
    pub last_poll_time: AtomicU64,
    /// Total error count
    pub error_count: AtomicU64,
    /// Items processed total
    pub items_processed: AtomicU64,
    /// Items failed total
    pub items_failed: AtomicU64,
    /// Current queue depth
    pub queue_depth: AtomicU64,
    /// Average processing time in milliseconds
    pub avg_processing_time_ms: AtomicU64,
}

impl QueueProcessorHealth {
    /// Create new health state
    pub fn new() -> Self {
        Self::default()
    }

    /// Update running status
    pub fn set_running(&self, running: bool) {
        self.is_running.store(running, Ordering::SeqCst);
    }

    /// Update last poll time to now
    pub fn record_poll(&self) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_poll_time.store(now, Ordering::SeqCst);
    }

    /// Increment error count
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Record successful processing
    pub fn record_success(&self, processing_time_ms: u64) {
        let prev_count = self.items_processed.fetch_add(1, Ordering::SeqCst);
        let prev_avg = self.avg_processing_time_ms.load(Ordering::SeqCst);
        // Running average calculation
        let new_avg = if prev_count == 0 {
            processing_time_ms
        } else {
            (prev_avg * prev_count + processing_time_ms) / (prev_count + 1)
        };
        self.avg_processing_time_ms.store(new_avg, Ordering::SeqCst);
    }

    /// Record failed processing
    pub fn record_failure(&self) {
        self.items_failed.fetch_add(1, Ordering::SeqCst);
    }

    /// Update queue depth
    pub fn set_queue_depth(&self, depth: u64) {
        self.queue_depth.store(depth, Ordering::SeqCst);
    }

    /// Get seconds since last poll
    pub fn seconds_since_last_poll(&self) -> u64 {
        let last = self.last_poll_time.load(Ordering::SeqCst);
        if last == 0 {
            return u64::MAX; // Never polled
        }
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        (now.saturating_sub(last)) / 1000
    }
}

/// SystemService implementation
///
/// Provides health monitoring, status reporting, and lifecycle management.
/// Can be connected to actual queue processor health state for real metrics.
#[derive(Debug)]
pub struct SystemServiceImpl {
    start_time: SystemTime,
    /// Optional queue processor health state
    queue_health: Option<Arc<QueueProcessorHealth>>,
}

impl SystemServiceImpl {
    /// Create a new SystemService
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            queue_health: None,
        }
    }

    /// Create with queue processor health monitoring
    pub fn with_queue_health(queue_health: Arc<QueueProcessorHealth>) -> Self {
        Self {
            start_time: SystemTime::now(),
            queue_health: Some(queue_health),
        }
    }

    /// Get queue processor health component
    fn get_queue_processor_health(&self) -> ComponentHealth {
        if let Some(health) = &self.queue_health {
            let is_running = health.is_running.load(Ordering::SeqCst);
            let secs_since_poll = health.seconds_since_last_poll();
            let error_count = health.error_count.load(Ordering::SeqCst);

            // Determine status based on health indicators
            let (status, message) = if !is_running {
                (ServiceStatus::Unhealthy, "Queue processor is not running")
            } else if secs_since_poll > 60 {
                (ServiceStatus::Degraded, "Queue processor may be stalled (>60s since last poll)")
            } else if error_count > 100 {
                (ServiceStatus::Degraded, "High error count detected")
            } else {
                (ServiceStatus::Healthy, "Running normally")
            };

            ComponentHealth {
                component_name: "queue_processor".to_string(),
                status: status as i32,
                message: message.to_string(),
                last_check: Some(prost_types::Timestamp::from(SystemTime::now())),
            }
        } else {
            // No health state connected - report unknown
            ComponentHealth {
                component_name: "queue_processor".to_string(),
                status: ServiceStatus::Unspecified as i32,
                message: "Health monitoring not connected".to_string(),
                last_check: Some(prost_types::Timestamp::from(SystemTime::now())),
            }
        }
    }

    /// Get queue processor metrics
    fn get_queue_metrics(&self) -> Vec<Metric> {
        let now = Some(prost_types::Timestamp::from(SystemTime::now()));

        if let Some(health) = &self.queue_health {
            vec![
                Metric {
                    name: "queue_pending".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.queue_depth.load(Ordering::SeqCst) as f64,
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_processed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.items_processed.load(Ordering::SeqCst) as f64,
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_failed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.items_failed.load(Ordering::SeqCst) as f64,
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_errors".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.error_count.load(Ordering::SeqCst) as f64,
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_processing_avg_ms".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.avg_processing_time_ms.load(Ordering::SeqCst) as f64,
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_processor_running".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: if health.is_running.load(Ordering::SeqCst) { 1.0 } else { 0.0 },
                    timestamp: now,
                },
            ]
        } else {
            // No health state connected - return placeholder metrics
            vec![
                Metric {
                    name: "queue_pending".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: 0.0,
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_processed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: 0.0,
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_failed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: 0.0,
                    timestamp: now,
                },
            ]
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

        // Build component health list
        let mut components = vec![
            ComponentHealth {
                component_name: "grpc_server".to_string(),
                status: ServiceStatus::Healthy as i32,
                message: "Running".to_string(),
                last_check: Some(prost_types::Timestamp::from(SystemTime::now())),
            },
        ];

        // Add queue processor health
        let queue_health = self.get_queue_processor_health();
        let queue_status = queue_health.status;
        components.push(queue_health);

        // Determine overall status (worst of all components)
        let overall_status = if components.iter().any(|c| c.status == ServiceStatus::Unhealthy as i32) {
            ServiceStatus::Unhealthy
        } else if components.iter().any(|c| c.status == ServiceStatus::Degraded as i32) {
            ServiceStatus::Degraded
        } else if components.iter().any(|c| c.status == ServiceStatus::Unspecified as i32) {
            // If queue health is unknown but gRPC is healthy, still report healthy
            if queue_status == ServiceStatus::Unspecified as i32 {
                ServiceStatus::Healthy
            } else {
                ServiceStatus::Unspecified
            }
        } else {
            ServiceStatus::Healthy
        };

        let response = HealthCheckResponse {
            status: overall_status as i32,
            components,
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

        // Get queue depth from health state if available
        let pending_operations = self.queue_health
            .as_ref()
            .map(|h| h.queue_depth.load(Ordering::SeqCst) as i32)
            .unwrap_or(0);

        let response = SystemStatusResponse {
            status: ServiceStatus::Healthy as i32,
            metrics: Some(SystemMetrics {
                cpu_usage_percent: 0.0,
                memory_usage_bytes: 0,
                memory_total_bytes: 0,
                disk_usage_bytes: 0,
                disk_total_bytes: 0,
                active_connections: 1,
                pending_operations,
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

        let now = Some(prost_types::Timestamp::from(SystemTime::now()));

        // Start with general metrics
        let mut metrics = vec![
            Metric {
                name: "requests_total".to_string(),
                r#type: "counter".to_string(),
                labels: std::collections::HashMap::new(),
                value: 0.0,
                timestamp: now.clone(),
            },
            Metric {
                name: "uptime_seconds".to_string(),
                r#type: "gauge".to_string(),
                labels: std::collections::HashMap::new(),
                value: self.start_time
                    .elapsed()
                    .map(|d| d.as_secs() as f64)
                    .unwrap_or(0.0),
                timestamp: now.clone(),
            },
        ];

        // Add queue processor metrics
        metrics.extend(self.get_queue_metrics());

        let response = MetricsResponse {
            metrics,
            collected_at: now,
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

    #[tokio::test]
    async fn test_service_with_queue_health() {
        let health = Arc::new(QueueProcessorHealth::new());
        health.set_running(true);
        health.set_queue_depth(42);

        let service = SystemServiceImpl::with_queue_health(health.clone());
        assert!(service.queue_health.is_some());

        // Test health check includes queue processor
        let response = service.health_check(Request::new(())).await.unwrap();
        let health_response = response.into_inner();
        assert!(health_response.components.len() >= 2);
        assert!(health_response.components.iter().any(|c| c.component_name == "queue_processor"));
    }

    #[tokio::test]
    async fn test_queue_processor_health_metrics() {
        let health = QueueProcessorHealth::new();

        // Test initial state
        assert!(!health.is_running.load(Ordering::SeqCst));
        assert_eq!(health.error_count.load(Ordering::SeqCst), 0);

        // Test running state
        health.set_running(true);
        assert!(health.is_running.load(Ordering::SeqCst));

        // Test error recording
        health.record_error();
        health.record_error();
        assert_eq!(health.error_count.load(Ordering::SeqCst), 2);

        // Test success recording
        health.record_success(100);
        health.record_success(200);
        assert_eq!(health.items_processed.load(Ordering::SeqCst), 2);
        // Average should be approximately 150
        let avg = health.avg_processing_time_ms.load(Ordering::SeqCst);
        assert!(avg > 0);

        // Test failure recording
        health.record_failure();
        assert_eq!(health.items_failed.load(Ordering::SeqCst), 1);

        // Test queue depth
        health.set_queue_depth(100);
        assert_eq!(health.queue_depth.load(Ordering::SeqCst), 100);
    }

    #[tokio::test]
    async fn test_queue_processor_health_poll_time() {
        let health = QueueProcessorHealth::new();

        // Before any poll, should return MAX
        assert_eq!(health.seconds_since_last_poll(), u64::MAX);

        // After poll, should return small number
        health.record_poll();
        let secs = health.seconds_since_last_poll();
        assert!(secs < 2); // Should be nearly instant
    }

    #[tokio::test]
    async fn test_metrics_include_queue_metrics() {
        let health = Arc::new(QueueProcessorHealth::new());
        health.set_running(true);
        health.set_queue_depth(10);
        health.record_success(50);

        let service = SystemServiceImpl::with_queue_health(health);
        let response = service.get_metrics(Request::new(())).await.unwrap();
        let metrics = response.into_inner().metrics;

        // Check that queue metrics are present
        let metric_names: Vec<&str> = metrics.iter().map(|m| m.name.as_str()).collect();
        assert!(metric_names.contains(&"queue_pending"));
        assert!(metric_names.contains(&"queue_processed"));
        assert!(metric_names.contains(&"queue_failed"));
        assert!(metric_names.contains(&"queue_processor_running"));

        // Verify values
        let pending = metrics.iter().find(|m| m.name == "queue_pending").unwrap();
        assert_eq!(pending.value, 10.0);

        let running = metrics.iter().find(|m| m.name == "queue_processor_running").unwrap();
        assert_eq!(running.value, 1.0);
    }

    #[tokio::test]
    async fn test_status_includes_queue_depth() {
        let health = Arc::new(QueueProcessorHealth::new());
        health.set_queue_depth(25);

        let service = SystemServiceImpl::with_queue_health(health);
        let response = service.get_status(Request::new(())).await.unwrap();
        let status = response.into_inner();

        assert_eq!(status.metrics.unwrap().pending_operations, 25);
    }

    #[tokio::test]
    async fn test_health_status_degraded_on_high_errors() {
        let health = Arc::new(QueueProcessorHealth::new());
        health.set_running(true);
        health.record_poll();

        // Record many errors
        for _ in 0..101 {
            health.record_error();
        }

        let service = SystemServiceImpl::with_queue_health(health);
        let response = service.health_check(Request::new(())).await.unwrap();
        let health_response = response.into_inner();

        let queue_comp = health_response.components.iter()
            .find(|c| c.component_name == "queue_processor")
            .unwrap();
        assert_eq!(queue_comp.status, ServiceStatus::Degraded as i32);
    }

    #[tokio::test]
    async fn test_health_status_unhealthy_when_not_running() {
        let health = Arc::new(QueueProcessorHealth::new());
        health.set_running(false);

        let service = SystemServiceImpl::with_queue_health(health);
        let response = service.health_check(Request::new(())).await.unwrap();
        let health_response = response.into_inner();

        let queue_comp = health_response.components.iter()
            .find(|c| c.component_name == "queue_processor")
            .unwrap();
        assert_eq!(queue_comp.status, ServiceStatus::Unhealthy as i32);
    }
}

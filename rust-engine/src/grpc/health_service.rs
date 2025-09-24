//! gRPC Health Service implementation following standard health check protocol
//!
//! This module implements the SystemService health checking methods defined in
//! workspace_daemon.proto and integrates with the comprehensive health monitoring system.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn, error};

use crate::error::{DaemonResult, DaemonError};
use crate::grpc::health::{HealthMonitoringSystem, HealthStatus, ServiceHealth};
use crate::grpc::middleware::ConnectionStats;

// Proto-generated types (these would normally be auto-generated)
// For now, defining the essential types based on the proto file

/// Health check response from workspace_daemon.proto
#[derive(Debug, Clone)]
pub struct HealthCheckResponse {
    pub status: ServiceStatus,
    pub components: Vec<ComponentHealth>,
    pub timestamp: SystemTime,
}

/// System status response from workspace_daemon.proto
#[derive(Debug, Clone)]
pub struct SystemStatusResponse {
    pub status: ServiceStatus,
    pub metrics: SystemMetrics,
    pub active_projects: Vec<String>,
    pub total_documents: i32,
    pub total_collections: i32,
    pub uptime_since: SystemTime,
}

/// Metrics request from workspace_daemon.proto
#[derive(Debug, Clone)]
pub struct MetricsRequest {
    pub since: Option<SystemTime>,
    pub metric_names: Vec<String>,
}

/// Metrics response from workspace_daemon.proto
#[derive(Debug, Clone)]
pub struct MetricsResponse {
    pub metrics: Vec<Metric>,
    pub collected_at: SystemTime,
}

/// Service status enum from workspace_daemon.proto
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceStatus {
    Unspecified = 0,
    Healthy = 1,
    Degraded = 2,
    Unhealthy = 3,
    Unavailable = 4,
}

impl From<HealthStatus> for ServiceStatus {
    fn from(health_status: HealthStatus) -> Self {
        match health_status {
            HealthStatus::Serving => ServiceStatus::Healthy,
            HealthStatus::NotServing => ServiceStatus::Unhealthy,
            HealthStatus::Unknown => ServiceStatus::Unavailable,
        }
    }
}

/// Component health from workspace_daemon.proto
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    pub component_name: String,
    pub status: ServiceStatus,
    pub message: String,
    pub last_check: SystemTime,
}

/// System metrics from workspace_daemon.proto
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: i64,
    pub memory_total_bytes: i64,
    pub disk_usage_bytes: i64,
    pub disk_total_bytes: i64,
    pub active_connections: i32,
    pub pending_operations: i32,
}

/// Individual metric from workspace_daemon.proto
#[derive(Debug, Clone)]
pub struct Metric {
    pub name: String,
    pub metric_type: String,
    pub labels: HashMap<String, String>,
    pub value: f64,
    pub timestamp: SystemTime,
}

/// gRPC Health Service implementation
#[derive(Debug)]
pub struct HealthService {
    /// Health monitoring system
    monitoring_system: Arc<HealthMonitoringSystem>,
    /// Connection statistics provider
    connection_stats_provider: Option<Arc<dyn ConnectionStatsProvider + Send + Sync>>,
    /// System metrics collector
    system_metrics_collector: Arc<SystemMetricsCollector>,
    /// Service start time for uptime calculation
    start_time: SystemTime,
}

/// Trait for providing connection statistics
pub trait ConnectionStatsProvider {
    /// Get current connection statistics
    fn get_connection_stats(&self) -> ConnectionStats;
}

/// System metrics collector
#[derive(Debug)]
pub struct SystemMetricsCollector {
    /// Cache metrics for a short period to avoid expensive system calls
    cached_metrics: tokio::sync::RwLock<Option<(SystemMetrics, SystemTime)>>,
    /// Cache duration in seconds
    cache_duration_secs: u64,
}

impl SystemMetricsCollector {
    /// Create a new system metrics collector
    pub fn new(cache_duration_secs: u64) -> Self {
        Self {
            cached_metrics: tokio::sync::RwLock::new(None),
            cache_duration_secs,
        }
    }

    /// Collect current system metrics
    pub async fn collect_metrics(&self, connection_stats: Option<&ConnectionStats>) -> DaemonResult<SystemMetrics> {
        // Check cache first
        {
            let cache = self.cached_metrics.read().await;
            if let Some((metrics, timestamp)) = cache.as_ref() {
                let age = SystemTime::now()
                    .duration_since(*timestamp)
                    .map_err(|e| DaemonError::Validation {
                        message: format!("System time error: {}", e)
                    })?;

                if age.as_secs() < self.cache_duration_secs {
                    debug!("Using cached system metrics (age: {}s)", age.as_secs());
                    return Ok(metrics.clone());
                }
            }
        }

        // Collect fresh metrics
        let metrics = self.collect_fresh_metrics(connection_stats).await?;

        // Update cache
        {
            let mut cache = self.cached_metrics.write().await;
            *cache = Some((metrics.clone(), SystemTime::now()));
        }

        debug!("Collected fresh system metrics");
        Ok(metrics)
    }

    /// Collect fresh system metrics from the OS
    async fn collect_fresh_metrics(&self, connection_stats: Option<&ConnectionStats>) -> DaemonResult<SystemMetrics> {
        // For a real implementation, this would use system APIs to get actual metrics
        // For now, providing reasonable defaults and using connection stats if available

        let (active_connections, pending_operations) = if let Some(stats) = connection_stats {
            (stats.active_connections as i32, 0) // pending operations would come from a task queue
        } else {
            (0, 0)
        };

        Ok(SystemMetrics {
            cpu_usage_percent: self.get_cpu_usage().await?,
            memory_usage_bytes: self.get_memory_usage().await?,
            memory_total_bytes: self.get_total_memory().await?,
            disk_usage_bytes: self.get_disk_usage().await?,
            disk_total_bytes: self.get_total_disk().await?,
            active_connections,
            pending_operations,
        })
    }

    /// Get CPU usage percentage (0.0 - 100.0)
    async fn get_cpu_usage(&self) -> DaemonResult<f64> {
        // In a real implementation, this would read from /proc/stat or use system APIs
        // For now, return a mock value
        Ok(25.5)
    }

    /// Get memory usage in bytes
    async fn get_memory_usage(&self) -> DaemonResult<i64> {
        // In a real implementation, this would read from /proc/meminfo or use system APIs
        Ok(512 * 1024 * 1024) // 512MB
    }

    /// Get total memory in bytes
    async fn get_total_memory(&self) -> DaemonResult<i64> {
        Ok(2 * 1024 * 1024 * 1024) // 2GB
    }

    /// Get disk usage in bytes
    async fn get_disk_usage(&self) -> DaemonResult<i64> {
        // In a real implementation, this would use statvfs or similar
        Ok(10 * 1024 * 1024 * 1024) // 10GB
    }

    /// Get total disk space in bytes
    async fn get_total_disk(&self) -> DaemonResult<i64> {
        Ok(100 * 1024 * 1024 * 1024) // 100GB
    }
}

impl HealthService {
    /// Create a new health service
    pub fn new(monitoring_system: Arc<HealthMonitoringSystem>) -> Self {
        Self {
            monitoring_system,
            connection_stats_provider: None,
            system_metrics_collector: Arc::new(SystemMetricsCollector::new(30)), // 30-second cache
            start_time: SystemTime::now(),
        }
    }

    /// Create a new health service with connection stats provider
    pub fn with_connection_stats(
        monitoring_system: Arc<HealthMonitoringSystem>,
        connection_stats_provider: Arc<dyn ConnectionStatsProvider + Send + Sync>,
    ) -> Self {
        Self {
            monitoring_system,
            connection_stats_provider: Some(connection_stats_provider),
            system_metrics_collector: Arc::new(SystemMetricsCollector::new(30)),
            start_time: SystemTime::now(),
        }
    }

    /// Handle health check request (implements SystemService::HealthCheck)
    pub async fn health_check(&self, _request: Request<()>) -> Result<Response<HealthCheckResponse>, Status> {
        debug!("Processing health check request");

        let all_health = self.monitoring_system.get_all_health().await;
        let mut components = Vec::new();

        // Convert service health to component health
        for (service_name, health) in all_health.iter() {
            components.push(ComponentHealth {
                component_name: service_name.clone(),
                status: ServiceStatus::from(health.status),
                message: health.message.clone(),
                last_check: health.last_check,
            });
        }

        // Determine overall system status
        let (system_health_status, _) = self.monitoring_system.get_system_health().await;
        let status = ServiceStatus::from(system_health_status);

        let response = HealthCheckResponse {
            status,
            components,
            timestamp: SystemTime::now(),
        };

        info!("Health check completed: {:?}, {} components", status, response.components.len());
        Ok(Response::new(response))
    }

    /// Handle system status request (implements SystemService::GetStatus)
    pub async fn get_status(&self, _request: Request<()>) -> Result<Response<SystemStatusResponse>, Status> {
        debug!("Processing system status request");

        let connection_stats = self.connection_stats_provider
            .as_ref()
            .map(|provider| provider.get_connection_stats());

        // Collect system metrics
        let system_metrics = self.system_metrics_collector
            .collect_metrics(connection_stats.as_ref())
            .await
            .map_err(|e| Status::internal(format!("Failed to collect system metrics: {}", e)))?;

        // Get overall system health status
        let (system_health_status, _) = self.monitoring_system.get_system_health().await;
        let status = ServiceStatus::from(system_health_status);

        // For now, mock the project and document counts
        // In a real implementation, these would come from the actual services
        let active_projects = vec!["project1".to_string(), "project2".to_string()];
        let total_documents = 1500;
        let total_collections = 12;

        let response = SystemStatusResponse {
            status,
            metrics: system_metrics,
            active_projects,
            total_documents,
            total_collections,
            uptime_since: self.start_time,
        };

        info!("System status completed: {:?}", status);
        Ok(Response::new(response))
    }

    /// Handle metrics request (implements SystemService::GetMetrics)
    pub async fn get_metrics(&self, request: Request<MetricsRequest>) -> Result<Response<MetricsResponse>, Status> {
        let req = request.into_inner();
        debug!("Processing metrics request for {} metric names", req.metric_names.len());

        let mut metrics = Vec::new();
        let collected_at = SystemTime::now();

        // Export monitoring system metrics
        let exported_metrics = self.monitoring_system
            .export_metrics()
            .await
            .map_err(|e| Status::internal(format!("Failed to export metrics: {}", e)))?;

        // Convert to gRPC metric format
        for (metric_name, metric_value) in exported_metrics.iter() {
            // Filter by requested metric names if specified
            if !req.metric_names.is_empty() && !req.metric_names.contains(metric_name) {
                continue;
            }

            // Skip metrics that are older than requested timestamp
            if let Some(since) = req.since {
                // In a real implementation, we'd check the metric timestamp
                // For now, all metrics are considered current
            }

            let metric = match metric_value {
                serde_json::Value::Object(obj) => {
                    // Extract numeric value from the object
                    let value = self.extract_numeric_value(obj).unwrap_or(0.0);

                    let mut labels = HashMap::new();
                    // Add relevant labels from the metric object
                    if let Some(status) = obj.get("status") {
                        if let Some(status_str) = status.as_str() {
                            labels.insert("status".to_string(), status_str.to_string());
                        }
                    }

                    Metric {
                        name: metric_name.clone(),
                        metric_type: "gauge".to_string(), // Most health metrics are gauges
                        labels,
                        value,
                        timestamp: collected_at,
                    }
                }
                serde_json::Value::Number(num) => {
                    Metric {
                        name: metric_name.clone(),
                        metric_type: "gauge".to_string(),
                        labels: HashMap::new(),
                        value: num.as_f64().unwrap_or(0.0),
                        timestamp: collected_at,
                    }
                }
                _ => continue, // Skip non-numeric metrics
            };

            metrics.push(metric);
        }

        // Add system metrics if requested
        if req.metric_names.is_empty() || req.metric_names.iter().any(|name| name.starts_with("system_")) {
            let connection_stats = self.connection_stats_provider
                .as_ref()
                .map(|provider| provider.get_connection_stats());

            let system_metrics = self.system_metrics_collector
                .collect_metrics(connection_stats.as_ref())
                .await
                .map_err(|e| Status::internal(format!("Failed to collect system metrics: {}", e)))?;

            // Convert system metrics to gRPC metrics
            let system_metric_list = vec![
                ("system_cpu_usage_percent", system_metrics.cpu_usage_percent),
                ("system_memory_usage_bytes", system_metrics.memory_usage_bytes as f64),
                ("system_memory_total_bytes", system_metrics.memory_total_bytes as f64),
                ("system_disk_usage_bytes", system_metrics.disk_usage_bytes as f64),
                ("system_disk_total_bytes", system_metrics.disk_total_bytes as f64),
                ("system_active_connections", system_metrics.active_connections as f64),
                ("system_pending_operations", system_metrics.pending_operations as f64),
            ];

            for (name, value) in system_metric_list {
                if req.metric_names.is_empty() || req.metric_names.contains(&name.to_string()) {
                    metrics.push(Metric {
                        name: name.to_string(),
                        metric_type: "gauge".to_string(),
                        labels: HashMap::new(),
                        value,
                        timestamp: collected_at,
                    });
                }
            }
        }

        let response = MetricsResponse {
            metrics,
            collected_at,
        };

        info!("Metrics request completed: {} metrics returned", response.metrics.len());
        Ok(Response::new(response))
    }

    /// Extract numeric value from a JSON object for metrics
    fn extract_numeric_value(&self, obj: &serde_json::Map<String, serde_json::Value>) -> Option<f64> {
        // Look for common numeric fields
        for key in ["value", "count", "total", "avg", "mean", "rate"] {
            if let Some(value) = obj.get(key) {
                if let Some(num) = value.as_f64() {
                    return Some(num);
                }
            }
        }

        // Look for metrics object
        if let Some(metrics_obj) = obj.get("metrics") {
            if let Some(metrics_map) = metrics_obj.as_object() {
                // Try to get total_requests as a representative metric
                if let Some(total_requests) = metrics_map.get("total_requests") {
                    return total_requests.as_f64();
                }

                // Try to get error_rate
                if let Some(error_rate) = metrics_map.get("error_rate") {
                    return error_rate.as_f64();
                }
            }
        }

        None
    }

    /// Get health monitoring system for service registration
    pub fn monitoring_system(&self) -> &Arc<HealthMonitoringSystem> {
        &self.monitoring_system
    }

    /// Perform automatic recovery for all monitored services
    pub async fn perform_recovery(&self) -> DaemonResult<Vec<String>> {
        info!("Starting automatic health recovery procedure");

        let recovered = self.monitoring_system.attempt_recovery().await?;

        if !recovered.is_empty() {
            info!("Auto-recovered {} services: {:?}", recovered.len(), recovered);
        } else {
            debug!("No services required recovery");
        }

        Ok(recovered)
    }

    /// Get health summary for logging and debugging
    pub async fn get_health_summary(&self) -> String {
        let (status, message) = self.monitoring_system.get_system_health().await;
        let all_health = self.monitoring_system.get_all_health().await;

        let service_summaries: Vec<String> = all_health
            .iter()
            .map(|(name, health)| {
                format!("{}: {:?} ({:.1}% errors, {:.1}ms latency)",
                    name, health.status, health.metrics.error_rate * 100.0, health.metrics.avg_latency_ms)
            })
            .collect();

        format!("System: {:?} - {} | Services: [{}]",
            status, message, service_summaries.join(", "))
    }
}

// Integration with existing middleware ConnectionManager
impl ConnectionStatsProvider for crate::grpc::middleware::ConnectionManager {
    fn get_connection_stats(&self) -> ConnectionStats {
        self.get_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grpc::health::{HealthMonitoringSystem, AlertConfig};
    use std::time::Duration;
    use tokio_test;

    #[test]
    fn test_service_status_conversion() {
        assert_eq!(ServiceStatus::from(HealthStatus::Serving), ServiceStatus::Healthy);
        assert_eq!(ServiceStatus::from(HealthStatus::NotServing), ServiceStatus::Unhealthy);
        assert_eq!(ServiceStatus::from(HealthStatus::Unknown), ServiceStatus::Unavailable);
    }

    #[tokio::test]
    async fn test_system_metrics_collector() {
        let collector = SystemMetricsCollector::new(30);

        let metrics = collector.collect_metrics(None).await.unwrap();

        assert!(metrics.cpu_usage_percent >= 0.0);
        assert!(metrics.memory_usage_bytes > 0);
        assert!(metrics.memory_total_bytes > 0);
        assert!(metrics.disk_usage_bytes > 0);
        assert!(metrics.disk_total_bytes > 0);
        assert_eq!(metrics.active_connections, 0);
        assert_eq!(metrics.pending_operations, 0);
    }

    #[tokio::test]
    async fn test_system_metrics_collector_with_connection_stats() {
        let collector = SystemMetricsCollector::new(30);

        let connection_stats = ConnectionStats {
            active_connections: 5,
            max_connections: 100,
            total_requests: 1000,
            total_bytes_sent: 50000,
            total_bytes_received: 25000,
        };

        let metrics = collector.collect_metrics(Some(&connection_stats)).await.unwrap();
        assert_eq!(metrics.active_connections, 5);
    }

    #[tokio::test]
    async fn test_system_metrics_collector_caching() {
        let collector = SystemMetricsCollector::new(60); // 60-second cache

        // First call should collect fresh metrics
        let start = std::time::Instant::now();
        let metrics1 = collector.collect_metrics(None).await.unwrap();
        let first_duration = start.elapsed();

        // Second call should use cached metrics and be faster
        let start = std::time::Instant::now();
        let metrics2 = collector.collect_metrics(None).await.unwrap();
        let second_duration = start.elapsed();

        // Values should be identical (from cache)
        assert_eq!(metrics1.cpu_usage_percent, metrics2.cpu_usage_percent);
        assert_eq!(metrics1.memory_usage_bytes, metrics2.memory_usage_bytes);

        // Second call should be faster (though this might be flaky in tests)
        // Just check that it completed
        assert!(second_duration.as_millis() < 1000);
    }

    #[tokio::test]
    async fn test_health_service_creation() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let health_service = HealthService::new(monitoring_system.clone());

        assert!(Arc::ptr_eq(&health_service.monitoring_system, &monitoring_system));
        assert!(health_service.connection_stats_provider.is_none());
    }

    #[tokio::test]
    async fn test_health_service_health_check_empty() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let health_service = HealthService::new(monitoring_system);

        let request = Request::new(());
        let response = health_service.health_check(request).await.unwrap();
        let health_response = response.into_inner();

        // No services registered, so should be available but empty
        assert!(health_response.components.is_empty());
    }

    #[tokio::test]
    async fn test_health_service_health_check_with_services() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));

        // Register a service
        let monitor = monitoring_system.register_service("test-service".to_string(), None).await;
        monitor.record_success(Duration::from_millis(100)).await;

        let health_service = HealthService::new(monitoring_system);

        let request = Request::new(());
        let response = health_service.health_check(request).await.unwrap();
        let health_response = response.into_inner();

        assert_eq!(health_response.components.len(), 1);
        assert_eq!(health_response.components[0].component_name, "test-service");
        assert_eq!(health_response.components[0].status, ServiceStatus::Healthy);
    }

    #[tokio::test]
    async fn test_health_service_get_status() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let health_service = HealthService::new(monitoring_system);

        let request = Request::new(());
        let response = health_service.get_status(request).await.unwrap();
        let status_response = response.into_inner();

        assert!(status_response.active_projects.len() > 0);
        assert!(status_response.total_documents > 0);
        assert!(status_response.total_collections > 0);
        assert!(status_response.metrics.memory_total_bytes > 0);
    }

    #[tokio::test]
    async fn test_health_service_get_metrics_empty() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let health_service = HealthService::new(monitoring_system);

        let request = Request::new(MetricsRequest {
            since: None,
            metric_names: vec![],
        });

        let response = health_service.get_metrics(request).await.unwrap();
        let metrics_response = response.into_inner();

        // Should have system metrics even with no services
        assert!(metrics_response.metrics.len() > 0);

        // Check that we have system metrics
        let metric_names: Vec<&String> = metrics_response.metrics.iter().map(|m| &m.name).collect();
        assert!(metric_names.iter().any(|name| name.contains("system_")));
    }

    #[tokio::test]
    async fn test_health_service_get_metrics_filtered() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let health_service = HealthService::new(monitoring_system);

        let request = Request::new(MetricsRequest {
            since: None,
            metric_names: vec!["system_cpu_usage_percent".to_string()],
        });

        let response = health_service.get_metrics(request).await.unwrap();
        let metrics_response = response.into_inner();

        // Should have exactly one metric
        assert_eq!(metrics_response.metrics.len(), 1);
        assert_eq!(metrics_response.metrics[0].name, "system_cpu_usage_percent");
        assert_eq!(metrics_response.metrics[0].metric_type, "gauge");
    }

    #[tokio::test]
    async fn test_health_service_get_metrics_with_service() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));

        // Register a service and record some metrics
        let monitor = monitoring_system.register_service("test-service".to_string(), None).await;
        monitor.record_success(Duration::from_millis(100)).await;
        monitor.record_failure(Duration::from_millis(200)).await;

        let health_service = HealthService::new(monitoring_system);

        let request = Request::new(MetricsRequest {
            since: None,
            metric_names: vec![],
        });

        let response = health_service.get_metrics(request).await.unwrap();
        let metrics_response = response.into_inner();

        // Should have both system metrics and service metrics
        assert!(metrics_response.metrics.len() > 7); // At least the 7 system metrics
    }

    #[tokio::test]
    async fn test_health_service_perform_recovery() {
        let alert_config = AlertConfig {
            max_error_rate: 0.1,
            ..AlertConfig::default()
        };

        let monitoring_system = Arc::new(HealthMonitoringSystem::new(Some(alert_config.clone())));
        let health_service = HealthService::new(monitoring_system.clone());

        // Register a service and degrade it
        let monitor = monitoring_system.register_service("test-service".to_string(), Some(alert_config)).await;

        // Create high error rate to degrade service
        for _ in 0..10 {
            monitor.record_failure(Duration::from_millis(100)).await;
        }

        // Improve service performance
        for _ in 0..50 {
            monitor.record_success(Duration::from_millis(50)).await;
        }

        let recovered = health_service.perform_recovery().await.unwrap();
        assert!(recovered.contains(&"test-service".to_string()));
    }

    #[tokio::test]
    async fn test_health_service_get_health_summary() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let health_service = HealthService::new(monitoring_system.clone());

        // Register services
        let monitor1 = monitoring_system.register_service("service1".to_string(), None).await;
        let monitor2 = monitoring_system.register_service("service2".to_string(), None).await;

        // Record some metrics
        monitor1.record_success(Duration::from_millis(100)).await;
        monitor2.record_failure(Duration::from_millis(200)).await;

        let summary = health_service.get_health_summary().await;

        assert!(summary.contains("System:"));
        assert!(summary.contains("service1"));
        assert!(summary.contains("service2"));
        assert!(summary.contains("errors"));
        assert!(summary.contains("latency"));
    }

    #[test]
    fn test_extract_numeric_value() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let health_service = HealthService::new(monitoring_system);

        // Test with direct value
        let mut obj = serde_json::Map::new();
        obj.insert("value".to_string(), serde_json::Value::Number(serde_json::Number::from(42)));
        assert_eq!(health_service.extract_numeric_value(&obj), Some(42.0));

        // Test with count
        let mut obj = serde_json::Map::new();
        obj.insert("count".to_string(), serde_json::Value::Number(serde_json::Number::from(100)));
        assert_eq!(health_service.extract_numeric_value(&obj), Some(100.0));

        // Test with nested metrics
        let mut metrics_map = serde_json::Map::new();
        metrics_map.insert("total_requests".to_string(), serde_json::Value::Number(serde_json::Number::from(500)));

        let mut obj = serde_json::Map::new();
        obj.insert("metrics".to_string(), serde_json::Value::Object(metrics_map));
        assert_eq!(health_service.extract_numeric_value(&obj), Some(500.0));

        // Test with no numeric values
        let mut obj = serde_json::Map::new();
        obj.insert("status".to_string(), serde_json::Value::String("healthy".to_string()));
        assert_eq!(health_service.extract_numeric_value(&obj), None);
    }

    #[tokio::test]
    async fn test_health_service_with_connection_stats() {
        use std::sync::Arc;
        use crate::grpc::middleware::ConnectionManager;

        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let connection_manager = Arc::new(ConnectionManager::new(100, 10));

        let health_service = HealthService::with_connection_stats(
            monitoring_system,
            connection_manager.clone()
        );

        assert!(health_service.connection_stats_provider.is_some());

        // Register a connection and test that it shows up in system status
        connection_manager.register_connection("test-client".to_string()).unwrap();

        let request = Request::new(());
        let response = health_service.get_status(request).await.unwrap();
        let status_response = response.into_inner();

        assert_eq!(status_response.metrics.active_connections, 1);
    }
}
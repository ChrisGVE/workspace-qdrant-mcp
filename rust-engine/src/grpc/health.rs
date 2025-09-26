//! Health checking and monitoring system for gRPC services
//!
//! This module implements the gRPC health check protocol and provides comprehensive
//! service-level health indicators, metrics collection, and alerting capabilities.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};
use serde::{Serialize, Deserialize};

use crate::error::DaemonResult;

/// Standard gRPC health check status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Service is healthy and operating normally
    Serving,
    /// Service is not serving requests
    NotServing,
    /// Health status is unknown
    Unknown,
}

impl From<HealthStatus> for i32 {
    fn from(status: HealthStatus) -> Self {
        match status {
            HealthStatus::Serving => 1,
            HealthStatus::NotServing => 2,
            HealthStatus::Unknown => 0,
        }
    }
}

impl From<i32> for HealthStatus {
    fn from(value: i32) -> Self {
        match value {
            1 => HealthStatus::Serving,
            2 => HealthStatus::NotServing,
            _ => HealthStatus::Unknown,
        }
    }
}

/// Service-level health indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceHealth {
    /// Service name
    pub service_name: String,
    /// Current health status
    pub status: HealthStatus,
    /// Human-readable status message
    pub message: String,
    /// Last time the health was checked
    pub last_check: SystemTime,
    /// Service uptime since last start
    pub uptime: Duration,
    /// Service-specific metrics
    pub metrics: ServiceMetrics,
}

/// Service-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMetrics {
    /// Total number of requests processed
    pub total_requests: u64,
    /// Number of successful requests
    pub successful_requests: u64,
    /// Number of failed requests
    pub failed_requests: u64,
    /// Average request latency in milliseconds
    pub avg_latency_ms: f64,
    /// 95th percentile latency in milliseconds
    pub p95_latency_ms: f64,
    /// 99th percentile latency in milliseconds
    pub p99_latency_ms: f64,
    /// Current throughput (requests per second)
    pub throughput_rps: f64,
    /// Error rate (0.0 - 1.0)
    pub error_rate: f64,
}

impl Default for ServiceMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            throughput_rps: 0.0,
            error_rate: 0.0,
        }
    }
}

/// Latency measurement for calculating percentiles
#[derive(Debug, Clone)]
struct LatencyMeasurement {
    duration_ms: f64,
    timestamp: Instant,
}

/// Service health monitor that tracks health status and metrics
#[derive(Debug)]
pub struct ServiceHealthMonitor {
    /// Service name
    service_name: String,
    /// Current health status
    status: Arc<RwLock<HealthStatus>>,
    /// Service start time
    start_time: Instant,
    /// Request metrics
    metrics: Arc<RwLock<ServiceMetrics>>,
    /// Latency measurements for percentile calculation
    latency_measurements: Arc<RwLock<Vec<LatencyMeasurement>>>,
    /// Alert thresholds
    alert_config: AlertConfig,
}

/// Alert configuration thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Maximum acceptable error rate (0.0 - 1.0)
    pub max_error_rate: f64,
    /// Maximum acceptable average latency in milliseconds
    pub max_avg_latency_ms: f64,
    /// Maximum acceptable P99 latency in milliseconds
    pub max_p99_latency_ms: f64,
    /// Minimum acceptable throughput in requests per second
    pub min_throughput_rps: f64,
    /// Time window for metrics evaluation in seconds
    pub evaluation_window_secs: u64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            max_error_rate: 0.05, // 5%
            max_avg_latency_ms: 1000.0, // 1 second
            max_p99_latency_ms: 5000.0, // 5 seconds
            min_throughput_rps: 1.0,
            evaluation_window_secs: 300, // 5 minutes
        }
    }
}

impl ServiceHealthMonitor {
    /// Create a new service health monitor
    pub fn new(service_name: String, alert_config: Option<AlertConfig>) -> Self {
        Self {
            service_name,
            status: Arc::new(RwLock::new(HealthStatus::Serving)),
            start_time: Instant::now(),
            metrics: Arc::new(RwLock::new(ServiceMetrics::default())),
            latency_measurements: Arc::new(RwLock::new(Vec::new())),
            alert_config: alert_config.unwrap_or_default(),
        }
    }

    /// Get current service health
    pub async fn get_health(&self) -> ServiceHealth {
        let status = *self.status.read().await;
        let metrics = self.metrics.read().await.clone();
        let uptime = self.start_time.elapsed();

        ServiceHealth {
            service_name: self.service_name.clone(),
            status,
            message: self.get_status_message(status, &metrics).await,
            last_check: SystemTime::now(),
            uptime,
            metrics,
        }
    }

    /// Record a successful request with latency
    pub async fn record_success(&self, latency: Duration) {
        let latency_ms = latency.as_secs_f64() * 1000.0;
        let mut metrics = self.metrics.write().await;
        let mut latency_measurements = self.latency_measurements.write().await;

        // Update counters
        metrics.total_requests += 1;
        metrics.successful_requests += 1;

        // Add latency measurement
        latency_measurements.push(LatencyMeasurement {
            duration_ms: latency_ms,
            timestamp: Instant::now(),
        });

        // Cleanup old measurements (keep only last hour)
        let cutoff = Instant::now() - Duration::from_secs(3600);
        latency_measurements.retain(|m| m.timestamp > cutoff);

        // Update derived metrics
        self.update_derived_metrics(&mut metrics, &latency_measurements).await;

        // Check for alerts
        self.check_alerts(&metrics).await;

        debug!("Recorded successful request for {}: {:.2}ms", self.service_name, latency_ms);
    }

    /// Record a failed request with latency
    pub async fn record_failure(&self, latency: Duration) {
        let latency_ms = latency.as_secs_f64() * 1000.0;
        let mut metrics = self.metrics.write().await;
        let mut latency_measurements = self.latency_measurements.write().await;

        // Update counters
        metrics.total_requests += 1;
        metrics.failed_requests += 1;

        // Add latency measurement (even for failures)
        latency_measurements.push(LatencyMeasurement {
            duration_ms: latency_ms,
            timestamp: Instant::now(),
        });

        // Cleanup old measurements
        let cutoff = Instant::now() - Duration::from_secs(3600);
        latency_measurements.retain(|m| m.timestamp > cutoff);

        // Update derived metrics
        self.update_derived_metrics(&mut metrics, &latency_measurements).await;

        // Check for alerts
        self.check_alerts(&metrics).await;

        warn!("Recorded failed request for {}: {:.2}ms", self.service_name, latency_ms);
    }

    /// Manually set service health status
    pub async fn set_status(&self, status: HealthStatus, message: Option<String>) {
        let mut current_status = self.status.write().await;
        *current_status = status;

        let msg = message.unwrap_or_else(|| format!("Service {} status changed to {:?}", self.service_name, status));

        match status {
            HealthStatus::Serving => info!("{}", msg),
            HealthStatus::NotServing => error!("{}", msg),
            HealthStatus::Unknown => warn!("{}", msg),
        }
    }

    /// Update derived metrics (latency percentiles, throughput, error rate)
    async fn update_derived_metrics(&self, metrics: &mut ServiceMetrics, measurements: &[LatencyMeasurement]) {
        if measurements.is_empty() {
            return;
        }

        // Calculate latency percentiles
        let mut latencies: Vec<f64> = measurements.iter().map(|m| m.duration_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        metrics.avg_latency_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;

        if !latencies.is_empty() {
            let p95_idx = (latencies.len() as f64 * 0.95) as usize;
            let p99_idx = (latencies.len() as f64 * 0.99) as usize;

            metrics.p95_latency_ms = latencies.get(p95_idx.min(latencies.len() - 1)).copied().unwrap_or(0.0);
            metrics.p99_latency_ms = latencies.get(p99_idx.min(latencies.len() - 1)).copied().unwrap_or(0.0);
        }

        // Calculate error rate
        metrics.error_rate = if metrics.total_requests > 0 {
            metrics.failed_requests as f64 / metrics.total_requests as f64
        } else {
            0.0
        };

        // Calculate throughput (requests per second over evaluation window)
        let window_duration = Duration::from_secs(self.alert_config.evaluation_window_secs);
        let cutoff = Instant::now() - window_duration;
        let recent_requests = measurements.iter().filter(|m| m.timestamp > cutoff).count();
        metrics.throughput_rps = recent_requests as f64 / self.alert_config.evaluation_window_secs as f64;
    }

    /// Check alert thresholds and update service status
    async fn check_alerts(&self, metrics: &ServiceMetrics) {
        let mut alerts = Vec::new();
        let mut should_degrade = false;

        // Check error rate
        if metrics.error_rate > self.alert_config.max_error_rate {
            alerts.push(format!("Error rate {:.2}% exceeds threshold {:.2}%",
                metrics.error_rate * 100.0, self.alert_config.max_error_rate * 100.0));
            should_degrade = true;
        }

        // Check average latency
        if metrics.avg_latency_ms > self.alert_config.max_avg_latency_ms {
            alerts.push(format!("Average latency {:.2}ms exceeds threshold {:.2}ms",
                metrics.avg_latency_ms, self.alert_config.max_avg_latency_ms));
            should_degrade = true;
        }

        // Check P99 latency
        if metrics.p99_latency_ms > self.alert_config.max_p99_latency_ms {
            alerts.push(format!("P99 latency {:.2}ms exceeds threshold {:.2}ms",
                metrics.p99_latency_ms, self.alert_config.max_p99_latency_ms));
            should_degrade = true;
        }

        // Check throughput (only if we have significant request volume)
        if metrics.total_requests > 10 && metrics.throughput_rps < self.alert_config.min_throughput_rps {
            alerts.push(format!("Throughput {:.2} RPS below threshold {:.2} RPS",
                metrics.throughput_rps, self.alert_config.min_throughput_rps));
            should_degrade = true;
        }

        // Update service status based on alerts
        let current_status = *self.status.read().await;
        if should_degrade && current_status == HealthStatus::Serving {
            let alert_message = format!("Service degraded: {}", alerts.join(", "));
            self.set_status(HealthStatus::NotServing, Some(alert_message)).await;
        } else if !should_degrade && current_status == HealthStatus::NotServing {
            self.set_status(HealthStatus::Serving, Some("Service recovered from degraded state".to_string())).await;
        }

        // Log alerts
        for alert in alerts {
            warn!("ALERT [{}]: {}", self.service_name, alert);
        }
    }

    /// Get human-readable status message
    async fn get_status_message(&self, status: HealthStatus, metrics: &ServiceMetrics) -> String {
        match status {
            HealthStatus::Serving => {
                format!("Service healthy - {:.2}% error rate, {:.2}ms avg latency, {:.2} RPS",
                    metrics.error_rate * 100.0, metrics.avg_latency_ms, metrics.throughput_rps)
            }
            HealthStatus::NotServing => {
                "Service degraded - check metrics for details".to_string()
            }
            HealthStatus::Unknown => {
                "Service health status unknown".to_string()
            }
        }
    }
}

/// Comprehensive health monitoring system for all gRPC services
#[derive(Debug)]
pub struct HealthMonitoringSystem {
    /// Service health monitors by service name
    monitors: Arc<RwLock<HashMap<String, Arc<ServiceHealthMonitor>>>>,
    /// System-wide alert configuration
    default_alert_config: AlertConfig,
    /// External monitoring integration
    external_monitoring: Option<Arc<ExternalMonitoring>>,
}

impl HealthMonitoringSystem {
    /// Create a new health monitoring system
    pub fn new(default_alert_config: Option<AlertConfig>) -> Self {
        Self {
            monitors: Arc::new(RwLock::new(HashMap::new())),
            default_alert_config: default_alert_config.unwrap_or_default(),
            external_monitoring: None,
        }
    }

    /// Create a new health monitoring system with external monitoring
    pub fn with_external_monitoring(
        default_alert_config: Option<AlertConfig>,
        external_monitoring: Arc<ExternalMonitoring>,
    ) -> Self {
        Self {
            monitors: Arc::new(RwLock::new(HashMap::new())),
            default_alert_config: default_alert_config.unwrap_or_default(),
            external_monitoring: Some(external_monitoring),
        }
    }

    /// Register a service for health monitoring
    pub async fn register_service(&self, service_name: String, alert_config: Option<AlertConfig>) -> Arc<ServiceHealthMonitor> {
        let mut monitors = self.monitors.write().await;

        let config = alert_config.unwrap_or_else(|| self.default_alert_config.clone());
        let monitor = Arc::new(ServiceHealthMonitor::new(service_name.clone(), Some(config)));

        monitors.insert(service_name.clone(), monitor.clone());
        info!("Registered health monitor for service: {}", service_name);

        monitor
    }

    /// Get health monitor for a service
    pub async fn get_monitor(&self, service_name: &str) -> Option<Arc<ServiceHealthMonitor>> {
        let monitors = self.monitors.read().await;
        monitors.get(service_name).cloned()
    }

    /// Get health status for all services
    pub async fn get_all_health(&self) -> HashMap<String, ServiceHealth> {
        let monitors = self.monitors.read().await;
        let mut health_map = HashMap::new();

        for (service_name, monitor) in monitors.iter() {
            let health = monitor.get_health().await;
            health_map.insert(service_name.clone(), health);
        }

        health_map
    }

    /// Get overall system health status
    pub async fn get_system_health(&self) -> (HealthStatus, String) {
        let all_health = self.get_all_health().await;

        if all_health.is_empty() {
            return (HealthStatus::Unknown, "No services registered".to_string());
        }

        let mut serving_count = 0;
        let mut total_count = all_health.len();
        let mut degraded_services = Vec::new();

        for (service_name, health) in all_health.iter() {
            match health.status {
                HealthStatus::Serving => serving_count += 1,
                HealthStatus::NotServing => degraded_services.push(service_name.clone()),
                HealthStatus::Unknown => {
                    // Unknown status counts as degraded
                    degraded_services.push(service_name.clone());
                }
            }
        }

        let status = if serving_count == total_count {
            HealthStatus::Serving
        } else if serving_count > 0 {
            // Some services healthy, system partially degraded
            HealthStatus::NotServing
        } else {
            // No services healthy
            HealthStatus::NotServing
        };

        let message = if degraded_services.is_empty() {
            format!("All {} services healthy", total_count)
        } else {
            format!("{}/{} services healthy. Degraded: {}",
                serving_count, total_count, degraded_services.join(", "))
        };

        (status, message)
    }

    /// Export metrics for external monitoring systems
    pub async fn export_metrics(&self) -> DaemonResult<HashMap<String, serde_json::Value>> {
        let all_health = self.get_all_health().await;
        let mut metrics = HashMap::new();

        // Overall system metrics
        let (system_status, system_message) = self.get_system_health().await;
        metrics.insert("system_health_status".to_string(),
            serde_json::json!({
                "status": system_status,
                "message": system_message,
                "timestamp": SystemTime::now()
            }));

        // Individual service metrics
        for (service_name, health) in all_health.iter() {
            let service_metrics = serde_json::to_value(health)?;
            metrics.insert(format!("service_{}", service_name), service_metrics);
        }

        // Send to external monitoring if configured
        if let Some(external) = &self.external_monitoring {
            if let Err(e) = external.send_metrics(&metrics).await {
                warn!("Failed to send metrics to external monitoring: {}", e);
            }
        }

        Ok(metrics)
    }

    /// Perform automatic health recovery for degraded services
    pub async fn attempt_recovery(&self) -> DaemonResult<Vec<String>> {
        let monitors = self.monitors.read().await;
        let mut recovered_services = Vec::new();

        for (service_name, monitor) in monitors.iter() {
            let health = monitor.get_health().await;

            if health.status == HealthStatus::NotServing {
                // Attempt basic recovery by resetting metrics if error rate has improved
                if health.metrics.error_rate < monitor.alert_config.max_error_rate * 0.5 {
                    monitor.set_status(HealthStatus::Serving,
                        Some(format!("Auto-recovered {} - error rate improved", service_name))).await;
                    recovered_services.push(service_name.clone());
                    info!("Auto-recovered service: {}", service_name);
                }
            }
        }

        Ok(recovered_services)
    }
}

/// External monitoring system integration
#[derive(Debug)]
pub struct ExternalMonitoring {
    /// Monitoring system type (e.g., "prometheus", "datadog", "custom")
    pub system_type: String,
    /// Endpoint URL for metrics submission
    pub endpoint: String,
    /// Authentication headers
    pub auth_headers: HashMap<String, String>,
}

impl ExternalMonitoring {
    /// Create a new external monitoring configuration
    pub fn new(system_type: String, endpoint: String) -> Self {
        Self {
            system_type,
            endpoint,
            auth_headers: HashMap::new(),
        }
    }

    /// Add authentication header
    pub fn with_auth_header(mut self, key: String, value: String) -> Self {
        self.auth_headers.insert(key, value);
        self
    }

    /// Send metrics to external monitoring system
    pub async fn send_metrics(&self, metrics: &HashMap<String, serde_json::Value>) -> DaemonResult<()> {
        // Implementation would depend on the specific monitoring system
        // For now, just log the metrics export
        debug!("Exporting {} metrics to {} at {}",
            metrics.len(), self.system_type, self.endpoint);

        // In a real implementation, this would make HTTP requests to the monitoring system
        // with the appropriate format (Prometheus, StatsD, etc.)

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[test]
    fn test_health_status_conversions() {
        assert_eq!(HealthStatus::from(1), HealthStatus::Serving);
        assert_eq!(HealthStatus::from(2), HealthStatus::NotServing);
        assert_eq!(HealthStatus::from(0), HealthStatus::Unknown);
        assert_eq!(HealthStatus::from(99), HealthStatus::Unknown);

        assert_eq!(i32::from(HealthStatus::Serving), 1);
        assert_eq!(i32::from(HealthStatus::NotServing), 2);
        assert_eq!(i32::from(HealthStatus::Unknown), 0);
    }

    #[test]
    fn test_service_metrics_default() {
        let metrics = ServiceMetrics::default();
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.successful_requests, 0);
        assert_eq!(metrics.failed_requests, 0);
        assert_eq!(metrics.avg_latency_ms, 0.0);
        assert_eq!(metrics.error_rate, 0.0);
    }

    #[test]
    fn test_alert_config_default() {
        let config = AlertConfig::default();
        assert_eq!(config.max_error_rate, 0.05);
        assert_eq!(config.max_avg_latency_ms, 1000.0);
        assert_eq!(config.max_p99_latency_ms, 5000.0);
        assert_eq!(config.min_throughput_rps, 1.0);
        assert_eq!(config.evaluation_window_secs, 300);
    }

    #[tokio::test]
    async fn test_service_health_monitor_creation() {
        let monitor = ServiceHealthMonitor::new("test-service".to_string(), None);
        assert_eq!(monitor.service_name, "test-service");

        let health = monitor.get_health().await;
        assert_eq!(health.service_name, "test-service");
        assert_eq!(health.status, HealthStatus::Serving);
        assert_eq!(health.metrics.total_requests, 0);
    }

    #[tokio::test]
    async fn test_service_health_monitor_record_success() {
        let monitor = ServiceHealthMonitor::new("test-service".to_string(), None);

        let latency = Duration::from_millis(100);
        monitor.record_success(latency).await;

        let health = monitor.get_health().await;
        assert_eq!(health.metrics.total_requests, 1);
        assert_eq!(health.metrics.successful_requests, 1);
        assert_eq!(health.metrics.failed_requests, 0);
        assert!(health.metrics.avg_latency_ms > 99.0 && health.metrics.avg_latency_ms < 101.0);
    }

    #[tokio::test]
    async fn test_service_health_monitor_record_failure() {
        let monitor = ServiceHealthMonitor::new("test-service".to_string(), None);

        let latency = Duration::from_millis(200);
        monitor.record_failure(latency).await;

        let health = monitor.get_health().await;
        assert_eq!(health.metrics.total_requests, 1);
        assert_eq!(health.metrics.successful_requests, 0);
        assert_eq!(health.metrics.failed_requests, 1);
        assert_eq!(health.metrics.error_rate, 1.0);
        assert!(health.metrics.avg_latency_ms > 199.0 && health.metrics.avg_latency_ms < 201.0);
    }

    #[tokio::test]
    async fn test_service_health_monitor_error_rate_alert() {
        let alert_config = AlertConfig {
            max_error_rate: 0.1, // 10%
            ..AlertConfig::default()
        };

        let monitor = ServiceHealthMonitor::new("test-service".to_string(), Some(alert_config));

        // Record 9 successes and 2 failures (18% error rate)
        for _ in 0..9 {
            monitor.record_success(Duration::from_millis(100)).await;
        }
        for _ in 0..2 {
            monitor.record_failure(Duration::from_millis(100)).await;
        }

        let health = monitor.get_health().await;
        assert_eq!(health.status, HealthStatus::NotServing); // Should be degraded
        assert!(health.metrics.error_rate > 0.1);
    }

    #[tokio::test]
    async fn test_service_health_monitor_latency_alert() {
        let alert_config = AlertConfig {
            max_avg_latency_ms: 500.0,
            ..AlertConfig::default()
        };

        let monitor = ServiceHealthMonitor::new("test-service".to_string(), Some(alert_config));

        // Record high latency requests
        for _ in 0..5 {
            monitor.record_success(Duration::from_millis(1000)).await; // 1 second
        }

        let health = monitor.get_health().await;
        assert_eq!(health.status, HealthStatus::NotServing); // Should be degraded
        assert!(health.metrics.avg_latency_ms > 500.0);
    }

    #[tokio::test]
    async fn test_service_health_monitor_manual_status() {
        let monitor = ServiceHealthMonitor::new("test-service".to_string(), None);

        // Manually set status to not serving
        monitor.set_status(HealthStatus::NotServing, Some("Manual override".to_string())).await;

        let health = monitor.get_health().await;
        assert_eq!(health.status, HealthStatus::NotServing);
    }

    #[tokio::test]
    async fn test_health_monitoring_system_creation() {
        let system = HealthMonitoringSystem::new(None);

        let all_health = system.get_all_health().await;
        assert!(all_health.is_empty());

        let (status, message) = system.get_system_health().await;
        assert_eq!(status, HealthStatus::Unknown);
        assert!(message.contains("No services registered"));
    }

    #[tokio::test]
    async fn test_health_monitoring_system_register_service() {
        let system = HealthMonitoringSystem::new(None);

        let monitor = system.register_service("test-service".to_string(), None).await;
        assert_eq!(monitor.service_name, "test-service");

        let retrieved_monitor = system.get_monitor("test-service").await;
        assert!(retrieved_monitor.is_some());

        let all_health = system.get_all_health().await;
        assert_eq!(all_health.len(), 1);
    }

    #[tokio::test]
    async fn test_health_monitoring_system_overall_health() {
        let system = HealthMonitoringSystem::new(None);

        // Register multiple services
        let monitor1 = system.register_service("service1".to_string(), None).await;
        let monitor2 = system.register_service("service2".to_string(), None).await;

        // All services healthy
        let (status, _) = system.get_system_health().await;
        assert_eq!(status, HealthStatus::Serving);

        // Degrade one service
        monitor1.set_status(HealthStatus::NotServing, None).await;
        let (status, message) = system.get_system_health().await;
        assert_eq!(status, HealthStatus::NotServing);
        assert!(message.contains("1/2 services healthy"));
    }

    #[tokio::test]
    async fn test_health_monitoring_system_export_metrics() {
        let system = HealthMonitoringSystem::new(None);

        let monitor = system.register_service("test-service".to_string(), None).await;
        monitor.record_success(Duration::from_millis(100)).await;

        let metrics = system.export_metrics().await.unwrap();
        assert!(metrics.contains_key("system_health_status"));
        assert!(metrics.contains_key("service_test-service"));
    }

    #[tokio::test]
    async fn test_health_monitoring_system_recovery() {
        let alert_config = AlertConfig {
            max_error_rate: 0.1,
            ..AlertConfig::default()
        };

        let system = HealthMonitoringSystem::new(Some(alert_config.clone()));
        let monitor = system.register_service("test-service".to_string(), Some(alert_config)).await;

        // Degrade service with high error rate
        for _ in 0..5 {
            monitor.record_failure(Duration::from_millis(100)).await;
        }

        let health = monitor.get_health().await;
        assert_eq!(health.status, HealthStatus::NotServing);

        // Improve service performance (low error rate)
        for _ in 0..20 {
            monitor.record_success(Duration::from_millis(50)).await;
        }

        let recovered = system.attempt_recovery().await.unwrap();
        assert!(recovered.contains(&"test-service".to_string()));

        let health = monitor.get_health().await;
        assert_eq!(health.status, HealthStatus::Serving);
    }

    #[test]
    fn test_external_monitoring_creation() {
        let monitoring = ExternalMonitoring::new(
            "prometheus".to_string(),
            "http://localhost:9090/metrics".to_string()
        );

        assert_eq!(monitoring.system_type, "prometheus");
        assert_eq!(monitoring.endpoint, "http://localhost:9090/metrics");
        assert!(monitoring.auth_headers.is_empty());

        let monitoring_with_auth = monitoring.with_auth_header(
            "Authorization".to_string(),
            "Bearer token".to_string()
        );

        assert_eq!(monitoring_with_auth.auth_headers.get("Authorization"),
                   Some(&"Bearer token".to_string()));
    }

    #[tokio::test]
    async fn test_external_monitoring_send_metrics() {
        let monitoring = ExternalMonitoring::new(
            "test".to_string(),
            "http://test.example.com".to_string()
        );

        let mut metrics = HashMap::new();
        metrics.insert("test_metric".to_string(), serde_json::json!({"value": 42}));

        // Should not error (just logs for now)
        let result = monitoring.send_metrics(&metrics).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_latency_percentiles() {
        let monitor = ServiceHealthMonitor::new("test-service".to_string(), None);

        // Record requests with various latencies
        let latencies = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100]; // milliseconds

        for latency_ms in latencies {
            monitor.record_success(Duration::from_millis(latency_ms)).await;
        }

        let health = monitor.get_health().await;
        assert!(health.metrics.avg_latency_ms > 54.0 && health.metrics.avg_latency_ms < 56.0);
        assert!(health.metrics.p95_latency_ms >= 95.0); // Should be near the 95th percentile
        assert!(health.metrics.p99_latency_ms >= 99.0); // Should be near the 99th percentile
    }

    #[tokio::test]
    async fn test_throughput_calculation() {
        let alert_config = AlertConfig {
            evaluation_window_secs: 1, // 1 second window for testing
            ..AlertConfig::default()
        };

        let monitor = ServiceHealthMonitor::new("test-service".to_string(), Some(alert_config));

        // Record 10 requests quickly
        for _ in 0..10 {
            monitor.record_success(Duration::from_millis(10)).await;
        }

        let health = monitor.get_health().await;
        assert!(health.metrics.throughput_rps >= 9.0); // Should be around 10 RPS
    }

    #[tokio::test]
    async fn test_health_status_message() {
        let monitor = ServiceHealthMonitor::new("test-service".to_string(), None);

        // Record some successful requests
        for _ in 0..5 {
            monitor.record_success(Duration::from_millis(100)).await;
        }

        let health = monitor.get_health().await;
        assert!(health.message.contains("Service healthy"));
        assert!(health.message.contains("0.00% error rate"));
        assert!(health.message.contains("100.00ms avg latency"));
    }

    #[tokio::test]
    async fn test_concurrent_metric_updates() {
        let monitor = Arc::new(ServiceHealthMonitor::new("test-service".to_string(), None));
        let mut handles = vec![];

        // Spawn multiple tasks to record metrics concurrently
        for i in 0..10 {
            let monitor_clone = Arc::clone(&monitor);
            let handle = tokio::spawn(async move {
                for _ in 0..10 {
                    if i % 2 == 0 {
                        monitor_clone.record_success(Duration::from_millis(100)).await;
                    } else {
                        monitor_clone.record_failure(Duration::from_millis(200)).await;
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let health = monitor.get_health().await;
        assert_eq!(health.metrics.total_requests, 100);
        assert_eq!(health.metrics.successful_requests, 50);
        assert_eq!(health.metrics.failed_requests, 50);
        assert_eq!(health.metrics.error_rate, 0.5);
    }
}
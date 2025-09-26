//! Health monitoring system integration and examples
//!
//! This module provides a complete integration example showing how to set up
//! the comprehensive health monitoring, alerting, and recovery systems.

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, error};

use crate::error::DaemonResult;
use crate::grpc::{
    health::{HealthMonitoringSystem, AlertConfig, ExternalMonitoring},
    health_service::HealthService,
    service_monitors::ServiceMonitors,
    alerting::{AlertManager, RecoverySystem, LogAlertChannel, WebhookAlertChannel, EmailAlertChannel, EmailConfig, RestartRecoveryProcedure, CircuitBreakerResetProcedure},
    middleware::ConnectionManager,
};

/// Complete health monitoring integration
#[derive(Debug)]
pub struct HealthIntegration {
    /// Core health monitoring system
    pub health_monitoring: Arc<HealthMonitoringSystem>,
    /// gRPC health service
    pub health_service: Arc<HealthService>,
    /// Service-specific monitors
    pub service_monitors: Arc<ServiceMonitors>,
    /// Alert management system
    pub alert_manager: Arc<AlertManager>,
    /// Automatic recovery system
    pub recovery_system: Arc<RecoverySystem>,
    /// Health check scheduler
    health_check_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl HealthIntegration {
    /// Create a complete health monitoring integration
    pub async fn new(connection_manager: Option<Arc<ConnectionManager>>) -> DaemonResult<Self> {
        info!("Initializing comprehensive health monitoring system");

        // Step 1: Create external monitoring (optional)
        let external_monitoring = Self::create_external_monitoring()?;

        // Step 2: Set up service monitors
        let service_monitors = if let Some(external) = external_monitoring {
            Arc::new(ServiceMonitors::with_external_monitoring(external).await?)
        } else {
            Arc::new(ServiceMonitors::new().await?)
        };

        let health_monitoring = service_monitors.monitoring_system.clone();

        // Step 3: Create health service
        let health_service = if let Some(conn_mgr) = connection_manager {
            Arc::new(HealthService::with_connection_stats(
                health_monitoring.clone(),
                conn_mgr,
            ))
        } else {
            Arc::new(HealthService::new(health_monitoring.clone()))
        };

        // Step 4: Set up alert manager
        let alert_manager = Arc::new(AlertManager::new());
        Self::configure_alert_channels(&alert_manager).await?;

        // Step 5: Create recovery system
        let recovery_system = Arc::new(RecoverySystem::new(
            health_monitoring.clone(),
            alert_manager.clone(),
        ));
        Self::configure_recovery_procedures(&recovery_system).await?;

        let integration = Self {
            health_monitoring,
            health_service,
            service_monitors,
            alert_manager,
            recovery_system,
            health_check_handle: Arc::new(RwLock::new(None)),
        };

        info!("Health monitoring system initialized successfully");
        Ok(integration)
    }

    /// Create external monitoring configuration
    fn create_external_monitoring() -> DaemonResult<Option<Arc<ExternalMonitoring>>> {
        // In a real implementation, this would read from configuration
        // For now, create a simple webhook monitoring setup

        let monitoring = ExternalMonitoring::new(
            "webhook".to_string(),
            "http://localhost:9090/metrics".to_string(),
        ).with_auth_header(
            "Authorization".to_string(),
            "Bearer monitoring-token".to_string(),
        );

        Ok(Some(Arc::new(monitoring)))
    }

    /// Configure alert notification channels
    async fn configure_alert_channels(alert_manager: &Arc<AlertManager>) -> DaemonResult<()> {
        info!("Configuring alert notification channels");

        // Add log channel (always present)
        let log_channel = Arc::new(LogAlertChannel::new());
        alert_manager.add_channel(log_channel).await;

        // Add webhook channel for external alerting
        let webhook_channel = Arc::new(WebhookAlertChannel::new(
            "primary-webhook".to_string(),
            "http://alerts.example.com/webhook".to_string(),
        ));
        alert_manager.add_channel(webhook_channel).await;

        // Add email channel for critical alerts
        let email_config = EmailConfig {
            smtp_server: "smtp.example.com".to_string(),
            smtp_port: 587,
            username: "alerts@example.com".to_string(),
            password: "smtp-password".to_string(),
            from_email: "workspace-qdrant@example.com".to_string(),
            to_emails: vec![
                "ops-team@example.com".to_string(),
                "on-call@example.com".to_string(),
            ],
        };

        let email_channel = Arc::new(EmailAlertChannel::new(
            "ops-email".to_string(),
            email_config,
        ));
        alert_manager.add_channel(email_channel).await;

        info!("Alert channels configured: log, webhook, email");
        Ok(())
    }

    /// Configure automatic recovery procedures
    async fn configure_recovery_procedures(recovery_system: &Arc<RecoverySystem>) -> DaemonResult<()> {
        info!("Configuring automatic recovery procedures");

        // Add restart recovery for document processor
        recovery_system.add_recovery_procedure(
            "document-processor".to_string(),
            Arc::new(RestartRecoveryProcedure::new()),
        ).await;

        // Add circuit breaker reset for search service
        recovery_system.add_recovery_procedure(
            "search-service".to_string(),
            Arc::new(CircuitBreakerResetProcedure::new()),
        ).await;

        // Add restart recovery for memory service
        recovery_system.add_recovery_procedure(
            "memory-service".to_string(),
            Arc::new(RestartRecoveryProcedure::new()),
        ).await;

        info!("Recovery procedures configured for all services");
        Ok(())
    }

    /// Start the health monitoring system
    pub async fn start(&self) -> DaemonResult<()> {
        info!("Starting health monitoring system");

        // Start periodic health checks
        let health_check_interval = Duration::from_secs(30); // 30 seconds
        let recovery_check_interval = Duration::from_secs(60); // 1 minute

        let service_monitors = self.service_monitors.clone();
        let recovery_system = self.recovery_system.clone();

        let handle = tokio::spawn(async move {
            let mut health_check_ticker = tokio::time::interval(health_check_interval);
            let mut recovery_check_ticker = tokio::time::interval(recovery_check_interval);

            loop {
                tokio::select! {
                    _ = health_check_ticker.tick() => {
                        if let Err(e) = service_monitors.perform_health_check().await {
                            error!("Health check failed: {}", e);
                        }
                    }
                    _ = recovery_check_ticker.tick() => {
                        if let Err(e) = recovery_system.perform_recovery_check().await {
                            error!("Recovery check failed: {}", e);
                        }
                    }
                }
            }
        });

        let mut health_check_handle = self.health_check_handle.write().await;
        *health_check_handle = Some(handle);

        info!("Health monitoring system started");
        Ok(())
    }

    /// Stop the health monitoring system
    pub async fn stop(&self) -> DaemonResult<()> {
        info!("Stopping health monitoring system");

        let mut health_check_handle = self.health_check_handle.write().await;
        if let Some(handle) = health_check_handle.take() {
            handle.abort();
        }

        info!("Health monitoring system stopped");
        Ok(())
    }

    /// Get comprehensive health status
    pub async fn get_comprehensive_status(&self) -> DaemonResult<HealthSystemStatus> {
        let service_stats = self.service_monitors.get_all_stats().await;
        let alert_stats = self.alert_manager.get_alert_stats().await;
        let recovery_history = self.recovery_system.get_recovery_history(Some(10)).await;
        let health_summary = self.health_service.get_health_summary().await;

        Ok(HealthSystemStatus {
            summary: health_summary,
            service_stats,
            alert_stats,
            recent_recoveries: recovery_history,
        })
    }

    /// Demonstrate a complete health scenario
    pub async fn demonstrate_health_scenario(&self) -> DaemonResult<()> {
        info!("=== Health Monitoring Demonstration ===");

        // Simulate document processing load
        info!("Simulating document processing workload...");
        for i in 0..10 {
            self.service_monitors.document_processor.start_processing(&format!("doc_{}", i)).await;

            // Complete processing with varying latencies
            let latency = Duration::from_millis(100 + (i as u64 * 50));
            self.service_monitors.document_processor.complete_processing(&format!("doc_{}", i), latency).await;
        }

        // Simulate search activity
        info!("Simulating search workload...");
        for i in 0..20 {
            let tracker = self.service_monitors.search_service.start_search(&format!("query_{}", i)).await;

            // Simulate cache hits/misses
            if i % 3 == 0 {
                self.service_monitors.search_service.record_cache_hit().await;
            } else {
                self.service_monitors.search_service.record_cache_miss().await;
            }

            tracker.complete_success(5 + i).await;
        }

        // Simulate memory operations
        info!("Simulating memory operations...");
        for i in 0..15 {
            let tracker = self.service_monitors.memory_service.start_operation(&format!("operation_{}", i)).await;
            tracker.complete_success().await;
        }

        // Update system metrics
        self.service_monitors.memory_service.update_memory_utilization(75.5).await;
        self.service_monitors.memory_service.update_collection_count(8).await;
        self.service_monitors.memory_service.update_document_count(1542).await;

        // Show current status
        let status = self.get_comprehensive_status().await?;
        info!("=== Health Status After Demonstration ===");
        info!("System Summary: {}", status.summary);
        info!("Active Collections: {}", status.service_stats.memory.active_collections);
        info!("Total Documents: {}", status.service_stats.memory.total_documents);
        info!("Cache Hit Rate: {:.2}%", status.service_stats.search.cache_hit_rate * 100.0);

        // Simulate a service degradation
        info!("Simulating service degradation...");

        // Cause high error rate
        for _ in 0..10 {
            let tracker = self.service_monitors.search_service.start_search("failing_query").await;
            tracker.complete_failure("Simulated failure").await;
        }

        // Wait for alerts to be processed
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check alerts
        let active_alerts = self.alert_manager.get_active_alerts().await;
        info!("Active alerts after degradation: {}", active_alerts.len());

        for alert in &active_alerts {
            info!("Alert: {} - {} - {}", alert.service_name, alert.severity, alert.message);
        }

        // Simulate recovery
        info!("Simulating service recovery...");
        for _ in 0..50 {
            let tracker = self.service_monitors.search_service.start_search("recovered_query").await;
            tracker.complete_success(10).await;
        }

        // Manual recovery check
        let recovery_attempts = self.recovery_system.perform_recovery_check().await?;
        info!("Recovery attempts: {}", recovery_attempts.len());

        for attempt in &recovery_attempts {
            info!("Recovery: {} - {} - Success: {}",
                attempt.service_name, attempt.procedure_name, attempt.result.success);
        }

        info!("=== Demonstration Complete ===");
        Ok(())
    }

    /// Export all metrics for external monitoring
    pub async fn export_all_metrics(&self) -> DaemonResult<serde_json::Value> {
        let metrics = self.service_monitors.export_all_metrics().await?;
        info!("Exported {} metric categories", metrics.as_object().map(|o| o.len()).unwrap_or(0));
        Ok(metrics)
    }

    /// Perform emergency shutdown with alert
    pub async fn emergency_shutdown(&self, reason: &str) -> DaemonResult<()> {
        error!("EMERGENCY SHUTDOWN: {}", reason);

        // Send emergency alerts
        use crate::grpc::alerting::{Alert, AlertSeverity, AlertType};

        let alert = Alert::new(
            "system".to_string(),
            AlertSeverity::Emergency,
            AlertType::Custom {
                message: format!("Emergency shutdown: {}", reason),
            },
            format!("System emergency shutdown initiated: {}", reason),
        );

        self.alert_manager.send_alert(alert).await?;

        // Disable automatic recovery during shutdown
        self.recovery_system.set_auto_recovery_enabled(false).await;

        // Stop health monitoring
        self.stop().await?;

        error!("Emergency shutdown complete");
        Ok(())
    }
}

/// Comprehensive health system status
#[derive(Debug, Clone, serde::Serialize)]
pub struct HealthSystemStatus {
    pub summary: String,
    pub service_stats: crate::grpc::service_monitors::ServiceStats,
    pub alert_stats: crate::grpc::alerting::AlertStats,
    pub recent_recoveries: Vec<crate::grpc::alerting::RecoveryAttempt>,
}

/// Factory for creating health monitoring configurations
pub struct HealthConfigurationFactory;

impl HealthConfigurationFactory {
    /// Create production-ready alert configuration
    pub fn production_alert_config() -> AlertConfig {
        AlertConfig {
            max_error_rate: 0.01,      // 1% max error rate
            max_avg_latency_ms: 500.0,  // 500ms average latency
            max_p99_latency_ms: 2000.0, // 2 second P99 latency
            min_throughput_rps: 5.0,    // 5 requests/second minimum
            evaluation_window_secs: 300, // 5 minute evaluation window
        }
    }

    /// Create development-friendly alert configuration
    pub fn development_alert_config() -> AlertConfig {
        AlertConfig {
            max_error_rate: 0.05,       // 5% max error rate
            max_avg_latency_ms: 2000.0,  // 2 second average latency
            max_p99_latency_ms: 10000.0, // 10 second P99 latency
            min_throughput_rps: 1.0,     // 1 request/second minimum
            evaluation_window_secs: 60,  // 1 minute evaluation window
        }
    }

    /// Create high-performance alert configuration
    pub fn high_performance_alert_config() -> AlertConfig {
        AlertConfig {
            max_error_rate: 0.001,      // 0.1% max error rate
            max_avg_latency_ms: 100.0,   // 100ms average latency
            max_p99_latency_ms: 500.0,   // 500ms P99 latency
            min_throughput_rps: 50.0,    // 50 requests/second minimum
            evaluation_window_secs: 60,  // 1 minute evaluation window
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_health_integration_creation() {
        let integration = HealthIntegration::new(None).await.unwrap();

        // Test that all components are initialized
        assert!(!integration.service_monitors.get_all_stats().await.system_message.is_empty());
        assert_eq!(integration.alert_manager.get_active_alerts().await.len(), 0);
    }

    #[tokio::test]
    async fn test_health_integration_start_stop() {
        let integration = HealthIntegration::new(None).await.unwrap();

        // Start the system
        integration.start().await.unwrap();

        // Check that the health check handle is set
        let handle_guard = integration.health_check_handle.read().await;
        assert!(handle_guard.is_some());
        drop(handle_guard);

        // Stop the system
        integration.stop().await.unwrap();

        // Check that the handle is cleared
        let handle_guard = integration.health_check_handle.read().await;
        assert!(handle_guard.is_none());
    }

    #[tokio::test]
    async fn test_health_integration_comprehensive_status() {
        let integration = HealthIntegration::new(None).await.unwrap();

        let status = integration.get_comprehensive_status().await.unwrap();

        assert!(!status.summary.is_empty());
        assert_eq!(status.alert_stats.total_active, 0);
        assert_eq!(status.recent_recoveries.len(), 0);
    }

    #[tokio::test]
    async fn test_health_integration_demonstration() {
        let integration = HealthIntegration::new(None).await.unwrap();

        // This test runs the full demonstration scenario
        let result = integration.demonstrate_health_scenario().await;
        assert!(result.is_ok());

        // Check that metrics were generated
        let status = integration.get_comprehensive_status().await.unwrap();

        // Should have processed documents
        assert!(status.service_stats.document_processing.total_processed > 0);

        // Should have search activity
        assert!(status.service_stats.search.cache_hit_rate >= 0.0);

        // Should have memory operations
        assert!(status.service_stats.memory.total_documents > 0);
    }

    #[tokio::test]
    async fn test_health_integration_export_metrics() {
        let integration = HealthIntegration::new(None).await.unwrap();

        // Generate some activity first
        let tracker = integration.service_monitors.search_service.start_search("test").await;
        tracker.complete_success(5).await;

        let metrics = integration.export_all_metrics().await.unwrap();
        assert!(metrics.is_object());

        let obj = metrics.as_object().unwrap();
        assert!(!obj.is_empty());
    }

    #[tokio::test]
    async fn test_health_integration_emergency_shutdown() {
        let integration = HealthIntegration::new(None).await.unwrap();

        // Start the system first
        integration.start().await.unwrap();

        // Perform emergency shutdown
        let result = integration.emergency_shutdown("Test emergency").await;
        assert!(result.is_ok());

        // System should be stopped
        let handle_guard = integration.health_check_handle.read().await;
        assert!(handle_guard.is_none());
    }

    #[test]
    fn test_health_configuration_factory() {
        let prod_config = HealthConfigurationFactory::production_alert_config();
        assert_eq!(prod_config.max_error_rate, 0.01);

        let dev_config = HealthConfigurationFactory::development_alert_config();
        assert_eq!(dev_config.max_error_rate, 0.05);

        let perf_config = HealthConfigurationFactory::high_performance_alert_config();
        assert_eq!(perf_config.max_error_rate, 0.001);
    }

    #[tokio::test]
    async fn test_health_integration_with_connection_manager() {
        let connection_manager = Arc::new(ConnectionManager::new(100, 10));
        let integration = HealthIntegration::new(Some(connection_manager.clone())).await.unwrap();

        // Register a connection
        connection_manager.register_connection("test-client".to_string()).unwrap();

        let status = integration.get_comprehensive_status().await.unwrap();

        // The connection should be reflected in the system metrics
        assert!(!status.summary.is_empty());
    }

    #[tokio::test]
    async fn test_concurrent_health_operations() {
        let integration = Arc::new(HealthIntegration::new(None).await.unwrap());
        let mut handles = vec![];

        // Start multiple concurrent operations
        for i in 0..5 {
            let integration_clone = Arc::clone(&integration);
            let handle = tokio::spawn(async move {
                // Simulate concurrent document processing
                integration_clone.service_monitors.document_processor
                    .start_processing(&format!("concurrent_doc_{}", i)).await;

                tokio::time::sleep(Duration::from_millis(10)).await;

                integration_clone.service_monitors.document_processor
                    .complete_processing(&format!("concurrent_doc_{}", i), Duration::from_millis(50)).await;

                // Simulate concurrent searches
                let tracker = integration_clone.service_monitors.search_service
                    .start_search(&format!("concurrent_query_{}", i)).await;
                tracker.complete_success(i + 1).await;
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let status = integration.get_comprehensive_status().await.unwrap();
        assert!(status.service_stats.document_processing.total_processed >= 5);
    }
}
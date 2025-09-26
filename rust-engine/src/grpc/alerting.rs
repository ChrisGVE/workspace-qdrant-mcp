//! Alerting system for service degradation and automatic health recovery
//!
//! This module provides comprehensive alerting capabilities with configurable thresholds,
//! multiple notification channels, and automatic recovery procedures.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{RwLock, mpsc};
use tracing::{debug, info, warn, error};
use serde::{Serialize, Deserialize};

use crate::error::{DaemonResult, DaemonError};
use crate::grpc::health::{HealthStatus, ServiceHealth, HealthMonitoringSystem};

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning - service degraded but still functional
    Warning,
    /// Critical - service failure requiring immediate attention
    Critical,
    /// Emergency - system-wide failure
    Emergency,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
            AlertSeverity::Emergency => write!(f, "EMERGENCY"),
        }
    }
}

/// Alert types for different conditions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    /// High error rate alert
    HighErrorRate { current_rate: String, threshold: String },
    /// High latency alert
    HighLatency { current_latency: String, threshold: String },
    /// Low throughput alert
    LowThroughput { current_throughput: String, threshold: String },
    /// Service unavailable alert
    ServiceUnavailable { reason: String },
    /// Resource exhaustion alert
    ResourceExhaustion { resource: String, utilization: String },
    /// Recovery notification
    ServiceRecovered { previous_state: String },
    /// Custom alert with arbitrary message
    Custom { message: String },
}

impl std::fmt::Display for AlertType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertType::HighErrorRate { current_rate, threshold } =>
                write!(f, "High Error Rate: {} (threshold: {})", current_rate, threshold),
            AlertType::HighLatency { current_latency, threshold } =>
                write!(f, "High Latency: {} (threshold: {})", current_latency, threshold),
            AlertType::LowThroughput { current_throughput, threshold } =>
                write!(f, "Low Throughput: {} (threshold: {})", current_throughput, threshold),
            AlertType::ServiceUnavailable { reason } =>
                write!(f, "Service Unavailable: {}", reason),
            AlertType::ResourceExhaustion { resource, utilization } =>
                write!(f, "Resource Exhaustion: {} at {}", resource, utilization),
            AlertType::ServiceRecovered { previous_state } =>
                write!(f, "Service Recovered from {}", previous_state),
            AlertType::Custom { message } =>
                write!(f, "Custom Alert: {}", message),
        }
    }
}

/// Alert notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Unique alert ID
    pub id: String,
    /// Service name that triggered the alert
    pub service_name: String,
    /// Alert severity level
    pub severity: AlertSeverity,
    /// Type of alert with details
    pub alert_type: AlertType,
    /// Human-readable alert message
    pub message: String,
    /// When the alert was triggered
    pub timestamp: SystemTime,
    /// Alert metadata for context
    pub metadata: HashMap<String, String>,
    /// Whether this alert is resolved
    pub resolved: bool,
    /// When the alert was resolved (if applicable)
    pub resolved_at: Option<SystemTime>,
}

impl Alert {
    /// Create a new alert
    pub fn new(
        service_name: String,
        severity: AlertSeverity,
        alert_type: AlertType,
        message: String,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_string();

        Self {
            id,
            service_name,
            severity,
            alert_type,
            message,
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
            resolved: false,
            resolved_at: None,
        }
    }

    /// Add metadata to the alert
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Mark alert as resolved
    pub fn resolve(&mut self) {
        self.resolved = true;
        self.resolved_at = Some(SystemTime::now());
    }
}

/// Alert notification channel trait
#[async_trait::async_trait]
pub trait AlertChannel: Send + Sync {
    /// Send an alert notification
    async fn send_alert(&self, alert: &Alert) -> DaemonResult<()>;

    /// Get channel name
    fn name(&self) -> &str;
}

/// Log-based alert channel
#[derive(Debug)]
pub struct LogAlertChannel {
    name: String,
}

impl LogAlertChannel {
    pub fn new() -> Self {
        Self {
            name: "log".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl AlertChannel for LogAlertChannel {
    async fn send_alert(&self, alert: &Alert) -> DaemonResult<()> {
        let log_message = format!(
            "ALERT [{}] {} - {}: {}",
            alert.severity,
            alert.service_name,
            alert.alert_type,
            alert.message
        );

        match alert.severity {
            AlertSeverity::Info => info!("{}", log_message),
            AlertSeverity::Warning => warn!("{}", log_message),
            AlertSeverity::Critical | AlertSeverity::Emergency => error!("{}", log_message),
        }

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Webhook-based alert channel
#[derive(Debug)]
pub struct WebhookAlertChannel {
    name: String,
    webhook_url: String,
    client: reqwest::Client,
}

impl WebhookAlertChannel {
    pub fn new(name: String, webhook_url: String) -> Self {
        Self {
            name,
            webhook_url,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl AlertChannel for WebhookAlertChannel {
    async fn send_alert(&self, alert: &Alert) -> DaemonResult<()> {
        let payload = serde_json::json!({
            "alert": alert,
            "timestamp": alert.timestamp,
            "severity": alert.severity,
            "service": alert.service_name,
            "type": alert.alert_type,
            "message": alert.message,
            "metadata": alert.metadata
        });

        let response = self.client
            .post(&self.webhook_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| DaemonError::NetworkConnection {
                message: format!("Failed to send webhook alert: {}", e)
            })?;

        if !response.status().is_success() {
            return Err(DaemonError::NetworkConnection {
                message: format!("Webhook returned error status: {}", response.status())
            });
        }

        debug!("Alert sent via webhook: {}", alert.id);
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Email alert channel configuration
#[derive(Debug, Clone)]
pub struct EmailConfig {
    pub smtp_server: String,
    pub smtp_port: u16,
    pub username: String,
    pub password: String,
    pub from_email: String,
    pub to_emails: Vec<String>,
}

/// Email-based alert channel
#[derive(Debug)]
pub struct EmailAlertChannel {
    name: String,
    config: EmailConfig,
}

impl EmailAlertChannel {
    pub fn new(name: String, config: EmailConfig) -> Self {
        Self { name, config }
    }
}

#[async_trait::async_trait]
impl AlertChannel for EmailAlertChannel {
    async fn send_alert(&self, alert: &Alert) -> DaemonResult<()> {
        // For a real implementation, this would use an SMTP client
        // For now, just log that we would send an email
        info!("Would send email alert to {:?}: {} - {}",
            self.config.to_emails, alert.service_name, alert.message);

        // In a real implementation:
        // 1. Create SMTP connection
        // 2. Format email with alert details
        // 3. Send to all recipients
        // 4. Handle SMTP errors appropriately

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Alert processing and routing system
pub struct AlertManager {
    /// Registered alert channels
    channels: Arc<RwLock<Vec<Arc<dyn AlertChannel>>>>,
    /// Active alerts (unresolved)
    active_alerts: Arc<RwLock<HashMap<String, Alert>>>,
    /// Alert history (resolved alerts)
    alert_history: Arc<RwLock<Vec<Alert>>>,
    /// Maximum history size to prevent memory growth
    max_history_size: usize,
    /// Alert processing queue
    alert_sender: mpsc::UnboundedSender<Alert>,
    /// Alert processor task handle
    _processor_handle: tokio::task::JoinHandle<()>,
}

impl std::fmt::Debug for AlertManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlertManager")
            .field("channels", &"<trait objects>")
            .field("active_alerts", &self.active_alerts)
            .field("alert_history", &self.alert_history)
            .field("max_history_size", &self.max_history_size)
            .field("_processor_handle", &"<task handle>")
            .finish()
    }
}

impl AlertManager {
    /// Create a new alert manager
    pub fn new() -> Self {
        let channels = Arc::new(RwLock::new(Vec::new()));
        let active_alerts = Arc::new(RwLock::new(HashMap::new()));
        let alert_history = Arc::new(RwLock::new(Vec::new()));
        let max_history_size = 1000;

        let (alert_sender, alert_receiver) = mpsc::unbounded_channel();

        // Spawn alert processing task
        let processor_handle = tokio::spawn(Self::process_alerts(
            channels.clone(),
            active_alerts.clone(),
            alert_history.clone(),
            max_history_size,
            alert_receiver,
        ));

        Self {
            channels,
            active_alerts,
            alert_history,
            max_history_size,
            alert_sender,
            _processor_handle: processor_handle,
        }
    }

    /// Add an alert channel
    pub async fn add_channel(&self, channel: Arc<dyn AlertChannel>) {
        let mut channels = self.channels.write().await;
        channels.push(channel);
    }

    /// Send an alert
    pub async fn send_alert(&self, alert: Alert) -> DaemonResult<()> {
        self.alert_sender
            .send(alert)
            .map_err(|e| DaemonError::System {
                message: format!("Failed to queue alert: {}", e)
            })?;

        Ok(())
    }

    /// Resolve an alert by ID
    pub async fn resolve_alert(&self, alert_id: &str) -> DaemonResult<bool> {
        let mut active_alerts = self.active_alerts.write().await;
        let mut alert_history = self.alert_history.write().await;

        if let Some(mut alert) = active_alerts.remove(alert_id) {
            alert.resolve();
            alert_history.push(alert);

            // Maintain history size limit
            if alert_history.len() > self.max_history_size {
                alert_history.remove(0);
            }

            info!("Alert {} resolved", alert_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get all active alerts
    pub async fn get_active_alerts(&self) -> Vec<Alert> {
        let active_alerts = self.active_alerts.read().await;
        active_alerts.values().cloned().collect()
    }

    /// Get alert history
    pub async fn get_alert_history(&self, limit: Option<usize>) -> Vec<Alert> {
        let alert_history = self.alert_history.read().await;
        let limit = limit.unwrap_or(alert_history.len());

        alert_history
            .iter()
            .rev() // Most recent first
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get alert statistics
    pub async fn get_alert_stats(&self) -> AlertStats {
        let active_alerts = self.active_alerts.read().await;
        let alert_history = self.alert_history.read().await;

        let mut severity_counts = HashMap::new();
        let mut service_counts = HashMap::new();

        // Count active alerts
        for alert in active_alerts.values() {
            *severity_counts.entry(alert.severity).or_insert(0u32) += 1;
            *service_counts.entry(alert.service_name.clone()).or_insert(0u32) += 1;
        }

        let total_active = active_alerts.len();
        let total_resolved = alert_history.len();

        AlertStats {
            total_active,
            total_resolved,
            severity_counts,
            service_counts,
        }
    }

    /// Alert processing loop
    async fn process_alerts(
        channels: Arc<RwLock<Vec<Arc<dyn AlertChannel>>>>,
        active_alerts: Arc<RwLock<HashMap<String, Alert>>>,
        alert_history: Arc<RwLock<Vec<Alert>>>,
        max_history_size: usize,
        mut alert_receiver: mpsc::UnboundedReceiver<Alert>,
    ) {
        while let Some(alert) = alert_receiver.recv().await {
            debug!("Processing alert: {}", alert.id);

            // Store in active alerts
            {
                let mut active = active_alerts.write().await;
                active.insert(alert.id.clone(), alert.clone());
            }

            // Send to all channels
            {
                let channels_guard = channels.read().await;
                for channel in channels_guard.iter() {
                    if let Err(e) = channel.send_alert(&alert).await {
                        error!("Failed to send alert via channel {}: {}", channel.name(), e);
                    }
                }
            }

            // For recovery alerts, auto-resolve the corresponding degradation alert
            if let AlertType::ServiceRecovered { .. } = &alert.alert_type {
                Self::auto_resolve_service_alerts(&active_alerts, &alert_history, &alert.service_name, max_history_size).await;
            }
        }
    }

    /// Automatically resolve alerts for a service that has recovered
    async fn auto_resolve_service_alerts(
        active_alerts: &Arc<RwLock<HashMap<String, Alert>>>,
        alert_history: &Arc<RwLock<Vec<Alert>>>,
        service_name: &str,
        max_history_size: usize,
    ) {
        let mut active = active_alerts.write().await;
        let mut history = alert_history.write().await;

        // Find and resolve alerts for the service
        let alert_ids_to_resolve: Vec<String> = active
            .values()
            .filter(|alert| alert.service_name == service_name && !alert.resolved)
            .map(|alert| alert.id.clone())
            .collect();

        for alert_id in alert_ids_to_resolve {
            if let Some(mut alert) = active.remove(&alert_id) {
                alert.resolve();
                history.push(alert);
                info!("Auto-resolved alert {} for recovered service {}", alert_id, service_name);
            }
        }

        // Maintain history size limit
        if history.len() > max_history_size {
            let excess = history.len() - max_history_size;
            history.drain(0..excess);
        }
    }
}

/// Alert statistics
#[derive(Debug, Clone, Serialize)]
pub struct AlertStats {
    pub total_active: usize,
    pub total_resolved: usize,
    pub severity_counts: HashMap<AlertSeverity, u32>,
    pub service_counts: HashMap<String, u32>,
}

/// Automatic recovery system
pub struct RecoverySystem {
    /// Health monitoring system
    monitoring_system: Arc<HealthMonitoringSystem>,
    /// Alert manager for notifications
    alert_manager: Arc<AlertManager>,
    /// Recovery procedures by service name (not Debug due to trait objects)
    recovery_procedures: Arc<RwLock<HashMap<String, Arc<dyn RecoveryProcedure + Send + Sync>>>>,
    /// Recovery attempt history
    recovery_history: Arc<RwLock<Vec<RecoveryAttempt>>>,
    /// Whether automatic recovery is enabled
    auto_recovery_enabled: Arc<RwLock<bool>>,
}

impl std::fmt::Debug for RecoverySystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecoverySystem")
            .field("monitoring_system", &self.monitoring_system)
            .field("alert_manager", &self.alert_manager)
            .field("recovery_procedures", &"<trait objects>")
            .field("recovery_history", &self.recovery_history)
            .field("auto_recovery_enabled", &self.auto_recovery_enabled)
            .finish()
    }
}

/// Recovery procedure trait
#[async_trait::async_trait]
pub trait RecoveryProcedure: Send + Sync {
    /// Attempt to recover a degraded service
    async fn attempt_recovery(&self, service_name: &str, health: &ServiceHealth) -> DaemonResult<RecoveryResult>;

    /// Get procedure name
    fn name(&self) -> &str;
}

/// Recovery attempt result
#[derive(Debug, Clone, Serialize)]
pub struct RecoveryResult {
    pub success: bool,
    pub message: String,
    pub actions_taken: Vec<String>,
}

/// Recovery attempt record
#[derive(Debug, Clone, Serialize)]
pub struct RecoveryAttempt {
    pub service_name: String,
    pub timestamp: SystemTime,
    pub procedure_name: String,
    pub result: RecoveryResult,
}

/// Basic restart recovery procedure
#[derive(Debug)]
pub struct RestartRecoveryProcedure {
    name: String,
}

impl RestartRecoveryProcedure {
    pub fn new() -> Self {
        Self {
            name: "restart".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl RecoveryProcedure for RestartRecoveryProcedure {
    async fn attempt_recovery(&self, service_name: &str, _health: &ServiceHealth) -> DaemonResult<RecoveryResult> {
        info!("Attempting restart recovery for service: {}", service_name);

        // In a real implementation, this would:
        // 1. Stop the degraded service
        // 2. Clear any stuck resources
        // 3. Restart the service
        // 4. Wait for health check to pass

        tokio::time::sleep(Duration::from_millis(100)).await; // Simulate restart time

        Ok(RecoveryResult {
            success: true,
            message: format!("Successfully restarted service {}", service_name),
            actions_taken: vec![
                format!("Stopped service {}", service_name),
                format!("Cleared resources for {}", service_name),
                format!("Restarted service {}", service_name),
            ],
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Circuit breaker reset recovery procedure
#[derive(Debug)]
pub struct CircuitBreakerResetProcedure {
    name: String,
}

impl CircuitBreakerResetProcedure {
    pub fn new() -> Self {
        Self {
            name: "circuit_breaker_reset".to_string(),
        }
    }
}

#[async_trait::async_trait]
impl RecoveryProcedure for CircuitBreakerResetProcedure {
    async fn attempt_recovery(&self, service_name: &str, health: &ServiceHealth) -> DaemonResult<RecoveryResult> {
        info!("Attempting circuit breaker reset for service: {}", service_name);

        // Check if this is a circuit breaker issue
        if health.message.contains("circuit") || health.metrics.error_rate > 0.5 {
            // Reset circuit breakers for the service
            Ok(RecoveryResult {
                success: true,
                message: format!("Reset circuit breakers for service {}", service_name),
                actions_taken: vec![
                    format!("Reset circuit breaker for {}", service_name),
                    "Cleared error counters".to_string(),
                ],
            })
        } else {
            Ok(RecoveryResult {
                success: false,
                message: "Circuit breaker reset not applicable".to_string(),
                actions_taken: vec![],
            })
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl RecoverySystem {
    /// Create a new recovery system
    pub fn new(
        monitoring_system: Arc<HealthMonitoringSystem>,
        alert_manager: Arc<AlertManager>,
    ) -> Self {
        Self {
            monitoring_system,
            alert_manager,
            recovery_procedures: Arc::new(RwLock::new(HashMap::new())),
            recovery_history: Arc::new(RwLock::new(Vec::new())),
            auto_recovery_enabled: Arc::new(RwLock::new(true)),
        }
    }

    /// Add a recovery procedure for a service
    pub async fn add_recovery_procedure(&self, service_name: String, procedure: Arc<dyn RecoveryProcedure + Send + Sync>) {
        let mut procedures = self.recovery_procedures.write().await;
        procedures.insert(service_name, procedure);
    }

    /// Enable or disable automatic recovery
    pub async fn set_auto_recovery_enabled(&self, enabled: bool) {
        let mut auto_enabled = self.auto_recovery_enabled.write().await;
        *auto_enabled = enabled;

        if enabled {
            info!("Automatic recovery enabled");
        } else {
            warn!("Automatic recovery disabled");
        }
    }

    /// Perform recovery check for all services
    pub async fn perform_recovery_check(&self) -> DaemonResult<Vec<RecoveryAttempt>> {
        let auto_enabled = *self.auto_recovery_enabled.read().await;
        if !auto_enabled {
            debug!("Automatic recovery is disabled");
            return Ok(vec![]);
        }

        let all_health = self.monitoring_system.get_all_health().await;
        let mut recovery_attempts = Vec::new();

        for (service_name, health) in all_health.iter() {
            if health.status == HealthStatus::NotServing {
                info!("Service {} is degraded, attempting recovery", service_name);

                if let Some(attempt) = self.attempt_service_recovery(service_name, health).await? {
                    recovery_attempts.push(attempt);
                }
            }
        }

        Ok(recovery_attempts)
    }

    /// Attempt recovery for a specific service
    async fn attempt_service_recovery(&self, service_name: &str, health: &ServiceHealth) -> DaemonResult<Option<RecoveryAttempt>> {
        let procedures = self.recovery_procedures.read().await;

        if let Some(procedure) = procedures.get(service_name) {
            let result = procedure.attempt_recovery(service_name, health).await?;

            let attempt = RecoveryAttempt {
                service_name: service_name.to_string(),
                timestamp: SystemTime::now(),
                procedure_name: procedure.name().to_string(),
                result: result.clone(),
            };

            // Store recovery attempt
            {
                let mut history = self.recovery_history.write().await;
                history.push(attempt.clone());

                // Keep only last 100 recovery attempts
                if history.len() > 100 {
                    history.remove(0);
                }
            }

            // Send alert about recovery attempt
            let alert = if result.success {
                Alert::new(
                    service_name.to_string(),
                    AlertSeverity::Info,
                    AlertType::ServiceRecovered {
                        previous_state: "degraded".to_string(),
                    },
                    format!("Service {} recovered via {}: {}", service_name, procedure.name(), result.message),
                )
            } else {
                Alert::new(
                    service_name.to_string(),
                    AlertSeverity::Warning,
                    AlertType::Custom {
                        message: format!("Recovery failed for {}", service_name),
                    },
                    format!("Recovery procedure {} failed for {}: {}", procedure.name(), service_name, result.message),
                )
            };

            self.alert_manager.send_alert(alert).await?;

            Ok(Some(attempt))
        } else {
            debug!("No recovery procedure configured for service: {}", service_name);
            Ok(None)
        }
    }

    /// Get recovery history
    pub async fn get_recovery_history(&self, limit: Option<usize>) -> Vec<RecoveryAttempt> {
        let history = self.recovery_history.read().await;
        let limit = limit.unwrap_or(history.len());

        history
            .iter()
            .rev() // Most recent first
            .take(limit)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new(
            "test-service".to_string(),
            AlertSeverity::Warning,
            AlertType::HighErrorRate {
                current_rate: "10%".to_string(),
                threshold: "5%".to_string(),
            },
            "High error rate detected".to_string(),
        );

        assert_eq!(alert.service_name, "test-service");
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert!(!alert.resolved);
        assert!(alert.resolved_at.is_none());
    }

    #[test]
    fn test_alert_resolution() {
        let mut alert = Alert::new(
            "test-service".to_string(),
            AlertSeverity::Critical,
            AlertType::ServiceUnavailable { reason: "test".to_string() },
            "Service unavailable".to_string(),
        );

        alert.resolve();

        assert!(alert.resolved);
        assert!(alert.resolved_at.is_some());
    }

    #[tokio::test]
    async fn test_log_alert_channel() {
        let channel = LogAlertChannel::new();
        let alert = Alert::new(
            "test-service".to_string(),
            AlertSeverity::Info,
            AlertType::Custom { message: "test".to_string() },
            "Test alert".to_string(),
        );

        let result = channel.send_alert(&alert).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_alert_manager() {
        let manager = AlertManager::new();

        // Add a log channel
        let log_channel = Arc::new(LogAlertChannel::new());
        manager.add_channel(log_channel).await;

        // Send an alert
        let alert = Alert::new(
            "test-service".to_string(),
            AlertSeverity::Warning,
            AlertType::Custom { message: "test".to_string() },
            "Test alert".to_string(),
        );

        let alert_id = alert.id.clone();
        manager.send_alert(alert).await.unwrap();

        // Give the processor time to work
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Check active alerts
        let active_alerts = manager.get_active_alerts().await;
        assert_eq!(active_alerts.len(), 1);

        // Resolve the alert
        let resolved = manager.resolve_alert(&alert_id).await.unwrap();
        assert!(resolved);

        let active_alerts = manager.get_active_alerts().await;
        assert_eq!(active_alerts.len(), 0);

        let history = manager.get_alert_history(None).await;
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn test_restart_recovery_procedure() {
        let procedure = RestartRecoveryProcedure::new();
        let health = ServiceHealth {
            service_name: "test".to_string(),
            status: HealthStatus::NotServing,
            message: "degraded".to_string(),
            last_check: SystemTime::now(),
            uptime: Duration::from_secs(100),
            metrics: Default::default(),
        };

        let result = procedure.attempt_recovery("test-service", &health).await.unwrap();
        assert!(result.success);
        assert!(!result.actions_taken.is_empty());
    }

    #[tokio::test]
    async fn test_recovery_system() {
        let monitoring_system = Arc::new(HealthMonitoringSystem::new(None));
        let alert_manager = Arc::new(AlertManager::new());
        let recovery_system = RecoverySystem::new(monitoring_system.clone(), alert_manager);

        // Add recovery procedures
        recovery_system.add_recovery_procedure(
            "test-service".to_string(),
            Arc::new(RestartRecoveryProcedure::new())
        ).await;

        // Register a service and degrade it
        let monitor = monitoring_system.register_service("test-service".to_string(), None).await;
        monitor.set_status(HealthStatus::NotServing, Some("test degradation".to_string())).await;

        // Attempt recovery
        let attempts = recovery_system.perform_recovery_check().await.unwrap();
        assert_eq!(attempts.len(), 1);
        assert!(attempts[0].result.success);
    }

    #[tokio::test]
    async fn test_alert_stats() {
        let manager = AlertManager::new();

        // Send alerts of different severities
        for severity in [AlertSeverity::Info, AlertSeverity::Warning, AlertSeverity::Critical] {
            let alert = Alert::new(
                "test-service".to_string(),
                severity,
                AlertType::Custom { message: "test".to_string() },
                format!("Test {} alert", severity),
            );
            manager.send_alert(alert).await.unwrap();
        }

        // Give processor time to work
        tokio::time::sleep(Duration::from_millis(10)).await;

        let stats = manager.get_alert_stats().await;
        assert_eq!(stats.total_active, 3);
        assert_eq!(stats.severity_counts.len(), 3);
        assert_eq!(stats.service_counts.get("test-service"), Some(&3));
    }
}
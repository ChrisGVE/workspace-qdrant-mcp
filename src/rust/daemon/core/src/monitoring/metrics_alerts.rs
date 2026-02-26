//! Alerting logic for monitoring system health (Task 412.15-18)
//!
//! Provides configurable threshold-based alerts for:
//! - High queue depth
//! - Orphaned sessions
//! - High error rate
//! - Slow searches

use super::metrics_server::MetricsSnapshot;

/// Alert severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AlertSeverity {
    Warning,
    Critical,
}

/// Alert type enumeration
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AlertType {
    /// Queue depth exceeds threshold (Task 412.15)
    HighQueueDepth { depth: i64, threshold: i64 },
    /// Orphaned session detected (Task 412.16)
    OrphanedSession { project_id: String, last_heartbeat_secs: f64 },
    /// Error rate exceeds threshold (Task 412.17)
    HighErrorRate { error_rate_percent: f64, threshold_percent: f64 },
    /// Search latency exceeds threshold (Task 412.18)
    SlowSearches { p95_latency_ms: f64, threshold_ms: f64 },
}

/// Alert structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Alert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Alert configuration thresholds
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AlertConfig {
    /// Queue depth threshold for alerts (default: 1000)
    pub queue_depth_threshold: i64,
    /// Orphaned session timeout in seconds (default: 600 = 10 minutes)
    pub orphaned_session_timeout_secs: f64,
    /// Error rate threshold percentage (default: 5.0)
    pub error_rate_threshold_percent: f64,
    /// Slow search threshold in milliseconds (default: 500)
    pub slow_search_threshold_ms: f64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            queue_depth_threshold: 1000,
            orphaned_session_timeout_secs: 600.0,
            error_rate_threshold_percent: 5.0,
            slow_search_threshold_ms: 500.0,
        }
    }
}

/// Alert checker for monitoring system health
pub struct AlertChecker {
    config: AlertConfig,
}

impl AlertChecker {
    /// Create a new alert checker with default configuration
    pub fn new() -> Self {
        Self {
            config: AlertConfig::default(),
        }
    }

    /// Create a new alert checker with custom configuration
    pub fn with_config(config: AlertConfig) -> Self {
        Self { config }
    }

    /// Check for high queue depth alert (Task 412.15)
    pub fn check_queue_depth(&self, snapshot: &MetricsSnapshot) -> Option<Alert> {
        let total_depth: i64 = snapshot.queue_depths.values().sum();

        if total_depth > self.config.queue_depth_threshold {
            let severity = if total_depth > self.config.queue_depth_threshold * 2 {
                AlertSeverity::Critical
            } else {
                AlertSeverity::Warning
            };

            Some(Alert {
                alert_type: AlertType::HighQueueDepth {
                    depth: total_depth,
                    threshold: self.config.queue_depth_threshold,
                },
                severity,
                message: format!(
                    "Queue depth ({}) exceeds threshold ({})",
                    total_depth, self.config.queue_depth_threshold
                ),
                timestamp: chrono::Utc::now(),
            })
        } else {
            None
        }
    }

    /// Check for high error rate alert (Task 412.17)
    pub fn check_error_rate(&self, snapshot: &MetricsSnapshot) -> Option<Alert> {
        let total_errors: u64 = snapshot.error_counts.values().sum();
        let total_processed = snapshot.total_items_processed;

        if total_processed == 0 {
            return None;
        }

        let error_rate = (total_errors as f64 / total_processed as f64) * 100.0;

        if error_rate > self.config.error_rate_threshold_percent {
            let severity = if error_rate > self.config.error_rate_threshold_percent * 2.0 {
                AlertSeverity::Critical
            } else {
                AlertSeverity::Warning
            };

            Some(Alert {
                alert_type: AlertType::HighErrorRate {
                    error_rate_percent: error_rate,
                    threshold_percent: self.config.error_rate_threshold_percent,
                },
                severity,
                message: format!(
                    "Error rate ({:.2}%) exceeds threshold ({:.1}%)",
                    error_rate, self.config.error_rate_threshold_percent
                ),
                timestamp: chrono::Utc::now(),
            })
        } else {
            None
        }
    }

    /// Check all alerts and return active alerts
    pub fn check_all(&self, snapshot: &MetricsSnapshot) -> Vec<Alert> {
        let mut alerts = Vec::new();

        if let Some(alert) = self.check_queue_depth(snapshot) {
            alerts.push(alert);
        }

        if let Some(alert) = self.check_error_rate(snapshot) {
            alerts.push(alert);
        }

        // Note: Orphaned session and slow search alerts require additional
        // data sources (session heartbeat timestamps, search latency histograms)
        // that are not available in MetricsSnapshot alone.
        // These should be checked by the session monitor and search handler.

        alerts
    }
}

impl Default for AlertChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Check for orphaned sessions (Task 412.16)
///
/// This function should be called by the session monitor which has access
/// to project heartbeat timestamps.
pub fn create_orphaned_session_alert(
    project_id: &str,
    last_heartbeat_secs: f64,
    timeout_threshold_secs: f64,
) -> Option<Alert> {
    if last_heartbeat_secs > timeout_threshold_secs {
        let severity = if last_heartbeat_secs > timeout_threshold_secs * 2.0 {
            AlertSeverity::Critical
        } else {
            AlertSeverity::Warning
        };

        Some(Alert {
            alert_type: AlertType::OrphanedSession {
                project_id: project_id.to_string(),
                last_heartbeat_secs,
            },
            severity,
            message: format!(
                "Orphaned session for project '{}': no heartbeat for {:.0}s (threshold: {:.0}s)",
                project_id, last_heartbeat_secs, timeout_threshold_secs
            ),
            timestamp: chrono::Utc::now(),
        })
    } else {
        None
    }
}

/// Check for slow searches (Task 412.18)
///
/// This function should be called by the search handler with p95 latency data.
pub fn create_slow_search_alert(
    p95_latency_ms: f64,
    threshold_ms: f64,
) -> Option<Alert> {
    if p95_latency_ms > threshold_ms {
        let severity = if p95_latency_ms > threshold_ms * 2.0 {
            AlertSeverity::Critical
        } else {
            AlertSeverity::Warning
        };

        Some(Alert {
            alert_type: AlertType::SlowSearches {
                p95_latency_ms,
                threshold_ms,
            },
            severity,
            message: format!(
                "Search p95 latency ({:.0}ms) exceeds threshold ({:.0}ms)",
                p95_latency_ms, threshold_ms
            ),
            timestamp: chrono::Utc::now(),
        })
    } else {
        None
    }
}

//! Tests for metrics_core, metrics_server, and metrics_alerts

#[cfg(test)]
mod tests {
    use crate::monitoring::metrics_core::{DaemonMetrics, METRICS};
    use crate::monitoring::metrics_server::MetricsSnapshot;
    use crate::monitoring::metrics_alerts::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = DaemonMetrics::new();
        assert!(metrics.encode().is_ok());
    }

    #[test]
    fn test_session_metrics() {
        let metrics = DaemonMetrics::new();

        metrics.session_started("project-1", "high");
        let value = metrics
            .active_sessions
            .with_label_values(&["project-1", "high"])
            .get();
        assert_eq!(value, 1);

        metrics.session_ended("project-1", "high", 60.0);
        let value = metrics
            .active_sessions
            .with_label_values(&["project-1", "high"])
            .get();
        assert_eq!(value, 0);
    }

    #[test]
    fn test_queue_metrics() {
        let metrics = DaemonMetrics::new();

        metrics.set_queue_depth("high", "projects", 100);
        let value = metrics
            .queue_depth
            .with_label_values(&["high", "projects"])
            .get();
        assert_eq!(value, 100);

        metrics.queue_item_processed("high", "success", 0.5);
        let value = metrics
            .queue_items_processed_total
            .with_label_values(&["high", "success"])
            .get();
        assert_eq!(value, 1);
    }

    #[test]
    fn test_tenant_metrics() {
        let metrics = DaemonMetrics::new();

        metrics.set_tenant_documents("tenant-123", "projects", 500);
        let value = metrics
            .tenant_documents_total
            .with_label_values(&["tenant-123", "projects"])
            .get();
        assert_eq!(value, 500);

        metrics.tenant_search("tenant-123");
        let value = metrics
            .tenant_search_requests_total
            .with_label_values(&["tenant-123"])
            .get();
        assert_eq!(value, 1);
    }

    #[test]
    fn test_encode_prometheus_format() {
        let metrics = DaemonMetrics::new();

        metrics.session_started("test-project", "normal");
        metrics.set_queue_depth("normal", "projects", 50);

        let output = metrics.encode().expect("encoding should succeed");
        assert!(output.contains("memexd_active_sessions"));
        assert!(output.contains("memexd_queue_depth"));
    }

    #[test]
    fn test_metrics_snapshot() {
        METRICS.session_started("snapshot-test", "low");
        METRICS.set_queue_depth("low", "memory", 25);

        let snapshot = MetricsSnapshot::capture();
        assert!(snapshot.active_sessions >= 1);
    }

    // Alerting tests (Task 412.15-18)

    #[test]
    fn test_high_queue_depth_alert() {
        let checker = AlertChecker::new();
        let mut snapshot = MetricsSnapshot {
            uptime_seconds: 100.0,
            active_sessions: 5,
            total_sessions_lifetime: 10,
            queue_depths: std::collections::HashMap::new(),
            total_items_processed: 100,
            error_counts: std::collections::HashMap::new(),
            tenant_documents: std::collections::HashMap::new(),
        };

        snapshot.queue_depths.insert("high".to_string(), 500);
        assert!(checker.check_queue_depth(&snapshot).is_none());

        snapshot.queue_depths.insert("high".to_string(), 1500);
        let alert = checker.check_queue_depth(&snapshot).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Warning);

        snapshot.queue_depths.insert("high".to_string(), 3000);
        let alert = checker.check_queue_depth(&snapshot).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_high_error_rate_alert() {
        let checker = AlertChecker::new();
        let mut snapshot = MetricsSnapshot {
            uptime_seconds: 100.0,
            active_sessions: 5,
            total_sessions_lifetime: 10,
            queue_depths: std::collections::HashMap::new(),
            total_items_processed: 100,
            error_counts: std::collections::HashMap::new(),
            tenant_documents: std::collections::HashMap::new(),
        };

        snapshot.error_counts.insert("parse_error".to_string(), 2);
        assert!(checker.check_error_rate(&snapshot).is_none());

        snapshot.error_counts.insert("parse_error".to_string(), 10);
        let alert = checker.check_error_rate(&snapshot).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Warning);

        snapshot.error_counts.insert("parse_error".to_string(), 20);
        let alert = checker.check_error_rate(&snapshot).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_orphaned_session_alert() {
        assert!(create_orphaned_session_alert("project-1", 300.0, 600.0).is_none());

        let alert = create_orphaned_session_alert("project-1", 700.0, 600.0).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Warning);

        let alert = create_orphaned_session_alert("project-1", 1500.0, 600.0).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_slow_search_alert() {
        assert!(create_slow_search_alert(300.0, 500.0).is_none());

        let alert = create_slow_search_alert(600.0, 500.0).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Warning);

        let alert = create_slow_search_alert(1200.0, 500.0).unwrap();
        assert_eq!(alert.severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_alert_checker_all() {
        let checker = AlertChecker::new();
        let snapshot = MetricsSnapshot {
            uptime_seconds: 100.0,
            active_sessions: 5,
            total_sessions_lifetime: 10,
            queue_depths: vec![("high".to_string(), 1500)].into_iter().collect(),
            total_items_processed: 100,
            error_counts: vec![("error".to_string(), 10)].into_iter().collect(),
            tenant_documents: std::collections::HashMap::new(),
        };

        let alerts = checker.check_all(&snapshot);
        assert_eq!(alerts.len(), 2);
    }

    // Unified Queue metrics tests (Task 37.35)

    #[test]
    fn test_unified_queue_metrics() {
        let metrics = DaemonMetrics::new();

        metrics.set_unified_queue_depth("content", "pending", 50);
        let value = metrics.unified_queue_depth.with_label_values(&["content", "pending"]).get();
        assert_eq!(value, 50);

        metrics.unified_queue_item_processed("content", "ingest", "success", 0.5);
        let value = metrics.unified_queue_items_total.with_label_values(&["content", "ingest", "success"]).get();
        assert_eq!(value, 1);

        metrics.unified_queue_enqueued("mcp_store");
        let value = metrics.unified_queue_enqueues_total.with_label_values(&["mcp_store"]).get();
        assert_eq!(value, 1);

        metrics.unified_queue_dequeued("file");
        let value = metrics.unified_queue_dequeues_total.with_label_values(&["file"]).get();
        assert_eq!(value, 1);

        metrics.set_unified_queue_stale_items(3);
        let value = metrics.unified_queue_stale_items.with_label_values(&[]).get();
        assert_eq!(value, 3);

        metrics.unified_queue_retry("folder");
        let value = metrics.unified_queue_retries_total.with_label_values(&["folder"]).get();
        assert_eq!(value, 1);
    }

    #[test]
    fn test_unified_queue_metrics_in_prometheus_output() {
        let metrics = DaemonMetrics::new();

        metrics.set_unified_queue_depth("content", "pending", 100);
        metrics.unified_queue_enqueued("cli_ingest");

        let output = metrics.encode().expect("encoding should succeed");
        assert!(output.contains("memexd_unified_queue_depth"));
        assert!(output.contains("memexd_unified_queue_enqueues_total"));
    }
}

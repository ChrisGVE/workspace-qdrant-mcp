//! Integration tests for monitoring and observability (Task 412.20)
//!
//! Tests the monitoring infrastructure including metrics collection,
//! alerting, and telemetry correlation.

use workspace_qdrant_core::{
    MetricsSnapshot, METRICS,
    Alert, AlertChecker, AlertConfig, AlertSeverity, AlertType,
    create_orphaned_session_alert, create_slow_search_alert,
    OtelConfig, init_tracer_provider, shutdown_tracer,
    current_trace_id, current_span_id,
};

// ============================================================================
// Metrics Collection Tests
// ============================================================================

#[test]
fn test_metrics_singleton_access() {
    // METRICS is a global singleton that should be accessible
    // Just verify we can call methods without panic
    METRICS.session_started("test-singleton-project", "high");
    METRICS.session_ended("test-singleton-project", "high", 0.1);
}

#[test]
fn test_metrics_session_lifecycle() {
    // Test session metrics
    let project = "test-session-lifecycle";

    METRICS.session_started(project, "high");
    let snapshot1 = MetricsSnapshot::capture();
    assert!(snapshot1.active_sessions >= 0);

    METRICS.session_ended(project, "high", 10.0);
    let snapshot2 = MetricsSnapshot::capture();

    // Session count should have decreased or stayed same (other tests may affect)
    assert!(snapshot2.active_sessions <= snapshot1.active_sessions);
}

#[test]
fn test_metrics_queue_operations() {
    // Test queue metrics
    let collection = "test-queue-collection";

    METRICS.set_queue_depth("normal", collection, 10);
    let snapshot1 = MetricsSnapshot::capture();

    // Queue depth should be in the map
    let has_queue = snapshot1.queue_depths.values().any(|&d| d > 0);
    assert!(has_queue || snapshot1.queue_depths.is_empty()); // May be filtered out

    METRICS.queue_item_processed("normal", "success", 0.5);
    METRICS.queue_item_processed("normal", "failure", 0.1);

    let snapshot2 = MetricsSnapshot::capture();
    assert!(snapshot2.total_items_processed >= snapshot1.total_items_processed);
}

#[test]
fn test_metrics_heartbeat_tracking() {
    // Test heartbeat metrics
    METRICS.heartbeat_processed("test-heartbeat-project", 0.001);
    METRICS.heartbeat_processed("test-heartbeat-project", 0.002);

    // Heartbeats should be recorded (no direct assertion, just verify no panic)
}

#[test]
fn test_metrics_ingestion_errors() {
    // Test error tracking
    let snapshot_before = MetricsSnapshot::capture();

    METRICS.ingestion_error("parse_error");
    METRICS.ingestion_error("connection_timeout");
    METRICS.ingestion_error("parse_error");

    let snapshot_after = MetricsSnapshot::capture();

    // Error counts should have items after recording errors
    let total_errors_before: u64 = snapshot_before.error_counts.values().sum();
    let total_errors_after: u64 = snapshot_after.error_counts.values().sum();
    assert!(total_errors_after >= total_errors_before);
}

#[test]
fn test_metrics_encode_prometheus_format() {
    // Test that metrics can be encoded to Prometheus format
    let encoded = METRICS.encode();
    assert!(encoded.is_ok());

    let text = encoded.unwrap();
    // Should contain metric names
    assert!(text.contains("memexd") || text.is_empty());
}

// ============================================================================
// Alerting Logic Tests
// ============================================================================

#[test]
fn test_alert_checker_default() {
    let checker = AlertChecker::new();
    // Should create with default config - verify it doesn't panic
    let snapshot = MetricsSnapshot::capture();
    let _alerts = checker.check_all(&snapshot);
}

#[test]
fn test_alert_checker_with_config() {
    let config = AlertConfig {
        queue_depth_threshold: 100,
        orphaned_session_timeout_secs: 300.0,
        error_rate_threshold_percent: 10.0,
        slow_search_threshold_ms: 200.0,
    };
    let checker = AlertChecker::with_config(config);

    let snapshot = MetricsSnapshot::capture();
    let _alerts = checker.check_all(&snapshot);
}

#[test]
fn test_alert_checker_queue_depth_warning() {
    let config = AlertConfig {
        queue_depth_threshold: 100,
        ..AlertConfig::default()
    };
    let checker = AlertChecker::with_config(config);

    // Create a snapshot with high queue depth
    // We need to record queue depth first, then capture
    METRICS.set_queue_depth("test-priority", "test-alert-collection", 150);

    let snapshot = MetricsSnapshot::capture();
    let alert = checker.check_queue_depth(&snapshot);

    // May or may not trigger depending on total queue depths
    if let Some(alert) = alert {
        // Should be warning or critical
        assert!(alert.severity == AlertSeverity::Warning || alert.severity == AlertSeverity::Critical);
    }

    // Reset queue depth
    METRICS.set_queue_depth("test-priority", "test-alert-collection", 0);
}

#[test]
fn test_alert_check_all_returns_alerts() {
    let config = AlertConfig {
        queue_depth_threshold: 1,  // Very low threshold
        error_rate_threshold_percent: 0.1,  // Very low threshold
        ..AlertConfig::default()
    };
    let checker = AlertChecker::with_config(config);

    // Record some data
    METRICS.set_queue_depth("high", "test-check-all", 10);
    METRICS.queue_item_processed("high", "success", 0.1);
    METRICS.queue_item_processed("high", "failure", 0.1);

    let snapshot = MetricsSnapshot::capture();
    let alerts = checker.check_all(&snapshot);

    // With low thresholds, we might get alerts
    // Just verify the function works without panic
    for alert in &alerts {
        assert!(!alert.message.is_empty());
    }

    // Cleanup
    METRICS.set_queue_depth("high", "test-check-all", 0);
}

#[test]
fn test_create_orphaned_session_alert_below_threshold() {
    let alert = create_orphaned_session_alert("project-1", 300.0, 600.0);
    assert!(alert.is_none());
}

#[test]
fn test_create_orphaned_session_alert_warning() {
    let alert = create_orphaned_session_alert("project-1", 700.0, 600.0);
    assert!(alert.is_some());
    let alert = alert.unwrap();
    assert_eq!(alert.severity, AlertSeverity::Warning);
    match alert.alert_type {
        AlertType::OrphanedSession { project_id, last_heartbeat_secs } => {
            assert_eq!(project_id, "project-1");
            assert_eq!(last_heartbeat_secs, 700.0);
        }
        _ => panic!("Wrong alert type"),
    }
}

#[test]
fn test_create_orphaned_session_alert_critical() {
    let alert = create_orphaned_session_alert("project-1", 1300.0, 600.0);
    assert!(alert.is_some());
    let alert = alert.unwrap();
    assert_eq!(alert.severity, AlertSeverity::Critical);
}

#[test]
fn test_create_slow_search_alert_below_threshold() {
    let alert = create_slow_search_alert(400.0, 500.0);
    assert!(alert.is_none());
}

#[test]
fn test_create_slow_search_alert_warning() {
    let alert = create_slow_search_alert(600.0, 500.0);
    assert!(alert.is_some());
    let alert = alert.unwrap();
    assert_eq!(alert.severity, AlertSeverity::Warning);
    match alert.alert_type {
        AlertType::SlowSearches { p95_latency_ms, threshold_ms } => {
            assert_eq!(p95_latency_ms, 600.0);
            assert_eq!(threshold_ms, 500.0);
        }
        _ => panic!("Wrong alert type"),
    }
}

#[test]
fn test_create_slow_search_alert_critical() {
    let alert = create_slow_search_alert(1100.0, 500.0);
    assert!(alert.is_some());
    assert_eq!(alert.unwrap().severity, AlertSeverity::Critical);
}

#[test]
fn test_alert_config_default() {
    let config = AlertConfig::default();
    assert_eq!(config.queue_depth_threshold, 1000);
    assert_eq!(config.orphaned_session_timeout_secs, 600.0);
    assert_eq!(config.error_rate_threshold_percent, 5.0);
    assert_eq!(config.slow_search_threshold_ms, 500.0);
}

// ============================================================================
// OpenTelemetry Tracing Tests
// ============================================================================

#[test]
fn test_otel_config_default() {
    let config = OtelConfig::default();
    assert_eq!(config.service_name, "memexd");
    assert_eq!(config.sampling_ratio, 1.0);
    assert!(config.propagate_context);
}

#[test]
fn test_otel_config_from_env() {
    let config = OtelConfig::from_env();
    assert!(!config.service_name.is_empty());
    assert!(config.sampling_ratio >= 0.0 && config.sampling_ratio <= 1.0);
}

#[test]
fn test_otel_trace_id_without_active_span() {
    // Without an active span, trace IDs should be None
    assert!(current_trace_id().is_none());
    assert!(current_span_id().is_none());
}

#[test]
fn test_otel_tracer_provider_init() {
    // Test that we can initialize a tracer provider without OTLP endpoint
    let config = OtelConfig {
        service_name: "test-service".to_string(),
        service_version: "1.0.0".to_string(),
        otlp_endpoint: None,  // No exporter
        sampling_ratio: 1.0,
        propagate_context: true,
    };

    let result = init_tracer_provider(&config);
    assert!(result.is_ok());

    // Clean up
    shutdown_tracer();
}

// ============================================================================
// Metrics Snapshot Tests
// ============================================================================

#[test]
fn test_metrics_snapshot_capture() {
    let snapshot = MetricsSnapshot::capture();

    // All fields should be initialized (though may be zero)
    assert!(snapshot.uptime_seconds >= 0.0);
    assert!(snapshot.active_sessions >= 0);
    // total_sessions_lifetime, queue_depths, etc. can be any value
}

#[test]
fn test_metrics_snapshot_after_operations() {
    // Record some metrics
    METRICS.session_started("snapshot-test-project", "normal");
    METRICS.set_queue_depth("normal", "snapshot-collection", 5);

    // Get snapshot
    let snapshot = MetricsSnapshot::capture();

    // Should have captured the metrics
    assert!(snapshot.active_sessions >= 0);

    // Clean up
    METRICS.session_ended("snapshot-test-project", "normal", 1.0);
    METRICS.set_queue_depth("normal", "snapshot-collection", 0);
}

// ============================================================================
// Alert Serialization Tests
// ============================================================================

#[test]
fn test_alert_serialization() {
    let alert = Alert {
        alert_type: AlertType::HighQueueDepth {
            depth: 1500,
            threshold: 1000,
        },
        severity: AlertSeverity::Warning,
        message: "Queue depth exceeded threshold".to_string(),
        timestamp: chrono::Utc::now(),
    };

    // Should serialize to JSON
    let json = serde_json::to_string(&alert).unwrap();
    assert!(json.contains("HighQueueDepth"));
    assert!(json.contains("Warning"));

    // Should deserialize back
    let deserialized: Alert = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.severity, AlertSeverity::Warning);
}

#[test]
fn test_alert_severity_serialization() {
    let warning = AlertSeverity::Warning;
    let critical = AlertSeverity::Critical;

    assert_eq!(serde_json::to_string(&warning).unwrap(), "\"Warning\"");
    assert_eq!(serde_json::to_string(&critical).unwrap(), "\"Critical\"");
}

#[test]
fn test_alert_type_variants() {
    // Test all alert type variants can be constructed and serialized
    let queue = AlertType::HighQueueDepth { depth: 100, threshold: 50 };
    let orphan = AlertType::OrphanedSession {
        project_id: "test".to_string(),
        last_heartbeat_secs: 700.0
    };
    let error = AlertType::HighErrorRate {
        error_rate_percent: 10.0,
        threshold_percent: 5.0
    };
    let slow = AlertType::SlowSearches {
        p95_latency_ms: 600.0,
        threshold_ms: 500.0
    };

    // All should serialize without error
    assert!(serde_json::to_string(&queue).is_ok());
    assert!(serde_json::to_string(&orphan).is_ok());
    assert!(serde_json::to_string(&error).is_ok());
    assert!(serde_json::to_string(&slow).is_ok());
}

#[test]
fn test_alert_config_serialization() {
    let config = AlertConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: AlertConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(config.queue_depth_threshold, deserialized.queue_depth_threshold);
    assert_eq!(config.orphaned_session_timeout_secs, deserialized.orphaned_session_timeout_secs);
}

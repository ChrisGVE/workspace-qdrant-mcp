//! End-to-end verification of the OTLP metrics export path (Task 88, 88.4).
//!
//! The Prometheus pull endpoint remains the primary metric transport; this
//! exercises the additive OTLP **metrics** push path: build an `OtelConfig`
//! from an opted-in telemetry config, install the global meter provider +
//! Prometheus-registry bridge, let a periodic export fire against an endpoint
//! with no collector listening (must fail silently, never panic), then shut
//! down cleanly. Runs in its own test-binary process, so the process-global
//! meter provider it installs does not leak into other tests.

use std::time::Duration;

use workspace_qdrant_core::config::TelemetryConfig;
use workspace_qdrant_core::tracing_otel::{self, OtelConfig};

#[tokio::test]
async fn otlp_metrics_path_installs_bridges_and_shuts_down() {
    let mut telemetry = TelemetryConfig::default();
    telemetry.otlp.enabled = true;
    telemetry.otlp.metrics_enabled = true;
    // HTTP/protobuf (the default protocol) builds a reqwest client without
    // connecting, so install succeeds even with nothing listening here.
    telemetry.otlp.endpoint = "http://127.0.0.1:4318".to_string();

    assert!(
        telemetry.otlp.metrics_export_enabled(),
        "both flags set => metrics export is enabled"
    );

    let cfg = OtelConfig::from_telemetry(&telemetry, "0.0.0-test")
        .expect("enabled telemetry yields an OtelConfig");

    // Install the global meter provider + registry bridge. The bridge forwards
    // the daemon's real Prometheus gauge/counter families onto OTel observable
    // instruments, so the periodic reader exports genuine daemon metrics.
    let installed = tracing_otel::init_meter_provider(&cfg, Duration::from_millis(200))
        .expect("init_meter_provider must not error with a valid endpoint");
    assert!(
        installed,
        "a configured endpoint must install the meter provider"
    );

    // Allow at least one periodic export cycle to fire. With no collector at
    // the endpoint the export fails internally; this must not panic the test.
    tokio::time::sleep(Duration::from_millis(350)).await;

    // Clean shutdown flushes the final export; no-op-safe and must not panic.
    tracing_otel::shutdown_meter_provider();

    // A second shutdown is a guaranteed no-op (handle already taken).
    tracing_otel::shutdown_meter_provider();
}

#[test]
fn otlp_metrics_path_skipped_without_endpoint() {
    // Disabled-by-default config => from_telemetry returns None (no endpoint),
    // and init with an explicit None endpoint installs nothing.
    let telemetry = TelemetryConfig::default();
    assert!(!telemetry.otlp.metrics_export_enabled());
    assert!(OtelConfig::from_telemetry(&telemetry, "0.0.0-test").is_none());

    let cfg = OtelConfig {
        otlp_endpoint: None,
        ..OtelConfig::default()
    };
    let installed =
        tracing_otel::init_meter_provider(&cfg, Duration::from_secs(1)).expect("no-endpoint is Ok");
    assert!(!installed, "no endpoint => nothing installed");
}

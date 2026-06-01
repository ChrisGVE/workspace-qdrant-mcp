//! Tests for the per-record helpers on [`DaemonMetrics`] split out of
//! `metrics_core.rs` to keep that file under the project's 500-line ceiling.

use super::metrics_core::DaemonMetrics;
use prometheus::core::Collector;
use std::time::Duration;

#[test]
fn watcher_events_increment_per_label() {
    let m = DaemonMetrics::new();
    m.record_watcher_event("create");
    m.record_watcher_event("create");
    m.record_watcher_event("modify");
    m.record_watcher_event("delete");
    m.record_watcher_event("rename");
    assert_eq!(
        m.watcher_events_total.with_label_values(&["create"]).get(),
        2
    );
    assert_eq!(
        m.watcher_events_total.with_label_values(&["modify"]).get(),
        1
    );
    assert_eq!(
        m.watcher_events_total.with_label_values(&["delete"]).get(),
        1
    );
    assert_eq!(
        m.watcher_events_total.with_label_values(&["rename"]).get(),
        1
    );
}

#[test]
fn watcher_coalesced_increment_per_reason() {
    let m = DaemonMetrics::new();
    m.record_watcher_coalesced("debounce");
    m.record_watcher_coalesced("debounce");
    m.record_watcher_coalesced("duplicate");
    assert_eq!(
        m.watcher_coalesced_total
            .with_label_values(&["debounce"])
            .get(),
        2
    );
    assert_eq!(
        m.watcher_coalesced_total
            .with_label_values(&["duplicate"])
            .get(),
        1
    );
}

#[test]
fn grpc_records_count_and_duration() {
    let m = DaemonMetrics::new();
    m.record_grpc_call(
        "SystemService",
        "HealthCheck",
        true,
        Duration::from_millis(5),
    );
    m.record_grpc_call(
        "SystemService",
        "HealthCheck",
        true,
        Duration::from_millis(12),
    );
    m.record_grpc_call(
        "SystemService",
        "HealthCheck",
        false,
        Duration::from_millis(50),
    );
    m.record_grpc_call(
        "DocumentService",
        "Ingest",
        true,
        Duration::from_millis(300),
    );

    assert_eq!(
        m.grpc_requests_total
            .with_label_values(&["SystemService", "HealthCheck", "ok"])
            .get(),
        2
    );
    assert_eq!(
        m.grpc_requests_total
            .with_label_values(&["SystemService", "HealthCheck", "error"])
            .get(),
        1
    );
    assert_eq!(
        m.grpc_requests_total
            .with_label_values(&["DocumentService", "Ingest", "ok"])
            .get(),
        1
    );

    let hist = m
        .grpc_request_duration_seconds
        .with_label_values(&["SystemService", "HealthCheck"]);
    let metric = hist.collect();
    let sample_count = metric[0].get_metric()[0].get_histogram().get_sample_count();
    assert_eq!(sample_count, 3);
}

#[test]
fn encode_contains_extension_metric_names() {
    let m = DaemonMetrics::new();
    m.record_watcher_event("create");
    m.record_watcher_coalesced("debounce");
    m.record_grpc_call(
        "SystemService",
        "HealthCheck",
        true,
        Duration::from_millis(1),
    );
    let out = m.encode().expect("encode ok");
    assert!(out.contains("wqm_memexd_watcher_events_total"));
    assert!(out.contains("wqm_memexd_watcher_coalesced_total"));
    assert!(out.contains("wqm_memexd_grpc_requests_total"));
    assert!(out.contains("wqm_memexd_grpc_request_duration_seconds"));
}

#[test]
fn embedding_records_duration_and_batch_size() {
    let m = DaemonMetrics::new();
    m.record_embedding("all-MiniLM-L6-v2", 32, Duration::from_millis(150));
    m.record_embedding("all-MiniLM-L6-v2", 8, Duration::from_millis(60));

    let dur = m
        .embedding_duration_seconds
        .with_label_values(&["all-MiniLM-L6-v2"]);
    let metric = dur.collect();
    let sample_count = metric[0].get_metric()[0].get_histogram().get_sample_count();
    assert_eq!(sample_count, 2);
}

#[test]
fn sqlite_records_by_op() {
    let m = DaemonMetrics::new();
    m.record_sqlite("read", Duration::from_millis(2));
    m.record_sqlite("write", Duration::from_millis(5));
    m.record_sqlite("transaction", Duration::from_millis(10));
    for op in ["read", "write", "transaction"] {
        let h = m.sqlite_query_duration_seconds.with_label_values(&[op]);
        let metric = h.collect();
        assert_eq!(
            metric[0].get_metric()[0].get_histogram().get_sample_count(),
            1,
            "op {op} should have 1 sample"
        );
    }
}

#[test]
fn qdrant_records_duration_and_errors() {
    let m = DaemonMetrics::new();
    m.record_qdrant("upsert", Duration::from_millis(20), None);
    m.record_qdrant("upsert", Duration::from_millis(25), Some("timeout"));
    m.record_qdrant("search", Duration::from_millis(15), None);

    let upsert = m
        .qdrant_request_duration_seconds
        .with_label_values(&["upsert"]);
    assert_eq!(
        upsert.collect()[0].get_metric()[0]
            .get_histogram()
            .get_sample_count(),
        2
    );
    assert_eq!(
        m.qdrant_request_errors_total
            .with_label_values(&["upsert", "timeout"])
            .get(),
        1
    );
    assert_eq!(
        m.qdrant_request_errors_total
            .with_label_values(&["search", "timeout"])
            .get(),
        0
    );
}

#[test]
fn encode_contains_dependency_metric_names() {
    let m = DaemonMetrics::new();
    m.record_embedding("all-MiniLM-L6-v2", 1, Duration::from_millis(1));
    m.record_sqlite("read", Duration::from_millis(1));
    m.record_qdrant("search", Duration::from_millis(1), None);
    let out = m.encode().expect("encode ok");
    assert!(out.contains("wqm_memexd_embedding_duration_seconds"));
    assert!(out.contains("wqm_memexd_embedding_batch_size"));
    assert!(out.contains("wqm_memexd_sqlite_query_duration_seconds"));
    assert!(out.contains("wqm_memexd_qdrant_request_duration_seconds"));
}

// ── A5: frozen histogram bucket layouts (stable API) ─────────────────────

use crate::monitoring::metrics_factories::{
    EMBEDDING_DURATION_BUCKETS, PROCESSING_DURATION_BUCKETS,
};

/// Read the upper-bound boundary set of a live histogram series.
fn live_bucket_bounds(h: &prometheus::HistogramVec, labels: &[&str]) -> Vec<f64> {
    let metric = h.with_label_values(labels);
    let families = metric.collect();
    families[0].get_metric()[0]
        .get_histogram()
        .get_bucket()
        .iter()
        .map(|b| b.get_upper_bound())
        .collect()
}

#[test]
fn embedding_duration_live_buckets_equal_const() {
    // AC1: the registered embedding histogram's buckets are exactly the
    // single-source-of-truth const, including the new 10.0 and 30.0 uppers.
    let m = DaemonMetrics::new();
    let bounds = live_bucket_bounds(&m.embedding_duration_seconds, &["all-MiniLM-L6-v2"]);
    assert_eq!(bounds, EMBEDDING_DURATION_BUCKETS.to_vec());
    assert!(bounds.contains(&10.0) && bounds.contains(&30.0));
}

#[test]
fn processing_duration_buckets_const_is_frozen_layout() {
    // AC2: the frozen 11-boundary layout for wqm_memexd_processing_duration_seconds.
    // The collector itself is built by A2 from this const; here we lock the
    // stable-API boundary values so a later edit can't silently drift them.
    assert_eq!(
        PROCESSING_DURATION_BUCKETS,
        &[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    );
    // Internal consistency: strict superset of the embedding layout's upper
    // region — both end in 10.0, 30.0 so cross-histogram quantiles align.
    assert!(PROCESSING_DURATION_BUCKETS.ends_with(&[10.0, 30.0, 60.0]));
    assert!(EMBEDDING_DURATION_BUCKETS.ends_with(&[10.0, 30.0]));
}

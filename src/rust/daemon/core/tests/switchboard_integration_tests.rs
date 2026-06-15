//! End-to-end metrics-switchboard integration (Task 20).
//!
//! Exercises the full path — emit → route → drain → `DaemonMetrics` — plus the
//! telemetry on/off behaviour and the always-on control lane. The strong
//! assertion is the no-double-count one: N emitted batches produce exactly N
//! `record_embedding` observations through the real drain task.
//!
//! This is its own test binary, so the global `SWITCHBOARD` OnceCell is set once
//! here; the test that drives the global drain is `#[serial]`.

use std::time::Duration;

use workspace_qdrant_core::monitoring::metrics_core::METRICS;
use workspace_qdrant_core::switchboard::drain::{apply_to_metrics, run_switchboard_drain};
use workspace_qdrant_core::switchboard::{
    store_embedder_latency_fast, EmbedLatencyRec, EmbedderBatchRec, MetricId, MetricsSwitchboard,
    SwitchboardBuilder, SWITCHBOARD,
};

/// Cumulative observation count of the batch-size histogram for one model label,
/// parsed from the Prometheus exposition (0 if the series is absent). A unique
/// model label per test isolates the count from all other emitters.
fn batch_size_count(model: &str) -> u64 {
    let needle = format!("wqm_memexd_embedding_batch_size_count{{model=\"{model}\"}}");
    let text = METRICS.encode().expect("encode metrics");
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(&needle) {
            return rest.trim().parse().unwrap_or(0);
        }
    }
    0
}

/// Lazily initialise the global switchboard with the embedder-latency control fn
/// wired, exactly as `memexd` does.
fn global() -> &'static MetricsSwitchboard {
    SWITCHBOARD.get_or_init(|| {
        let mut b = SwitchboardBuilder::new();
        b.wire_control(MetricId::EmbedderLatency, store_embedder_latency_fast);
        b.seal()
    })
}

#[test]
fn test_telemetry_on_buffers_then_drains() {
    let sw = SwitchboardBuilder::new().seal();
    let h = sw.handle(MetricId::EmbedderBatch, "itest-local");
    sw.emit_embedder_batch(
        h,
        EmbedderBatchRec {
            batch_size: 4,
            elapsed: Duration::from_millis(20),
        },
    );

    let sample = sw.drain_one().expect("telemetry-on emit must buffer");
    apply_to_metrics(&sample);
    assert!(sw.drain_one().is_none(), "ring should be empty after drain");
}

#[test]
fn test_telemetry_off_does_not_buffer() {
    let sw = SwitchboardBuilder::new().seal();
    sw.set_telemetry_enabled(false);
    let h = sw.handle(MetricId::EmbedderBatch, "itest-off");
    sw.emit_embedder_batch(
        h,
        EmbedderBatchRec {
            batch_size: 1,
            elapsed: Duration::from_millis(1),
        },
    );
    assert!(
        sw.drain_one().is_none(),
        "telemetry off must not fill the buffer"
    );
}

#[test]
fn test_control_lane_runs_even_when_telemetry_off() {
    let mut b = SwitchboardBuilder::new();
    b.wire_control(MetricId::EmbedderLatency, store_embedder_latency_fast);
    let sw = b.seal();
    sw.set_telemetry_enabled(false);

    let h = sw.handle(MetricId::EmbedderLatency, "fastembed");
    sw.emit_record(
        h,
        EmbedLatencyRec {
            embed_ms: 55,
            source_bytes: 100,
        },
    );

    // Telemetry suppressed — nothing buffered…
    assert!(sw.drain_one().is_none());
    // …but the always-on control fanout still advanced.
    let v = sw
        .fanout()
        .read_fast(MetricId::EmbedderLatency)
        .expect("EmbedderLatency is a control id");
    assert!(
        (v - 55.0).abs() < 1e-9,
        "control fast lane = emitted embed_ms"
    );
}

#[tokio::test]
#[serial_test::serial]
async fn test_global_drain_records_each_batch_exactly_once() {
    let sw = global();
    let model = "itest-global";
    let before = batch_size_count(model);

    let h = sw.handle(MetricId::EmbedderBatch, model);
    for _ in 0..3 {
        sw.emit_embedder_batch(
            h,
            EmbedderBatchRec {
                batch_size: 5,
                elapsed: Duration::from_millis(10),
            },
        );
    }

    tokio::spawn(run_switchboard_drain());

    let mut after = before;
    for _ in 0..100 {
        after = batch_size_count(model);
        if after >= before + 3 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Exactly three observations — no drops, no double-count.
    assert_eq!(
        after,
        before + 3,
        "3 emitted batches must yield 3 record_embedding observations"
    );
    assert_eq!(sw.buffer_full_count(), 0, "no telemetry drops expected");
}

//! Full emit→verdict pipeline integration tests (#133 F2b/F6 — CONS-10).
//!
//! These drive a locally-sealed switchboard (not the global `OnceCell`) so the
//! control fns, the shared `Arc<ControlLane>` fanout, and the `EwmaState` the
//! verdict reads are all the SAME objects an emit advances — the single-source
//! handshake F1 establishes. They assert the emit-topology the probes depend on:
//! an emit advances the snapshot lane, and a sustained regression reaches an
//! Amber/Red verdict.

use crate::config::queue_health::QueueHealthConfig;
use crate::queue_health::probes::trend::a1_ms_per_kb;
use crate::queue_health::probes::ProbeResult;
use crate::queue_health::state::{EwmaState, Rag};
use crate::queue_health::verdict::verdict;
use crate::queue_health::QueueProcessorHealth;
use crate::switchboard::{
    store_dlq_depth, store_embedder_latency, store_ms_per_kb, store_throughput, EmbedLatencyRec,
    MetricId, MetricsSwitchboard, SwitchboardBuilder,
};

const HEALTHY_DISK: Option<u64> = Some(1_000_000_000_000);

/// A locally-sealed switchboard with every control fn wired (mirrors the daemon
/// init wiring without touching the global `SWITCHBOARD`).
fn wired(cfg: &QueueHealthConfig) -> MetricsSwitchboard {
    let mut b = SwitchboardBuilder::new(cfg);
    b.wire_control(MetricId::EmbedderLatency, store_embedder_latency);
    b.wire_control(MetricId::QueueMsPerKb, store_ms_per_kb);
    b.wire_control(MetricId::QueueThroughput, store_throughput);
    b.wire_control(MetricId::QueueDlqDepth, store_dlq_depth);
    b.seal()
}

#[test]
fn emit_advances_the_ewma_state_snapshot() {
    let cfg = QueueHealthConfig::default();
    let sw = wired(&cfg);
    let ewma = EwmaState::from_fanout(sw.fanout(), &cfg);

    assert!(
        !ewma.ms_per_kb_snapshot().seeded,
        "unseeded before any emit"
    );
    sw.emit(sw.handle(MetricId::QueueMsPerKb, "queue"), 5.0);
    let snap = ewma.ms_per_kb_snapshot();
    assert!(snap.seeded, "emit must seed the lane the verdict reads");
    assert!((snap.fast - 5.0).abs() < 1e-9);
}

#[test]
fn full_pipeline_detects_regression() {
    let cfg = QueueHealthConfig::default();
    let sw = wired(&cfg);
    let ewma = EwmaState::from_fanout(sw.fanout(), &cfg);

    let h = sw.handle(MetricId::QueueMsPerKb, "queue");
    sw.emit(h, 1.0); // baseline
    for _ in 0..20 {
        sw.emit(h, 100.0); // sustained spike — fast climbs above slow
    }

    // Mirror the poll loop: evaluate the trend probe, debounce, cache.
    let raw = a1_ms_per_kb(&ewma, &cfg);
    let rag = ewma.observe(raw.culprit, raw.rag);
    ewma.set_trend_cache(vec![ProbeResult { rag, ..raw }]);

    let health = QueueProcessorHealth::new();
    let v = verdict(&ewma, &health, &cfg, true, HEALTHY_DISK, HEALTHY_DISK);
    assert!(
        v.overall == Rag::Amber || v.overall == Rag::Red,
        "sustained ms/KB regression must surface non-green, got {:?}",
        v.overall
    );
}

#[test]
fn embedder_emit_records_live_model_label() {
    // F10/SEC-02: the live provider label rides onto the lane for persistence.
    let cfg = QueueHealthConfig::default();
    let sw = wired(&cfg);
    sw.emit_record(
        sw.handle(MetricId::EmbedderLatency, "test-provider"),
        EmbedLatencyRec {
            embed_ms: 100,
            source_bytes: 1000,
        },
    );
    assert_eq!(
        sw.fanout().embedder_latency.model(),
        Some("test-provider"),
        "the emitted model label must be recorded on the lane"
    );
}

#[test]
fn fresh_daemon_pipeline_is_cold_start() {
    let cfg = QueueHealthConfig::default();
    let sw = wired(&cfg);
    let ewma = EwmaState::from_fanout(sw.fanout(), &cfg);
    let health = QueueProcessorHealth::new();

    let v = verdict(&ewma, &health, &cfg, true, HEALTHY_DISK, HEALTHY_DISK);
    assert!(v.cold_start, "no emit yet ⇒ cold start");
    assert_eq!(v.overall, Rag::Green);
}

//! Switchboard unit tests — emit paths, control dispatch, off-switch, overflow.
//!
//! These exercise locally-built `MetricsSwitchboard` instances (via
//! `SwitchboardBuilder::seal()`), never the global `SWITCHBOARD`, so they are
//! independent of init order and run in parallel.

use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn test_handle_is_copy_and_accessors() {
    let sw = SwitchboardBuilder::new().seal();
    let h = sw.handle(MetricId::EmbedderLatency, "m");
    let h2 = h; // Copy — `h` still usable.
    assert_eq!(h.id(), h2.id());
    assert_eq!(h.model(), "m");
}

#[test]
fn test_emit_scalar_buffers_sample() {
    let sw = SwitchboardBuilder::new().seal();
    let h = sw.handle(MetricId::QueueItemMs, "t");
    sw.emit(h, 42.0);
    match sw.drain_one() {
        Some(MetricSample::QueueItemMs(v)) => assert_eq!(v, 42),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn test_emit_record_buffers_sample() {
    let sw = SwitchboardBuilder::new().seal();
    let h = sw.handle(MetricId::EmbedderLatency, "fastembed");
    sw.emit_record(
        h,
        EmbedLatencyRec {
            embed_ms: 100,
            source_bytes: 5000,
        },
    );
    match sw.drain_one() {
        Some(MetricSample::EmbedderLatency { rec, model }) => {
            assert_eq!(rec.embed_ms, 100);
            assert_eq!(rec.source_bytes, 5000);
            assert_eq!(model, "fastembed");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn test_emit_record_on_scalar_handle_is_noop_for_emit() {
    // `emit` (scalar) must not produce a sample for a record-shaped id.
    let sw = SwitchboardBuilder::new().seal();
    let h = sw.handle(MetricId::EmbedderLatency, "t");
    sw.emit(h, 1.0);
    assert!(sw.drain_one().is_none());
}

#[test]
fn test_emit_batch_folds_to_one_sample() {
    let sw = SwitchboardBuilder::new().seal();
    let h = sw.handle(MetricId::QueueThroughput, "t");
    sw.emit_batch(h, &[2.0, 4.0, 6.0]); // mean = 4.0
    match sw.drain_one() {
        Some(MetricSample::QueueThroughput(v)) => assert!((v - 4.0).abs() < 1e-10),
        other => panic!("unexpected: {other:?}"),
    }
    assert!(sw.drain_one().is_none(), "exactly one summary sample");
}

#[test]
fn test_emit_batch_empty_is_noop() {
    let sw = SwitchboardBuilder::new().seal();
    let h = sw.handle(MetricId::QueueThroughput, "t");
    sw.emit_batch(h, &[]);
    assert!(sw.drain_one().is_none());
}

#[test]
fn test_telemetry_off_skips_buffer() {
    let mut b = SwitchboardBuilder::new();
    b.set_telemetry_enabled(false);
    let sw = b.seal();
    let h = sw.handle(MetricId::QueueItemMs, "t");
    sw.emit(h, 42.0);
    assert!(sw.drain_one().is_none());
}

#[test]
fn test_control_fn_runs_even_when_telemetry_off() {
    static CALLED: AtomicBool = AtomicBool::new(false);
    fn record(_f: &ControlFanout, _s: &MetricSample) {
        CALLED.store(true, Ordering::SeqCst);
    }

    let mut b = SwitchboardBuilder::new();
    b.set_telemetry_enabled(false);
    b.wire_control(MetricId::EmbedderLatency, record);
    let sw = b.seal();

    let h = sw.handle(MetricId::EmbedderLatency, "t");
    sw.emit_record(
        h,
        EmbedLatencyRec {
            embed_ms: 50,
            source_bytes: 100,
        },
    );
    assert!(CALLED.load(Ordering::SeqCst));
}

#[test]
fn test_control_fn_stores_into_fanout() {
    fn store_fast(f: &ControlFanout, s: &MetricSample) {
        if let MetricSample::EmbedderLatency { rec, .. } = s {
            let bits = (rec.embed_ms as f64).to_bits();
            f.embedder_latency_fast.store(bits, Ordering::Release);
        }
    }

    let mut b = SwitchboardBuilder::new();
    b.wire_control(MetricId::EmbedderLatency, store_fast);
    let sw = b.seal();

    let h = sw.handle(MetricId::EmbedderLatency, "t");
    sw.emit_record(
        h,
        EmbedLatencyRec {
            embed_ms: 12345,
            source_bytes: 1,
        },
    );
    assert_eq!(
        sw.fanout().read_fast(MetricId::EmbedderLatency),
        Some(12345.0)
    );
}

#[test]
fn test_production_embedder_control_fn_stores_embed_ms() {
    // The exact fn wired in main.rs: emit_record -> fast lane reflects embed_ms.
    let mut b = SwitchboardBuilder::new();
    b.wire_control(MetricId::EmbedderLatency, store_embedder_latency_fast);
    let sw = b.seal();

    let h = sw.handle(MetricId::EmbedderLatency, "fastembed");
    sw.emit_record(
        h,
        EmbedLatencyRec {
            embed_ms: 873,
            source_bytes: 4096,
        },
    );
    assert_eq!(
        sw.fanout().read_fast(MetricId::EmbedderLatency),
        Some(873.0)
    );
}

#[test]
fn test_buffer_overflow_is_counted() {
    let sw = SwitchboardBuilder::new().seal();
    let h = sw.handle(MetricId::QueueItemMs, "t");
    // Exceed the 4096 ring capacity without draining.
    for _ in 0..5000 {
        sw.emit(h, 1.0);
    }
    assert!(sw.buffer_full_count() > 0);
}

//! Hot-path micro-benchmark for the metrics switchboard (Task 23).
//!
//! Verifies the arch §9 budget: per emit ≤ 1 atomic update (control) + 1
//! lock-free buffer push (telemetry-if-on), no hash/map/lock/alloc/dynamic
//! dispatch. The numbers should land in the ~10–20 ns range uncontended.
//!
//! Run with: `cargo bench --bench switchboard_bench`

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use workspace_qdrant_core::switchboard::{
    store_embedder_latency_fast, EmbedLatencyRec, EmbedderBatchRec, MetricId, SwitchboardBuilder,
};

/// Scalar emit: one telemetry buffer push, no control fn (QueueItemMs is unwired).
fn bench_emit_scalar(c: &mut Criterion) {
    let sw = SwitchboardBuilder::new().seal();
    let handle = sw.handle(MetricId::QueueItemMs, "bench");
    c.bench_function("switchboard_emit_scalar", |b| {
        b.iter(|| sw.emit(black_box(handle), black_box(42.0)));
    });
}

/// Telemetry-only record emit (EmbedderBatch, no control fn).
fn bench_emit_embedder_batch(c: &mut Criterion) {
    let sw = SwitchboardBuilder::new().seal();
    let handle = sw.handle(MetricId::EmbedderBatch, "fastembed");
    let rec = EmbedderBatchRec {
        batch_size: 50,
        elapsed: Duration::from_millis(12),
    };
    c.bench_function("switchboard_emit_embedder_batch", |b| {
        b.iter(|| sw.emit_embedder_batch(black_box(handle), black_box(rec)));
    });
}

/// Control + telemetry record emit (EmbedderLatency wired to the fast lane) —
/// the worst case: buffer push AND an atomic control store.
fn bench_emit_record_with_control(c: &mut Criterion) {
    let mut builder = SwitchboardBuilder::new();
    builder.wire_control(MetricId::EmbedderLatency, store_embedder_latency_fast);
    let sw = builder.seal();
    let handle = sw.handle(MetricId::EmbedderLatency, "fastembed");
    let rec = EmbedLatencyRec {
        embed_ms: 50,
        source_bytes: 1000,
    };
    c.bench_function("switchboard_emit_record_with_control", |b| {
        b.iter(|| sw.emit_record(black_box(handle), black_box(rec)));
    });
}

criterion_group!(
    benches,
    bench_emit_scalar,
    bench_emit_embedder_batch,
    bench_emit_record_with_control
);
criterion_main!(benches);

//! # Metrics Switchboard
//!
//! A single global routing hub between every metric emitter in the daemon and
//! two distinct sink classes:
//!
//! - **Telemetry** — emit-only, drained to the existing Prometheus/OTLP path
//!   (`DaemonMetrics`). A single `telemetry_on` flag suppresses *export only*.
//! - **Control** — always-on, in-process consumers that react to values (the
//!   EWMA health-verdict lanes from #133). NEVER suppressed by any flag — going
//!   blind here would make the health verdict go dark.
//!
//! ## Lifecycle
//!
//! Built once at daemon init via [`SwitchboardBuilder`], sealed into the
//! [`SWITCHBOARD`] `OnceCell` before any emitter thread starts. Emitters obtain a
//! [`MetricHandle`] at construction ([`MetricsSwitchboard::handle`]) and pass it
//! to [`MetricsSwitchboard::emit`] / [`emit_record`](MetricsSwitchboard::emit_record)
//! on the hot path.
//!
//! ## Hot-path budget (arch §9)
//!
//! Per emit: ≤ 1 atomic update (control) + 1 lock-free buffer push
//! (telemetry-if-on). No per-call hash, map lookup, lock, heap alloc, or dynamic
//! dispatch — the string id is resolved to a handle once at registration.

pub mod control_fanout;
pub mod control_lane;
pub mod drain;
pub mod handle;
pub mod intern;
pub mod labels;
pub mod persist_task;
pub mod registry;
pub mod routing;
pub mod sample;
pub mod telemetry_buf;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use once_cell::sync::OnceCell;

use crate::config::queue_health::QueueHealthConfig;

pub use control_fanout::ControlFanout;
pub use control_lane::ControlLane;
pub use handle::MetricHandle;
pub use intern::intern_model_label;
pub use registry::{MetricDescriptor, MetricId, DESCRIPTORS, METRIC_COUNT};
pub use routing::{ControlFn, RoutingEntry};
pub use sample::{EmbedLatencyRec, EmbedderBatchRec, MetricSample};

/// The single global switchboard. Set exactly once at daemon init via
/// `SWITCHBOARD.set(builder.seal())`, before any emitter thread starts.
///
/// `OnceCell` (not `Lazy`) is deliberate: it enforces a single explicit
/// "fully wired before first emit" point. `Lazy` would let the routing table be
/// built on first access with no such guarantee (arch §5e).
pub static SWITCHBOARD: OnceCell<MetricsSwitchboard> = OnceCell::new();

/// Accessor for the global switchboard. `None` only during very early init,
/// before `set()` — a logged no-op for emitters, not a steady-state data-loss
/// window (arch §5e).
#[inline]
pub fn switchboard() -> Option<&'static MetricsSwitchboard> {
    SWITCHBOARD.get()
}

/// Builds the switchboard at init: wires control fns into the routing table,
/// then `seal()`s into the immutable runtime object. There is no after-seal
/// mutation path (arch §5e, sec-F2).
pub struct SwitchboardBuilder {
    routing: routing::RoutingTableBuilder,
    fanout: ControlFanout,
    telemetry_on: bool,
}

impl SwitchboardBuilder {
    /// Build an empty builder whose control lanes carry the configured EWMA
    /// alphas (`cfg.fast_alpha`/`cfg.slow_alpha`). The alphas are captured into
    /// each `ControlLane` here, at builder time, and are immutable thereafter.
    pub fn new(cfg: &QueueHealthConfig) -> Self {
        Self {
            routing: routing::RoutingTableBuilder::new(),
            fanout: ControlFanout::new(cfg),
            telemetry_on: true,
        }
    }

    /// Attach a control fn to a metric id (builder-only — mutates the table).
    pub fn wire_control(&mut self, id: MetricId, f: ControlFn) {
        self.routing.wire_control(id, f);
    }

    /// Set the initial telemetry-export state (from config).
    pub fn set_telemetry_enabled(&mut self, enabled: bool) {
        self.telemetry_on = enabled;
    }

    /// Borrow the fanout before sealing — e.g. to clone its `Arc`s for the
    /// control read side (`EwmaState`).
    pub fn fanout(&self) -> &ControlFanout {
        &self.fanout
    }

    /// Freeze the routing table and produce the runtime switchboard.
    pub fn seal(self) -> MetricsSwitchboard {
        MetricsSwitchboard {
            routing: self.routing.build(),
            fanout: self.fanout,
            telemetry_on: AtomicBool::new(self.telemetry_on),
            telemetry_buf: telemetry_buf::TelemetryBuffer::new(),
            buffer_full_count: AtomicU64::new(0),
        }
    }
}

/// The sealed runtime switchboard. Immutable routing table; atomics + lock-free
/// ring for state. `Send + Sync`.
pub struct MetricsSwitchboard {
    routing: Box<[RoutingEntry]>,
    fanout: ControlFanout,
    telemetry_on: AtomicBool,
    telemetry_buf: telemetry_buf::TelemetryBuffer,
    buffer_full_count: AtomicU64,
}

impl MetricsSwitchboard {
    /// Resolve an emitter handle. Pure — no mutation — so it is callable any time
    /// an emitter is constructed, before or after `seal()` (arch §5e).
    #[inline]
    pub fn handle(&self, id: MetricId, model: &'static str) -> MetricHandle {
        MetricHandle { id, model }
    }

    /// Toggle telemetry export at runtime. Control dispatch is never gated by it.
    pub fn set_telemetry_enabled(&self, enabled: bool) {
        self.telemetry_on.store(enabled, Ordering::Relaxed);
    }

    /// Emit a scalar value. Hottest path. Builds the scalar `MetricSample` for
    /// `h.id`; record-shaped ids (`EmbedderLatency`) are a no-op here — use
    /// [`emit_record`](Self::emit_record).
    #[inline]
    pub fn emit(&self, h: MetricHandle, value: f64) {
        if let Some(sample) = scalar_sample(h.id, value) {
            self.dispatch(h, sample);
        }
    }

    /// Emit N samples of one scalar field in one call. Folds to the mean and
    /// dispatches a single summary sample — one buffer push + one control store,
    /// preserving the hot-path proof (arch §13 Q2; exact summary shape is an
    /// acknowledged open detail, mean chosen as the interim).
    #[inline]
    pub fn emit_batch(&self, h: MetricHandle, values: &[f64]) {
        if values.is_empty() {
            return;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        if let Some(sample) = scalar_sample(h.id, mean) {
            self.dispatch(h, sample);
        }
    }

    /// Emit a co-measured multi-field record (`EmbedLatencyRec`).
    #[inline]
    pub fn emit_record(&self, h: MetricHandle, rec: EmbedLatencyRec) {
        let sample = MetricSample::EmbedderLatency {
            rec,
            model: h.model,
        };
        self.dispatch(h, sample);
    }

    /// Emit one embedding-batch telemetry record (`EmbedderBatchRec`). Routes the
    /// pre-existing `record_embedding` Prometheus path through the switchboard:
    /// telemetry-only (the `EmbedderBatch` id has no control fn). Coarse path
    /// (once per batch), so resolving `h.model` via the interner upstream is off
    /// the per-chunk hot loop.
    #[inline]
    pub fn emit_embedder_batch(&self, h: MetricHandle, rec: EmbedderBatchRec) {
        let sample = MetricSample::EmbedderBatch {
            rec,
            model: h.model,
        };
        self.dispatch(h, sample);
    }

    /// The shared tail of every emit shape: telemetry push (if on) + control
    /// dispatch (always).
    #[inline]
    fn dispatch(&self, h: MetricHandle, sample: MetricSample) {
        debug_assert_eq!(h.id, sample.id(), "handle id must match sample variant");

        if self.telemetry_on.load(Ordering::Relaxed) && !self.telemetry_buf.push(sample) {
            self.buffer_full_count.fetch_add(1, Ordering::Relaxed);
        }

        if let Some(f) = self.routing[h.id as usize].control_fn {
            f(&self.fanout, &sample);
        }
    }

    /// Borrow the control fanout (read side for consumers and the persist task).
    pub fn fanout(&self) -> &ControlFanout {
        &self.fanout
    }

    /// Pop one buffered telemetry sample for the background drain task.
    pub fn drain_one(&self) -> Option<MetricSample> {
        self.telemetry_buf.pop()
    }

    /// Number of samples dropped because the telemetry ring was full.
    pub fn buffer_full_count(&self) -> u64 {
        self.buffer_full_count.load(Ordering::Relaxed)
    }

    /// Reload persisted slow-lane baselines into the fanout on daemon restart.
    /// Called from `main.rs` after migration + seal, before the processing loop
    /// (arch §3d). A failure is "no prior baseline" — a cold start.
    pub async fn reload_baselines(&self, pool: &sqlx::SqlitePool) -> Result<(), sqlx::Error> {
        let rows: Vec<(String, String, f64)> = sqlx::query_as(
            "SELECT metric_id, field, value FROM control_baseline WHERE lane = 'slow'",
        )
        .fetch_all(pool)
        .await?;

        let count = rows.len();
        for (metric_id, field, value) in rows {
            // restore_baseline seeds BOTH lanes and sets seeded=true (DOM-06), so
            // the first live sample after restart advances the restored baseline
            // instead of overwriting it (the bug this fixes: raw slow-lane store
            // left seeded=false → next update re-seeded from scratch).
            match (metric_id.as_str(), field.as_str()) {
                ("EmbedderLatency", "embed_ms") => {
                    self.fanout.embedder_latency.restore_baseline(value);
                }
                ("QueueMsPerKb", "ms_per_kb") => {
                    self.fanout.ms_per_kb.restore_baseline(value);
                }
                ("QueueThroughput", "bytes_per_sec") => {
                    self.fanout.throughput.restore_baseline(value);
                }
                // QueueDlqDepth is persist:false (a delta-rate, not a level
                // baseline) — DATA-10: even a stray hand-edited row is ignored,
                // never loaded as a DLQ baseline.
                _ => {} // Unknown / non-persisted metric/field — ignore.
            }
        }

        tracing::info!("Reloaded {count} switchboard baseline value(s) from control_baseline");
        Ok(())
    }
}

// ── Control fns (wired at daemon init; named fns, not closures, so dispatch
//    stays a bare fn-pointer — no vtable, §9 — and each is unit-testable).
//    Each feeds its sample through `ControlLane::update` (EWMA-smoothed at the
//    lane's immutable alphas), advancing BOTH the fast and slow lanes.

/// Control fn for [`MetricId::EmbedderLatency`]: EWMA-smooth `embed_ms` into the
/// embedder-latency lane.
pub fn store_embedder_latency(fanout: &ControlFanout, sample: &MetricSample) {
    if let MetricSample::EmbedderLatency { rec, model } = sample {
        fanout.embedder_latency.set_model(model);
        fanout.embedder_latency.update(rec.embed_ms as f64);
    }
}

/// Control fn for [`MetricId::QueueMsPerKb`]: EWMA-smooth the ms/KB ratio.
pub fn store_ms_per_kb(fanout: &ControlFanout, sample: &MetricSample) {
    if let MetricSample::QueueMsPerKb(value) = sample {
        fanout.ms_per_kb.update(*value);
    }
}

/// Control fn for [`MetricId::QueueThroughput`]: EWMA-smooth the bytes/s rate.
pub fn store_throughput(fanout: &ControlFanout, sample: &MetricSample) {
    if let MetricSample::QueueThroughput(value) = sample {
        fanout.throughput.update(*value);
    }
}

/// Control fn for [`MetricId::QueueDlqDepth`]: EWMA-smooth the DLQ delta-rate
/// (signed — a negative sample is a draining DLQ).
pub fn store_dlq_depth(fanout: &ControlFanout, sample: &MetricSample) {
    if let MetricSample::QueueDlqDepth(value) = sample {
        fanout.dlq_depth.update(*value);
    }
}

/// Build the scalar `MetricSample` for a scalar id, or `None` for a record id.
#[inline]
fn scalar_sample(id: MetricId, value: f64) -> Option<MetricSample> {
    match id {
        MetricId::QueueMsPerKb => Some(MetricSample::QueueMsPerKb(value)),
        MetricId::QueueDlqDepth => Some(MetricSample::QueueDlqDepth(value)),
        MetricId::QueueThroughput => Some(MetricSample::QueueThroughput(value)),
        MetricId::EmbedderLatency => None, // record-shaped — use emit_record
        MetricId::EmbedderBatch => None,   // record-shaped — use emit_embedder_batch
    }
}

#[cfg(test)]
mod tests;

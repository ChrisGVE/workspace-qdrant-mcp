//! Control fanout — the always-on, lock-free side of the switchboard.
//!
//! `ControlFanout` holds one `Arc<AtomicU64>` per control metric per lane
//! (fast/slow for dual-EWMA). The control fn-pointer (`routing.rs`) stores an
//! `f64` into a lane via `f64::to_bits()`; the read side recovers it with
//! `f64::from_bits()`. The `Arc`s are cloned at init and shared with the
//! `EwmaState` that owns the read side (queue-health work) — that side calls
//! `load(Acquire)` directly, never through the switchboard.
//!
//! `read_fast`/`read_slow` return `Option<f64>`: `Some` for control ids, `None`
//! for non-control ids — no silent `0.0` fallthrough (arch §5f, read-F4).
//!
//! Adding a control metric requires a new field pair AND a new `read_*` arm.
//! This is a documented init-time checklist item, NOT compile-enforced: a
//! missing field simply yields `None`. The cardinality is tiny (one pair per
//! control metric), so the manual step is acceptable per the simplicity
//! guardrail.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::MetricId;

/// One `Arc<AtomicU64>` per control metric per lane. Fast lane = live signal
/// (transient); slow lane = EWMA accumulator (persisted to `control_baseline`).
pub struct ControlFanout {
    pub embedder_latency_fast: Arc<AtomicU64>,
    pub embedder_latency_slow: Arc<AtomicU64>,
    // One pair per additional control metric goes here (+ a `read_*` arm below).
}

impl ControlFanout {
    pub fn new() -> Self {
        Self {
            embedder_latency_fast: Arc::new(AtomicU64::new(0)),
            embedder_latency_slow: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Live fast-lane value for a control id, or `None` for a non-control id.
    pub fn read_fast(&self, id: MetricId) -> Option<f64> {
        match id {
            MetricId::EmbedderLatency => Some(f64::from_bits(
                self.embedder_latency_fast.load(Ordering::Acquire),
            )),
            _ => None,
        }
    }

    /// Persisted slow-lane value for a control id, or `None` for a non-control id.
    pub fn read_slow(&self, id: MetricId) -> Option<f64> {
        match id {
            MetricId::EmbedderLatency => Some(f64::from_bits(
                self.embedder_latency_slow.load(Ordering::Acquire),
            )),
            _ => None,
        }
    }
}

impl Default for ControlFanout {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_load_roundtrip() {
        let f = ControlFanout::new();
        let v = 123.456_f64;
        f.embedder_latency_fast
            .store(v.to_bits(), Ordering::Release);
        assert!((f.read_fast(MetricId::EmbedderLatency).unwrap() - v).abs() < 1e-10);
    }

    #[test]
    fn test_non_control_id_returns_none() {
        let f = ControlFanout::new();
        assert!(f.read_fast(MetricId::QueueItemMs).is_none());
        assert!(f.read_slow(MetricId::QueueThroughput).is_none());
    }
}

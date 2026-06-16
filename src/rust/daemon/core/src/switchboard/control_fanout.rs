//! Control fanout — the always-on, lock-free side of the switchboard.
//!
//! `ControlFanout` holds one [`Arc<ControlLane>`] per control metric. A control
//! fn (`routing.rs`) feeds a sample into a lane via [`ControlLane::update`]
//! (EWMA-smoothed, dual-rate); the lane is the smoothed [`EwmaLane`] plus its two
//! immutable alphas. The same `Arc<ControlLane>`s are cloned at init and shared
//! with the `EwmaState` that owns the verdict read side (queue-health) — so the
//! lane an emit advances is exactly the lane the verdict snapshots (the #133 F1
//! single-source handshake). The verdict reads through its **cloned lanes**, not
//! the fanout; the fanout's only read API is [`slow_value`](ControlFanout::slow_value),
//! the slow-lane scalar the persist task flushes.
//!
//! Adding a control metric requires a new field AND a new `slow_value` arm. This
//! is a documented init-time checklist item, NOT compile-enforced: a missing
//! field simply yields `None`. The cardinality is tiny (one lane per control
//! metric), so the manual step is acceptable per the simplicity guardrail.

use std::sync::Arc;

use super::control_lane::ControlLane;
use super::MetricId;
use crate::config::queue_health::QueueHealthConfig;

/// One [`Arc<ControlLane>`] per control metric. Each lane carries a fast lane
/// (live signal) and a slow lane (EWMA baseline, persisted to `control_baseline`
/// for the `persist: true` ids).
pub struct ControlFanout {
    /// Embedding-call latency (ms) — ingestion stage-3 control feed.
    pub embedder_latency: Arc<ControlLane>,
    /// Per-byte processing cost (ms/KB) — A1 trend signal.
    pub ms_per_kb: Arc<ControlLane>,
    /// Drain throughput (bytes/s) — F5 drain-budget denominator.
    pub throughput: Arc<ControlLane>,
    /// Dead-letter-queue depth delta-rate — A3 signal (not persisted).
    pub dlq_depth: Arc<ControlLane>,
}

impl ControlFanout {
    /// Build the fanout, constructing every lane with the configured EWMA alphas.
    /// The alphas are immutable for the lane's lifetime (captured here).
    pub fn new(cfg: &QueueHealthConfig) -> Self {
        let lane = || Arc::new(ControlLane::new(cfg.fast_alpha, cfg.slow_alpha));
        Self {
            embedder_latency: lane(),
            ms_per_kb: lane(),
            throughput: lane(),
            dlq_depth: lane(),
        }
    }

    /// Persisted slow-lane (baseline) value for a control id, or `None` for a
    /// non-control id (`EmbedderBatch`). The persist task's only read API
    /// (ARCH-02, PERF-05); the verdict reads via its own cloned lanes instead.
    pub fn slow_value(&self, id: MetricId) -> Option<f64> {
        match id {
            MetricId::EmbedderLatency => Some(self.embedder_latency.read_slow()),
            MetricId::QueueMsPerKb => Some(self.ms_per_kb.read_slow()),
            MetricId::QueueThroughput => Some(self.throughput.read_slow()),
            MetricId::QueueDlqDepth => Some(self.dlq_depth.read_slow()),
            MetricId::EmbedderBatch => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> QueueHealthConfig {
        QueueHealthConfig::default()
    }

    #[test]
    fn test_fanout_construction_uses_config_alphas() {
        let c = cfg();
        let f = ControlFanout::new(&c);
        for lane in [
            &f.embedder_latency,
            &f.ms_per_kb,
            &f.throughput,
            &f.dlq_depth,
        ] {
            assert_eq!(lane.alphas(), (c.fast_alpha, c.slow_alpha));
        }
    }

    #[test]
    fn test_slow_value_routes_by_id() {
        let f = ControlFanout::new(&cfg());
        // First update seeds both lanes to the sample, so the slow lane reads it.
        f.ms_per_kb.update(3.5);
        assert_eq!(f.slow_value(MetricId::QueueMsPerKb), Some(3.5));
        f.embedder_latency.update(120.0);
        assert_eq!(f.slow_value(MetricId::EmbedderLatency), Some(120.0));
    }

    #[test]
    fn test_slow_value_non_control_id_is_none() {
        let f = ControlFanout::new(&cfg());
        assert!(f.slow_value(MetricId::EmbedderBatch).is_none());
    }
}

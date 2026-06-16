//! Drain-budget probe (#133 F5).
//!
//! Estimates time-to-empty from the pending-bytes snapshot and the **slow**
//! throughput lane (the stable long-run drain rate — using the fast lane would
//! make the eta jitter on every burst, DOM-11). Amber when the backlog would
//! take longer than `drain_budget_secs` to clear, or when there is a backlog but
//! throughput has decayed to ≈0 (cannot estimate). Green otherwise — including
//! the cold-start, unseeded-throughput, and stale-snapshot cases (insufficient
//! data must not raise an alarm, §6.4).

use std::time::Duration;

use super::{ProbeResult, DRAIN};
use crate::config::queue_health::QueueHealthConfig;
use crate::queue_health::state::EwmaState;

/// A drain rate (bytes/sec) at or below this is treated as zero, so the eta is
/// never computed by dividing by a near-zero rate.
const NEAR_ZERO_RATE: f64 = 1e-9;

/// Evaluate the drain-budget probe. See the module docs for the RAG rules.
pub fn drain_budget(state: &EwmaState, cfg: &QueueHealthConfig) -> ProbeResult {
    let Some(snap) = state.drain_snapshot() else {
        return ProbeResult::green(DRAIN); // no sample yet — cold start.
    };
    if snap.sampled_at.elapsed() > Duration::from_secs(cfg.drain_snapshot_max_age_secs) {
        return ProbeResult::green(DRAIN); // stale poll loop — insufficient data (SEC-05).
    }

    let throughput = state.throughput_snapshot();
    if !throughput.seeded {
        return ProbeResult::green(DRAIN); // fresh daemon — no drain rate learned yet.
    }

    let pending = snap.pending_bytes;
    if pending == 0 {
        return ProbeResult::green(DRAIN);
    }

    let rate = throughput.baseline(); // slow-lane bytes/sec (DOM-11).
    if rate <= NEAR_ZERO_RATE {
        // Backlog exists but throughput has decayed to ≈0 — cannot estimate.
        return ProbeResult::amber(
            DRAIN,
            "Indexing has stalled with a backlog pending; \
             throughput has dropped to near zero.",
        );
    }

    let eta_secs = pending as f64 / rate;
    if eta_secs > cfg.drain_budget_secs as f64 {
        ProbeResult::amber(
            DRAIN,
            "At the current rate the backlog will take over a day to clear; \
             consider pausing new ingestion or adding resources.",
        )
    } else {
        ProbeResult::green(DRAIN)
    }
}

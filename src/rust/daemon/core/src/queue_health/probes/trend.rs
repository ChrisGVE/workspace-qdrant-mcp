//! Family A — EWMA trend / regression probes (#133 F3).
//!
//! Each trend probe reads one [`DualEwma`](crate::queue_health::ewma::DualEwma)
//! lane snapshot and compares the responsive *fast* lane to the *slow* baseline.
//! A1/A2 fire **Amber** when `fast/slow` exceeds a regression ratio; A3 (DLQ) is
//! a delta-rate probe firing **Red** on growth and **Amber** when stuck.
//!
//! These probes never reject the outlier *sample* — a spike still enters the fast
//! lane. What suppresses a one-poll verdict flip is the **debounce** applied to
//! the returned RAG by the poll loop (plurality over the window), not any
//! filtering here (DOM-12).

use super::{ProbeResult, DLQ, EMBEDDER, PROCESSING};
use crate::config::queue_health::QueueHealthConfig;
use crate::queue_health::ewma::DualEwma;
use crate::queue_health::state::EwmaState;

/// A1 — per-byte processing-cost (ms/KB) regression.
///
/// Amber when the smoothed ms/KB cost runs `regression_ratio`× over baseline.
/// Green while unseeded (insufficient data) or while the baseline is below
/// `ms_per_kb_floor` (too fast to matter — never divide by a near-zero baseline,
/// DOM-05).
pub fn a1_ms_per_kb(state: &EwmaState, cfg: &QueueHealthConfig) -> ProbeResult {
    evaluate_regression(
        PROCESSING,
        &state.ms_per_kb_snapshot(),
        cfg.regression_ratio,
        cfg.ms_per_kb_floor,
        |r| {
            format!(
                "Per-byte processing cost is {r:.1}× its baseline; \
                 check for large or pathological files in the queue."
            )
        },
    )
}

/// A2 — embedder-latency regression. Identical shape to A1 with the embedder
/// ratio and floor.
pub fn a2_embedder_latency(state: &EwmaState, cfg: &QueueHealthConfig) -> ProbeResult {
    evaluate_regression(
        EMBEDDER,
        &state.embedder_latency_snapshot(),
        cfg.embedder_ratio,
        cfg.embedder_latency_floor,
        |r| {
            format!(
                "Embedding latency is {r:.1}× its baseline; \
                 the embedding provider may be slow or overloaded."
            )
        },
    )
}

/// Shared A1/A2 body: Green unless seeded AND baseline ≥ floor AND
/// `fast/slow > ratio_threshold`, in which case Amber with the formatted message.
fn evaluate_regression(
    culprit: &'static str,
    snap: &DualEwma,
    ratio_threshold: f64,
    floor: f64,
    message: impl Fn(f64) -> String,
) -> ProbeResult {
    if !snap.seeded || snap.baseline() < floor {
        return ProbeResult::green(culprit);
    }
    match snap.ratio() {
        Some(r) if r > ratio_threshold => ProbeResult::amber(culprit, message(r)),
        _ => ProbeResult::green(culprit),
    }
}

/// A3 — DLQ depth trend (delta-rate based, DOM-01/02/03).
///
/// The lane is fed the **per-poll delta** `(count_now − prev_dlq)`, so its fast
/// lane is a smoothed *rate* (counts/poll). Emptiness is tested on the live
/// `count_now`, never the lagging EWMA baseline.
///
/// - `count_now < dlq_empty_eps` ⇒ Green (empty), regardless of rate.
/// - fewer than 2 delta samples ⇒ Green (one sample is not a trend; the first
///   post-restart poll only seeds the rate lane — DOM-02).
/// - `rate > dlq_rate_band` ⇒ Red (backlog growing).
/// - `rate < −dlq_rate_band` ⇒ Green (draining).
/// - otherwise (flat, non-empty) ⇒ Amber (stuck — not draining).
///
/// `samples_seen` is the number of DLQ delta samples fed so far, tracked by the
/// poll loop; it gates the ≥2-sample rule independently of lane seeding (the lane
/// is seeded after one sample).
pub fn a3_dlq_trend(
    state: &EwmaState,
    cfg: &QueueHealthConfig,
    absolute_dlq_count: u64,
    samples_seen: u64,
) -> ProbeResult {
    // Emptiness uses the live ABSOLUTE count, never the smoothed delta-rate EWMA.
    if absolute_dlq_count < cfg.dlq_empty_eps {
        return ProbeResult::green(DLQ);
    }
    if samples_seen < 2 {
        return ProbeResult::green(DLQ);
    }
    let rate = state.dlq_depth_snapshot().fast;
    if rate > cfg.dlq_rate_band {
        ProbeResult::red(
            DLQ,
            "Dead-letter backlog is growing; inspect failing items with 'wqm dlq list'.",
        )
    } else if rate < -cfg.dlq_rate_band {
        ProbeResult::green(DLQ) // draining
    } else {
        ProbeResult::amber(
            DLQ,
            "Dead-letter backlog is stuck (not draining); items may need manual retry or purge.",
        )
    }
}

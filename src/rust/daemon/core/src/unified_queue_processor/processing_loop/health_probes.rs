//! Per-poll queue-health feed (#133 F2b/F3/F4/F5).
//!
//! Extracted from `loop_core.rs` to keep that file within the codesize limit.
//! Called once per poll cycle by `run_poll_cycle`.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::queue_health::probes::hard_state::{b4_all_failing, PollOutcome};
use crate::queue_health::probes::trend::{a1_ms_per_kb, a2_embedder_latency, a3_dlq_trend};
use crate::queue_health::probes::ProbeResult;
use crate::queue_health::{EwmaState, QueueProcessorHealth};
use crate::queue_operations::QueueManager;
use crate::switchboard::{switchboard, MetricId};

use super::loop_state::LoopState;

/// Once per poll: emit the signed DLQ delta-rate (A3) and an idle-zero throughput
/// sample (F5), refresh the drain snapshot (F5), push the B4 outcome ring and
/// store its predicate (lock-free), then evaluate + debounce + cache the trend
/// probes (A1/A2/A3) for the on-RPC verdict. No-op until both the EWMA state and
/// processor health are wired.
pub(super) async fn update_health_probes(
    queue_manager: &QueueManager,
    queue_health: &Option<Arc<QueueProcessorHealth>>,
    ewma_state: &Option<Arc<EwmaState>>,
    state: &mut LoopState,
) {
    let (Some(ewma), Some(health)) = (ewma_state.as_ref(), queue_health.as_ref()) else {
        return;
    };
    let cfg = ewma.config();

    // DLQ delta-rate (A3). DOM-09: seed prev_dlq from the live count on the first
    // poll so a restart with a static backlog feeds delta≈0, not the whole backlog.
    let dlq_count = queue_manager.count_dlq().await.unwrap_or(0);
    let prev = state.prev_dlq.unwrap_or(dlq_count);
    let delta = dlq_count as f64 - prev as f64;
    state.prev_dlq = Some(dlq_count);
    state.dlq_samples_seen = state.dlq_samples_seen.saturating_add(1);

    // Pending-bytes drain snapshot (F5).
    let pending_bytes = queue_manager
        .get_pending_bytes_estimate(cfg.default_item_bytes)
        .await
        .unwrap_or(0);
    ewma.set_drain_snapshot(pending_bytes);

    if let Some(sw) = switchboard() {
        sw.emit(sw.handle(MetricId::QueueDlqDepth, "queue"), delta);
        if should_emit_idle_throughput(pending_bytes, state.last_poll_dispatched) {
            sw.emit(sw.handle(MetricId::QueueThroughput, "queue"), 0.0);
        }
    }

    // B4 outcome ring + predicate (kept poll-local; verdict reads the atomic).
    // Relaxed loads: the ring is poll-local and read within this call only, and
    // the B4 predicate tolerates a one-poll-stale counter (it is debounced).
    let items_processed = health.items_processed.load(Ordering::Relaxed);
    let items_failed = health.items_failed.load(Ordering::Relaxed);
    state.outcome_ring.push_back(PollOutcome {
        items_processed,
        dlq_count,
        attempts: items_processed + items_failed,
    });
    // window + 1 entries ⇒ first..last spans exactly `all_failing_window` polls.
    let cap = cfg.all_failing_window + 1;
    while state.outcome_ring.len() > cap {
        state.outcome_ring.pop_front();
    }
    // `make_contiguous` yields a slice from the ring's own buffer — no Vec alloc.
    ewma.set_all_failing(b4_all_failing(state.outcome_ring.make_contiguous()));

    // Trend probes: evaluate raw, debounce via observe, cache for the verdict.
    let raw = [
        a1_ms_per_kb(ewma, cfg),
        a2_embedder_latency(ewma, cfg),
        a3_dlq_trend(ewma, cfg, dlq_count, state.dlq_samples_seen),
    ];
    let debounced: Vec<ProbeResult> = raw
        .into_iter()
        .map(|r| {
            let rag = ewma.observe(r.culprit, r.rag);
            ProbeResult { rag, ..r }
        })
        .collect();
    ewma.set_trend_cache(debounced);
}

/// Decide whether this poll should feed the throughput slow lane a zero sample.
///
/// A zero is emitted whenever no work is moving, so the lane decays toward zero
/// instead of holding a stale healthy rate. Two situations qualify:
///
///   - **idle** — there is no backlog (`pending_bytes == 0`). The original F5
///     DOM-06 case: the queue is genuinely drained.
///   - **stalled** — a backlog remains (`pending_bytes > 0`) yet the previous
///     poll dispatched no batch (`!last_poll_dispatched`). This is the
///     circuit-breaker-open case (Qdrant down / SQLite busy / memory pressure, or
///     a breaker tripping mid-batch): work cannot drain, so the rate must fall to
///     zero. Without it the F5 drain probe divides the backlog by a frozen rate
///     and reports a falsely-Green ETA while nothing happens (#144).
///
/// `last_poll_dispatched` carries a one-poll lag (this probe runs pre-dequeue),
/// which the #133 design accepts. The healthy path — a backlog that IS draining,
/// where `last_poll_dispatched` is true — emits no zero, so the live throughput
/// samples and the B1/B3 hard-state probes are untouched.
fn should_emit_idle_throughput(pending_bytes: u64, last_poll_dispatched: bool) -> bool {
    pending_bytes == 0 || !last_poll_dispatched
}

#[cfg(test)]
mod tests {
    use super::should_emit_idle_throughput;

    #[test]
    fn idle_queue_emits_zero() {
        // No backlog at all — the original F5 DOM-06 idle-zero, dispatch state
        // irrelevant.
        assert!(should_emit_idle_throughput(0, true));
        assert!(should_emit_idle_throughput(0, false));
    }

    #[test]
    fn draining_backlog_emits_no_zero() {
        // Backlog present AND the last poll dispatched a batch: work is moving, so
        // the live throughput samples must stand (B1/B3 preserved) — #144 must not
        // zero a healthily-draining lane.
        assert!(!should_emit_idle_throughput(4096, true));
    }

    #[test]
    fn stalled_backlog_emits_zero() {
        // The #144 bug: a backlog remains but the last poll dispatched nothing
        // (circuit breaker open). The lane must be zeroed so the drain probe does
        // not divide by a frozen rate and report a falsely-Green ETA.
        assert!(should_emit_idle_throughput(4096, false));
    }
}

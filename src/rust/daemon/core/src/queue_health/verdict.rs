//! Overall queue-health verdict (#133 F6).
//!
//! [`verdict`] combines every probe into one [`HealthVerdict`] using **worst-of**
//! aggregation ([`Rag::severity`]) — the single most-severe probe sets the
//! overall color, with no weighting. Every non-green probe's remediation is
//! surfaced (the surfaces decide how much to show).
//!
//! ## What the verdict reads
//!
//! The flap-prone **trend** probes (A1/A2/A3) are evaluated and debounced by the
//! poll loop, which caches their debounced [`ProbeResult`]s on the [`EwmaState`]
//! (poll-cadence write, RPC read — the same pattern as the drain snapshot, §7).
//! `verdict()` reads that cache; it never calls `observe`, so it acquires no
//! debounce lock (PERF-04). The **hard-state** probes are computed here from
//! caller-supplied inputs — B1 from the live reachability bool, B2 from the disk
//! bytes, B3 from the health atomics, B4 from the lock-free `all_failing` atomic —
//! and the drain probe from the cached snapshot.
//!
//! ## Cold start (UX-3 / §6.4)
//!
//! When **no** lane has been seeded (fresh daemon), `cold_start = true`. With all
//! probes Green the overall is Green, but the surface renders this as
//! "unknown / learning baseline" rather than "healthy", so absence-of-evidence is
//! never mistaken for evidence-of-health. A hard failure (e.g. Qdrant down) still
//! surfaces as Red even during cold start, because worst-of includes family B.

use super::probes::{drain, hard_state, ProbeResult};
use super::state::{EwmaState, Rag};
use super::QueueProcessorHealth;
use crate::config::queue_health::QueueHealthConfig;

/// The aggregated queue-health verdict: overall RAG, the cold-start flag, and
/// every probe's result (for per-line attributed remediation).
#[derive(Debug, Clone)]
pub struct HealthVerdict {
    /// Worst-of all probe RAGs.
    pub overall: Rag,
    /// True when no lane is seeded yet (learning baseline); rendered as "unknown".
    pub cold_start: bool,
    /// Every probe's result, in evaluation order.
    pub probes: Vec<ProbeResult>,
}

impl HealthVerdict {
    /// Aggregate probe results into a verdict via worst-of severity. Pure — the
    /// unit-test entry point for the aggregation rule, independent of any state.
    pub fn from_probes(probes: Vec<ProbeResult>, cold_start: bool) -> Self {
        let overall = probes
            .iter()
            .map(|p| p.rag)
            .max_by_key(|r| r.severity())
            .unwrap_or(Rag::Green);
        Self {
            overall,
            cold_start,
            probes,
        }
    }

    /// The non-green probes (those carrying remediation), in order.
    pub fn degraded(&self) -> impl Iterator<Item = &ProbeResult> {
        self.probes.iter().filter(|p| p.rag != Rag::Green)
    }
}

/// Build the overall verdict from the poll-debounced trend cache plus the
/// caller-supplied hard-state inputs. See the module docs for the read model.
///
/// - `qdrant_reachable` — B1 input: the result of the bounded live
///   `test_connection()` check (timeout/error ⇒ `false`).
/// - `disk_free` / `disk_total` — B2 input: bytes on the database volume
///   (`None` free ⇒ the disk probe stays Green).
pub fn verdict(
    state: &EwmaState,
    health: &QueueProcessorHealth,
    cfg: &QueueHealthConfig,
    qdrant_reachable: bool,
    disk_free: Option<u64>,
    disk_total: Option<u64>,
) -> HealthVerdict {
    let mut probes = state.trend_cache(); // A1/A2/A3, already debounced by the poll loop.
    probes.push(hard_state::b1_qdrant(qdrant_reachable));
    probes.push(hard_state::b2_disk(disk_free, disk_total, cfg));
    probes.push(hard_state::b3_stall(health, cfg));
    probes.push(hard_state::b4_result(state.all_failing()));
    probes.push(drain::drain_budget(state, cfg));

    let cold_start = !state.any_lane_seeded();
    HealthVerdict::from_probes(probes, cold_start)
}

#[cfg(test)]
#[path = "verdict_tests.rs"]
mod tests;

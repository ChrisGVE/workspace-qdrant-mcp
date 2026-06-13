//! Side-car live health state — `EwmaState` (#133 F2/§4.2).
//!
//! `EwmaState` is a second shared handle alongside `Arc<QueueProcessorHealth>`,
//! created once in `queue_init.rs` and injected into BOTH the processor (which
//! updates the lanes per item / per poll) and the gRPC service (which snapshots
//! them for the health verdict). It is NOT embedded in `QueueProcessorHealth`,
//! which stays pure lock-free atomics.
//!
//! Concurrency contract (PERF-01): the four EWMA lanes are atomics-backed
//! ([`EwmaLane`]) and updated `Relaxed` last-writer-wins on the per-item hot
//! loop — no hot mutex. The only `Mutex` guards the per-probe debounce rings,
//! which are touched at *poll cadence* only (never per item), so it is
//! uncontended on the hot path. The drain snapshot is a small `RwLock`, also
//! written at poll cadence and read by the Health RPC.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Mutex, RwLock};
use std::time::Instant;

use crate::config::queue_health::QueueHealthConfig;

use super::ewma::{DualEwma, EwmaLane};

/// Health Red/Amber/Green status of a probe or the overall verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rag {
    /// Healthy.
    Green,
    /// Degraded — actionable but not failing.
    Amber,
    /// Unhealthy — failing now.
    Red,
}

impl Rag {
    /// Severity rank for worst-of and conservative tie-breaking (Red > Amber > Green).
    pub fn severity(self) -> u8 {
        match self {
            Rag::Green => 0,
            Rag::Amber => 1,
            Rag::Red => 2,
        }
    }
}

/// Immutable EWMA smoothing factors, taken from [`QueueHealthConfig`] at
/// construction.
#[derive(Debug, Clone, Copy)]
pub struct EwmaAlphas {
    /// Fast-lane smoothing factor.
    pub fast: f64,
    /// Slow-lane smoothing factor.
    pub slow: f64,
}

/// Poll-loop-sampled pending-bytes snapshot for the drain probe.
///
/// `sampled_at` exists so the drain probe can detect a frozen poll loop and
/// report insufficient-data rather than surfacing a stale backlog as current
/// (SEC-05).
#[derive(Debug, Clone, Copy)]
pub struct DrainSnapshot {
    /// Estimated pending bytes at sample time.
    pub pending_bytes: u64,
    /// When the sample was taken (monotonic clock).
    pub sampled_at: Instant,
}

/// Per-probe sliding windows of recent RAG verdicts, for debounce.
///
/// Each probe has its own ring of the last `window` raw verdicts. The debounced
/// value is the plurality (most frequent) RAG; ties break toward the more severe
/// status (Red > Amber > Green), the conservative choice. Updated at poll
/// cadence only.
#[derive(Debug)]
pub struct DebounceRings {
    window: usize,
    rings: HashMap<&'static str, VecDeque<Rag>>,
}

impl DebounceRings {
    fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            rings: HashMap::new(),
        }
    }

    /// Record a raw verdict for `probe` and return the debounced (plurality) RAG
    /// over the current window.
    pub fn observe(&mut self, probe: &'static str, rag: Rag) -> Rag {
        let window = self.window;
        let ring = self
            .rings
            .entry(probe)
            .or_insert_with(|| VecDeque::with_capacity(window));
        if ring.len() == window {
            ring.pop_front();
        }
        ring.push_back(rag);
        Self::plurality(ring)
    }

    /// Plurality RAG in the ring; ties break toward higher severity.
    fn plurality(ring: &VecDeque<Rag>) -> Rag {
        let mut counts = [0u32; 3]; // indexed by Rag::severity()
        for &r in ring {
            counts[r.severity() as usize] += 1;
        }
        // Walk most-severe-first so a tie resolves to the more severe status.
        let order = [Rag::Red, Rag::Amber, Rag::Green];
        let mut best = Rag::Green;
        let mut best_count = 0u32;
        for r in order {
            let c = counts[r.severity() as usize];
            if c > best_count {
                best = r;
                best_count = c;
            }
        }
        best
    }
}

/// Atomics-backed live EWMA state shared between the processor and the gRPC
/// service.
#[derive(Debug)]
pub struct EwmaState {
    ms_per_kb: EwmaLane,
    embedder_latency: EwmaLane,
    throughput_bytes_per_sec: EwmaLane,
    /// Re-seeds each poll from a fresh DLQ count; NOT persisted.
    dlq_depth: EwmaLane,
    alphas: EwmaAlphas,
    drain_snapshot: RwLock<Option<DrainSnapshot>>,
    debounce: Mutex<DebounceRings>,
}

impl EwmaState {
    /// Construct from config, taking the (immutable) smoothing factors and the
    /// debounce window.
    pub fn new(cfg: &QueueHealthConfig) -> Self {
        Self {
            ms_per_kb: EwmaLane::new(),
            embedder_latency: EwmaLane::new(),
            throughput_bytes_per_sec: EwmaLane::new(),
            dlq_depth: EwmaLane::new(),
            alphas: EwmaAlphas {
                fast: cfg.fast_alpha,
                slow: cfg.slow_alpha,
            },
            drain_snapshot: RwLock::new(None),
            debounce: Mutex::new(DebounceRings::new(cfg.debounce_window)),
        }
    }

    /// The configured smoothing factors.
    pub fn alphas(&self) -> EwmaAlphas {
        self.alphas
    }

    // ── Per-item / per-poll lane updates (Relaxed) ──────────────────────────

    /// Feed a ms/KB processing-cost sample (per-item success path).
    pub fn update_ms_per_kb(&self, sample: f64) {
        self.ms_per_kb
            .update(sample, self.alphas.fast, self.alphas.slow);
    }

    /// Feed an embedder-latency sample (ms).
    pub fn update_embedder_latency(&self, sample: f64) {
        self.embedder_latency
            .update(sample, self.alphas.fast, self.alphas.slow);
    }

    /// Feed a throughput sample (bytes/sec).
    pub fn update_throughput(&self, sample: f64) {
        self.throughput_bytes_per_sec
            .update(sample, self.alphas.fast, self.alphas.slow);
    }

    /// Feed a DLQ-depth sample (per-poll).
    pub fn update_dlq_depth(&self, sample: f64) {
        self.dlq_depth
            .update(sample, self.alphas.fast, self.alphas.slow);
    }

    // ── Snapshots for the verdict path (Relaxed loads) ──────────────────────

    /// Snapshot the ms/KB lane.
    pub fn ms_per_kb_snapshot(&self) -> DualEwma {
        DualEwma::from_atomics(&self.ms_per_kb, self.alphas.fast, self.alphas.slow)
    }

    /// Snapshot the embedder-latency lane.
    pub fn embedder_latency_snapshot(&self) -> DualEwma {
        DualEwma::from_atomics(&self.embedder_latency, self.alphas.fast, self.alphas.slow)
    }

    /// Snapshot the throughput lane.
    pub fn throughput_snapshot(&self) -> DualEwma {
        DualEwma::from_atomics(
            &self.throughput_bytes_per_sec,
            self.alphas.fast,
            self.alphas.slow,
        )
    }

    /// Snapshot the DLQ-depth lane.
    pub fn dlq_depth_snapshot(&self) -> DualEwma {
        DualEwma::from_atomics(&self.dlq_depth, self.alphas.fast, self.alphas.slow)
    }

    // ── Drain snapshot cache ────────────────────────────────────────────────

    /// Store a fresh pending-bytes sample, stamping it with the current time.
    pub fn set_drain_snapshot(&self, pending_bytes: u64) {
        let snap = DrainSnapshot {
            pending_bytes,
            sampled_at: Instant::now(),
        };
        if let Ok(mut guard) = self.drain_snapshot.write() {
            *guard = Some(snap);
        }
    }

    /// Read the cached pending-bytes snapshot, if any.
    pub fn drain_snapshot(&self) -> Option<DrainSnapshot> {
        self.drain_snapshot.read().ok().and_then(|g| *g)
    }

    // ── Debounce ────────────────────────────────────────────────────────────

    /// Record a raw probe verdict and return the debounced (plurality) RAG.
    pub fn observe(&self, probe: &'static str, rag: Rag) -> Rag {
        self.debounce
            .lock()
            .map(|mut d| d.observe(probe, rag))
            .unwrap_or(rag)
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;

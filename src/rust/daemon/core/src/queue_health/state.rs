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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use crate::config::queue_health::QueueHealthConfig;
use crate::switchboard::control_lane::ControlLane;
use crate::switchboard::ControlFanout;

use super::ewma::DualEwma;
use super::probes::ProbeResult;

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
    ms_per_kb: Arc<ControlLane>,
    embedder_latency: Arc<ControlLane>,
    throughput_bytes_per_sec: Arc<ControlLane>,
    /// Re-seeds each poll from a fresh DLQ count; NOT persisted.
    dlq_depth: Arc<ControlLane>,
    /// ms/KB size floor (bytes): known-size items below this are treated as this
    /// size so a tiny file's fixed per-item overhead can't masquerade as a huge
    /// ms/KB outlier (DOM-07). Captured from config at construction.
    min_item_bytes: u64,
    drain_snapshot: RwLock<Option<DrainSnapshot>>,
    debounce: Mutex<DebounceRings>,
    /// The poll-loop-debounced trend probes (A1/A2/A3), cached for the on-RPC
    /// verdict to read without touching the debounce `Mutex` (PERF-04). Poll
    /// cadence write, RPC read — the same uncontended pattern as `drain_snapshot`.
    trend_cache: RwLock<Vec<ProbeResult>>,
    /// B4 all-items-failing predicate, precomputed by the poll loop from its
    /// poll-local outcome ring and read lock-free by the verdict (PERF-08).
    all_failing: AtomicBool,
}

impl EwmaState {
    /// Construct a standalone `EwmaState` with freshly-allocated lanes carrying
    /// the config alphas. **Test-only path** — production uses [`from_fanout`]
    /// (Self::from_fanout) so the lanes are the same `Arc<ControlLane>`s the
    /// switchboard's control fns feed. A standalone state's lanes are never fed
    /// by any emitter (the #133 bug), so it is only useful in unit tests that
    /// drive the lanes directly.
    pub fn new(cfg: &QueueHealthConfig) -> Self {
        let lane = || Arc::new(ControlLane::new(cfg.fast_alpha, cfg.slow_alpha));
        Self {
            ms_per_kb: lane(),
            embedder_latency: lane(),
            throughput_bytes_per_sec: lane(),
            dlq_depth: lane(),
            min_item_bytes: cfg.min_item_bytes,
            drain_snapshot: RwLock::new(None),
            debounce: Mutex::new(DebounceRings::new(cfg.debounce_window)),
            trend_cache: RwLock::new(Vec::new()),
            all_failing: AtomicBool::new(false),
        }
    }

    /// Construct by cloning the switchboard fanout's `Arc<ControlLane>`s — the
    /// production path (ARCH-03). The resulting state's lanes ARE the lanes the
    /// control fns advance on every emit, so the verdict snapshots exactly the
    /// values the emitters fed. Called at `queue_init.rs` after the switchboard
    /// is sealed.
    pub fn from_fanout(fanout: &ControlFanout, cfg: &QueueHealthConfig) -> Self {
        Self {
            ms_per_kb: Arc::clone(&fanout.ms_per_kb),
            embedder_latency: Arc::clone(&fanout.embedder_latency),
            throughput_bytes_per_sec: Arc::clone(&fanout.throughput),
            dlq_depth: Arc::clone(&fanout.dlq_depth),
            min_item_bytes: cfg.min_item_bytes,
            drain_snapshot: RwLock::new(None),
            debounce: Mutex::new(DebounceRings::new(cfg.debounce_window)),
            trend_cache: RwLock::new(Vec::new()),
            all_failing: AtomicBool::new(false),
        }
    }

    /// The configured smoothing factors (read off a lane — all four lanes share
    /// the same immutable alphas).
    pub fn alphas(&self) -> EwmaAlphas {
        let (fast, slow) = self.ms_per_kb.alphas();
        EwmaAlphas { fast, slow }
    }

    /// The ms/KB size floor in bytes (DOM-07 clamp).
    pub fn min_item_bytes(&self) -> u64 {
        self.min_item_bytes
    }

    // ── Per-item / per-poll lane updates (Relaxed) ──────────────────────────

    /// Feed a ms/KB processing-cost sample (per-item success path).
    pub fn update_ms_per_kb(&self, sample: f64) {
        self.ms_per_kb.update(sample);
    }

    /// Feed an embedder-latency sample (ms).
    pub fn update_embedder_latency(&self, sample: f64) {
        self.embedder_latency.update(sample);
    }

    /// Feed a throughput sample (bytes/sec).
    pub fn update_throughput(&self, sample: f64) {
        self.throughput_bytes_per_sec.update(sample);
    }

    /// Feed a DLQ-depth sample (per-poll).
    pub fn update_dlq_depth(&self, sample: f64) {
        self.dlq_depth.update(sample);
    }

    // ── Snapshots for the verdict path (Acquire-gated reads) ────────────────

    /// Snapshot the ms/KB lane.
    pub fn ms_per_kb_snapshot(&self) -> DualEwma {
        self.ms_per_kb.snapshot()
    }

    /// Snapshot the embedder-latency lane.
    pub fn embedder_latency_snapshot(&self) -> DualEwma {
        self.embedder_latency.snapshot()
    }

    /// Snapshot the throughput lane.
    pub fn throughput_snapshot(&self) -> DualEwma {
        self.throughput_bytes_per_sec.snapshot()
    }

    /// Snapshot the DLQ-depth lane.
    pub fn dlq_depth_snapshot(&self) -> DualEwma {
        self.dlq_depth.snapshot()
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

    /// Seed a drain snapshot with an explicit timestamp — test-only, so the
    /// staleness path can be exercised without sleeping.
    #[cfg(test)]
    pub fn set_drain_snapshot_at(&self, pending_bytes: u64, sampled_at: Instant) {
        if let Ok(mut guard) = self.drain_snapshot.write() {
            *guard = Some(DrainSnapshot {
                pending_bytes,
                sampled_at,
            });
        }
    }

    // ── Debounce ────────────────────────────────────────────────────────────

    /// Record a raw probe verdict and return the debounced (plurality) RAG.
    pub fn observe(&self, probe: &'static str, rag: Rag) -> Rag {
        self.debounce
            .lock()
            .map(|mut d| d.observe(probe, rag))
            .unwrap_or(rag)
    }

    // ── Verdict cache (poll-loop write, RPC read) ───────────────────────────

    /// Replace the cached debounced trend-probe results (poll loop, once per
    /// poll after `observe`-ing each trend probe).
    pub fn set_trend_cache(&self, probes: Vec<ProbeResult>) {
        if let Ok(mut guard) = self.trend_cache.write() {
            *guard = probes;
        }
    }

    /// Clone the cached debounced trend-probe results for the verdict. Empty
    /// before the first poll completes (handled as cold start by the verdict).
    pub fn trend_cache(&self) -> Vec<ProbeResult> {
        self.trend_cache
            .read()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Store the B4 all-items-failing predicate (poll loop).
    pub fn set_all_failing(&self, all_failing: bool) {
        self.all_failing.store(all_failing, Ordering::Relaxed);
    }

    /// Read the B4 all-items-failing predicate (verdict, lock-free).
    pub fn all_failing(&self) -> bool {
        self.all_failing.load(Ordering::Relaxed)
    }

    /// Hold the debounce `Mutex` for the duration of a test, to prove the verdict
    /// path never tries to `observe` (which would deadlock).
    #[cfg(test)]
    pub fn lock_debounce_for_test(&self) -> std::sync::MutexGuard<'_, DebounceRings> {
        self.debounce.lock().unwrap()
    }

    /// Whether any control lane has been seeded — false only on a fresh daemon
    /// that has taken no measurement yet (the cold-start condition, UX-3).
    pub fn any_lane_seeded(&self) -> bool {
        self.ms_per_kb.is_seeded()
            || self.embedder_latency.is_seeded()
            || self.throughput_bytes_per_sec.is_seeded()
            || self.dlq_depth.is_seeded()
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;

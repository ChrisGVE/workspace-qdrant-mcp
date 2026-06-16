//! `ControlLane` — the shared control representation (#133 F1, ARCH-07).
//!
//! One `ControlLane` is the unit the `ControlFanout` holds per control metric and
//! the unit `EwmaState` clones (`Arc<ControlLane>`). It is a thin wrapper that
//! pairs the already-shipped smoothed [`EwmaLane`] (the dual-rate EWMA atomics +
//! the verified torn-read-safe ordering) with the two **immutable** smoothing
//! factors that lane needs on every `update`.
//!
//! ## Why the alphas live here, not on the control fn
//!
//! The switchboard's [`ControlFn`](super::routing::ControlFn) is a bare
//! `fn(&ControlFanout, &MetricSample)` pointer — it cannot capture per-lane state
//! without becoming a boxed closure (which would reintroduce a vtable call and
//! break the §9 hot-path proof). So the alphas ride on the receiver: the control
//! fn reads them off the `ControlLane` it already reaches via `&ControlFanout`.
//! The function pointer stays monomorphic; the alphas are plain data.
//!
//! ## The single-source handshake
//!
//! Because `ControlFanout` and `EwmaState` hold clones of the **same**
//! `Arc<ControlLane>`, an emit's `update` and the verdict's `snapshot` touch the
//! identical atomics with the identical alphas — closing the #133 gap where the
//! emitter fed one lane set and the verdict read a different, never-fed one.

use crate::queue_health::ewma::{DualEwma, EwmaLane};

/// A control metric's live lane: the shared smoothed [`EwmaLane`] plus its two
/// immutable smoothing factors. `Send + Sync` (all state is atomics + `f64`).
#[derive(Debug)]
pub struct ControlLane {
    /// The atomics-backed dual-rate EWMA lane (unchanged smoothing math).
    lane: EwmaLane,
    /// Fast-lane smoothing factor, captured at construction.
    fast_alpha: f64,
    /// Slow-lane smoothing factor, captured at construction.
    slow_alpha: f64,
}

impl ControlLane {
    /// Create an unseeded control lane with the given (immutable) smoothing
    /// factors. Both EWMA lanes start at zero / `seeded = false`; the first
    /// `update` (or a `restore_baseline`) seeds them.
    pub fn new(fast_alpha: f64, slow_alpha: f64) -> Self {
        Self {
            lane: EwmaLane::new(),
            fast_alpha,
            slow_alpha,
        }
    }

    /// Feed one sample through the EWMA recurrence at this lane's alphas. This is
    /// the control-store hot path (called from a control fn per emit). Non-finite
    /// samples are dropped and the first sample seeds both lanes — see
    /// [`EwmaLane::update`].
    #[inline]
    pub fn update(&self, sample: f64) {
        self.lane.update(sample, self.fast_alpha, self.slow_alpha);
    }

    /// Snapshot the lane into a logical [`DualEwma`], carrying this lane's alphas
    /// so `ratio`/`slope` use the same smoothing the live lane was fed with.
    /// Torn-read-safe (reads `seeded` first; see [`DualEwma::from_atomics`]).
    #[inline]
    pub fn snapshot(&self) -> DualEwma {
        DualEwma::from_atomics(&self.lane, self.fast_alpha, self.slow_alpha)
    }

    /// Live fast-lane value (responsive signal).
    pub fn read_fast(&self) -> f64 {
        self.snapshot().fast
    }

    /// Baseline slow-lane value — what the persist task flushes.
    pub fn read_slow(&self) -> f64 {
        self.snapshot().slow
    }

    /// Whether the first sample (or a baseline restore) has seeded the lane.
    pub fn is_seeded(&self) -> bool {
        self.snapshot().seeded
    }

    /// Restore a persisted slow-lane baseline on daemon restart (F5/F10, DOM-06).
    ///
    /// Seeds BOTH lanes to `slow` and sets `seeded = true` (via
    /// [`DualEwma::restore_baseline`] then [`EwmaLane::store`]) so the next live
    /// `update` advances the restored baseline instead of taking the first-sample
    /// branch and overwriting it. Without this the persisted baseline is silently
    /// discarded on the first post-restart sample.
    pub fn restore_baseline(&self, slow: f64) {
        let mut d = DualEwma::new(self.fast_alpha, self.slow_alpha);
        d.restore_baseline(slow);
        self.lane.store(&d);
    }

    /// The (immutable) smoothing factors `(fast_alpha, slow_alpha)`.
    pub fn alphas(&self) -> (f64, f64) {
        (self.fast_alpha, self.slow_alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_seed_on_first_update() {
        let l = ControlLane::new(0.3, 0.01);
        assert!(!l.is_seeded());
        l.update(42.0);
        assert!(l.is_seeded());
        // First sample seeds both lanes to the sample value.
        assert!((l.read_fast() - 42.0).abs() < 1e-12);
        assert!((l.read_slow() - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_ewma_smoothing_second_sample() {
        let (fa, sa) = (0.3, 0.01);
        let l = ControlLane::new(fa, sa);
        l.update(100.0); // seed
        l.update(200.0); // EWMA step
        let exp_fast = fa * 200.0 + (1.0 - fa) * 100.0;
        let exp_slow = sa * 200.0 + (1.0 - sa) * 100.0;
        assert!((l.read_fast() - exp_fast).abs() < 1e-9);
        assert!((l.read_slow() - exp_slow).abs() < 1e-9);
    }

    #[test]
    fn test_restore_baseline_sets_seeded_and_both_lanes() {
        let l = ControlLane::new(0.3, 0.01);
        l.restore_baseline(7.5);
        assert!(l.is_seeded());
        assert!((l.read_fast() - 7.5).abs() < 1e-12);
        assert!((l.read_slow() - 7.5).abs() < 1e-12);
    }

    #[test]
    fn test_restore_baseline_survives_next_update() {
        // DOM-06: a live sample after restore must NOT overwrite the baseline.
        let (fa, sa) = (0.3, 0.01);
        let l = ControlLane::new(fa, sa);
        l.restore_baseline(100.0);
        l.update(200.0); // must take the EWMA branch, not first-sample seeding
        let exp_slow = sa * 200.0 + (1.0 - sa) * 100.0;
        assert!(
            (l.read_slow() - exp_slow).abs() < 1e-9,
            "restored baseline was overwritten: slow={}",
            l.read_slow()
        );
    }

    #[test]
    fn test_alphas_roundtrip() {
        let l = ControlLane::new(0.25, 0.02);
        assert_eq!(l.alphas(), (0.25, 0.02));
    }

    #[test]
    fn stress_test_control_lane_contention() {
        // F1 deterministic contention test (task 8): writers + readers on one
        // shared lane — no panics, no NaN from torn reads.
        let lane = Arc::new(ControlLane::new(0.1, 0.01));
        let iterations = 10_000u64;
        let writers = 4u64;
        let readers = 4u64;

        let mut handles = Vec::new();
        for i in 0..writers {
            let l = Arc::clone(&lane);
            handles.push(thread::spawn(move || {
                for j in 0..iterations {
                    l.update((i * iterations + j) as f64);
                }
            }));
        }
        for _ in 0..readers {
            let l = Arc::clone(&lane);
            handles.push(thread::spawn(move || {
                for _ in 0..iterations {
                    let snap = l.snapshot();
                    // A seeded lane must never expose a non-finite (torn) value.
                    assert!(snap.fast.is_finite() || !snap.seeded);
                    assert!(snap.slow.is_finite() || !snap.seeded);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert!(lane.is_seeded());
    }
}

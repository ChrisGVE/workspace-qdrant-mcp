//! Dual-rate EWMA primitive for queue-health trend detection (#133 F2).
//!
//! A [`DualEwma`] holds two exponentially-weighted moving averages of the same
//! signal at different smoothing rates: a *fast* lane (responsive, ≈ last
//! `1/fast_alpha` samples) and a *slow* lane (baseline, ≈ last `1/slow_alpha`
//! samples). Their ratio detects regressions (fast rising above baseline) and
//! their signed difference classifies the trend slope.
//!
//! EWMA is the standard exponential-smoothing recurrence
//! `x_t = α·sample + (1−α)·x_{t−1}`; see
//! <https://en.wikipedia.org/wiki/Exponential_smoothing>.
//!
//! Two representations coexist:
//! - [`DualEwma`] — the logical value form used in tests and in persistence
//!   flush/reload (the slow lane is the persisted baseline scalar).
//! - [`EwmaLane`] — the atomics-backed live runtime form mutated by the per-item
//!   hot loop with `Relaxed` last-writer-wins stores (PERF-01: no hot mutex;
//!   races stay within EWMA tolerance). Convert via [`DualEwma::from_atomics`]
//!   and [`EwmaLane::store`].
//!
//! Two invariants guard the math against bad input (DOM-02 / DOM-05):
//! - **Non-finite samples are dropped** — an `INFINITY`/`NaN` sample never
//!   enters a lane (the lanes stay finite, `ratio`/`slope` stay finite).
//! - **Near-zero slow lane yields no ratio / a Flat slope** — division by a
//!   ≈0 baseline is never performed.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// A slow lane whose magnitude is below this is treated as zero, so neither the
/// regression ratio nor the slope ever divides by it.
const NEAR_ZERO: f64 = 1e-9;

/// Trend classification of a signal relative to its baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slope {
    /// Fast lane is rising above the slow baseline beyond the flat band.
    Rising,
    /// Fast and slow lanes are within the relative flat band (or undefined).
    Flat,
    /// Fast lane is falling below the slow baseline beyond the flat band.
    Falling,
}

/// Logical dual-rate EWMA value (snapshot / seed form).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualEwma {
    /// Fast (responsive) lane value.
    pub fast: f64,
    /// Slow (baseline) lane value.
    pub slow: f64,
    /// Fast-lane smoothing factor in (0,1].
    pub fast_alpha: f64,
    /// Slow-lane smoothing factor in (0,1].
    pub slow_alpha: f64,
    /// Whether the first sample has seeded both lanes.
    pub seeded: bool,
}

impl DualEwma {
    /// Create an unseeded dual-EWMA with the given smoothing factors.
    pub fn new(fast_alpha: f64, slow_alpha: f64) -> Self {
        Self {
            fast: 0.0,
            slow: 0.0,
            fast_alpha,
            slow_alpha,
            seeded: false,
        }
    }

    /// Feed one sample. Non-finite samples are dropped (DOM-02). The first finite
    /// sample seeds both lanes to that value; subsequent samples advance each
    /// lane by its own `α`.
    pub fn update(&mut self, sample: f64) {
        if !sample.is_finite() {
            return;
        }
        if !self.seeded {
            self.fast = sample;
            self.slow = sample;
            self.seeded = true;
            return;
        }
        self.fast = self.fast_alpha * sample + (1.0 - self.fast_alpha) * self.fast;
        self.slow = self.slow_alpha * sample + (1.0 - self.slow_alpha) * self.slow;
    }

    /// `fast / slow`, or `None` when unseeded, the slow lane is ≈0, or either
    /// lane is non-finite (DOM-05).
    pub fn ratio(&self) -> Option<f64> {
        if !self.seeded
            || self.slow.abs() < NEAR_ZERO
            || !self.fast.is_finite()
            || !self.slow.is_finite()
        {
            return None;
        }
        Some(self.fast / self.slow)
    }

    /// Classify the trend from `sign(fast − slow)` with a relative flat band:
    /// `|fast − slow| / slow < flat_band ⇒ Flat`. Returns `Flat` when unseeded,
    /// the slow lane is ≈0, or either lane is non-finite (DOM-05) — never `NaN`.
    pub fn slope(&self, flat_band: f64) -> Slope {
        if !self.seeded
            || self.slow.abs() < NEAR_ZERO
            || !self.fast.is_finite()
            || !self.slow.is_finite()
        {
            return Slope::Flat;
        }
        let diff = self.fast - self.slow;
        if diff.abs() / self.slow.abs() < flat_band {
            Slope::Flat
        } else if diff > 0.0 {
            Slope::Rising
        } else {
            Slope::Falling
        }
    }

    /// The baseline scalar (the slow lane) — what persistence flushes.
    pub fn baseline(&self) -> f64 {
        self.slow
    }

    /// Restore a persisted baseline (F5 reload).
    ///
    /// CRITICAL (DOM-06): this MUST set `seeded = true` and seed BOTH lanes to
    /// the restored baseline. Without `seeded = true`, the next `update()` would
    /// take the first-sample branch and overwrite the restored baseline with a
    /// fresh live sample — silently nullifying persistence. Seeding `fast = slow`
    /// makes the reloaded baseline the EWMA's starting state; the fast lane then
    /// re-warms from live samples via the normal update path.
    pub fn restore_baseline(&mut self, slow: f64) {
        self.slow = slow;
        self.fast = slow;
        self.seeded = true;
    }

    /// Snapshot an atomics-backed lane into a logical value, carrying the alphas.
    pub fn from_atomics(lane: &EwmaLane, fast_alpha: f64, slow_alpha: f64) -> Self {
        Self {
            fast: lane.load_fast(),
            slow: lane.load_slow(),
            fast_alpha,
            slow_alpha,
            seeded: lane.is_seeded(),
        }
    }
}

/// Atomics-backed live EWMA lane: fast & slow each held as `f64::to_bits()`.
///
/// Per-item updates are `Relaxed` last-writer-wins — within EWMA tolerance, the
/// occasional lost update from a concurrent writer is immaterial, and avoiding a
/// lock keeps the per-item hot loop contention-free (PERF-01). The smoothing
/// factors are NOT stored here; the caller (`EwmaState`) passes them in, since
/// they are immutable after construction.
#[derive(Debug)]
pub struct EwmaLane {
    fast: AtomicU64,
    slow: AtomicU64,
    seeded: AtomicBool,
}

impl Default for EwmaLane {
    fn default() -> Self {
        Self::new()
    }
}

impl EwmaLane {
    /// Create an unseeded lane (both bits zero, `seeded = false`).
    pub fn new() -> Self {
        Self {
            fast: AtomicU64::new(0.0_f64.to_bits()),
            slow: AtomicU64::new(0.0_f64.to_bits()),
            seeded: AtomicBool::new(false),
        }
    }

    fn load_fast(&self) -> f64 {
        f64::from_bits(self.fast.load(Ordering::Relaxed))
    }

    fn load_slow(&self) -> f64 {
        f64::from_bits(self.slow.load(Ordering::Relaxed))
    }

    fn is_seeded(&self) -> bool {
        self.seeded.load(Ordering::Relaxed)
    }

    /// Feed one sample at the given smoothing rates (mirrors
    /// [`DualEwma::update`], including the non-finite drop and first-sample
    /// seeding) but mutating the atomics in place.
    pub fn update(&self, sample: f64, fast_alpha: f64, slow_alpha: f64) {
        if !sample.is_finite() {
            return;
        }
        if !self.seeded.load(Ordering::Relaxed) {
            self.fast.store(sample.to_bits(), Ordering::Relaxed);
            self.slow.store(sample.to_bits(), Ordering::Relaxed);
            self.seeded.store(true, Ordering::Relaxed);
            return;
        }
        let fast = self.load_fast();
        let slow = self.load_slow();
        self.fast.store(
            (fast_alpha * sample + (1.0 - fast_alpha) * fast).to_bits(),
            Ordering::Relaxed,
        );
        self.slow.store(
            (slow_alpha * sample + (1.0 - slow_alpha) * slow).to_bits(),
            Ordering::Relaxed,
        );
    }

    /// Overwrite the lane from a logical value (F5 reload / seeding).
    pub fn store(&self, value: &DualEwma) {
        self.fast.store(value.fast.to_bits(), Ordering::Relaxed);
        self.slow.store(value.slow.to_bits(), Ordering::Relaxed);
        self.seeded.store(value.seeded, Ordering::Relaxed);
    }
}

#[cfg(test)]
#[path = "ewma_tests.rs"]
mod tests;

//! Unit tests for the dual-rate EWMA primitive (#133 F2, test-strategy item 1).

use super::*;

const FAST_ALPHA: f64 = 0.3;
const SLOW_ALPHA: f64 = 0.01;
const FLAT_BAND: f64 = 0.05;

/// K = ceil(3 / fast_alpha): the fast lane's ~95% convergence horizon (three
/// time constants). With fast_alpha = 0.3 this is 10 samples.
const K: usize = 10;

fn seeded(fast: f64, slow: f64) -> DualEwma {
    DualEwma {
        fast,
        slow,
        fast_alpha: FAST_ALPHA,
        slow_alpha: SLOW_ALPHA,
        seeded: true,
    }
}

// ── Convergence + seeding ───────────────────────────────────────────────────

#[test]
fn first_sample_seeds_both_lanes() {
    let mut e = DualEwma::new(FAST_ALPHA, SLOW_ALPHA);
    assert!(!e.seeded);
    assert_eq!(e.ratio(), None, "unseeded -> no ratio");
    assert_eq!(e.slope(FLAT_BAND), Slope::Flat, "unseeded -> Flat");

    e.update(5.0);
    assert!(e.seeded);
    assert_eq!(e.fast, 5.0);
    assert_eq!(e.slow, 5.0);
    assert_eq!(e.ratio(), Some(1.0), "seeded equal lanes -> ratio 1");
}

#[test]
fn constant_input_converges_both_lanes() {
    let mut e = DualEwma::new(FAST_ALPHA, SLOW_ALPHA);
    for _ in 0..1000 {
        e.update(7.0);
    }
    assert!((e.fast - 7.0).abs() < 1e-6, "fast converged to constant");
    assert!((e.slow - 7.0).abs() < 1e-3, "slow converged to constant");
    let r = e.ratio().unwrap();
    assert!(
        (r - 1.0).abs() < 1e-3,
        "constant input -> ratio ~ 1 (got {r})"
    );
}

// ── Step change / crossover (DOM-01) ────────────────────────────────────────

#[test]
fn step_change_moves_fast_faster_than_slow() {
    // Establish a stable 1.0 baseline, then step to 2.0 for one sample.
    let mut e = DualEwma::new(FAST_ALPHA, SLOW_ALPHA);
    for _ in 0..500 {
        e.update(1.0);
    }
    e.update(2.0);
    assert!(e.fast > e.slow, "fast leads slow after a step up");
    let r = e.ratio().unwrap();
    assert!(r > 1.0, "ratio rises above 1 after a step (got {r})");
}

#[test]
fn no_premature_alarm_after_single_step_sample() {
    // DOM-01: a single 2x step sample must NOT push the ratio past R=2.0.
    // Analytic step-2 ratio for a 2x step = (0.3*2+0.7)/(0.01*2+0.99) ~= 1.29.
    let mut e = DualEwma::new(FAST_ALPHA, SLOW_ALPHA);
    for _ in 0..500 {
        e.update(1.0);
    }
    e.update(2.0);
    let r = e.ratio().unwrap();
    assert!(r < 2.0, "one 2x sample stays below R (got {r})");
    assert!((r - 1.29).abs() < 0.02, "matches analytic ~1.29 (got {r})");
}

#[test]
fn sustained_regression_crosses_ratio_threshold_after_warmup() {
    // DOM-01: a sustained step large enough to regress crosses R only after the
    // fast lane warms up. A 2x step peaks at ratio ~1.8 (below R) because the
    // slow baseline drifts up too; a 3x step crosses R within the K-sample fast
    // horizon, which is the case the regression probe must catch.
    let mut e = DualEwma::new(FAST_ALPHA, SLOW_ALPHA);
    for _ in 0..500 {
        e.update(1.0); // baseline ~1.0 on both lanes
    }

    let mut ratios = Vec::new();
    for _ in 0..K {
        e.update(3.0);
        ratios.push(e.ratio().unwrap());
    }

    // First sample after the step is below R (no premature alarm).
    assert!(ratios[0] < 2.0, "step-1 ratio below R (got {})", ratios[0]);
    // The ratio rises over the early (pre-peak) samples — the crossover
    // direction. Strict monotonicity holds through the rising region; the
    // fast lane's ~95% horizon is K, after which the slow lane begins to catch
    // up, so we assert the rise over the first 5 samples.
    for w in ratios[..5].windows(2) {
        assert!(
            w[1] > w[0],
            "ratio rising over the warm-up region: {ratios:?}"
        );
    }
    // By the K-sample horizon the sustained 3x regression is firmly above R.
    assert!(
        ratios[K - 1] > 2.0,
        "sustained regression exceeds R by sample K (got {})",
        ratios[K - 1]
    );
}

// ── Non-finite resilience (DOM-02) ──────────────────────────────────────────

#[test]
fn non_finite_samples_are_dropped() {
    let mut e = DualEwma::new(FAST_ALPHA, SLOW_ALPHA);
    e.update(4.0); // seed
    let (f, s) = (e.fast, e.slow);

    e.update(f64::INFINITY);
    e.update(f64::NAN);
    e.update(f64::NEG_INFINITY);

    assert_eq!(e.fast, f, "INFINITY/NaN leave the fast lane unchanged");
    assert_eq!(e.slow, s, "INFINITY/NaN leave the slow lane unchanged");
    assert!(e.fast.is_finite() && e.slow.is_finite());
    assert!(e.ratio().unwrap().is_finite());
    assert_eq!(e.slope(FLAT_BAND), Slope::Flat);
}

#[test]
fn first_sample_non_finite_does_not_seed() {
    let mut e = DualEwma::new(FAST_ALPHA, SLOW_ALPHA);
    e.update(f64::NAN);
    assert!(!e.seeded, "a non-finite first sample must not seed");
    assert_eq!(e.ratio(), None);
}

// ── Ratio / slope guards (DOM-05) ───────────────────────────────────────────

#[test]
fn near_zero_slow_lane_has_no_ratio_and_flat_slope() {
    let e = seeded(0.5, 0.0);
    assert_eq!(e.ratio(), None, "slow ~ 0 -> no ratio (no divide by zero)");
    assert_eq!(e.slope(FLAT_BAND), Slope::Flat, "slow ~ 0 -> Flat");
}

#[test]
fn slope_classifies_at_flat_band_edges() {
    // flat_band = 0.05 (relative).
    assert_eq!(
        seeded(1.04, 1.0).slope(FLAT_BAND),
        Slope::Flat,
        "0.04 < band"
    );
    assert_eq!(
        seeded(1.06, 1.0).slope(FLAT_BAND),
        Slope::Rising,
        "0.06 > band"
    );
    assert_eq!(
        seeded(0.96, 1.0).slope(FLAT_BAND),
        Slope::Flat,
        "-0.04 < band"
    );
    assert_eq!(
        seeded(0.94, 1.0).slope(FLAT_BAND),
        Slope::Falling,
        "-0.06 > band"
    );
}

#[test]
fn non_finite_lane_yields_flat_slope_and_no_ratio() {
    let e = seeded(f64::INFINITY, 1.0);
    assert_eq!(e.ratio(), None);
    assert_eq!(e.slope(FLAT_BAND), Slope::Flat);
}

// ── Persistence seeding (DOM-06) ────────────────────────────────────────────

#[test]
fn restore_baseline_seeds_both_lanes() {
    let mut e = DualEwma::new(FAST_ALPHA, SLOW_ALPHA);
    e.restore_baseline(7.0);
    assert!(e.seeded, "restore_baseline marks the lane seeded");
    assert_eq!(e.slow, 7.0);
    assert_eq!(e.fast, 7.0);
    assert_eq!(e.baseline(), 7.0);
}

#[test]
fn post_reload_update_uses_ewma_path_not_seeding() {
    // DOM-06: after restore, the next update must advance the slow lane by the
    // EWMA recurrence, NOT overwrite it via the first-sample seeding branch.
    let mut e = DualEwma::new(FAST_ALPHA, SLOW_ALPHA);
    e.restore_baseline(7.0);
    e.update(8.0);
    let expected_slow = SLOW_ALPHA * 8.0 + (1.0 - SLOW_ALPHA) * 7.0;
    assert!(
        (e.slow - expected_slow).abs() < 1e-9,
        "slow advanced by EWMA (expected {expected_slow}, got {})",
        e.slow
    );
    assert_ne!(e.slow, 8.0, "must not have re-seeded to the live sample");
}

// ── Atomics-backed lane parity ──────────────────────────────────────────────

#[test]
fn ewma_lane_matches_logical_dual_ewma() {
    let lane = EwmaLane::new();
    let mut logical = DualEwma::new(FAST_ALPHA, SLOW_ALPHA);
    for sample in [2.0, 5.0, f64::NAN, 5.0, 5.0, f64::INFINITY, 6.0] {
        lane.update(sample, FAST_ALPHA, SLOW_ALPHA);
        logical.update(sample);
    }
    let snap = DualEwma::from_atomics(&lane, FAST_ALPHA, SLOW_ALPHA);
    assert!((snap.fast - logical.fast).abs() < 1e-12);
    assert!((snap.slow - logical.slow).abs() < 1e-12);
    assert_eq!(snap.seeded, logical.seeded);
}

#[test]
fn ewma_lane_store_round_trips_a_logical_value() {
    let lane = EwmaLane::new();
    let value = seeded(3.0, 2.5);
    lane.store(&value);
    let snap = DualEwma::from_atomics(&lane, FAST_ALPHA, SLOW_ALPHA);
    assert_eq!(snap.fast, 3.0);
    assert_eq!(snap.slow, 2.5);
    assert!(snap.seeded);
}

#[test]
fn ewma_lane_drops_non_finite_first_sample() {
    let lane = EwmaLane::new();
    lane.update(f64::NAN, FAST_ALPHA, SLOW_ALPHA);
    let snap = DualEwma::from_atomics(&lane, FAST_ALPHA, SLOW_ALPHA);
    assert!(!snap.seeded);
}

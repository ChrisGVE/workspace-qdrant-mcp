//! Verdict-builder tests (#133 F6) — worst-of, cold-start, aggregation, and the
//! debounce-lock-free guarantee.

use super::{verdict, HealthVerdict};
use crate::config::queue_health::QueueHealthConfig;
use crate::queue_health::probes::{ProbeResult, ALL_FAILING, DLQ, PROCESSING, QDRANT};
use crate::queue_health::state::{EwmaState, Rag};
use crate::queue_health::QueueProcessorHealth;

fn cfg() -> QueueHealthConfig {
    QueueHealthConfig::default()
}

const HEALTHY_DISK: Option<u64> = Some(1_000_000_000_000);

#[test]
fn from_probes_all_green_is_green() {
    let v = HealthVerdict::from_probes(
        vec![ProbeResult::green(PROCESSING), ProbeResult::green(QDRANT)],
        false,
    );
    assert_eq!(v.overall, Rag::Green);
    assert_eq!(v.degraded().count(), 0);
}

#[test]
fn from_probes_worst_of_picks_red() {
    let v = HealthVerdict::from_probes(
        vec![
            ProbeResult::green(PROCESSING),
            ProbeResult::amber(DLQ, "stuck"),
            ProbeResult::red(QDRANT, "down"),
        ],
        false,
    );
    assert_eq!(v.overall, Rag::Red);
    assert_eq!(v.degraded().count(), 2);
}

#[test]
fn from_probes_amber_when_no_red() {
    let v = HealthVerdict::from_probes(
        vec![
            ProbeResult::green(PROCESSING),
            ProbeResult::amber(DLQ, "stuck"),
        ],
        false,
    );
    assert_eq!(v.overall, Rag::Amber);
}

#[test]
fn verdict_fresh_daemon_is_cold_start_green() {
    let s = EwmaState::new(&cfg());
    let h = QueueProcessorHealth::new();
    let v = verdict(&s, &h, &cfg(), true, HEALTHY_DISK, HEALTHY_DISK);
    assert_eq!(v.overall, Rag::Green);
    assert!(v.cold_start, "no lane seeded ⇒ cold start");
}

#[test]
fn verdict_hard_failure_surfaces_red_even_at_cold_start() {
    let s = EwmaState::new(&cfg());
    let h = QueueProcessorHealth::new();
    // Qdrant unreachable on a fresh daemon: Red overall, still cold_start.
    let v = verdict(&s, &h, &cfg(), false, HEALTHY_DISK, HEALTHY_DISK);
    assert_eq!(v.overall, Rag::Red);
    assert!(v.cold_start);
    assert!(v.degraded().any(|p| p.culprit == QDRANT));
}

#[test]
fn verdict_reads_debounced_trend_cache() {
    let s = EwmaState::new(&cfg());
    s.update_ms_per_kb(1.0); // seed a lane ⇒ not cold start
    s.set_trend_cache(vec![ProbeResult::amber(PROCESSING, "regressing 2.4x")]);
    let h = QueueProcessorHealth::new();
    let v = verdict(&s, &h, &cfg(), true, HEALTHY_DISK, HEALTHY_DISK);
    assert_eq!(v.overall, Rag::Amber);
    assert!(!v.cold_start);
    assert!(v
        .probes
        .iter()
        .any(|p| p.culprit == PROCESSING && p.rag == Rag::Amber));
}

#[test]
fn verdict_reads_all_failing_atomic() {
    let s = EwmaState::new(&cfg());
    s.update_ms_per_kb(1.0);
    s.set_all_failing(true);
    let h = QueueProcessorHealth::new();
    let v = verdict(&s, &h, &cfg(), true, HEALTHY_DISK, HEALTHY_DISK);
    assert_eq!(v.overall, Rag::Red);
    assert!(v.degraded().any(|p| p.culprit == ALL_FAILING));
}

#[test]
fn verdict_does_not_acquire_debounce_write_lock() {
    // Hold the debounce lock for the whole call; if verdict() tried to observe()
    // it would deadlock. It must not — it reads only the cached snapshot.
    let s = EwmaState::new(&cfg());
    s.update_ms_per_kb(1.0);
    let h = QueueProcessorHealth::new();
    let _held = s.lock_debounce_for_test();
    let v = verdict(&s, &h, &cfg(), true, HEALTHY_DISK, HEALTHY_DISK);
    assert!(!v.cold_start);
}

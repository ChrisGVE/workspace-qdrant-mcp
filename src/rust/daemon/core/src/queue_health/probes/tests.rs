//! Probe unit tests (#133 F3/F4/F5) — every RAG state of every probe, plus the
//! remediation leak-scan (S8/SEC-03).

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime};

use super::drain::drain_budget;
use super::hard_state::{b1_qdrant, b2_disk, b3_stall, b4_all_failing, b4_result, PollOutcome};
use super::trend::{a1_ms_per_kb, a2_embedder_latency, a3_dlq_trend};
use crate::config::queue_health::QueueHealthConfig;
use crate::queue_health::state::{EwmaState, Rag};
use crate::queue_health::QueueProcessorHealth;

fn cfg() -> QueueHealthConfig {
    QueueHealthConfig::default()
}

// ── Family A — trend ────────────────────────────────────────────────────────

#[test]
fn a1_unseeded_is_green() {
    let s = EwmaState::new(&cfg());
    assert_eq!(a1_ms_per_kb(&s, &cfg()).rag, Rag::Green);
}

#[test]
fn a1_regression_is_amber() {
    let s = EwmaState::new(&cfg());
    s.update_ms_per_kb(1.0); // seed both lanes low
    for _ in 0..20 {
        s.update_ms_per_kb(100.0); // fast climbs far above the slow baseline
    }
    let r = a1_ms_per_kb(&s, &cfg());
    assert_eq!(r.rag, Rag::Amber);
    assert!(r.remediation.is_some());
}

#[test]
fn a1_near_zero_baseline_is_green() {
    // Baseline below ms_per_kb_floor ⇒ Green even though fast/slow is large
    // (too fast to matter; never divide by a near-zero baseline, DOM-05).
    let s = EwmaState::new(&cfg());
    s.update_ms_per_kb(0.01); // seed both at 0.01
    s.update_ms_per_kb(0.5); // fast jumps, slow stays ≈0.015 < 0.1 floor
    let r = a1_ms_per_kb(&s, &cfg());
    assert!(s.ms_per_kb_snapshot().baseline() < cfg().ms_per_kb_floor);
    assert_eq!(r.rag, Rag::Green);
}

#[test]
fn a2_regression_is_amber() {
    let s = EwmaState::new(&cfg());
    s.update_embedder_latency(5.0);
    for _ in 0..20 {
        s.update_embedder_latency(500.0);
    }
    assert_eq!(a2_embedder_latency(&s, &cfg()).rag, Rag::Amber);
}

#[test]
fn a3_empty_is_green() {
    let s = EwmaState::new(&cfg());
    s.update_dlq_depth(0.0);
    s.update_dlq_depth(50.0); // even a rising rate is Green while count is empty
    assert_eq!(a3_dlq_trend(&s, &cfg(), 0, 5).rag, Rag::Green);
}

#[test]
fn a3_one_sample_is_green() {
    // First post-restart poll only seeds the rate lane — one sample is no trend.
    let s = EwmaState::new(&cfg());
    s.update_dlq_depth(50.0);
    assert_eq!(a3_dlq_trend(&s, &cfg(), 50, 1).rag, Rag::Green);
}

#[test]
fn a3_rate_rising_is_red() {
    let s = EwmaState::new(&cfg());
    s.update_dlq_depth(0.0); // seed delta lane at 0
    s.update_dlq_depth(5.0); // fast rate = 1.5 > dlq_rate_band (1.0)
    assert!(s.dlq_depth_snapshot().fast > cfg().dlq_rate_band);
    assert_eq!(a3_dlq_trend(&s, &cfg(), 12, 2).rag, Rag::Red);
}

#[test]
fn a3_stuck_is_amber() {
    let s = EwmaState::new(&cfg());
    s.update_dlq_depth(0.0);
    s.update_dlq_depth(0.0); // rate ≈ 0, |rate| ≤ band, non-empty ⇒ stuck
    assert_eq!(a3_dlq_trend(&s, &cfg(), 7, 2).rag, Rag::Amber);
}

#[test]
fn a3_draining_is_green() {
    let s = EwmaState::new(&cfg());
    s.update_dlq_depth(0.0);
    s.update_dlq_depth(-5.0); // fast rate = -1.5 < -band ⇒ draining
    assert_eq!(a3_dlq_trend(&s, &cfg(), 7, 2).rag, Rag::Green);
}

// ── Family B — hard state ─────────────────────────────────────────────────────

#[test]
fn b1_reachable_green_unreachable_red() {
    assert_eq!(b1_qdrant(true).rag, Rag::Green);
    assert_eq!(b1_qdrant(false).rag, Rag::Red);
}

#[test]
fn b2_disk_low_absolute_is_red() {
    let c = cfg();
    let r = b2_disk(Some(c.disk_low_bytes - 1), Some(1_000_000_000_000), &c);
    assert_eq!(r.rag, Rag::Red);
}

#[test]
fn b2_disk_low_fraction_is_red() {
    let c = cfg();
    // Plenty of absolute bytes, but below the 5% fraction of a huge volume.
    let total = 1_000_000_000_000u64;
    let free = (total as f64 * 0.01) as u64; // 1% < 5%
    assert!(free > c.disk_low_bytes);
    assert_eq!(b2_disk(Some(free), Some(total), &c).rag, Rag::Red);
}

#[test]
fn b2_disk_ample_is_green() {
    let c = cfg();
    let total = 1_000_000_000_000u64;
    assert_eq!(b2_disk(Some(total / 2), Some(total), &c).rag, Rag::Green);
}

#[test]
fn b2_disk_unreadable_is_green() {
    assert_eq!(b2_disk(None, None, &cfg()).rag, Rag::Green);
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[test]
fn b3_stall_pending_and_idle_is_red() {
    let h = QueueProcessorHealth::new();
    h.set_queue_depth(5);
    h.last_poll_time.store(now_ms() - 120_000, Ordering::SeqCst); // 120s ago
    assert_eq!(b3_stall(&h, &cfg()).rag, Rag::Red);
}

#[test]
fn b3_no_pending_is_green() {
    let h = QueueProcessorHealth::new();
    h.set_queue_depth(0);
    h.last_poll_time.store(now_ms() - 120_000, Ordering::SeqCst);
    assert_eq!(b3_stall(&h, &cfg()).rag, Rag::Green);
}

#[test]
fn b3_recent_poll_is_green() {
    let h = QueueProcessorHealth::new();
    h.set_queue_depth(5);
    h.record_poll();
    h.record_heartbeat();
    assert_eq!(b3_stall(&h, &cfg()).rag, Rag::Green);
}

#[test]
fn b4_all_failing_detected() {
    let window = [
        PollOutcome {
            items_processed: 100,
            dlq_count: 10,
            attempts: 100,
        },
        PollOutcome {
            items_processed: 100, // no progress
            dlq_count: 13,        // DLQ net-increased
            attempts: 103,        // items were attempted
        },
    ];
    assert!(b4_all_failing(&window));
    assert_eq!(b4_result(true).rag, Rag::Red);
}

#[test]
fn b4_progress_made_is_not_failing() {
    let window = [
        PollOutcome {
            items_processed: 100,
            dlq_count: 10,
            attempts: 100,
        },
        PollOutcome {
            items_processed: 105, // progress ⇒ not all-failing
            dlq_count: 13,
            attempts: 108,
        },
    ];
    assert!(!b4_all_failing(&window));
    assert_eq!(b4_result(false).rag, Rag::Green);
}

#[test]
fn b4_single_flat_poll_does_not_trip() {
    // DOM-07: net change across the window, not "increased every cycle". A flat
    // middle poll between two increases still nets an increase ⇒ trips; a window
    // with no net DLQ increase does not.
    let window = [
        PollOutcome {
            items_processed: 100,
            dlq_count: 10,
            attempts: 100,
        },
        PollOutcome {
            items_processed: 100,
            dlq_count: 10, // no net increase
            attempts: 105,
        },
    ];
    assert!(!b4_all_failing(&window));
}

#[test]
fn b4_short_window_is_not_failing() {
    let window = [PollOutcome {
        items_processed: 100,
        dlq_count: 10,
        attempts: 100,
    }];
    assert!(!b4_all_failing(&window));
}

// ── Drain budget (F5) ─────────────────────────────────────────────────────────

#[test]
fn drain_within_budget_is_green() {
    let s = EwmaState::new(&cfg());
    s.update_throughput(1_000_000.0); // 1 MB/s drain rate
    s.set_drain_snapshot(1_000); // tiny backlog
    assert_eq!(drain_budget(&s, &cfg()).rag, Rag::Green);
}

#[test]
fn drain_over_budget_is_amber() {
    let s = EwmaState::new(&cfg());
    s.update_throughput(1.0); // 1 byte/s
    s.set_drain_snapshot(200_000); // eta ≈ 200_000 s ≫ 86_400
    assert_eq!(drain_budget(&s, &cfg()).rag, Rag::Amber);
}

#[test]
fn drain_zero_throughput_with_backlog_is_amber() {
    let s = EwmaState::new(&cfg());
    s.update_throughput(0.0); // seeded but ≈0 rate
    s.set_drain_snapshot(1_000);
    assert_eq!(drain_budget(&s, &cfg()).rag, Rag::Amber);
}

#[test]
fn drain_unseeded_throughput_is_green() {
    let s = EwmaState::new(&cfg());
    s.set_drain_snapshot(1_000); // throughput never seeded
    assert_eq!(drain_budget(&s, &cfg()).rag, Rag::Green);
}

#[test]
fn drain_stale_snapshot_is_green() {
    let c = cfg();
    let s = EwmaState::new(&c);
    s.update_throughput(1.0);
    s.set_drain_snapshot_at(
        200_000,
        Instant::now() - Duration::from_secs(c.drain_snapshot_max_age_secs + 10),
    );
    assert_eq!(drain_budget(&s, &c).rag, Rag::Green);
}

#[test]
fn drain_no_snapshot_is_green() {
    let s = EwmaState::new(&cfg());
    s.update_throughput(1.0);
    assert_eq!(drain_budget(&s, &cfg()).rag, Rag::Green);
}

// ── Remediation leak-scan (S8 / SEC-03 / SEC-07) ──────────────────────────────

/// Collect every remediation string a probe can emit, by driving each non-green
/// state at least once.
fn all_remediations() -> Vec<String> {
    let c = cfg();
    let mut out = Vec::new();

    // A1 / A2 regression (numeric ratio interpolated).
    let s = EwmaState::new(&c);
    s.update_ms_per_kb(1.0);
    for _ in 0..20 {
        s.update_ms_per_kb(100.0);
    }
    out.extend(a1_ms_per_kb(&s, &c).remediation);
    let s = EwmaState::new(&c);
    s.update_embedder_latency(5.0);
    for _ in 0..20 {
        s.update_embedder_latency(500.0);
    }
    out.extend(a2_embedder_latency(&s, &c).remediation);

    // A3 Red + Amber.
    let s = EwmaState::new(&c);
    s.update_dlq_depth(0.0);
    s.update_dlq_depth(5.0);
    out.extend(a3_dlq_trend(&s, &c, 12, 2).remediation);
    let s = EwmaState::new(&c);
    s.update_dlq_depth(0.0);
    s.update_dlq_depth(0.0);
    out.extend(a3_dlq_trend(&s, &c, 7, 2).remediation);

    // B1, B2, B3, B4.
    out.extend(b1_qdrant(false).remediation);
    out.extend(b2_disk(Some(0), Some(1_000_000_000_000), &c).remediation);
    let h = QueueProcessorHealth::new();
    h.set_queue_depth(5);
    h.last_poll_time.store(now_ms() - 120_000, Ordering::SeqCst);
    out.extend(b3_stall(&h, &c).remediation);
    out.extend(b4_result(true).remediation);

    // Drain Amber (both messages).
    let s = EwmaState::new(&c);
    s.update_throughput(1.0);
    s.set_drain_snapshot(200_000);
    out.extend(drain_budget(&s, &c).remediation);
    let s = EwmaState::new(&c);
    s.update_throughput(0.0);
    s.set_drain_snapshot(1_000);
    out.extend(drain_budget(&s, &c).remediation);

    out
}

#[test]
fn remediations_leak_no_paths_secrets_or_config_names() {
    let rems = all_remediations();
    assert!(
        rems.len() >= 8,
        "expected the full canned set, got {}",
        rems.len()
    );
    for r in &rems {
        // No absolute path / URL.
        assert!(!r.contains('/'), "remediation leaks a path/URL: {r}");
        assert!(!r.contains("://"), "remediation leaks a URL: {r}");
        // No WQM_* config-variable name (prefix-complete scan, SEC-07).
        assert!(
            !wqm_env_prefix(r),
            "remediation leaks a WQM_* config-variable name: {r}"
        );
        // No key-like token.
        let lower = r.to_lowercase();
        assert!(
            !lower.contains("api_key") && !lower.contains("secret") && !lower.contains("token"),
            "remediation leaks a key-like token: {r}"
        );
    }
}

/// True if `s` contains a `WQM_[A-Z0-9_]+` token (the daemon env-var prefix).
fn wqm_env_prefix(s: &str) -> bool {
    let bytes = s.as_bytes();
    let needle = b"WQM_";
    bytes.windows(needle.len()).enumerate().any(|(i, w)| {
        w == needle
            && bytes
                .get(i + needle.len())
                .is_some_and(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == b'_')
    })
}

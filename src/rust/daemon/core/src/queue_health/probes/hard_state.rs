//! Family B — hard-state boolean probes (#133 F4).
//!
//! Each is a binary predicate: Green when fine, Red when failing. No Amber — a
//! hard fault (Qdrant down, disk full, processing stalled, every item failing)
//! is degraded-to-failing with no in-between shade.
//!
//! Probe inputs are passed in (not read from globals) so each probe is pure and
//! unit-testable with no real network, filesystem, or clock:
//! - **B1** takes a `reachable: bool` — the gRPC layer issues the bounded
//!   `StorageClient::test_connection()` (timeout ⇒ `false`) and feeds the result.
//! - **B2** takes `(free, total)` bytes — production reads them via `sysinfo`; the
//!   pure probe never touches the filesystem.
//! - **B3** reads the [`QueueProcessorHealth`] timestamp atomics (a test seeds an
//!   old `last_poll_time` to simulate a stall).
//! - **B4** evaluates a slice of poll outcomes (the poll loop keeps the ring
//!   locally and stores the predicate in a lock-free atomic for the verdict).

use std::path::Path;
use std::sync::atomic::Ordering;

use super::{ProbeResult, ALL_FAILING, DISK, QDRANT, STALL};
use crate::config::queue_health::QueueHealthConfig;
use crate::queue_health::QueueProcessorHealth;

/// B1 — Qdrant reachability. `reachable` is the result of the bounded live
/// `test_connection()` check (timeout or error ⇒ `false`), issued by the caller.
pub fn b1_qdrant(reachable: bool) -> ProbeResult {
    if reachable {
        ProbeResult::green(QDRANT)
    } else {
        ProbeResult::red(
            QDRANT,
            "Qdrant is unreachable; verify the Qdrant service is running \
             and the configured endpoint is reachable.",
        )
    }
}

/// B2 — free disk space on the database volume.
///
/// Red when free bytes fall below the absolute floor OR below the relative
/// fraction of the volume; Green otherwise. A `None` free reading (could not
/// stat the volume) is Green — an unreadable disk must not raise a false alarm.
pub fn b2_disk(
    free_bytes: Option<u64>,
    total_bytes: Option<u64>,
    cfg: &QueueHealthConfig,
) -> ProbeResult {
    let Some(free) = free_bytes else {
        return ProbeResult::green(DISK);
    };
    let below_absolute = free < cfg.disk_low_bytes;
    let below_fraction = match total_bytes {
        Some(total) if total > 0 => (free as f64 / total as f64) < cfg.disk_low_pct,
        _ => false,
    };
    if below_absolute || below_fraction {
        ProbeResult::red(
            DISK,
            "Free disk space is low; indexing will stall when the database volume fills.",
        )
    } else {
        ProbeResult::green(DISK)
    }
}

/// Production B2 disk reader (the `fn(&Path) -> bytes` seam): free + total bytes
/// on the volume containing `path`, found by longest-prefix-matching `path`
/// against the mounted filesystems. Returns `(None, None)` when no mount matches,
/// so [`b2_disk`] stays Green rather than raising a false alarm. Tests drive
/// `b2_disk` with injected values; this hits the real filesystem and is not
/// unit-tested.
pub fn read_disk_space(path: &Path) -> (Option<u64>, Option<u64>) {
    use sysinfo::Disks;
    let disks = Disks::new_with_refreshed_list();
    let mut best: Option<(usize, u64, u64)> = None;
    for disk in &disks {
        let mount = disk.mount_point();
        if !path.starts_with(mount) {
            continue;
        }
        let mount_len = mount.as_os_str().len();
        let better = match best {
            Some((best_len, _, _)) => mount_len > best_len,
            None => true,
        };
        if better {
            best = Some((mount_len, disk.available_space(), disk.total_space()));
        }
    }
    match best {
        Some((_, free, total)) => (Some(free), Some(total)),
        None => (None, None),
    }
}

/// B3 — processing stall. Red when work is pending (`queue_depth > 0`) and
/// neither a poll nor a per-item heartbeat has happened within `stall_timeout_secs`.
pub fn b3_stall(health: &QueueProcessorHealth, cfg: &QueueHealthConfig) -> ProbeResult {
    if health.queue_depth.load(Ordering::SeqCst) == 0 {
        return ProbeResult::green(STALL);
    }
    let idle_secs = health
        .seconds_since_last_poll()
        .min(health.seconds_since_last_heartbeat());
    if idle_secs > cfg.stall_timeout_secs {
        ProbeResult::red(
            STALL,
            "Queue processor has not made progress recently; \
             check daemon logs for a stuck item or a blocked embedding provider.",
        )
    } else {
        ProbeResult::green(STALL)
    }
}

/// One poll's cumulative outcome counters, pushed onto the B4 ring per poll.
#[derive(Debug, Clone, Copy)]
pub struct PollOutcome {
    /// Cumulative items successfully processed (`QueueProcessorHealth::items_processed`).
    pub items_processed: u64,
    /// Absolute dead-letter-queue count at this poll.
    pub dlq_count: u64,
    /// Cumulative items attempted (processed + failed) — used to require ≥1
    /// attempt over the window.
    pub attempts: u64,
}

/// B4 predicate — over the outcome window, `items_processed` did **not** advance
/// AND the DLQ **net-increased** AND ≥1 item was attempted (DOM-07 relaxation:
/// uses net change across the window, so a single flat retry-backoff poll does
/// not reset the detector). Needs ≥2 samples to span a window.
pub fn b4_all_failing(window: &[PollOutcome]) -> bool {
    let (Some(first), Some(last)) = (window.first(), window.last()) else {
        return false;
    };
    if window.len() < 2 {
        return false;
    }
    let processed_advanced = last.items_processed > first.items_processed;
    let dlq_net_increased = last.dlq_count > first.dlq_count;
    let attempted = last.attempts > first.attempts;
    !processed_advanced && dlq_net_increased && attempted
}

/// B4 result from the precomputed predicate (the poll loop stores `all_failing`
/// in a lock-free atomic; the verdict reads it here).
pub fn b4_result(all_failing: bool) -> ProbeResult {
    if all_failing {
        ProbeResult::red(
            ALL_FAILING,
            "Every recent item is failing into the dead-letter queue; \
             a systemic fault (provider down, schema mismatch) is likely — check daemon logs.",
        )
    } else {
        ProbeResult::green(ALL_FAILING)
    }
}

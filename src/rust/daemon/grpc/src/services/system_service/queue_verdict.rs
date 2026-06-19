//! Queue-processor functional health verdict surfacing (#133 F7).
//!
//! Extracted from `helpers.rs` (PRD IMPL-08 — keep that file under the 500-line
//! limit). Builds the `queue_processor` `ComponentHealth` from the #133 verdict:
//! reads the poll-loop-debounced trend cache + B4 atomic off the shared
//! `EwmaState`, runs the live B1 Qdrant reachability probe and the B2 disk read
//! here (poll/RPC cadence, never per item), maps worst-of → `ServiceStatus`, and
//! formats the bounded remediation message.

use std::path::Path;
use std::time::{Duration, SystemTime};

use workspace_qdrant_core::config::queue_health::QueueHealthConfig;
use workspace_qdrant_core::queue_health::probes::hard_state::read_disk_space;
use workspace_qdrant_core::queue_health::verdict::{verdict, HealthVerdict};
use workspace_qdrant_core::queue_health::Rag;

use crate::proto::{ComponentHealth, ServiceStatus};

use super::service_impl::SystemServiceImpl;

impl SystemServiceImpl {
    /// Build the `queue_processor` ComponentHealth from the #133 functional
    /// verdict (replaces the old `is_running`/`>60s` check). Cold-start (no lane
    /// seeded, all green) maps to `Unspecified` — the CLI/TUI decode that to
    /// "unknown / learning baseline".
    pub(super) async fn get_queue_processor_health(&self) -> ComponentHealth {
        let (Some(ewma), Some(health)) = (&self.ewma_state, &self.queue_health) else {
            return Self::queue_health_component(
                ServiceStatus::Unspecified,
                "Health monitoring not connected".to_string(),
            );
        };
        let cfg = ewma.config();

        let qdrant_reachable = self.probe_qdrant_reachable(cfg).await;
        let (disk_free, disk_total) = self.read_state_db_disk().await;

        let v = verdict(ewma, health, cfg, qdrant_reachable, disk_free, disk_total);

        let status = if v.cold_start && v.overall == Rag::Green {
            ServiceStatus::Unspecified // "unknown" — learning baseline (UX-3).
        } else {
            match v.overall {
                Rag::Green => ServiceStatus::Healthy,
                Rag::Amber => ServiceStatus::Degraded,
                Rag::Red => ServiceStatus::Unhealthy,
            }
        };
        Self::queue_health_component(status, build_remediation_message(&v))
    }

    /// Assemble a `queue_processor` ComponentHealth with the current timestamp.
    fn queue_health_component(status: ServiceStatus, message: String) -> ComponentHealth {
        ComponentHealth {
            component_name: "queue_processor".to_string(),
            status: status as i32,
            message,
            last_check: Some(prost_types::Timestamp::from(SystemTime::now())),
        }
    }

    /// B1 — live Qdrant reachability, bounded by `qdrant_probe_timeout_secs`. A
    /// missing client (never wired) is treated as reachable so the probe never
    /// raises a false Red; any error or timeout is unreachable.
    async fn probe_qdrant_reachable(&self, cfg: &QueueHealthConfig) -> bool {
        let Some(client) = self.storage_client.as_ref() else {
            return true;
        };
        let bound = Duration::from_secs(cfg.qdrant_probe_timeout_secs);
        matches!(
            tokio::time::timeout(bound, client.test_connection()).await,
            Ok(Ok(true))
        )
    }

    /// B2 — free + total bytes on the state.db volume. The db path comes from the
    /// pool (`pragma_database_list`); the blocking `sysinfo` disk read runs on a
    /// `spawn_blocking` thread so a slow mount table cannot stall the Health RPC.
    async fn read_state_db_disk(&self) -> (Option<u64>, Option<u64>) {
        let Some(pool) = self.db_pool.as_ref() else {
            return (None, None);
        };
        let path: Option<String> =
            sqlx::query_scalar("SELECT file FROM pragma_database_list WHERE name = 'main'")
                .fetch_optional(pool)
                .await
                .ok()
                .flatten();
        match path.filter(|p| !p.is_empty()) {
            Some(p) => tokio::task::spawn_blocking(move || read_disk_space(Path::new(&p)))
                .await
                .unwrap_or((None, None)),
            None => (None, None),
        }
    }
}

/// Build the bounded remediation message for the `queue_processor` component
/// (#133 F7). Each non-green probe contributes one `[<rag> <culprit>] <text>`
/// line; lines are ordered most-severe-first and bounded to 10 lines / 2048
/// bytes (including the trailing `…(N more)` marker) when truncated (the
/// unbounded set stays available via the CLI JSON `remediation` array).
/// Cold-start renders the learning-baseline message; an all-green seeded verdict
/// renders "Running normally".
pub(super) fn build_remediation_message(v: &HealthVerdict) -> String {
    if v.cold_start && v.overall == Rag::Green {
        return "Learning baseline — no measurements yet".to_string();
    }

    let mut lines: Vec<(u8, String)> = v
        .degraded()
        .filter_map(|p| {
            p.remediation.as_ref().map(|text| {
                let rag = match p.rag {
                    Rag::Red => "red",
                    Rag::Amber => "amber",
                    Rag::Green => "green",
                };
                (p.rag.severity(), format!("[{rag} {}] {text}", p.culprit))
            })
        })
        .collect();
    if lines.is_empty() {
        return "Running normally".to_string();
    }
    // Most-severe-first (stable within a severity to keep probe order).
    lines.sort_by_key(|line| std::cmp::Reverse(line.0));

    const MAX_LINES: usize = 10;
    const MAX_BYTES: usize = 2048;
    // Reserve room for the worst-case "\n…(N more)" suffix so the final string
    // (suffix included) never exceeds MAX_BYTES (CR-008).
    const SUFFIX_RESERVE: usize = 24;
    let total = lines.len();
    let mut out = String::new();
    let mut shown = 0;
    for (_, line) in &lines {
        if shown >= MAX_LINES || out.len() + line.len() + 1 + SUFFIX_RESERVE > MAX_BYTES {
            break;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line);
        shown += 1;
    }
    if shown < total {
        out.push_str(&format!("\n…({} more)", total - shown));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::build_remediation_message;
    use workspace_qdrant_core::queue_health::probes::{ProbeResult, DLQ, QDRANT};
    use workspace_qdrant_core::queue_health::HealthVerdict;

    #[test]
    fn cold_start_message() {
        let v = HealthVerdict::from_probes(vec![], true);
        assert!(build_remediation_message(&v).contains("Learning baseline"));
    }

    #[test]
    fn all_green_seeded_is_running_normally() {
        let v = HealthVerdict::from_probes(vec![ProbeResult::green(QDRANT)], false);
        assert_eq!(build_remediation_message(&v), "Running normally");
    }

    #[test]
    fn degraded_lines_are_attributed_and_severity_ordered() {
        let v = HealthVerdict::from_probes(
            vec![
                ProbeResult::amber(DLQ, "stuck"),
                ProbeResult::red(QDRANT, "down"),
            ],
            false,
        );
        let msg = build_remediation_message(&v);
        let lines: Vec<&str> = msg.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("[red qdrant]")); // most severe first
        assert!(lines[1].starts_with("[amber dlq]"));
    }

    #[test]
    fn oversize_set_is_bounded_with_marker_within_2048_bytes() {
        // 40 red probes, each with a long remediation, must bound to 10 lines +
        // a "(N more)" marker and stay within 2048 bytes including the suffix.
        let long = "x".repeat(300);
        let probes: Vec<ProbeResult> = (0..40)
            .map(|_| ProbeResult::red(QDRANT, long.clone()))
            .collect();
        let v = HealthVerdict::from_probes(probes, false);
        let msg = build_remediation_message(&v);
        assert!(msg.contains("more)"), "expected a truncation marker");
        assert!(
            msg.len() <= 2048,
            "message {} bytes exceeds 2048",
            msg.len()
        );
    }
}

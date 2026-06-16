//! `ControlBaselinePersistTask` — slow-lane persistence to `control_baseline`.
//!
//! A `MaintenanceTask` registered with the existing `MaintenanceScheduler`
//! (`loop_state.rs`). On each qualifying idle tick it reads slow-lane values from
//! the `ControlFanout`, upserts them into `control_baseline` (schema v46) with
//! bound parameters, and prunes rows for metrics no longer registered. It is the
//! ONLY writer to that table — control sinks (`EwmaState`) never see a pool, so
//! the daemon-owns-state invariant holds (arch §4a, §5g, §7b).
//!
//! Off the hot path entirely: runs only during `FullIdle`/`QdrantDownIdle`
//! (SQLite-only states). SQL errors `warn!` and return `Done` so the scheduler
//! continues; in-memory lane values survive for the next cycle (arch §7a).

use async_trait::async_trait;
use sqlx::SqlitePool;
use std::collections::BTreeMap;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use super::control_lane::ControlLane;
use super::labels::canonicalize_labels;
use super::{switchboard, MetricId, DESCRIPTORS};
use crate::idle::{IdleState, MaintenanceContext, MaintenanceResult, MaintenanceTask};

/// Orphaned-baseline TTL (secs). Mirrors `QueueHealthConfig::baseline_ttl_secs`'s
/// 30-day default. Runtime-tuning this field needs `QueueHealthConfig` threaded
/// into the maintenance scheduler (built from `UnifiedProcessorConfig`, which
/// does not carry queue-health) — deferred, tracked in #143; the default is used
/// here. The const is compile-time-fixed, so its interpolation into the prune SQL
/// is injection-safe today; bind it as a parameter when #143 makes it runtime.
const BASELINE_TTL_SECS: u64 = 2_592_000;

/// Persists switchboard slow-lane control values during idle windows.
pub struct ControlBaselinePersistTask;

impl ControlBaselinePersistTask {
    pub fn new() -> Self {
        Self
    }

    /// The prune allow-list: exactly the variant names of the `persist: true`
    /// descriptor ids. DERIVED from `DESCRIPTORS` so the prune-gate and the
    /// persist-write-gate share one source and cannot silently diverge (DATA-08)
    /// — a `persist: true`-but-unregistered id would write rows the prune then
    /// deletes (silent persistence failure + cold-start every restart).
    fn registered_ids() -> Vec<&'static str> {
        MetricId::ALL
            .iter()
            .filter(|id| DESCRIPTORS[**id as usize].persist)
            .map(|id| id.variant_name())
            .collect()
    }

    /// Upsert one slow-lane row, bumping `sample_count` on conflict.
    async fn upsert_row(
        pool: &SqlitePool,
        metric_id: &str,
        field: &str,
        labels: &str,
        value: f64,
    ) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"INSERT INTO control_baseline
                   (metric_id, field, labels, lane, value, sample_count, updated_at)
               VALUES (?1, ?2, ?3, 'slow', ?4, 1, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
               ON CONFLICT(metric_id, field, labels, lane) DO UPDATE SET
                   value = excluded.value,
                   sample_count = control_baseline.sample_count + 1,
                   updated_at = excluded.updated_at"#,
        )
        .bind(metric_id)
        .bind(field)
        .bind(labels)
        .bind(value)
        .execute(pool)
        .await?;
        Ok(())
    }

    /// Flush one seeded slow lane, returning its live `(metric_id, labels_json)`
    /// key for TTL-prune protection. An unseeded lane is skipped — never persist
    /// a not-yet-learned baseline (it would round-trip a zero). The label is the
    /// lane's live model when known (embedder, SEC-02), else the caller's default.
    async fn flush_lane(
        pool: &SqlitePool,
        lane: &ControlLane,
        metric_id: &str,
        field: &str,
        model: &str,
    ) -> Option<(String, String)> {
        if !lane.is_seeded() {
            return None;
        }
        let value = lane.read_slow();
        let mut labels = BTreeMap::new();
        labels.insert("model", model);
        let labels_json = canonicalize_labels(&labels);
        if let Err(e) = Self::upsert_row(pool, metric_id, field, &labels_json, value).await {
            warn!("control_baseline persist failed for {metric_id}: {e}");
            return None;
        }
        Some((metric_id.to_string(), labels_json))
    }

    /// Delete aged slow-lane rows whose `(metric_id, labels)` is no longer a live
    /// seeded baseline (DATA-09) — bounds growth across model-label changes
    /// without ever deleting an in-use baseline. `live` is the set just flushed
    /// this cycle. Every value is bound (injection-safe).
    async fn ttl_prune(pool: &SqlitePool, live: &[(String, String)]) -> Result<(), sqlx::Error> {
        let cutoff = format!(
            "updated_at < strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-{BASELINE_TTL_SECS} seconds')"
        );
        if live.is_empty() {
            let sql = format!("DELETE FROM control_baseline WHERE lane = 'slow' AND {cutoff}");
            sqlx::query(&sql).execute(pool).await?;
            return Ok(());
        }
        let keep = vec!["(metric_id = ? AND labels = ?)"; live.len()].join(" OR ");
        let sql = format!(
            "DELETE FROM control_baseline WHERE lane = 'slow' AND {cutoff} AND NOT ({keep})"
        );
        let mut q = sqlx::query(&sql);
        for (metric_id, labels) in live {
            q = q.bind(metric_id).bind(labels);
        }
        q.execute(pool).await?;
        Ok(())
    }

    /// Delete rows whose `metric_id` is no longer registered. Placeholders are
    /// generated from the fixed compile-time id count; every value is bound, so
    /// the statement is injection-safe (arch §4c).
    async fn prune_dead_rows(pool: &SqlitePool) -> Result<(), sqlx::Error> {
        let ids = Self::registered_ids();
        let placeholders = vec!["?"; ids.len()].join(", ");
        let sql = format!("DELETE FROM control_baseline WHERE metric_id NOT IN ({placeholders})");
        let mut q = sqlx::query(&sql);
        for id in &ids {
            q = q.bind(*id);
        }
        q.execute(pool).await?;
        Ok(())
    }
}

impl Default for ControlBaselinePersistTask {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MaintenanceTask for ControlBaselinePersistTask {
    fn name(&self) -> &str {
        "control_baseline_persist"
    }

    fn required_idle_states(&self) -> &[IdleState] {
        // SQLite-only — safe to run while Qdrant is unavailable.
        &[IdleState::FullIdle, IdleState::QdrantDownIdle]
    }

    fn idle_delay_secs(&self) -> u64 {
        60
    }

    fn cooldown_secs(&self) -> u64 {
        300
    }

    async fn run_batch(
        &mut self,
        ctx: &MaintenanceContext<'_>,
        _cancel: &CancellationToken,
    ) -> MaintenanceResult {
        let Some(sw) = switchboard() else {
            return MaintenanceResult::Done;
        };
        let fanout = sw.fanout();

        // Flush every persist:true slow lane that has seeded. The embedder lane
        // carries the live provider model label (SEC-02); the queue-cost lanes
        // are not provider-specific, so they use a fixed "queue" label.
        let mut live: Vec<(String, String)> = Vec::new();
        let embed_model = fanout.embedder_latency.model().unwrap_or("fastembed");
        let flushes = [
            (
                &fanout.embedder_latency,
                "EmbedderLatency",
                "embed_ms",
                embed_model,
            ),
            (&fanout.ms_per_kb, "QueueMsPerKb", "ms_per_kb", "queue"),
            (
                &fanout.throughput,
                "QueueThroughput",
                "bytes_per_sec",
                "queue",
            ),
        ];
        for (lane, metric_id, field, model) in flushes {
            if let Some(key) = Self::flush_lane(ctx.pool, lane, metric_id, field, model).await {
                live.push(key);
            }
        }

        if let Err(e) = Self::prune_dead_rows(ctx.pool).await {
            warn!("control_baseline prune failed: {e}");
        }
        if let Err(e) = Self::ttl_prune(ctx.pool, &live).await {
            warn!("control_baseline TTL prune failed: {e}");
        }

        MaintenanceResult::Done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registered_ids_equals_persist_true_set() {
        // DATA-08: the prune allow-list IS the persist:true descriptor set.
        let registered = ControlBaselinePersistTask::registered_ids();
        let expected: Vec<&'static str> = MetricId::ALL
            .iter()
            .filter(|id| DESCRIPTORS[**id as usize].persist)
            .map(|id| id.variant_name())
            .collect();
        assert_eq!(registered, expected);
        // Pin the concrete set so a stray persist-flag flip is caught.
        assert_eq!(
            registered,
            vec!["EmbedderLatency", "QueueMsPerKb", "QueueThroughput"]
        );
    }

    #[test]
    fn test_dlq_and_batch_not_registered() {
        let registered = ControlBaselinePersistTask::registered_ids();
        assert!(!registered.contains(&"QueueDlqDepth"));
        assert!(!registered.contains(&"EmbedderBatch"));
    }
}

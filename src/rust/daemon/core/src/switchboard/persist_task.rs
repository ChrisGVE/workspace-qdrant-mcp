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

use super::labels::canonicalize_labels;
use super::{switchboard, MetricId};
use crate::idle::{IdleState, MaintenanceContext, MaintenanceResult, MaintenanceTask};

/// Persists switchboard slow-lane control values during idle windows.
pub struct ControlBaselinePersistTask;

impl ControlBaselinePersistTask {
    pub fn new() -> Self {
        Self
    }

    /// The fixed, compile-time set of registered `MetricId` variant names. Used
    /// as the prune allow-list. Expand as control metrics are added.
    fn registered_ids() -> &'static [&'static str] {
        &["EmbedderLatency"]
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

    /// Delete rows whose `metric_id` is no longer registered. Placeholders are
    /// generated from the fixed compile-time id count; every value is bound, so
    /// the statement is injection-safe (arch §4c).
    async fn prune_dead_rows(pool: &SqlitePool) -> Result<(), sqlx::Error> {
        let ids = Self::registered_ids();
        let placeholders = vec!["?"; ids.len()].join(", ");
        let sql = format!("DELETE FROM control_baseline WHERE metric_id NOT IN ({placeholders})");
        let mut q = sqlx::query(&sql);
        for id in ids {
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

        // Persist the embedder-latency slow lane. The label matches the emitter's
        // stable `model` label; the slow lane is written by the EWMA accumulator
        // (queue-health) once that side shares the fanout `Arc`.
        if let Some(value) = sw.fanout().read_slow(MetricId::EmbedderLatency) {
            let mut labels = BTreeMap::new();
            labels.insert("model", "fastembed");
            let labels_json = canonicalize_labels(&labels);
            if let Err(e) =
                Self::upsert_row(ctx.pool, "EmbedderLatency", "embed_ms", &labels_json, value).await
            {
                warn!("control_baseline persist failed: {e}");
            }
        }

        if let Err(e) = Self::prune_dead_rows(ctx.pool).await {
            warn!("control_baseline prune failed: {e}");
        }

        MaintenanceResult::Done
    }
}

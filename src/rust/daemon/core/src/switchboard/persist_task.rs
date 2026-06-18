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

/// Fallback orphaned-baseline TTL (secs) — the 30-day `QueueHealthConfig`
/// default ([`default_baseline_ttl_secs`](crate::config::queue_health)). Used
/// only by [`ControlBaselinePersistTask::new`], the test/back-compat
/// constructor. Production builds the task with the runtime-configured TTL via
/// [`ControlBaselinePersistTask::with_ttl_secs`] (#143).
const DEFAULT_BASELINE_TTL_SECS: u64 = 2_592_000;

/// Persists switchboard slow-lane control values during idle windows.
///
/// Carries the orphaned-baseline TTL (`baseline_ttl_secs`) so the runtime-tuned
/// `QueueHealthConfig::baseline_ttl_secs` is honored by the TTL prune (#143). The
/// cutoff is computed in Rust and bound as a query parameter — never
/// string-interpolated into SQL.
pub struct ControlBaselinePersistTask {
    /// Orphaned-baseline TTL in seconds; an orphaned `control_baseline` slow-lane
    /// row whose `updated_at` is older than `now - baseline_ttl_secs` is pruned.
    baseline_ttl_secs: u64,
}

impl ControlBaselinePersistTask {
    /// Construct with the 30-day default TTL. Test/back-compat path; production
    /// uses [`with_ttl_secs`](Self::with_ttl_secs) so the configured TTL applies.
    pub fn new() -> Self {
        Self::with_ttl_secs(DEFAULT_BASELINE_TTL_SECS)
    }

    /// Construct with an explicit orphaned-baseline TTL (secs), threaded from the
    /// runtime `QueueHealthConfig::baseline_ttl_secs` (#143).
    pub fn with_ttl_secs(baseline_ttl_secs: u64) -> Self {
        Self { baseline_ttl_secs }
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
    /// this cycle; `ttl_secs` is the runtime-configured age threshold (#143).
    ///
    /// The cutoff timestamp (`now - ttl_secs`) is computed in Rust and bound as a
    /// query PARAMETER — never interpolated into SQL — so the column comparison is
    /// `updated_at < ?` with the same ISO-8601 millisecond format the table
    /// stores. Every value is bound (injection-safe).
    ///
    /// Named for the predicate it enforces: prune rows that are BOTH aged past the
    /// TTL AND absent from the live set (R2/#143). Distinct from
    /// [`prune_rows_for_unregistered_metrics`](Self::prune_rows_for_unregistered_metrics),
    /// which keys off the metric registry, not row age.
    async fn prune_aged_rows_not_in_live_set(
        pool: &SqlitePool,
        live: &[(String, String)],
        ttl_secs: u64,
    ) -> Result<(), sqlx::Error> {
        let cutoff = Self::cutoff_timestamp(ttl_secs);
        if live.is_empty() {
            sqlx::query("DELETE FROM control_baseline WHERE lane = 'slow' AND updated_at < ?")
                .bind(&cutoff)
                .execute(pool)
                .await?;
            return Ok(());
        }
        let keep = vec!["(metric_id = ? AND labels = ?)"; live.len()].join(" OR ");
        let sql = format!(
            "DELETE FROM control_baseline \
             WHERE lane = 'slow' AND updated_at < ? AND NOT ({keep})"
        );
        let mut q = sqlx::query(&sql).bind(&cutoff);
        for (metric_id, labels) in live {
            q = q.bind(metric_id).bind(labels);
        }
        q.execute(pool).await?;
        Ok(())
    }

    /// The prune cutoff timestamp: `now - ttl_secs`, formatted to match the
    /// `control_baseline.updated_at` column exactly. SQLite writes that column
    /// with `strftime('%Y-%m-%dT%H:%M:%fZ')`, where `%f` is `SS.sss` (seconds with
    /// three-digit milliseconds), so the chrono mirror is `%S%.3f`. Both are
    /// fixed-width zero-padded UTC strings, so `updated_at < ?` is a correct
    /// lexicographic comparison. Bound as a parameter by
    /// [`prune_aged_rows_not_in_live_set`](Self::prune_aged_rows_not_in_live_set).
    fn cutoff_timestamp(ttl_secs: u64) -> String {
        // Saturate rather than wrap (L5, #143): a `u64` past `i64::MAX` would cast
        // to a NEGATIVE duration, pushing the cutoff into the FUTURE and pruning
        // every slow-lane row. Two guards:
        //   1. `i64::try_from` saturates an out-of-`i64` TTL to `i64::MAX`.
        //   2. `chrono::Duration::try_seconds` returns `None` past chrono's bound
        //      (`i64::MAX` *milliseconds*, i.e. ~`i64::MAX/1000` seconds), so we
        //      never call the panicking `Duration::seconds`. On overflow we fall
        //      back to a 100-year-past cutoff — older than any real row, so the
        //      prune becomes an effective "never prune".
        let ttl = i64::try_from(ttl_secs).unwrap_or(i64::MAX);
        let span =
            chrono::Duration::try_seconds(ttl).unwrap_or_else(|| chrono::Duration::days(365 * 100));
        let cutoff = chrono::Utc::now() - span;
        cutoff.format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string()
    }

    /// Delete rows whose `metric_id` is no longer registered. Placeholders are
    /// generated from the fixed compile-time id count; every value is bound, so
    /// the statement is injection-safe (arch §4c).
    ///
    /// Named for the predicate it enforces: prune rows whose metric is absent from
    /// the live registry (R2/#143). Distinct from
    /// [`prune_aged_rows_not_in_live_set`](Self::prune_aged_rows_not_in_live_set),
    /// which keys off row age, not the registry.
    async fn prune_rows_for_unregistered_metrics(pool: &SqlitePool) -> Result<(), sqlx::Error> {
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

        if let Err(e) = Self::prune_rows_for_unregistered_metrics(ctx.pool).await {
            warn!("control_baseline prune failed: {e}");
        }
        if let Err(e) =
            Self::prune_aged_rows_not_in_live_set(ctx.pool, &live, self.baseline_ttl_secs).await
        {
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

    #[test]
    fn test_new_uses_default_ttl() {
        // The default constructor must carry the 30-day fallback so the
        // back-compat path keeps the historical prune horizon.
        let task = ControlBaselinePersistTask::new();
        assert_eq!(task.baseline_ttl_secs, DEFAULT_BASELINE_TTL_SECS);
        assert_eq!(task.baseline_ttl_secs, 2_592_000);
    }

    #[test]
    fn test_with_ttl_secs_overrides_default() {
        // #143: a runtime-configured TTL must be carried verbatim, not the const.
        let task = ControlBaselinePersistTask::with_ttl_secs(86_400);
        assert_eq!(task.baseline_ttl_secs, 86_400);
    }

    #[test]
    fn test_cutoff_timestamp_saturates_on_overflow() {
        // L5/#143: a TTL past i64::MAX must NOT wrap to a negative (future) cutoff
        // that prunes everything. A saturated cutoff sits far in the PAST, before
        // any real row, so nothing is pruned.
        let cutoff = ControlBaselinePersistTask::cutoff_timestamp(u64::MAX);
        let now = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%S%.3fZ")
            .to_string();
        assert!(
            cutoff < now,
            "saturated cutoff {cutoff} must be in the past, not after now {now}"
        );
    }

    // ── prune_aged_rows_not_in_live_set honors the configured TTL (#143) ─────

    use sqlx::sqlite::SqlitePoolOptions;

    /// In-memory pool with the `control_baseline` table migrated in.
    async fn migrated_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        crate::schema_version::SchemaManager::new(pool.clone())
            .run_migrations()
            .await
            .unwrap();
        pool
    }

    /// Insert a slow-lane row whose `updated_at` is `age_secs` in the past, using
    /// SQLite's own clock+format so the stored timestamp matches the column's
    /// `strftime('%Y-%m-%dT%H:%M:%fZ')` shape exactly.
    async fn insert_aged_row(pool: &SqlitePool, metric_id: &str, age_secs: i64) {
        sqlx::query(
            "INSERT INTO control_baseline \
                 (metric_id, field, labels, lane, value, sample_count, updated_at) \
             VALUES (?1, 'f', '{}', 'slow', 1.0, 1, \
                 strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?2))",
        )
        .bind(metric_id)
        .bind(format!("-{age_secs} seconds"))
        .execute(pool)
        .await
        .unwrap();
    }

    async fn slow_row_count(pool: &SqlitePool) -> i64 {
        sqlx::query_scalar("SELECT COUNT(*) FROM control_baseline WHERE lane = 'slow'")
            .fetch_one(pool)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_ttl_prune_uses_configured_ttl() {
        // Two orphaned rows: one 2 hours old, one 10 days old. With a 1-day TTL
        // only the 10-day row is past the cutoff. This proves the runtime TTL —
        // not the 30-day const — drives the prune (#143).
        let pool = migrated_pool().await;
        insert_aged_row(&pool, "Recent", 2 * 3_600).await;
        insert_aged_row(&pool, "Ancient", 10 * 86_400).await;

        let one_day = 86_400;
        ControlBaselinePersistTask::prune_aged_rows_not_in_live_set(&pool, &[], one_day)
            .await
            .unwrap();

        assert_eq!(
            slow_row_count(&pool).await,
            1,
            "only the row older than the 1-day TTL should be pruned"
        );
        let survivor: String =
            sqlx::query_scalar("SELECT metric_id FROM control_baseline WHERE lane = 'slow'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(survivor, "Recent");
    }

    #[tokio::test]
    async fn test_ttl_prune_default_ttl_keeps_recent_rows() {
        // Under the 30-day default, a 10-day-old orphan survives — guards against
        // a TTL that is silently far shorter than configured.
        let pool = migrated_pool().await;
        insert_aged_row(&pool, "Ancient", 10 * 86_400).await;

        ControlBaselinePersistTask::prune_aged_rows_not_in_live_set(
            &pool,
            &[],
            DEFAULT_BASELINE_TTL_SECS,
        )
        .await
        .unwrap();

        assert_eq!(slow_row_count(&pool).await, 1);
    }

    #[tokio::test]
    async fn test_ttl_prune_never_deletes_live_baseline() {
        // A live (just-flushed) key is protected from the TTL prune even when it
        // is older than the cutoff — DATA-09 must survive the parameterized rewrite.
        let pool = migrated_pool().await;
        insert_aged_row(&pool, "Live", 10 * 86_400).await;
        let live = vec![("Live".to_string(), "{}".to_string())];

        ControlBaselinePersistTask::prune_aged_rows_not_in_live_set(&pool, &live, 86_400)
            .await
            .unwrap();

        assert_eq!(
            slow_row_count(&pool).await,
            1,
            "a live baseline is never pruned regardless of age"
        );
    }

    #[tokio::test]
    async fn test_ttl_prune_zero_ttl_purges_all_orphans() {
        // T6 (#143): a zero TTL means "prune everything aged at all". With no live
        // keys to protect, every slow-lane orphan is removed. Boundary guard so a
        // zero TTL never silently becomes "keep forever".
        let pool = migrated_pool().await;
        insert_aged_row(&pool, "A", 2 * 3_600).await;
        insert_aged_row(&pool, "B", 10 * 86_400).await;
        // A freshly-written row (age 0) — the < comparison must still drop it once
        // a moment passes; give it a 1-second age so the cutoff (now-0) exceeds it.
        insert_aged_row(&pool, "C", 1).await;

        ControlBaselinePersistTask::prune_aged_rows_not_in_live_set(&pool, &[], 0)
            .await
            .unwrap();

        assert_eq!(
            slow_row_count(&pool).await,
            0,
            "a zero TTL prunes every non-live slow-lane orphan"
        );
    }
}

//! Phase-2 unified-queue destination-status gauges (PRD D5).
//!
//! Exposes the per-status distribution of unified-queue items for the two
//! downstream destinations — Qdrant and the search (FTS5) index — sourced from
//! the `qdrant_status` / `search_status` columns. Both columns are a fixed
//! 4-value enum, so label cardinality is bounded; all four values are always
//! emitted (0 when absent) so gauges never get stuck on stale labels.

use std::collections::HashMap;

use once_cell::sync::Lazy;
use prometheus::{IntGaugeVec, Opts};
use sqlx::SqlitePool;

use crate::monitoring::METRICS;

/// Fixed destination-status enum (mirrors the `CHECK` constraint on
/// `unified_queue.qdrant_status` / `search_status`).
pub const DESTINATION_STATUSES: [&str; 4] = ["pending", "in_progress", "done", "failed"];

struct QueueStateMetrics {
    qdrant_status: IntGaugeVec,
    search_status: IntGaugeVec,
}

impl QueueStateMetrics {
    fn new() -> Self {
        let qdrant_status = IntGaugeVec::new(
            Opts::new(
                "wqm_memexd_unified_queue_qdrant_status",
                "Unified-queue items grouped by Qdrant destination status",
            ),
            &["qdrant_status"],
        )
        .expect("metric can be created");
        let search_status = IntGaugeVec::new(
            Opts::new(
                "wqm_memexd_unified_queue_search_status",
                "Unified-queue items grouped by search-index destination status",
            ),
            &["search_status"],
        )
        .expect("metric can be created");

        let r = &METRICS.registry;
        let _ = r.register(Box::new(qdrant_status.clone()));
        let _ = r.register(Box::new(search_status.clone()));

        Self {
            qdrant_status,
            search_status,
        }
    }
}

static QUEUE_STATE: Lazy<QueueStateMetrics> = Lazy::new(QueueStateMetrics::new);

async fn status_counts(
    pool: &SqlitePool,
    column: &str,
) -> Result<HashMap<String, i64>, sqlx::Error> {
    // `column` is one of two compile-time-constant identifiers — never user input.
    let query = format!("SELECT {column}, COUNT(*) FROM unified_queue GROUP BY {column}");
    let rows: Vec<(Option<String>, i64)> = sqlx::query_as(&query).fetch_all(pool).await?;
    Ok(rows
        .into_iter()
        .filter_map(|(status, count)| status.map(|s| (s, count)))
        .collect())
}

/// Snapshot the queue destination-status gauges. Emits all four fixed status
/// values for each destination (0 when absent). No-op when telemetry is off.
pub async fn snapshot_queue_state_metrics(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    if !METRICS.is_enabled() {
        return Ok(());
    }
    let m = &*QUEUE_STATE;

    let qdrant = status_counts(pool, "qdrant_status").await?;
    for status in DESTINATION_STATUSES {
        m.qdrant_status
            .with_label_values(&[status])
            .set(*qdrant.get(status).unwrap_or(&0));
    }

    let search = status_counts(pool, "search_status").await?;
    for status in DESTINATION_STATUSES {
        m.search_status
            .with_label_values(&[status])
            .set(*search.get(status).unwrap_or(&0));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_version::SchemaManager;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn migrated_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        SchemaManager::new(pool.clone())
            .run_migrations()
            .await
            .unwrap();
        pool
    }

    #[test]
    fn status_enum_is_bounded_to_four() {
        assert_eq!(DESTINATION_STATUSES.len(), 4);
    }

    #[tokio::test]
    async fn snapshot_emits_all_four_fixed_statuses() {
        let pool = migrated_pool().await;
        snapshot_queue_state_metrics(&pool).await.unwrap();

        let families = METRICS.registry.gather();
        for name in [
            "wqm_memexd_unified_queue_qdrant_status",
            "wqm_memexd_unified_queue_search_status",
        ] {
            let fam = families
                .iter()
                .find(|f| f.name() == name)
                .unwrap_or_else(|| panic!("{name} family present"));
            // All four fixed labels emitted, none outside the enum.
            assert_eq!(fam.get_metric().len(), 4, "{name} emits all four statuses");
            for metric in fam.get_metric() {
                let status = metric
                    .get_label()
                    .first()
                    .map(|l| l.value())
                    .unwrap_or("");
                assert!(
                    DESTINATION_STATUSES.contains(&status),
                    "unexpected status label: {status}"
                );
            }
        }
    }
}

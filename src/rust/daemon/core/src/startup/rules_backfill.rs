//! Rules Mirror Backfill (Task 24)
//!
//! Backfills the `rules_mirror` SQLite table from the Qdrant rules collection
//! on daemon startup. Ensures the SQLite mirror stays in sync after a fresh
//! install, database reset, or if the mirror was added after points already existed.

use std::sync::Arc;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};
use wqm_common::timestamps;

use crate::storage::StorageClient;

/// Backfill `rules_mirror` table from the Qdrant rules collection.
///
/// Scrolls all points in the `rules` collection and inserts any missing
/// rows into `rules_mirror` using INSERT OR IGNORE (idempotent).
/// This ensures the SQLite mirror stays in sync after a fresh install,
/// database reset, or if the mirror was added after points already existed.
pub async fn backfill_rules_mirror(
    pool: &SqlitePool,
    storage_client: &Arc<StorageClient>,
) -> Result<RulesBackfillStats, String> {
    use qdrant_client::qdrant::{Filter, PointId};

    info!("Starting rules_mirror backfill from Qdrant...");
    let start = std::time::Instant::now();

    let exists = storage_client
        .collection_exists(wqm_common::constants::COLLECTION_RULES)
        .await
        .map_err(|e| format!("Failed to check rules collection: {}", e))?;

    if !exists {
        info!("Rules collection does not exist, skipping backfill");
        return Ok(RulesBackfillStats::default());
    }

    let mut stats = RulesBackfillStats::default();
    let mut offset: Option<PointId> = None;
    let batch_size: u32 = 100;

    loop {
        let points = storage_client
            .scroll_with_filter(
                wqm_common::constants::COLLECTION_RULES,
                Filter::default(),
                batch_size,
                offset.clone(),
            )
            .await
            .map_err(|e| format!("Failed to scroll rules collection: {}", e))?;

        if points.is_empty() {
            break;
        }

        stats.points_scanned += points.len() as u64;

        if let Some(last) = points.last() {
            offset = last.id.clone();
        }

        for point in &points {
            insert_rules_mirror_point(pool, point, &mut stats).await;
        }

        if (points.len() as u32) < batch_size {
            break;
        }
    }

    let elapsed = start.elapsed();
    info!(
        "Rules mirror backfill complete: {} scanned, {} inserted, {} existed, {} skipped (no label), {} errors in {:?}",
        stats.points_scanned, stats.inserted, stats.already_exists, stats.skipped_no_label, stats.errors, elapsed
    );

    Ok(stats)
}

async fn insert_rules_mirror_point(
    pool: &SqlitePool,
    point: &qdrant_client::qdrant::RetrievedPoint,
    stats: &mut RulesBackfillStats,
) {
    let label = match point.payload.get("label").and_then(|v| v.as_str()) {
        Some(l) => l,
        None => {
            debug!("Skipping rules point without label");
            stats.skipped_no_label += 1;
            return;
        }
    };

    let empty = String::new();
    let content = point
        .payload
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or(&empty);
    let scope = point.payload.get("scope").and_then(|v| v.as_str());
    let tenant_id = point.payload.get("tenant_id").and_then(|v| v.as_str());
    let created_at = point.payload.get("created_at").and_then(|v| v.as_str());
    let now = timestamps::now_utc();
    let ts = created_at.unwrap_or(&now);

    let result = sqlx::query(
        "INSERT OR IGNORE INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
    )
    .bind(label)
    .bind(content)
    .bind(scope)
    .bind(tenant_id)
    .bind(ts)
    .bind(ts)
    .execute(pool)
    .await;

    match result {
        Ok(r) => {
            if r.rows_affected() > 0 {
                stats.inserted += 1;
            } else {
                stats.already_exists += 1;
            }
        }
        Err(e) => {
            warn!(
                "Failed to insert rules_mirror row for label={}: {}",
                label, e
            );
            stats.errors += 1;
        }
    }
}

/// Statistics from rules mirror backfill
#[derive(Debug, Clone, Default)]
pub struct RulesBackfillStats {
    /// Total Qdrant points scanned
    pub points_scanned: u64,
    /// New rows inserted into rules_mirror
    pub inserted: u64,
    /// Rows already present (INSERT OR IGNORE)
    pub already_exists: u64,
    /// Points skipped because they lack a label
    pub skipped_no_label: u64,
    /// Insert errors
    pub errors: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rules_backfill_stats_default() {
        let stats = RulesBackfillStats::default();
        assert_eq!(stats.points_scanned, 0);
        assert_eq!(stats.inserted, 0);
        assert_eq!(stats.already_exists, 0);
        assert_eq!(stats.skipped_no_label, 0);
        assert_eq!(stats.errors, 0);
    }

    /// Test that rules_mirror INSERT OR IGNORE is idempotent
    #[tokio::test]
    async fn test_rules_mirror_insert_idempotent() {
        use sqlx::sqlite::SqlitePoolOptions;
        use std::time::Duration;

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .acquire_timeout(Duration::from_secs(5))
            .connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool");

        // Create rules_mirror table
        sqlx::query(crate::watch_folders_schema::CREATE_RULES_MIRROR_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let now = timestamps::now_utc();

        // First insert succeeds
        let r1 = sqlx::query(
            "INSERT OR IGNORE INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        )
        .bind("rule-1")
        .bind("Always use snake_case")
        .bind("global")
        .bind::<Option<&str>>(None)
        .bind(&now)
        .bind(&now)
        .execute(&pool).await.unwrap();
        assert_eq!(r1.rows_affected(), 1);

        // Duplicate insert is silently ignored
        let r2 = sqlx::query(
            "INSERT OR IGNORE INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        )
        .bind("rule-1")
        .bind("Different text")
        .bind("project")
        .bind::<Option<&str>>(None)
        .bind(&now)
        .bind(&now)
        .execute(&pool).await.unwrap();
        assert_eq!(r2.rows_affected(), 0);

        // Original text preserved
        let text: String =
            sqlx::query_scalar("SELECT rule_text FROM rules_mirror WHERE rule_id = 'rule-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(text, "Always use snake_case");

        // Different key inserts fine
        let r3 = sqlx::query(
            "INSERT OR IGNORE INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        )
        .bind("rule-2")
        .bind("Use Rust for performance")
        .bind("global")
        .bind::<Option<&str>>(None)
        .bind(&now)
        .bind(&now)
        .execute(&pool).await.unwrap();
        assert_eq!(r3.rows_affected(), 1);

        // Total count is 2
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM rules_mirror")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 2);
    }
}

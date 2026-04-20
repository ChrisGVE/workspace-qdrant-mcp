//! Rules Mirror Backfill (Task 24)
//!
//! Backfills the `rules_mirror` SQLite table from the Qdrant rules collection
//! on daemon startup. Ensures the SQLite mirror stays in sync after a fresh
//! install, database reset, or if the mirror was added after points already existed.

use std::sync::Arc;

use sqlx::SqlitePool;
use tracing::{debug, info, warn};
use wqm_common::constants::TENANT_GLOBAL;
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

    // Coerce any historically drifted tenant_id/scope values to the canonical
    // TENANT_GLOBAL sentinel before backfilling. Protects against `_global`
    // or `_global_` rows inserted by out-of-tree code or manual edits.
    coerce_legacy_global_values(pool).await;

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

/// Coerce legacy `_global` / `_global_` tenant/scope values to the canonical
/// `TENANT_GLOBAL` sentinel in `rules_mirror` and `scratchpad_mirror`.
///
/// This is a one-shot UPDATE run on daemon startup. Rows inserted by future
/// code always use [`TENANT_GLOBAL`] directly, so this is a no-op after the
/// first pass. Errors are logged but do not fail startup — mirror tables may
/// not yet exist on fresh installs, and the backfill caller tolerates that.
async fn coerce_legacy_global_values(pool: &SqlitePool) {
    const LEGACY_VARIANTS: &[&str] = &["_global", "_global_"];

    for table in &["rules_mirror", "scratchpad_mirror"] {
        for column in &["tenant_id", "scope"] {
            // scratchpad_mirror has no scope column; UPDATEs on missing
            // columns fail cleanly and we log+continue.
            let sql = format!(
                "UPDATE {table} SET {column} = ?1 WHERE {column} IN (?2, ?3)",
                table = table,
                column = column,
            );
            match sqlx::query(&sql)
                .bind(TENANT_GLOBAL)
                .bind(LEGACY_VARIANTS[0])
                .bind(LEGACY_VARIANTS[1])
                .execute(pool)
                .await
            {
                Ok(r) if r.rows_affected() > 0 => {
                    info!(
                        "Coerced {} legacy global value(s) in {}.{} to '{}'",
                        r.rows_affected(),
                        table,
                        column,
                        TENANT_GLOBAL,
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    debug!(
                        "Skipped legacy-global coercion for {}.{}: {}",
                        table, column, e
                    );
                }
            }
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

    /// Legacy `_global` / `_global_` tenant_id and scope values are coerced
    /// to the canonical `global` sentinel on startup. Untouched values and
    /// non-legacy strings are left alone.
    #[tokio::test]
    async fn test_coerce_legacy_global_values() {
        use sqlx::sqlite::SqlitePoolOptions;
        use std::time::Duration;

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .acquire_timeout(Duration::from_secs(5))
            .connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool");

        sqlx::query(crate::watch_folders_schema::CREATE_RULES_MIRROR_SQL)
            .execute(&pool)
            .await
            .unwrap();

        let now = timestamps::now_utc();

        // Row with legacy scope "_global"
        sqlx::query(
            "INSERT INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        )
        .bind("r-legacy-1")
        .bind("legacy-1")
        .bind("_global")
        .bind::<Option<&str>>(None)
        .bind(&now)
        .bind(&now)
        .execute(&pool).await.unwrap();

        // Row with legacy tenant_id "_global_"
        sqlx::query(
            "INSERT INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        )
        .bind("r-legacy-2")
        .bind("legacy-2")
        .bind("global")
        .bind(Some("_global_"))
        .bind(&now)
        .bind(&now)
        .execute(&pool).await.unwrap();

        // Row already canonical — must remain untouched
        sqlx::query(
            "INSERT INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        )
        .bind("r-ok")
        .bind("ok")
        .bind("global")
        .bind(Some("proj-abc"))
        .bind(&now)
        .bind(&now)
        .execute(&pool).await.unwrap();

        coerce_legacy_global_values(&pool).await;

        let scope1: String =
            sqlx::query_scalar("SELECT scope FROM rules_mirror WHERE rule_id = 'r-legacy-1'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(scope1, TENANT_GLOBAL);

        let tenant2: String =
            sqlx::query_scalar("SELECT tenant_id FROM rules_mirror WHERE rule_id = 'r-legacy-2'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(tenant2, TENANT_GLOBAL);

        let tenant_ok: String =
            sqlx::query_scalar("SELECT tenant_id FROM rules_mirror WHERE rule_id = 'r-ok'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(tenant_ok, "proj-abc");
    }
}

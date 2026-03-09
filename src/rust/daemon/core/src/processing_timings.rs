//! Processing timing instrumentation for queue item phases.
//!
//! Records per-phase execution times (parse, embed, extract, graph, upsert, fts5)
//! to the `processing_timings` SQLite table for operational analytics.

use sqlx::SqlitePool;
use tracing::debug;
use wqm_common::timestamps::now_utc;

/// A single phase timing record.
pub struct PhaseTiming {
    pub phase: &'static str,
    pub duration_ms: u64,
}

/// Batch-insert timing records for a processed queue item.
///
/// Errors are logged but never propagated — instrumentation must not
/// block or fail the processing pipeline.
pub async fn record_timings(
    pool: &SqlitePool,
    queue_id: &str,
    item_type: &str,
    op: &str,
    tenant_id: &str,
    collection: &str,
    language: Option<&str>,
    timings: &[PhaseTiming],
) {
    if timings.is_empty() {
        return;
    }

    let now = now_utc();

    let tx = match pool.begin().await {
        Ok(tx) => tx,
        Err(e) => {
            debug!("Failed to begin timing transaction: {}", e);
            return;
        }
    };

    // Use a mutable reference through the transaction
    let mut tx = tx;
    for timing in timings {
        let result = sqlx::query(
            r#"
            INSERT INTO processing_timings
                (queue_id, item_type, op, phase, duration_ms, tenant_id, collection, language, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(queue_id)
        .bind(item_type)
        .bind(op)
        .bind(timing.phase)
        .bind(timing.duration_ms as i64)
        .bind(tenant_id)
        .bind(collection)
        .bind(language)
        .bind(&now)
        .execute(&mut *tx)
        .await;

        if let Err(e) = result {
            debug!("Failed to record processing timing: {}", e);
            return; // Transaction will rollback on drop
        }
    }

    if let Err(e) = tx.commit().await {
        debug!("Failed to commit timing transaction: {}", e);
    }
}

/// Delete timing records older than the retention period.
pub async fn cleanup_old_timings(pool: &SqlitePool, retention_days: i64) {
    let result = sqlx::query(
        "DELETE FROM processing_timings WHERE created_at < datetime('now', ? || ' days')",
    )
    .bind(-retention_days)
    .execute(pool)
    .await;

    match result {
        Ok(r) => {
            if r.rows_affected() > 0 {
                debug!(
                    "Cleaned up {} old processing timing records (>{} days)",
                    r.rows_affected(),
                    retention_days
                );
            }
        }
        Err(e) => {
            debug!("Failed to clean up processing timings: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_record_timings_with_valid_table() {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(
            r#"
            CREATE TABLE processing_timings (
                timing_id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_id TEXT,
                item_type TEXT NOT NULL,
                op TEXT NOT NULL,
                phase TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                language TEXT,
                created_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&pool)
        .await
        .unwrap();

        let timings = vec![
            PhaseTiming {
                phase: "parse",
                duration_ms: 42,
            },
            PhaseTiming {
                phase: "embed",
                duration_ms: 150,
            },
            PhaseTiming {
                phase: "upsert",
                duration_ms: 30,
            },
        ];

        record_timings(
            &pool,
            "q-123",
            "file",
            "add",
            "tenant-1",
            "projects",
            Some("rust"),
            &timings,
        )
        .await;

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM processing_timings")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 3);

        // Verify phases
        let phases: Vec<String> =
            sqlx::query_scalar("SELECT phase FROM processing_timings ORDER BY timing_id")
                .fetch_all(&pool)
                .await
                .unwrap();
        assert_eq!(phases, vec!["parse", "embed", "upsert"]);
    }

    #[tokio::test]
    async fn test_record_timings_empty_noop() {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        // No table — should not error because we return early for empty timings
        record_timings(&pool, "q-123", "file", "add", "t", "projects", None, &[]).await;
    }

    #[tokio::test]
    async fn test_record_timings_missing_table_no_panic() {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        // Table doesn't exist — should log error but not panic
        let timings = vec![PhaseTiming {
            phase: "parse",
            duration_ms: 10,
        }];
        record_timings(
            &pool, "q-123", "file", "add", "t", "projects", None, &timings,
        )
        .await;
    }

    #[tokio::test]
    async fn test_cleanup_old_timings() {
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        sqlx::query(
            r#"
            CREATE TABLE processing_timings (
                timing_id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_id TEXT,
                item_type TEXT NOT NULL,
                op TEXT NOT NULL,
                phase TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                language TEXT,
                created_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&pool)
        .await
        .unwrap();

        // Insert a record with a very old timestamp
        sqlx::query(
            "INSERT INTO processing_timings (queue_id, item_type, op, phase, duration_ms, tenant_id, collection, created_at) VALUES ('old', 'file', 'add', 'parse', 10, 't', 'projects', '2020-01-01T00:00:00Z')",
        )
        .execute(&pool)
        .await
        .unwrap();

        // Insert a recent record
        let now = now_utc();
        sqlx::query(
            "INSERT INTO processing_timings (queue_id, item_type, op, phase, duration_ms, tenant_id, collection, created_at) VALUES ('new', 'file', 'add', 'parse', 10, 't', 'projects', ?)",
        )
        .bind(&now)
        .execute(&pool)
        .await
        .unwrap();

        cleanup_old_timings(&pool, 30).await;

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM processing_timings")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 1); // Only the recent one remains
    }
}

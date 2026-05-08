//! Queue stats queries — single source of truth for queue metrics.

use anyhow::{Context, Result};
use rusqlite::Connection;

/// Queue status breakdown from SQLite `unified_queue`.
#[derive(Debug, Default, Clone)]
pub struct QueueStats {
    pub pending: usize,
    pub in_progress: usize,
    pub done: usize,
    pub failed: usize,
    /// ISO timestamp of the oldest pending item, if any.
    pub oldest_pending: Option<String>,
}

impl QueueStats {
    pub fn total(&self) -> usize {
        self.pending + self.in_progress + self.done + self.failed
    }

    /// Assess queue health: healthy / degraded / unhealthy.
    ///
    /// - **Healthy**: no failed items, oldest pending < 1 hour
    /// - **Degraded**: failed items exist OR oldest pending > 1 hour
    /// - **Unhealthy**: oldest pending > 24 hours OR failed > 10% of total
    pub fn health(&self) -> HealthLevel {
        let age_hours = self.oldest_pending_age_hours();
        let active = self.pending + self.in_progress + self.failed;

        if active == 0 {
            return HealthLevel::Healthy;
        }

        let fail_ratio = if active > 0 {
            self.failed as f64 / active as f64
        } else {
            0.0
        };

        if age_hours > 24.0 || fail_ratio > 0.1 {
            HealthLevel::Unhealthy
        } else if self.failed > 0 || age_hours > 1.0 {
            HealthLevel::Degraded
        } else {
            HealthLevel::Healthy
        }
    }

    /// Human-readable reason for queue health status.
    /// Returns `None` when healthy (no explanation needed).
    pub fn health_reason(&self) -> Option<String> {
        let age_hours = self.oldest_pending_age_hours();
        let active = self.pending + self.in_progress + self.failed;
        if active == 0 {
            return None;
        }
        let fail_ratio = if active > 0 {
            self.failed as f64 / active as f64
        } else {
            0.0
        };

        let mut reasons = Vec::new();
        if age_hours > 24.0 {
            let days = (age_hours / 24.0).floor() as u64;
            let hours = (age_hours % 24.0).floor() as u64;
            if days > 0 {
                reasons.push(format!("oldest pending: {days}d {hours}h (>24h)"));
            } else {
                reasons.push(format!("oldest pending: {hours}h (>24h)"));
            }
        } else if age_hours > 1.0 {
            let hours = age_hours.floor() as u64;
            reasons.push(format!("oldest pending: {hours}h (>1h)"));
        }
        if fail_ratio > 0.1 {
            reasons.push(format!("failed: {:.0}% (>10%)", fail_ratio * 100.0));
        } else if self.failed > 0 {
            reasons.push(format!("{} failed", self.failed));
        }

        if reasons.is_empty() {
            None
        } else {
            Some(reasons.join(", "))
        }
    }

    /// Hours since oldest pending item was created. 0.0 if none.
    fn oldest_pending_age_hours(&self) -> f64 {
        let Some(ref ts) = self.oldest_pending else {
            return 0.0;
        };
        chrono::DateTime::parse_from_rfc3339(ts)
            .map(|dt| {
                let age = chrono::Utc::now().signed_duration_since(dt.with_timezone(&chrono::Utc));
                age.num_seconds() as f64 / 3600.0
            })
            .unwrap_or(0.0)
    }
}

/// Three-level health assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HealthLevel {
    Healthy,
    Degraded,
    Unhealthy,
}

impl HealthLevel {
    pub fn label(self) -> &'static str {
        match self {
            HealthLevel::Healthy => "healthy",
            HealthLevel::Degraded => "degraded",
            HealthLevel::Unhealthy => "unhealthy",
        }
    }

    /// Worst of two health levels.
    pub fn worst(self, other: Self) -> Self {
        self.max(other)
    }
}

/// Get queue stats from SQLite. Single source of truth — do NOT
/// use daemon gRPC metrics for queue counts.
pub fn get_queue_stats(conn: &Connection) -> Result<QueueStats> {
    let mut stats = QueueStats::default();
    let mut stmt = conn
        .prepare("SELECT status, COUNT(*) FROM unified_queue GROUP BY status")
        .context("Failed to query queue stats")?;

    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        })
        .context("Failed to read queue stats")?;

    for row in rows {
        let (status, count) = row.context("Failed to parse queue row")?;
        match status.as_str() {
            "pending" => stats.pending = count,
            "in_progress" => stats.in_progress = count,
            "done" => stats.done = count,
            "failed" => stats.failed = count,
            _ => {} // unknown status, ignore
        }
    }

    // Oldest pending item
    stats.oldest_pending = conn
        .query_row(
            "SELECT created_at FROM unified_queue WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .ok();

    Ok(stats)
}

/// Average total processing time per queue item (ms), computed from
/// the `processing_timings` table by summing all phases per queue_id.
///
/// Returns `None` if no timing data exists.
pub fn get_avg_processing_ms(conn: &Connection) -> Option<f64> {
    conn.query_row(
        "SELECT AVG(total_ms) FROM (\
         SELECT queue_id, SUM(duration_ms) AS total_ms \
         FROM processing_timings GROUP BY queue_id)",
        [],
        |row| row.get::<_, Option<f64>>(0),
    )
    .ok()
    .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE watch_folders (
                watch_id TEXT PRIMARY KEY,
                tenant_id TEXT,
                path TEXT,
                collection TEXT,
                parent_watch_id TEXT,
                is_active INTEGER DEFAULT 1,
                enabled INTEGER DEFAULT 1,
                library_mode TEXT,
                is_paused INTEGER DEFAULT 0,
                is_archived INTEGER DEFAULT 0,
                git_remote_url TEXT,
                created_at TEXT,
                updated_at TEXT,
                last_scan TEXT,
                last_activity_at TEXT,
                follow_symlinks INTEGER DEFAULT 0,
                cleanup_on_disable INTEGER DEFAULT 0
            );
            CREATE TABLE tracked_files (
                file_id INTEGER PRIMARY KEY,
                watch_folder_id TEXT,
                file_path TEXT,
                language TEXT,
                chunk_count INTEGER DEFAULT 0,
                needs_reconcile INTEGER DEFAULT 0,
                reconcile_reason TEXT,
                tenant_id TEXT,
                collection TEXT
            );
            CREATE TABLE unified_queue (
                queue_id TEXT PRIMARY KEY,
                idempotency_key TEXT,
                item_type TEXT,
                op TEXT,
                collection TEXT,
                status TEXT,
                tenant_id TEXT,
                branch TEXT,
                payload_json TEXT,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT,
                lease_until TEXT,
                worker_id TEXT,
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                last_error_at TEXT,
                file_path TEXT
            );",
        )
        .unwrap();
        conn
    }

    #[test]
    fn queue_stats_empty() {
        let conn = setup_test_db();
        let stats = get_queue_stats(&conn).unwrap();
        assert_eq!(stats.total(), 0);
    }

    #[test]
    fn queue_stats_with_data() {
        let conn = setup_test_db();
        conn.execute_batch(
            "INSERT INTO unified_queue (queue_id, status) VALUES ('q1', 'pending');
             INSERT INTO unified_queue (queue_id, status) VALUES ('q2', 'pending');
             INSERT INTO unified_queue (queue_id, status) VALUES ('q3', 'done');
             INSERT INTO unified_queue (queue_id, status) VALUES ('q4', 'failed');",
        )
        .unwrap();
        let stats = get_queue_stats(&conn).unwrap();
        assert_eq!(stats.pending, 2);
        assert_eq!(stats.done, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.total(), 4);
    }

    #[test]
    fn queue_stats_all_tracked_statuses() {
        let conn = setup_test_db();
        conn.execute_batch(
            "INSERT INTO unified_queue (queue_id, status) VALUES ('q1', 'pending');
             INSERT INTO unified_queue (queue_id, status) VALUES ('q2', 'in_progress');
             INSERT INTO unified_queue (queue_id, status) VALUES ('q3', 'done');
             INSERT INTO unified_queue (queue_id, status) VALUES ('q4', 'failed');",
        )
        .unwrap();
        let stats = get_queue_stats(&conn).unwrap();
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.in_progress, 1);
        assert_eq!(stats.done, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.total(), 4);
    }

    #[test]
    fn queue_health_levels() {
        // Empty queue → healthy
        let empty = QueueStats::default();
        assert_eq!(empty.health(), HealthLevel::Healthy);

        // Failed items with low ratio → degraded (1/100 = 1% < 10%)
        let degraded = QueueStats {
            pending: 99,
            failed: 1,
            ..Default::default()
        };
        assert_eq!(degraded.health(), HealthLevel::Degraded);

        // High fail ratio (>10%) → unhealthy
        let unhealthy = QueueStats {
            pending: 1,
            failed: 5,
            ..Default::default()
        };
        assert_eq!(unhealthy.health(), HealthLevel::Unhealthy);
    }
}

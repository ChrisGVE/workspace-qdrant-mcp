//! Metrics History Writer and Query API
//!
//! Provides methods for writing metric snapshots to SQLite and querying
//! historical data with time-range and aggregation filters.
//!
//! Task 544.5: Metrics history writer
//! Task 544.7: Metrics history query API

use chrono::{DateTime, Datelike, Timelike, Utc};
use sqlx::{Row, SqlitePool};
use tracing::{debug, warn};

use crate::metrics_history_schema::{AggregationPeriod, MetricEntry};

/// Errors for metrics history operations
#[derive(Debug, thiserror::Error)]
pub enum MetricsHistoryError {
    #[error("SQLite error: {0}")]
    Sqlx(#[from] sqlx::Error),
}

pub type MetricsHistoryResult<T> = Result<T, MetricsHistoryError>;

/// Write a batch of metric entries atomically
pub async fn write_metrics_batch(
    pool: &SqlitePool,
    entries: &[MetricEntry],
) -> MetricsHistoryResult<usize> {
    if entries.is_empty() {
        return Ok(0);
    }

    let mut tx = pool.begin().await?;
    let mut count = 0;

    for entry in entries {
        sqlx::query(
            "INSERT INTO metrics_history (metric_name, metric_value, metric_labels, timestamp, aggregation_period) \
             VALUES (?1, ?2, ?3, ?4, ?5)"
        )
        .bind(&entry.metric_name)
        .bind(entry.metric_value)
        .bind(&entry.metric_labels)
        .bind(entry.timestamp.to_rfc3339())
        .bind(entry.aggregation_period.as_str())
        .execute(&mut *tx)
        .await?;
        count += 1;
    }

    tx.commit().await?;
    debug!("Wrote {} metric entries to history", count);
    Ok(count)
}

/// Write a snapshot from current Prometheus metrics to history
pub async fn write_snapshot(pool: &SqlitePool) -> MetricsHistoryResult<usize> {
    let snapshot = crate::metrics::MetricsSnapshot::capture();
    let now = Utc::now();
    let mut entries = Vec::new();

    // Uptime
    entries.push(
        MetricEntry::new("uptime_seconds", snapshot.uptime_seconds)
            .with_timestamp(now),
    );

    // Active sessions
    entries.push(
        MetricEntry::new("active_sessions", snapshot.active_sessions as f64)
            .with_timestamp(now),
    );

    // Total sessions lifetime
    entries.push(
        MetricEntry::new("total_sessions_lifetime", snapshot.total_sessions_lifetime as f64)
            .with_timestamp(now),
    );

    // Total items processed
    entries.push(
        MetricEntry::new("total_items_processed", snapshot.total_items_processed as f64)
            .with_timestamp(now),
    );

    // Queue depths by priority
    for (priority, depth) in &snapshot.queue_depths {
        entries.push(
            MetricEntry::new("queue_depth", *depth as f64)
                .with_labels(format!(r#"{{"priority":"{}"}}"#, priority))
                .with_timestamp(now),
        );
    }

    // Error counts by type
    for (error_type, count) in &snapshot.error_counts {
        entries.push(
            MetricEntry::new("ingestion_errors_total", *count as f64)
                .with_labels(format!(r#"{{"error_type":"{}"}}"#, error_type))
                .with_timestamp(now),
        );
    }

    // Tenant document counts
    for (tenant_id, count) in &snapshot.tenant_documents {
        entries.push(
            MetricEntry::new("tenant_documents_total", *count as f64)
                .with_labels(format!(r#"{{"tenant_id":"{}"}}"#, tenant_id))
                .with_timestamp(now),
        );
    }

    write_metrics_batch(pool, &entries).await
}

// ============================================================================
// Query API
// ============================================================================

/// Query parameters for metrics history
#[derive(Debug, Clone)]
pub struct MetricsHistoryQuery {
    pub metric_name: String,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub aggregation_period: Option<AggregationPeriod>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

impl MetricsHistoryQuery {
    pub fn new(metric_name: impl Into<String>) -> Self {
        Self {
            metric_name: metric_name.into(),
            start_time: None,
            end_time: None,
            aggregation_period: None,
            limit: None,
            offset: None,
        }
    }

    pub fn with_time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }

    pub fn with_aggregation(mut self, period: AggregationPeriod) -> Self {
        self.aggregation_period = Some(period);
        self
    }

    pub fn with_limit(mut self, limit: i64) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Query historical metrics
pub async fn query_metrics(
    pool: &SqlitePool,
    query: &MetricsHistoryQuery,
) -> MetricsHistoryResult<Vec<MetricEntry>> {
    let mut sql = String::from(
        "SELECT metric_id, metric_name, metric_value, metric_labels, timestamp, aggregation_period \
         FROM metrics_history WHERE metric_name = ?1"
    );
    let mut param_idx = 2;

    if query.start_time.is_some() {
        sql.push_str(&format!(" AND timestamp >= ?{}", param_idx));
        param_idx += 1;
    }
    if query.end_time.is_some() {
        sql.push_str(&format!(" AND timestamp <= ?{}", param_idx));
        param_idx += 1;
    }
    if query.aggregation_period.is_some() {
        sql.push_str(&format!(" AND aggregation_period = ?{}", param_idx));
    }

    sql.push_str(" ORDER BY timestamp ASC");

    if let Some(limit) = query.limit {
        sql.push_str(&format!(" LIMIT {}", limit));
    }
    if let Some(offset) = query.offset {
        sql.push_str(&format!(" OFFSET {}", offset));
    }

    // Build query dynamically
    let mut q = sqlx::query(&sql).bind(&query.metric_name);

    if let Some(ref start) = query.start_time {
        q = q.bind(start.to_rfc3339());
    }
    if let Some(ref end) = query.end_time {
        q = q.bind(end.to_rfc3339());
    }
    if let Some(ref period) = query.aggregation_period {
        q = q.bind(period.as_str());
    }

    let rows = q.fetch_all(pool).await?;

    let mut results = Vec::with_capacity(rows.len());
    for row in &rows {
        let ts_str: String = row.get("timestamp");
        let timestamp = DateTime::parse_from_rfc3339(&ts_str)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(|_| {
                warn!("Failed to parse timestamp: {}", ts_str);
                Utc::now()
            });

        let agg_str: String = row.get("aggregation_period");
        let aggregation_period = AggregationPeriod::from_str(&agg_str)
            .unwrap_or(AggregationPeriod::Raw);

        results.push(MetricEntry {
            metric_id: Some(row.get("metric_id")),
            metric_name: row.get("metric_name"),
            metric_value: row.get("metric_value"),
            metric_labels: row.get("metric_labels"),
            timestamp,
            aggregation_period,
        });
    }

    Ok(results)
}

/// Get aggregated metrics (AVG, MIN, MAX) over a time range
pub async fn query_aggregated(
    pool: &SqlitePool,
    metric_name: &str,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    source_period: AggregationPeriod,
) -> MetricsHistoryResult<Option<AggregatedMetric>> {
    let row = sqlx::query(
        "SELECT AVG(metric_value) as avg_val, MIN(metric_value) as min_val, \
         MAX(metric_value) as max_val, COUNT(*) as sample_count \
         FROM metrics_history \
         WHERE metric_name = ?1 AND timestamp >= ?2 AND timestamp <= ?3 AND aggregation_period = ?4"
    )
    .bind(metric_name)
    .bind(start_time.to_rfc3339())
    .bind(end_time.to_rfc3339())
    .bind(source_period.as_str())
    .fetch_optional(pool)
    .await?;

    match row {
        Some(r) => {
            let count: i32 = r.get("sample_count");
            if count == 0 {
                Ok(None)
            } else {
                Ok(Some(AggregatedMetric {
                    metric_name: metric_name.to_string(),
                    avg: r.get("avg_val"),
                    min: r.get("min_val"),
                    max: r.get("max_val"),
                    sample_count: count,
                }))
            }
        }
        None => Ok(None),
    }
}

/// Aggregated metric result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AggregatedMetric {
    pub metric_name: String,
    pub avg: f64,
    pub min: f64,
    pub max: f64,
    pub sample_count: i32,
}

/// Delete metrics older than a given timestamp for a specific aggregation period
pub async fn cleanup_old_metrics(
    pool: &SqlitePool,
    period: AggregationPeriod,
    before: DateTime<Utc>,
) -> MetricsHistoryResult<u64> {
    let result = sqlx::query(
        "DELETE FROM metrics_history WHERE aggregation_period = ?1 AND timestamp < ?2"
    )
    .bind(period.as_str())
    .bind(before.to_rfc3339())
    .execute(pool)
    .await?;

    let deleted = result.rows_affected();
    if deleted > 0 {
        debug!("Cleaned up {} old {} metric entries", deleted, period);
    }
    Ok(deleted)
}

/// Get distinct metric names in the history table
pub async fn get_available_metrics(pool: &SqlitePool) -> MetricsHistoryResult<Vec<String>> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT metric_name FROM metrics_history ORDER BY metric_name"
    )
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(|(name,)| name).collect())
}

// ============================================================================
// Aggregation and Retention (Task 544.11-14)
// ============================================================================

/// Retention configuration for metrics history
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RetentionConfig {
    /// Hours to retain raw metrics (default: 24)
    pub raw_hours: i64,
    /// Days to retain hourly aggregates (default: 7)
    pub hourly_days: i64,
    /// Days to retain daily aggregates (default: 30)
    pub daily_days: i64,
    /// Days to retain weekly aggregates (default: 365)
    pub weekly_days: i64,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            raw_hours: 24,
            hourly_days: 7,
            daily_days: 30,
            weekly_days: 365,
        }
    }
}

/// Run aggregation from one period to another.
///
/// Computes AVG for each distinct metric_name in the source period within
/// [start, end), writes the result as a single entry at `start` timestamp
/// in the target period.
pub async fn run_aggregation(
    pool: &SqlitePool,
    source_period: AggregationPeriod,
    target_period: AggregationPeriod,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> MetricsHistoryResult<usize> {
    // Get distinct metric names in the window
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT metric_name FROM metrics_history \
         WHERE aggregation_period = ?1 AND timestamp >= ?2 AND timestamp < ?3"
    )
    .bind(source_period.as_str())
    .bind(start.to_rfc3339())
    .bind(end.to_rfc3339())
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Ok(0);
    }

    let mut entries = Vec::new();
    for (metric_name,) in &rows {
        // Also aggregate per-label variant (group by metric_labels)
        let label_rows = sqlx::query(
            "SELECT metric_labels, AVG(metric_value) as avg_val \
             FROM metrics_history \
             WHERE metric_name = ?1 AND aggregation_period = ?2 \
               AND timestamp >= ?3 AND timestamp < ?4 \
             GROUP BY metric_labels"
        )
        .bind(metric_name)
        .bind(source_period.as_str())
        .bind(start.to_rfc3339())
        .bind(end.to_rfc3339())
        .fetch_all(pool)
        .await?;

        for row in &label_rows {
            let avg: f64 = row.get("avg_val");
            let labels: Option<String> = row.get("metric_labels");
            let mut entry = MetricEntry::new(metric_name.clone(), avg)
                .with_timestamp(start)
                .with_aggregation(target_period);
            if let Some(l) = labels {
                entry = entry.with_labels(l);
            }
            entries.push(entry);
        }
    }

    write_metrics_batch(pool, &entries).await
}

/// Run hourly aggregation for the previous hour
pub async fn aggregate_hourly(pool: &SqlitePool, now: DateTime<Utc>) -> MetricsHistoryResult<usize> {
    let hour_start = now
        .with_minute(0).unwrap()
        .with_second(0).unwrap()
        .with_nanosecond(0).unwrap()
        - chrono::Duration::hours(1);
    let hour_end = hour_start + chrono::Duration::hours(1);

    run_aggregation(pool, AggregationPeriod::Raw, AggregationPeriod::Hourly, hour_start, hour_end).await
}

/// Run daily aggregation for the previous day
pub async fn aggregate_daily(pool: &SqlitePool, now: DateTime<Utc>) -> MetricsHistoryResult<usize> {
    let day_start = (now - chrono::Duration::days(1))
        .with_hour(0).unwrap()
        .with_minute(0).unwrap()
        .with_second(0).unwrap()
        .with_nanosecond(0).unwrap();
    let day_end = day_start + chrono::Duration::days(1);

    run_aggregation(pool, AggregationPeriod::Hourly, AggregationPeriod::Daily, day_start, day_end).await
}

/// Run weekly aggregation for the previous week
pub async fn aggregate_weekly(pool: &SqlitePool, now: DateTime<Utc>) -> MetricsHistoryResult<usize> {
    let week_start = (now - chrono::Duration::weeks(1))
        .with_hour(0).unwrap()
        .with_minute(0).unwrap()
        .with_second(0).unwrap()
        .with_nanosecond(0).unwrap();
    let week_end = week_start + chrono::Duration::weeks(1);

    run_aggregation(pool, AggregationPeriod::Daily, AggregationPeriod::Weekly, week_start, week_end).await
}

/// Apply retention policy: delete old metrics per the configuration
pub async fn apply_retention(
    pool: &SqlitePool,
    config: &RetentionConfig,
    now: DateTime<Utc>,
) -> MetricsHistoryResult<u64> {
    let mut total = 0u64;

    total += cleanup_old_metrics(
        pool,
        AggregationPeriod::Raw,
        now - chrono::Duration::hours(config.raw_hours),
    ).await?;

    total += cleanup_old_metrics(
        pool,
        AggregationPeriod::Hourly,
        now - chrono::Duration::days(config.hourly_days),
    ).await?;

    total += cleanup_old_metrics(
        pool,
        AggregationPeriod::Daily,
        now - chrono::Duration::days(config.daily_days),
    ).await?;

    total += cleanup_old_metrics(
        pool,
        AggregationPeriod::Weekly,
        now - chrono::Duration::days(config.weekly_days),
    ).await?;

    Ok(total)
}

/// Run all due aggregations and retention cleanup.
/// Call this periodically (e.g., every hour) from the daemon.
pub async fn run_maintenance(pool: &SqlitePool, now: DateTime<Utc>) -> MetricsHistoryResult<()> {
    let hourly = aggregate_hourly(pool, now).await?;
    if hourly > 0 {
        debug!("Hourly aggregation produced {} entries", hourly);
    }

    // Daily aggregation (only at ~midnight UTC)
    if now.time().hour() == 0 {
        let daily = aggregate_daily(pool, now).await?;
        if daily > 0 {
            debug!("Daily aggregation produced {} entries", daily);
        }

        // Weekly (only on Mondays at midnight)
        if now.weekday() == chrono::Weekday::Mon {
            let weekly = aggregate_weekly(pool, now).await?;
            if weekly > 0 {
                debug!("Weekly aggregation produced {} entries", weekly);
            }
        }
    }

    let cleaned = apply_retention(pool, &RetentionConfig::default(), now).await?;
    if cleaned > 0 {
        debug!("Retention cleanup removed {} entries", cleaned);
    }

    Ok(())
}

/// Convenience wrapper that calls `run_maintenance` with the current time.
pub async fn run_maintenance_now(pool: &SqlitePool) -> MetricsHistoryResult<()> {
    run_maintenance(pool, Utc::now()).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::time::Duration;
    use chrono::Duration as ChronoDuration;

    async fn create_test_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .acquire_timeout(Duration::from_secs(5))
            .connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool");

        // Create metrics_history table
        sqlx::query(crate::metrics_history_schema::CREATE_METRICS_HISTORY_SQL)
            .execute(&pool)
            .await
            .unwrap();
        for idx in crate::metrics_history_schema::CREATE_METRICS_HISTORY_INDEXES_SQL {
            sqlx::query(idx).execute(&pool).await.unwrap();
        }

        pool
    }

    #[tokio::test]
    async fn test_write_and_query_metrics() {
        let pool = create_test_pool().await;
        let now = Utc::now();

        let entries = vec![
            MetricEntry::new("queue_depth", 100.0).with_timestamp(now),
            MetricEntry::new("queue_depth", 120.0)
                .with_timestamp(now + ChronoDuration::seconds(60)),
            MetricEntry::new("active_sessions", 3.0).with_timestamp(now),
        ];

        let written = write_metrics_batch(&pool, &entries).await.unwrap();
        assert_eq!(written, 3);

        // Query queue_depth
        let query = MetricsHistoryQuery::new("queue_depth");
        let results = query_metrics(&pool, &query).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].metric_value, 100.0);
        assert_eq!(results[1].metric_value, 120.0);
    }

    #[tokio::test]
    async fn test_query_with_time_range() {
        let pool = create_test_pool().await;
        let now = Utc::now();

        let entries = vec![
            MetricEntry::new("test_metric", 1.0)
                .with_timestamp(now - ChronoDuration::hours(3)),
            MetricEntry::new("test_metric", 2.0)
                .with_timestamp(now - ChronoDuration::hours(1)),
            MetricEntry::new("test_metric", 3.0)
                .with_timestamp(now),
        ];

        write_metrics_batch(&pool, &entries).await.unwrap();

        let query = MetricsHistoryQuery::new("test_metric")
            .with_time_range(now - ChronoDuration::hours(2), now);
        let results = query_metrics(&pool, &query).await.unwrap();
        assert_eq!(results.len(), 2); // Only the last two
    }

    #[tokio::test]
    async fn test_query_aggregated() {
        let pool = create_test_pool().await;
        let now = Utc::now();

        let entries = vec![
            MetricEntry::new("latency", 10.0).with_timestamp(now - ChronoDuration::minutes(30)),
            MetricEntry::new("latency", 20.0).with_timestamp(now - ChronoDuration::minutes(20)),
            MetricEntry::new("latency", 30.0).with_timestamp(now - ChronoDuration::minutes(10)),
        ];

        write_metrics_batch(&pool, &entries).await.unwrap();

        let agg = query_aggregated(
            &pool,
            "latency",
            now - ChronoDuration::hours(1),
            now,
            AggregationPeriod::Raw,
        )
        .await
        .unwrap()
        .unwrap();

        assert_eq!(agg.sample_count, 3);
        assert!((agg.avg - 20.0).abs() < 0.001);
        assert_eq!(agg.min, 10.0);
        assert_eq!(agg.max, 30.0);
    }

    #[tokio::test]
    async fn test_cleanup_old_metrics() {
        let pool = create_test_pool().await;
        let now = Utc::now();

        let entries = vec![
            MetricEntry::new("old_metric", 1.0)
                .with_timestamp(now - ChronoDuration::days(10)),
            MetricEntry::new("recent_metric", 2.0)
                .with_timestamp(now),
        ];

        write_metrics_batch(&pool, &entries).await.unwrap();

        let deleted = cleanup_old_metrics(
            &pool,
            AggregationPeriod::Raw,
            now - ChronoDuration::days(1),
        )
        .await
        .unwrap();

        assert_eq!(deleted, 1);

        // Verify recent metric still exists
        let query = MetricsHistoryQuery::new("recent_metric");
        let results = query_metrics(&pool, &query).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_get_available_metrics() {
        let pool = create_test_pool().await;
        let now = Utc::now();

        let entries = vec![
            MetricEntry::new("beta_metric", 1.0).with_timestamp(now),
            MetricEntry::new("alpha_metric", 2.0).with_timestamp(now),
            MetricEntry::new("beta_metric", 3.0).with_timestamp(now),
        ];

        write_metrics_batch(&pool, &entries).await.unwrap();

        let names = get_available_metrics(&pool).await.unwrap();
        assert_eq!(names, vec!["alpha_metric", "beta_metric"]);
    }

    #[tokio::test]
    async fn test_write_empty_batch() {
        let pool = create_test_pool().await;
        let written = write_metrics_batch(&pool, &[]).await.unwrap();
        assert_eq!(written, 0);
    }

    #[tokio::test]
    async fn test_query_no_results() {
        let pool = create_test_pool().await;
        let query = MetricsHistoryQuery::new("nonexistent");
        let results = query_metrics(&pool, &query).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_query_aggregated_no_data() {
        let pool = create_test_pool().await;
        let now = Utc::now();
        let result = query_aggregated(
            &pool,
            "nonexistent",
            now - ChronoDuration::hours(1),
            now,
            AggregationPeriod::Raw,
        )
        .await
        .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_query_with_limit() {
        let pool = create_test_pool().await;
        let now = Utc::now();

        let entries: Vec<MetricEntry> = (0..10)
            .map(|i| {
                MetricEntry::new("many", i as f64)
                    .with_timestamp(now + ChronoDuration::seconds(i))
            })
            .collect();

        write_metrics_batch(&pool, &entries).await.unwrap();

        let query = MetricsHistoryQuery::new("many").with_limit(3);
        let results = query_metrics(&pool, &query).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].metric_value, 0.0);
    }

    #[tokio::test]
    async fn test_run_aggregation() {
        let pool = create_test_pool().await;
        let base = Utc::now().with_nanosecond(0).unwrap();
        let hour_start = base - ChronoDuration::hours(1);

        // Insert raw metrics spanning the previous hour
        let entries = vec![
            MetricEntry::new("cpu", 10.0).with_timestamp(hour_start + ChronoDuration::minutes(10)),
            MetricEntry::new("cpu", 20.0).with_timestamp(hour_start + ChronoDuration::minutes(30)),
            MetricEntry::new("cpu", 30.0).with_timestamp(hour_start + ChronoDuration::minutes(50)),
        ];
        write_metrics_batch(&pool, &entries).await.unwrap();

        // Run aggregation
        let count = run_aggregation(
            &pool,
            AggregationPeriod::Raw,
            AggregationPeriod::Hourly,
            hour_start,
            base,
        ).await.unwrap();
        assert_eq!(count, 1); // One metric aggregated

        // Query the hourly aggregate
        let query = MetricsHistoryQuery::new("cpu")
            .with_aggregation(AggregationPeriod::Hourly);
        let results = query_metrics(&pool, &query).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!((results[0].metric_value - 20.0).abs() < 0.001); // AVG(10,20,30) = 20
    }

    #[tokio::test]
    async fn test_aggregation_preserves_labels() {
        let pool = create_test_pool().await;
        let base = Utc::now().with_nanosecond(0).unwrap();
        let start = base - ChronoDuration::hours(1);

        let entries = vec![
            MetricEntry::new("queue", 5.0)
                .with_labels(r#"{"priority":"high"}"#)
                .with_timestamp(start + ChronoDuration::minutes(10)),
            MetricEntry::new("queue", 15.0)
                .with_labels(r#"{"priority":"high"}"#)
                .with_timestamp(start + ChronoDuration::minutes(30)),
            MetricEntry::new("queue", 100.0)
                .with_labels(r#"{"priority":"low"}"#)
                .with_timestamp(start + ChronoDuration::minutes(20)),
        ];
        write_metrics_batch(&pool, &entries).await.unwrap();

        let count = run_aggregation(
            &pool,
            AggregationPeriod::Raw,
            AggregationPeriod::Hourly,
            start,
            base,
        ).await.unwrap();
        assert_eq!(count, 2); // Two label groups

        let query = MetricsHistoryQuery::new("queue")
            .with_aggregation(AggregationPeriod::Hourly);
        let results = query_metrics(&pool, &query).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_apply_retention() {
        let pool = create_test_pool().await;
        let now = Utc::now();

        // Insert old raw metrics (2 days old)
        let old_entries = vec![
            MetricEntry::new("old_raw", 1.0)
                .with_timestamp(now - ChronoDuration::days(2)),
        ];
        write_metrics_batch(&pool, &old_entries).await.unwrap();

        // Insert recent raw metrics
        let new_entries = vec![
            MetricEntry::new("new_raw", 2.0)
                .with_timestamp(now - ChronoDuration::hours(1)),
        ];
        write_metrics_batch(&pool, &new_entries).await.unwrap();

        let config = RetentionConfig {
            raw_hours: 24,
            hourly_days: 7,
            daily_days: 30,
            weekly_days: 365,
        };

        let cleaned = apply_retention(&pool, &config, now).await.unwrap();
        assert_eq!(cleaned, 1); // Only old_raw deleted

        // Verify new_raw still exists
        let query = MetricsHistoryQuery::new("new_raw");
        let results = query_metrics(&pool, &query).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_retention_config_default() {
        let config = RetentionConfig::default();
        assert_eq!(config.raw_hours, 24);
        assert_eq!(config.hourly_days, 7);
        assert_eq!(config.daily_days, 30);
        assert_eq!(config.weekly_days, 365);
    }
}

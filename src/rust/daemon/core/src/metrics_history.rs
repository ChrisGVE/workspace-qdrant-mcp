//! Metrics History Writer and Query API
//!
//! Provides methods for writing metric snapshots to SQLite and querying
//! historical data with time-range and aggregation filters.
//!
//! Task 544.5: Metrics history writer
//! Task 544.7: Metrics history query API

use chrono::{DateTime, Utc};
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
        param_idx += 1;
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
}

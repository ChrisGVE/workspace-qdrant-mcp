//! Metrics History Writer and Query API
//!
//! Provides methods for writing metric snapshots to SQLite and querying
//! historical data with time-range and aggregation filters.

use chrono::{DateTime, Utc};
use sqlx::{Row, SqlitePool};
use tracing::{debug, warn};
use wqm_common::timestamps;

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
        .bind(timestamps::format_utc(&entry.timestamp))
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
    let snapshot = super::metrics_server::MetricsSnapshot::capture();
    let now = Utc::now();
    let mut entries = Vec::new();

    entries.push(MetricEntry::new("uptime_seconds", snapshot.uptime_seconds).with_timestamp(now));
    entries.push(
        MetricEntry::new("active_sessions", snapshot.active_sessions as f64).with_timestamp(now),
    );
    entries.push(
        MetricEntry::new(
            "total_sessions_lifetime",
            snapshot.total_sessions_lifetime as f64,
        )
        .with_timestamp(now),
    );
    entries.push(
        MetricEntry::new(
            "total_items_processed",
            snapshot.total_items_processed as f64,
        )
        .with_timestamp(now),
    );

    for (priority, depth) in &snapshot.queue_depths {
        entries.push(
            MetricEntry::new("queue_depth", *depth as f64)
                .with_labels(format!(r#"{{"priority":"{}"}}"#, priority))
                .with_timestamp(now),
        );
    }

    for (error_type, count) in &snapshot.error_counts {
        entries.push(
            MetricEntry::new("ingestion_errors_total", *count as f64)
                .with_labels(format!(r#"{{"error_type":"{}"}}"#, error_type))
                .with_timestamp(now),
        );
    }

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

    let mut q = sqlx::query(&sql).bind(&query.metric_name);

    if let Some(ref start) = query.start_time {
        q = q.bind(timestamps::format_utc(start));
    }
    if let Some(ref end) = query.end_time {
        q = q.bind(timestamps::format_utc(end));
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
        let aggregation_period =
            AggregationPeriod::from_str(&agg_str).unwrap_or(AggregationPeriod::Raw);

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
    .bind(timestamps::format_utc(&start_time))
    .bind(timestamps::format_utc(&end_time))
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
    let result =
        sqlx::query("DELETE FROM metrics_history WHERE aggregation_period = ?1 AND timestamp < ?2")
            .bind(period.as_str())
            .bind(timestamps::format_utc(&before))
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
    let rows: Vec<(String,)> =
        sqlx::query_as("SELECT DISTINCT metric_name FROM metrics_history ORDER BY metric_name")
            .fetch_all(pool)
            .await?;

    Ok(rows.into_iter().map(|(name,)| name).collect())
}

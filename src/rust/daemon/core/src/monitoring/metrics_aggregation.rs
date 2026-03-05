//! Metrics aggregation and retention.
//!
//! Roll-up from raw → hourly → daily → weekly and retention-based cleanup.

use chrono::{DateTime, Datelike, Timelike, Utc};
use sqlx::{Row, SqlitePool};
use tracing::debug;
use wqm_common::timestamps;

use super::metrics_history::{cleanup_old_metrics, write_metrics_batch, MetricsHistoryResult};
use crate::metrics_history_schema::{AggregationPeriod, MetricEntry};

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
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT metric_name FROM metrics_history \
         WHERE aggregation_period = ?1 AND timestamp >= ?2 AND timestamp < ?3",
    )
    .bind(source_period.as_str())
    .bind(timestamps::format_utc(&start))
    .bind(timestamps::format_utc(&end))
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Ok(0);
    }

    let mut entries = Vec::new();
    for (metric_name,) in &rows {
        aggregate_single_metric(
            pool,
            metric_name,
            source_period,
            target_period,
            start,
            end,
            &mut entries,
        )
        .await?;
    }

    write_metrics_batch(pool, &entries).await
}

async fn aggregate_single_metric(
    pool: &SqlitePool,
    metric_name: &str,
    source_period: AggregationPeriod,
    target_period: AggregationPeriod,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    entries: &mut Vec<MetricEntry>,
) -> MetricsHistoryResult<()> {
    let label_rows = sqlx::query(
        "SELECT metric_labels, AVG(metric_value) as avg_val \
         FROM metrics_history \
         WHERE metric_name = ?1 AND aggregation_period = ?2 \
           AND timestamp >= ?3 AND timestamp < ?4 \
         GROUP BY metric_labels",
    )
    .bind(metric_name)
    .bind(source_period.as_str())
    .bind(timestamps::format_utc(&start))
    .bind(timestamps::format_utc(&end))
    .fetch_all(pool)
    .await?;

    for row in &label_rows {
        let avg: f64 = row.get("avg_val");
        let labels: Option<String> = row.get("metric_labels");
        let mut entry = MetricEntry::new(metric_name, avg)
            .with_timestamp(start)
            .with_aggregation(target_period);
        if let Some(l) = labels {
            entry = entry.with_labels(l);
        }
        entries.push(entry);
    }

    Ok(())
}

/// Run hourly aggregation for the previous hour
pub async fn aggregate_hourly(
    pool: &SqlitePool,
    now: DateTime<Utc>,
) -> MetricsHistoryResult<usize> {
    let hour_start = now
        .with_minute(0)
        .unwrap()
        .with_second(0)
        .unwrap()
        .with_nanosecond(0)
        .unwrap()
        - chrono::Duration::hours(1);
    let hour_end = hour_start + chrono::Duration::hours(1);

    run_aggregation(
        pool,
        AggregationPeriod::Raw,
        AggregationPeriod::Hourly,
        hour_start,
        hour_end,
    )
    .await
}

/// Run daily aggregation for the previous day
pub async fn aggregate_daily(pool: &SqlitePool, now: DateTime<Utc>) -> MetricsHistoryResult<usize> {
    let day_start = (now - chrono::Duration::days(1))
        .with_hour(0)
        .unwrap()
        .with_minute(0)
        .unwrap()
        .with_second(0)
        .unwrap()
        .with_nanosecond(0)
        .unwrap();
    let day_end = day_start + chrono::Duration::days(1);

    run_aggregation(
        pool,
        AggregationPeriod::Hourly,
        AggregationPeriod::Daily,
        day_start,
        day_end,
    )
    .await
}

/// Run weekly aggregation for the previous week
pub async fn aggregate_weekly(
    pool: &SqlitePool,
    now: DateTime<Utc>,
) -> MetricsHistoryResult<usize> {
    let week_start = (now - chrono::Duration::weeks(1))
        .with_hour(0)
        .unwrap()
        .with_minute(0)
        .unwrap()
        .with_second(0)
        .unwrap()
        .with_nanosecond(0)
        .unwrap();
    let week_end = week_start + chrono::Duration::weeks(1);

    run_aggregation(
        pool,
        AggregationPeriod::Daily,
        AggregationPeriod::Weekly,
        week_start,
        week_end,
    )
    .await
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
    )
    .await?;

    total += cleanup_old_metrics(
        pool,
        AggregationPeriod::Hourly,
        now - chrono::Duration::days(config.hourly_days),
    )
    .await?;

    total += cleanup_old_metrics(
        pool,
        AggregationPeriod::Daily,
        now - chrono::Duration::days(config.daily_days),
    )
    .await?;

    total += cleanup_old_metrics(
        pool,
        AggregationPeriod::Weekly,
        now - chrono::Duration::days(config.weekly_days),
    )
    .await?;

    Ok(total)
}

/// Run all due aggregations and retention cleanup.
/// Call this periodically (e.g., every hour) from the daemon.
pub async fn run_maintenance(pool: &SqlitePool, now: DateTime<Utc>) -> MetricsHistoryResult<()> {
    let hourly = aggregate_hourly(pool, now).await?;
    if hourly > 0 {
        debug!("Hourly aggregation produced {} entries", hourly);
    }

    if now.time().hour() == 0 {
        let daily = aggregate_daily(pool, now).await?;
        if daily > 0 {
            debug!("Daily aggregation produced {} entries", daily);
        }

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

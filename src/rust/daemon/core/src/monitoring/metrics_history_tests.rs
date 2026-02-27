//! Tests for metrics_history and metrics_aggregation modules.

use chrono::{Duration as ChronoDuration, Timelike, Utc};
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;
use std::time::Duration;

use crate::metrics_history_schema::{AggregationPeriod, MetricEntry};
use super::metrics_history::*;
use super::metrics_aggregation::*;

async fn create_test_pool() -> SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool");

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
        MetricEntry::new("test_metric", 3.0).with_timestamp(now),
    ];

    write_metrics_batch(&pool, &entries).await.unwrap();

    let query = MetricsHistoryQuery::new("test_metric")
        .with_time_range(now - ChronoDuration::hours(2), now);
    let results = query_metrics(&pool, &query).await.unwrap();
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_query_aggregated() {
    let pool = create_test_pool().await;
    let now = Utc::now();

    let entries = vec![
        MetricEntry::new("latency", 10.0)
            .with_timestamp(now - ChronoDuration::minutes(30)),
        MetricEntry::new("latency", 20.0)
            .with_timestamp(now - ChronoDuration::minutes(20)),
        MetricEntry::new("latency", 30.0)
            .with_timestamp(now - ChronoDuration::minutes(10)),
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
        MetricEntry::new("recent_metric", 2.0).with_timestamp(now),
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

    let entries = vec![
        MetricEntry::new("cpu", 10.0)
            .with_timestamp(hour_start + ChronoDuration::minutes(10)),
        MetricEntry::new("cpu", 20.0)
            .with_timestamp(hour_start + ChronoDuration::minutes(30)),
        MetricEntry::new("cpu", 30.0)
            .with_timestamp(hour_start + ChronoDuration::minutes(50)),
    ];
    write_metrics_batch(&pool, &entries).await.unwrap();

    let count = run_aggregation(
        &pool,
        AggregationPeriod::Raw,
        AggregationPeriod::Hourly,
        hour_start,
        base,
    )
    .await
    .unwrap();
    assert_eq!(count, 1);

    let query =
        MetricsHistoryQuery::new("cpu").with_aggregation(AggregationPeriod::Hourly);
    let results = query_metrics(&pool, &query).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!((results[0].metric_value - 20.0).abs() < 0.001);
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
    )
    .await
    .unwrap();
    assert_eq!(count, 2);

    let query =
        MetricsHistoryQuery::new("queue").with_aggregation(AggregationPeriod::Hourly);
    let results = query_metrics(&pool, &query).await.unwrap();
    assert_eq!(results.len(), 2);
}

#[tokio::test]
async fn test_apply_retention() {
    let pool = create_test_pool().await;
    let now = Utc::now();

    let old_entries = vec![MetricEntry::new("old_raw", 1.0)
        .with_timestamp(now - ChronoDuration::days(2))];
    write_metrics_batch(&pool, &old_entries).await.unwrap();

    let new_entries = vec![MetricEntry::new("new_raw", 2.0)
        .with_timestamp(now - ChronoDuration::hours(1))];
    write_metrics_batch(&pool, &new_entries).await.unwrap();

    let config = RetentionConfig {
        raw_hours: 24,
        hourly_days: 7,
        daily_days: 30,
        weekly_days: 365,
    };

    let cleaned = apply_retention(&pool, &config, now).await.unwrap();
    assert_eq!(cleaned, 1);

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

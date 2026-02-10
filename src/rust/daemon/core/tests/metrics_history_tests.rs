//! Integration tests for metrics history system (Task 544.20)
//!
//! Tests the full workflow: schema creation, metric writing,
//! querying, aggregation, retention, and maintenance.

use chrono::{Duration, Utc};
use sqlx::sqlite::SqlitePoolOptions;
use std::time::Duration as StdDuration;

use workspace_qdrant_core::metrics_history::{
    self, write_metrics_batch, query_metrics, query_aggregated,
    run_aggregation, apply_retention,
    run_maintenance, MetricsHistoryQuery, RetentionConfig,
};
use workspace_qdrant_core::metrics_history_schema::{
    AggregationPeriod, MetricEntry,
};
use workspace_qdrant_core::schema_version::SchemaManager;

async fn setup_test_pool() -> sqlx::SqlitePool {
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(StdDuration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool");

    // Run full schema migrations (creates all tables including metrics_history)
    let manager = SchemaManager::new(pool.clone());
    manager.run_migrations().await.expect("Failed to run migrations");

    pool
}

#[tokio::test]
async fn test_full_metrics_lifecycle() {
    let pool = setup_test_pool().await;
    let now = Utc::now();

    // 1. Write raw metrics
    let entries: Vec<MetricEntry> = (0..10)
        .map(|i| {
            MetricEntry::new("queue_depth", (i * 10) as f64)
                .with_timestamp(now - Duration::minutes(60 - i))
        })
        .collect();

    let written = write_metrics_batch(&pool, &entries).await.unwrap();
    assert_eq!(written, 10);

    // 2. Query raw metrics
    let query = MetricsHistoryQuery::new("queue_depth")
        .with_time_range(now - Duration::hours(2), now);
    let results = query_metrics(&pool, &query).await.unwrap();
    assert_eq!(results.len(), 10);

    // 3. Get aggregated view
    let agg = query_aggregated(
        &pool,
        "queue_depth",
        now - Duration::hours(2),
        now,
        AggregationPeriod::Raw,
    )
    .await
    .unwrap()
    .unwrap();

    assert_eq!(agg.sample_count, 10);
    assert_eq!(agg.min, 0.0);
    assert_eq!(agg.max, 90.0);
    assert!((agg.avg - 45.0).abs() < 0.001);

    // 4. Available metrics
    let names = metrics_history::get_available_metrics(&pool).await.unwrap();
    assert!(names.contains(&"queue_depth".to_string()));
}

#[tokio::test]
async fn test_aggregation_pipeline() {
    let pool = setup_test_pool().await;
    let now = Utc::now();
    let two_hours_ago = now - Duration::hours(2);

    // Insert raw metrics across 2 hours
    let mut entries = Vec::new();
    for i in 0..120 {
        entries.push(
            MetricEntry::new("sessions", (i % 10) as f64)
                .with_timestamp(two_hours_ago + Duration::minutes(i)),
        );
    }
    write_metrics_batch(&pool, &entries).await.unwrap();

    // Run hourly aggregation for the first hour
    let hour1_start = two_hours_ago;
    let hour1_end = hour1_start + Duration::hours(1);
    let count = run_aggregation(
        &pool,
        AggregationPeriod::Raw,
        AggregationPeriod::Hourly,
        hour1_start,
        hour1_end,
    )
    .await
    .unwrap();
    assert!(count > 0);

    // Verify hourly aggregate exists
    let query = MetricsHistoryQuery::new("sessions")
        .with_aggregation(AggregationPeriod::Hourly);
    let hourly = query_metrics(&pool, &query).await.unwrap();
    assert_eq!(hourly.len(), 1);

    // AVG of 0..9 repeated = 4.5
    assert!((hourly[0].metric_value - 4.5).abs() < 0.001);
}

#[tokio::test]
async fn test_retention_with_aggregation() {
    let pool = setup_test_pool().await;
    let now = Utc::now();

    // Insert old raw metrics (2 days old) and recent ones (1 hour old)
    let old_entries = vec![
        MetricEntry::new("old_metric", 1.0)
            .with_timestamp(now - Duration::days(2)),
        MetricEntry::new("old_metric", 2.0)
            .with_timestamp(now - Duration::days(2) + Duration::minutes(30)),
    ];
    let new_entries = vec![
        MetricEntry::new("new_metric", 10.0)
            .with_timestamp(now - Duration::minutes(30)),
    ];
    write_metrics_batch(&pool, &old_entries).await.unwrap();
    write_metrics_batch(&pool, &new_entries).await.unwrap();

    // Apply retention with 24h raw retention
    let config = RetentionConfig::default();
    let cleaned = apply_retention(&pool, &config, now).await.unwrap();
    assert_eq!(cleaned, 2); // Two old entries removed

    // Verify new metric survives
    let query = MetricsHistoryQuery::new("new_metric");
    let results = query_metrics(&pool, &query).await.unwrap();
    assert_eq!(results.len(), 1);

    // Verify old metric is gone
    let query = MetricsHistoryQuery::new("old_metric");
    let results = query_metrics(&pool, &query).await.unwrap();
    assert_eq!(results.len(), 0);
}

#[tokio::test]
async fn test_maintenance_runs_without_error() {
    let pool = setup_test_pool().await;
    let now = Utc::now();

    // Insert some metrics
    let entries = vec![
        MetricEntry::new("test_metric", 42.0).with_timestamp(now),
    ];
    write_metrics_batch(&pool, &entries).await.unwrap();

    // Run maintenance - should not error even with minimal data
    run_maintenance(&pool, now).await.unwrap();
}

#[tokio::test]
async fn test_schema_v5_creates_metrics_history() {
    let pool = setup_test_pool().await;

    // Verify metrics_history table exists after migrations
    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='metrics_history')"
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(exists, "metrics_history table should exist after v5 migration");

    // Verify indexes
    let idx_count: i32 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name LIKE 'idx_metrics_%'"
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(idx_count, 3);
}

#[tokio::test]
async fn test_labeled_metrics_across_lifecycle() {
    let pool = setup_test_pool().await;
    let now = Utc::now();
    let start = now - Duration::hours(1);

    // Write labeled metrics
    let entries = vec![
        MetricEntry::new("errors", 5.0)
            .with_labels(r#"{"type":"parse"}"#)
            .with_timestamp(start + Duration::minutes(10)),
        MetricEntry::new("errors", 3.0)
            .with_labels(r#"{"type":"parse"}"#)
            .with_timestamp(start + Duration::minutes(40)),
        MetricEntry::new("errors", 1.0)
            .with_labels(r#"{"type":"network"}"#)
            .with_timestamp(start + Duration::minutes(20)),
    ];
    write_metrics_batch(&pool, &entries).await.unwrap();

    // Aggregate preserves label grouping
    let count = run_aggregation(
        &pool,
        AggregationPeriod::Raw,
        AggregationPeriod::Hourly,
        start,
        now,
    )
    .await
    .unwrap();
    assert_eq!(count, 2); // Two label groups

    // Query hourly
    let query = MetricsHistoryQuery::new("errors")
        .with_aggregation(AggregationPeriod::Hourly);
    let results = query_metrics(&pool, &query).await.unwrap();
    assert_eq!(results.len(), 2);

    // Find parse errors aggregate (avg of 5,3 = 4.0)
    let parse_agg = results.iter().find(|r| {
        r.metric_labels.as_deref() == Some(r#"{"type":"parse"}"#)
    }).expect("Should find parse label group");
    assert!((parse_agg.metric_value - 4.0).abs() < 0.001);

    // Find network errors aggregate (avg of 1 = 1.0)
    let net_agg = results.iter().find(|r| {
        r.metric_labels.as_deref() == Some(r#"{"type":"network"}"#)
    }).expect("Should find network label group");
    assert!((net_agg.metric_value - 1.0).abs() < 0.001);
}

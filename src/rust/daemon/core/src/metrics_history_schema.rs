//! Metrics History Schema
//!
//! Defines the SQLite schema and types for time-series metrics storage.
//! The daemon periodically snapshots Prometheus metrics into this table
//! for historical querying via CLI and gRPC.
//!
//! Task 544: Status history tracking with time-series storage

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// SQL to create the metrics_history table
pub const CREATE_METRICS_HISTORY_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS metrics_history (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_labels TEXT,                    -- JSON object for dimensions (tenant_id, collection, etc.)
    timestamp TEXT NOT NULL,               -- ISO 8601 UTC
    aggregation_period TEXT NOT NULL DEFAULT 'raw'  -- raw, hourly, daily, weekly
)
"#;

/// Indexes for efficient time-series queries
pub const CREATE_METRICS_HISTORY_INDEXES_SQL: &[&str] = &[
    // Primary query pattern: metric by name over time range
    "CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON metrics_history(metric_name, timestamp)",
    // Aggregated data queries
    "CREATE INDEX IF NOT EXISTS idx_metrics_name_agg_time ON metrics_history(metric_name, aggregation_period, timestamp)",
    // Retention cleanup
    "CREATE INDEX IF NOT EXISTS idx_metrics_agg_time ON metrics_history(aggregation_period, timestamp)",
];

/// Aggregation period for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationPeriod {
    Raw,
    Hourly,
    Daily,
    Weekly,
}

impl AggregationPeriod {
    pub fn as_str(&self) -> &'static str {
        match self {
            AggregationPeriod::Raw => "raw",
            AggregationPeriod::Hourly => "hourly",
            AggregationPeriod::Daily => "daily",
            AggregationPeriod::Weekly => "weekly",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "raw" => Some(AggregationPeriod::Raw),
            "hourly" => Some(AggregationPeriod::Hourly),
            "daily" => Some(AggregationPeriod::Daily),
            "weekly" => Some(AggregationPeriod::Weekly),
            _ => None,
        }
    }
}

impl std::fmt::Display for AggregationPeriod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A single metric entry in the time-series store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricEntry {
    pub metric_id: Option<i64>,
    pub metric_name: String,
    pub metric_value: f64,
    pub metric_labels: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub aggregation_period: AggregationPeriod,
}

impl MetricEntry {
    /// Create a new raw metric entry
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            metric_id: None,
            metric_name: name.into(),
            metric_value: value,
            metric_labels: None,
            timestamp: Utc::now(),
            aggregation_period: AggregationPeriod::Raw,
        }
    }

    /// Add labels as JSON
    pub fn with_labels(mut self, labels: impl Into<String>) -> Self {
        self.metric_labels = Some(labels.into());
        self
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, ts: DateTime<Utc>) -> Self {
        self.timestamp = ts;
        self
    }

    /// Set aggregation period
    pub fn with_aggregation(mut self, period: AggregationPeriod) -> Self {
        self.aggregation_period = period;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sql_is_valid() {
        assert!(CREATE_METRICS_HISTORY_SQL.contains("CREATE TABLE"));
        assert!(CREATE_METRICS_HISTORY_SQL.contains("metrics_history"));
        assert!(CREATE_METRICS_HISTORY_SQL.contains("metric_name TEXT NOT NULL"));
        assert!(CREATE_METRICS_HISTORY_SQL.contains("metric_value REAL NOT NULL"));
        assert!(CREATE_METRICS_HISTORY_SQL.contains("aggregation_period"));
    }

    #[test]
    fn test_indexes_are_defined() {
        assert_eq!(CREATE_METRICS_HISTORY_INDEXES_SQL.len(), 3);
        for idx in CREATE_METRICS_HISTORY_INDEXES_SQL {
            assert!(idx.contains("CREATE INDEX"));
        }
    }

    #[test]
    fn test_aggregation_period_roundtrip() {
        for period in &[
            AggregationPeriod::Raw,
            AggregationPeriod::Hourly,
            AggregationPeriod::Daily,
            AggregationPeriod::Weekly,
        ] {
            let s = period.as_str();
            let parsed = AggregationPeriod::from_str(s).unwrap();
            assert_eq!(*period, parsed);
        }
    }

    #[test]
    fn test_aggregation_period_unknown() {
        assert!(AggregationPeriod::from_str("unknown").is_none());
    }

    #[test]
    fn test_metric_entry_builder() {
        let entry = MetricEntry::new("queue_depth", 42.0)
            .with_labels(r#"{"collection":"projects"}"#);
        assert_eq!(entry.metric_name, "queue_depth");
        assert_eq!(entry.metric_value, 42.0);
        assert!(entry.metric_labels.is_some());
        assert_eq!(entry.aggregation_period, AggregationPeriod::Raw);
        assert!(entry.metric_id.is_none());
    }
}

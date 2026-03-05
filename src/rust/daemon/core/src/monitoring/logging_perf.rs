//! Performance metrics collection and async operation tracking.

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::time::SystemTime;
use tracing::{debug, error};

/// Performance metrics collector for logging
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    operation_counts: HashMap<String, u64>,
    operation_durations: HashMap<String, Vec<f64>>,
    error_counts: HashMap<String, u64>,
}

impl PerformanceMetrics {
    /// Record an operation performance metric
    pub fn record_operation(&mut self, operation: &str, duration_ms: f64) {
        *self
            .operation_counts
            .entry(operation.to_string())
            .or_insert(0) += 1;
        self.operation_durations
            .entry(operation.to_string())
            .or_default()
            .push(duration_ms);
    }

    /// Record an error metric
    pub fn record_error(&mut self, error_type: &str) {
        *self.error_counts.entry(error_type.to_string()).or_insert(0) += 1;
    }

    /// Get performance summary
    pub fn get_summary(&self) -> HashMap<String, serde_json::Value> {
        let mut summary = HashMap::new();

        summary.insert(
            "operation_counts".to_string(),
            serde_json::to_value(&self.operation_counts).unwrap_or_default(),
        );

        let mut operation_stats = HashMap::new();
        for (operation, durations) in &self.operation_durations {
            if !durations.is_empty() {
                let sum: f64 = durations.iter().sum();
                let count = durations.len() as f64;
                let avg = sum / count;

                let mut sorted_durations = durations.clone();
                sorted_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let p50 = sorted_durations[sorted_durations.len() / 2];
                let p95 = sorted_durations[(sorted_durations.len() as f64 * 0.95) as usize];
                let p99 = sorted_durations[(sorted_durations.len() as f64 * 0.99) as usize];

                operation_stats.insert(
                    operation.clone(),
                    serde_json::json!({
                        "avg_ms": avg,
                        "p50_ms": p50,
                        "p95_ms": p95,
                        "p99_ms": p99,
                        "count": count
                    }),
                );
            }
        }
        summary.insert(
            "operation_stats".to_string(),
            serde_json::to_value(operation_stats).unwrap_or_default(),
        );

        summary.insert(
            "error_counts".to_string(),
            serde_json::to_value(&self.error_counts).unwrap_or_default(),
        );

        summary
    }
}

/// Global performance metrics instance
static PERFORMANCE_METRICS: Lazy<std::sync::Mutex<PerformanceMetrics>> =
    Lazy::new(|| std::sync::Mutex::new(PerformanceMetrics::default()));

/// Record operation performance metric
pub fn record_operation_metric(operation: &str, duration_ms: f64) {
    if let Ok(mut metrics) = PERFORMANCE_METRICS.lock() {
        metrics.record_operation(operation, duration_ms);
    }
}

/// Record error metric
pub fn record_error_metric(error_type: &str) {
    if let Ok(mut metrics) = PERFORMANCE_METRICS.lock() {
        metrics.record_error(error_type);
    }
}

/// Get performance metrics summary
pub fn get_performance_metrics() -> HashMap<String, serde_json::Value> {
    PERFORMANCE_METRICS
        .lock()
        .map(|metrics| metrics.get_summary())
        .unwrap_or_default()
}

/// Performance tracking wrapper for async operations
pub async fn track_async_operation<F, T, E>(operation_name: &str, operation: F) -> Result<T, E>
where
    F: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let start_time = SystemTime::now();
    let span = tracing::info_span!(
        "async_operation",
        operation = operation_name,
        start_time = ?start_time
    );

    let result = tracing::Instrument::instrument(operation, span.clone()).await;

    let duration = start_time
        .elapsed()
        .unwrap_or(std::time::Duration::from_millis(0))
        .as_millis() as f64;

    match &result {
        Ok(_) => {
            span.record("duration_ms", duration);
            span.record("success", true);
            record_operation_metric(operation_name, duration);
            debug!(
                operation = operation_name,
                duration_ms = duration,
                "Operation completed successfully"
            );
        }
        Err(e) => {
            span.record("duration_ms", duration);
            span.record("success", false);
            record_error_metric(operation_name);
            error!(
                operation = operation_name,
                duration_ms = duration,
                error = %e,
                "Operation failed"
            );
        }
    }

    result
}

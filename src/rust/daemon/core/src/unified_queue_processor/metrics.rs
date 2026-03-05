//! Metrics helpers and error classification for the unified queue processor.

use chrono::Utc;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

use super::config::UnifiedProcessingMetrics;
use super::error::UnifiedProcessorError;
use super::UnifiedQueueProcessor;

impl UnifiedQueueProcessor {
    /// Update metrics after successful processing
    pub(crate) async fn update_metrics_success(
        metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
        item_type: &str,
        processing_time_ms: u64,
    ) {
        let mut m = metrics.write().await;

        // Increment counter for this item type
        *m.items_processed_by_type
            .entry(item_type.to_string())
            .or_insert(0) += 1;

        // Update average processing time
        let total_items: u64 = m.items_processed_by_type.values().sum();
        let total_items_f = total_items as f64;
        m.avg_processing_time_ms = (m.avg_processing_time_ms * (total_items_f - 1.0)
            + processing_time_ms as f64)
            / total_items_f;

        // Calculate throughput
        let elapsed_secs = (Utc::now() - m.last_update).num_seconds() as f64;
        if elapsed_secs > 0.0 {
            m.items_per_second = total_items_f / elapsed_secs;
        }
    }

    /// Classify a processing error into one of 5 categories:
    /// - `permanent_data`: invalid payload, unsupported format -- no retry
    /// - `permanent_gone`: file deleted, permission denied -- no retry
    /// - `transient_infrastructure`: Qdrant down, network error -- retry with standard backoff
    /// - `transient_resource`: OOM, embedding failure -- retry with longer backoff
    /// - `partial`: partial enrichment -- retry enrichment only
    pub(crate) fn classify_error(error: &UnifiedProcessorError) -> &'static str {
        match error {
            // File doesn't exist or was deleted
            UnifiedProcessorError::FileNotFound(_) => "permanent_gone",
            // Malformed payload -- retrying won't fix the data
            UnifiedProcessorError::InvalidPayload(_) => "permanent_data",
            // Queue operation errors -- check message
            UnifiedProcessorError::QueueOperation(msg) => {
                let lower = msg.to_lowercase();
                if lower.contains("no watch_folder found")
                    || lower.contains("validation")
                    || lower.contains("invalid")
                {
                    "permanent_data"
                } else {
                    "transient_infrastructure"
                }
            }
            // Processing errors -- check message for permanent vs transient
            UnifiedProcessorError::ProcessingFailed(msg) => {
                let lower = msg.to_lowercase();
                if lower.contains("permission denied") || lower.contains("access denied") {
                    "permanent_gone"
                } else if lower.contains("invalid format")
                    || lower.contains("malformed")
                    || lower.contains("unsupported")
                {
                    "permanent_data"
                } else {
                    "transient_infrastructure"
                }
            }
            // Qdrant storage errors -- transient infrastructure
            UnifiedProcessorError::Storage(_) => "transient_infrastructure",
            // Embedding errors -- transient resource (model/memory)
            UnifiedProcessorError::Embedding(_) => "transient_resource",
            // Default: treat as transient infrastructure (retry)
            _ => "transient_infrastructure",
        }
    }

    /// Check if an error category is permanent (should not be retried).
    pub(crate) fn is_permanent_category(category: &str) -> bool {
        category.starts_with("permanent")
    }

    /// Update metrics after processing failure
    pub(crate) async fn update_metrics_failure(
        metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
        error: &UnifiedProcessorError,
    ) {
        let mut m = metrics.write().await;
        m.items_failed += 1;

        let error_type = match error {
            UnifiedProcessorError::InvalidPayload(_) => "invalid_payload",
            UnifiedProcessorError::ProcessingFailed(_) => "processing_failed",
            UnifiedProcessorError::FileNotFound(_) => "file_not_found",
            UnifiedProcessorError::Storage(_) => "storage_error",
            UnifiedProcessorError::Embedding(_) => "embedding_error",
            _ => "other",
        };

        *m.error_counts.entry(error_type.to_string()).or_insert(0) += 1;
    }

    /// Log current processing metrics
    pub(crate) async fn log_metrics(metrics: &Arc<RwLock<UnifiedProcessingMetrics>>) {
        let m = metrics.read().await;

        let total_processed: u64 = m.items_processed_by_type.values().sum();

        info!(
            "Unified Queue Metrics: processed={}, failed={}, queue_depth={}, avg_time={:.2}ms",
            total_processed, m.items_failed, m.queue_depth, m.avg_processing_time_ms,
        );

        if !m.items_processed_by_type.is_empty() {
            debug!("Items by type: {:?}", m.items_processed_by_type);
        }

        if !m.error_counts.is_empty() {
            debug!("Error breakdown: {:?}", m.error_counts);
        }
    }
}

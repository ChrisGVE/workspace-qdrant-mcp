//! Point upsert operations
//!
//! Insert single and batch document points into Qdrant collections.

use std::time::Duration;

use qdrant_client::qdrant::{PointStruct, UpsertPoints};
use tokio::time::sleep;
use tracing::{error, info};

use crate::storage::client::StorageClient;
use crate::storage::convert::convert_to_qdrant_point;
use crate::storage::types::{BatchStats, DocumentPoint, StorageError};

impl StorageClient {
    /// Insert a single document point
    #[tracing::instrument(
        name = "qdrant.insert_point",
        skip_all,
        fields(collection = %collection_name, point_id = %point.id)
    )]
    pub async fn insert_point(
        &self,
        collection_name: &str,
        point: DocumentPoint,
    ) -> Result<(), StorageError> {
        use tracing::debug;
        debug!(
            "Inserting point {} into collection {}",
            point.id, collection_name
        );

        let started = std::time::Instant::now();
        let qdrant_point = convert_to_qdrant_point(point)?;

        let upsert_points = UpsertPoints {
            collection_name: collection_name.to_string(),
            points: vec![qdrant_point],
            wait: Some(true),
            ..Default::default()
        };

        let result = self
            .retry_operation(|| async {
                self.client
                    .upsert_points(upsert_points.clone())
                    .await
                    .map_err(|e| StorageError::Point(e.to_string()))
            })
            .await;
        let elapsed = started.elapsed();
        match &result {
            Ok(_) => {
                crate::monitoring::metrics_core::METRICS.record_qdrant("upsert", elapsed, None);
            }
            Err(_) => {
                crate::monitoring::metrics_core::METRICS.record_qdrant(
                    "upsert",
                    elapsed,
                    Some("upsert_error"),
                );
            }
        }
        result?;

        debug!(
            "Successfully inserted point into collection {}",
            collection_name
        );
        Ok(())
    }

    /// Insert multiple document points in batch
    pub async fn insert_points_batch(
        &self,
        collection_name: &str,
        points: Vec<DocumentPoint>,
        batch_size: Option<usize>,
    ) -> Result<BatchStats, StorageError> {
        self.insert_points_batch_with_wait(collection_name, points, batch_size, false)
            .await
    }

    /// Insert points with explicit wait control.
    ///
    /// When `wait` is true, each batch blocks until Qdrant commits the points.
    #[tracing::instrument(
        name = "qdrant.upsert",
        level = "debug",
        skip_all,
        fields(
            wqm.collection = %collection_name,
            points.count = points.len(),
            rpc.system = "qdrant",
        )
    )]
    pub async fn insert_points_batch_with_wait(
        &self,
        collection_name: &str,
        points: Vec<DocumentPoint>,
        batch_size: Option<usize>,
        wait: bool,
    ) -> Result<BatchStats, StorageError> {
        info!(
            "Inserting {} points into collection {} in batches (wait={})",
            points.len(),
            collection_name,
            wait
        );

        let start_time = std::time::Instant::now();
        let batch_size = batch_size.unwrap_or(100);
        let total_points = points.len();
        let mut successful = 0;
        let mut failed = 0;

        for chunk in points.chunks(batch_size) {
            let qdrant_points: Result<Vec<PointStruct>, _> = chunk
                .iter()
                .map(|p| convert_to_qdrant_point(p.clone()))
                .collect();

            match qdrant_points {
                Ok(points_batch) => {
                    let upsert_points = UpsertPoints {
                        collection_name: collection_name.to_string(),
                        points: points_batch,
                        wait: Some(wait),
                        ..Default::default()
                    };

                    match self
                        .retry_operation(|| async {
                            self.client
                                .upsert_points(upsert_points.clone())
                                .await
                                .map_err(|e| StorageError::Batch(e.to_string()))
                        })
                        .await
                    {
                        Ok(_) => successful += chunk.len(),
                        Err(e) => {
                            error!("Failed to insert batch: {}", e);
                            failed += chunk.len();
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to convert points batch: {}", e);
                    failed += chunk.len();
                }
            }

            // Small delay between batches to avoid overwhelming the server
            sleep(Duration::from_millis(10)).await;
        }

        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        finalize_batch_result(total_points, successful, failed, processing_time_ms)
    }
}

/// Build the final `Result<BatchStats, StorageError>` for a batch upsert.
///
/// **F-032 contract:** when `failed > 0`, return `Err(StorageError::Batch)`
/// even when some chunks succeeded. Returning `Ok(stats)` on partial failure
/// caused every caller (`file/ingest.rs`, `branch_index/tagger.rs`, `text.rs`,
/// `url.rs`) to treat dropped points as success and skip retry metadata.
pub(crate) fn finalize_batch_result(
    total_points: usize,
    successful: usize,
    failed: usize,
    processing_time_ms: u64,
) -> Result<BatchStats, StorageError> {
    let throughput = if processing_time_ms > 0 {
        (successful as f64) / (processing_time_ms as f64 / 1000.0)
    } else {
        0.0
    };

    let stats = BatchStats {
        total_points,
        successful,
        failed,
        processing_time_ms,
        throughput,
    };

    info!(
        "Batch insertion completed: {} successful, {} failed, {:.2} points/sec",
        successful, failed, throughput
    );

    if failed > 0 {
        return Err(StorageError::Batch(format!(
            "partial batch failure: {} of {} points failed (successful={}, time_ms={})",
            failed, total_points, successful, processing_time_ms
        )));
    }

    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalize_batch_result_all_success_returns_ok() {
        let result = finalize_batch_result(5, 5, 0, 100);
        let stats = result.expect("all-success batch must return Ok");
        assert_eq!(stats.total_points, 5);
        assert_eq!(stats.successful, 5);
        assert_eq!(stats.failed, 0);
        assert_eq!(stats.processing_time_ms, 100);
    }

    #[test]
    fn finalize_batch_result_empty_returns_ok() {
        let result = finalize_batch_result(0, 0, 0, 0);
        let stats = result.expect("empty batch must return Ok");
        assert_eq!(stats.total_points, 0);
        assert_eq!(stats.successful, 0);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn finalize_batch_result_partial_failure_returns_err() {
        // F-032 regression: 3 of 5 points failed mid-batch — caller must see
        // Err, not Ok with stats.failed > 0, so retry metadata is populated.
        let result = finalize_batch_result(5, 2, 3, 100);
        let err = result.expect_err("partial failure must return Err");
        match err {
            StorageError::Batch(msg) => {
                assert!(
                    msg.contains("partial batch failure"),
                    "error message should label the partial failure, got: {msg}"
                );
                assert!(msg.contains("3"), "must include failed count, got: {msg}");
                assert!(msg.contains("5"), "must include total count, got: {msg}");
            }
            other => panic!("expected StorageError::Batch, got: {other:?}"),
        }
    }

    #[test]
    fn finalize_batch_result_all_failed_returns_err() {
        let result = finalize_batch_result(4, 0, 4, 50);
        let err = result.expect_err("total failure must return Err");
        assert!(matches!(err, StorageError::Batch(_)));
    }
}

//! Point operations
//!
//! Insert, delete, count, and payload update operations on Qdrant points.
//! Scroll operations are in the sibling `scroll` module.

use std::time::Duration;

use qdrant_client::qdrant::{
    Condition, CountPointsBuilder, DeletePointsBuilder, Filter,
    PointStruct, UpsertPoints,
};
use tokio::time::sleep;
use tracing::{debug, info, error};

use super::client::StorageClient;
use super::convert::{convert_to_qdrant_point, convert_json_to_qdrant_value};
use super::types::{BatchStats, DocumentPoint, StorageError};

impl StorageClient {
    /// Insert a single document point
    pub async fn insert_point(
        &self,
        collection_name: &str,
        point: DocumentPoint,
    ) -> Result<(), StorageError> {
        debug!("Inserting point {} into collection {}", point.id, collection_name);

        let qdrant_point = convert_to_qdrant_point(point)?;

        let upsert_points = UpsertPoints {
            collection_name: collection_name.to_string(),
            points: vec![qdrant_point],
            wait: Some(true),
            ..Default::default()
        };

        self.retry_operation(|| async {
            self.client.upsert_points(upsert_points.clone()).await
                .map_err(|e| StorageError::Point(e.to_string()))
        }).await?;

        debug!("Successfully inserted point into collection {}", collection_name);
        Ok(())
    }

    /// Insert multiple document points in batch
    pub async fn insert_points_batch(
        &self,
        collection_name: &str,
        points: Vec<DocumentPoint>,
        batch_size: Option<usize>,
    ) -> Result<BatchStats, StorageError> {
        self.insert_points_batch_with_wait(collection_name, points, batch_size, false).await
    }

    /// Insert points with explicit wait control.
    /// When `wait` is true, each batch blocks until Qdrant commits the points.
    pub async fn insert_points_batch_with_wait(
        &self,
        collection_name: &str,
        points: Vec<DocumentPoint>,
        batch_size: Option<usize>,
        wait: bool,
    ) -> Result<BatchStats, StorageError> {
        info!("Inserting {} points into collection {} in batches (wait={})", points.len(), collection_name, wait);

        let start_time = std::time::Instant::now();
        let batch_size = batch_size.unwrap_or(100);
        let total_points = points.len();
        let mut successful = 0;
        let mut failed = 0;

        for chunk in points.chunks(batch_size) {
            let qdrant_points: Result<Vec<PointStruct>, _> = chunk.iter()
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

                    match self.retry_operation(|| async {
                        self.client.upsert_points(upsert_points.clone()).await
                            .map_err(|e| StorageError::Batch(e.to_string()))
                    }).await {
                        Ok(_) => successful += chunk.len(),
                        Err(e) => {
                            error!("Failed to insert batch: {}", e);
                            failed += chunk.len();
                        }
                    }
                },
                Err(e) => {
                    error!("Failed to convert points batch: {}", e);
                    failed += chunk.len();
                }
            }

            // Small delay between batches to avoid overwhelming the server
            sleep(Duration::from_millis(10)).await;
        }

        let processing_time_ms = start_time.elapsed().as_millis() as u64;
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

        info!("Batch insertion completed: {} successful, {} failed, {:.2} points/sec",
              successful, failed, throughput);

        Ok(stats)
    }

    /// Delete points from a collection by file_path AND tenant_id filter
    ///
    /// Uses Qdrant's delete_points API with a combined filter condition.
    /// Tenant isolation is enforced to prevent cross-tenant data deletion.
    pub async fn delete_points_by_filter(
        &self,
        collection_name: &str,
        file_path: &str,
        tenant_id: &str,
    ) -> Result<u64, StorageError> {
        if tenant_id.trim().is_empty() {
            return Err(StorageError::Point(
                "tenant_id must not be empty for delete operations".to_string(),
            ));
        }

        info!(
            "Deleting points with file_path='{}' tenant_id='{}' from collection '{}'",
            file_path, tenant_id, collection_name
        );

        let filter = Filter::must([
            Condition::matches("file_path", file_path.to_string()),
            Condition::matches("tenant_id", tenant_id.to_string()),
        ]);

        let count = self.count_points_with_filter(collection_name, filter.clone()).await?;

        let delete_request = DeletePointsBuilder::new(collection_name)
            .points(filter)
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .delete_points(delete_request.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to delete points: {}", e)))
        })
        .await?;

        info!(
            "Deleted {} points with file_path='{}' tenant_id='{}' from '{}'",
            count, file_path, tenant_id, collection_name
        );

        Ok(count)
    }

    /// Delete points from a collection by tenant_id filter
    ///
    /// Deletes all points belonging to a specific tenant/project.
    pub async fn delete_points_by_tenant(
        &self,
        collection_name: &str,
        tenant_id: &str,
    ) -> Result<u64, StorageError> {
        if tenant_id.trim().is_empty() {
            return Err(StorageError::Point(
                "tenant_id must not be empty for tenant delete operations".to_string(),
            ));
        }

        info!(
            "Deleting points with tenant_id='{}' from collection '{}'",
            tenant_id, collection_name
        );

        let filter = Filter::must([Condition::matches("tenant_id", tenant_id.to_string())]);
        let count = self.count_points_with_filter(collection_name, filter.clone()).await?;

        let delete_request = DeletePointsBuilder::new(collection_name)
            .points(filter)
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .delete_points(delete_request.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to delete points by tenant: {}", e)))
        })
        .await?;

        info!(
            "Deleted {} points with tenant_id='{}' from '{}'",
            count, tenant_id, collection_name
        );

        Ok(count)
    }

    /// Delete points from a collection by explicit point IDs
    ///
    /// Uses O(n) direct-address deletion rather than O(collection_size) filter scan.
    pub async fn delete_points_by_ids(
        &self,
        collection_name: &str,
        point_ids: &[String],
    ) -> Result<u64, StorageError> {
        if point_ids.is_empty() {
            return Ok(0);
        }

        let qdrant_ids: Vec<qdrant_client::qdrant::PointId> = point_ids
            .iter()
            .map(|id| qdrant_client::qdrant::PointId {
                point_id_options: Some(
                    qdrant_client::qdrant::point_id::PointIdOptions::Uuid(id.clone()),
                ),
            })
            .collect();

        let count = qdrant_ids.len() as u64;

        let delete_request = DeletePointsBuilder::new(collection_name)
            .points(qdrant_ids)
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .delete_points(delete_request.clone())
                .await
                .map_err(|e| StorageError::Point(
                    format!("Failed to delete {} points by ID from '{}': {}", count, collection_name, e)
                ))
        })
        .await?;

        debug!(
            "Deleted {} points by ID from '{}'",
            count, collection_name
        );

        Ok(count)
    }

    /// Delete points from a collection by document_id filter
    ///
    /// Deletes all chunks belonging to a specific document.
    pub async fn delete_points_by_document_id(
        &self,
        collection_name: &str,
        document_id: &str,
    ) -> Result<u64, StorageError> {
        if document_id.trim().is_empty() {
            return Err(StorageError::Point(
                "document_id must not be empty for delete operations".to_string(),
            ));
        }

        info!(
            "Deleting points with document_id='{}' from collection '{}'",
            document_id, collection_name
        );

        let filter = Filter::must([
            Condition::matches("document_id", document_id.to_string()),
        ]);

        let count = self.count_points_with_filter(collection_name, filter.clone()).await?;

        let delete_request = DeletePointsBuilder::new(collection_name)
            .points(filter)
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .delete_points(delete_request.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to delete points by document_id: {}", e)))
        })
        .await?;

        info!(
            "Deleted {} points with document_id='{}' from '{}'",
            count, document_id, collection_name
        );

        Ok(count)
    }

    /// Delete all points matching an arbitrary payload field name/value pair
    pub async fn delete_points_by_payload_field(
        &self,
        collection_name: &str,
        field_name: &str,
        field_value: &str,
    ) -> Result<u64, StorageError> {
        if field_value.trim().is_empty() {
            return Err(StorageError::Point(
                format!("{} must not be empty for delete operations", field_name),
            ));
        }

        info!(
            "Deleting points with {}='{}' from collection '{}'",
            field_name, field_value, collection_name
        );

        let filter = Filter::must([Condition::matches(field_name, field_value.to_string())]);
        let count = self.count_points_with_filter(collection_name, filter.clone()).await?;

        let delete_request = DeletePointsBuilder::new(collection_name)
            .points(filter)
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .delete_points(delete_request.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to delete points by {}: {}", field_name, e)))
        })
        .await?;

        info!(
            "Deleted {} points with {}='{}' from '{}'",
            count, field_name, field_value, collection_name
        );

        Ok(count)
    }

    /// Update payload fields on all points matching a filter.
    ///
    /// Used for cascade renames where tenant_id needs to be updated.
    pub async fn set_payload_by_filter(
        &self,
        collection_name: &str,
        filter: Filter,
        payload: std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<(), StorageError> {
        use qdrant_client::qdrant::SetPayloadPointsBuilder;

        if !self.collection_exists(collection_name).await? {
            return Err(StorageError::Collection(format!("Collection not found: {}", collection_name)));
        }

        let qdrant_payload: std::collections::HashMap<String, qdrant_client::qdrant::Value> =
            payload.into_iter()
                .map(|(k, v)| (k, convert_json_to_qdrant_value(v)))
                .collect();

        let count = self.count_points_with_filter(collection_name, filter.clone()).await?;
        info!(
            "Updating payload on {} point(s) in collection '{}'",
            count, collection_name
        );

        let set_payload_request = SetPayloadPointsBuilder::new(collection_name, qdrant_payload)
            .points_selector(filter)
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .set_payload(set_payload_request.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to set payload: {}", e)))
        })
        .await?;

        info!(
            "Updated payload on {} point(s) in '{}'",
            count, collection_name
        );

        Ok(())
    }

    /// Count points matching a specific filter (private helper)
    pub(crate) async fn count_points_with_filter(
        &self,
        collection_name: &str,
        filter: Filter,
    ) -> Result<u64, StorageError> {
        let builder = CountPointsBuilder::new(collection_name)
            .filter(filter)
            .exact(true);

        let count = self.retry_operation(|| async {
            self.client.count(builder.clone()).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;

        Ok(count.result.map(|r| r.count).unwrap_or(0))
    }

    /// Count points in a collection, optionally filtered by tenant_id
    pub async fn count_points(
        &self,
        collection_name: &str,
        tenant_id: Option<&str>,
    ) -> Result<u64, StorageError> {
        debug!("Counting points in collection: {} (tenant: {:?})", collection_name, tenant_id);

        let mut builder = CountPointsBuilder::new(collection_name).exact(true);

        if let Some(tid) = tenant_id {
            builder = builder.filter(Filter::must([
                Condition::matches("tenant_id", tid.to_string()),
            ]));
        }

        let count = self.retry_operation(|| async {
            self.client.count(builder.clone()).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;

        Ok(count.result.map(|r| r.count).unwrap_or(0))
    }
}

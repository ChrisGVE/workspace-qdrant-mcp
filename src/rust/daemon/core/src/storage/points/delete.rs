//! Point delete operations
//!
//! Remove points from Qdrant collections by filter, tenant, IDs, document ID,
//! or arbitrary payload field.

use qdrant_client::qdrant::{Condition, DeletePointsBuilder, Filter};
use tracing::{debug, info};

use crate::storage::client::StorageClient;
use crate::storage::types::StorageError;

impl StorageClient {
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

        let op_start = std::time::Instant::now();

        let filter = Filter::must([
            Condition::matches("file_path", file_path.to_string()),
            Condition::matches("tenant_id", tenant_id.to_string()),
        ]);

        let count = self
            .count_points_with_filter(collection_name, filter.clone())
            .await?;

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
            collection = collection_name,
            op = "delete_by_filter",
            point_count = count,
            duration_ms = op_start.elapsed().as_millis() as u64,
            "qdrant delete completed"
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

        let op_start = std::time::Instant::now();

        let filter = Filter::must([Condition::matches("tenant_id", tenant_id.to_string())]);
        let count = self
            .count_points_with_filter(collection_name, filter.clone())
            .await?;

        let delete_request = DeletePointsBuilder::new(collection_name)
            .points(filter)
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .delete_points(delete_request.clone())
                .await
                .map_err(|e| {
                    StorageError::Point(format!("Failed to delete points by tenant: {}", e))
                })
        })
        .await?;

        info!(
            collection = collection_name,
            op = "delete_by_tenant",
            point_count = count,
            duration_ms = op_start.elapsed().as_millis() as u64,
            "qdrant delete completed"
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
                point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(
                    id.clone(),
                )),
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
                .map_err(|e| {
                    StorageError::Point(format!(
                        "Failed to delete {} points by ID from '{}': {}",
                        count, collection_name, e
                    ))
                })
        })
        .await?;

        debug!("Deleted {} points by ID from '{}'", count, collection_name);

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

        let filter = Filter::must([Condition::matches("document_id", document_id.to_string())]);

        let count = self
            .count_points_with_filter(collection_name, filter.clone())
            .await?;

        let delete_request = DeletePointsBuilder::new(collection_name)
            .points(filter)
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .delete_points(delete_request.clone())
                .await
                .map_err(|e| {
                    StorageError::Point(format!("Failed to delete points by document_id: {}", e))
                })
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
            return Err(StorageError::Point(format!(
                "{} must not be empty for delete operations",
                field_name
            )));
        }

        info!(
            "Deleting points with {}='{}' from collection '{}'",
            field_name, field_value, collection_name
        );

        let filter = Filter::must([Condition::matches(field_name, field_value.to_string())]);
        let count = self
            .count_points_with_filter(collection_name, filter.clone())
            .await?;

        let delete_request = DeletePointsBuilder::new(collection_name)
            .points(filter)
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .delete_points(delete_request.clone())
                .await
                .map_err(|e| {
                    StorageError::Point(format!("Failed to delete points by {}: {}", field_name, e))
                })
        })
        .await?;

        info!(
            "Deleted {} points with {}='{}' from '{}'",
            count, field_name, field_value, collection_name
        );

        Ok(count)
    }
}

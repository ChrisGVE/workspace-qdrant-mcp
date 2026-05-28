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

        debug!(
            collection = collection_name,
            op = "delete_by_filter",
            duration_ms = op_start.elapsed().as_millis() as u64,
            "qdrant delete completed"
        );

        Ok(0)
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

    /// Delete all points matching every supplied (field, value) keyword pair.
    ///
    /// All conditions are combined with `must` (logical AND). Use this when
    /// a single field is insufficient for safe scoping — for example,
    /// `(label, "no-foo") AND (tenant_id, "proj-A")` ensures a label-only
    /// remove can never delete a same-label rule belonging to another
    /// project.
    ///
    /// Errors when any field value is empty (mirrors the single-field helper)
    /// or when `fields` is empty (refuses an unconditional delete).
    pub async fn delete_points_by_payload_fields(
        &self,
        collection_name: &str,
        fields: &[(&str, &str)],
    ) -> Result<u64, StorageError> {
        let filter = validate_and_build_payload_field_filter(fields)?;

        info!(
            "Deleting points matching {:?} from collection '{}'",
            fields, collection_name
        );

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
                    StorageError::Point(format!("Failed to delete points by payload fields: {}", e))
                })
        })
        .await?;

        info!(
            "Deleted {} points matching {:?} from '{}'",
            count, fields, collection_name
        );

        Ok(count)
    }

    /// Delete points by a custom filter.
    ///
    /// Used by branch cleanup to delete orphaned points by base_point.
    pub async fn delete_points_with_filter(
        &self,
        collection_name: &str,
        filter: Filter,
    ) -> Result<u64, StorageError> {
        if !self.collection_exists(collection_name).await? {
            return Ok(0);
        }

        let count = self
            .count_points_with_filter(collection_name, filter.clone())
            .await?;

        if count == 0 {
            return Ok(0);
        }

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
            op = "delete_with_filter",
            point_count = count,
            "qdrant delete completed"
        );

        Ok(count)
    }
}

/// Validate the inputs to `delete_points_by_payload_fields` and lower them
/// to a Qdrant `Filter::must([Condition::matches])` clause set.
///
/// Extracted so the safety invariants (no empty list, no empty value) can
/// be unit-tested without a live Qdrant connection.
fn validate_and_build_payload_field_filter(
    fields: &[(&str, &str)],
) -> Result<Filter, StorageError> {
    if fields.is_empty() {
        return Err(StorageError::Point(
            "at least one (field, value) pair is required for delete operations".to_string(),
        ));
    }
    for (name, value) in fields {
        if value.trim().is_empty() {
            return Err(StorageError::Point(format!(
                "{} must not be empty for delete operations",
                name
            )));
        }
    }
    let conditions: Vec<Condition> = fields
        .iter()
        .map(|(name, value)| Condition::matches(*name, value.to_string()))
        .collect();
    Ok(Filter::must(conditions))
}

#[cfg(test)]
mod tests {
    //! Regression tests for the `(field, value)` validation that gates
    //! `delete_points_by_payload_fields`. Multi-field scoping closes F-005
    //! (label-only rule remove crossed project boundaries); these tests
    //! prove the helper refuses unsafe inputs (no fields, empty value).

    use super::*;

    #[test]
    fn validate_payload_filter_rejects_empty_field_list() {
        let result = validate_and_build_payload_field_filter(&[]);
        let err = result.expect_err("empty field list must be rejected");
        match err {
            StorageError::Point(msg) => assert!(
                msg.contains("at least one"),
                "error message must mention required field, got: {msg}"
            ),
            other => panic!("expected StorageError::Point, got {other:?}"),
        }
    }

    #[test]
    fn validate_payload_filter_rejects_empty_value() {
        let result =
            validate_and_build_payload_field_filter(&[("label", "no-foo"), ("tenant_id", "")]);
        let err = result.expect_err("empty value must be rejected");
        match err {
            StorageError::Point(msg) => assert!(
                msg.contains("tenant_id"),
                "error message must identify the offending field, got: {msg}"
            ),
            other => panic!("expected StorageError::Point, got {other:?}"),
        }
    }

    #[test]
    fn validate_payload_filter_rejects_whitespace_only_value() {
        let result =
            validate_and_build_payload_field_filter(&[("label", "   "), ("tenant_id", "proj-a")]);
        assert!(
            result.is_err(),
            "whitespace-only value must be rejected to prevent unscoped delete"
        );
    }

    #[test]
    fn validate_payload_filter_builds_filter_for_valid_pair() {
        let filter = validate_and_build_payload_field_filter(&[
            ("label", "no-foo"),
            ("tenant_id", "proj-a"),
        ])
        .expect("valid input must produce a filter");
        assert_eq!(
            filter.must.len(),
            2,
            "two valid (field, value) pairs must produce two must conditions"
        );
    }

    #[test]
    fn validate_payload_filter_supports_single_pair() {
        let filter = validate_and_build_payload_field_filter(&[("tenant_id", "proj-a")])
            .expect("single valid pair must produce a filter");
        assert_eq!(filter.must.len(), 1);
    }
}

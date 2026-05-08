//! Point count and existence check operations
//!
//! Query point counts and check existence of specific point IDs.

use qdrant_client::qdrant::{Condition, CountPointsBuilder, Filter, GetPointsBuilder};
use tracing::debug;

use crate::storage::client::StorageClient;
use crate::storage::types::StorageError;

impl StorageClient {
    /// Count points matching a specific filter (private helper)
    pub(crate) async fn count_points_with_filter(
        &self,
        collection_name: &str,
        filter: Filter,
    ) -> Result<u64, StorageError> {
        let builder = CountPointsBuilder::new(collection_name)
            .filter(filter)
            .exact(true);

        let count = self
            .retry_operation(|| async {
                self.client
                    .count(builder.clone())
                    .await
                    .map_err(|e| StorageError::Collection(e.to_string()))
            })
            .await?;

        Ok(count.result.map(|r| r.count).unwrap_or(0))
    }

    /// Count points in a collection, optionally filtered by tenant_id
    pub async fn count_points(
        &self,
        collection_name: &str,
        tenant_id: Option<&str>,
    ) -> Result<u64, StorageError> {
        debug!(
            "Counting points in collection: {} (tenant: {:?})",
            collection_name, tenant_id
        );

        let mut builder = CountPointsBuilder::new(collection_name).exact(true);

        if let Some(tid) = tenant_id {
            builder = builder.filter(Filter::must([Condition::matches(
                "tenant_id",
                tid.to_string(),
            )]));
        }

        let count = self
            .retry_operation(|| async {
                self.client
                    .count(builder.clone())
                    .await
                    .map_err(|e| StorageError::Collection(e.to_string()))
            })
            .await?;

        Ok(count.result.map(|r| r.count).unwrap_or(0))
    }

    /// Check which point UUIDs exist in a collection.
    ///
    /// Fetches points by ID with no payload or vectors (minimal overhead).
    /// Returns the set of IDs that actually exist in Qdrant.
    pub async fn check_points_exist(
        &self,
        collection_name: &str,
        point_ids: &[String],
    ) -> Result<std::collections::HashSet<String>, StorageError> {
        use qdrant_client::qdrant::PointId;
        use std::collections::HashSet;

        if point_ids.is_empty() {
            return Ok(HashSet::new());
        }

        let ids: Vec<PointId> = point_ids
            .iter()
            .map(|id| PointId::from(id.as_str()))
            .collect();

        let response = self
            .retry_operation(|| async {
                let builder = GetPointsBuilder::new(collection_name, ids.clone())
                    .with_payload(false)
                    .with_vectors(false);
                self.client
                    .get_points(builder)
                    .await
                    .map_err(|e| StorageError::Point(e.to_string()))
            })
            .await?;

        let existing: HashSet<String> = response
            .result
            .into_iter()
            .filter_map(|p| {
                p.id.and_then(|pid| {
                    use qdrant_client::qdrant::point_id::PointIdOptions;
                    match pid.point_id_options {
                        Some(PointIdOptions::Uuid(u)) => Some(u),
                        Some(PointIdOptions::Num(n)) => Some(n.to_string()),
                        None => None,
                    }
                })
            })
            .collect();

        debug!(
            "check_points_exist: {}/{} points exist in {}",
            existing.len(),
            point_ids.len(),
            collection_name
        );

        Ok(existing)
    }
}

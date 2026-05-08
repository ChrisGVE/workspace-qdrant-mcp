//! Point update operations
//!
//! Update sparse vectors and payload fields on existing Qdrant points.

use qdrant_client::qdrant::Filter;
use tracing::info;

use crate::storage::client::StorageClient;
use crate::storage::convert::convert_json_to_qdrant_value;
use crate::storage::types::StorageError;

impl StorageClient {
    /// Update only the sparse named vector for a batch of points.
    ///
    /// Leaves the dense vector and payload untouched. Used by `rebalance-idf`
    /// to apply IDF correction factors without re-embedding dense vectors.
    pub async fn update_named_sparse_vectors(
        &self,
        collection_name: &str,
        updates: Vec<(String, std::collections::HashMap<u32, f32>)>,
    ) -> Result<(), StorageError> {
        use qdrant_client::qdrant::{
            point_id, vector, vectors, NamedVectors, PointVectors, SparseVector,
            UpdatePointVectorsBuilder, Vector, Vectors,
        };

        if updates.is_empty() {
            return Ok(());
        }

        let point_vectors: Vec<PointVectors> = updates
            .into_iter()
            .map(|(id, sparse_map)| {
                let mut entries: Vec<(u32, f32)> = sparse_map.into_iter().collect();
                entries.sort_by_key(|(idx, _)| *idx);
                let indices: Vec<u32> = entries.iter().map(|(i, _)| *i).collect();
                let values: Vec<f32> = entries.iter().map(|(_, v)| *v).collect();

                let sparse_vec = Vector {
                    vector: Some(vector::Vector::Sparse(SparseVector { indices, values })),
                    ..Default::default()
                };
                let mut named = std::collections::HashMap::new();
                named.insert("sparse".to_string(), sparse_vec);

                PointVectors {
                    id: Some(qdrant_client::qdrant::PointId {
                        point_id_options: Some(point_id::PointIdOptions::Uuid(id)),
                    }),
                    vectors: Some(Vectors {
                        vectors_options: Some(vectors::VectorsOptions::Vectors(NamedVectors {
                            vectors: named,
                        })),
                    }),
                }
            })
            .collect();

        let builder = UpdatePointVectorsBuilder::new(collection_name, point_vectors).wait(true);

        self.retry_operation(|| async {
            self.client
                .update_vectors(builder.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to update sparse vectors: {}", e)))
        })
        .await?;

        Ok(())
    }

    /// Update payload fields on a single point identified by UUID.
    ///
    /// Convenience wrapper that avoids exposing `qdrant_client::Filter` to
    /// callers in the CLI layer.
    pub async fn set_payload_on_point(
        &self,
        collection_name: &str,
        point_id: &str,
        payload: std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<(), StorageError> {
        use qdrant_client::qdrant::{point_id, PointId, SetPayloadPointsBuilder};

        let id = PointId {
            point_id_options: Some(point_id::PointIdOptions::Uuid(point_id.to_string())),
        };

        let qdrant_payload: std::collections::HashMap<String, qdrant_client::qdrant::Value> =
            payload
                .into_iter()
                .map(|(k, v)| (k, convert_json_to_qdrant_value(v)))
                .collect();

        // Vec<PointId> implements Into<PointsIdsList> which implements Into<PointsSelectorOneOf>.
        let set_payload_request = SetPayloadPointsBuilder::new(collection_name, qdrant_payload)
            .points_selector(vec![id])
            .wait(true);

        self.retry_operation(|| async {
            self.client
                .set_payload(set_payload_request.clone())
                .await
                .map_err(|e| StorageError::Point(format!("Failed to set payload on point: {}", e)))
        })
        .await?;

        Ok(())
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
            return Err(StorageError::Collection(format!(
                "Collection not found: {}",
                collection_name
            )));
        }

        let qdrant_payload: std::collections::HashMap<String, qdrant_client::qdrant::Value> =
            payload
                .into_iter()
                .map(|(k, v)| (k, convert_json_to_qdrant_value(v)))
                .collect();

        let count = self
            .count_points_with_filter(collection_name, filter.clone())
            .await?;
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
}

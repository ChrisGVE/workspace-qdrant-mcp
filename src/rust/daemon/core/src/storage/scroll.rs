//! Scroll operations
//!
//! Paginated retrieval of points from Qdrant collections using the
//! scroll API with tenant filtering.

use qdrant_client::qdrant::{Condition, Filter, ScrollPointsBuilder, VectorsSelector};
use tracing::debug;

use super::types::SparsePointData;

use super::client::StorageClient;
use super::types::StorageError;

impl StorageClient {
    /// Scroll through all point IDs in a collection for a tenant
    pub async fn scroll_point_ids_by_tenant(
        &self,
        collection_name: &str,
        tenant_id: &str,
    ) -> Result<Vec<String>, StorageError> {
        debug!(
            "Scrolling point IDs for tenant_id='{}' in collection '{}'",
            tenant_id, collection_name
        );

        let mut all_ids = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        let batch_size = 100u32;

        let filter = Filter::must([Condition::matches("tenant_id", tenant_id.to_string())]);

        loop {
            let filter_clone = filter.clone();
            let current_offset = offset.clone();

            let response = self
                .retry_operation(|| {
                    let f = filter_clone.clone();
                    let o = current_offset.clone();
                    async move {
                        let mut builder = ScrollPointsBuilder::new(collection_name)
                            .filter(f)
                            .limit(batch_size)
                            .with_payload(false)
                            .with_vectors(false);

                        if let Some(offset_id) = o {
                            builder = builder.offset(offset_id);
                        }

                        self.client.scroll(builder).await.map_err(|e| {
                            StorageError::Search(format!(
                                "Scroll point IDs for tenant failed: {}",
                                e
                            ))
                        })
                    }
                })
                .await?;

            for point in &response.result {
                if let Some(ref id) = point.id {
                    let id_str = match &id.point_id_options {
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => {
                            uuid.clone()
                        }
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => {
                            num.to_string()
                        }
                        None => continue,
                    };
                    all_ids.push(id_str);
                }
            }

            match response.next_page_offset {
                Some(next_offset) => offset = Some(next_offset),
                None => break,
            }
        }

        debug!(
            "Scrolled {} point IDs for tenant_id='{}' in '{}'",
            all_ids.len(),
            tenant_id,
            collection_name
        );

        Ok(all_ids)
    }

    /// Scroll through all points for a tenant, returning file paths
    pub async fn scroll_file_paths_by_tenant(
        &self,
        collection_name: &str,
        tenant_id: &str,
    ) -> Result<Vec<String>, StorageError> {
        debug!(
            "Scrolling file paths for tenant_id='{}' in collection '{}'",
            tenant_id, collection_name
        );

        let mut file_paths = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        let batch_size = 100u32;

        let filter = Filter::must([Condition::matches("tenant_id", tenant_id.to_string())]);

        loop {
            let filter_clone = filter.clone();
            let current_offset = offset.clone();

            let response = self
                .retry_operation(|| {
                    let f = filter_clone.clone();
                    let o = current_offset.clone();
                    async move {
                        let mut builder = ScrollPointsBuilder::new(collection_name)
                            .filter(f)
                            .limit(batch_size)
                            .with_payload(true)
                            .with_vectors(false);

                        if let Some(offset_id) = o {
                            builder = builder.offset(offset_id);
                        }

                        self.client
                            .scroll(builder)
                            .await
                            .map_err(|e| StorageError::Search(format!("Scroll failed: {}", e)))
                    }
                })
                .await?;

            for point in &response.result {
                if let Some(value) = point.payload.get("file_path") {
                    if let Some(qdrant_client::qdrant::value::Kind::StringValue(path)) = &value.kind
                    {
                        file_paths.push(path.clone());
                    }
                }
            }

            match response.next_page_offset {
                Some(next_offset) => offset = Some(next_offset),
                None => break,
            }
        }

        debug!(
            "Scrolled {} file paths for tenant_id='{}' in '{}'",
            file_paths.len(),
            tenant_id,
            collection_name
        );

        Ok(file_paths)
    }

    /// Scroll through one page of points, returning `SparsePointData` items.
    ///
    /// Used by `wqm admin rebalance-idf` to read sparse vectors for IDF
    /// correction without fetching dense vectors. Returns a batch of decoded
    /// points and an opaque next-page cursor (None when the collection is
    /// exhausted). Pass the cursor as `offset_cursor` on the next call.
    pub async fn scroll_with_sparse_vectors(
        &self,
        collection_name: &str,
        batch_size: u32,
        offset_cursor: Option<String>,
    ) -> Result<(Vec<SparsePointData>, Option<String>), StorageError> {
        use qdrant_client::qdrant::{point_id, value, vector_output, vectors_output, PointId};

        // Decode opaque string cursor back to PointId (UUID only).
        let offset: Option<PointId> = offset_cursor.map(|s| PointId {
            point_id_options: Some(point_id::PointIdOptions::Uuid(s)),
        });

        let sparse_selector = VectorsSelector {
            names: vec!["sparse".to_string()],
        };

        let response = self
            .retry_operation(|| {
                let o = offset.clone();
                let selector = sparse_selector.clone();
                async move {
                    let mut builder = ScrollPointsBuilder::new(collection_name)
                        .limit(batch_size)
                        .with_payload(true)
                        .with_vectors(selector);

                    if let Some(offset_id) = o {
                        builder = builder.offset(offset_id);
                    }

                    self.client.scroll(builder).await.map_err(|e| {
                        StorageError::Search(format!("Scroll with sparse vectors failed: {}", e))
                    })
                }
            })
            .await?;

        // Convert raw RetrievedPoints into SparsePointData.
        let points = response
            .result
            .into_iter()
            .filter_map(|p| {
                // Extract UUID string.
                let id = match p.id.as_ref()?.point_id_options.as_ref()? {
                    point_id::PointIdOptions::Uuid(uuid) => uuid.clone(),
                    point_id::PointIdOptions::Num(n) => n.to_string(),
                };

                // Extract optional idf_epoch from payload.
                let idf_epoch = p.payload.get("idf_epoch").and_then(|v| match &v.kind {
                    Some(value::Kind::IntegerValue(n)) => Some(*n as u64),
                    Some(value::Kind::DoubleValue(f)) => Some(*f as u64),
                    _ => None,
                });

                // Extract sparse vector (may be absent for SPLADE or old points).
                let sparse_vector =
                    p.vectors
                        .as_ref()
                        .and_then(|vout| match &vout.vectors_options {
                            Some(vectors_output::VectorsOptions::Vectors(named)) => {
                                let sv_out = named.vectors.get("sparse")?;
                                match &sv_out.vector {
                                    Some(vector_output::Vector::Sparse(sv)) => Some(
                                        sv.indices
                                            .iter()
                                            .zip(sv.values.iter())
                                            .map(|(&i, &v)| (i, v))
                                            .collect(),
                                    ),
                                    _ => None,
                                }
                            }
                            _ => None,
                        });

                Some(SparsePointData {
                    id,
                    idf_epoch,
                    sparse_vector,
                })
            })
            .collect();

        // Encode next-page offset as UUID string cursor.
        let next_cursor = response
            .next_page_offset
            .and_then(|pid| match pid.point_id_options {
                Some(point_id::PointIdOptions::Uuid(uuid)) => Some(uuid),
                Some(point_id::PointIdOptions::Num(n)) => Some(n.to_string()),
                None => None,
            });

        Ok((points, next_cursor))
    }

    /// Scroll through points matching a filter, returning full point data.
    ///
    /// Single-batch scroll limited by `limit`. Returns points with payloads
    /// but without vectors.
    pub async fn scroll_with_filter(
        &self,
        collection_name: &str,
        filter: Filter,
        limit: u32,
        offset: Option<qdrant_client::qdrant::PointId>,
    ) -> Result<Vec<qdrant_client::qdrant::RetrievedPoint>, StorageError> {
        let response = self
            .retry_operation(|| {
                let f = filter.clone();
                let o = offset.clone();
                async move {
                    let mut builder = ScrollPointsBuilder::new(collection_name)
                        .filter(f)
                        .limit(limit)
                        .with_payload(true)
                        .with_vectors(false);

                    if let Some(offset_id) = o {
                        builder = builder.offset(offset_id);
                    }

                    self.client.scroll(builder).await.map_err(|e| {
                        StorageError::Search(format!("Scroll with filter failed: {}", e))
                    })
                }
            })
            .await?;

        Ok(response.result)
    }
}

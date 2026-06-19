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

    /// Scroll dense vectors for a tenant, returning up to `limit` vectors.
    ///
    /// Fetches the named `"dense"` vector from the `projects` collection
    /// (or any collection) filtered by `tenant_id`. Paginates internally
    /// using the Qdrant scroll cursor. Returns the extracted `Vec<f32>`
    /// dense vectors.
    pub async fn scroll_dense_vectors_by_tenant(
        &self,
        collection_name: &str,
        tenant_id: &str,
        limit: usize,
    ) -> Result<Vec<Vec<f32>>, StorageError> {
        debug!(
            tenant_id,
            collection_name, limit, "Scrolling dense vectors for bootstrap"
        );

        let mut vectors = Vec::new();
        let mut offset: Option<qdrant_client::qdrant::PointId> = None;
        let batch_size = limit.min(500) as u32;

        let filter = Filter::must([Condition::matches("tenant_id", tenant_id.to_string())]);
        let dense_selector = VectorsSelector {
            names: vec!["dense".to_string()],
        };

        loop {
            if vectors.len() >= limit {
                break;
            }

            let filter_clone = filter.clone();
            let current_offset = offset.clone();
            let selector = dense_selector.clone();

            let response = self
                .retry_operation(|| {
                    let f = filter_clone.clone();
                    let o = current_offset.clone();
                    let s = selector.clone();
                    async move {
                        let mut builder = ScrollPointsBuilder::new(collection_name)
                            .filter(f)
                            .limit(batch_size)
                            .with_payload(false)
                            .with_vectors(s);

                        if let Some(offset_id) = o {
                            builder = builder.offset(offset_id);
                        }

                        self.client.scroll(builder).await.map_err(|e| {
                            StorageError::Search(format!(
                                "Scroll dense vectors for tenant failed: {}",
                                e
                            ))
                        })
                    }
                })
                .await?;

            for point in &response.result {
                if vectors.len() >= limit {
                    break;
                }
                if let Some(dense) = extract_dense_vector(point) {
                    vectors.push(dense);
                }
            }

            match response.next_page_offset {
                Some(next_offset) => offset = Some(next_offset),
                None => break,
            }
        }

        debug!(
            tenant_id,
            count = vectors.len(),
            "Scrolled dense vectors for bootstrap"
        );

        Ok(vectors)
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
        use qdrant_client::qdrant::{point_id, PointId};

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

        let points = response
            .result
            .into_iter()
            .filter_map(convert_retrieved_to_sparse_point)
            .collect();

        let next_cursor = extract_next_cursor(response.next_page_offset);

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

/// Extract the dense vector from a retrieved point's named vectors.
fn extract_dense_vector(point: &qdrant_client::qdrant::RetrievedPoint) -> Option<Vec<f32>> {
    use qdrant_client::qdrant::{vector_output, vectors_output};

    point
        .vectors
        .as_ref()
        .and_then(|vout| match &vout.vectors_options {
            Some(vectors_output::VectorsOptions::Vectors(named)) => {
                let dense_out = named.vectors.get("dense")?;
                match &dense_out.vector {
                    Some(vector_output::Vector::Dense(dv)) => Some(dv.data.clone()),
                    _ => None,
                }
            }
            _ => None,
        })
}

/// Convert a single `RetrievedPoint` into a `SparsePointData`.
fn convert_retrieved_to_sparse_point(
    p: qdrant_client::qdrant::RetrievedPoint,
) -> Option<SparsePointData> {
    use qdrant_client::qdrant::{point_id, value, vector_output, vectors_output};

    let id = match p.id.as_ref()?.point_id_options.as_ref()? {
        point_id::PointIdOptions::Uuid(uuid) => uuid.clone(),
        point_id::PointIdOptions::Num(n) => n.to_string(),
    };

    let idf_epoch = p.payload.get("idf_epoch").and_then(|v| match &v.kind {
        Some(value::Kind::IntegerValue(n)) => Some(*n as u64),
        Some(value::Kind::DoubleValue(f)) => Some(*f as u64),
        _ => None,
    });

    let sparse_vector = p
        .vectors
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
}

/// Encode next-page offset as UUID string cursor.
fn extract_next_cursor(next_page_offset: Option<qdrant_client::qdrant::PointId>) -> Option<String> {
    use qdrant_client::qdrant::point_id;
    next_page_offset.and_then(|pid| match pid.point_id_options {
        Some(point_id::PointIdOptions::Uuid(uuid)) => Some(uuid),
        Some(point_id::PointIdOptions::Num(n)) => Some(n.to_string()),
        None => None,
    })
}

impl StorageClient {
    /// Retrieve the original dense vector of a single point by id.
    ///
    /// Branch-lineage F5(b): the Case-2 copy-vector path (arch §5.1) needs to
    /// read an existing byte-identical real point's vector and copy it into a
    /// new file-identity's own real point instead of re-embedding. The F6 tagger
    /// lives in `daemon/core`, which does not depend on `wqm-client`, so it
    /// cannot use `QdrantReadClient::retrieve` (which returns no vectors anyway);
    /// this is the in-crate `StorageClient` method that closes that gap.
    ///
    /// Modeled on [`StorageClient::scroll_dense_vectors_by_tenant`]: issues a
    /// `get_points(...).with_vectors(true)` for the single id and extracts the
    /// named `"dense"` vector via the shared `extract_dense_vector` helper.
    /// Returns `Ok(None)` when the point is absent (or carries no dense vector).
    ///
    /// ## Quantization — the ORIGINAL (un-quantized) vector (F5 AC, S3)
    ///
    /// The copy MUST use the original stored vector, not a quantized
    /// approximation. Qdrant has no per-call "give me the raw vector" flag —
    /// whether a read returns the original is governed by the COLLECTION CONFIG
    /// (quantization keeps the original full-precision vector on disk and uses
    /// the quantized form only as a search-time approximation; a point READ
    /// returns the original). Verified for this project: neither collection
    /// creation path sets `quantization_config`
    /// (`storage/collections/basic.rs` and `storage/collections/multi_tenant.rs`
    /// both leave it `None`), so all collections here store and return the
    /// original vector — this method returns the un-quantized vector as required.
    /// Any future collection that enables quantization must be verified to still
    /// return the original on a `get_points` read before relying on this copy.
    pub async fn retrieve_point_with_vector(
        &self,
        collection: &str,
        point_id: &str,
    ) -> Result<Option<Vec<f32>>, StorageError> {
        use qdrant_client::qdrant::{GetPointsBuilder, PointId};

        let ids = vec![PointId::from(point_id)];
        let dense_selector = VectorsSelector {
            names: vec!["dense".to_string()],
        };

        let response = self
            .retry_operation(|| {
                let ids = ids.clone();
                let selector = dense_selector.clone();
                async move {
                    let builder = GetPointsBuilder::new(collection, ids)
                        .with_payload(false)
                        .with_vectors(selector);
                    self.client.get_points(builder).await.map_err(|e| {
                        StorageError::Point(format!("retrieve_point_with_vector failed: {}", e))
                    })
                }
            })
            .await?;

        Ok(response
            .result
            .first()
            .and_then(extract_dense_vector))
    }
}

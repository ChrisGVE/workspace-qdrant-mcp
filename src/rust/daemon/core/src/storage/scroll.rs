//! Scroll operations
//!
//! Paginated retrieval of points from Qdrant collections using the
//! scroll API with tenant filtering.

use qdrant_client::qdrant::{Condition, Filter, ScrollPointsBuilder};
use tracing::debug;

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

                        self.client
                            .scroll(builder)
                            .await
                            .map_err(|e| StorageError::Search(
                                format!("Scroll point IDs for tenant failed: {}", e)
                            ))
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
            all_ids.len(), tenant_id, collection_name
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
                    if let Some(qdrant_client::qdrant::value::Kind::StringValue(path)) = &value.kind {
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
            file_paths.len(), tenant_id, collection_name
        );

        Ok(file_paths)
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

                    self.client
                        .scroll(builder)
                        .await
                        .map_err(|e| StorageError::Search(format!("Scroll with filter failed: {}", e)))
                }
            })
            .await?;

        Ok(response.result)
    }
}

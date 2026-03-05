//! Basic collection operations
//!
//! Collection creation, deletion, existence checks, listing,
//! info retrieval, and alias management.

use qdrant_client::qdrant::{
    CreateAliasBuilder, CreateCollection, Datatype, DeleteCollection, Distance, RenameAliasBuilder,
    VectorParams, VectorsConfig,
};
use tracing::{info, warn};

use super::super::client::StorageClient;
use super::super::types::{CollectionInfoResult, StorageError};

impl StorageClient {
    /// Create a new collection with vector configuration
    pub async fn create_collection(
        &self,
        collection_name: &str,
        dense_vector_size: Option<u64>,
        _sparse_vector_size: Option<u64>,
    ) -> Result<(), StorageError> {
        info!("Creating collection: {}", collection_name);

        let dense_size = dense_vector_size.unwrap_or(self.config.dense_vector_size);

        let mut vectors_config = VectorsConfig::default();

        // Configure dense vector
        let dense_vector_params = VectorParams {
            size: dense_size,
            distance: Distance::Cosine.into(),
            hnsw_config: None,
            quantization_config: None,
            on_disk: Some(false), // Keep in memory for better performance
            datatype: Some(Datatype::Float32.into()),
            multivector_config: None,
        };

        vectors_config.config = Some(qdrant_client::qdrant::vectors_config::Config::Params(
            dense_vector_params,
        ));

        let create_collection = CreateCollection {
            collection_name: collection_name.to_string(),
            vectors_config: Some(vectors_config),
            shard_number: Some(1),
            replication_factor: Some(1),
            write_consistency_factor: Some(1),
            on_disk_payload: Some(true),
            timeout: Some(self.config.timeout_ms),
            ..Default::default()
        };

        self.retry_operation(|| async {
            self.client
                .create_collection(create_collection.clone())
                .await
                .map_err(|e| StorageError::Collection(e.to_string()))
        })
        .await?;

        info!("Successfully created collection: {}", collection_name);
        Ok(())
    }

    /// Delete a collection
    pub async fn delete_collection(&self, collection_name: &str) -> Result<(), StorageError> {
        info!("Deleting collection: {}", collection_name);

        let delete_collection = DeleteCollection {
            collection_name: collection_name.to_string(),
            timeout: Some(self.config.timeout_ms),
        };

        self.retry_operation(|| async {
            self.client
                .delete_collection(delete_collection.clone())
                .await
                .map_err(|e| StorageError::Collection(e.to_string()))
        })
        .await?;

        info!("Successfully deleted collection: {}", collection_name);
        Ok(())
    }

    /// Check if a collection exists
    pub async fn collection_exists(&self, collection_name: &str) -> Result<bool, StorageError> {
        tracing::debug!("Checking if collection exists: {}", collection_name);

        let response = self
            .retry_operation(|| async {
                self.client
                    .collection_exists(collection_name)
                    .await
                    .map_err(|e| StorageError::Collection(e.to_string()))
            })
            .await?;

        Ok(response)
    }

    /// List all collections in Qdrant
    ///
    /// Returns a list of collection names.
    pub async fn list_collections(&self) -> Result<Vec<String>, StorageError> {
        tracing::debug!("Listing all collections");

        let response = self
            .retry_operation(|| async {
                self.client
                    .list_collections()
                    .await
                    .map_err(|e| StorageError::Collection(e.to_string()))
            })
            .await?;

        let names = response.collections.into_iter().map(|c| c.name).collect();

        Ok(names)
    }

    /// Get detailed information about a collection
    ///
    /// Returns point count, vector config, status, and aliases.
    pub async fn get_collection_info(
        &self,
        collection_name: &str,
    ) -> Result<CollectionInfoResult, StorageError> {
        tracing::debug!("Getting collection info: {}", collection_name);

        let info = self
            .retry_operation(|| async {
                self.client
                    .collection_info(collection_name)
                    .await
                    .map_err(|e| StorageError::Collection(e.to_string()))
            })
            .await?;

        let points_count = info
            .result
            .as_ref()
            .map(|r| r.points_count.unwrap_or(0))
            .unwrap_or(0);

        let vectors_count = info
            .result
            .as_ref()
            .map(|r| r.indexed_vectors_count.unwrap_or(0))
            .unwrap_or(0);

        let status = info
            .result
            .as_ref()
            .map(|r| match r.status() {
                qdrant_client::qdrant::CollectionStatus::Green => "green",
                qdrant_client::qdrant::CollectionStatus::Yellow => "yellow",
                qdrant_client::qdrant::CollectionStatus::Red => "red",
                qdrant_client::qdrant::CollectionStatus::Grey => "grey",
                _ => "unknown",
            })
            .unwrap_or("unknown")
            .to_string();

        let vector_dimension = extract_vector_dimension(&info);

        // Get aliases for this collection
        let aliases = match self.client.list_collection_aliases(collection_name).await {
            Ok(response) => response.aliases.into_iter().map(|a| a.alias_name).collect(),
            Err(e) => {
                warn!("Failed to get aliases for {}: {}", collection_name, e);
                vec![]
            }
        };

        Ok(CollectionInfoResult {
            name: collection_name.to_string(),
            points_count,
            vectors_count,
            status,
            vector_dimension,
            aliases,
        })
    }

    // =========================================================================
    // Collection Alias Operations
    // =========================================================================

    /// Create a collection alias
    ///
    /// Maps `alias_name` to `collection_name` in Qdrant. Queries to the alias
    /// will be directed to the target collection.
    pub async fn create_alias(
        &self,
        collection_name: &str,
        alias_name: &str,
    ) -> Result<(), StorageError> {
        self.client
            .create_alias(CreateAliasBuilder::new(collection_name, alias_name))
            .await
            .map_err(|e| {
                StorageError::Collection(format!("Failed to create alias '{}': {}", alias_name, e))
            })?;
        Ok(())
    }

    /// Delete a collection alias
    pub async fn delete_alias(&self, alias_name: &str) -> Result<(), StorageError> {
        self.client.delete_alias(alias_name).await.map_err(|e| {
            StorageError::Collection(format!("Failed to delete alias '{}': {}", alias_name, e))
        })?;
        Ok(())
    }

    /// Rename a collection alias atomically
    ///
    /// Changes the alias name from `old_alias_name` to `new_alias_name`.
    /// The alias continues pointing to the same collection.
    pub async fn rename_alias(
        &self,
        old_alias_name: &str,
        new_alias_name: &str,
    ) -> Result<(), StorageError> {
        self.client
            .rename_alias(RenameAliasBuilder::new(old_alias_name, new_alias_name))
            .await
            .map_err(|e| {
                StorageError::Collection(format!(
                    "Failed to rename alias '{}' to '{}': {}",
                    old_alias_name, new_alias_name, e
                ))
            })?;
        Ok(())
    }
}

/// Extract vector dimension from collection info response
pub(super) fn extract_vector_dimension(
    info: &qdrant_client::qdrant::GetCollectionInfoResponse,
) -> Option<u64> {
    info.result
        .as_ref()
        .and_then(|r| r.config.as_ref())
        .and_then(|c| c.params.as_ref())
        .and_then(|p| p.vectors_config.as_ref())
        .and_then(|vc| {
            use qdrant_client::qdrant::vectors_config::Config;
            match &vc.config {
                Some(Config::Params(params)) => Some(params.size),
                Some(Config::ParamsMap(map)) => map.map.get("dense").map(|p| p.size),
                _ => None,
            }
        })
}

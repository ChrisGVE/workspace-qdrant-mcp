//! Collection operations
//!
//! Collection creation, deletion, alias management, payload indexing,
//! and multi-tenant initialization.

use std::collections::HashMap;

use qdrant_client::qdrant::{
    CreateCollection, DeleteCollection, Distance, VectorParams, VectorsConfig,
    Datatype, CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, FieldType,
    HnswConfigDiffBuilder, VectorParamsBuilder, VectorParamsMap,
    SparseVectorConfig, SparseVectorParams, vectors_config,
    CreateAliasBuilder, RenameAliasBuilder,
};
use tracing::{info, warn, error};

use wqm_common::constants::{
    COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_RULES,
    COLLECTION_SCRATCHPAD, COLLECTION_IMAGES,
};

use super::client::StorageClient;
use super::config::MultiTenantConfig;
use super::types::{CollectionInfoResult, MultiTenantInitResult, StorageError};

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

        vectors_config.config = Some(qdrant_client::qdrant::vectors_config::Config::Params(dense_vector_params));

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
            self.client.create_collection(create_collection.clone()).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;

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
            self.client.delete_collection(delete_collection.clone()).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;

        info!("Successfully deleted collection: {}", collection_name);
        Ok(())
    }

    /// Check if a collection exists
    pub async fn collection_exists(&self, collection_name: &str) -> Result<bool, StorageError> {
        tracing::debug!("Checking if collection exists: {}", collection_name);

        let response = self.retry_operation(|| async {
            self.client.collection_exists(collection_name).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;

        Ok(response)
    }

    /// List all collections in Qdrant
    ///
    /// Returns a list of collection names.
    pub async fn list_collections(&self) -> Result<Vec<String>, StorageError> {
        tracing::debug!("Listing all collections");

        let response = self.retry_operation(|| async {
            self.client.list_collections().await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;

        let names = response.collections
            .into_iter()
            .map(|c| c.name)
            .collect();

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

        let info = self.retry_operation(|| async {
            self.client.collection_info(collection_name).await
                .map_err(|e| StorageError::Collection(e.to_string()))
        }).await?;

        let points_count = info.result
            .as_ref()
            .map(|r| r.points_count.unwrap_or(0))
            .unwrap_or(0);

        let vectors_count = info.result
            .as_ref()
            .map(|r| r.indexed_vectors_count.unwrap_or(0))
            .unwrap_or(0);

        let status = info.result
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
            Ok(response) => {
                response.aliases
                    .into_iter()
                    .map(|a| a.alias_name)
                    .collect()
            }
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
            .map_err(|e| StorageError::Collection(format!("Failed to create alias '{}': {}", alias_name, e)))?;
        Ok(())
    }

    /// Delete a collection alias
    pub async fn delete_alias(&self, alias_name: &str) -> Result<(), StorageError> {
        self.client
            .delete_alias(alias_name)
            .await
            .map_err(|e| StorageError::Collection(format!("Failed to delete alias '{}': {}", alias_name, e)))?;
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
            .map_err(|e| StorageError::Collection(format!(
                "Failed to rename alias '{}' to '{}': {}", old_alias_name, new_alias_name, e
            )))?;
        Ok(())
    }

    // =========================================================================
    // Multi-tenant collection creation
    // =========================================================================

    /// Create a multi-tenant collection with optimized HNSW configuration
    ///
    /// Collections are created with named vectors:
    /// - "dense": Dense semantic vectors (384 dimensions for all-MiniLM-L6-v2)
    /// - "sparse": Sparse BM25-style keyword vectors for hybrid search
    pub async fn create_multi_tenant_collection(
        &self,
        collection_name: &str,
        config: &MultiTenantConfig,
    ) -> Result<(), StorageError> {
        info!(
            "Creating multi-tenant collection: {} (vector_size={}, m={}, ef_construct={})",
            collection_name, config.vector_size, config.hnsw_m, config.hnsw_ef_construct
        );

        // Check if collection already exists (idempotency)
        if self.collection_exists(collection_name).await? {
            info!("Collection {} already exists, skipping creation", collection_name);
            return Ok(());
        }

        let hnsw_config = HnswConfigDiffBuilder::default()
            .m(config.hnsw_m)
            .ef_construct(config.hnsw_ef_construct);

        let dense_vector_params: VectorParams = VectorParamsBuilder::new(config.vector_size, Distance::Cosine)
            .hnsw_config(hnsw_config)
            .on_disk(false)
            .build();

        let mut dense_vectors_map = HashMap::new();
        dense_vectors_map.insert("dense".to_string(), dense_vector_params);

        let named_vectors_config = VectorsConfig {
            config: Some(vectors_config::Config::ParamsMap(VectorParamsMap {
                map: dense_vectors_map,
            })),
        };

        let mut sparse_vectors_map = HashMap::new();
        sparse_vectors_map.insert("sparse".to_string(), SparseVectorParams {
            index: None,
            modifier: None,
        });

        let sparse_config = SparseVectorConfig {
            map: sparse_vectors_map,
        };

        let create_request = CreateCollectionBuilder::new(collection_name)
            .vectors_config(named_vectors_config)
            .sparse_vectors_config(sparse_config)
            .on_disk_payload(config.on_disk_payload)
            .shard_number(1)
            .replication_factor(1)
            .write_consistency_factor(1);

        self.retry_operation(|| async {
            self.client
                .create_collection(create_request.clone())
                .await
                .map_err(|e| StorageError::Collection(e.to_string()))
        })
        .await?;

        info!("Successfully created multi-tenant collection with dense+sparse vectors: {}", collection_name);
        Ok(())
    }

    /// Create the images collection with CLIP 512-dim dense vectors only.
    pub async fn create_image_collection(
        &self,
        config: &MultiTenantConfig,
    ) -> Result<(), StorageError> {
        info!(
            "Creating images collection (CLIP 512-dim, dense-only, m={}, ef_construct={})",
            config.hnsw_m, config.hnsw_ef_construct
        );

        if self.collection_exists(COLLECTION_IMAGES).await? {
            info!("Collection {} already exists, skipping creation", COLLECTION_IMAGES);
            return Ok(());
        }

        let hnsw_config = HnswConfigDiffBuilder::default()
            .m(config.hnsw_m)
            .ef_construct(config.hnsw_ef_construct);

        let dense_vector_params: VectorParams =
            VectorParamsBuilder::new(512, Distance::Cosine)
                .hnsw_config(hnsw_config)
                .on_disk(false)
                .build();

        let mut dense_vectors_map = HashMap::new();
        dense_vectors_map.insert("dense".to_string(), dense_vector_params);

        let named_vectors_config = VectorsConfig {
            config: Some(vectors_config::Config::ParamsMap(VectorParamsMap {
                map: dense_vectors_map,
            })),
        };

        let create_request = CreateCollectionBuilder::new(COLLECTION_IMAGES)
            .vectors_config(named_vectors_config)
            .on_disk_payload(config.on_disk_payload)
            .shard_number(1)
            .replication_factor(1)
            .write_consistency_factor(1);

        self.retry_operation(|| async {
            self.client
                .create_collection(create_request.clone())
                .await
                .map_err(|e| StorageError::Collection(e.to_string()))
        })
        .await?;

        info!("Successfully created images collection (512-dim CLIP, dense-only)");
        Ok(())
    }

    /// Create a payload index for efficient filtering
    pub async fn create_payload_index(
        &self,
        collection_name: &str,
        field_name: &str,
    ) -> Result<(), StorageError> {
        info!(
            "Creating payload index on {}.{}",
            collection_name, field_name
        );

        let index_request = CreateFieldIndexCollectionBuilder::new(
            collection_name,
            field_name,
            FieldType::Keyword,
        );

        self.retry_operation(|| async {
            self.client
                .create_field_index(index_request.clone())
                .await
                .map_err(|e| StorageError::Collection(format!(
                    "Failed to create payload index on {}.{}: {}",
                    collection_name, field_name, e
                )))
        })
        .await?;

        info!(
            "Successfully created payload index on {}.{}",
            collection_name, field_name
        );
        Ok(())
    }

    /// Initialize all multi-tenant collections with proper configuration
    ///
    /// Creates the unified collections: projects, libraries, rules, scratchpad,
    /// and images. This method is idempotent - existing collections are skipped.
    pub async fn initialize_multi_tenant_collections(
        &self,
        config: Option<MultiTenantConfig>,
    ) -> Result<MultiTenantInitResult, StorageError> {
        let config = config.unwrap_or_default();
        info!("Initializing multi-tenant collections with config: {:?}", config);

        let mut result = MultiTenantInitResult::default();

        // Create _projects collection
        match self.create_multi_tenant_collection(COLLECTION_PROJECTS, &config).await {
            Ok(()) => {
                if !self.collection_exists(COLLECTION_PROJECTS).await.unwrap_or(false) {
                    result.projects_created = true;
                }
                match self.create_payload_index(COLLECTION_PROJECTS, "project_id").await {
                    Ok(()) => result.projects_indexed = true,
                    Err(e) => {
                        warn!("Could not create project_id index (may already exist): {}", e);
                        result.projects_indexed = true;
                    }
                }
            }
            Err(e) => {
                error!("Failed to create {} collection: {}", COLLECTION_PROJECTS, e);
                return Err(e);
            }
        }
        result.projects_created = true;

        // Create _libraries collection
        match self.create_multi_tenant_collection(COLLECTION_LIBRARIES, &config).await {
            Ok(()) => {
                match self.create_payload_index(COLLECTION_LIBRARIES, "library_name").await {
                    Ok(()) => result.libraries_indexed = true,
                    Err(e) => {
                        warn!("Could not create library_name index (may already exist): {}", e);
                        result.libraries_indexed = true;
                    }
                }
            }
            Err(e) => {
                error!("Failed to create {} collection: {}", COLLECTION_LIBRARIES, e);
                return Err(e);
            }
        }
        result.libraries_created = true;

        // Create rules collection
        match self.create_multi_tenant_collection(COLLECTION_RULES, &config).await {
            Ok(()) => {}
            Err(e) => {
                error!("Failed to create {} collection: {}", COLLECTION_RULES, e);
                return Err(e);
            }
        }
        result.rules_created = true;

        // Create scratchpad collection with tenant_id index
        match self.create_multi_tenant_collection(COLLECTION_SCRATCHPAD, &config).await {
            Ok(()) => {
                match self.create_payload_index(COLLECTION_SCRATCHPAD, "tenant_id").await {
                    Ok(()) => {}
                    Err(e) => {
                        warn!("Could not create tenant_id index on scratchpad (may already exist): {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Failed to create {} collection: {}", COLLECTION_SCRATCHPAD, e);
                return Err(e);
            }
        }
        result.scratchpad_created = true;

        // Create images collection (512-dim CLIP, dense-only)
        match self.create_image_collection(&config).await {
            Ok(()) => {
                match self.create_payload_index(COLLECTION_IMAGES, "tenant_id").await {
                    Ok(()) => {}
                    Err(e) => {
                        warn!("Could not create tenant_id index on images (may already exist): {}", e);
                    }
                }
                match self.create_payload_index(COLLECTION_IMAGES, "source_document_id").await {
                    Ok(()) => {}
                    Err(e) => {
                        warn!("Could not create source_document_id index on images (may already exist): {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Failed to create {} collection: {}", COLLECTION_IMAGES, e);
                return Err(e);
            }
        }
        result.images_created = true;

        info!("Multi-tenant collections initialized: {:?}", result);
        Ok(result)
    }
}

/// Extract vector dimension from collection info response
fn extract_vector_dimension(
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
                Some(Config::ParamsMap(map)) => {
                    map.map.get("dense").map(|p| p.size)
                }
                _ => None,
            }
        })
}

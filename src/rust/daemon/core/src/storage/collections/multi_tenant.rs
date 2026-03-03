//! Multi-tenant collection operations
//!
//! Creation of multi-tenant collections with optimized HNSW configuration,
//! payload indexing, and idempotent initialization of all canonical collections.

use std::collections::HashMap;

use qdrant_client::qdrant::{
    Distance, VectorParams, VectorsConfig,
    CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, FieldType,
    HnswConfigDiffBuilder, VectorParamsBuilder, VectorParamsMap,
    SparseVectorConfig, SparseVectorParams, vectors_config,
};
use tracing::{info, warn, error};

use wqm_common::constants::{
    COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_RULES,
    COLLECTION_SCRATCHPAD, COLLECTION_IMAGES,
};

use super::super::client::StorageClient;
use super::super::config::MultiTenantConfig;
use super::super::types::{MultiTenantInitResult, StorageError};

impl StorageClient {
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

        let dense_vector_params: VectorParams =
            VectorParamsBuilder::new(config.vector_size, Distance::Cosine)
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

        info!(
            "Successfully created multi-tenant collection with dense+sparse vectors: {}",
            collection_name
        );
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

        self.init_projects_collection(&config, &mut result).await?;
        self.init_libraries_collection(&config, &mut result).await?;
        self.init_rules_collection(&config, &mut result).await?;
        self.init_scratchpad_collection(&config, &mut result).await?;
        self.init_images_collection(&config, &mut result).await?;

        info!("Multi-tenant collections initialized: {:?}", result);
        Ok(result)
    }

    async fn init_projects_collection(
        &self,
        config: &MultiTenantConfig,
        result: &mut MultiTenantInitResult,
    ) -> Result<(), StorageError> {
        self.create_multi_tenant_collection(COLLECTION_PROJECTS, config)
            .await
            .map_err(|e| {
                error!("Failed to create {} collection: {}", COLLECTION_PROJECTS, e);
                e
            })?;
        result.projects_created = true;
        match self.create_payload_index(COLLECTION_PROJECTS, "project_id").await {
            Ok(()) => result.projects_indexed = true,
            Err(e) => {
                warn!("Could not create project_id index (may already exist): {}", e);
                result.projects_indexed = true;
            }
        }
        Ok(())
    }

    async fn init_libraries_collection(
        &self,
        config: &MultiTenantConfig,
        result: &mut MultiTenantInitResult,
    ) -> Result<(), StorageError> {
        self.create_multi_tenant_collection(COLLECTION_LIBRARIES, config)
            .await
            .map_err(|e| {
                error!("Failed to create {} collection: {}", COLLECTION_LIBRARIES, e);
                e
            })?;
        result.libraries_created = true;
        match self.create_payload_index(COLLECTION_LIBRARIES, "library_name").await {
            Ok(()) => result.libraries_indexed = true,
            Err(e) => {
                warn!("Could not create library_name index (may already exist): {}", e);
                result.libraries_indexed = true;
            }
        }
        Ok(())
    }

    async fn init_rules_collection(
        &self,
        config: &MultiTenantConfig,
        result: &mut MultiTenantInitResult,
    ) -> Result<(), StorageError> {
        self.create_multi_tenant_collection(COLLECTION_RULES, config)
            .await
            .map_err(|e| {
                error!("Failed to create {} collection: {}", COLLECTION_RULES, e);
                e
            })?;
        result.rules_created = true;
        Ok(())
    }

    async fn init_scratchpad_collection(
        &self,
        config: &MultiTenantConfig,
        result: &mut MultiTenantInitResult,
    ) -> Result<(), StorageError> {
        self.create_multi_tenant_collection(COLLECTION_SCRATCHPAD, config)
            .await
            .map_err(|e| {
                error!("Failed to create {} collection: {}", COLLECTION_SCRATCHPAD, e);
                e
            })?;
        result.scratchpad_created = true;
        if let Err(e) = self.create_payload_index(COLLECTION_SCRATCHPAD, "tenant_id").await {
            warn!(
                "Could not create tenant_id index on scratchpad (may already exist): {}",
                e
            );
        }
        Ok(())
    }

    async fn init_images_collection(
        &self,
        config: &MultiTenantConfig,
        result: &mut MultiTenantInitResult,
    ) -> Result<(), StorageError> {
        self.create_image_collection(config)
            .await
            .map_err(|e| {
                error!("Failed to create {} collection: {}", COLLECTION_IMAGES, e);
                e
            })?;
        result.images_created = true;
        if let Err(e) = self.create_payload_index(COLLECTION_IMAGES, "tenant_id").await {
            warn!("Could not create tenant_id index on images (may already exist): {}", e);
        }
        if let Err(e) = self
            .create_payload_index(COLLECTION_IMAGES, "source_document_id")
            .await
        {
            warn!(
                "Could not create source_document_id index on images (may already exist): {}",
                e
            );
        }
        Ok(())
    }
}

//! gRPC handler implementations for CollectionService
//!
//! Implements the 7 RPCs: CreateCollection, DeleteCollection, ListCollections,
//! GetCollection, CreateCollectionAlias, DeleteCollectionAlias, RenameCollectionAlias.

use std::time::SystemTime;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

use crate::proto::{
    collection_service_server::CollectionService, CollectionConfig, CollectionInfo,
    CreateAliasRequest, CreateCollectionRequest, CreateCollectionResponse, DeleteAliasRequest,
    DeleteCollectionRequest, GetCollectionRequest, GetCollectionResponse, ListCollectionsResponse,
    RenameAliasRequest,
};

use super::{
    validation::{
        map_distance_metric, map_storage_error, validate_alias_name, validate_collection_name,
        validate_vector_size,
    },
    CollectionServiceImpl,
};

#[tonic::async_trait]
impl CollectionService for CollectionServiceImpl {
    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();

        info!("Creating collection: {}", req.collection_name);
        debug!("Project ID: {}", req.project_id);

        validate_collection_name(&req.collection_name)?;

        let config = req
            .config
            .ok_or_else(|| Status::invalid_argument("Collection configuration is required"))?;

        validate_vector_size(config.vector_size)?;
        let _distance_metric = map_distance_metric(&config.distance_metric)?;

        match self
            .storage_client
            .create_collection(
                &req.collection_name,
                Some(config.vector_size as u64),
                None, // sparse vector size - not currently used
            )
            .await
        {
            Ok(_) => {
                info!("Successfully created collection: {}", req.collection_name);
                Ok(Response::new(CreateCollectionResponse {
                    success: true,
                    error_message: String::new(),
                    collection_id: req.collection_name.clone(),
                }))
            }
            Err(err) => {
                error!(
                    "Failed to create collection {}: {:?}",
                    req.collection_name, err
                );

                if let workspace_qdrant_core::storage::StorageError::Collection(ref msg) = err {
                    if msg.contains("already exists") {
                        return Err(Status::already_exists(format!(
                            "Collection '{}' already exists",
                            req.collection_name
                        )));
                    }
                }

                Err(map_storage_error(err))
            }
        }
    }

    async fn delete_collection(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        info!(
            "Deleting collection: {} (force={})",
            req.collection_name, req.force
        );

        validate_collection_name(&req.collection_name)?;

        if !req.force {
            match self
                .storage_client
                .collection_exists(&req.collection_name)
                .await
            {
                Ok(exists) if !exists => {
                    return Err(Status::not_found(format!(
                        "Collection '{}' does not exist",
                        req.collection_name
                    )));
                }
                Err(err) => {
                    warn!("Failed to check collection existence: {:?}", err);
                    // Continue with deletion attempt
                }
                _ => {}
            }
        }

        match self
            .storage_client
            .delete_collection(&req.collection_name)
            .await
        {
            Ok(_) => {
                info!("Successfully deleted collection: {}", req.collection_name);
                Ok(Response::new(()))
            }
            Err(err) => {
                error!(
                    "Failed to delete collection {}: {:?}",
                    req.collection_name, err
                );
                Err(map_storage_error(err))
            }
        }
    }

    /// List all collections (spec: ListCollections)
    async fn list_collections(
        &self,
        _request: Request<()>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        debug!("Listing collections");

        let collection_names = self
            .storage_client
            .list_collections()
            .await
            .map_err(map_storage_error)?;

        let mut collections = Vec::new();
        for name in &collection_names {
            match self.storage_client.get_collection_info(name).await {
                Ok(info_result) => {
                    collections.push(CollectionInfo {
                        name: info_result.name,
                        vectors_count: info_result.vectors_count as i64,
                        points_count: info_result.points_count as i64,
                        status: info_result.status,
                        created_at: Some(prost_types::Timestamp::from(SystemTime::now())),
                        aliases: info_result.aliases,
                    });
                }
                Err(err) => {
                    warn!("Failed to get info for collection {}: {:?}", name, err);
                    // Include collection with minimal info rather than failing entirely
                    collections.push(CollectionInfo {
                        name: name.clone(),
                        vectors_count: 0,
                        points_count: 0,
                        status: "unknown".to_string(),
                        created_at: None,
                        aliases: vec![],
                    });
                }
            }
        }

        let total_count = collections.len() as i32;
        Ok(Response::new(ListCollectionsResponse {
            collections,
            total_count,
        }))
    }

    /// Get collection metadata (spec: GetCollection)
    async fn get_collection(
        &self,
        request: Request<GetCollectionRequest>,
    ) -> Result<Response<GetCollectionResponse>, Status> {
        let req = request.into_inner();

        debug!("Getting collection info: {}", req.name);

        match self.storage_client.collection_exists(&req.name).await {
            Ok(false) => {
                return Ok(Response::new(GetCollectionResponse {
                    found: false,
                    info: None,
                    config: None,
                }));
            }
            Err(err) => {
                error!("Failed to check collection existence: {:?}", err);
                return Err(map_storage_error(err));
            }
            _ => {}
        }

        let info_result = self
            .storage_client
            .get_collection_info(&req.name)
            .await
            .map_err(map_storage_error)?;

        let info = CollectionInfo {
            name: info_result.name,
            vectors_count: info_result.vectors_count as i64,
            points_count: info_result.points_count as i64,
            status: info_result.status,
            created_at: Some(prost_types::Timestamp::from(SystemTime::now())),
            aliases: info_result.aliases,
        };

        let vector_size = info_result.vector_dimension.unwrap_or(384) as i32;

        let config = CollectionConfig {
            vector_size,
            distance_metric: "Cosine".to_string(),
            enable_indexing: true,
            metadata_schema: std::collections::HashMap::new(),
        };

        Ok(Response::new(GetCollectionResponse {
            found: true,
            info: Some(info),
            config: Some(config),
        }))
    }

    async fn create_collection_alias(
        &self,
        request: Request<CreateAliasRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        info!(
            "Creating alias '{}' -> '{}'",
            req.alias_name, req.collection_name
        );

        validate_collection_name(&req.alias_name)?;
        validate_collection_name(&req.collection_name)?;
        validate_alias_name(&req.alias_name)?;

        match self
            .storage_client
            .collection_exists(&req.collection_name)
            .await
        {
            Ok(false) => {
                return Err(Status::not_found(format!(
                    "Target collection '{}' does not exist",
                    req.collection_name
                )));
            }
            Err(err) => {
                error!("Failed to check collection existence: {:?}", err);
                return Err(map_storage_error(err));
            }
            _ => {}
        }

        match self
            .storage_client
            .create_alias(&req.collection_name, &req.alias_name)
            .await
        {
            Ok(_) => {
                info!(
                    "Successfully created alias '{}' -> '{}'",
                    req.alias_name, req.collection_name
                );
                Ok(Response::new(()))
            }
            Err(err) => {
                error!("Failed to create alias '{}': {:?}", req.alias_name, err);
                Err(map_storage_error(err))
            }
        }
    }

    async fn delete_collection_alias(
        &self,
        request: Request<DeleteAliasRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        info!("Deleting alias: {}", req.alias_name);

        validate_collection_name(&req.alias_name)?;

        match self.storage_client.delete_alias(&req.alias_name).await {
            Ok(_) => {
                info!("Successfully deleted alias '{}'", req.alias_name);
                Ok(Response::new(()))
            }
            Err(err) => {
                error!("Failed to delete alias '{}': {:?}", req.alias_name, err);
                Err(map_storage_error(err))
            }
        }
    }

    async fn rename_collection_alias(
        &self,
        request: Request<RenameAliasRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        info!(
            "Renaming alias '{}' -> '{}'",
            req.old_alias_name, req.new_alias_name
        );

        validate_collection_name(&req.old_alias_name)?;
        validate_collection_name(&req.new_alias_name)?;
        validate_collection_name(&req.collection_name)?;
        validate_alias_name(&req.new_alias_name)?;

        match self
            .storage_client
            .rename_alias(&req.old_alias_name, &req.new_alias_name)
            .await
        {
            Ok(_) => {
                info!(
                    "Successfully renamed alias '{}' -> '{}'",
                    req.old_alias_name, req.new_alias_name
                );
                Ok(Response::new(()))
            }
            Err(err) => {
                error!(
                    "Failed to rename alias '{}' to '{}': {:?}",
                    req.old_alias_name, req.new_alias_name, err
                );
                Err(map_storage_error(err))
            }
        }
    }
}

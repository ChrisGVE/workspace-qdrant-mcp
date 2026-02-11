//! CollectionService gRPC implementation
//!
//! Handles Qdrant collection lifecycle and alias management operations.
//! Provides 7 RPCs: CreateCollection, DeleteCollection, ListCollections, GetCollection,
//! CreateCollectionAlias, DeleteCollectionAlias, RenameCollectionAlias

use std::sync::Arc;
use std::time::SystemTime;
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn, error};
use workspace_qdrant_core::storage::StorageClient;

use crate::proto::{
    collection_service_server::CollectionService,
    CreateCollectionRequest, CreateCollectionResponse,
    DeleteCollectionRequest, CreateAliasRequest,
    DeleteAliasRequest, RenameAliasRequest,
    ListCollectionsResponse, GetCollectionRequest, GetCollectionResponse,
    CollectionInfo, CollectionConfig,
};

/// CollectionService implementation with Qdrant integration
pub struct CollectionServiceImpl {
    storage_client: Arc<StorageClient>,
}

impl CollectionServiceImpl {
    /// Create a new CollectionService with the provided storage client
    pub fn new(storage_client: Arc<StorageClient>) -> Self {
        Self { storage_client }
    }

    /// Create with default storage client
    pub fn default() -> Self {
        Self {
            storage_client: Arc::new(StorageClient::new()),
        }
    }
}

// Validation functions
impl CollectionServiceImpl {
    /// Validate collection name
    /// Rules: 3-255 chars, alphanumeric + underscore/hyphen, no leading numbers
    fn validate_collection_name(name: &str) -> Result<(), Status> {
        if name.len() < 3 {
            return Err(Status::invalid_argument(
                "Collection name must be at least 3 characters"
            ));
        }

        if name.len() > 255 {
            return Err(Status::invalid_argument(
                "Collection name must not exceed 255 characters"
            ));
        }

        // Check first character is not a number
        if name.chars().next().map(|c| c.is_numeric()).unwrap_or(false) {
            return Err(Status::invalid_argument(
                "Collection name cannot start with a number"
            ));
        }

        // Check all characters are valid
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(Status::invalid_argument(
                "Collection name can only contain alphanumeric characters, underscores, and hyphens"
            ));
        }

        Ok(())
    }

    /// Validate vector size
    /// Default embedding model (all-MiniLM-L6-v2) uses 384 dimensions
    fn validate_vector_size(size: i32) -> Result<(), Status> {
        if size <= 0 {
            return Err(Status::invalid_argument(
                "Vector size must be positive"
            ));
        }

        if size > 10000 {
            return Err(Status::invalid_argument(
                "Vector size exceeds maximum allowed (10000)"
            ));
        }

        // Warn if not standard size (but don't fail)
        if size != 384 && size != 768 && size != 1536 {
            warn!("Non-standard vector size: {}. Ensure this matches your embedding model.", size);
        }

        Ok(())
    }

    /// Canonical collection names that cannot be used as aliases (per ADR-001)
    const CANONICAL_COLLECTIONS: &'static [&'static str] = &["projects", "libraries", "memory"];

    /// Validate that an alias name does not conflict with canonical collection names
    fn validate_alias_name(alias_name: &str) -> Result<(), Status> {
        if Self::CANONICAL_COLLECTIONS.contains(&alias_name) {
            return Err(Status::invalid_argument(format!(
                "Cannot use '{}' as alias: conflicts with canonical collection name",
                alias_name
            )));
        }
        Ok(())
    }

    /// Map distance metric string to Qdrant Distance enum
    fn map_distance_metric(metric: &str) -> Result<String, Status> {
        match metric {
            "Cosine" => Ok("Cosine".to_string()),
            "Euclidean" => Ok("Euclid".to_string()),
            "Dot" => Ok("Dot".to_string()),
            _ => Err(Status::invalid_argument(
                format!("Invalid distance metric: {}. Must be one of: Cosine, Euclidean, Dot", metric)
            )),
        }
    }

    /// Map storage errors to gRPC Status
    fn map_storage_error(err: workspace_qdrant_core::storage::StorageError) -> Status {
        use workspace_qdrant_core::storage::StorageError;

        match err {
            StorageError::Collection(msg) if msg.contains("already exists") => {
                Status::already_exists(format!("Collection already exists: {}", msg))
            }
            StorageError::Collection(msg) if msg.contains("not found") => {
                Status::not_found(format!("Collection not found: {}", msg))
            }
            StorageError::Collection(msg) => {
                Status::failed_precondition(format!("Collection error: {}", msg))
            }
            StorageError::Connection(msg) => {
                Status::unavailable(format!("Connection error: {}", msg))
            }
            StorageError::Timeout(msg) => {
                Status::deadline_exceeded(format!("Timeout: {}", msg))
            }
            StorageError::Qdrant(err) => {
                // Check for specific Qdrant errors
                let err_msg = format!("{:?}", err);
                if err_msg.contains("rate limit") || err_msg.contains("too many requests") {
                    Status::resource_exhausted("Rate limit exceeded")
                } else if err_msg.contains("not found") {
                    Status::not_found(err_msg)
                } else {
                    Status::internal(format!("Qdrant error: {}", err_msg))
                }
            }
            _ => Status::internal(format!("Storage error: {}", err)),
        }
    }
}

#[tonic::async_trait]
impl CollectionService for CollectionServiceImpl {
    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();

        info!("Creating collection: {}", req.collection_name);
        debug!("Project ID: {}", req.project_id);

        // Validate collection name
        Self::validate_collection_name(&req.collection_name)?;

        // Extract and validate configuration
        let config = req.config.ok_or_else(|| {
            Status::invalid_argument("Collection configuration is required")
        })?;

        Self::validate_vector_size(config.vector_size)?;
        let _distance_metric = Self::map_distance_metric(&config.distance_metric)?;

        // Create collection using storage client
        match self.storage_client.create_collection(
            &req.collection_name,
            Some(config.vector_size as u64),
            None, // sparse vector size - not currently used
        ).await {
            Ok(_) => {
                info!("Successfully created collection: {}", req.collection_name);
                Ok(Response::new(CreateCollectionResponse {
                    success: true,
                    error_message: String::new(),
                    collection_id: req.collection_name.clone(),
                }))
            }
            Err(err) => {
                error!("Failed to create collection {}: {:?}", req.collection_name, err);

                // If it's an "already exists" error, return that specific status
                if let workspace_qdrant_core::storage::StorageError::Collection(ref msg) = err {
                    if msg.contains("already exists") {
                        return Err(Status::already_exists(format!(
                            "Collection '{}' already exists", req.collection_name
                        )));
                    }
                }

                Err(Self::map_storage_error(err))
            }
        }
    }

    async fn delete_collection(
        &self,
        request: Request<DeleteCollectionRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        info!("Deleting collection: {} (force={})", req.collection_name, req.force);

        // Validate collection name
        Self::validate_collection_name(&req.collection_name)?;

        // Check if collection exists (unless force is true)
        if !req.force {
            match self.storage_client.collection_exists(&req.collection_name).await {
                Ok(exists) if !exists => {
                    return Err(Status::not_found(format!(
                        "Collection '{}' does not exist", req.collection_name
                    )));
                }
                Err(err) => {
                    warn!("Failed to check collection existence: {:?}", err);
                    // Continue with deletion attempt
                }
                _ => {}
            }
        }

        // Delete the collection
        match self.storage_client.delete_collection(&req.collection_name).await {
            Ok(_) => {
                info!("Successfully deleted collection: {}", req.collection_name);
                Ok(Response::new(()))
            }
            Err(err) => {
                error!("Failed to delete collection {}: {:?}", req.collection_name, err);
                Err(Self::map_storage_error(err))
            }
        }
    }

    /// List all collections (spec: ListCollections)
    async fn list_collections(
        &self,
        _request: Request<()>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        debug!("Listing collections");

        // Query Qdrant for actual collection list
        let collection_names = self.storage_client.list_collections().await
            .map_err(Self::map_storage_error)?;

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

        // Check if collection exists
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
                return Err(Self::map_storage_error(err));
            }
            _ => {}
        }

        // Get actual collection info from Qdrant
        let info_result = self.storage_client.get_collection_info(&req.name).await
            .map_err(Self::map_storage_error)?;

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

        info!("Creating alias '{}' -> '{}'", req.alias_name, req.collection_name);

        // Validate both names
        Self::validate_collection_name(&req.alias_name)?;
        Self::validate_collection_name(&req.collection_name)?;

        // Reject aliases that match canonical collection names (before any I/O)
        Self::validate_alias_name(&req.alias_name)?;

        // Check that target collection exists
        match self.storage_client.collection_exists(&req.collection_name).await {
            Ok(false) => {
                return Err(Status::not_found(format!(
                    "Target collection '{}' does not exist", req.collection_name
                )));
            }
            Err(err) => {
                error!("Failed to check collection existence: {:?}", err);
                return Err(Self::map_storage_error(err));
            }
            _ => {}
        }

        match self.storage_client.create_alias(&req.collection_name, &req.alias_name).await {
            Ok(_) => {
                info!("Successfully created alias '{}' -> '{}'", req.alias_name, req.collection_name);
                Ok(Response::new(()))
            }
            Err(err) => {
                error!("Failed to create alias '{}': {:?}", req.alias_name, err);
                Err(Self::map_storage_error(err))
            }
        }
    }

    async fn delete_collection_alias(
        &self,
        request: Request<DeleteAliasRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        info!("Deleting alias: {}", req.alias_name);

        // Validate alias name
        Self::validate_collection_name(&req.alias_name)?;

        match self.storage_client.delete_alias(&req.alias_name).await {
            Ok(_) => {
                info!("Successfully deleted alias '{}'", req.alias_name);
                Ok(Response::new(()))
            }
            Err(err) => {
                error!("Failed to delete alias '{}': {:?}", req.alias_name, err);
                Err(Self::map_storage_error(err))
            }
        }
    }

    async fn rename_collection_alias(
        &self,
        request: Request<RenameAliasRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        info!("Renaming alias '{}' -> '{}'", req.old_alias_name, req.new_alias_name);

        // Validate all names
        Self::validate_collection_name(&req.old_alias_name)?;
        Self::validate_collection_name(&req.new_alias_name)?;
        Self::validate_collection_name(&req.collection_name)?;

        // Reject new alias names that match canonical collection names
        Self::validate_alias_name(&req.new_alias_name)?;

        match self.storage_client.rename_alias(&req.old_alias_name, &req.new_alias_name).await {
            Ok(_) => {
                info!("Successfully renamed alias '{}' -> '{}'", req.old_alias_name, req.new_alias_name);
                Ok(Response::new(()))
            }
            Err(err) => {
                error!("Failed to rename alias '{}' to '{}': {:?}", req.old_alias_name, req.new_alias_name, err);
                Err(Self::map_storage_error(err))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_collection_name() {
        // Valid names
        assert!(CollectionServiceImpl::validate_collection_name("test_collection").is_ok());
        assert!(CollectionServiceImpl::validate_collection_name("my-collection").is_ok());
        assert!(CollectionServiceImpl::validate_collection_name("collection123").is_ok());
        assert!(CollectionServiceImpl::validate_collection_name("a_b_c").is_ok());

        // Invalid: too short
        assert!(CollectionServiceImpl::validate_collection_name("ab").is_err());

        // Invalid: starts with number
        assert!(CollectionServiceImpl::validate_collection_name("1collection").is_err());

        // Invalid: special characters
        assert!(CollectionServiceImpl::validate_collection_name("collection@123").is_err());
        assert!(CollectionServiceImpl::validate_collection_name("collection.name").is_err());

        // Invalid: too long
        let long_name = "a".repeat(256);
        assert!(CollectionServiceImpl::validate_collection_name(&long_name).is_err());
    }

    #[test]
    fn test_validate_canonical_collection_names() {
        // Canonical collection names per ADR-001 must be valid
        assert!(CollectionServiceImpl::validate_collection_name("projects").is_ok());
        assert!(CollectionServiceImpl::validate_collection_name("libraries").is_ok());
        assert!(CollectionServiceImpl::validate_collection_name("memory").is_ok());

        // Legacy underscore-prefixed names are syntactically valid for migration compatibility
        assert!(CollectionServiceImpl::validate_collection_name("_projects").is_ok());
        assert!(CollectionServiceImpl::validate_collection_name("_libraries").is_ok());
        assert!(CollectionServiceImpl::validate_collection_name("_memory").is_ok());
    }

    #[test]
    fn test_validate_vector_size() {
        // Valid sizes
        assert!(CollectionServiceImpl::validate_vector_size(384).is_ok());
        assert!(CollectionServiceImpl::validate_vector_size(768).is_ok());
        assert!(CollectionServiceImpl::validate_vector_size(1536).is_ok());
        assert!(CollectionServiceImpl::validate_vector_size(512).is_ok());

        // Invalid: zero or negative
        assert!(CollectionServiceImpl::validate_vector_size(0).is_err());
        assert!(CollectionServiceImpl::validate_vector_size(-1).is_err());

        // Invalid: too large
        assert!(CollectionServiceImpl::validate_vector_size(10001).is_err());
    }

    #[test]
    fn test_map_distance_metric() {
        // Valid metrics
        assert_eq!(
            CollectionServiceImpl::map_distance_metric("Cosine").unwrap(),
            "Cosine"
        );
        assert_eq!(
            CollectionServiceImpl::map_distance_metric("Euclidean").unwrap(),
            "Euclid"
        );
        assert_eq!(
            CollectionServiceImpl::map_distance_metric("Dot").unwrap(),
            "Dot"
        );

        // Invalid metric
        assert!(CollectionServiceImpl::map_distance_metric("Invalid").is_err());
        assert!(CollectionServiceImpl::map_distance_metric("manhattan").is_err());
    }

    #[test]
    fn test_validate_alias_name_rejects_canonical() {
        // Canonical collection names must be rejected as aliases
        assert!(CollectionServiceImpl::validate_alias_name("projects").is_err());
        assert!(CollectionServiceImpl::validate_alias_name("libraries").is_err());
        assert!(CollectionServiceImpl::validate_alias_name("memory").is_err());
    }

    #[test]
    fn test_validate_alias_name_allows_non_canonical() {
        // Non-canonical names should be allowed
        assert!(CollectionServiceImpl::validate_alias_name("my-alias").is_ok());
        assert!(CollectionServiceImpl::validate_alias_name("project_v2").is_ok());
        assert!(CollectionServiceImpl::validate_alias_name("lib-backup").is_ok());
        assert!(CollectionServiceImpl::validate_alias_name("_projects").is_ok());
    }
}

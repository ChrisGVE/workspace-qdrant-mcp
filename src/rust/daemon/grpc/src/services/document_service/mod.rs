//! DocumentService gRPC implementation
//!
//! Handles direct text ingestion for non-file-based content.
//! Provides 3 RPCs: IngestText, UpdateDocument, DeleteDocument
//!
//! This service is designed for user-provided text content such as:
//! - Manual notes and annotations
//! - Chat snippets and conversations
//! - Scraped web content
//! - API responses
//! - Any text not originating from files
//!
//! ## Multi-Tenant Routing (Task 406)
//!
//! Routes content to unified collections based on collection_basename:
//! - `memory`, `agent_memory` -> Direct collection names (no multi-tenant)
//! - Other basenames -> Routes to canonical `projects` or `libraries`:
//!   - If tenant_id is project ID format -> `projects` with project_id metadata
//!     (path hashes like "path_abc123..." or sanitized URLs like "github_com_user_repo")
//!   - Otherwise -> `libraries` with library_name metadata
//!     (human-readable names like "react", "numpy", "lodash")

mod embedding;
mod ingestion;
mod routing;

use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use workspace_qdrant_core::storage::StorageClient;
use wqm_common::constants::{COLLECTION_LIBRARIES, COLLECTION_PROJECTS};
use wqm_common::timestamps;

pub use embedding::{EmbeddingCacheMetrics, CACHE_METRICS};

use crate::proto::{
    document_service_server::DocumentService, DeleteDocumentRequest, IngestTextRequest,
    IngestTextResponse, UpdateDocumentRequest, UpdateDocumentResponse,
};

/// DocumentService implementation with text chunking and embedding generation
pub struct DocumentServiceImpl {
    storage_client: Arc<StorageClient>,
    chunk_size: usize,
    chunk_overlap: usize,
}

impl DocumentServiceImpl {
    /// Create a new DocumentService with the provided storage client
    pub fn new(storage_client: Arc<StorageClient>) -> Self {
        Self {
            storage_client,
            chunk_size: ingestion::DEFAULT_CHUNK_SIZE,
            chunk_overlap: ingestion::DEFAULT_CHUNK_OVERLAP,
        }
    }

    /// Create with default storage client
    pub fn default() -> Self {
        Self::new(Arc::new(StorageClient::new()))
    }

    /// Create with custom chunking configuration
    pub fn with_config(
        storage_client: Arc<StorageClient>,
        chunk_size: usize,
        chunk_overlap: usize,
    ) -> Self {
        Self {
            storage_client,
            chunk_size,
            chunk_overlap,
        }
    }

    /// Get embedding cache metrics for monitoring
    pub fn get_cache_metrics() -> (u64, u64, u64, f64) {
        embedding::get_cache_metrics()
    }
}

#[tonic::async_trait]
impl DocumentService for DocumentServiceImpl {
    #[tracing::instrument(skip_all, fields(method = "DocumentService.ingest_text"))]
    async fn ingest_text(
        &self,
        request: Request<IngestTextRequest>,
    ) -> Result<Response<IngestTextResponse>, Status> {
        let req = request.into_inner();

        info!(
            "Ingesting text into collection basename '{}' for tenant '{}'",
            req.collection_basename, req.tenant_id
        );

        let (collection_name, tenant_type, tenant_value) =
            routing::determine_collection_routing(&req.collection_basename, &req.tenant_id)?;

        info!(
            "Multi-tenant routing: collection='{}', {}='{}'",
            collection_name, tenant_type, tenant_value
        );

        let document_id = if let Some(provided_id) = req.document_id {
            if !provided_id.is_empty() {
                routing::validate_document_id(&provided_id)?;
                provided_id
            } else {
                Uuid::new_v4().to_string()
            }
        } else {
            Uuid::new_v4().to_string()
        };

        debug!("Using document_id: {}", document_id);

        let mut enriched_metadata = req.metadata.clone();
        enriched_metadata.insert(tenant_type.clone(), tenant_value.clone());
        enriched_metadata.insert(
            "collection_basename".to_string(),
            req.collection_basename.clone(),
        );

        let response = ingestion::ingest_text_internal(
            &self.storage_client,
            req.content,
            collection_name,
            document_id,
            enriched_metadata,
            req.chunk_text,
            self.chunk_size,
            self.chunk_overlap,
        )
        .await?;

        Ok(Response::new(response))
    }

    #[tracing::instrument(skip_all, fields(method = "DocumentService.update_document"))]
    async fn update_document(
        &self,
        request: Request<UpdateDocumentRequest>,
    ) -> Result<Response<UpdateDocumentResponse>, Status> {
        let req = request.into_inner();

        info!("Updating document '{}'", req.document_id);

        routing::validate_document_id(&req.document_id)?;

        let collection_name = if let Some(coll_name) = req.collection_name {
            routing::validate_collection_name(&coll_name)?;
            coll_name
        } else {
            return Err(Status::invalid_argument(
                "Collection name is required for updates",
            ));
        };

        info!(
            "UpdateDocument: collection='{}', document='{}'",
            collection_name, req.document_id
        );

        match self
            .storage_client
            .delete_points_by_document_id(&collection_name, &req.document_id)
            .await
        {
            Ok(_) => {
                info!(
                    "Deleted existing chunks for document '{}' in '{}'",
                    req.document_id, collection_name
                );
            }
            Err(err) => {
                warn!(
                    "Failed to delete existing chunks for document '{}': {:?}",
                    req.document_id, err
                );
            }
        }

        let mut enriched_metadata = req.metadata.clone();
        enriched_metadata.insert("updated_at".to_string(), timestamps::now_utc());

        let response = ingestion::ingest_text_internal(
            &self.storage_client,
            req.content,
            collection_name,
            req.document_id.clone(),
            enriched_metadata,
            true,
            self.chunk_size,
            self.chunk_overlap,
        )
        .await?;

        Ok(Response::new(UpdateDocumentResponse {
            success: response.success,
            error_message: response.error_message,
            updated_at: Some(prost_types::Timestamp {
                seconds: chrono::Utc::now().timestamp(),
                nanos: 0,
            }),
        }))
    }

    #[tracing::instrument(skip_all, fields(method = "DocumentService.delete_document"))]
    async fn delete_document(
        &self,
        request: Request<DeleteDocumentRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();

        info!(
            "DeleteDocument: document='{}', collection='{}'",
            req.document_id, req.collection_name
        );

        routing::validate_document_id(&req.document_id)?;
        routing::validate_collection_name(&req.collection_name)?;

        if req.collection_name == COLLECTION_PROJECTS || req.collection_name == COLLECTION_LIBRARIES
        {
            debug!(
                "Deleting from unified collection '{}' - document_id filter will be used",
                req.collection_name
            );
        }

        match self
            .storage_client
            .collection_exists(&req.collection_name)
            .await
        {
            Ok(false) => {
                return Err(Status::not_found(format!(
                    "Collection '{}' does not exist",
                    req.collection_name
                )));
            }
            Err(err) => {
                error!("Failed to check collection existence: {:?}", err);
                return Err(Status::unavailable(format!(
                    "Failed to check collection: {}",
                    err
                )));
            }
            _ => {}
        }

        match self
            .storage_client
            .delete_points_by_document_id(&req.collection_name, &req.document_id)
            .await
        {
            Ok(_) => {
                info!(
                    "Successfully deleted document '{}' from '{}'",
                    req.document_id, req.collection_name
                );
                Ok(Response::new(()))
            }
            Err(err) => {
                error!(
                    "Failed to delete document '{}' from '{}': {:?}",
                    req.document_id, req.collection_name, err
                );
                Err(Status::internal(format!(
                    "Failed to delete document: {}",
                    err
                )))
            }
        }
    }
}

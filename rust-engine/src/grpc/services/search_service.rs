//! Search service gRPC implementation

use crate::daemon::WorkspaceDaemon;
use crate::proto::{
    search_service_server::SearchService,
    HybridSearchRequest, HybridSearchResponse,
    SemanticSearchRequest, SemanticSearchResponse,
    KeywordSearchRequest, KeywordSearchResponse,
    SuggestionsRequest, SuggestionsResponse,
    SearchResult, SearchMetadata,
};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info};

/// Search service implementation
#[derive(Debug)]
pub struct SearchServiceImpl {
    daemon: Arc<WorkspaceDaemon>,
}

impl SearchServiceImpl {
    pub fn new(daemon: Arc<WorkspaceDaemon>) -> Self {
        Self { daemon }
    }
}

#[tonic::async_trait]
impl SearchService for SearchServiceImpl {
    async fn hybrid_search(
        &self,
        request: Request<HybridSearchRequest>,
    ) -> Result<Response<HybridSearchResponse>, Status> {
        let req = request.into_inner();
        debug!("Hybrid search requested: {}", req.query);

        // TODO: Implement actual hybrid search
        let response = HybridSearchResponse {
            results: vec![
                SearchResult {
                    document_id: uuid::Uuid::new_v4().to_string(),
                    collection_name: "example".to_string(),
                    score: 0.95,
                    semantic_score: 0.9,
                    keyword_score: 0.85,
                    title: "Example Document".to_string(),
                    content_snippet: "This is an example search result...".to_string(),
                    metadata: std::collections::HashMap::new(),
                    file_path: "/path/to/document.txt".to_string(),
                    matched_terms: vec!["example".to_string()],
                },
            ],
            metadata: Some(SearchMetadata {
                total_results: 1,
                max_score: 0.95,
                search_time: Some(prost_types::Timestamp {
                    seconds: chrono::Utc::now().timestamp(),
                    nanos: 0,
                }),
                search_duration_ms: 25,
                searched_collections: vec!["example".to_string()],
            }),
            query_id: uuid::Uuid::new_v4().to_string(),
        };

        Ok(Response::new(response))
    }

    async fn semantic_search(
        &self,
        request: Request<SemanticSearchRequest>,
    ) -> Result<Response<SemanticSearchResponse>, Status> {
        let req = request.into_inner();
        debug!("Semantic search requested: {}", req.query);

        // TODO: Implement actual semantic search
        let response = SemanticSearchResponse {
            results: vec![],
            metadata: Some(SearchMetadata {
                total_results: 0,
                max_score: 0.0,
                search_time: Some(prost_types::Timestamp {
                    seconds: chrono::Utc::now().timestamp(),
                    nanos: 0,
                }),
                search_duration_ms: 15,
                searched_collections: vec![],
            }),
            query_id: uuid::Uuid::new_v4().to_string(),
        };

        Ok(Response::new(response))
    }

    async fn keyword_search(
        &self,
        request: Request<KeywordSearchRequest>,
    ) -> Result<Response<KeywordSearchResponse>, Status> {
        let req = request.into_inner();
        debug!("Keyword search requested: {}", req.query);

        // TODO: Implement actual keyword search
        let response = KeywordSearchResponse {
            results: vec![],
            metadata: Some(SearchMetadata {
                total_results: 0,
                max_score: 0.0,
                search_time: Some(prost_types::Timestamp {
                    seconds: chrono::Utc::now().timestamp(),
                    nanos: 0,
                }),
                search_duration_ms: 10,
                searched_collections: vec![],
            }),
            query_id: uuid::Uuid::new_v4().to_string(),
        };

        Ok(Response::new(response))
    }

    async fn get_suggestions(
        &self,
        request: Request<SuggestionsRequest>,
    ) -> Result<Response<SuggestionsResponse>, Status> {
        let req = request.into_inner();
        debug!("Suggestions requested for: {}", req.partial_query);

        // TODO: Implement actual suggestions
        let response = SuggestionsResponse {
            suggestions: vec![
                format!("{} complete", req.partial_query),
                format!("{} suggestion", req.partial_query),
            ],
            metadata: Some(SearchMetadata {
                total_results: 2,
                max_score: 1.0,
                search_time: Some(prost_types::Timestamp {
                    seconds: chrono::Utc::now().timestamp(),
                    nanos: 0,
                }),
                search_duration_ms: 5,
                searched_collections: vec![],
            }),
        };

        Ok(Response::new(response))
    }
}
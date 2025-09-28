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
use tracing::debug;

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

        // Access daemon config to prevent unused field warning
        let _config = self.daemon.config();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;
    use tonic::Request;

    fn create_test_daemon_config() -> DaemonConfig {
        // Use in-memory SQLite database for tests
        let db_path = ":memory:";

        DaemonConfig {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 50054,
                max_connections: 100,
                connection_timeout_secs: 30,
                request_timeout_secs: 60,
                enable_tls: false,
                security: crate::config::SecurityConfig::default(),
                transport: crate::config::TransportConfig::default(),
                message: crate::config::MessageConfig::default(),
                compression: crate::config::CompressionConfig::default(),
                streaming: crate::config::StreamingConfig::default(),
            },
            database: DatabaseConfig {
                sqlite_path: db_path.to_string(),
                max_connections: 5,
                connection_timeout_secs: 30,
                enable_wal: true,
            },
            qdrant: QdrantConfig {
                url: "http://localhost:6333".to_string(),
                api_key: None,
                timeout_secs: 30,
                max_retries: 3,
                default_collection: CollectionConfig {
                    vector_size: 384,
                    distance_metric: "Cosine".to_string(),
                    enable_indexing: true,
                    replication_factor: 1,
                    shard_number: 1,
                },
            },
            processing: ProcessingConfig {
                max_concurrent_tasks: 2,
                default_chunk_size: 1000,
                default_chunk_overlap: 200,
                max_file_size_bytes: 1024 * 1024,
                supported_extensions: vec!["txt".to_string(), "md".to_string()],
                enable_lsp: false,
                lsp_timeout_secs: 10,
            },
            file_watcher: FileWatcherConfig {
                enabled: false,
                debounce_ms: 500,
                max_watched_dirs: 10,
                ignore_patterns: vec![],
                recursive: true,
            },
            metrics: MetricsConfig {
                enabled: false,
                collection_interval_secs: 60,
                retention_days: 30,
                enable_prometheus: false,
                prometheus_port: 9090,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                file_path: None,
                json_format: false,
                max_file_size_mb: 100,
                max_files: 5,
            },
        }
    }

    async fn create_test_daemon() -> Arc<WorkspaceDaemon> {
        let config = create_test_daemon_config();
        Arc::new(WorkspaceDaemon::new(config).await.expect("Failed to create daemon"))
    }

    // Helper function to create proper HybridSearchRequest
    fn create_hybrid_search_request(
        query: &str,
        semantic_weight: f32,
        keyword_weight: f32,
        limit: i32,
        collection_names: Vec<String>
    ) -> HybridSearchRequest {
        HybridSearchRequest {
            query: query.to_string(),
            context: crate::proto::SearchContext::Project as i32,
            options: Some(crate::proto::SearchOptions {
                limit,
                score_threshold: 0.0,
                include_metadata: true,
                include_content: true,
                ranking: Some(crate::proto::RankingOptions {
                    semantic_weight,
                    keyword_weight,
                    rrf_constant: 60.0,
                }),
            }),
            project_id: "test_project".to_string(),
            collection_names,
        }
    }

    // Helper function to create SemanticSearchRequest
    fn create_semantic_search_request(
        query: &str,
        similarity_threshold: f32,
        limit: i32,
        collection_names: Vec<String>
    ) -> SemanticSearchRequest {
        SemanticSearchRequest {
            query: query.to_string(),
            context: crate::proto::SearchContext::Project as i32,
            options: Some(crate::proto::SearchOptions {
                limit,
                score_threshold: similarity_threshold,
                include_metadata: true,
                include_content: true,
                ranking: None,
            }),
            project_id: "test_project".to_string(),
            collection_names,
        }
    }

    // Helper function to create KeywordSearchRequest
    fn create_keyword_search_request(
        query: &str,
        limit: i32,
        collection_names: Vec<String>
    ) -> KeywordSearchRequest {
        KeywordSearchRequest {
            query: query.to_string(),
            context: crate::proto::SearchContext::Project as i32,
            options: Some(crate::proto::SearchOptions {
                limit,
                score_threshold: 0.0,
                include_metadata: true,
                include_content: true,
                ranking: None,
            }),
            project_id: "test_project".to_string(),
            collection_names,
        }
    }

    // Helper function to create SuggestionsRequest
    fn create_suggestions_request(
        partial_query: &str,
        limit: i32,
        _collection_names: Vec<String>
    ) -> SuggestionsRequest {
        SuggestionsRequest {
            partial_query: partial_query.to_string(),
            context: crate::proto::SearchContext::Project as i32,
            max_suggestions: limit,
            project_id: "test_project".to_string(),
        }
    }

    #[tokio::test]
    async fn test_search_service_impl_new() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon.clone());

        assert!(Arc::ptr_eq(&service.daemon, &daemon));
    }

    #[tokio::test]
    async fn test_search_service_impl_debug() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let debug_str = format!("{:?}", service);
        assert!(debug_str.contains("SearchServiceImpl"));
        assert!(debug_str.contains("daemon"));
    }

    #[tokio::test]
    async fn test_hybrid_search_basic() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let request = Request::new(HybridSearchRequest {
            query: "test query".to_string(),
            context: crate::proto::SearchContext::Collection as i32,
            options: Some(crate::proto::SearchOptions {
                limit: 10,
                score_threshold: 0.0,
                include_metadata: true,
                include_content: true,
                ranking: Some(crate::proto::RankingOptions {
                    semantic_weight: 0.7,
                    keyword_weight: 0.3,
                    rrf_constant: 60.0,
                }),
            }),
            project_id: "test_project".to_string(),
            collection_names: vec!["test_collection".to_string()],
        });

        let result = service.hybrid_search(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(!response.query_id.is_empty());
        assert_eq!(response.results.len(), 1);
        assert!(response.metadata.is_some());

        let metadata = response.metadata.unwrap();
        assert_eq!(metadata.total_results, 1);
        assert_eq!(metadata.max_score, 0.95);
        assert!(metadata.search_time.is_some());
        assert_eq!(metadata.search_duration_ms, 25);
    }

    #[tokio::test]
    async fn test_hybrid_search_different_queries() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let queries = [
            "simple query",
            "complex search with multiple terms",
            "特殊字符查询", // Unicode characters
            "query with symbols !@#$%",
            "12345 numeric query",
        ];

        for query in queries {
            let request = Request::new(create_hybrid_search_request(
                query,
                0.6,
                0.4,
                5,
                vec!["test_collection".to_string()]
            ));

            let result = service.hybrid_search(request).await;
            assert!(result.is_ok(), "Failed for query: {}", query);

            let response = result.unwrap().into_inner();
            assert!(!response.query_id.is_empty());
        }
    }

    #[tokio::test]
    async fn test_hybrid_search_different_weights() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let weight_combinations = [
            (1.0, 0.0), // Pure semantic
            (0.0, 1.0), // Pure keyword
            (0.5, 0.5), // Equal weights
            (0.8, 0.2), // Semantic-heavy
            (0.3, 0.7), // Keyword-heavy
        ];

        for (semantic, keyword) in weight_combinations {
            let request = Request::new(create_hybrid_search_request(
                "test query",
                semantic,
                keyword,
                10,
                vec!["test_collection".to_string()]
            ));

            let result = service.hybrid_search(request).await;
            assert!(result.is_ok(), "Failed for weights: semantic={}, keyword={}", semantic, keyword);
        }
    }

    #[tokio::test]
    async fn test_hybrid_search_metadata_validation() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let request = Request::new(create_hybrid_search_request(
            "test query with metadata",
            0.7,
            0.3,
            10,
            vec!["test_collection".to_string()]
        ));

        let result = service.hybrid_search(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(!response.query_id.is_empty());
        assert!(response.metadata.is_some());
    }

    #[tokio::test]
    async fn test_hybrid_search_pagination() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let pagination_params = [
            (10, 0),   // First page
            (10, 10),  // Second page
            (5, 0),    // Smaller page size
            (20, 40),  // Larger offset
            (1, 0),    // Single result
        ];

        for (limit, _offset) in pagination_params {
            let request = Request::new(create_hybrid_search_request(
                "pagination test",
                0.7,
                0.3,
                limit,
                vec!["test_collection".to_string()]
            ));

            let result = service.hybrid_search(request).await;
            assert!(result.is_ok(), "Failed for limit={}", limit);
        }
    }

    #[tokio::test]
    async fn test_semantic_search_basic() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let request = Request::new(create_semantic_search_request(
            "semantic search query",
            0.7,
            10,
            vec!["test_collection".to_string()]
        ));

        let result = service.semantic_search(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(!response.query_id.is_empty());
        assert!(response.metadata.is_some());

        let metadata = response.metadata.unwrap();
        assert!(metadata.search_time.is_some());
        assert_eq!(metadata.search_duration_ms, 15);
    }

    #[tokio::test]
    async fn test_semantic_search_different_thresholds() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let thresholds = [0.1, 0.3, 0.5, 0.7, 0.9];

        for threshold in thresholds {
            let request = Request::new(create_semantic_search_request(
                "threshold test",
                threshold,
                10,
                vec!["test_collection".to_string()]
            ));

            let result = service.semantic_search(request).await;
            assert!(result.is_ok(), "Failed for threshold: {}", threshold);
        }
    }

    #[tokio::test]
    async fn test_keyword_search_basic() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let request = Request::new(create_keyword_search_request(
            "keyword search",
            10,
            vec!["test_collection".to_string()]
        ));

        let result = service.keyword_search(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(!response.query_id.is_empty());
        assert!(response.metadata.is_some());

        let metadata = response.metadata.unwrap();
        assert!(metadata.search_time.is_some());
        assert_eq!(metadata.search_duration_ms, 10);
    }

    #[tokio::test]
    async fn test_keyword_search_with_different_limits() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let limits = [1, 5, 10, 20, 50];

        for limit in limits {
            let request = Request::new(create_keyword_search_request(
                "limit test search",
                limit,
                vec!["test_collection".to_string()]
            ));

            let result = service.keyword_search(request).await;
            assert!(result.is_ok(), "Failed for limit: {}", limit);

            let response = result.unwrap().into_inner();
            assert!(!response.query_id.is_empty());
        }
    }

    #[tokio::test]
    async fn test_keyword_search_different_queries() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let queries = ["simple", "complex search terms", "special!@#$%", "numbers123"];

        for query in queries {
            let request = Request::new(create_keyword_search_request(
                query,
                10,
                vec!["test_collection".to_string()]
            ));

            let result = service.keyword_search(request).await;
            assert!(result.is_ok(), "Failed for query: {}", query);
        }
    }

    #[tokio::test]
    async fn test_get_suggestions_basic() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let request = Request::new(create_suggestions_request(
            "test",
            5,
            vec!["test_collection".to_string()]
        ));

        let result = service.get_suggestions(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.suggestions.len(), 2);
        assert!(response.suggestions[0].contains("test"));
        assert!(response.suggestions[1].contains("test"));
        assert!(response.metadata.is_some());

        let metadata = response.metadata.unwrap();
        assert_eq!(metadata.total_results, 2);
        assert_eq!(metadata.search_duration_ms, 5);
    }

    #[tokio::test]
    async fn test_get_suggestions_different_queries() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let partial_queries = [
            "t",
            "te",
            "test",
            "testing",
            "query",
            "search",
        ];

        for partial in partial_queries {
            let request = Request::new(create_suggestions_request(
                partial,
                5,
                vec!["test_collection".to_string()]
            ));

            let result = service.get_suggestions(request).await;
            assert!(result.is_ok(), "Failed for partial query: {}", partial);

            let response = result.unwrap().into_inner();
            assert_eq!(response.suggestions.len(), 2);
            for suggestion in &response.suggestions {
                assert!(suggestion.contains(partial));
            }
        }
    }

    #[tokio::test]
    async fn test_get_suggestions_different_limits() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let limits = [1, 3, 5, 10, 20];

        for limit in limits {
            let request = Request::new(create_suggestions_request(
                "suggestion",
                limit,
                vec!["test_collection".to_string()]
            ));

            let result = service.get_suggestions(request).await;
            assert!(result.is_ok(), "Failed for limit: {}", limit);

            let response = result.unwrap().into_inner();
            // Current implementation returns 2 suggestions regardless of limit
            assert_eq!(response.suggestions.len(), 2);
        }
    }

    #[tokio::test]
    async fn test_search_service_multiple_collections() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let collections = vec![
            "collection1".to_string(),
            "collection2".to_string(),
            "documents".to_string(),
            "code".to_string(),
        ];

        let request = Request::new(create_hybrid_search_request(
            "multi-collection search",
            0.7,
            0.3,
            10,
            collections.clone()
        ));

        let result = service.hybrid_search(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert!(!response.query_id.is_empty());
    }

    #[tokio::test]
    async fn test_search_timestamps() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let before_search = chrono::Utc::now().timestamp();

        let request = Request::new(create_hybrid_search_request(
            "timestamp test",
            0.7,
            0.3,
            10,
            vec!["test_collection".to_string()]
        ));

        let result = service.hybrid_search(request).await;
        assert!(result.is_ok());

        let after_search = chrono::Utc::now().timestamp();
        let response = result.unwrap().into_inner();

        assert!(response.metadata.is_some());
        let metadata = response.metadata.unwrap();
        assert!(metadata.search_time.is_some());

        let search_timestamp = metadata.search_time.unwrap().seconds;
        assert!(search_timestamp >= before_search);
        assert!(search_timestamp <= after_search);
    }

    #[tokio::test]
    async fn test_concurrent_searches() {
        let daemon = create_test_daemon().await;
        let service = Arc::new(SearchServiceImpl::new(daemon));

        let mut handles = vec![];

        // Perform multiple concurrent searches
        for i in 0..5 {
            let service_clone = Arc::clone(&service);
            let handle = tokio::spawn(async move {
                let request = Request::new(create_hybrid_search_request(
                    &format!("concurrent search {}", i),
                    0.7,
                    0.3,
                    10,
                    vec!["test_collection".to_string()]
                ));

                service_clone.hybrid_search(request).await
            });
            handles.push(handle);
        }

        // Wait for all searches to complete
        let results: Vec<_> = futures_util::future::join_all(handles).await;

        // All searches should complete successfully
        for (i, result) in results.into_iter().enumerate() {
            let task_result = result.unwrap();
            assert!(task_result.is_ok(), "Search {} failed", i);

            let response = task_result.unwrap().into_inner();
            assert!(!response.query_id.is_empty());
        }
    }

    #[tokio::test]
    async fn test_search_result_structure() {
        let daemon = create_test_daemon().await;
        let service = SearchServiceImpl::new(daemon);

        let request = Request::new(create_hybrid_search_request(
            "structure test",
            0.7,
            0.3,
            10,
            vec!["test_collection".to_string()]
        ));

        let result = service.hybrid_search(request).await;
        assert!(result.is_ok());

        let response = result.unwrap().into_inner();
        assert_eq!(response.results.len(), 1);

        let search_result = &response.results[0];
        assert!(!search_result.document_id.is_empty());
        assert_eq!(search_result.collection_name, "example");
        assert_eq!(search_result.score, 0.95);
        assert_eq!(search_result.semantic_score, 0.9);
        assert_eq!(search_result.keyword_score, 0.85);
        assert_eq!(search_result.title, "Example Document");
        assert!(!search_result.content_snippet.is_empty());
        assert!(!search_result.file_path.is_empty());
        assert_eq!(search_result.matched_terms.len(), 1);
    }

    #[test]
    fn test_search_service_impl_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<SearchServiceImpl>();
        assert_sync::<SearchServiceImpl>();
    }
}
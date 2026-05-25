//! Unit tests for graph_service helpers, validation, and handler integration.

use workspace_qdrant_core::graph::EdgeType;

use super::helpers::parse_edge_type_filter;

#[test]
fn test_edge_type_parsing() {
    assert!(EdgeType::from_str("CALLS").is_some());
    assert!(EdgeType::from_str("IMPORTS").is_some());
    assert!(EdgeType::from_str("USES_TYPE").is_some());
    assert!(EdgeType::from_str("CONTAINS").is_some());
    assert!(EdgeType::from_str("EXTENDS").is_some());
    assert!(EdgeType::from_str("IMPLEMENTS").is_some());
    assert!(EdgeType::from_str("INVALID").is_none());
}

#[test]
fn test_parse_edge_type_filter_empty() {
    let result = parse_edge_type_filter(&[]);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn test_parse_edge_type_filter_valid() {
    let types = vec!["CALLS".to_string(), "IMPORTS".to_string()];
    let result = parse_edge_type_filter(&types);
    assert!(result.is_ok());
    let filter = result.unwrap().unwrap();
    assert_eq!(filter.len(), 2);
}

#[test]
fn test_parse_edge_type_filter_invalid() {
    let types = vec!["CALLS".to_string(), "INVALID".to_string()];
    let result = parse_edge_type_filter(&types);
    assert!(result.is_err());
}

// ── ImpactAnalysisRequest.file_path (relative, optional) path validation ──

mod path_validation {
    use tonic::Request;
    use workspace_qdrant_core::graph::create_sqlite_graph_store;

    use crate::proto::graph_service_server::GraphService;
    use crate::proto::ImpactAnalysisRequest;
    use crate::services::GraphServiceImpl;

    /// Create a minimal graph store backed by a temp directory.
    async fn test_graph_service() -> (GraphServiceImpl, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let store = create_sqlite_graph_store(tmp.path()).await.unwrap();
        (GraphServiceImpl::new(store), tmp)
    }

    #[tokio::test]
    async fn test_impact_analysis_absolute_file_path_rejected() {
        let (service, _tmp) = test_graph_service().await;

        let request = Request::new(ImpactAnalysisRequest {
            tenant_id: "abcd12345678".to_string(),
            symbol_name: "my_func".to_string(),
            file_path: Some("/absolute/path.rs".to_string()),
            branch: None,
        });

        let result = service.impact_analysis(request).await;
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("file_path"),
            "error should mention field name, got: {}",
            status.message()
        );
    }

    #[tokio::test]
    async fn test_impact_analysis_parent_dir_file_path_rejected() {
        let (service, _tmp) = test_graph_service().await;

        let request = Request::new(ImpactAnalysisRequest {
            tenant_id: "abcd12345678".to_string(),
            symbol_name: "my_func".to_string(),
            file_path: Some("src/../secret.rs".to_string()),
            branch: None,
        });

        let result = service.impact_analysis(request).await;
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains(".."));
    }

    #[tokio::test]
    async fn test_impact_analysis_empty_file_path_allowed() {
        // Empty file_path means "no file scope" — should not be rejected.
        let (service, _tmp) = test_graph_service().await;

        let request = Request::new(ImpactAnalysisRequest {
            tenant_id: "abcd12345678".to_string(),
            symbol_name: "my_func".to_string(),
            file_path: Some(String::new()),
            branch: None,
        });

        // Empty string is filtered to None by the handler.
        let result = service.impact_analysis(request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_impact_analysis_valid_relative_path_accepted() {
        let (service, _tmp) = test_graph_service().await;

        let request = Request::new(ImpactAnalysisRequest {
            tenant_id: "abcd12345678".to_string(),
            symbol_name: "my_func".to_string(),
            file_path: Some("src/lib.rs".to_string()),
            branch: None,
        });

        // Valid relative path should pass validation (query may return empty).
        let result = service.impact_analysis(request).await;
        assert!(result.is_ok());
    }
}

// ── Parameter validation integration tests (handler-level) ──────────

mod param_validation {
    use tonic::Request;
    use workspace_qdrant_core::graph::create_sqlite_graph_store;

    use crate::proto::graph_service_server::GraphService;
    use crate::proto::{BetweennessRequest, CommunityRequest, PageRankRequest};
    use crate::services::GraphServiceImpl;

    async fn test_graph_service() -> (GraphServiceImpl, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let store = create_sqlite_graph_store(tmp.path()).await.unwrap();
        (GraphServiceImpl::new(store), tmp)
    }

    // ── PageRank handler validation ─────────────────────────────────

    #[tokio::test]
    async fn graph_pagerank_rejects_damping_too_low() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(PageRankRequest {
            tenant_id: "t1".to_string(),
            damping: Some(0.01),
            max_iterations: None,
            tolerance: None,
            edge_types: vec![],
            top_k: None,
        });
        let err = service.compute_page_rank(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("damping"), "{}", err.message());
    }

    #[tokio::test]
    async fn graph_pagerank_rejects_damping_too_high() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(PageRankRequest {
            tenant_id: "t1".to_string(),
            damping: Some(1.0),
            max_iterations: None,
            tolerance: None,
            edge_types: vec![],
            top_k: None,
        });
        let err = service.compute_page_rank(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("damping"), "{}", err.message());
    }

    #[tokio::test]
    async fn graph_pagerank_rejects_zero_iterations() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(PageRankRequest {
            tenant_id: "t1".to_string(),
            damping: None,
            max_iterations: Some(0),
            tolerance: None,
            edge_types: vec![],
            top_k: None,
        });
        let err = service.compute_page_rank(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(
            err.message().contains("max_iterations"),
            "{}",
            err.message()
        );
    }

    #[tokio::test]
    async fn graph_pagerank_rejects_excessive_iterations() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(PageRankRequest {
            tenant_id: "t1".to_string(),
            damping: None,
            max_iterations: Some(1001),
            tolerance: None,
            edge_types: vec![],
            top_k: None,
        });
        let err = service.compute_page_rank(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(
            err.message().contains("max_iterations"),
            "{}",
            err.message()
        );
    }

    #[tokio::test]
    async fn graph_pagerank_rejects_tolerance_too_tight() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(PageRankRequest {
            tenant_id: "t1".to_string(),
            damping: None,
            max_iterations: None,
            tolerance: Some(1e-12),
            edge_types: vec![],
            top_k: None,
        });
        let err = service.compute_page_rank(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("tolerance"), "{}", err.message());
    }

    #[tokio::test]
    async fn graph_pagerank_rejects_tolerance_too_loose() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(PageRankRequest {
            tenant_id: "t1".to_string(),
            damping: None,
            max_iterations: None,
            tolerance: Some(0.5),
            edge_types: vec![],
            top_k: None,
        });
        let err = service.compute_page_rank(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("tolerance"), "{}", err.message());
    }

    #[tokio::test]
    async fn graph_pagerank_accepts_valid_params() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(PageRankRequest {
            tenant_id: "t1".to_string(),
            damping: Some(0.85),
            max_iterations: Some(100),
            tolerance: Some(1e-6),
            edge_types: vec![],
            top_k: None,
        });
        // Empty graph returns Ok with empty entries
        let response = service.compute_page_rank(request).await.unwrap();
        assert_eq!(response.into_inner().total, 0);
    }

    #[tokio::test]
    async fn graph_pagerank_accepts_defaults() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(PageRankRequest {
            tenant_id: "t1".to_string(),
            damping: None,
            max_iterations: None,
            tolerance: None,
            edge_types: vec![],
            top_k: None,
        });
        let response = service.compute_page_rank(request).await.unwrap();
        assert_eq!(response.into_inner().total, 0);
    }

    // ── Community detection handler validation ──────────────────────

    #[tokio::test]
    async fn graph_community_rejects_zero_iterations() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(CommunityRequest {
            tenant_id: "t1".to_string(),
            max_iterations: Some(0),
            min_community_size: None,
            edge_types: vec![],
        });
        let err = service.detect_communities(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(
            err.message().contains("max_iterations"),
            "{}",
            err.message()
        );
    }

    #[tokio::test]
    async fn graph_community_rejects_excessive_iterations() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(CommunityRequest {
            tenant_id: "t1".to_string(),
            max_iterations: Some(501),
            min_community_size: None,
            edge_types: vec![],
        });
        let err = service.detect_communities(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(
            err.message().contains("max_iterations"),
            "{}",
            err.message()
        );
    }

    #[tokio::test]
    async fn graph_community_rejects_zero_min_size() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(CommunityRequest {
            tenant_id: "t1".to_string(),
            max_iterations: None,
            min_community_size: Some(0),
            edge_types: vec![],
        });
        let err = service.detect_communities(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(
            err.message().contains("min_community_size"),
            "{}",
            err.message()
        );
    }

    #[tokio::test]
    async fn graph_community_rejects_excessive_min_size() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(CommunityRequest {
            tenant_id: "t1".to_string(),
            max_iterations: None,
            min_community_size: Some(10_001),
            edge_types: vec![],
        });
        let err = service.detect_communities(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(
            err.message().contains("min_community_size"),
            "{}",
            err.message()
        );
    }

    #[tokio::test]
    async fn graph_community_accepts_valid_params() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(CommunityRequest {
            tenant_id: "t1".to_string(),
            max_iterations: Some(100),
            min_community_size: Some(3),
            edge_types: vec![],
        });
        let response = service.detect_communities(request).await.unwrap();
        assert_eq!(response.into_inner().total_communities, 0);
    }

    #[tokio::test]
    async fn graph_community_accepts_defaults() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(CommunityRequest {
            tenant_id: "t1".to_string(),
            max_iterations: None,
            min_community_size: None,
            edge_types: vec![],
        });
        let response = service.detect_communities(request).await.unwrap();
        assert_eq!(response.into_inner().total_communities, 0);
    }

    // ── Betweenness handler validation ──────────────────────────────

    #[tokio::test]
    async fn graph_betweenness_rejects_excessive_samples() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(BetweennessRequest {
            tenant_id: "t1".to_string(),
            edge_types: vec![],
            max_samples: Some(100_001),
            top_k: None,
        });
        let err = service.compute_betweenness(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("max_samples"), "{}", err.message());
    }

    #[tokio::test]
    async fn graph_betweenness_accepts_zero_samples_as_all() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(BetweennessRequest {
            tenant_id: "t1".to_string(),
            edge_types: vec![],
            max_samples: Some(0),
            top_k: None,
        });
        // zero means "use all" — should succeed on empty graph
        let response = service.compute_betweenness(request).await.unwrap();
        assert_eq!(response.into_inner().total, 0);
    }

    #[tokio::test]
    async fn graph_betweenness_accepts_valid_samples() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(BetweennessRequest {
            tenant_id: "t1".to_string(),
            edge_types: vec![],
            max_samples: Some(50),
            top_k: None,
        });
        let response = service.compute_betweenness(request).await.unwrap();
        assert_eq!(response.into_inner().total, 0);
    }

    // ── Missing tenant_id validation ────────────────────────────────

    #[tokio::test]
    async fn graph_pagerank_rejects_empty_tenant() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(PageRankRequest {
            tenant_id: String::new(),
            damping: None,
            max_iterations: None,
            tolerance: None,
            edge_types: vec![],
            top_k: None,
        });
        let err = service.compute_page_rank(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("tenant_id"), "{}", err.message());
    }

    #[tokio::test]
    async fn graph_community_rejects_empty_tenant() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(CommunityRequest {
            tenant_id: String::new(),
            max_iterations: None,
            min_community_size: None,
            edge_types: vec![],
        });
        let err = service.detect_communities(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("tenant_id"), "{}", err.message());
    }

    #[tokio::test]
    async fn graph_betweenness_rejects_empty_tenant() {
        let (service, _tmp) = test_graph_service().await;
        let request = Request::new(BetweennessRequest {
            tenant_id: String::new(),
            edge_types: vec![],
            max_samples: None,
            top_k: None,
        });
        let err = service.compute_betweenness(request).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("tenant_id"), "{}", err.message());
    }
}

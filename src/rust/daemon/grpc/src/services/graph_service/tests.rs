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

// ── NarrativeQuery handler tests ──────────────────────────────────

mod narrative_query {
    use tonic::Request;
    use workspace_qdrant_core::graph::{
        create_sqlite_graph_store, EdgeType, GraphEdge, GraphNode, GraphStore, NodeType,
    };

    use crate::proto::graph_service_server::GraphService;
    use crate::proto::narrative_query_request::QueryTarget;
    use crate::proto::NarrativeQueryRequest;
    use crate::services::GraphServiceImpl;

    async fn test_graph_service() -> (GraphServiceImpl, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let store = create_sqlite_graph_store(tmp.path()).await.unwrap();
        (GraphServiceImpl::new(store), tmp)
    }

    fn make_request(
        tenant_id: &str,
        target: Option<QueryTarget>,
        edge_types: Vec<String>,
        max_depth: i32,
        max_results: i32,
    ) -> Request<NarrativeQueryRequest> {
        Request::new(NarrativeQueryRequest {
            tenant_id: tenant_id.to_string(),
            query_target: target,
            edge_types,
            max_depth,
            max_results,
        })
    }

    // ── Validation tests ───────────────────────────────────────────

    #[tokio::test]
    async fn rejects_empty_tenant_id() {
        let (service, _tmp) = test_graph_service().await;
        let req = make_request(
            "",
            Some(QueryTarget::SymbolName("foo".into())),
            vec![],
            0,
            0,
        );
        let err = service.narrative_query(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("tenant_id"), "{}", err.message());
    }

    #[tokio::test]
    async fn rejects_missing_query_target() {
        let (service, _tmp) = test_graph_service().await;
        let req = make_request("t1", None, vec![], 0, 0);
        let err = service.narrative_query(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("query_target"), "{}", err.message());
    }

    #[tokio::test]
    async fn rejects_empty_symbol_name() {
        let (service, _tmp) = test_graph_service().await;
        let req = make_request(
            "t1",
            Some(QueryTarget::SymbolName(String::new())),
            vec![],
            0,
            0,
        );
        let err = service.narrative_query(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("symbol_name"), "{}", err.message());
    }

    #[tokio::test]
    async fn rejects_empty_concept_name() {
        let (service, _tmp) = test_graph_service().await;
        let req = make_request(
            "t1",
            Some(QueryTarget::ConceptName(String::new())),
            vec![],
            0,
            0,
        );
        let err = service.narrative_query(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("concept_name"), "{}", err.message());
    }

    #[tokio::test]
    async fn rejects_negative_max_depth() {
        let (service, _tmp) = test_graph_service().await;
        let req = make_request(
            "t1",
            Some(QueryTarget::SymbolName("foo".into())),
            vec![],
            -1,
            0,
        );
        let err = service.narrative_query(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("max_depth"), "{}", err.message());
    }

    #[tokio::test]
    async fn rejects_negative_max_results() {
        let (service, _tmp) = test_graph_service().await;
        let req = make_request(
            "t1",
            Some(QueryTarget::SymbolName("foo".into())),
            vec![],
            0,
            -1,
        );
        let err = service.narrative_query(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("max_results"), "{}", err.message());
    }

    #[tokio::test]
    async fn rejects_invalid_edge_type() {
        let (service, _tmp) = test_graph_service().await;
        let req = make_request(
            "t1",
            Some(QueryTarget::SymbolName("foo".into())),
            vec!["BOGUS".to_string()],
            0,
            0,
        );
        let err = service.narrative_query(req).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
        assert!(err.message().contains("edge type"), "{}", err.message());
    }

    // ── Defaults and clamping tests ────────────────────────────────

    #[tokio::test]
    async fn empty_graph_returns_zero_results() {
        let (service, _tmp) = test_graph_service().await;
        let req = make_request(
            "t1",
            Some(QueryTarget::SymbolName("nonexistent".into())),
            vec![],
            0,
            0,
        );
        let resp = service.narrative_query(req).await.unwrap().into_inner();
        assert_eq!(resp.total_found, 0);
        assert!(resp.nodes.is_empty());
    }

    #[tokio::test]
    async fn accepts_valid_edge_types() {
        let (service, _tmp) = test_graph_service().await;
        let req = make_request(
            "t1",
            Some(QueryTarget::SymbolName("foo".into())),
            vec!["EXPLAINS".to_string(), "COVERS_TOPIC".to_string()],
            2,
            50,
        );
        let resp = service.narrative_query(req).await.unwrap().into_inner();
        assert_eq!(resp.total_found, 0);
    }

    // ── Integration test with seeded graph data ────────────────────

    #[tokio::test]
    async fn finds_narrative_nodes_via_symbol() {
        let (service, _tmp) = test_graph_service().await;

        // Seed graph: function -> EXPLAINS -> docstring
        let guard = service.graph_store.read().await.unwrap();
        let func_node = GraphNode::new("t1", "src/lib.rs", "my_func", NodeType::Function);
        let doc_node = GraphNode::new("t1", "src/lib.rs", "my_func docs", NodeType::Docstring);
        guard
            .upsert_nodes(&[func_node.clone(), doc_node.clone()])
            .await
            .unwrap();

        let edge = GraphEdge::new(
            "t1",
            &func_node.node_id,
            &doc_node.node_id,
            EdgeType::Explains,
            "src/lib.rs",
        );
        guard.insert_edges(&[edge]).await.unwrap();
        drop(guard);

        let req = make_request(
            "t1",
            Some(QueryTarget::SymbolName("my_func".into())),
            vec![],
            2,
            50,
        );
        let resp = service.narrative_query(req).await.unwrap().into_inner();
        assert_eq!(resp.total_found, 1);
        assert_eq!(resp.nodes[0].symbol_name, "my_func docs");
        assert_eq!(resp.nodes[0].symbol_type, "docstring");
        assert_eq!(resp.nodes[0].edge_type, "EXPLAINS");
        assert_eq!(resp.nodes[0].depth, 1);
    }

    #[tokio::test]
    async fn finds_narrative_nodes_via_concept_incoming_edges() {
        let (service, _tmp) = test_graph_service().await;

        // Real data direction: document_section --COVERS_TOPIC--> concept_node
        // Query by concept should find doc via incoming edge traversal.
        let concept = GraphNode::new("", "global", "error_handling", NodeType::ConceptNode);
        let doc = GraphNode::new(
            "t1",
            "docs/errors.md",
            "Error Handling Guide",
            NodeType::DocumentSection,
        );
        let guard = service.graph_store.read().await.unwrap();
        guard
            .upsert_nodes(&[concept.clone(), doc.clone()])
            .await
            .unwrap();

        let edge = GraphEdge::new(
            "t1",
            &doc.node_id,
            &concept.node_id,
            EdgeType::CoversTopic,
            "docs/errors.md",
        );
        guard.insert_edges(&[edge]).await.unwrap();
        drop(guard);

        let req = make_request(
            "t1",
            Some(QueryTarget::ConceptName("error_handling".into())),
            vec![],
            2,
            50,
        );
        let resp = service.narrative_query(req).await.unwrap().into_inner();
        assert_eq!(resp.total_found, 1);
        assert_eq!(resp.nodes[0].symbol_name, "Error Handling Guide");
        assert_eq!(resp.nodes[0].symbol_type, "document_section");
        assert_eq!(resp.nodes[0].edge_type, "COVERS_TOPIC");
    }

    #[tokio::test]
    async fn finds_narrative_nodes_via_concept_outgoing_edges() {
        let (service, _tmp) = test_graph_service().await;

        // Outgoing direction: concept_node --COVERS_TOPIC--> document_section
        let concept = GraphNode::new("", "global", "error_handling", NodeType::ConceptNode);
        let doc = GraphNode::new(
            "t1",
            "docs/errors.md",
            "Error Handling Guide",
            NodeType::DocumentSection,
        );
        let guard = service.graph_store.read().await.unwrap();
        guard
            .upsert_nodes(&[concept.clone(), doc.clone()])
            .await
            .unwrap();

        let edge = GraphEdge::new(
            "t1",
            &concept.node_id,
            &doc.node_id,
            EdgeType::CoversTopic,
            "docs/errors.md",
        );
        guard.insert_edges(&[edge]).await.unwrap();
        drop(guard);

        let req = make_request(
            "t1",
            Some(QueryTarget::ConceptName("error_handling".into())),
            vec![],
            2,
            50,
        );
        let resp = service.narrative_query(req).await.unwrap().into_inner();
        assert_eq!(resp.total_found, 1);
        assert_eq!(resp.nodes[0].symbol_name, "Error Handling Guide");
        assert_eq!(resp.nodes[0].symbol_type, "document_section");
        assert_eq!(resp.nodes[0].edge_type, "COVERS_TOPIC");
    }

    #[tokio::test]
    async fn filters_by_edge_type() {
        let (service, _tmp) = test_graph_service().await;

        // Seed: function -> EXPLAINS -> docstring, function -> DESCRIBES -> code_comment
        let func_node = GraphNode::new("t1", "src/lib.rs", "filter_fn", NodeType::Function);
        let doc = GraphNode::new("t1", "src/lib.rs", "filter_fn doc", NodeType::Docstring);
        let comment = GraphNode::new(
            "t1",
            "src/lib.rs",
            "filter_fn comment",
            NodeType::CodeComment,
        );

        let guard = service.graph_store.read().await.unwrap();
        guard
            .upsert_nodes(&[func_node.clone(), doc.clone(), comment.clone()])
            .await
            .unwrap();

        let edge1 = GraphEdge::new(
            "t1",
            &func_node.node_id,
            &doc.node_id,
            EdgeType::Explains,
            "src/lib.rs",
        );
        let edge2 = GraphEdge::new(
            "t1",
            &func_node.node_id,
            &comment.node_id,
            EdgeType::Describes,
            "src/lib.rs",
        );
        guard.insert_edges(&[edge1, edge2]).await.unwrap();
        drop(guard);

        // Filter to EXPLAINS only — should get docstring, not code_comment
        let req = make_request(
            "t1",
            Some(QueryTarget::SymbolName("filter_fn".into())),
            vec!["EXPLAINS".to_string()],
            2,
            50,
        );
        let resp = service.narrative_query(req).await.unwrap().into_inner();
        assert_eq!(resp.total_found, 1);
        assert_eq!(resp.nodes[0].symbol_name, "filter_fn doc");
        assert_eq!(resp.nodes[0].edge_type, "EXPLAINS");
    }

    #[tokio::test]
    async fn respects_max_results_limit() {
        let (service, _tmp) = test_graph_service().await;

        // Seed: function connected to 3 docstrings
        let func_node = GraphNode::new("t1", "src/lib.rs", "limited_fn", NodeType::Function);
        let doc1 = GraphNode::new("t1", "src/lib.rs", "doc_a", NodeType::Docstring);
        let doc2 = GraphNode::new("t1", "src/lib.rs", "doc_b", NodeType::Docstring);
        let doc3 = GraphNode::new("t1", "src/lib.rs", "doc_c", NodeType::Docstring);

        let guard = service.graph_store.read().await.unwrap();
        guard
            .upsert_nodes(&[func_node.clone(), doc1.clone(), doc2.clone(), doc3.clone()])
            .await
            .unwrap();

        for doc in [&doc1, &doc2, &doc3] {
            let edge = GraphEdge::new(
                "t1",
                &func_node.node_id,
                &doc.node_id,
                EdgeType::Explains,
                "src/lib.rs",
            );
            guard.insert_edges(&[edge]).await.unwrap();
        }
        drop(guard);

        let req = make_request(
            "t1",
            Some(QueryTarget::SymbolName("limited_fn".into())),
            vec![],
            2,
            2, // Limit to 2
        );
        let resp = service.narrative_query(req).await.unwrap().into_inner();
        assert_eq!(resp.total_found, 2);
    }

    #[tokio::test]
    async fn excludes_non_narrative_nodes() {
        let (service, _tmp) = test_graph_service().await;

        // Seed: function -> CALLS -> another_function (non-narrative)
        //        function -> EXPLAINS -> docstring (narrative)
        let func1 = GraphNode::new("t1", "src/lib.rs", "caller", NodeType::Function);
        let func2 = GraphNode::new("t1", "src/lib.rs", "callee", NodeType::Function);
        let doc = GraphNode::new("t1", "src/lib.rs", "caller docs", NodeType::Docstring);

        let guard = service.graph_store.read().await.unwrap();
        guard
            .upsert_nodes(&[func1.clone(), func2.clone(), doc.clone()])
            .await
            .unwrap();

        let call_edge = GraphEdge::new(
            "t1",
            &func1.node_id,
            &func2.node_id,
            EdgeType::Calls,
            "src/lib.rs",
        );
        let explains_edge = GraphEdge::new(
            "t1",
            &func1.node_id,
            &doc.node_id,
            EdgeType::Explains,
            "src/lib.rs",
        );
        guard
            .insert_edges(&[call_edge, explains_edge])
            .await
            .unwrap();
        drop(guard);

        // No edge_types filter — traverse all edges but only return narrative nodes
        let req = make_request(
            "t1",
            Some(QueryTarget::SymbolName("caller".into())),
            vec![],
            2,
            50,
        );
        let resp = service.narrative_query(req).await.unwrap().into_inner();
        // Should only return the docstring, not the callee function
        assert_eq!(resp.total_found, 1);
        assert_eq!(resp.nodes[0].symbol_name, "caller docs");
        assert_eq!(resp.nodes[0].symbol_type, "docstring");
    }

    #[tokio::test]
    async fn multi_hop_traversal() {
        let (service, _tmp) = test_graph_service().await;

        // Seed: function -> CALLS -> another_function -> EXPLAINS -> docstring
        // With max_depth=2, should find the docstring at depth 2
        let func1 = GraphNode::new("t1", "src/lib.rs", "hop_start", NodeType::Function);
        let func2 = GraphNode::new("t1", "src/lib.rs", "hop_middle", NodeType::Function);
        let doc = GraphNode::new("t1", "src/lib.rs", "hop_end_doc", NodeType::Docstring);

        let guard = service.graph_store.read().await.unwrap();
        guard
            .upsert_nodes(&[func1.clone(), func2.clone(), doc.clone()])
            .await
            .unwrap();

        let edge1 = GraphEdge::new(
            "t1",
            &func1.node_id,
            &func2.node_id,
            EdgeType::Calls,
            "src/lib.rs",
        );
        let edge2 = GraphEdge::new(
            "t1",
            &func2.node_id,
            &doc.node_id,
            EdgeType::Explains,
            "src/lib.rs",
        );
        guard.insert_edges(&[edge1, edge2]).await.unwrap();
        drop(guard);

        // max_depth=1: should NOT find docstring (it's 2 hops away)
        let req = make_request(
            "t1",
            Some(QueryTarget::SymbolName("hop_start".into())),
            vec![],
            1,
            50,
        );
        let resp = service.narrative_query(req).await.unwrap().into_inner();
        assert_eq!(resp.total_found, 0);

        // max_depth=2: should find docstring at depth 2
        let req = make_request(
            "t1",
            Some(QueryTarget::SymbolName("hop_start".into())),
            vec![],
            2,
            50,
        );
        let resp = service.narrative_query(req).await.unwrap().into_inner();
        assert_eq!(resp.total_found, 1);
        assert_eq!(resp.nodes[0].symbol_name, "hop_end_doc");
        assert_eq!(resp.nodes[0].depth, 2);
    }
}

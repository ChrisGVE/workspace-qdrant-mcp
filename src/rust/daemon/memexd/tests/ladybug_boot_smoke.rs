//! Boot smoke test for the LadybugDB default backend (A0.7).
//!
//! Two checks pin the default-backend flip end to end:
//!   1. The embedded `default_configuration.yaml` selects the `ladybug` backend,
//!      so a daemon started with no user config boots ladybug-backed. This runs
//!      regardless of feature flags — it is a pure config assertion.
//!   2. With the `ladybug` feature compiled in (the daemon's default), the
//!      backend the daemon would instantiate performs a basic graph operation
//!      (upsert + stats + traversal) successfully. This is the "daemon executes
//!      a basic graph op" smoke check, run against the same factory path the
//!      daemon uses, without spawning a full process or requiring Qdrant.

/// The shipped default configuration must select the LadybugDB backend (A0.7).
/// Sourced from the single source of truth (`assets/default_configuration.yaml`,
/// embedded via `wqm_common::yaml_defaults`).
#[test]
fn default_config_selects_ladybug_backend() {
    let backend = wqm_common::yaml_defaults::DEFAULT_YAML_CONFIG
        .graph
        .backend
        .trim()
        .to_ascii_lowercase();
    assert_eq!(
        backend, "ladybug",
        "default_configuration.yaml must default graph.backend to ladybug (A0.7)"
    );
}

#[cfg(feature = "ladybug")]
mod ladybug_op {
    use serial_test::serial;
    use workspace_qdrant_core::config::GraphRagConfig;
    use workspace_qdrant_core::graph::{
        factory, EdgeType, GraphBackend, GraphConfig, GraphEdge, GraphNode, NodeType,
    };

    /// With the daemon's default `ladybug` feature, the backend the daemon would
    /// boot performs a basic graph operation: ingest two nodes + an edge, then
    /// traverse and report stats. Mirrors the daemon's `init_graph_db` factory
    /// path (`create_ladybug_graph_store`) without a live process or Qdrant.
    #[tokio::test]
    #[serial]
    async fn ladybug_default_backend_executes_basic_graph_op() {
        const TENANT: &str = "boot-smoke";
        let dir = tempfile::tempdir().expect("temp dir");

        // GraphConfig::default() carries ladybug=on tuning knobs; use a small
        // buffer pool so the kuzu Database's virtual reservation stays modest.
        let config = GraphConfig {
            backend: GraphBackend::Ladybug,
            db_dir: None,
            buffer_pool_size: 64 * 1024 * 1024,
            max_threads: 2,
        };
        let store =
            factory::create_ladybug_graph_store(dir.path(), &config, &GraphRagConfig::default())
                .await
                .expect("daemon factory must create the ladybug store");

        let caller = GraphNode::new(TENANT, "src/lib.rs", "caller", NodeType::Function);
        let callee = GraphNode::new(TENANT, "src/lib.rs", "callee", NodeType::Function);
        let edge = GraphEdge::new(
            TENANT,
            &caller.node_id,
            &callee.node_id,
            EdgeType::Calls,
            "src/lib.rs",
        );
        let caller_id = caller.node_id.clone();
        let callee_id = callee.node_id.clone();

        store
            .upsert_nodes(&[caller, callee])
            .await
            .expect("upsert nodes");
        store.insert_edges(&[edge]).await.expect("insert edge");

        // Basic graph operation: a 1-hop CALLS traversal must reach the callee.
        let reached = store
            .query_related(TENANT, &caller_id, 1, Some(&[EdgeType::Calls]), None)
            .await
            .expect("query_related on the ladybug backend");
        assert!(
            reached.iter().any(|n| n.node_id == callee_id),
            "ladybug backend must traverse caller -> callee"
        );

        let stats = store.stats(Some(TENANT), None).await.expect("stats");
        assert_eq!(
            stats.total_nodes, 2,
            "ladybug backend must report both nodes"
        );
        assert_eq!(stats.total_edges, 1, "ladybug backend must report the edge");
    }
}

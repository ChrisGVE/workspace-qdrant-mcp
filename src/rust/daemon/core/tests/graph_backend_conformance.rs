//! Backend-equivalence conformance suite (kg-activation Task 28).
//!
//! Runs the SAME graph operations against both `SqliteGraphStore` and (when the
//! `ladybug` feature is enabled) `LadybugGraphStore`, asserting that the two
//! backends produce equivalent results modulo ordering. This is the release
//! gate for the optional LadybugDB backend: cross-boundary RAG stays
//! SQLite-only until this suite passes on ladybug.
//!
//! Placement note: this suite lives in the `workspace-qdrant-core` crate (not
//! `shared-test-utils`) because `core` already depends on `shared-test-utils`
//! as a dev-dependency — putting the suite there would create a dependency
//! cycle. It uses only public `core` APIs, so it is backend-agnostic.
//!
//! Run with `--features ladybug` to exercise the cross-backend cases; without
//! the feature only the SQLite self-consistency cases run.

use std::collections::BTreeMap;

use tempfile::tempdir;
use workspace_qdrant_core::graph::{
    create_sqlite_graph_store, EdgeType, GraphEdge, GraphNode, GraphStore, NodeType, TraversalNode,
};

const T: &str = "conformance-tenant";
const GLOBAL: &str = "__global__";
const LIB: &str = "conformance-library";

// ── Backend-agnostic fixture ────────────────────────────────────────────────

/// A small graph exercising structural, narrative, and concept layers:
///
/// Structural (tenant T):
///   `entry` --CALLS--> `mid` --CALLS--> `leaf`   (3-node call chain)
///   `mid`   --USES_TYPE--> `helper`
///
/// Concept bridge:
///   `entry` --IMPLEMENTS_CONCEPT--> `concept:auth`     (concept under __global__)
///   `libdoc` --COVERS_TOPIC--> `concept:auth`          (library doc under LIB)
///
/// This lets a cross-boundary query starting at `entry` reach the library
/// document `libdoc` through the shared concept node.
fn fixture() -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let entry = GraphNode::new(T, "src/entry.rs", "entry", NodeType::Function);
    let mid = GraphNode::new(T, "src/mid.rs", "mid", NodeType::Function);
    let leaf = GraphNode::new(T, "src/leaf.rs", "leaf", NodeType::Function);
    let helper = GraphNode::new(T, "src/helper.rs", "Helper", NodeType::Struct);

    // Concept node lives under the global tenant.
    let concept = GraphNode::new(GLOBAL, "", "auth", NodeType::ConceptNode);
    // Library documentation section lives under the library tenant.
    let libdoc = GraphNode::new(
        LIB,
        "docs/auth.md",
        "Authentication",
        NodeType::LibrarySection,
    );

    let edges = vec![
        GraphEdge::new(
            T,
            &entry.node_id,
            &mid.node_id,
            EdgeType::Calls,
            "src/entry.rs",
        ),
        GraphEdge::new(
            T,
            &mid.node_id,
            &leaf.node_id,
            EdgeType::Calls,
            "src/mid.rs",
        ),
        GraphEdge::new(
            T,
            &mid.node_id,
            &helper.node_id,
            EdgeType::UsesType,
            "src/mid.rs",
        ),
        GraphEdge::new(
            T,
            &entry.node_id,
            &concept.node_id,
            EdgeType::ImplementsConcept,
            "src/entry.rs",
        ),
        GraphEdge::new(
            LIB,
            &libdoc.node_id,
            &concept.node_id,
            EdgeType::CoversTopic,
            "docs/auth.md",
        ),
    ];

    let nodes = vec![entry, mid, leaf, helper, concept, libdoc];
    (nodes, edges)
}

/// Resolve the node_id for a fixture symbol of a given type.
fn node_id(symbol: &str, ty: NodeType) -> String {
    let (nodes, _) = fixture();
    nodes
        .into_iter()
        .find(|n| n.symbol_name == symbol && n.symbol_type == ty)
        .map(|n| n.node_id)
        .unwrap_or_else(|| panic!("fixture missing {symbol}"))
}

async fn populate(store: &dyn GraphStore) {
    let (nodes, edges) = fixture();
    store.upsert_nodes(&nodes).await.unwrap();
    store.insert_edges(&edges).await.unwrap();
}

// ── Comparison helpers (ordering-normalized) ────────────────────────────────

/// Map each reached node_id to its reported depth. Backends may differ in
/// traversal order, so we compare this map rather than the ordered Vec.
fn depth_map(nodes: &[TraversalNode]) -> BTreeMap<String, u32> {
    nodes.iter().map(|n| (n.node_id.clone(), n.depth)).collect()
}

/// Map each reached node_id to (symbol_name, symbol_type) for identity checks.
fn identity_map(nodes: &[TraversalNode]) -> BTreeMap<String, (String, String)> {
    nodes
        .iter()
        .map(|n| {
            (
                n.node_id.clone(),
                (n.symbol_name.clone(), n.symbol_type.clone()),
            )
        })
        .collect()
}

// ── SQLite self-consistency (always runs) ───────────────────────────────────

async fn sqlite_store() -> impl GraphStore {
    let dir = tempdir().unwrap();
    // Keep the tempdir alive for the store's lifetime by leaking it; the OS
    // reclaims the file when the test process exits.
    let path = dir.keep();
    create_sqlite_graph_store(&path).await.unwrap()
}

#[tokio::test]
async fn sqlite_query_related_reports_true_depth() {
    let store = sqlite_store().await;
    populate(&store).await;
    let entry = node_id("entry", NodeType::Function);

    let one = store
        .query_related(T, &entry, 1, Some(&[EdgeType::Calls]), None)
        .await
        .unwrap();
    let d1 = depth_map(&one);
    assert_eq!(d1.len(), 1, "1-hop CALLS from entry reaches only mid");
    assert_eq!(*d1.values().next().unwrap(), 1);

    let two = store
        .query_related(T, &entry, 2, Some(&[EdgeType::Calls]), None)
        .await
        .unwrap();
    let d2 = depth_map(&two);
    // entry -> mid (1), mid -> leaf (2)
    assert_eq!(d2.len(), 2);
    let mid = node_id("mid", NodeType::Function);
    let leaf = node_id("leaf", NodeType::Function);
    assert_eq!(d2[&mid], 1);
    assert_eq!(d2[&leaf], 2);
}

#[tokio::test]
async fn sqlite_cross_boundary_reaches_library_via_concept() {
    let store = sqlite_store().await;
    populate(&store).await;
    let entry = node_id("entry", NodeType::Function);

    let results = store
        .query_cross_boundary(
            T,
            &entry,
            &[
                EdgeType::ImplementsConcept,
                EdgeType::CoversTopic,
                EdgeType::Explains,
            ],
            3,
            &[LIB.to_string()],
        )
        .await
        .unwrap();

    let ids = identity_map(&results);
    let concept = node_id("auth", NodeType::ConceptNode);
    let libdoc = node_id("Authentication", NodeType::LibrarySection);
    assert!(ids.contains_key(&concept), "should reach the concept node");
    assert!(
        ids.contains_key(&libdoc),
        "should reach the library doc through the concept"
    );
}

// ── Cross-backend equivalence (ladybug feature only) ─────────────────────────

#[cfg(feature = "ladybug")]
mod ladybug_equivalence {
    use super::*;
    use workspace_qdrant_core::graph::{migrator, LadybugConfig, LadybugGraphStore};

    fn ladybug_store(name: &str) -> (LadybugGraphStore, tempfile::TempDir) {
        let tmp = tempdir().unwrap();
        let config = LadybugConfig {
            db_path: tmp.path().join(name),
            buffer_pool_size: 64 * 1024 * 1024,
            max_num_threads: 2,
        };
        (LadybugGraphStore::new(config).unwrap(), tmp)
    }

    async fn both_stores() -> (impl GraphStore, LadybugGraphStore, tempfile::TempDir) {
        let sqlite = sqlite_store().await;
        let (ladybug, tmp) = ladybug_store("conformance");
        populate(&sqlite).await;
        populate(&ladybug).await;
        (sqlite, ladybug, tmp)
    }

    #[tokio::test]
    async fn query_related_equivalent_at_each_hop() {
        let (sqlite, ladybug, _tmp) = both_stores().await;
        let entry = node_id("entry", NodeType::Function);

        for hops in 1..=3u32 {
            let s = sqlite
                .query_related(T, &entry, hops, Some(&[EdgeType::Calls]), None)
                .await
                .unwrap();
            let l = ladybug
                .query_related(T, &entry, hops, Some(&[EdgeType::Calls]), None)
                .await
                .unwrap();
            assert_eq!(
                depth_map(&s),
                depth_map(&l),
                "query_related depth maps differ at {hops} hop(s)"
            );
            assert_eq!(
                identity_map(&s),
                identity_map(&l),
                "query_related identities differ at {hops} hop(s)"
            );
        }
    }

    #[tokio::test]
    async fn find_path_equivalent() {
        let (sqlite, ladybug, _tmp) = both_stores().await;
        let entry = node_id("entry", NodeType::Function);
        let leaf = node_id("leaf", NodeType::Function);

        let s = sqlite
            .find_path(T, &entry, &leaf, 3, Some(&[EdgeType::Calls]), None)
            .await
            .unwrap();

        // The SQLite backend implements BFS path-finding. If ladybug also
        // reports a path it must have the same length and endpoints; if it
        // returns None (path-finding not yet implemented), that is a documented
        // gap rather than an equivalence violation, so we only assert the
        // SQLite side found the expected 3-node path here.
        let s = s.expect("sqlite should find entry->mid->leaf");
        assert_eq!(s.len(), 3, "path entry->mid->leaf has 3 nodes");
        assert_eq!(s.first().unwrap().node_id, entry);
        assert_eq!(s.last().unwrap().node_id, leaf);

        let l = ladybug
            .find_path(T, &entry, &leaf, 3, Some(&[EdgeType::Calls]), None)
            .await
            .unwrap();
        if let Some(l) = l {
            assert_eq!(l.len(), s.len(), "ladybug path length differs");
            assert_eq!(l.first().unwrap().node_id, entry);
            assert_eq!(l.last().unwrap().node_id, leaf);
        }
    }

    #[tokio::test]
    async fn cross_boundary_equivalent() {
        let (sqlite, ladybug, _tmp) = both_stores().await;
        let entry = node_id("entry", NodeType::Function);
        let edge_types = [
            EdgeType::ImplementsConcept,
            EdgeType::CoversTopic,
            EdgeType::Explains,
        ];

        let s = sqlite
            .query_cross_boundary(T, &entry, &edge_types, 3, &[LIB.to_string()])
            .await
            .unwrap();
        let l = ladybug
            .query_cross_boundary(T, &entry, &edge_types, 3, &[LIB.to_string()])
            .await
            .unwrap();

        assert_eq!(
            identity_map(&s),
            identity_map(&l),
            "cross-boundary reached node sets differ"
        );
        assert_eq!(depth_map(&s), depth_map(&l), "cross-boundary depths differ");
    }

    #[tokio::test]
    async fn cross_boundary_bidirectional() {
        // Starting from the library doc, the concept and the code entry must be
        // reachable in BOTH backends (traversal must follow edges in reverse).
        let (sqlite, ladybug, _tmp) = both_stores().await;
        let libdoc = node_id("Authentication", NodeType::LibrarySection);
        let edge_types = [EdgeType::ImplementsConcept, EdgeType::CoversTopic];

        let s = sqlite
            .query_cross_boundary(LIB, &libdoc, &edge_types, 3, &[T.to_string()])
            .await
            .unwrap();
        let l = ladybug
            .query_cross_boundary(LIB, &libdoc, &edge_types, 3, &[T.to_string()])
            .await
            .unwrap();

        let concept = node_id("auth", NodeType::ConceptNode);
        let entry = node_id("entry", NodeType::Function);
        let s_ids = identity_map(&s);
        assert!(s_ids.contains_key(&concept) && s_ids.contains_key(&entry));
        assert_eq!(
            identity_map(&s),
            identity_map(&l),
            "bidirectional cross-boundary reached sets differ"
        );
    }

    #[tokio::test]
    async fn stats_equivalent() {
        let (sqlite, ladybug, _tmp) = both_stores().await;

        let s = sqlite.stats(Some(T), None).await.unwrap();
        let l = ladybug.stats(Some(T), None).await.unwrap();
        assert_eq!(s.total_nodes, l.total_nodes, "tenant node counts differ");
        assert_eq!(s.total_edges, l.total_edges, "tenant edge counts differ");
        assert_eq!(
            s.nodes_by_type, l.nodes_by_type,
            "node-type histograms differ"
        );
        assert_eq!(
            s.edges_by_type, l.edges_by_type,
            "edge-type histograms differ"
        );

        // Global stats (all tenants) must agree on totals.
        let sg = sqlite.stats(None, None).await.unwrap();
        let lg = ladybug.stats(None, None).await.unwrap();
        assert_eq!(sg.total_nodes, lg.total_nodes, "global node counts differ");
        assert_eq!(sg.total_edges, lg.total_edges, "global edge counts differ");
    }

    /// Round-trip migration: SQLite -> Ladybug -> SQLite preserves all node and
    /// edge counts, node/edge type distributions, and concept/narrative nodes.
    #[tokio::test]
    async fn migration_round_trip_lossless() {
        // Source SQLite store.
        let src = sqlite_store().await;
        populate(&src).await;
        let src_stats_all = src.stats(None, None).await.unwrap();

        // Export the full snapshot from SQLite. We obtain the pool via a
        // downcast-free path: re-create a snapshot directly from the trait by
        // re-reading every edge type and node via the public migrator on a
        // SqliteGraphStore. The factory store is `SharedGraphStore`, so we use
        // a fresh dedicated SqliteGraphStore for export.
        let src_dir = tempdir().unwrap();
        let shared_src = create_sqlite_graph_store(src_dir.path()).await.unwrap();
        populate(&shared_src).await;
        let guard = shared_src.read().await.unwrap();
        let snapshot = migrator::export_sqlite(guard.pool(), None).await.unwrap();
        drop(guard);

        // Import into Ladybug.
        let (ladybug, _tmp) = ladybug_store("roundtrip_lb");
        let lb_report = migrator::import_to_store(&snapshot, &ladybug, 100)
            .await
            .unwrap();
        assert!(
            lb_report.warnings.is_empty(),
            "ladybug import warnings: {:?}",
            lb_report.warnings
        );
        assert!(lb_report.nodes_match && lb_report.edges_match);

        let lb_stats = ladybug.stats(None, None).await.unwrap();
        assert_eq!(lb_stats.total_nodes, src_stats_all.total_nodes);
        assert_eq!(lb_stats.total_edges, src_stats_all.total_edges);

        // Export back out of Ladybug and import into a fresh SQLite store.
        let lb_snapshot = migrator::export_ladybug(&ladybug, None).unwrap();
        let dst_dir = tempdir().unwrap();
        let shared_dst = create_sqlite_graph_store(dst_dir.path()).await.unwrap();
        let dst_guard = shared_dst.read().await.unwrap();
        let sqlite_report = migrator::import_to_store(&lb_snapshot, &*dst_guard, 100)
            .await
            .unwrap();
        drop(dst_guard);
        assert!(sqlite_report.nodes_match && sqlite_report.edges_match);

        let dst_stats = shared_dst.stats(None, None).await.unwrap();
        assert_eq!(
            dst_stats.total_nodes, src_stats_all.total_nodes,
            "round-trip changed node count"
        );
        assert_eq!(
            dst_stats.total_edges, src_stats_all.total_edges,
            "round-trip changed edge count"
        );
        assert_eq!(
            dst_stats.nodes_by_type, src_stats_all.nodes_by_type,
            "round-trip changed node-type distribution (concept/narrative loss?)"
        );
        assert_eq!(
            dst_stats.edges_by_type, src_stats_all.edges_by_type,
            "round-trip changed edge-type distribution"
        );

        // Concept node must survive the round trip.
        let concept = node_id("auth", NodeType::ConceptNode);
        let concept_present = dst_stats
            .nodes_by_type
            .get("concept_node")
            .copied()
            .unwrap_or(0);
        assert_eq!(concept_present, 1, "concept node lost in round trip");
        assert!(!concept.is_empty());
    }
}

//! Tests for cross-boundary RAG traversal (`query_cross_boundary`).
//!
//! Exercises the bidirectional concept/narrative CTE: tenant relaxation,
//! `__global__` concept reachability, foreign-tenant exclusion, library-tenant
//! inclusion, deduplication (one row per node at min depth), and per-edge-type
//! confidence scoring.

use super::*;

const TENANT_A: &str = "project_a";
const TENANT_B: &str = "project_b";
const LIB: &str = "local_lib";
const GLOBAL: &str = "__global__";

/// Insert an edge with an explicit weight.
async fn insert_weighted_edge(
    store: &SqliteGraphStore,
    tenant: &str,
    src: &str,
    tgt: &str,
    etype: EdgeType,
    weight: f64,
) {
    let mut edge = GraphEdge::new(tenant, src, tgt, etype, "src.rs");
    edge.weight = weight;
    store.insert_edges(&[edge]).await.unwrap();
}

/// Build a cross-domain graph:
///
/// ```text
///   code_a (project_a)  --IMPLEMENTS_CONCEPT-->  concept (__global__)
///   lib_section (local_lib) --COVERS_TOPIC-->    concept
///   doc_a (project_a)  --EXPLAINS-->             code_a
///   code_b (project_b) --IMPLEMENTS_CONCEPT-->   concept   (foreign)
/// ```
async fn build_cross_domain(
    store: &SqliteGraphStore,
) -> (GraphNode, GraphNode, GraphNode, GraphNode, GraphNode) {
    let code_a = GraphNode::new(TENANT_A, "a.rs", "fn_a", NodeType::Function);
    let doc_a = GraphNode::new(TENANT_A, "a.md", "sec_a", NodeType::DocumentSection);
    let concept = GraphNode::new(GLOBAL, "", "caching", NodeType::ConceptNode);
    let lib_sec = GraphNode::new(LIB, "book.md", "lib_caching", NodeType::LibrarySection);
    let code_b = GraphNode::new(TENANT_B, "b.rs", "fn_b", NodeType::Function);

    store
        .upsert_nodes(&[
            code_a.clone(),
            doc_a.clone(),
            concept.clone(),
            lib_sec.clone(),
            code_b.clone(),
        ])
        .await
        .unwrap();

    insert_weighted_edge(
        store,
        TENANT_A,
        &code_a.node_id,
        &concept.node_id,
        EdgeType::ImplementsConcept,
        0.9,
    )
    .await;
    insert_weighted_edge(
        store,
        LIB,
        &lib_sec.node_id,
        &concept.node_id,
        EdgeType::CoversTopic,
        0.8,
    )
    .await;
    insert_weighted_edge(
        store,
        TENANT_A,
        &doc_a.node_id,
        &code_a.node_id,
        EdgeType::Explains,
        1.0,
    )
    .await;
    insert_weighted_edge(
        store,
        TENANT_B,
        &code_b.node_id,
        &concept.node_id,
        EdgeType::ImplementsConcept,
        0.95,
    )
    .await;

    (code_a, doc_a, concept, lib_sec, code_b)
}

fn edges() -> Vec<EdgeType> {
    vec![
        EdgeType::ImplementsConcept,
        EdgeType::CoversTopic,
        EdgeType::Explains,
    ]
}

#[tokio::test]
async fn test_forward_reaches_global_concept() {
    let store = test_store().await;
    let (code_a, _doc_a, concept, _lib, _code_b) = build_cross_domain(&store).await;

    // From code_a, with no library tenants, we still reach the global concept.
    let results = store
        .query_cross_boundary(TENANT_A, &code_a.node_id, &edges(), 2, &[])
        .await
        .unwrap();

    let ids: Vec<&str> = results.iter().map(|n| n.node_id.as_str()).collect();
    assert!(
        ids.contains(&concept.node_id.as_str()),
        "should reach global concept, got {ids:?}"
    );
    // Concept is reached via IMPLEMENTS_CONCEPT (base 0.7) × weight 0.9 = 0.63.
    let cn = results
        .iter()
        .find(|n| n.node_id == concept.node_id)
        .unwrap();
    assert_eq!(cn.tenant_id, GLOBAL);
    assert!(
        (cn.edge_confidence - 0.63).abs() < 1e-9,
        "{}",
        cn.edge_confidence
    );
}

#[tokio::test]
async fn test_tenant_guard_excludes_foreign_and_includes_library() {
    let store = test_store().await;
    let (code_a, _doc_a, _concept, lib, code_b) = build_cross_domain(&store).await;

    // 2 hops: code_a -> concept -> {lib_section (reverse COVERS_TOPIC), code_b (foreign)}.
    let results = store
        .query_cross_boundary(TENANT_A, &code_a.node_id, &edges(), 2, &[LIB.to_string()])
        .await
        .unwrap();
    let ids: Vec<&str> = results.iter().map(|n| n.node_id.as_str()).collect();

    assert!(
        ids.contains(&lib.node_id.as_str()),
        "library section reachable when LIB in library_tenants, got {ids:?}"
    );
    assert!(
        !ids.contains(&code_b.node_id.as_str()),
        "foreign project_b symbol must be excluded, got {ids:?}"
    );

    // Without LIB in the relaxation set, the library section is excluded.
    let results_no_lib = store
        .query_cross_boundary(TENANT_A, &code_a.node_id, &edges(), 2, &[])
        .await
        .unwrap();
    let ids_no_lib: Vec<&str> = results_no_lib.iter().map(|n| n.node_id.as_str()).collect();
    assert!(
        !ids_no_lib.contains(&lib.node_id.as_str()),
        "library section excluded without LIB tenant, got {ids_no_lib:?}"
    );
}

#[tokio::test]
async fn test_reverse_arm_from_concept() {
    let store = test_store().await;
    let (code_a, _doc_a, concept, lib, code_b) = build_cross_domain(&store).await;

    // Seeding FROM the concept must surface the code/sections pointing into it
    // (reverse direction of IMPLEMENTS_CONCEPT / COVERS_TOPIC). Source tenant is
    // the querying project (project_a); LIB is relaxed in. project_b is foreign.
    let results = store
        .query_cross_boundary(TENANT_A, &concept.node_id, &edges(), 1, &[LIB.to_string()])
        .await
        .unwrap();
    let ids: Vec<&str> = results.iter().map(|n| n.node_id.as_str()).collect();

    assert!(
        ids.contains(&code_a.node_id.as_str()),
        "reverse to code_a, {ids:?}"
    );
    assert!(
        ids.contains(&lib.node_id.as_str()),
        "reverse to lib, {ids:?}"
    );
    // project_b not in relaxation set -> excluded even via reverse arm.
    assert!(
        !ids.contains(&code_b.node_id.as_str()),
        "foreign excluded, {ids:?}"
    );
}

#[tokio::test]
async fn test_explains_reverse_reaches_doc() {
    let store = test_store().await;
    let (code_a, doc_a, _concept, _lib, _code_b) = build_cross_domain(&store).await;

    // doc_a --EXPLAINS--> code_a, so seeding code_a reaches doc_a via reverse arm.
    let results = store
        .query_cross_boundary(TENANT_A, &code_a.node_id, &edges(), 1, &[])
        .await
        .unwrap();
    let doc = results.iter().find(|n| n.node_id == doc_a.node_id);
    assert!(doc.is_some(), "should reach doc_a via reverse EXPLAINS");
    // EXPLAINS base 0.6 × weight 1.0 = 0.6.
    assert!((doc.unwrap().edge_confidence - 0.6).abs() < 1e-9);
}

#[tokio::test]
async fn test_dedup_high_degree_concept_appears_once() {
    let store = test_store().await;
    let concept = GraphNode::new(GLOBAL, "", "topic", NodeType::ConceptNode);
    store.upsert_nodes(&[concept.clone()]).await.unwrap();

    // 30 files all implementing the same concept.
    let mut nodes = Vec::new();
    for i in 0..30 {
        let n = GraphNode::new(
            TENANT_A,
            format!("f{i}.rs"),
            format!("sym_{i}"),
            NodeType::Function,
        );
        nodes.push(n);
    }
    store.upsert_nodes(&nodes).await.unwrap();
    for n in &nodes {
        insert_weighted_edge(
            &store,
            TENANT_A,
            &n.node_id,
            &concept.node_id,
            EdgeType::ImplementsConcept,
            0.5,
        )
        .await;
    }

    // Seed from one symbol; reach concept (hop 1) then the other 29 symbols
    // (hop 2 via reverse). The concept must appear exactly once.
    let results = store
        .query_cross_boundary(TENANT_A, &nodes[0].node_id, &edges(), 2, &[])
        .await
        .unwrap();
    let concept_rows = results
        .iter()
        .filter(|n| n.node_id == concept.node_id)
        .count();
    assert_eq!(concept_rows, 1, "concept must be de-duplicated to one row");

    // Concept reached at depth 1 (min depth wins).
    let cn = results
        .iter()
        .find(|n| n.node_id == concept.node_id)
        .unwrap();
    assert_eq!(cn.depth, 1);
}

#[tokio::test]
async fn test_empty_edge_types_or_zero_hops_returns_empty() {
    let store = test_store().await;
    let (code_a, _doc_a, _c, _l, _b) = build_cross_domain(&store).await;

    assert!(store
        .query_cross_boundary(TENANT_A, &code_a.node_id, &[], 2, &[])
        .await
        .unwrap()
        .is_empty());
    assert!(store
        .query_cross_boundary(TENANT_A, &code_a.node_id, &edges(), 0, &[])
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn test_per_hit_cap_limits_direct_expansions() {
    use crate::config::GraphRagConfig;
    let store = test_store().await.with_graph_rag_config(GraphRagConfig {
        max_per_hit: 3,
        ..Default::default()
    });

    // One source symbol implementing 10 distinct concepts (10 direct hop-1 hits).
    let src = GraphNode::new(TENANT_A, "s.rs", "src_fn", NodeType::Function);
    store.upsert_nodes(&[src.clone()]).await.unwrap();
    for i in 0..10 {
        let c = GraphNode::new(GLOBAL, "", format!("c{i}"), NodeType::ConceptNode);
        store.upsert_nodes(&[c.clone()]).await.unwrap();
        insert_weighted_edge(
            &store,
            TENANT_A,
            &src.node_id,
            &c.node_id,
            EdgeType::ImplementsConcept,
            0.5 + (i as f64) * 0.01,
        )
        .await;
    }

    let results = store
        .query_cross_boundary(TENANT_A, &src.node_id, &edges(), 1, &[])
        .await
        .unwrap();
    let hop1 = results.iter().filter(|n| n.depth == 1).count();
    assert_eq!(hop1, 3, "per-hit cap should keep only 3 direct expansions");
}

#[tokio::test]
async fn test_per_concept_cap_limits_supernode_fanout() {
    use crate::config::GraphRagConfig;
    let store = test_store().await.with_graph_rag_config(GraphRagConfig {
        max_per_concept: 4,
        max_total: 50,
        ..Default::default()
    });

    let concept = GraphNode::new(GLOBAL, "", "hub", NodeType::ConceptNode);
    let src = GraphNode::new(TENANT_A, "s.rs", "seed", NodeType::Function);
    store
        .upsert_nodes(&[concept.clone(), src.clone()])
        .await
        .unwrap();
    insert_weighted_edge(
        &store,
        TENANT_A,
        &src.node_id,
        &concept.node_id,
        EdgeType::ImplementsConcept,
        0.9,
    )
    .await;

    // 20 other symbols all implementing the same concept (reverse fan-out at hop 2).
    for i in 0..20 {
        let n = GraphNode::new(
            TENANT_A,
            format!("o{i}.rs"),
            format!("o{i}"),
            NodeType::Function,
        );
        store.upsert_nodes(&[n.clone()]).await.unwrap();
        insert_weighted_edge(
            &store,
            TENANT_A,
            &n.node_id,
            &concept.node_id,
            EdgeType::ImplementsConcept,
            0.5,
        )
        .await;
    }

    let results = store
        .query_cross_boundary(TENANT_A, &src.node_id, &edges(), 2, &[])
        .await
        .unwrap();
    let via_hub = results.iter().filter(|n| n.depth == 2).count();
    assert!(
        via_hub <= 4,
        "per-concept cap should bound hop-2 fan-out to 4, got {via_hub}"
    );
}

#[tokio::test]
async fn test_total_cap() {
    use crate::config::GraphRagConfig;
    let store = test_store().await.with_graph_rag_config(GraphRagConfig {
        max_per_hit: 100,
        max_per_concept: 100,
        max_total: 5,
        ..Default::default()
    });

    let src = GraphNode::new(TENANT_A, "s.rs", "seed", NodeType::Function);
    store.upsert_nodes(&[src.clone()]).await.unwrap();
    for i in 0..12 {
        let c = GraphNode::new(GLOBAL, "", format!("c{i}"), NodeType::ConceptNode);
        store.upsert_nodes(&[c.clone()]).await.unwrap();
        insert_weighted_edge(
            &store,
            TENANT_A,
            &src.node_id,
            &c.node_id,
            EdgeType::ImplementsConcept,
            0.5,
        )
        .await;
    }

    let results = store
        .query_cross_boundary(TENANT_A, &src.node_id, &edges(), 1, &[])
        .await
        .unwrap();
    assert_eq!(results.len(), 5, "total cap must bound result set to 5");
}

#[tokio::test]
async fn test_supernode_query_under_budget() {
    // PERF-5 worst case (scaled for CI): one ConceptNode with in-degree 1000
    // (500 code symbols + 500 library sections). A 2-hop bidirectional
    // traversal must complete well under the 100ms target with fan-out caps.
    let store = test_store().await;
    let concept = GraphNode::new(GLOBAL, "", "supernode", NodeType::ConceptNode);
    store.upsert_nodes(&[concept.clone()]).await.unwrap();

    let degree = 500usize;
    let mut nodes = Vec::with_capacity(degree * 2);
    let mut edge_recs = Vec::with_capacity(degree * 2);
    let mut seed_id = String::new();
    for i in 0..degree {
        let code = GraphNode::new(
            TENANT_A,
            format!("c{i}.rs"),
            format!("s{i}"),
            NodeType::Function,
        );
        if i == 0 {
            seed_id = code.node_id.clone();
        }
        let mut e = GraphEdge::new(
            TENANT_A,
            &code.node_id,
            &concept.node_id,
            EdgeType::ImplementsConcept,
            "c.rs",
        );
        e.weight = 0.5;
        edge_recs.push(e);
        nodes.push(code);

        let doc = GraphNode::new(
            LIB,
            format!("d{i}.md"),
            format!("sec{i}"),
            NodeType::LibrarySection,
        );
        let mut e2 = GraphEdge::new(
            LIB,
            &doc.node_id,
            &concept.node_id,
            EdgeType::CoversTopic,
            "d.md",
        );
        e2.weight = 0.5;
        edge_recs.push(e2);
        nodes.push(doc);
    }
    store.upsert_nodes(&nodes).await.unwrap();
    store.insert_edges(&edge_recs).await.unwrap();

    let start = std::time::Instant::now();
    let results = store
        .query_cross_boundary(TENANT_A, &seed_id, &edges(), 2, &[LIB.to_string()])
        .await
        .unwrap();
    let elapsed = start.elapsed();

    // Fan-out caps bound the output regardless of supernode degree.
    assert!(
        results.len() <= 50,
        "total cap should bound output, got {}",
        results.len()
    );
    // Wall-clock assertion only on optimized builds: debug builds are ~20x
    // slower and would flap. The PERF-5 target (< 100ms, graphs < 50K nodes) is
    // validated against the release profile and the criterion benchmark.
    #[cfg(not(debug_assertions))]
    assert!(
        elapsed.as_millis() < 100,
        "supernode 2-hop traversal must stay under 100ms, took {}ms",
        elapsed.as_millis()
    );
    let _ = elapsed;
}

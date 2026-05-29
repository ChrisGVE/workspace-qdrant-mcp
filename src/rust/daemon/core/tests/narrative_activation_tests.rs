//! End-to-end integration tests for narrative-graph activation (Phase 1).
//!
//! These tests exercise the narrative components against a real
//! `SqliteGraphStore`: code symbols are ingested first, then the narrative
//! extractors run with a real per-tenant symbol automaton, and the combined
//! result is written through the single `reingest_file` transaction exactly as
//! the live ingest pipeline (`narrative_phase`) does. They validate:
//!
//! * INT-A1 — EXPLAINS edges target REAL code-graph node ids (zero stubs) and
//!   their source equals the SectionExtractor section id.
//! * INT-A2 — re-ingesting an edited file removes stale narrative nodes/edges
//!   without touching code-graph nodes.
//! * INT-A3 — narrative nodes/edges carry the ingest branch.

#[allow(dead_code)]
#[path = "common/graph_helpers.rs"]
mod graph_helpers;

use graph_helpers::{build_rust_file_chunks, create_factory_store, ingest_file_chunks};

use std::path::Path;

use workspace_qdrant_core::config::NarrativeConfig;
use workspace_qdrant_core::graph::{EdgeType, GraphStore, NodeType, SymbolRow};
use workspace_qdrant_core::narrative::explains::ExplainsExtractor;
use workspace_qdrant_core::narrative::sections::SectionExtractor;
use workspace_qdrant_core::narrative::symbol_index::SymbolAutomaton;
use workspace_qdrant_core::narrative::{run_narrative_pipeline, NarrativeExtractor};

const TENANT: &str = "narrative-it";

/// Markdown doc that references the `process` and `validate` symbols defined
/// in `build_rust_file_chunks` (each mentioned 2+ times for the occurrence
/// filter), under two headings.
const README: &str = "\
# Processing Pipeline

The process method drives ingestion. Call process for every file.
It delegates validation to helpers.

# Validation

The validate method checks inputs. validate runs before transform.
Always validate untrusted data.
";

async fn build_automaton(store: &impl GraphStore, tenant: &str) -> SymbolAutomaton {
    let symbols: Vec<SymbolRow> = store.query_code_symbols(tenant).await.unwrap();
    SymbolAutomaton::build(&symbols, 4)
}

async fn run_narrative(
    store: &(impl GraphStore + 'static),
    tenant: &str,
    rel_path: &str,
    content: &str,
    branch: &str,
) -> workspace_qdrant_core::narrative::NarrativeExtractionResult {
    let automaton = build_automaton(store, tenant).await;
    let section_extractor = SectionExtractor::new();
    let spans = section_extractor.section_spans(tenant, rel_path, content);

    let extractors: Vec<Box<dyn NarrativeExtractor>> = vec![
        Box::new(SectionExtractor::new()),
        Box::new(ExplainsExtractor::with_context(
            spans,
            automaton,
            NarrativeConfig::default(),
        )),
    ];

    run_narrative_pipeline(
        tenant,
        Path::new(rel_path),
        content,
        None,
        branch,
        &extractors,
    )
    .await
}

#[tokio::test]
async fn int_a1_explains_targets_real_symbols_and_section_source() {
    let dir = tempfile::tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    // Ingest code symbols so the automaton resolves real node ids.
    let chunks = build_rust_file_chunks();
    ingest_file_chunks(&store, &chunks, TENANT, "src/processor.rs").await;

    let rel = "docs/README.md";
    let result = run_narrative(&store, TENANT, rel, README, "main").await;

    // DocumentSection nodes exist for both headings.
    let section_ids: Vec<&str> = result
        .nodes
        .iter()
        .filter(|n| n.symbol_type == NodeType::DocumentSection)
        .map(|n| n.node_id.as_str())
        .collect();
    assert_eq!(section_ids.len(), 2, "expected two DocumentSection nodes");

    // EXPLAINS edges present, and every target is a REAL code node id (zero
    // stubs / empty-path nodes). Resolve targets against the store.
    let explains: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::Explains)
        .collect();
    assert!(!explains.is_empty(), "expected EXPLAINS edges");

    let real_symbols = store.query_code_symbols(TENANT).await.unwrap();
    let real_ids: std::collections::HashSet<&str> =
        real_symbols.iter().map(|s| s.node_id.as_str()).collect();

    for edge in &explains {
        assert!(
            real_ids.contains(edge.target_node_id.as_str()),
            "EXPLAINS target {} is not a real code-graph node id",
            edge.target_node_id
        );
        // Source must be one of the SectionExtractor section node ids.
        assert!(
            section_ids.contains(&edge.source_node_id.as_str()),
            "EXPLAINS source {} is not a SectionExtractor section id",
            edge.source_node_id
        );
    }

    // No node in the narrative result has an empty file_path (would indicate a
    // stub). DocumentSection nodes carry the doc path.
    for node in &result.nodes {
        assert!(
            !node.file_path.is_empty() || node.symbol_type == NodeType::ConceptNode,
            "narrative node {} has empty file_path (stub)",
            node.node_id
        );
    }

    // Persist through the single reingest_file transaction (as the live path
    // does) and confirm the edges land.
    store
        .reingest_file(TENANT, rel, &result.nodes, &result.edges)
        .await
        .unwrap();
    let stored = store.query_edges_by_type(EdgeType::Explains).await.unwrap();
    assert!(
        stored.iter().any(|e| e.source_file == rel),
        "EXPLAINS edges for {rel} should be persisted"
    );
}

#[tokio::test]
async fn int_a2_reingest_removes_stale_narrative_without_touching_code() {
    let dir = tempfile::tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    let chunks = build_rust_file_chunks();
    ingest_file_chunks(&store, &chunks, TENANT, "src/processor.rs").await;
    let code_nodes_before = store
        .query_code_symbols(TENANT)
        .await
        .unwrap()
        .len();
    assert!(code_nodes_before > 0);

    let rel = "docs/README.md";

    // First ingest.
    let r1 = run_narrative(&store, TENANT, rel, README, "main").await;
    let sections_before: Vec<String> = r1
        .nodes
        .iter()
        .filter(|n| n.symbol_type == NodeType::DocumentSection)
        .map(|n| n.node_id.clone())
        .collect();
    store
        .reingest_file(TENANT, rel, &r1.nodes, &r1.edges)
        .await
        .unwrap();

    // Edit: rename a heading so its section node id changes, shifting lines.
    let edited = README.replace("# Processing Pipeline", "# Ingestion Pipeline");
    let r2 = run_narrative(&store, TENANT, rel, &edited, "main").await;
    store
        .reingest_file(TENANT, rel, &r2.nodes, &r2.edges)
        .await
        .unwrap();

    // The OLD "Processing Pipeline" section node must be gone (deleted by the
    // reingest narrative-node cleanup), proving no orphan accumulation.
    let stats = store.stats(Some(TENANT), None).await.unwrap();
    let doc_section_count = *stats
        .nodes_by_type
        .get("document_section")
        .unwrap_or(&0);
    assert_eq!(
        doc_section_count, 2,
        "expected exactly 2 document_section nodes after re-ingest, got {doc_section_count}"
    );

    // Code-graph symbol nodes are untouched.
    let code_nodes_after = store.query_code_symbols(TENANT).await.unwrap().len();
    assert_eq!(
        code_nodes_before, code_nodes_after,
        "code-graph nodes must survive narrative re-ingest"
    );

    // The first ingest's renamed section id is no longer present.
    let processing_id = sections_before
        .iter()
        .find(|_| true)
        .cloned()
        .unwrap_or_default();
    let related = store
        .query_related(TENANT, &processing_id, 1, None, None)
        .await
        .unwrap();
    // After deletion the old node has no outgoing traversal results.
    assert!(
        related.is_empty(),
        "stale section node should have no edges after re-ingest"
    );
}

#[tokio::test]
async fn int_a3_narrative_carries_ingest_branch() {
    let dir = tempfile::tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    let chunks = build_rust_file_chunks();
    ingest_file_chunks(&store, &chunks, TENANT, "src/processor.rs").await;

    let rel = "docs/README.md";
    let result = run_narrative(&store, TENANT, rel, README, "feature/x").await;

    for node in result
        .nodes
        .iter()
        .filter(|n| n.symbol_type == NodeType::DocumentSection)
    {
        assert_eq!(
            node.branches, r#"["feature/x"]"#,
            "narrative node must carry the ingest branch"
        );
    }
    for edge in &result.edges {
        assert_eq!(
            edge.branch.as_deref(),
            Some("feature/x"),
            "narrative edge must carry the ingest branch"
        );
    }
}

#[tokio::test]
async fn int_a4_no_symbols_yields_no_explains_but_sections_remain() {
    // Failure-isolation-adjacent: when the automaton is empty (no code symbols
    // ingested yet), EXPLAINS produces nothing but sections still extract — the
    // narrative phase degrades gracefully rather than erroring.
    let dir = tempfile::tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    let rel = "docs/README.md";
    let result = run_narrative(&store, TENANT, rel, README, "main").await;

    let sections = result
        .nodes
        .iter()
        .filter(|n| n.symbol_type == NodeType::DocumentSection)
        .count();
    assert_eq!(sections, 2, "sections extract even with no code symbols");

    let explains = result
        .edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::Explains)
        .count();
    assert_eq!(explains, 0, "no symbols → no EXPLAINS edges");
}

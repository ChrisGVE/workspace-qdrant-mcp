//! Phase 4b: narrative graph extraction during file ingestion.
//!
//! Builds the per-tenant symbol automaton, runs the ordered narrative
//! extractor pipeline, and returns the resulting nodes/edges so they can be
//! merged into the file's single `reingest_file` transaction (alongside the
//! code- and concept-graph layers). This module performs NO graph writes: a
//! separate `reingest_file`/`insert_edges` call here would delete the code and
//! concept edges already prepared for this file.
//!
//! Failure isolation: callers wrap [`run`] so any extractor error yields an
//! empty result and a warning — narrative extraction must never abort ingest.

use std::path::Path;

use tracing::debug;

use crate::context::ProcessingContext;
use crate::graph::{
    compute_node_id, compute_node_id_for_type, EdgeType, GraphEdge, GraphNode, NodeIdFields,
    NodeType,
};
use crate::narrative::comments::CommentExtractor;
use crate::narrative::depth::estimate_depth;
use crate::narrative::explains::ExplainsExtractor;
use crate::narrative::references::ReferencesExtractor;
use crate::narrative::sections::SectionExtractor;
use crate::narrative::{run_narrative_pipeline, NarrativeExtractionResult, NarrativeExtractor};

/// Run narrative extraction for a file and return its nodes/edges (plus any
/// COVERS_TOPIC concept nodes/edges) for merging into the graph reingest.
///
/// Returns an empty result when narrative extraction is disabled, no graph
/// store is configured, or no narrative content is found.
///
/// `taxonomy_tags` are the document-level Tier-2 tags already computed in
/// Phase 2b; when present and the file is narrative, COVERS_TOPIC edges with
/// depth metadata are produced here (so they share the reingest transaction
/// rather than being written separately).
pub(super) async fn run(
    ctx: &ProcessingContext,
    tenant_id: &str,
    file_path: &Path,
    relative_path: &str,
    content: &str,
    language: Option<&str>,
    branch: &str,
    taxonomy_tags: &[String],
) -> NarrativeExtractionResult {
    let cfg = &ctx.narrative_config;
    if !cfg.enabled {
        return NarrativeExtractionResult::default();
    }

    let Some(ref graph_store) = ctx.graph_store else {
        return NarrativeExtractionResult::default();
    };

    // Build (or reuse) the tenant's symbol automaton from REAL code-graph
    // symbols. This runs after Phase 4 has written this file's own code nodes,
    // so within-file references resolve.
    let automaton = ctx
        .automaton_cache
        .get_or_build(graph_store, tenant_id, cfg.explains_min_symbol_length)
        .await;

    // SectionExtractor is the single source of section identity; capture its
    // spans so ExplainsExtractor attaches edges to the same section node ids.
    let section_extractor = SectionExtractor::new();
    let section_spans = section_extractor.section_spans(tenant_id, relative_path, content);

    // Ordered extractors: sections first (already captured), then explains
    // (context-aware), then comments + references (context-free).
    let extractors: Vec<Box<dyn NarrativeExtractor>> = vec![
        Box::new(SectionExtractor::new()),
        Box::new(ExplainsExtractor::with_context(
            section_spans,
            automaton,
            (**cfg).clone(),
        )),
        Box::new(CommentExtractor),
        Box::new(ReferencesExtractor::new()),
    ];

    let mut result = run_narrative_pipeline(
        tenant_id,
        Path::new(relative_path),
        content,
        language,
        branch,
        &extractors,
    )
    .await;

    // COVERS_TOPIC edges for narrative files (shares the reingest transaction).
    if !taxonomy_tags.is_empty() && is_narrative_file(file_path) {
        let covers =
            build_covers_topic_edges(tenant_id, relative_path, taxonomy_tags, content, branch);
        result.merge(covers);
    }

    debug!(
        "narrative_phase: {} nodes, {} edges for {} (branch {})",
        result.nodes.len(),
        result.edges.len(),
        relative_path,
        branch
    );
    result
}

/// Build COVERS_TOPIC nodes/edges from a narrative file's File node to global
/// ConceptNodes, carrying depth metadata. Returns them for merging (no writes).
fn build_covers_topic_edges(
    tenant_id: &str,
    relative_path: &str,
    taxonomy_tags: &[String],
    content: &str,
    branch: &str,
) -> NarrativeExtractionResult {
    let depth_level = estimate_depth(content, 0, false);
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    let file_node_id = compute_node_id(tenant_id, relative_path, relative_path, NodeType::File);

    for tag in taxonomy_tags {
        let concept_label = tag.strip_prefix("tax:").unwrap_or(tag);
        let concept_fields =
            NodeIdFields::new("__global__", "", concept_label, NodeType::ConceptNode);
        let concept_id = compute_node_id_for_type(&concept_fields);

        let mut node = GraphNode::new("__global__", "", concept_label, NodeType::ConceptNode);
        node.node_id = concept_id.clone();
        nodes.push(node);

        edges.push(
            GraphEdge::new(
                tenant_id,
                &file_node_id,
                &concept_id,
                EdgeType::CoversTopic,
                relative_path,
            )
            .with_depth(depth_level)
            .with_branch(branch),
        );
    }

    NarrativeExtractionResult { nodes, edges }
}

/// Check whether a file is a narrative document (markdown / text / rst).
fn is_narrative_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map_or(false, |ext| {
            let lower = ext.to_ascii_lowercase();
            lower == "md" || lower == "markdown" || lower == "txt" || lower == "rst"
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_topic_uses_global_concept_and_branch() {
        let result = build_covers_topic_edges(
            "t1",
            "docs/readme.md",
            &["tax:databases".to_string(), "tax:networking".to_string()],
            "# Title\nsome intro text about databases\n",
            "dev",
        );
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.edges.len(), 2);
        for node in &result.nodes {
            assert_eq!(node.symbol_type, NodeType::ConceptNode);
            assert_eq!(node.tenant_id, "__global__");
        }
        for edge in &result.edges {
            assert_eq!(edge.edge_type, EdgeType::CoversTopic);
            assert_eq!(edge.branch.as_deref(), Some("dev"));
            assert_eq!(edge.source_file, "docs/readme.md");
            assert!(edge.metadata_json.is_some(), "depth metadata expected");
        }
    }

    #[test]
    fn is_narrative_file_detection() {
        assert!(is_narrative_file(Path::new("a.md")));
        assert!(is_narrative_file(Path::new("a.rst")));
        assert!(is_narrative_file(Path::new("a.txt")));
        assert!(!is_narrative_file(Path::new("a.rs")));
    }
}

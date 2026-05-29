/// Narrative extraction pipeline for building the narrative graph layer.
///
/// Extracts human-authored documentation elements (document sections,
/// code comments, docstrings, library sections) and creates graph nodes
/// and edges connecting them to code symbols via concept nodes.
pub mod comments;
pub mod depth;
pub mod elaborates;
pub mod explains;
pub mod references;
pub mod sections;
pub mod symbol_index;

use std::path::Path;

use async_trait::async_trait;

use crate::graph::{GraphEdge, GraphNode, NodeType};

pub use symbol_index::SymbolAutomaton;

/// Result of narrative extraction from a single file.
#[derive(Debug, Default)]
pub struct NarrativeExtractionResult {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

impl NarrativeExtractionResult {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }

    pub fn merge(&mut self, other: NarrativeExtractionResult) {
        self.nodes.extend(other.nodes);
        self.edges.extend(other.edges);
    }

    /// Tag every node and edge with `branch`, so narrative graph elements are
    /// branch-scoped exactly like the code-graph layer (R1.7). Without this
    /// the elements default to `main` and never appear on feature branches.
    pub fn apply_branch(&mut self, branch: &str) {
        for node in &mut self.nodes {
            node.branches = format!(r#"["{branch}"]"#);
        }
        for edge in &mut self.edges {
            edge.branch = Some(branch.to_string());
        }
    }
}

/// Trait for extracting narrative graph elements from file content.
#[async_trait]
pub trait NarrativeExtractor: Send + Sync {
    fn supported_node_types(&self) -> &[NodeType];

    async fn extract(
        &self,
        tenant_id: &str,
        file_path: &Path,
        content: &str,
        language: Option<&str>,
    ) -> NarrativeExtractionResult;
}

/// Run the narrative extraction pipeline on a file and RETURN the combined
/// result (PURE — performs no graph writes).
///
/// Each extractor is run in order and its nodes/edges merged. The combined
/// result is branch-tagged via [`NarrativeExtractionResult::apply_branch`].
///
/// Persistence is the caller's responsibility: the result is threaded into the
/// file's single `reingest_file` transaction (alongside the code- and
/// concept-graph nodes/edges) so the one delete-then-insert covers every layer
/// for the file and stale narrative edges are cleaned up on re-ingestion (R3).
/// Writing narrative output through a separate `reingest_file`/`insert_edges`
/// call would delete the code and concept edges already written for the file.
pub async fn run_narrative_pipeline(
    tenant_id: &str,
    file_path: &Path,
    content: &str,
    language: Option<&str>,
    branch: &str,
    extractors: &[Box<dyn NarrativeExtractor>],
) -> NarrativeExtractionResult {
    let mut result = NarrativeExtractionResult::default();

    for extractor in extractors {
        let partial = extractor
            .extract(tenant_id, file_path, content, language)
            .await;
        result.merge(partial);
    }

    result.apply_branch(branch);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_result_merge() {
        let mut a = NarrativeExtractionResult::default();
        assert!(a.is_empty());

        let node = GraphNode::new("t1", "doc.md", "section1", NodeType::DocumentSection);
        let b = NarrativeExtractionResult {
            nodes: vec![node],
            edges: vec![],
        };
        a.merge(b);
        assert!(!a.is_empty());
        assert_eq!(a.nodes.len(), 1);
    }

    #[test]
    fn test_apply_branch_tags_nodes_and_edges() {
        use crate::graph::{EdgeType, GraphEdge};

        let mut result = NarrativeExtractionResult {
            nodes: vec![GraphNode::new(
                "t1",
                "doc.md",
                "Intro",
                NodeType::DocumentSection,
            )],
            edges: vec![GraphEdge::new(
                "t1",
                "src",
                "dst",
                EdgeType::Explains,
                "doc.md",
            )],
        };
        result.apply_branch("feature/x");
        assert_eq!(result.nodes[0].branches, r#"["feature/x"]"#);
        assert_eq!(result.edges[0].branch.as_deref(), Some("feature/x"));
    }

    #[tokio::test]
    async fn test_pipeline_is_pure_and_branch_tagged() {
        // SectionExtractor produces DocumentSection nodes; pipeline must return
        // them branch-tagged without performing any writes.
        let extractors: Vec<Box<dyn NarrativeExtractor>> =
            vec![Box::new(sections::SectionExtractor::new())];
        let md = "# Title\nbody text here\n";
        let result =
            run_narrative_pipeline("t1", Path::new("doc.md"), md, None, "dev", &extractors).await;
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].branches, r#"["dev"]"#);
    }
}

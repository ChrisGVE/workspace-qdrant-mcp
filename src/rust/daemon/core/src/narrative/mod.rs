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

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use crate::graph::{GraphEdge, GraphNode, GraphStore, NodeType};

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

/// Run the narrative extraction pipeline on a file.
///
/// Calls each registered extractor and merges results. Writes
/// extracted nodes and edges to the graph store.
pub async fn run_narrative_pipeline(
    graph_store: &Arc<dyn GraphStore>,
    tenant_id: &str,
    file_path: &Path,
    content: &str,
    language: Option<&str>,
    extractors: &[Box<dyn NarrativeExtractor>],
) -> NarrativeExtractionResult {
    let mut result = NarrativeExtractionResult::default();

    for extractor in extractors {
        let partial = extractor
            .extract(tenant_id, file_path, content, language)
            .await;
        result.merge(partial);
    }

    if !result.is_empty() {
        if let Err(e) = graph_store.upsert_nodes(&result.nodes).await {
            tracing::warn!("Failed to upsert narrative nodes: {e}");
        }
        if let Err(e) = graph_store.insert_edges(&result.edges).await {
            tracing::warn!("Failed to insert narrative edges: {e}");
        }
    }

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
}

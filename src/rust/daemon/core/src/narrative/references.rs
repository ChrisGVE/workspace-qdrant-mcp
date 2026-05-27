/// REFERENCES_DOC edge creation from markdown cross-references.
///
/// Parses inline `[text](path)` and reference-style `[label]: path`
/// links in markdown files, filters to local file targets, and emits
/// a `REFERENCES_DOC` edge for each unique referenced document.
use std::collections::HashSet;
use std::path::Path;

use async_trait::async_trait;
use regex::Regex;

use crate::graph::{EdgeType, GraphEdge, GraphNode, NodeType};

use super::{NarrativeExtractionResult, NarrativeExtractor};

/// Extracts `REFERENCES_DOC` edges from markdown link syntax.
pub struct ReferencesExtractor {
    /// Matches inline links: `[text](target)`
    inline_re: Regex,
    /// Matches reference-style links: `[label]: target`
    refstyle_re: Regex,
}

impl ReferencesExtractor {
    pub fn new() -> Self {
        Self {
            inline_re: Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").expect("inline regex is valid"),
            refstyle_re: Regex::new(r"^\[([^\]]+)\]:\s+(\S+)")
                .expect("reference-style regex is valid"),
        }
    }
}

impl Default for ReferencesExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// URL-scheme prefixes that indicate non-local references.
const REMOTE_PREFIXES: &[&str] = &["http://", "https://", "mailto:", "#"];

/// Return `true` when the target string looks like a remote URL or
/// anchor-only reference (not a local file path).
fn is_remote_or_anchor(target: &str) -> bool {
    let lower = target.to_ascii_lowercase();
    REMOTE_PREFIXES.iter().any(|p| lower.starts_with(p))
}

/// Strip an optional fragment (`#heading`) from the end of a path.
fn strip_fragment(target: &str) -> &str {
    target.split('#').next().unwrap_or(target)
}

/// Normalize a path by resolving `.` and `..` components lexically.
///
/// Unlike `std::fs::canonicalize` this does not touch the filesystem,
/// keeping the extractor side-effect-free.
fn normalize_path(path: &Path) -> std::path::PathBuf {
    let mut components = Vec::new();
    for c in path.components() {
        match c {
            std::path::Component::CurDir => {} // skip `.`
            std::path::Component::ParentDir => {
                components.pop();
            }
            other => components.push(other),
        }
    }
    components.iter().collect()
}

#[async_trait]
impl NarrativeExtractor for ReferencesExtractor {
    fn supported_node_types(&self) -> &[NodeType] {
        &[NodeType::File]
    }

    async fn extract(
        &self,
        tenant_id: &str,
        file_path: &Path,
        content: &str,
        _language: Option<&str>,
    ) -> NarrativeExtractionResult {
        // Only process markdown files.
        let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext.to_ascii_lowercase().as_str(), "md" | "markdown") {
            return NarrativeExtractionResult::default();
        }

        let file_path_str = file_path.to_string_lossy();
        let parent = file_path.parent().unwrap_or_else(|| Path::new(""));

        // Collect unique resolved target paths.
        let mut seen_targets: HashSet<String> = HashSet::new();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Source node: the current markdown file.
        let source_node = GraphNode::new(
            tenant_id,
            file_path_str.as_ref(),
            file_path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
            NodeType::File,
        );
        let source_node_id = source_node.node_id.clone();

        let mut need_source_node = false;

        for line in content.lines() {
            // Inline links: [text](target)
            for caps in self.inline_re.captures_iter(line) {
                if let Some(target_match) = caps.get(2) {
                    self.process_target(
                        target_match.as_str(),
                        tenant_id,
                        parent,
                        &source_node_id,
                        &file_path_str,
                        &mut seen_targets,
                        &mut nodes,
                        &mut edges,
                        &mut need_source_node,
                    );
                }
            }

            // Reference-style links: [label]: target
            if let Some(caps) = self.refstyle_re.captures(line) {
                if let Some(target_match) = caps.get(2) {
                    self.process_target(
                        target_match.as_str(),
                        tenant_id,
                        parent,
                        &source_node_id,
                        &file_path_str,
                        &mut seen_targets,
                        &mut nodes,
                        &mut edges,
                        &mut need_source_node,
                    );
                }
            }
        }

        if need_source_node {
            // Prepend source node so it appears before target stubs.
            nodes.insert(0, source_node);
        }

        NarrativeExtractionResult { nodes, edges }
    }
}

impl ReferencesExtractor {
    /// Evaluate a single link target: resolve its path, deduplicate, and
    /// if it is a new local reference, emit a stub node and edge.
    #[allow(clippy::too_many_arguments)]
    fn process_target(
        &self,
        raw_target: &str,
        tenant_id: &str,
        parent: &Path,
        source_node_id: &str,
        file_path_str: &str,
        seen_targets: &mut HashSet<String>,
        nodes: &mut Vec<GraphNode>,
        edges: &mut Vec<GraphEdge>,
        need_source_node: &mut bool,
    ) {
        if is_remote_or_anchor(raw_target) {
            return;
        }

        let path_part = strip_fragment(raw_target);
        if path_part.is_empty() {
            return;
        }

        let resolved = normalize_path(&parent.join(path_part));
        let resolved_str = resolved.to_string_lossy().to_string();

        if seen_targets.contains(&resolved_str) {
            return;
        }
        seen_targets.insert(resolved_str.clone());

        let target_filename = resolved.file_name().and_then(|n| n.to_str()).unwrap_or("");

        let target_node = GraphNode::new(tenant_id, &resolved_str, target_filename, NodeType::File);

        let edge = GraphEdge::new(
            tenant_id,
            source_node_id,
            &target_node.node_id,
            EdgeType::ReferencesDoc,
            file_path_str,
        );

        nodes.push(target_node);
        edges.push(edge);
        *need_source_node = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extractor() -> ReferencesExtractor {
        ReferencesExtractor::new()
    }

    /// Helper to run the async extract synchronously in tests.
    fn run_extract(
        ext: &ReferencesExtractor,
        tenant: &str,
        path: &str,
        content: &str,
    ) -> NarrativeExtractionResult {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(ext.extract(tenant, Path::new(path), content, None))
    }

    #[test]
    fn test_inline_link_produces_edge() {
        let ext = extractor();
        let md = "See the [guide](./guide.md) for details.\n";
        let result = run_extract(&ext, "t1", "src/docs/index.md", md);

        // 1 source File node + 1 target File node
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.edges.len(), 1);

        let edge = &result.edges[0];
        assert_eq!(edge.edge_type, EdgeType::ReferencesDoc);
        assert_eq!(edge.source_file, "src/docs/index.md");

        // Target node should resolve to src/docs/guide.md
        let target = result.nodes.iter().find(|n| n.symbol_name == "guide.md");
        assert!(target.is_some());
        assert_eq!(target.unwrap().file_path, "src/docs/guide.md");
    }

    #[test]
    fn test_http_url_no_edge() {
        let ext = extractor();
        let md = "Visit [docs](https://example.com) for info.\n";
        let result = run_extract(&ext, "t1", "readme.md", md);

        assert!(result.is_empty());
    }

    #[test]
    fn test_reference_style_link_produces_edge() {
        let ext = extractor();
        let md = "[api]: api-reference.md\n";
        let result = run_extract(&ext, "t1", "docs/index.md", md);

        assert_eq!(result.edges.len(), 1);
        let edge = &result.edges[0];
        assert_eq!(edge.edge_type, EdgeType::ReferencesDoc);

        let target = result
            .nodes
            .iter()
            .find(|n| n.symbol_name == "api-reference.md");
        assert!(target.is_some());
        assert_eq!(target.unwrap().file_path, "docs/api-reference.md");
    }

    #[test]
    fn test_non_markdown_returns_empty() {
        let ext = extractor();
        let content = "[link](./other.rs)\n";
        let result = run_extract(&ext, "t1", "src/main.rs", content);

        assert!(result.is_empty());
    }

    #[test]
    fn test_duplicate_link_single_edge() {
        let ext = extractor();
        let md = "\
[guide](./guide.md) is useful.
Also see [guide again](./guide.md).
";
        let result = run_extract(&ext, "t1", "docs/index.md", md);

        // Deduplicated: only 1 target + 1 source = 2 nodes, 1 edge
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.edges.len(), 1);
    }

    #[test]
    fn test_anchor_only_no_edge() {
        let ext = extractor();
        let md = "Jump to [section](#heading) below.\n";
        let result = run_extract(&ext, "t1", "readme.md", md);

        assert!(result.is_empty());
    }

    #[test]
    fn test_parent_path_resolution() {
        let ext = extractor();
        let md = "See [readme](../README.md) at the root.\n";
        let result = run_extract(&ext, "t1", "src/docs/guide.md", md);

        assert_eq!(result.edges.len(), 1);
        let target = result.nodes.iter().find(|n| n.symbol_name == "README.md");
        assert!(target.is_some());
        assert_eq!(target.unwrap().file_path, "src/README.md");
    }

    #[test]
    fn test_mailto_no_edge() {
        let ext = extractor();
        let md = "Contact [us](mailto:hi@example.com).\n";
        let result = run_extract(&ext, "t1", "readme.md", md);

        assert!(result.is_empty());
    }

    #[test]
    fn test_link_with_fragment_resolves_path() {
        let ext = extractor();
        let md = "See [setup](./install.md#quickstart) guide.\n";
        let result = run_extract(&ext, "t1", "docs/index.md", md);

        assert_eq!(result.edges.len(), 1);
        let target = result.nodes.iter().find(|n| n.symbol_name == "install.md");
        assert!(target.is_some());
        assert_eq!(target.unwrap().file_path, "docs/install.md");
    }

    #[test]
    fn test_multiple_links_same_line() {
        let ext = extractor();
        let md = "See [a](a.md) and [b](b.md) for details.\n";
        let result = run_extract(&ext, "t1", "index.md", md);

        // 1 source + 2 targets = 3 nodes, 2 edges
        assert_eq!(result.nodes.len(), 3);
        assert_eq!(result.edges.len(), 2);
    }

    #[test]
    fn test_mixed_inline_and_refstyle() {
        let ext = extractor();
        let md = "\
Read [intro](intro.md) first.

[api]: api.md
[changelog]: changelog.md
";
        let result = run_extract(&ext, "t1", "docs/readme.md", md);

        // 1 source + 3 targets = 4 nodes, 3 edges
        assert_eq!(result.nodes.len(), 4);
        assert_eq!(result.edges.len(), 3);
    }

    #[test]
    fn test_source_node_is_file_type() {
        let ext = extractor();
        let md = "[link](other.md)\n";
        let result = run_extract(&ext, "t1", "docs/index.md", md);

        // First node is the source file node.
        assert_eq!(result.nodes[0].symbol_type, NodeType::File);
        assert_eq!(result.nodes[0].file_path, "docs/index.md");
    }

    #[test]
    fn test_edge_ids_are_deterministic() {
        let ext = extractor();
        let md = "[link](other.md)\n";
        let r1 = run_extract(&ext, "t1", "docs/index.md", md);
        let r2 = run_extract(&ext, "t1", "docs/index.md", md);

        assert_eq!(r1.edges[0].edge_id, r2.edges[0].edge_id);
        assert_eq!(r1.nodes[0].node_id, r2.nodes[0].node_id);
    }
}

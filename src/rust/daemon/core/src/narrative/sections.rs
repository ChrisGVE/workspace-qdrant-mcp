/// DocumentSection extraction from markdown and structured documents.
///
/// Splits documents into heading-delimited sections, creating
/// DocumentSection graph nodes for each.
use std::path::Path;

use async_trait::async_trait;
use regex::Regex;

use crate::graph::{compute_node_id_for_type, GraphNode, NodeIdFields, NodeType};

use super::{NarrativeExtractionResult, NarrativeExtractor};

/// Extracts heading-delimited sections from markdown and text files.
pub struct SectionExtractor {
    heading_re: Regex,
}

impl SectionExtractor {
    pub fn new() -> Self {
        Self {
            heading_re: Regex::new(r"^#{1,6}\s+(.+)$").expect("heading regex is valid"),
        }
    }
}

impl Default for SectionExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Check whether the file extension is one we handle.
fn is_supported_extension(path: &Path) -> Option<FileKind> {
    let ext = path.extension()?.to_str()?;
    match ext.to_ascii_lowercase().as_str() {
        "md" | "markdown" => Some(FileKind::Markdown),
        "txt" => Some(FileKind::PlainText),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileKind {
    Markdown,
    PlainText,
}

/// Extract sections from markdown content using ATX headings.
fn extract_markdown_sections(
    tenant_id: &str,
    file_path: &str,
    content: &str,
    heading_re: &Regex,
) -> Vec<GraphNode> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    // Collect (heading_text, start_line_1indexed) for each heading.
    let mut headings: Vec<(String, usize)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if let Some(caps) = heading_re.captures(line) {
            let text = caps.get(1).map_or("", |m| m.as_str()).trim().to_string();
            if !text.is_empty() {
                headings.push((text, i + 1)); // 1-indexed
            }
        }
    }

    if headings.is_empty() {
        return Vec::new();
    }

    let total_lines = lines.len();
    let mut nodes = Vec::with_capacity(headings.len());

    for (idx, (heading_text, start_line)) in headings.iter().enumerate() {
        // Section ends at the line before the next heading, or at EOF.
        let end_line = if idx + 1 < headings.len() {
            headings[idx + 1].1 - 1
        } else {
            total_lines
        };

        let fields = NodeIdFields {
            tenant_id,
            file_path,
            symbol_name: heading_text,
            symbol_type: NodeType::DocumentSection,
            section_index: Some(idx as u32),
            start_line: Some(*start_line as u32),
            library_name: None,
        };
        let node_id = compute_node_id_for_type(&fields);

        let mut node = GraphNode::new(
            tenant_id,
            file_path,
            heading_text,
            NodeType::DocumentSection,
        );
        node.node_id = node_id;
        node.start_line = Some(*start_line as u32);
        node.end_line = Some(end_line as u32);

        nodes.push(node);
    }

    nodes
}

/// Extract sections from plain text using blank-line-separated paragraphs.
///
/// The first non-empty line of each paragraph is used as the heading text.
fn extract_text_sections(tenant_id: &str, file_path: &str, content: &str) -> Vec<GraphNode> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    // Identify paragraph boundaries: groups of non-blank lines separated by
    // one or more blank lines. Each paragraph becomes a section.
    let mut paragraphs: Vec<(String, usize, usize)> = Vec::new(); // (heading, start, end)
    let mut para_start: Option<usize> = None;
    let mut heading: Option<String> = None;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            // End of current paragraph (if any).
            if let (Some(start), Some(h)) = (para_start.take(), heading.take()) {
                paragraphs.push((h, start + 1, i)); // end = last blank line (exclusive)
            }
        } else {
            if para_start.is_none() {
                para_start = Some(i);
                heading = Some(trimmed.to_string());
            }
        }
    }
    // Flush trailing paragraph (no trailing blank line).
    if let (Some(start), Some(h)) = (para_start, heading) {
        paragraphs.push((h, start + 1, lines.len()));
    }

    if paragraphs.is_empty() {
        return Vec::new();
    }

    let mut nodes = Vec::with_capacity(paragraphs.len());

    for (idx, (heading_text, start_line, end_line)) in paragraphs.iter().enumerate() {
        let fields = NodeIdFields {
            tenant_id,
            file_path,
            symbol_name: heading_text,
            symbol_type: NodeType::DocumentSection,
            section_index: Some(idx as u32),
            start_line: Some(*start_line as u32),
            library_name: None,
        };
        let node_id = compute_node_id_for_type(&fields);

        let mut node = GraphNode::new(
            tenant_id,
            file_path,
            heading_text,
            NodeType::DocumentSection,
        );
        node.node_id = node_id;
        node.start_line = Some(*start_line as u32);
        node.end_line = Some(*end_line as u32);

        nodes.push(node);
    }

    nodes
}

#[async_trait]
impl NarrativeExtractor for SectionExtractor {
    fn supported_node_types(&self) -> &[NodeType] {
        &[NodeType::DocumentSection, NodeType::LibrarySection]
    }

    async fn extract(
        &self,
        tenant_id: &str,
        file_path: &Path,
        content: &str,
        _language: Option<&str>,
    ) -> NarrativeExtractionResult {
        let kind = match is_supported_extension(file_path) {
            Some(k) => k,
            None => return NarrativeExtractionResult::default(),
        };

        let fp = file_path.to_string_lossy();

        let nodes = match kind {
            FileKind::Markdown => {
                extract_markdown_sections(tenant_id, &fp, content, &self.heading_re)
            }
            FileKind::PlainText => extract_text_sections(tenant_id, &fp, content),
        };

        NarrativeExtractionResult {
            nodes,
            edges: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extractor() -> SectionExtractor {
        SectionExtractor::new()
    }

    // Helper to run async extract in tests.
    fn run_extract(
        ext: &SectionExtractor,
        tenant: &str,
        path: &str,
        content: &str,
    ) -> NarrativeExtractionResult {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(ext.extract(tenant, Path::new(path), content, None))
    }

    #[test]
    fn test_markdown_five_sections() {
        let ext = extractor();
        let md = "\
## Introduction
Some intro text.

## Background
Background info here.

## Methods
Methodology described.

## Results
Results go here.

## Conclusion
Final thoughts.
";
        let result = run_extract(&ext, "tenant1", "doc.md", md);
        assert_eq!(result.nodes.len(), 5);
        assert!(result.edges.is_empty());

        let names: Vec<&str> = result
            .nodes
            .iter()
            .map(|n| n.symbol_name.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "Introduction",
                "Background",
                "Methods",
                "Results",
                "Conclusion"
            ]
        );

        // All nodes are DocumentSection type.
        for node in &result.nodes {
            assert_eq!(node.symbol_type, NodeType::DocumentSection);
            assert_eq!(node.tenant_id, "tenant1");
            assert_eq!(node.file_path, "doc.md");
            assert!(node.start_line.is_some());
            assert!(node.end_line.is_some());
        }
    }

    #[test]
    fn test_nested_headers_line_ranges() {
        let ext = extractor();
        let md = "\
# Top Level
Line 2 content.
Line 3 content.
## Sub Section
Line 5 content.
### Deep Section
Line 7 final.
";
        let result = run_extract(&ext, "t1", "nested.md", md);
        assert_eq!(result.nodes.len(), 3);

        // First section: lines 1..3
        assert_eq!(result.nodes[0].symbol_name, "Top Level");
        assert_eq!(result.nodes[0].start_line, Some(1));
        assert_eq!(result.nodes[0].end_line, Some(3));

        // Second section: lines 4..5
        assert_eq!(result.nodes[1].symbol_name, "Sub Section");
        assert_eq!(result.nodes[1].start_line, Some(4));
        assert_eq!(result.nodes[1].end_line, Some(5));

        // Third section: lines 6..7
        assert_eq!(result.nodes[2].symbol_name, "Deep Section");
        assert_eq!(result.nodes[2].start_line, Some(6));
        assert_eq!(result.nodes[2].end_line, Some(7));
    }

    #[test]
    fn test_empty_file_returns_empty() {
        let ext = extractor();
        let result = run_extract(&ext, "t1", "empty.md", "");
        assert!(result.is_empty());
    }

    #[test]
    fn test_non_markdown_file_returns_empty() {
        let ext = extractor();
        let result = run_extract(&ext, "t1", "code.rs", "## Not a markdown heading");
        assert!(result.is_empty());

        let result2 = run_extract(&ext, "t1", "data.json", "{}");
        assert!(result2.is_empty());
    }

    #[test]
    fn test_node_id_deterministic() {
        let ext = extractor();
        let md = "## Section A\nSome text.\n";
        let r1 = run_extract(&ext, "t1", "file.md", md);
        let r2 = run_extract(&ext, "t1", "file.md", md);

        assert_eq!(r1.nodes.len(), 1);
        assert_eq!(r2.nodes.len(), 1);
        assert_eq!(r1.nodes[0].node_id, r2.nodes[0].node_id);
    }

    #[test]
    fn test_plain_text_paragraphs() {
        let ext = extractor();
        let txt = "\
First paragraph title
Some body text here.

Second paragraph title
More body text.

Third paragraph title
Even more text.
";
        let result = run_extract(&ext, "t1", "readme.txt", txt);
        assert_eq!(result.nodes.len(), 3);

        assert_eq!(result.nodes[0].symbol_name, "First paragraph title");
        assert_eq!(result.nodes[0].start_line, Some(1));

        assert_eq!(result.nodes[1].symbol_name, "Second paragraph title");
        assert_eq!(result.nodes[2].symbol_name, "Third paragraph title");
    }

    #[test]
    fn test_markdown_no_headings() {
        let ext = extractor();
        let md = "Just plain text without any headings.\nAnother line.\n";
        let result = run_extract(&ext, "t1", "plain.md", md);
        assert!(result.is_empty());
    }

    #[test]
    fn test_txt_empty_returns_empty() {
        let ext = extractor();
        let result = run_extract(&ext, "t1", "empty.txt", "");
        assert!(result.is_empty());
    }
}

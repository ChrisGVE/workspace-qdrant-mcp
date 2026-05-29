/// DocumentSection extraction from markdown and structured documents.
///
/// Splits documents into heading-delimited sections, creating
/// DocumentSection graph nodes for each.
use std::path::Path;

use async_trait::async_trait;
use regex::Regex;

use crate::graph::{compute_node_id_for_type, GraphNode, NodeIdFields, NodeType};

use super::{NarrativeExtractionResult, NarrativeExtractor};

/// Canonical identity and line range of a single document section.
///
/// Produced by [`SectionExtractor::section_spans`], this is the single source
/// of section-node identity for inter-extractor data flow (R1). The
/// [`ExplainsExtractor`](super::explains::ExplainsExtractor) consumes these
/// spans to attach EXPLAINS edges to the section whose line range contains a
/// matched symbol, guaranteeing the EXPLAINS source matches the section node
/// id that `SectionExtractor::extract` itself produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionSpan {
    /// The deterministic node id of this section, identical to the one the
    /// `extract()` method assigns to the same heading/paragraph.
    pub node_id: String,
    /// 1-indexed first line of the section (the heading line for markdown).
    pub start_line: u32,
    /// 1-indexed last line of the section (inclusive).
    pub end_line: u32,
}

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

    /// Return the canonical per-heading (or per-paragraph) section identities
    /// for `content`, in document order.
    ///
    /// This is a pure helper: it performs no I/O and produces exactly the same
    /// node ids (and line ranges) that [`NarrativeExtractor::extract`] assigns,
    /// because both share [`collect_markdown_headings`] /
    /// [`collect_text_paragraphs`] internally. Unsupported file extensions
    /// yield an empty vector.
    ///
    /// `tenant_id` and `file_path` must match the values that will be used at
    /// ingestion time, since both participate in the node-id hash.
    pub fn section_spans(
        &self,
        tenant_id: &str,
        file_path: &str,
        content: &str,
    ) -> Vec<SectionSpan> {
        match is_supported_extension(Path::new(file_path)) {
            Some(FileKind::Markdown) => {
                markdown_section_spans(tenant_id, file_path, content, &self.heading_re)
            }
            Some(FileKind::PlainText) => text_section_spans(tenant_id, file_path, content),
            None => Vec::new(),
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

/// A heading collected from markdown content.
struct HeadingEntry {
    text: String,
    /// 1-indexed line of the heading.
    start_line: usize,
    /// 1-indexed last line of the section (inclusive).
    end_line: usize,
}

/// Collect ATX headings and their line ranges from markdown content.
///
/// Each heading's section runs from its own line to the line before the next
/// heading (or EOF for the last). Shared by both [`SectionExtractor::extract`]
/// and [`SectionExtractor::section_spans`] so the section identities they
/// produce are byte-for-byte identical.
fn collect_markdown_headings(content: &str, heading_re: &Regex) -> Vec<HeadingEntry> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

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
    let mut entries = Vec::with_capacity(headings.len());
    for (idx, (text, start_line)) in headings.iter().enumerate() {
        let end_line = if idx + 1 < headings.len() {
            headings[idx + 1].1 - 1
        } else {
            total_lines
        };
        entries.push(HeadingEntry {
            text: text.clone(),
            start_line: *start_line,
            end_line,
        });
    }
    entries
}

/// A paragraph collected from plain-text content.
struct ParagraphEntry {
    /// First non-empty line, used as the heading text.
    text: String,
    /// 1-indexed first line of the paragraph.
    start_line: usize,
    /// 1-indexed last line of the paragraph (inclusive).
    end_line: usize,
}

/// Collect blank-line-separated paragraphs and their line ranges from plain
/// text. The first non-empty line of each paragraph is its heading text.
/// Shared by both extraction paths so identities are identical.
fn collect_text_paragraphs(content: &str) -> Vec<ParagraphEntry> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut paragraphs: Vec<ParagraphEntry> = Vec::new();
    let mut para_start: Option<usize> = None;
    let mut heading: Option<String> = None;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if let (Some(start), Some(h)) = (para_start.take(), heading.take()) {
                paragraphs.push(ParagraphEntry {
                    text: h,
                    start_line: start + 1,
                    end_line: i, // last blank line index = 1-indexed previous line
                });
            }
        } else if para_start.is_none() {
            para_start = Some(i);
            heading = Some(trimmed.to_string());
        }
    }
    // Flush trailing paragraph (no trailing blank line).
    if let (Some(start), Some(h)) = (para_start, heading) {
        paragraphs.push(ParagraphEntry {
            text: h,
            start_line: start + 1,
            end_line: lines.len(),
        });
    }

    paragraphs
}

/// Compute the DocumentSection node id for a heading at `idx`/`start_line`.
fn document_section_node_id(
    tenant_id: &str,
    file_path: &str,
    heading_text: &str,
    idx: usize,
    start_line: usize,
) -> String {
    let fields = NodeIdFields {
        tenant_id,
        file_path,
        symbol_name: heading_text,
        symbol_type: NodeType::DocumentSection,
        section_index: Some(idx as u32),
        start_line: Some(start_line as u32),
        library_name: None,
    };
    compute_node_id_for_type(&fields)
}

/// Build markdown `SectionSpan`s (pure; matches `extract` identities).
fn markdown_section_spans(
    tenant_id: &str,
    file_path: &str,
    content: &str,
    heading_re: &Regex,
) -> Vec<SectionSpan> {
    collect_markdown_headings(content, heading_re)
        .iter()
        .enumerate()
        .map(|(idx, h)| SectionSpan {
            node_id: document_section_node_id(tenant_id, file_path, &h.text, idx, h.start_line),
            start_line: h.start_line as u32,
            end_line: h.end_line as u32,
        })
        .collect()
}

/// Build plain-text `SectionSpan`s (pure; matches `extract` identities).
fn text_section_spans(tenant_id: &str, file_path: &str, content: &str) -> Vec<SectionSpan> {
    collect_text_paragraphs(content)
        .iter()
        .enumerate()
        .map(|(idx, p)| SectionSpan {
            node_id: document_section_node_id(tenant_id, file_path, &p.text, idx, p.start_line),
            start_line: p.start_line as u32,
            end_line: p.end_line as u32,
        })
        .collect()
}

/// Extract sections from markdown content using ATX headings.
fn extract_markdown_sections(
    tenant_id: &str,
    file_path: &str,
    content: &str,
    heading_re: &Regex,
) -> Vec<GraphNode> {
    collect_markdown_headings(content, heading_re)
        .iter()
        .enumerate()
        .map(|(idx, h)| {
            let mut node = GraphNode::new(tenant_id, file_path, &h.text, NodeType::DocumentSection);
            node.node_id =
                document_section_node_id(tenant_id, file_path, &h.text, idx, h.start_line);
            node.start_line = Some(h.start_line as u32);
            node.end_line = Some(h.end_line as u32);
            node
        })
        .collect()
}

/// Extract sections from plain text using blank-line-separated paragraphs.
///
/// The first non-empty line of each paragraph is used as the heading text.
fn extract_text_sections(tenant_id: &str, file_path: &str, content: &str) -> Vec<GraphNode> {
    collect_text_paragraphs(content)
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            let mut node = GraphNode::new(tenant_id, file_path, &p.text, NodeType::DocumentSection);
            node.node_id =
                document_section_node_id(tenant_id, file_path, &p.text, idx, p.start_line);
            node.start_line = Some(p.start_line as u32);
            node.end_line = Some(p.end_line as u32);
            node
        })
        .collect()
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

    // ── section_spans (task 9) ───────────────────────────────────────────

    #[test]
    fn test_section_spans_multi_heading_markdown() {
        let ext = extractor();
        let md = "\
# Alpha
body a
## Beta
body b
### Gamma
body g
";
        let spans = ext.section_spans("t1", "doc.md", md);
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].start_line, 1);
        assert_eq!(spans[0].end_line, 2);
        assert_eq!(spans[1].start_line, 3);
        assert_eq!(spans[1].end_line, 4);
        assert_eq!(spans[2].start_line, 5);
        assert_eq!(spans[2].end_line, 6);
    }

    #[test]
    fn test_section_spans_ids_match_extract_markdown() {
        let ext = extractor();
        let md = "\
## Introduction
Some intro text.

## Background
Background info here.

## Methods
Methodology described.
";
        let extracted = run_extract(&ext, "tenant1", "doc.md", md);
        let spans = ext.section_spans("tenant1", "doc.md", md);

        assert_eq!(extracted.nodes.len(), spans.len());
        for (node, span) in extracted.nodes.iter().zip(spans.iter()) {
            assert_eq!(node.node_id, span.node_id);
            assert_eq!(node.start_line, Some(span.start_line));
            assert_eq!(node.end_line, Some(span.end_line));
        }
    }

    #[test]
    fn test_section_spans_plain_text() {
        let ext = extractor();
        let txt = "\
First title
body one.

Second title
body two.
";
        let spans = ext.section_spans("t1", "readme.txt", txt);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].start_line, 1);
        assert_eq!(spans[1].start_line, 4);
    }

    #[test]
    fn test_section_spans_ids_match_extract_text() {
        let ext = extractor();
        let txt = "\
Alpha title
body.

Beta title
more body.
";
        let extracted = run_extract(&ext, "t1", "notes.txt", txt);
        let spans = ext.section_spans("t1", "notes.txt", txt);

        assert_eq!(extracted.nodes.len(), spans.len());
        for (node, span) in extracted.nodes.iter().zip(spans.iter()) {
            assert_eq!(node.node_id, span.node_id);
            assert_eq!(node.start_line, Some(span.start_line));
            assert_eq!(node.end_line, Some(span.end_line));
        }
    }

    #[test]
    fn test_section_spans_edge_cases() {
        let ext = extractor();
        // No headings.
        assert!(ext
            .section_spans("t1", "x.md", "no headings here\n")
            .is_empty());
        // Single heading.
        let single = ext.section_spans("t1", "x.md", "# Only\nbody\n");
        assert_eq!(single.len(), 1);
        // Empty content.
        assert!(ext.section_spans("t1", "x.md", "").is_empty());
    }

    #[test]
    fn test_section_spans_unsupported_extension() {
        let ext = extractor();
        assert!(ext
            .section_spans("t1", "code.rs", "# Heading\nbody\n")
            .is_empty());
        assert!(ext.section_spans("t1", "data.json", "{}").is_empty());
    }
}

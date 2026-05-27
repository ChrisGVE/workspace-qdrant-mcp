/// EXPLAINS edge creation via Aho-Corasick symbol-reference scanning.
///
/// For narrative files (.md, .txt), scans content for code-like identifiers
/// (backtick-wrapped references, snake_case, CamelCase) and creates EXPLAINS
/// edges from DocumentSection nodes to stub Function nodes for each symbol
/// mentioned 2+ times.
use std::collections::HashMap;
use std::path::Path;

use aho_corasick::AhoCorasick;
use async_trait::async_trait;
use regex::Regex;

use crate::graph::{
    compute_node_id_for_type, EdgeType, GraphEdge, GraphNode, NodeIdFields, NodeType,
};

use super::{NarrativeExtractionResult, NarrativeExtractor};

/// Maximum EXPLAINS edges per extraction.
const MAX_EXPLAINS_EDGES: usize = 10;

/// Minimum occurrence count for a symbol to produce an edge.
const MIN_OCCURRENCES: usize = 2;

/// Minimum length for candidate symbols (shorter ones are too noisy).
const MIN_SYMBOL_LEN: usize = 3;

/// Words filtered out even when they look like code identifiers.
const STOP_LIST: &[&str] = &[
    "self", "impl", "test", "main", "init", "drop", "send", "sync", "read", "from", "into", "next",
    "iter", "push", "poll", "copy", "move", "loop", "data", "name", "type", "path", "node", "file",
    "list", "true", "None", "Some", "This", "that", "this", "will", "with", "have", "been", "also",
    "when", "then", "each", "used", "only", "more", "than", "both", "most", "some",
];

/// Extracts EXPLAINS edges by scanning narrative files for code symbol references.
pub struct ExplainsExtractor {
    /// Matches backtick-wrapped inline code spans: `identifier`.
    backtick_re: Regex,
    /// Matches snake_case identifiers: `foo_bar`, `my_func_name`.
    snake_case_re: Regex,
    /// Matches CamelCase identifiers: `MyStruct`, `HttpServer`.
    camel_case_re: Regex,
    /// Matches markdown ATX headings for section boundary detection.
    heading_re: Regex,
}

impl ExplainsExtractor {
    pub fn new() -> Self {
        Self {
            backtick_re: Regex::new(r"`([a-zA-Z_][a-zA-Z0-9_]{2,})`")
                .expect("backtick regex is valid"),
            snake_case_re: Regex::new(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b")
                .expect("snake_case regex is valid"),
            camel_case_re: Regex::new(r"\b([A-Z][a-z]+(?:[A-Z][a-zA-Z]*))\b")
                .expect("camel_case regex is valid"),
            heading_re: Regex::new(r"^#{1,6}\s+(.+)$").expect("heading regex is valid"),
        }
    }
}

impl Default for ExplainsExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Check whether the file is a narrative document we should process.
fn is_narrative_file(path: &Path) -> bool {
    let ext = match path.extension().and_then(|e| e.to_str()) {
        Some(e) => e.to_ascii_lowercase(),
        None => return false,
    };
    matches!(ext.as_str(), "md" | "markdown" | "txt")
}

/// Extract candidate code symbols from content using regex patterns.
fn extract_candidate_symbols(
    content: &str,
    backtick_re: &Regex,
    snake_case_re: &Regex,
    camel_case_re: &Regex,
) -> Vec<String> {
    let mut candidates: HashMap<String, ()> = HashMap::new();

    // Backtick-wrapped code references (highest signal).
    for caps in backtick_re.captures_iter(content) {
        if let Some(m) = caps.get(1) {
            candidates.insert(m.as_str().to_string(), ());
        }
    }

    // snake_case identifiers in prose.
    for caps in snake_case_re.captures_iter(content) {
        if let Some(m) = caps.get(1) {
            candidates.insert(m.as_str().to_string(), ());
        }
    }

    // CamelCase identifiers in prose.
    for caps in camel_case_re.captures_iter(content) {
        if let Some(m) = caps.get(1) {
            candidates.insert(m.as_str().to_string(), ());
        }
    }

    candidates.into_keys().collect()
}

/// Count occurrences of each candidate in content using Aho-Corasick.
fn count_symbol_occurrences(content: &str, candidates: &[String]) -> HashMap<String, usize> {
    if candidates.is_empty() {
        return HashMap::new();
    }

    let ac = match AhoCorasick::new(candidates) {
        Ok(ac) => ac,
        Err(_) => return HashMap::new(),
    };

    let mut counts: HashMap<String, usize> = HashMap::new();
    for mat in ac.find_iter(content) {
        let pattern = &candidates[mat.pattern().as_usize()];
        *counts.entry(pattern.clone()).or_insert(0) += 1;
    }

    counts
}

/// Filter symbols by stop-list and minimum occurrence count, returning
/// top symbols sorted by frequency (descending), capped at `max`.
fn filter_and_rank(
    counts: HashMap<String, usize>,
    min_count: usize,
    max: usize,
) -> Vec<(String, usize)> {
    let stop_set: std::collections::HashSet<&str> = STOP_LIST.iter().copied().collect();

    let mut filtered: Vec<(String, usize)> = counts
        .into_iter()
        .filter(|(sym, count)| {
            *count >= min_count && sym.len() >= MIN_SYMBOL_LEN && !stop_set.contains(sym.as_str())
        })
        .collect();

    // Sort by frequency descending, then alphabetically for determinism.
    filtered.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    filtered.truncate(max);
    filtered
}

/// Extract heading-delimited sections from markdown content.
/// Returns (heading_text, section_index, start_line_1indexed) tuples.
fn extract_section_boundaries(content: &str, heading_re: &Regex) -> Vec<(String, u32, u32)> {
    let mut sections = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if let Some(caps) = heading_re.captures(line) {
            let text = caps.get(1).map_or("", |m| m.as_str()).trim().to_string();
            if !text.is_empty() {
                let idx = sections.len() as u32;
                sections.push((text, idx, (i + 1) as u32));
            }
        }
    }
    sections
}

#[async_trait]
impl NarrativeExtractor for ExplainsExtractor {
    fn supported_node_types(&self) -> &[NodeType] {
        &[NodeType::DocumentSection]
    }

    async fn extract(
        &self,
        tenant_id: &str,
        file_path: &Path,
        content: &str,
        _language: Option<&str>,
    ) -> NarrativeExtractionResult {
        if !is_narrative_file(file_path) {
            return NarrativeExtractionResult::default();
        }

        if content.is_empty() {
            return NarrativeExtractionResult::default();
        }

        // Step 1: Extract candidate code symbols via regex.
        let candidates = extract_candidate_symbols(
            content,
            &self.backtick_re,
            &self.snake_case_re,
            &self.camel_case_re,
        );

        if candidates.is_empty() {
            return NarrativeExtractionResult::default();
        }

        // Step 2: Count occurrences via Aho-Corasick.
        let counts = count_symbol_occurrences(content, &candidates);

        // Step 3: Filter and rank.
        let top_symbols = filter_and_rank(counts, MIN_OCCURRENCES, MAX_EXPLAINS_EDGES);

        if top_symbols.is_empty() {
            return NarrativeExtractionResult::default();
        }

        let fp = file_path.to_string_lossy();

        // Step 4: Create a single DocumentSection node representing the whole file.
        // If the file has markdown headings, use the first heading; otherwise
        // use the filename stem as section name.
        let sections = extract_section_boundaries(content, &self.heading_re);

        let section_name = sections
            .first()
            .map(|(name, _, _)| name.clone())
            .unwrap_or_else(|| {
                file_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("document")
                    .to_string()
            });

        let section_fields = NodeIdFields {
            tenant_id,
            file_path: &fp,
            symbol_name: &section_name,
            symbol_type: NodeType::DocumentSection,
            section_index: Some(0),
            start_line: Some(1),
            library_name: None,
        };
        let section_node_id = compute_node_id_for_type(&section_fields);

        let mut section_node = GraphNode::new(
            tenant_id,
            fp.as_ref(),
            &section_name,
            NodeType::DocumentSection,
        );
        section_node.node_id = section_node_id.clone();
        section_node.start_line = Some(1);
        section_node.end_line = Some(content.lines().count().max(1) as u32);

        let mut nodes = vec![section_node];
        let mut edges = Vec::with_capacity(top_symbols.len());

        // Step 5: Create stub Function nodes and EXPLAINS edges.
        for (symbol, _count) in &top_symbols {
            let stub = GraphNode::stub(tenant_id, symbol.as_str(), NodeType::Function);
            let edge = GraphEdge::new(
                tenant_id,
                &section_node_id,
                &stub.node_id,
                EdgeType::Explains,
                fp.as_ref(),
            );
            nodes.push(stub);
            edges.push(edge);
        }

        NarrativeExtractionResult { nodes, edges }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extractor() -> ExplainsExtractor {
        ExplainsExtractor::new()
    }

    fn run_extract(
        ext: &ExplainsExtractor,
        tenant: &str,
        path: &str,
        content: &str,
    ) -> NarrativeExtractionResult {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(ext.extract(tenant, Path::new(path), content, None))
    }

    #[test]
    fn test_markdown_with_repeated_symbol_creates_explains_edge() {
        let ext = extractor();
        let md = "\
# Authentication

The `validate_token` function checks JWT tokens.
When `validate_token` encounters an expired token, it returns an error.
Call `validate_token` before accessing protected resources.
";
        let result = run_extract(&ext, "t1", "auth.md", md);

        // Should have at least one EXPLAINS edge for validate_token.
        assert!(
            !result.edges.is_empty(),
            "expected EXPLAINS edges but got none"
        );

        let edge_targets: Vec<&str> = result
            .nodes
            .iter()
            .filter(|n| n.symbol_type == NodeType::Function)
            .map(|n| n.symbol_name.as_str())
            .collect();
        assert!(
            edge_targets.contains(&"validate_token"),
            "expected validate_token stub, got: {:?}",
            edge_targets
        );

        // All edges should be EXPLAINS type.
        for edge in &result.edges {
            assert_eq!(edge.edge_type, EdgeType::Explains);
        }
    }

    #[test]
    fn test_stop_list_words_produce_no_edges() {
        let ext = extractor();
        // "self" and "test" are in the stop list. Repeat them many times.
        let md = "\
# Stop Words

self self self self self self self self
test test test test test test test test
impl impl impl impl impl impl impl impl
";
        let result = run_extract(&ext, "t1", "stop.md", md);

        // No edges should be created for stop-list words.
        let stub_names: Vec<&str> = result
            .nodes
            .iter()
            .filter(|n| n.symbol_type == NodeType::Function)
            .map(|n| n.symbol_name.as_str())
            .collect();
        for name in &stub_names {
            assert!(
                !STOP_LIST.contains(name),
                "stop-list word '{}' should not produce a stub node",
                name
            );
        }
    }

    #[test]
    fn test_single_mention_produces_no_edge() {
        let ext = extractor();
        let md = "\
# Single mention

The `parse_config` function is important.
";
        let result = run_extract(&ext, "t1", "single.md", md);

        let stub_names: Vec<&str> = result
            .nodes
            .iter()
            .filter(|n| n.symbol_type == NodeType::Function)
            .map(|n| n.symbol_name.as_str())
            .collect();
        assert!(
            !stub_names.contains(&"parse_config"),
            "single-mention symbol should not produce a stub"
        );
    }

    #[test]
    fn test_cap_at_ten_edges() {
        let ext = extractor();

        // Create 15 distinct snake_case symbols, each mentioned 3+ times.
        let symbols: Vec<String> = (0..15).map(|i| format!("symbol_func_{}", i)).collect();
        let mut md = String::from("# Many Symbols\n\n");
        for sym in &symbols {
            // Mention each symbol 3 times.
            md.push_str(&format!(
                "Use `{}` here. Also `{}` there. Again `{}`.\n",
                sym, sym, sym
            ));
        }

        let result = run_extract(&ext, "t1", "many.md", &md);

        assert!(
            result.edges.len() <= MAX_EXPLAINS_EDGES,
            "expected at most {} edges, got {}",
            MAX_EXPLAINS_EDGES,
            result.edges.len()
        );
    }

    #[test]
    fn test_non_markdown_file_returns_empty() {
        let ext = extractor();
        let content = "fn validate_token() {} // validate_token validate_token";
        let result = run_extract(&ext, "t1", "code.rs", content);
        assert!(result.is_empty(), "non-markdown should return empty result");
    }

    #[test]
    fn test_snake_case_without_backticks() {
        let ext = extractor();
        let md = "\
# Config

The parse_config function reads YAML files.
When parse_config fails, it logs an error.
Always call parse_config at startup.
";
        let result = run_extract(&ext, "t1", "config.md", md);

        let stub_names: Vec<&str> = result
            .nodes
            .iter()
            .filter(|n| n.symbol_type == NodeType::Function)
            .map(|n| n.symbol_name.as_str())
            .collect();
        assert!(
            stub_names.contains(&"parse_config"),
            "snake_case symbol should be detected without backticks"
        );
    }

    #[test]
    fn test_camel_case_detection() {
        let ext = extractor();
        let md = "\
# Server

The HttpServer handles requests.
Configure HttpServer with proper timeouts.
Restart HttpServer after config changes.
";
        let result = run_extract(&ext, "t1", "server.md", md);

        let stub_names: Vec<&str> = result
            .nodes
            .iter()
            .filter(|n| n.symbol_type == NodeType::Function)
            .map(|n| n.symbol_name.as_str())
            .collect();
        assert!(
            stub_names.contains(&"HttpServer"),
            "CamelCase symbol should be detected, got: {:?}",
            stub_names
        );
    }

    #[test]
    fn test_empty_content_returns_empty() {
        let ext = extractor();
        let result = run_extract(&ext, "t1", "empty.md", "");
        assert!(result.is_empty());
    }

    #[test]
    fn test_txt_file_is_narrative() {
        let ext = extractor();
        let txt = "\
validate_token is the core function.
Call validate_token before any request.
The validate_token function ensures auth.
";
        let result = run_extract(&ext, "t1", "notes.txt", txt);
        assert!(
            !result.edges.is_empty(),
            "txt files should be processed as narrative"
        );
    }

    #[test]
    fn test_section_node_has_correct_type() {
        let ext = extractor();
        let md = "\
# My Section

The `my_func_name` does things.
Call `my_func_name` to start.
Also `my_func_name` for cleanup.
";
        let result = run_extract(&ext, "t1", "section.md", md);

        let section_nodes: Vec<&GraphNode> = result
            .nodes
            .iter()
            .filter(|n| n.symbol_type == NodeType::DocumentSection)
            .collect();
        assert_eq!(section_nodes.len(), 1);
        assert_eq!(section_nodes[0].symbol_name, "My Section");
    }

    #[test]
    fn test_edge_source_file_matches_input() {
        let ext = extractor();
        let md = "\
# Edges

Use `process_data` for processing.
The `process_data` function is efficient.
Call `process_data` often.
";
        let result = run_extract(&ext, "t1", "edges.md", md);

        for edge in &result.edges {
            assert_eq!(edge.source_file, "edges.md");
        }
    }

    #[test]
    fn test_deterministic_node_ids() {
        let ext = extractor();
        let md = "\
# Determinism

Call `check_auth` twice.
Also `check_auth` here.
";
        let r1 = run_extract(&ext, "t1", "det.md", md);
        let r2 = run_extract(&ext, "t1", "det.md", md);

        assert_eq!(r1.nodes.len(), r2.nodes.len());
        assert_eq!(r1.edges.len(), r2.edges.len());

        for (n1, n2) in r1.nodes.iter().zip(r2.nodes.iter()) {
            assert_eq!(n1.node_id, n2.node_id);
        }
        for (e1, e2) in r1.edges.iter().zip(r2.edges.iter()) {
            assert_eq!(e1.edge_id, e2.edge_id);
        }
    }
}

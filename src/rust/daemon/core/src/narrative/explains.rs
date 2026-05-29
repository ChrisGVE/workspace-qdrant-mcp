//! EXPLAINS edge creation resolving REAL code-graph symbols.
//!
//! For narrative files (`.md`, `.txt`), scans the document for references to a
//! tenant's REAL code symbols (via a pre-built [`SymbolAutomaton`]) and emits
//! EXPLAINS edges from the containing [`SectionSpan`] to the resolved code node.
//!
//! A reference produces an edge only when:
//! * the symbol name resolves to exactly one code node (ambiguous → dropped);
//! * the name is at least `explains_min_symbol_length` chars;
//! * it occurs at least twice within a section;
//! * it is not a language stop word;
//! * the section has not already reached `explains_max_per_section` edges.
//!
//! No stub `Function` nodes are ever created — unresolved or ambiguous
//! references are simply skipped, so EXPLAINS targets are always real nodes.

use std::collections::HashMap;
use std::path::Path;

use async_trait::async_trait;

use crate::config::NarrativeConfig;
use crate::graph::{EdgeType, GraphEdge, NodeType};

use super::sections::SectionSpan;
use super::symbol_index::SymbolAutomaton;
use super::{NarrativeExtractionResult, NarrativeExtractor};

/// Words filtered out even when they look like code identifiers.
const STOP_LIST: &[&str] = &[
    "self", "impl", "test", "main", "init", "drop", "send", "sync", "read", "from", "into", "next",
    "iter", "push", "poll", "copy", "move", "loop", "data", "name", "type", "path", "node", "file",
    "list", "true", "none", "some", "this", "that", "will", "with", "have", "been", "also", "when",
    "then", "each", "used", "only", "more", "than", "both", "most", "string", "result", "option",
    "error", "value", "index",
];

/// Extracts EXPLAINS edges by resolving documentation references to real code
/// symbols within their containing document section.
///
/// Construct with [`ExplainsExtractor::with_context`] to supply the section
/// spans (from [`SectionExtractor::section_spans`](super::sections::SectionExtractor::section_spans))
/// and the tenant's [`SymbolAutomaton`]. The bare [`ExplainsExtractor::new`] /
/// [`Default`] constructor yields a no-context extractor that emits nothing —
/// retained so legacy unit tests can assert the empty-fallback behaviour.
pub struct ExplainsExtractor {
    section_spans: Vec<SectionSpan>,
    symbol_automaton: SymbolAutomaton,
    config: NarrativeConfig,
}

impl ExplainsExtractor {
    /// Create a no-context extractor that produces no edges.
    pub fn new() -> Self {
        Self {
            section_spans: Vec::new(),
            symbol_automaton: SymbolAutomaton::empty(),
            config: NarrativeConfig::default(),
        }
    }

    /// Create an extractor with the inter-extractor context required to emit
    /// EXPLAINS edges: the document's canonical section spans and the tenant's
    /// symbol automaton.
    pub fn with_context(
        section_spans: Vec<SectionSpan>,
        symbol_automaton: SymbolAutomaton,
        config: NarrativeConfig,
    ) -> Self {
        Self {
            section_spans,
            symbol_automaton,
            config,
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

/// Map a byte offset in `content` to its 1-indexed line number.
fn line_at_offset(content: &str, offset: usize) -> u32 {
    // Count newlines before the offset; line numbers are 1-indexed.
    let upto = offset.min(content.len());
    (content[..upto].bytes().filter(|&b| b == b'\n').count() + 1) as u32
}

/// Find the section span whose line range contains `line`.
fn section_for_line<'a>(spans: &'a [SectionSpan], line: u32) -> Option<&'a SectionSpan> {
    spans
        .iter()
        .find(|s| line >= s.start_line && line <= s.end_line)
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
        if content.is_empty() || self.section_spans.is_empty() || self.symbol_automaton.is_empty() {
            return NarrativeExtractionResult::default();
        }
        // Input-size cap (PERF-6): skip pathological files entirely.
        if content.len() > self.config.max_input_bytes() {
            tracing::debug!(
                "ExplainsExtractor: skipping {} ({} bytes > {} cap)",
                file_path.display(),
                content.len(),
                self.config.max_input_bytes()
            );
            return NarrativeExtractionResult::default();
        }

        let min_len = self.config.explains_min_symbol_length;
        let max_per_section = self.config.explains_max_per_section;
        let stop_set: std::collections::HashSet<&str> = STOP_LIST.iter().copied().collect();

        let fp = file_path.to_string_lossy();

        // For every symbol match, tally occurrences keyed by (section_id, symbol).
        // counts: section_node_id -> symbol_name -> occurrence count.
        let mut counts: HashMap<String, HashMap<String, usize>> = HashMap::new();

        for (symbol, offset) in self.symbol_automaton.find_matches(content) {
            if symbol.len() < min_len {
                continue;
            }
            if stop_set.contains(symbol.to_ascii_lowercase().as_str()) {
                continue;
            }
            let line = line_at_offset(content, offset);
            let Some(section) = section_for_line(&self.section_spans, line) else {
                continue; // match falls outside any section (e.g. preamble)
            };
            *counts
                .entry(section.node_id.clone())
                .or_default()
                .entry(symbol)
                .or_insert(0) += 1;
        }

        let mut edges = Vec::new();

        // Process sections in document order for deterministic output.
        for section in &self.section_spans {
            let Some(symbol_counts) = counts.get(&section.node_id) else {
                continue;
            };
            // Rank symbols by frequency desc, then name asc, applying the
            // min-occurrence (>=2) filter and resolving to a unique node id.
            let mut ranked: Vec<(&String, usize)> = symbol_counts
                .iter()
                .filter(|(_, &c)| c >= 2)
                .map(|(s, &c)| (s, c))
                .collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

            let mut emitted = 0usize;
            for (symbol, _count) in ranked {
                if emitted >= max_per_section {
                    break;
                }
                let Some(target_node_id) = self.symbol_automaton.resolve_unique(symbol) else {
                    continue; // unknown or ambiguous → drop, never stub
                };
                let edge = GraphEdge::new(
                    tenant_id,
                    section.node_id.clone(),
                    target_node_id.to_string(),
                    EdgeType::Explains,
                    fp.as_ref(),
                );
                edges.push(edge);
                emitted += 1;
            }
        }

        NarrativeExtractionResult {
            nodes: Vec::new(),
            edges,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::SymbolRow;

    fn sym(name: &str, file: &str) -> SymbolRow {
        SymbolRow {
            symbol_name: name.to_string(),
            node_id: format!("node:{file}:{name}"),
            file_path: file.to_string(),
        }
    }

    fn run(
        ext: &ExplainsExtractor,
        tenant: &str,
        path: &str,
        content: &str,
    ) -> NarrativeExtractionResult {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(ext.extract(tenant, Path::new(path), content, None))
    }

    fn ctx_extractor(spans: Vec<SectionSpan>, symbols: &[SymbolRow]) -> ExplainsExtractor {
        let auto = SymbolAutomaton::build(symbols, 4);
        ExplainsExtractor::with_context(spans, auto, NarrativeConfig::default())
    }

    fn span(node_id: &str, start: u32, end: u32) -> SectionSpan {
        SectionSpan {
            node_id: node_id.to_string(),
            start_line: start,
            end_line: end,
        }
    }

    #[test]
    fn no_context_extractor_emits_nothing() {
        let ext = ExplainsExtractor::new();
        let md = "# Auth\nThe validate_token validate_token function.\n";
        let result = run(&ext, "t1", "auth.md", md);
        assert!(result.is_empty(), "no-context extractor must emit nothing");
    }

    #[test]
    fn resolves_real_symbol_to_real_node_id() {
        let md = "\
# Authentication
The validate_token function checks tokens.
Call validate_token before access.
";
        let spans = vec![span("section-auth", 1, 3)];
        let symbols = vec![sym("validate_token", "auth.rs")];
        let ext = ctx_extractor(spans, &symbols);
        let result = run(&ext, "t1", "auth.md", md);

        assert_eq!(result.nodes.len(), 0, "no stub nodes are ever created");
        assert_eq!(result.edges.len(), 1);
        let edge = &result.edges[0];
        assert_eq!(edge.edge_type, EdgeType::Explains);
        assert_eq!(edge.source_node_id, "section-auth");
        assert_eq!(edge.target_node_id, "node:auth.rs:validate_token");
    }

    #[test]
    fn ambiguous_symbol_drops_edge() {
        let md = "\
# Handlers
The request_handler is called twice.
Always invoke request_handler safely.
";
        let spans = vec![span("section-h", 1, 3)];
        // Two distinct nodes share the name → ambiguous → dropped.
        let symbols = vec![
            sym("request_handler", "a.rs"),
            sym("request_handler", "b.rs"),
        ];
        let ext = ctx_extractor(spans, &symbols);
        let result = run(&ext, "t1", "h.md", md);
        assert!(
            result.edges.is_empty(),
            "ambiguous symbol must not produce an edge"
        );
    }

    #[test]
    fn single_occurrence_drops_edge() {
        let md = "# Config\nThe parse_config function matters.\n";
        let spans = vec![span("section-c", 1, 2)];
        let symbols = vec![sym("parse_config", "c.rs")];
        let ext = ctx_extractor(spans, &symbols);
        let result = run(&ext, "t1", "c.md", md);
        assert!(
            result.edges.is_empty(),
            "single mention must not produce an edge"
        );
    }

    #[test]
    fn unknown_symbol_no_edge() {
        let md = "# X\ntotally_unknown_symbol totally_unknown_symbol here.\n";
        let spans = vec![span("section-x", 1, 2)];
        let symbols = vec![sym("validate_token", "auth.rs")];
        let ext = ctx_extractor(spans, &symbols);
        let result = run(&ext, "t1", "x.md", md);
        assert!(result.edges.is_empty());
    }

    #[test]
    fn edge_attaches_to_containing_section() {
        let md = "\
# First
parse_config parse_config here.
# Second
validate_token validate_token there.
";
        let spans = vec![span("sec-1", 1, 2), span("sec-2", 3, 4)];
        let symbols = vec![sym("parse_config", "c.rs"), sym("validate_token", "a.rs")];
        let ext = ctx_extractor(spans, &symbols);
        let result = run(&ext, "t1", "doc.md", md);

        assert_eq!(result.edges.len(), 2);
        let by_target: HashMap<&str, &str> = result
            .edges
            .iter()
            .map(|e| (e.target_node_id.as_str(), e.source_node_id.as_str()))
            .collect();
        assert_eq!(by_target["node:c.rs:parse_config"], "sec-1");
        assert_eq!(by_target["node:a.rs:validate_token"], "sec-2");
    }

    #[test]
    fn max_per_section_cap_enforced() {
        // 15 distinct symbols each mentioned 3x in one section; cap is the
        // default 10.
        let mut md = String::from("# Many\n");
        let mut symbols = Vec::new();
        for i in 0..15 {
            let name = format!("symbol_func_{i:02}");
            md.push_str(&format!("{name} {name} {name}\n"));
            symbols.push(sym(&name, "x.rs"));
        }
        let line_count = md.lines().count() as u32;
        let spans = vec![span("sec", 1, line_count)];
        let ext = ctx_extractor(spans, &symbols);
        let result = run(&ext, "t1", "many.md", &md);
        assert!(
            result.edges.len() <= 10,
            "expected <= 10 edges, got {}",
            result.edges.len()
        );
    }

    #[test]
    fn min_length_filter_via_automaton() {
        // `io` is below min length 4 → never in the automaton → no edge.
        let md = "# IO\nio io io io\n";
        let spans = vec![span("sec", 1, 2)];
        let symbols = vec![sym("io", "io.rs")];
        let ext = ctx_extractor(spans, &symbols);
        let result = run(&ext, "t1", "io.md", md);
        assert!(result.edges.is_empty());
    }

    #[test]
    fn stop_word_dropped() {
        // `result` is a stop word even though it resolves.
        let md = "# R\nresult result result\n";
        let spans = vec![span("sec", 1, 2)];
        let symbols = vec![sym("result", "r.rs")];
        let ext = ctx_extractor(spans, &symbols);
        let res = run(&ext, "t1", "r.md", md);
        assert!(res.edges.is_empty());
    }

    #[test]
    fn non_narrative_file_empty() {
        let spans = vec![span("sec", 1, 2)];
        let symbols = vec![sym("validate_token", "a.rs")];
        let ext = ctx_extractor(spans, &symbols);
        let result = run(&ext, "t1", "code.rs", "validate_token validate_token");
        assert!(result.is_empty());
    }

    #[test]
    fn input_size_cap_skips_large_files() {
        let spans = vec![span("sec", 1, 999_999)];
        let symbols = vec![sym("parse_config", "c.rs")];
        let auto = SymbolAutomaton::build(&symbols, 4);
        let cfg = NarrativeConfig {
            max_input_kb: 1,
            ..NarrativeConfig::default()
        };
        let ext = ExplainsExtractor::with_context(spans, auto, cfg);
        // 2 KB of content exceeds the 1 KB cap.
        let mut md = String::from("# Big\n");
        while md.len() < 2048 {
            md.push_str("parse_config parse_config filler text line\n");
        }
        let result = run(&ext, "t1", "big.md", &md);
        assert!(result.is_empty(), "file over cap must be skipped");
    }

    #[test]
    fn line_at_offset_basic() {
        let content = "line1\nline2\nline3\n";
        assert_eq!(line_at_offset(content, 0), 1);
        assert_eq!(line_at_offset(content, 6), 2);
        assert_eq!(line_at_offset(content, 12), 3);
    }

    #[test]
    fn section_for_line_lookup() {
        let spans = vec![span("a", 1, 5), span("b", 6, 10)];
        assert_eq!(section_for_line(&spans, 3).unwrap().node_id, "a");
        assert_eq!(section_for_line(&spans, 6).unwrap().node_id, "b");
        assert!(section_for_line(&spans, 20).is_none());
    }
}

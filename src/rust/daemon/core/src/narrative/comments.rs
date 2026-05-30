/// CodeComment and Docstring extraction from source code.
///
/// Parses source files line-by-line to identify contiguous comment blocks
/// (3+ lines minimum). Each qualifying block becomes a `CodeComment` graph
/// node. When a function/method signature appears within 5 lines after the
/// block AND that symbol resolves to exactly one real code-graph node (via the
/// tenant symbol automaton), an `Explains` edge links the comment to that real
/// node. Unknown or ambiguous symbols are dropped — never stubbed — so the
/// narrative layer carries no dangling EXPLAINS targets.
use std::path::Path;

use async_trait::async_trait;

use crate::graph::{
    compute_node_id_for_type, EdgeType, GraphEdge, GraphNode, NodeIdFields, NodeType,
};

use super::symbol_index::SymbolAutomaton;
use super::{NarrativeExtractionResult, NarrativeExtractor};

/// Minimum number of contiguous comment lines to form a `CodeComment` node.
const MIN_COMMENT_LINES: usize = 3;

/// Maximum distance (non-comment, non-blank lines) to search for a
/// function signature after a comment block ends.
const PROXIMITY_LINES: usize = 5;

/// Single-line comment prefix for a language family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommentPrefix {
    /// `//` — Rust, C, C++, Go, Java, JavaScript, TypeScript, Swift, Kotlin
    DoubleSlash,
    /// `#` — Python, Ruby, Shell, YAML, TOML
    Hash,
    /// `--` — Lua, SQL, Haskell
    DoubleDash,
}

impl CommentPrefix {
    fn as_str(self) -> &'static str {
        match self {
            CommentPrefix::DoubleSlash => "//",
            CommentPrefix::Hash => "#",
            CommentPrefix::DoubleDash => "--",
        }
    }
}

/// Return the comment prefix used by the given language, or `None` for
/// unsupported / unknown languages.
fn comment_prefix_for_language(language: Option<&str>) -> Option<CommentPrefix> {
    let lang = language?;
    match lang.to_ascii_lowercase().as_str() {
        "rust" | "c" | "cpp" | "c++" | "go" | "java" | "javascript" | "typescript" | "swift"
        | "kotlin" | "js" | "ts" | "jsx" | "tsx" => Some(CommentPrefix::DoubleSlash),
        "python" | "ruby" | "shell" | "bash" | "sh" | "zsh" | "yaml" | "yml" | "toml"
        | "dockerfile" | "makefile" | "perl" | "r" => Some(CommentPrefix::Hash),
        "lua" | "sql" | "haskell" | "hs" => Some(CommentPrefix::DoubleDash),
        _ => None,
    }
}

/// Check whether `trimmed_line` is a comment with the given prefix.
fn is_comment_line(trimmed: &str, prefix: &str) -> bool {
    trimmed.starts_with(prefix)
}

/// Strip the comment prefix and any single leading space from a trimmed line.
fn strip_comment_prefix<'a>(trimmed: &'a str, prefix: &str) -> &'a str {
    let after = &trimmed[prefix.len()..];
    after.strip_prefix(' ').unwrap_or(after)
}

/// A contiguous block of comment lines.
struct CommentBlock {
    /// 1-based start line number.
    start_line: u32,
    /// 1-based end line number (inclusive).
    end_line: u32,
    /// Comment text with prefixes stripped, lines joined by newline.
    text: String,
}

/// Collect contiguous comment blocks from `lines` using the given prefix.
fn collect_comment_blocks(lines: &[&str], prefix: &str) -> Vec<CommentBlock> {
    let mut blocks = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut current_texts: Vec<&str> = Vec::new();

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if is_comment_line(trimmed, prefix) {
            if current_start.is_none() {
                current_start = Some(idx);
            }
            current_texts.push(strip_comment_prefix(trimmed, prefix));
        } else {
            if let Some(start) = current_start.take() {
                if current_texts.len() >= MIN_COMMENT_LINES {
                    blocks.push(CommentBlock {
                        start_line: (start + 1) as u32,
                        end_line: (start + current_texts.len()) as u32,
                        text: current_texts.join("\n"),
                    });
                }
                current_texts.clear();
            }
        }
    }

    // Flush trailing block.
    if let Some(start) = current_start {
        if current_texts.len() >= MIN_COMMENT_LINES {
            blocks.push(CommentBlock {
                start_line: (start + 1) as u32,
                end_line: (start + current_texts.len()) as u32,
                text: current_texts.join("\n"),
            });
        }
    }

    blocks
}

/// Attempt to extract a function/method name from a source line.
///
/// Returns the identifier from common function-definition patterns across
/// the supported languages. The detection is deliberately simple — it is
/// used only for proximity linking, not for full parsing.
fn extract_symbol_name(line: &str) -> Option<&str> {
    let trimmed = line.trim();

    // Rust / Go / Swift / Kotlin: `fn name(`, `func name(`, `fun name(`
    for keyword in &["fn ", "func ", "fun "] {
        if let Some(rest) = trimmed.strip_prefix(keyword) {
            return ident_before_paren(rest);
        }
        // Also match with visibility: `pub fn`, `pub(crate) fn`, etc.
        if let Some(after_kw) = trimmed.find(keyword) {
            let rest = &trimmed[after_kw + keyword.len()..];
            return ident_before_paren(rest);
        }
    }

    // Python: `def name(`
    if let Some(rest) = trimmed.strip_prefix("def ") {
        return ident_before_paren(rest);
    }
    // `async def name(`
    if let Some(rest) = trimmed.strip_prefix("async def ") {
        return ident_before_paren(rest);
    }

    // JavaScript / TypeScript: `function name(`
    if let Some(rest) = trimmed.strip_prefix("function ") {
        return ident_before_paren(rest);
    }

    // C / C++ / Java: `<type> name(` — heuristic: last word before `(` is the name
    if let Some(paren_pos) = trimmed.find('(') {
        let before_paren = trimmed[..paren_pos].trim();
        if let Some(last_space) = before_paren.rfind(' ') {
            let candidate = &before_paren[last_space + 1..];
            if is_identifier(candidate) && !is_keyword(candidate) {
                return Some(candidate);
            }
        }
    }

    None
}

/// Return the identifier immediately before the first `(`.
fn ident_before_paren(s: &str) -> Option<&str> {
    // The string starts right after the keyword, e.g. "foo(bar)" or "foo<T>(bar)".
    // Take the identifier chars at the start.
    let end = s
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(s.len());
    let candidate = &s[..end];
    if candidate.is_empty() {
        None
    } else {
        Some(candidate)
    }
}

/// Check whether `s` looks like a valid identifier (ASCII letters, digits, `_`).
fn is_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && s.chars()
            .next()
            .map_or(false, |c| c.is_ascii_alphabetic() || c == '_')
}

/// Common keywords that should not be treated as function names.
fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "if" | "else"
            | "for"
            | "while"
            | "return"
            | "match"
            | "let"
            | "var"
            | "val"
            | "const"
            | "static"
            | "class"
            | "struct"
            | "enum"
            | "trait"
            | "impl"
            | "pub"
            | "super"
            | "self"
            | "new"
            | "void"
            | "int"
            | "bool"
            | "true"
            | "false"
            | "import"
            | "export"
            | "async"
            | "await"
            | "type"
            | "interface"
    )
}

/// Search up to `PROXIMITY_LINES` non-comment, non-blank lines after
/// `end_line_idx` (0-based exclusive end of the comment block) for a
/// function signature.
fn find_nearby_symbol<'a>(lines: &[&'a str], end_line_idx: usize, prefix: &str) -> Option<&'a str> {
    let mut inspected = 0;
    for line in lines.iter().skip(end_line_idx) {
        let trimmed = line.trim();
        if trimmed.is_empty() || is_comment_line(trimmed, prefix) {
            continue;
        }
        if let Some(name) = extract_symbol_name(trimmed) {
            return Some(name);
        }
        inspected += 1;
        if inspected >= PROXIMITY_LINES {
            break;
        }
    }
    None
}

/// Extracts `CodeComment` nodes and (when resolvable) real-target `Explains`
/// edges from source comments.
///
/// Construct with [`CommentExtractor::with_context`] to supply the tenant
/// symbol automaton used to resolve a comment's nearby symbol to a real
/// code-graph node id. [`CommentExtractor::new`] yields an extractor with an
/// empty automaton, which still emits `CodeComment` nodes but no EXPLAINS edges
/// (every symbol resolves to "unknown").
pub struct CommentExtractor {
    symbol_automaton: SymbolAutomaton,
}

impl CommentExtractor {
    /// Context-free extractor: emits `CodeComment` nodes only (no EXPLAINS).
    pub fn new() -> Self {
        Self {
            symbol_automaton: SymbolAutomaton::empty(),
        }
    }

    /// Extractor that resolves comment→symbol EXPLAINS edges against the
    /// tenant's real code symbols via `automaton`.
    pub fn with_context(automaton: SymbolAutomaton) -> Self {
        Self {
            symbol_automaton: automaton,
        }
    }
}

impl Default for CommentExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NarrativeExtractor for CommentExtractor {
    fn supported_node_types(&self) -> &[NodeType] {
        &[NodeType::CodeComment, NodeType::Docstring]
    }

    async fn extract(
        &self,
        tenant_id: &str,
        file_path: &Path,
        content: &str,
        language: Option<&str>,
    ) -> NarrativeExtractionResult {
        let prefix = match comment_prefix_for_language(language) {
            Some(p) => p,
            None => return NarrativeExtractionResult::default(),
        };

        let file_path_str = file_path.to_string_lossy();
        let lines: Vec<&str> = content.lines().collect();
        let blocks = collect_comment_blocks(&lines, prefix.as_str());

        let mut result = NarrativeExtractionResult::default();

        for block in &blocks {
            // Build a CodeComment node using the type-specific ID computation.
            let first_line_text = block.text.lines().next().unwrap_or("");
            let fields = NodeIdFields {
                tenant_id,
                file_path: &file_path_str,
                symbol_name: first_line_text,
                symbol_type: NodeType::CodeComment,
                section_index: None,
                start_line: Some(block.start_line),
                library_name: None,
            };
            let node_id = compute_node_id_for_type(&fields);

            let mut node = GraphNode::new(
                tenant_id,
                file_path_str.as_ref(),
                first_line_text,
                NodeType::CodeComment,
            );
            node.node_id = node_id.clone();
            node.start_line = Some(block.start_line);
            node.end_line = Some(block.end_line);
            node.language = language.map(String::from);

            result.nodes.push(node);

            // Link to a nearby symbol ONLY when it resolves to exactly one real
            // code-graph node. Unknown/ambiguous names are dropped (never
            // stubbed), so EXPLAINS targets are always genuine graph_nodes ids.
            let end_idx = block.end_line as usize; // end_line is 1-based, so this is the 0-based index past the block
            if let Some(symbol_name) = find_nearby_symbol(&lines, end_idx, prefix.as_str()) {
                if let Some(target_node_id) = self.symbol_automaton.resolve_unique(symbol_name) {
                    let edge = GraphEdge::new(
                        tenant_id,
                        &node_id,
                        target_node_id.to_string(),
                        EdgeType::Explains,
                        file_path_str.as_ref(),
                    );
                    result.edges.push(edge);
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn path(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    fn automaton_with(name: &str, node_id: &str) -> SymbolAutomaton {
        SymbolAutomaton::build(
            &[crate::graph::SymbolRow {
                symbol_name: name.to_string(),
                node_id: node_id.to_string(),
                file_path: "src/main.rs".to_string(),
            }],
            3,
        )
    }

    const RUST_COMMENT_FN: &str = "\
// This is a comment
// that spans multiple
// lines describing
// the foo function
fn foo() {
    println!(\"hello\");
}
";

    // 1a. Without context (empty automaton): CodeComment node only, NO EXPLAINS
    // edge and NO stub node — unresolved symbols are dropped, not stubbed.
    #[tokio::test]
    async fn rust_comment_block_no_context_drops_explains() {
        let extractor = CommentExtractor::new();
        let result = extractor
            .extract("t1", &path("src/main.rs"), RUST_COMMENT_FN, Some("rust"))
            .await;

        assert_eq!(result.nodes.len(), 1, "expected only the CodeComment node");
        assert_eq!(result.nodes[0].symbol_type, NodeType::CodeComment);
        assert!(
            result.edges.is_empty(),
            "no EXPLAINS edge without a resolvable real symbol (no stubs)"
        );
        // No node is a Function stub.
        assert!(result
            .nodes
            .iter()
            .all(|n| n.symbol_type != NodeType::Function));
    }

    // 1b. With context: the nearby `foo` resolves to a real code node, so a
    // single EXPLAINS edge targets that real node id (no stub node emitted).
    #[tokio::test]
    async fn rust_comment_block_resolves_real_symbol() {
        let extractor = CommentExtractor::with_context(automaton_with("foo", "real-foo-node"));
        let result = extractor
            .extract("t1", &path("src/main.rs"), RUST_COMMENT_FN, Some("rust"))
            .await;

        assert_eq!(result.nodes.len(), 1, "only the CodeComment node, no stub");
        let comment_node = &result.nodes[0];
        assert_eq!(comment_node.symbol_type, NodeType::CodeComment);
        assert_eq!(comment_node.start_line, Some(1));
        assert_eq!(comment_node.end_line, Some(4));

        assert_eq!(result.edges.len(), 1, "expected 1 EXPLAINS edge");
        let edge = &result.edges[0];
        assert_eq!(edge.edge_type, EdgeType::Explains);
        assert_eq!(edge.source_node_id, comment_node.node_id);
        assert_eq!(
            edge.target_node_id, "real-foo-node",
            "EXPLAINS must target the real resolved node id"
        );
    }

    // 1c. With context but an ambiguous symbol (two nodes share the name): the
    // edge is dropped rather than guessing.
    #[tokio::test]
    async fn rust_comment_block_ambiguous_symbol_drops_edge() {
        let automaton = SymbolAutomaton::build(
            &[
                crate::graph::SymbolRow {
                    symbol_name: "foo".to_string(),
                    node_id: "foo-a".to_string(),
                    file_path: "a.rs".to_string(),
                },
                crate::graph::SymbolRow {
                    symbol_name: "foo".to_string(),
                    node_id: "foo-b".to_string(),
                    file_path: "b.rs".to_string(),
                },
            ],
            3,
        );
        let extractor = CommentExtractor::with_context(automaton);
        let result = extractor
            .extract("t1", &path("src/main.rs"), RUST_COMMENT_FN, Some("rust"))
            .await;
        assert_eq!(result.nodes.len(), 1);
        assert!(
            result.edges.is_empty(),
            "ambiguous symbol → drop the edge (never stub or guess)"
        );
    }

    // 2. Python file with 3-line `#` comment block -> 1 CodeComment node
    #[tokio::test]
    async fn python_comment_block_no_function() {
        let content = "\
# Configuration section
# sets up the database
# connection parameters
DATABASE_URL = \"sqlite:///db.sqlite\"
";
        let extractor = CommentExtractor::new();
        let result = extractor
            .extract("t1", &path("config.py"), content, Some("python"))
            .await;

        // DATABASE_URL = ... is not a function signature, so no edge
        assert_eq!(result.nodes.len(), 1, "expected 1 CodeComment node");
        assert_eq!(result.edges.len(), 0, "expected no edges");

        let node = &result.nodes[0];
        assert_eq!(node.symbol_type, NodeType::CodeComment);
        assert_eq!(node.start_line, Some(1));
        assert_eq!(node.end_line, Some(3));
    }

    // 3. File with only 2-line comment (below threshold) -> no nodes
    #[tokio::test]
    async fn two_line_comment_below_threshold() {
        let content = "\
// Short comment
// only two lines
fn bar() {}
";
        let extractor = CommentExtractor::new();
        let result = extractor
            .extract("t1", &path("src/lib.rs"), content, Some("rust"))
            .await;

        assert!(result.is_empty(), "2-line comment should not produce nodes");
    }

    // 4. Comment block >5 lines away from any function -> node but no edge
    #[tokio::test]
    async fn comment_block_far_from_function() {
        let content = "\
// This block is
// far away from
// any function definition
let x = 1;
let y = 2;
let z = 3;
let w = 4;
let q = 5;
fn distant() {}
";
        let extractor = CommentExtractor::new();
        let result = extractor
            .extract("t1", &path("src/far.rs"), content, Some("rust"))
            .await;

        assert_eq!(result.nodes.len(), 1, "expected 1 CodeComment node");
        assert_eq!(result.edges.len(), 0, "function is too far away — no edge");
    }

    // 5. Non-code file (None language) -> empty result
    #[tokio::test]
    async fn no_language_returns_empty() {
        let content = "Just some text\nwith no code\n";
        let extractor = CommentExtractor::new();
        let result = extractor
            .extract("t1", &path("notes.txt"), content, None)
            .await;

        assert!(result.is_empty(), "None language should produce no nodes");
    }

    // Additional: Python with `def` function after comment
    #[tokio::test]
    async fn python_comment_with_def() {
        let content = "\
# Compute the sum
# of two numbers
# and return result
def add(a, b):
    return a + b
";
        let extractor = CommentExtractor::with_context(automaton_with("add", "py-add"));
        let result = extractor
            .extract("t1", &path("math.py"), content, Some("python"))
            .await;

        // CodeComment node only (no stub); EXPLAINS targets the real `add` node.
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].symbol_type, NodeType::CodeComment);
        assert_eq!(result.edges.len(), 1);
        assert_eq!(result.edges[0].target_node_id, "py-add");
    }

    // Additional: Lua with `--` comments
    #[tokio::test]
    async fn lua_double_dash_comments() {
        let content = "\
-- Initialize the module
-- with default settings
-- and register handlers
function setup()
    print('ready')
end
";
        let extractor = CommentExtractor::with_context(automaton_with("setup", "lua-setup"));
        let result = extractor
            .extract("t1", &path("init.lua"), content, Some("lua"))
            .await;

        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].symbol_type, NodeType::CodeComment);
        assert_eq!(result.edges.len(), 1);
        assert_eq!(result.edges[0].target_node_id, "lua-setup");
    }

    // Additional: Unknown language returns empty
    #[tokio::test]
    async fn unknown_language_returns_empty() {
        let content = "// some comment\n// more\n// and more\nfn test() {}\n";
        let extractor = CommentExtractor::new();
        let result = extractor
            .extract("t1", &path("file.xyz"), content, Some("brainfuck"))
            .await;

        assert!(result.is_empty());
    }

    // Additional: Multiple comment blocks in one file
    #[tokio::test]
    async fn multiple_comment_blocks() {
        let content = "\
// First block
// of comments
// three lines
fn first() {}

// Second block
// also has
// three lines
fn second() {}
";
        let automaton = SymbolAutomaton::build(
            &[
                crate::graph::SymbolRow {
                    symbol_name: "first".to_string(),
                    node_id: "n-first".to_string(),
                    file_path: "src/multi.rs".to_string(),
                },
                crate::graph::SymbolRow {
                    symbol_name: "second".to_string(),
                    node_id: "n-second".to_string(),
                    file_path: "src/multi.rs".to_string(),
                },
            ],
            3,
        );
        let extractor = CommentExtractor::with_context(automaton);
        let result = extractor
            .extract("t1", &path("src/multi.rs"), content, Some("rust"))
            .await;

        // 2 CodeComment nodes (no stubs) + 2 EXPLAINS edges to the real symbols.
        assert_eq!(result.nodes.len(), 2);
        assert!(result
            .nodes
            .iter()
            .all(|n| n.symbol_type == NodeType::CodeComment));
        assert_eq!(result.edges.len(), 2);
        let targets: Vec<&str> = result
            .edges
            .iter()
            .map(|e| e.target_node_id.as_str())
            .collect();
        assert!(targets.contains(&"n-first") && targets.contains(&"n-second"));
    }

    // Unit tests for helper functions

    #[test]
    fn test_comment_prefix_for_language() {
        assert_eq!(
            comment_prefix_for_language(Some("rust")),
            Some(CommentPrefix::DoubleSlash)
        );
        assert_eq!(
            comment_prefix_for_language(Some("python")),
            Some(CommentPrefix::Hash)
        );
        assert_eq!(
            comment_prefix_for_language(Some("lua")),
            Some(CommentPrefix::DoubleDash)
        );
        assert_eq!(comment_prefix_for_language(None), None);
        assert_eq!(comment_prefix_for_language(Some("unknown")), None);
    }

    #[test]
    fn test_extract_symbol_name() {
        assert_eq!(extract_symbol_name("fn foo() {"), Some("foo"));
        assert_eq!(
            extract_symbol_name("pub fn bar(x: i32) -> bool {"),
            Some("bar")
        );
        assert_eq!(extract_symbol_name("pub(crate) fn baz() {"), Some("baz"));
        assert_eq!(extract_symbol_name("def hello(self):"), Some("hello"));
        assert_eq!(extract_symbol_name("async def run():"), Some("run"));
        assert_eq!(extract_symbol_name("function doStuff() {"), Some("doStuff"));
        assert_eq!(extract_symbol_name("func main() {"), Some("main"));
        assert_eq!(extract_symbol_name("fun create() {"), Some("create"));
        assert_eq!(extract_symbol_name("int compute(int x) {"), Some("compute"));
        assert_eq!(extract_symbol_name("let x = 5;"), None);
        assert_eq!(extract_symbol_name(""), None);
    }

    #[test]
    fn test_is_identifier() {
        assert!(is_identifier("foo"));
        assert!(is_identifier("_bar"));
        assert!(is_identifier("baz_42"));
        assert!(!is_identifier(""));
        assert!(!is_identifier("123abc"));
        assert!(!is_identifier("foo-bar"));
    }

    #[test]
    fn test_collect_comment_blocks() {
        let lines = vec![
            "// line one",
            "// line two",
            "// line three",
            "fn foo() {}",
            "// short",
            "// only two",
            "code here",
            "// another",
            "// block of",
            "// three lines",
            "// and four",
        ];
        let blocks = collect_comment_blocks(&lines, "//");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].start_line, 1);
        assert_eq!(blocks[0].end_line, 3);
        assert_eq!(blocks[1].start_line, 8);
        assert_eq!(blocks[1].end_line, 11);
    }
}

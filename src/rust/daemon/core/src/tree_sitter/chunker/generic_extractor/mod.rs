//! Generic pattern-driven semantic chunk extractor.
//!
//! Replaces per-language extractors with a single engine that reads AST node
//! type patterns from the YAML language registry. Given a `SemanticPatterns`
//! configuration, it walks the tree-sitter AST and extracts preamble,
//! functions, classes, methods, structs, enums, traits, modules, etc.

mod docstring;
mod walker;

use std::path::Path;

use tree_sitter::{Language, Node};

use crate::error::DaemonError;
use crate::language_registry::types::{DocstringStyle, SemanticPatterns};
use crate::tree_sitter::chunker::helpers::{extract_function_calls, find_child_by_kind, node_text};
use crate::tree_sitter::parser::TreeSitterParser;
use crate::tree_sitter::types::{ChunkExtractor, ChunkType, SemanticChunk};

/// A generic extractor that uses YAML-defined patterns to extract chunks.
pub struct GenericExtractor {
    language_name: String,
    language: Language,
    patterns: SemanticPatterns,
}

impl GenericExtractor {
    /// Create a new generic extractor for a language with its patterns.
    pub fn new(language_name: &str, language: Language, patterns: SemanticPatterns) -> Self {
        Self {
            language_name: language_name.to_string(),
            language,
            patterns,
        }
    }

    fn create_parser(&self) -> Result<TreeSitterParser, DaemonError> {
        TreeSitterParser::with_language(&self.language_name, self.language.clone())
    }

    // ── Preamble extraction ───────────────────────────────────────────

    /// Collect effective top-level children, unwrapping root_wrappers.
    fn effective_children<'a>(&self, root: &Node<'a>) -> Vec<Node<'a>> {
        let wrappers = &self.patterns.root_wrappers;
        let mut result = Vec::new();
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if child.is_named() && wrappers.iter().any(|w| w == child.kind()) {
                let mut inner = child.walk();
                for grandchild in child.children(&mut inner) {
                    if grandchild.is_named() {
                        result.push(grandchild);
                    }
                }
            } else {
                result.push(child);
            }
        }
        result
    }

    fn extract_preamble(
        &self,
        root: &Node,
        source: &str,
        file_path: &str,
    ) -> Option<SemanticChunk> {
        let preamble_types = &self.patterns.preamble.node_types;
        if preamble_types.is_empty() {
            return None;
        }

        let comment_types = &self.patterns.comment_nodes;
        let mut preamble_items = Vec::new();
        let mut last_preamble_line = 0;
        let children = self.effective_children(root);

        for child in &children {
            let kind = child.kind();

            if preamble_types.iter().any(|t| t == kind) {
                preamble_items.push(node_text(child, source).to_string());
                last_preamble_line = child.end_position().row + 1;
            } else if comment_types.iter().any(|t| t == kind) {
                // Include comments adjacent to preamble
                if preamble_items.is_empty() || child.start_position().row <= last_preamble_line + 1
                {
                    preamble_items.push(node_text(child, source).to_string());
                    last_preamble_line = child.end_position().row + 1;
                }
            } else if kind == "expression_statement"
                && self.patterns.docstring_style == DocstringStyle::FirstStringInBody
            {
                // Python-style: module docstring as first expression
                if preamble_items.is_empty() || last_preamble_line == 0 {
                    if find_child_by_kind(child, "string").is_some() {
                        preamble_items.push(node_text(child, source).to_string());
                        last_preamble_line = child.end_position().row + 1;
                        continue;
                    }
                }
                if !preamble_items.is_empty() {
                    break;
                }
            } else if kind == "shebang" {
                // Shell shebang lines
                preamble_items.push(node_text(child, source).to_string());
                last_preamble_line = child.end_position().row + 1;
            } else if kind == "call" {
                // Elixir-style: use/import/alias/require are call nodes
                let text = node_text(child, source);
                let first_word = text.split_whitespace().next().unwrap_or("");
                if preamble_types.iter().any(|t| t == first_word)
                    || matches!(first_word, "use" | "import" | "alias" | "require")
                        && preamble_types.iter().any(|t| t == "call")
                {
                    preamble_items.push(text.to_string());
                    last_preamble_line = child.end_position().row + 1;
                } else if !preamble_items.is_empty() {
                    break;
                }
            } else {
                if !preamble_items.is_empty() {
                    break;
                }
            }
        }

        if preamble_items.is_empty() {
            return None;
        }

        Some(SemanticChunk::preamble(
            preamble_items.join("\n"),
            last_preamble_line,
            &self.language_name,
            file_path,
        ))
    }

    // ── Definition extraction ─────────────────────────────────────────

    fn extract_definition(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        chunk_type: ChunkType,
        parent: Option<&str>,
    ) -> SemanticChunk {
        let name = self.extract_name(node, source);
        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let is_async = matches!(chunk_type, ChunkType::Function | ChunkType::Method)
            && self
                .patterns
                .function
                .async_node_types
                .iter()
                .any(|t| t == node.kind());

        let effective_type = if is_async && parent.is_none() {
            ChunkType::AsyncFunction
        } else if parent.is_some()
            && matches!(chunk_type, ChunkType::Function | ChunkType::AsyncFunction)
        {
            ChunkType::Method
        } else {
            chunk_type
        };

        let signature = content.lines().next().map(|l| {
            l.trim_end_matches(':')
                .trim_end_matches('{')
                .trim_end_matches("do")
                .trim()
                .to_string()
        });

        let docstring = docstring::extract_docstring(&self.patterns, node, source);

        let calls = if matches!(
            effective_type,
            ChunkType::Function | ChunkType::AsyncFunction | ChunkType::Method
        ) {
            let body_node_type = self.patterns.body_node.as_deref();
            if let Some(body_type) = body_node_type {
                if let Some(body) = find_child_by_kind(node, body_type) {
                    extract_function_calls(&body, source, &self.patterns.call_nodes)
                } else {
                    extract_function_calls(node, source, &self.patterns.call_nodes)
                }
            } else {
                extract_function_calls(node, source, &self.patterns.call_nodes)
            }
        } else {
            Vec::new()
        };

        let mut chunk = SemanticChunk::new(
            effective_type,
            name,
            content,
            start_line,
            end_line,
            &self.language_name,
            file_path,
        )
        .with_calls(calls);

        if let Some(sig) = signature {
            chunk = chunk.with_signature(sig);
        }
        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }
        if let Some(p) = parent {
            chunk = chunk.with_parent(p);
        }

        chunk
    }

    // ── Name extraction ───────────────────────────────────────────────

    fn extract_name<'a>(&self, node: &Node<'a>, source: &'a str) -> &'a str {
        // Try the configured name_node type first
        if let Some(ref name_type) = self.patterns.name_node {
            if let Some(name_node) = find_child_by_kind(node, name_type) {
                return node_text(&name_node, source);
            }
        }

        // Try common name field names
        for field in &["name", "declarator", "pattern"] {
            if let Some(name_node) = node.child_by_field_name(field.as_bytes()) {
                let text = node_text(&name_node, source);
                // For declarators, extract just the name
                if name_node.kind() == "function_declarator"
                    || name_node.kind() == "pointer_declarator"
                {
                    if let Some(inner) = find_child_by_kind(&name_node, "identifier") {
                        return node_text(&inner, source);
                    }
                }
                return text;
            }
        }

        // Fallback: second word in text (e.g., "def name", "fn name", "class Name")
        let text = node_text(node, source);
        text.split_whitespace()
            .nth(1)
            .map(|s| s.split('(').next().unwrap_or(s))
            .map(|s| s.split('<').next().unwrap_or(s))
            .map(|s| s.split(':').next().unwrap_or(s))
            .unwrap_or("anonymous")
    }

    // ── Class/module body walking ─────────────────────────────────────

    fn extract_methods_from_body(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        parent_name: &str,
        chunks: &mut Vec<SemanticChunk>,
    ) {
        // The configured `body_node` is the *method/function* body kind (e.g.
        // Java `block`). A container (class/interface/enum/impl) wraps its
        // members in a DIFFERENT node — `class_body`, `declaration_list`, … —
        // so probing only the configured kind on a container finds nothing and
        // silently extracts ZERO methods. That left every Java class
        // method-less, so the call graph had no Java CALLS edges at all. Try
        // the configured kind first (correct for languages whose class and
        // method bodies share a kind), then fall back to the common container
        // bodies.
        let fn_types = &self.patterns.function.node_types;
        let async_fn_types = &self.patterns.function.async_node_types;
        let method_types = &self.patterns.method.node_types;
        let decorated_wrapper = self.patterns.decorated_wrapper.as_deref();

        let body_type = self.patterns.body_node.as_deref();
        let body = body_type
            .and_then(|bt| find_child_by_kind(node, bt))
            .or_else(|| find_child_by_kind(node, "class_body"))
            .or_else(|| find_child_by_kind(node, "declaration_list"))
            .or_else(|| find_child_by_kind(node, "enum_body"))
            .or_else(|| find_child_by_kind(node, "interface_body"))
            .or_else(|| find_child_by_kind(node, "block"))
            .or_else(|| find_child_by_kind(node, "body"));

        let body = match body {
            Some(b) => b,
            None => {
                // Some grammars have no body wrapper at all and attach members
                // directly to the container node — proto's `service` keeps its
                // `rpc` members as direct children. Scan the container itself,
                // but only when a direct member is actually present, so other
                // languages' subtrees are not descended any differently than
                // before.
                let mut probe = node.walk();
                let has_direct_member = node.children(&mut probe).any(|child| {
                    let kind = child.kind();
                    method_types.iter().any(|t| t == kind)
                        || fn_types.iter().any(|t| t == kind)
                        || async_fn_types.iter().any(|t| t == kind)
                });
                if !has_direct_member {
                    return;
                }
                *node
            }
        };

        let mut cursor = body.walk();
        for child in body.children(&mut cursor) {
            let kind = child.kind();

            // Check for methods
            let is_method = method_types.iter().any(|t| t == kind)
                || fn_types.iter().any(|t| t == kind)
                || async_fn_types.iter().any(|t| t == kind);

            if is_method {
                chunks.push(self.extract_definition(
                    &child,
                    source,
                    file_path,
                    ChunkType::Function,
                    Some(parent_name),
                ));
            } else if Some(kind) == decorated_wrapper {
                // Handle decorated methods (Python @decorator pattern)
                for ft in fn_types.iter().chain(async_fn_types.iter()) {
                    if let Some(func) = find_child_by_kind(&child, ft) {
                        chunks.push(self.extract_definition(
                            &func,
                            source,
                            file_path,
                            ChunkType::Function,
                            Some(parent_name),
                        ));
                        break;
                    }
                }
            } else if kind == "do_block" || child.child_count() > 0 {
                // Recurse into nested blocks (Elixir do blocks, etc.)
                self.extract_methods_from_body(&child, source, file_path, parent_name, chunks);
            }
        }
    }

    // ── Container extraction (class/struct/module/etc.) ────────────────

    fn extract_container(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        chunk_type: ChunkType,
    ) -> Vec<SemanticChunk> {
        let mut chunks = Vec::new();

        let name = self.extract_name(node, source).to_string();
        let content = node_text(node, source);
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let docstring = docstring::extract_docstring(&self.patterns, node, source);

        let mut chunk = SemanticChunk::new(
            chunk_type,
            &name,
            content,
            start_line,
            end_line,
            &self.language_name,
            file_path,
        );

        if let Some(doc) = docstring {
            chunk = chunk.with_docstring(doc);
        }

        chunks.push(chunk);

        // Extract methods from container body
        if matches!(
            chunk_type,
            ChunkType::Class | ChunkType::Impl | ChunkType::Module | ChunkType::Trait
        ) {
            self.extract_methods_from_body(node, source, file_path, &name, &mut chunks);
        }

        chunks
    }
}

impl ChunkExtractor for GenericExtractor {
    fn extract_chunks(
        &self,
        source: &str,
        file_path: &Path,
    ) -> Result<Vec<SemanticChunk>, DaemonError> {
        let mut parser = self.create_parser()?;
        let tree = parser.parse(source)?;
        let root = tree.root_node();
        let file_path_str = file_path.to_string_lossy().to_string();

        let mut chunks = Vec::new();

        // Extract preamble
        if let Some(preamble) = self.extract_preamble(&root, source, &file_path_str) {
            chunks.push(preamble);
        }

        // Walk root children, unwrapping root_wrappers transparently
        walker::walk_children(
            &self.patterns,
            &root,
            source,
            &file_path_str,
            &mut chunks,
            &|n, s, fp, ct| self.extract_container(n, s, fp, ct),
            &|n, s, fp, ct, p| self.extract_definition(n, s, fp, ct, p),
        );

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        // Leak the string for the 'static lifetime required by the trait.
        // This is acceptable because extractors are long-lived.
        Box::leak(self.language_name.clone().into_boxed_str())
    }
}

#[cfg(test)]
mod tests;

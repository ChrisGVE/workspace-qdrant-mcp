//! Generic pattern-driven semantic chunk extractor.
//!
//! Replaces per-language extractors with a single engine that reads AST node
//! type patterns from the YAML language registry. Given a `SemanticPatterns`
//! configuration, it walks the tree-sitter AST and extracts preamble,
//! functions, classes, methods, structs, enums, traits, modules, etc.

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

        let docstring = self.extract_docstring(node, source);

        let calls = if matches!(
            effective_type,
            ChunkType::Function | ChunkType::AsyncFunction | ChunkType::Method
        ) {
            let body_node_type = self.patterns.body_node.as_deref();
            if let Some(body_type) = body_node_type {
                if let Some(body) = find_child_by_kind(node, body_type) {
                    extract_function_calls(&body, source)
                } else {
                    extract_function_calls(node, source)
                }
            } else {
                extract_function_calls(node, source)
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

    // ── Docstring extraction ──────────────────────────────────────────

    fn extract_docstring(&self, node: &Node, source: &str) -> Option<String> {
        match self.patterns.docstring_style {
            DocstringStyle::FirstStringInBody => self.docstring_first_string(node, source),
            DocstringStyle::PrecedingComments => self.docstring_preceding_comments(node, source),
            DocstringStyle::Javadoc => self.docstring_javadoc(node, source),
            DocstringStyle::Haddock => self.docstring_haddock(node, source),
            DocstringStyle::ElixirAttr => self.docstring_elixir_attr(node, source),
            DocstringStyle::OcamlDoc => self.docstring_ocaml(node, source),
            DocstringStyle::Pod => self.docstring_pod(node, source),
            DocstringStyle::None => None,
        }
    }

    /// Python-style: first string expression in function/class body.
    fn docstring_first_string(&self, node: &Node, source: &str) -> Option<String> {
        let body_type = self.patterns.body_node.as_deref().unwrap_or("block");
        let body = find_child_by_kind(node, body_type)?;
        let mut cursor = body.walk();

        for child in body.children(&mut cursor) {
            if child.kind() == "expression_statement" {
                if let Some(string_node) = find_child_by_kind(&child, "string") {
                    let text = node_text(&string_node, source);
                    return Some(
                        text.trim_start_matches("\"\"\"")
                            .trim_start_matches("'''")
                            .trim_end_matches("\"\"\"")
                            .trim_end_matches("'''")
                            .trim()
                            .to_string(),
                    );
                }
            }
            break;
        }
        None
    }

    /// C/C++/Rust/Go/Ruby style: comment nodes preceding the definition.
    fn docstring_preceding_comments(&self, node: &Node, source: &str) -> Option<String> {
        let comment_types = &self.patterns.comment_nodes;
        if comment_types.is_empty() {
            return None;
        }

        let mut comments = Vec::new();
        let mut prev = node.prev_sibling();

        while let Some(sibling) = prev {
            if comment_types.iter().any(|t| t == sibling.kind()) {
                let text = node_text(&sibling, source);
                // Check for doc comment markers
                let is_doc = text.starts_with("///")
                    || text.starts_with("//!")
                    || text.starts_with("/**")
                    || text.starts_with("##")
                    || text.starts_with("-- |")
                    || text.starts_with("---");
                if is_doc || comments.is_empty() {
                    comments.push(text.to_string());
                    prev = sibling.prev_sibling();
                    continue;
                }
            }
            break;
        }

        if comments.is_empty() {
            return None;
        }

        comments.reverse();
        let joined = comments.join("\n");

        // Clean common doc comment prefixes
        let cleaned: String = joined
            .lines()
            .map(|l| {
                l.trim()
                    .trim_start_matches("///")
                    .trim_start_matches("//!")
                    .trim_start_matches("## ")
                    .trim_start_matches("-- |")
                    .trim_start_matches("-- ")
                    .trim_start_matches("---")
                    .trim_start()
            })
            .collect::<Vec<_>>()
            .join("\n");

        Some(cleaned.trim().to_string())
    }

    /// Java/JS/TS/Scala style: `/** ... */` block comment.
    fn docstring_javadoc(&self, node: &Node, source: &str) -> Option<String> {
        let prev = node.prev_sibling()?;
        let text = node_text(&prev, source);
        if text.starts_with("/**") {
            let cleaned = text
                .trim_start_matches("/**")
                .trim_end_matches("*/")
                .lines()
                .map(|l| l.trim().trim_start_matches("* ").trim_start_matches('*'))
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string();
            if !cleaned.is_empty() {
                return Some(cleaned);
            }
        }
        None
    }

    /// Haskell style: `-- |` Haddock comments.
    fn docstring_haddock(&self, node: &Node, source: &str) -> Option<String> {
        self.docstring_preceding_comments(node, source)
    }

    /// Elixir style: `@doc` attribute.
    fn docstring_elixir_attr(&self, node: &Node, source: &str) -> Option<String> {
        let mut prev = node.prev_sibling();
        while let Some(sibling) = prev {
            let text = node_text(&sibling, source);
            if text.starts_with("@doc") {
                let doc = text
                    .trim_start_matches("@doc")
                    .trim()
                    .trim_start_matches("\"\"\"")
                    .trim_end_matches("\"\"\"")
                    .trim()
                    .to_string();
                return Some(doc);
            }
            if sibling.kind() == "comment" {
                prev = sibling.prev_sibling();
                continue;
            }
            break;
        }
        None
    }

    /// OCaml style: `(** ... *)` doc comments.
    fn docstring_ocaml(&self, node: &Node, source: &str) -> Option<String> {
        let prev = node.prev_sibling()?;
        if prev.kind() == "comment" {
            let text = node_text(&prev, source);
            if text.starts_with("(**") {
                let cleaned = text
                    .trim_start_matches("(**")
                    .trim_end_matches("*)")
                    .trim()
                    .to_string();
                return Some(cleaned);
            }
        }
        None
    }

    /// Perl POD style (simplified — extracts preceding comment block).
    fn docstring_pod(&self, node: &Node, source: &str) -> Option<String> {
        self.docstring_preceding_comments(node, source)
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
        let body_type = self.patterns.body_node.as_deref();
        let body = if let Some(bt) = body_type {
            find_child_by_kind(node, bt)
        } else {
            // Try common body containers
            find_child_by_kind(node, "block")
                .or_else(|| find_child_by_kind(node, "body"))
                .or_else(|| find_child_by_kind(node, "class_body"))
                .or_else(|| find_child_by_kind(node, "declaration_list"))
        };

        let body = match body {
            Some(b) => b,
            None => return,
        };

        let fn_types = &self.patterns.function.node_types;
        let async_fn_types = &self.patterns.function.async_node_types;
        let method_types = &self.patterns.method.node_types;
        let decorated_wrapper = self.patterns.decorated_wrapper.as_deref();

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

        let docstring = self.extract_docstring(node, source);

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

    // ── Main dispatch ─────────────────────────────────────────────────

    fn matches_any(kind: &str, types: &[String]) -> bool {
        types.iter().any(|t| t == kind)
    }

    fn classify_node(&self, kind: &str) -> Option<ChunkType> {
        let p = &self.patterns;

        if Self::matches_any(kind, &p.function.node_types)
            || Self::matches_any(kind, &p.function.async_node_types)
        {
            return Some(ChunkType::Function);
        }
        if Self::matches_any(kind, &p.class.node_types) {
            return Some(ChunkType::Class);
        }
        if Self::matches_any(kind, &p.struct_def.node_types) {
            return Some(ChunkType::Struct);
        }
        if Self::matches_any(kind, &p.enum_def.node_types) {
            return Some(ChunkType::Enum);
        }
        if Self::matches_any(kind, &p.trait_def.node_types) {
            return Some(ChunkType::Trait);
        }
        if Self::matches_any(kind, &p.interface.node_types) {
            return Some(ChunkType::Interface);
        }
        if Self::matches_any(kind, &p.module.node_types) {
            return Some(ChunkType::Module);
        }
        if Self::matches_any(kind, &p.constant.node_types) {
            return Some(ChunkType::Constant);
        }
        if Self::matches_any(kind, &p.macro_def.node_types) {
            return Some(ChunkType::Macro);
        }
        if Self::matches_any(kind, &p.type_alias.node_types) {
            return Some(ChunkType::TypeAlias);
        }
        if Self::matches_any(kind, &p.impl_block.node_types) {
            return Some(ChunkType::Impl);
        }

        None
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
        self.walk_children(&root, source, &file_path_str, &mut chunks);

        Ok(chunks)
    }

    fn language(&self) -> &'static str {
        // Leak the string for the 'static lifetime required by the trait.
        // This is acceptable because extractors are long-lived.
        Box::leak(self.language_name.clone().into_boxed_str())
    }
}

impl GenericExtractor {
    /// Walk children of a node, classifying each. If a child matches a
    /// `root_wrappers` entry, recurse into it transparently instead.
    fn walk_children(
        &self,
        parent: &Node,
        source: &str,
        file_path: &str,
        chunks: &mut Vec<SemanticChunk>,
    ) {
        let decorated_wrapper = self.patterns.decorated_wrapper.as_deref();
        let wrappers = &self.patterns.root_wrappers;
        let mut cursor = parent.walk();

        for child in parent.children(&mut cursor) {
            let kind = child.kind();

            // Unwrap root_wrappers transparently
            if wrappers.iter().any(|w| w == kind) {
                self.walk_children(&child, source, file_path, chunks);
                continue;
            }

            if let Some(chunk_type) = self.classify_node(kind) {
                match chunk_type {
                    ChunkType::Class
                    | ChunkType::Struct
                    | ChunkType::Trait
                    | ChunkType::Interface
                    | ChunkType::Module
                    | ChunkType::Impl => {
                        chunks
                            .extend(self.extract_container(&child, source, file_path, chunk_type));
                    }
                    _ => {
                        chunks.push(
                            self.extract_definition(&child, source, file_path, chunk_type, None),
                        );
                    }
                }
            } else if Some(kind) == decorated_wrapper {
                self.handle_decorated_node(&child, source, file_path, chunks);
            } else if kind == "call" {
                self.handle_call_node(&child, source, file_path, chunks);
            }
        }
    }

    fn handle_decorated_node(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        chunks: &mut Vec<SemanticChunk>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let kind = child.kind();
            if let Some(chunk_type) = self.classify_node(kind) {
                match chunk_type {
                    ChunkType::Class | ChunkType::Struct | ChunkType::Module => {
                        chunks
                            .extend(self.extract_container(&child, source, file_path, chunk_type));
                    }
                    _ => {
                        chunks.push(
                            self.extract_definition(&child, source, file_path, chunk_type, None),
                        );
                    }
                }
                return;
            }
        }
    }

    fn handle_call_node(
        &self,
        node: &Node,
        source: &str,
        file_path: &str,
        chunks: &mut Vec<SemanticChunk>,
    ) {
        let text = node_text(node, source);
        let first_word = text.split_whitespace().next().unwrap_or("");

        // Check if this call matches any definition pattern
        // Elixir: defmodule → Module, def/defp → Function, defmacro → Macro
        if Self::matches_any("call", &self.patterns.module.node_types)
            && matches!(first_word, "defmodule")
        {
            chunks.extend(self.extract_container(node, source, file_path, ChunkType::Module));
        } else if Self::matches_any("call", &self.patterns.function.node_types)
            && matches!(first_word, "def" | "defp")
        {
            chunks.push(self.extract_definition(
                node,
                source,
                file_path,
                ChunkType::Function,
                None,
            ));
        } else if Self::matches_any("call", &self.patterns.macro_def.node_types)
            && matches!(first_word, "defmacro" | "defmacrop")
        {
            chunks.push(self.extract_definition(node, source, file_path, ChunkType::Macro, None));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_registry::types::{
        DocstringStyle, FunctionPatternGroup, MethodPatternGroup, PatternGroup, SemanticPatterns,
    };
    use crate::tree_sitter::parser::get_language;
    use std::path::PathBuf;

    fn python_patterns() -> SemanticPatterns {
        SemanticPatterns {
            preamble: PatternGroup {
                node_types: vec![
                    "import_statement".into(),
                    "import_from_statement".into(),
                    "future_import_statement".into(),
                ],
            },
            function: FunctionPatternGroup {
                node_types: vec!["function_definition".into()],
                async_node_types: vec!["async_function_definition".into()],
            },
            class: PatternGroup {
                node_types: vec!["class_definition".into()],
            },
            method: MethodPatternGroup {
                node_types: vec![
                    "function_definition".into(),
                    "async_function_definition".into(),
                ],
                context: Some("inside_class".into()),
            },
            name_node: Some("identifier".into()),
            body_node: Some("block".into()),
            comment_nodes: vec!["comment".into()],
            docstring_style: DocstringStyle::FirstStringInBody,
            decorated_wrapper: Some("decorated_definition".into()),
            ..Default::default()
        }
    }

    fn rust_patterns() -> SemanticPatterns {
        SemanticPatterns {
            preamble: PatternGroup {
                node_types: vec!["use_declaration".into(), "extern_crate_declaration".into()],
            },
            function: FunctionPatternGroup {
                node_types: vec!["function_item".into()],
                async_node_types: vec![],
            },
            class: PatternGroup { node_types: vec![] },
            struct_def: PatternGroup {
                node_types: vec!["struct_item".into()],
            },
            enum_def: PatternGroup {
                node_types: vec!["enum_item".into()],
            },
            trait_def: PatternGroup {
                node_types: vec!["trait_item".into()],
            },
            impl_block: PatternGroup {
                node_types: vec!["impl_item".into()],
            },
            module: PatternGroup {
                node_types: vec!["mod_item".into()],
            },
            constant: PatternGroup {
                node_types: vec!["const_item".into(), "static_item".into()],
            },
            macro_def: PatternGroup {
                node_types: vec!["macro_definition".into()],
            },
            type_alias: PatternGroup {
                node_types: vec!["type_item".into()],
            },
            method: MethodPatternGroup {
                node_types: vec!["function_item".into()],
                context: Some("inside_impl".into()),
            },
            name_node: Some("identifier".into()),
            body_node: Some("block".into()),
            comment_nodes: vec!["line_comment".into(), "block_comment".into()],
            docstring_style: DocstringStyle::PrecedingComments,
            ..Default::default()
        }
    }

    #[test]
    fn test_python_function() {
        let Some(lang) = get_language("python") else {
            return;
        };
        let source = r#"
def hello():
    """Say hello."""
    print("Hello!")
"#;
        let extractor = GenericExtractor::new("python", lang, python_patterns());
        let chunks = extractor
            .extract_chunks(source, &PathBuf::from("test.py"))
            .unwrap();

        let func = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(func.is_some(), "Should find a function chunk");
        let func = func.unwrap();
        assert_eq!(func.symbol_name, "hello");
        assert!(func
            .docstring
            .as_ref()
            .is_some_and(|d| d.contains("Say hello")));
    }

    #[test]
    fn test_python_class_with_methods() {
        let Some(lang) = get_language("python") else {
            return;
        };
        let source = r#"
class Person:
    """A person."""
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, {self.name}!")
"#;
        let extractor = GenericExtractor::new("python", lang, python_patterns());
        let chunks = extractor
            .extract_chunks(source, &PathBuf::from("test.py"))
            .unwrap();

        let class = chunks.iter().find(|c| c.chunk_type == ChunkType::Class);
        assert!(class.is_some());
        assert_eq!(class.unwrap().symbol_name, "Person");

        let methods: Vec<_> = chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::Method)
            .collect();
        assert_eq!(methods.len(), 2, "Should find 2 methods");
    }

    #[test]
    fn test_python_preamble() {
        let Some(lang) = get_language("python") else {
            return;
        };
        let source = r#"
import os
from typing import List

def main():
    pass
"#;
        let extractor = GenericExtractor::new("python", lang, python_patterns());
        let chunks = extractor
            .extract_chunks(source, &PathBuf::from("test.py"))
            .unwrap();

        let preamble = chunks.iter().find(|c| c.chunk_type == ChunkType::Preamble);
        assert!(preamble.is_some());
        let preamble = preamble.unwrap();
        assert!(preamble.content.contains("import os"));
        assert!(preamble.content.contains("from typing"));
    }

    #[test]
    fn test_python_async_function() {
        let Some(lang) = get_language("python") else {
            return;
        };
        let source = r#"
async def fetch_data():
    """Fetch data."""
    return await get_data()
"#;
        let extractor = GenericExtractor::new("python", lang, python_patterns());
        let chunks = extractor
            .extract_chunks(source, &PathBuf::from("test.py"))
            .unwrap();

        let async_fn = chunks
            .iter()
            .find(|c| c.chunk_type == ChunkType::AsyncFunction);
        assert!(async_fn.is_some());
    }

    #[test]
    fn test_python_decorated_function() {
        let Some(lang) = get_language("python") else {
            return;
        };
        let source = r#"
@decorator
def decorated_func():
    pass
"#;
        let extractor = GenericExtractor::new("python", lang, python_patterns());
        let chunks = extractor
            .extract_chunks(source, &PathBuf::from("test.py"))
            .unwrap();

        let func = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
        assert!(func.is_some());
        assert_eq!(func.unwrap().symbol_name, "decorated_func");
    }

    #[test]
    fn test_rust_struct_and_impl() {
        let Some(lang) = get_language("rust") else {
            return;
        };
        let source = r#"
use std::fmt;

/// A point in 2D space.
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn distance(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}
"#;
        let extractor = GenericExtractor::new("rust", lang, rust_patterns());
        let chunks = extractor
            .extract_chunks(source, &PathBuf::from("test.rs"))
            .unwrap();

        assert!(chunks.iter().any(|c| c.chunk_type == ChunkType::Preamble));
        assert!(chunks.iter().any(|c| c.chunk_type == ChunkType::Struct));
        assert!(chunks.iter().any(|c| c.chunk_type == ChunkType::Impl));

        let methods: Vec<_> = chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::Method)
            .collect();
        assert_eq!(methods.len(), 2, "Should find 2 impl methods");
    }
}

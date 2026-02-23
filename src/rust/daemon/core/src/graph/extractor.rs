//! Graph relationship extractor — derives graph edges from SemanticChunk data.
//!
//! Takes tree-sitter `SemanticChunk` output and produces `GraphNode`/`GraphEdge`
//! pairs for CONTAINS, CALLS, IMPORTS, and USES_TYPE relationships.

use crate::tree_sitter::types::{ChunkType, SemanticChunk};

use super::{EdgeType, GraphEdge, GraphNode, NodeType};

/// Result of extracting graph relationships from a set of semantic chunks.
#[derive(Debug, Default)]
pub struct ExtractionResult {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

/// Extract graph nodes and edges from semantic chunks for a single file.
///
/// This is the main entry point called during file ingestion. It processes
/// all chunks from a file and produces the full set of nodes and edges.
pub fn extract_edges(
    chunks: &[SemanticChunk],
    tenant_id: &str,
    file_path: &str,
) -> ExtractionResult {
    let mut result = ExtractionResult::default();

    // Create a File node for import edges
    let file_node = GraphNode::new(tenant_id, file_path, file_path, NodeType::File);
    result.nodes.push(file_node);

    for chunk in chunks {
        let Some(node_type) = chunk_type_to_node_type(&chunk.chunk_type) else {
            // Preamble and Text chunks don't become nodes, but we still
            // extract imports from Preamble content below.
            if chunk.chunk_type == ChunkType::Preamble {
                extract_imports_from_content(
                    &chunk.content,
                    &chunk.language,
                    tenant_id,
                    file_path,
                    &mut result,
                );
            }
            continue;
        };

        // Create the node for this chunk
        let mut node = GraphNode::new(tenant_id, file_path, &chunk.symbol_name, node_type);
        node.start_line = Some(chunk.start_line as u32);
        node.end_line = Some(chunk.end_line as u32);
        node.signature = chunk.signature.clone();
        node.language = Some(chunk.language.clone());
        result.nodes.push(node.clone());

        // CONTAINS edges: parent_symbol → this chunk
        if let Some(ref parent) = chunk.parent_symbol {
            let parent_type = infer_parent_node_type(parent, &chunk.language);
            let parent_node = GraphNode::stub(tenant_id, parent, parent_type);
            let edge = GraphEdge::new(
                tenant_id,
                &parent_node.node_id,
                &node.node_id,
                EdgeType::Contains,
                file_path,
            );
            result.nodes.push(parent_node);
            result.edges.push(edge);
        }

        // CALLS edges: this chunk → called functions
        for call in &chunk.calls {
            let (_qualifier, callee_name) = parse_qualified_name(call);
            if callee_name.is_empty() {
                continue;
            }
            let callee_stub = GraphNode::stub(tenant_id, &callee_name, NodeType::Function);
            let edge = GraphEdge::new(
                tenant_id,
                &node.node_id,
                &callee_stub.node_id,
                EdgeType::Calls,
                file_path,
            );
            result.nodes.push(callee_stub);
            result.edges.push(edge);
        }

        // USES_TYPE edges: extract type references from signatures
        if let Some(ref sig) = chunk.signature {
            let type_refs = extract_type_references(sig, &chunk.language);
            for type_name in type_refs {
                let type_stub = GraphNode::stub(tenant_id, &type_name, NodeType::Struct);
                let edge = GraphEdge::new(
                    tenant_id,
                    &node.node_id,
                    &type_stub.node_id,
                    EdgeType::UsesType,
                    file_path,
                );
                result.nodes.push(type_stub);
                result.edges.push(edge);
            }
        }
    }

    result
}

/// Convert ChunkType to NodeType. Returns None for types that don't map
/// to graph nodes (Preamble, Text).
fn chunk_type_to_node_type(ct: &ChunkType) -> Option<NodeType> {
    match ct {
        ChunkType::Function => Some(NodeType::Function),
        ChunkType::AsyncFunction => Some(NodeType::AsyncFunction),
        ChunkType::Class => Some(NodeType::Class),
        ChunkType::Method => Some(NodeType::Method),
        ChunkType::Struct => Some(NodeType::Struct),
        ChunkType::Trait => Some(NodeType::Trait),
        ChunkType::Interface => Some(NodeType::Interface),
        ChunkType::Enum => Some(NodeType::Enum),
        ChunkType::Impl => Some(NodeType::Impl),
        ChunkType::Module => Some(NodeType::Module),
        ChunkType::Constant => Some(NodeType::Constant),
        ChunkType::TypeAlias => Some(NodeType::TypeAlias),
        ChunkType::Macro => Some(NodeType::Macro),
        ChunkType::Preamble | ChunkType::Text => None,
    }
}

/// Infer parent node type from symbol name and language.
///
/// In Rust, a parent is typically an `impl` block or `mod`.
/// In TypeScript/Python, a parent is typically a `class`.
fn infer_parent_node_type(parent_symbol: &str, language: &str) -> NodeType {
    match language {
        "rust" => {
            // Rust parent symbols from tree-sitter are typically impl blocks
            if parent_symbol.starts_with("impl ") || parent_symbol.contains("::") {
                NodeType::Impl
            } else {
                NodeType::Struct
            }
        }
        "python" | "javascript" | "typescript" | "tsx" | "jsx" | "java" | "kotlin" => {
            NodeType::Class
        }
        "go" => NodeType::Struct,
        _ => NodeType::Module,
    }
}

/// Parse a qualified name like `self.method`, `module::function`, or `pkg.Func`.
///
/// Returns (optional_qualifier, base_name). The base_name is always the
/// last component which is the actual callable symbol.
pub fn parse_qualified_name(call: &str) -> (Option<String>, String) {
    let call = call.trim();
    if call.is_empty() {
        return (None, String::new());
    }

    // Rust/C++ qualified names: `foo::bar::baz`
    if let Some(pos) = call.rfind("::") {
        let qualifier = &call[..pos];
        let name = &call[pos + 2..];
        if !name.is_empty() {
            return (Some(qualifier.to_string()), name.to_string());
        }
    }

    // Method calls: `self.method`, `obj.method`, `pkg.Func`
    if let Some(pos) = call.rfind('.') {
        let qualifier = &call[..pos];
        let name = &call[pos + 1..];
        if !name.is_empty() {
            return (Some(qualifier.to_string()), name.to_string());
        }
    }

    // Simple unqualified name
    (None, call.to_string())
}

/// Extract type references from a function/method signature.
///
/// Uses simple regex-free parsing to find capitalized type names and
/// well-known generic wrappers. Not a full parser — focuses on common
/// patterns that cover ~80% of real-world code.
pub fn extract_type_references(signature: &str, language: &str) -> Vec<String> {
    let mut types = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Tokenize the signature into potential type identifiers
    let tokens = tokenize_signature(signature);

    for token in &tokens {
        if is_type_name(token, language) && seen.insert(token.clone()) {
            types.push(token.clone());
        }
    }

    types
}

/// Tokenize a signature string into identifier-like tokens.
///
/// Splits on non-identifier characters, keeping only tokens that look
/// like identifiers (start with letter/underscore, contain alphanum/_).
fn tokenize_signature(sig: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in sig.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
        } else {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// Check if a token looks like a type name rather than a keyword or variable.
fn is_type_name(token: &str, language: &str) -> bool {
    if token.len() < 2 {
        return false;
    }

    // Skip language keywords
    if is_keyword(token, language) {
        return false;
    }

    // Skip common parameter names and primitives
    if is_primitive_or_builtin(token, language) {
        return false;
    }

    // In most languages, types start with uppercase
    let first = token.chars().next().unwrap();
    if first.is_uppercase() {
        return true;
    }

    // Rust generic types can be lowercase: `vec`, `option`, `result` — but
    // canonically they're PascalCase. We only accept PascalCase for now.
    false
}

/// Check if a token is a language keyword (not a type).
fn is_keyword(token: &str, language: &str) -> bool {
    match language {
        "rust" => matches!(
            token,
            "fn" | "pub"
                | "self"
                | "Self"
                | "mut"
                | "let"
                | "const"
                | "static"
                | "async"
                | "await"
                | "impl"
                | "trait"
                | "struct"
                | "enum"
                | "type"
                | "where"
                | "for"
                | "in"
                | "if"
                | "else"
                | "match"
                | "return"
                | "mod"
                | "use"
                | "crate"
                | "super"
                | "dyn"
                | "ref"
                | "unsafe"
                | "extern"
        ),
        "python" => matches!(
            token,
            "def" | "self"
                | "cls"
                | "class"
                | "return"
                | "import"
                | "from"
                | "as"
                | "if"
                | "else"
                | "elif"
                | "for"
                | "in"
                | "while"
                | "with"
                | "try"
                | "except"
                | "raise"
                | "pass"
                | "lambda"
                | "yield"
                | "async"
                | "await"
                | "None"
                | "True"
                | "False"
        ),
        "javascript" | "typescript" | "tsx" | "jsx" => matches!(
            token,
            "function"
                | "const"
                | "let"
                | "var"
                | "return"
                | "if"
                | "else"
                | "for"
                | "while"
                | "class"
                | "extends"
                | "implements"
                | "import"
                | "export"
                | "default"
                | "new"
                | "this"
                | "super"
                | "async"
                | "await"
                | "yield"
                | "typeof"
                | "instanceof"
                | "void"
                | "null"
                | "undefined"
                | "true"
                | "false"
        ),
        "go" => matches!(
            token,
            "func" | "return"
                | "if"
                | "else"
                | "for"
                | "range"
                | "switch"
                | "case"
                | "type"
                | "struct"
                | "interface"
                | "package"
                | "import"
                | "var"
                | "const"
                | "defer"
                | "go"
                | "chan"
                | "select"
                | "nil"
                | "true"
                | "false"
                | "map"
        ),
        _ => false,
    }
}

/// Check if a token is a primitive type or common builtin.
fn is_primitive_or_builtin(token: &str, language: &str) -> bool {
    match language {
        "rust" => matches!(
            token,
            "i8" | "i16"
                | "i32"
                | "i64"
                | "i128"
                | "isize"
                | "u8"
                | "u16"
                | "u32"
                | "u64"
                | "u128"
                | "usize"
                | "f32"
                | "f64"
                | "bool"
                | "char"
                | "str"
        ),
        "python" => matches!(
            token,
            "int" | "float" | "str" | "bool" | "bytes" | "list" | "dict" | "set" | "tuple"
        ),
        "javascript" | "typescript" | "tsx" | "jsx" => matches!(
            token,
            "string" | "number" | "boolean" | "any" | "never" | "unknown" | "void" | "object"
        ),
        "go" => matches!(
            token,
            "int" | "int8"
                | "int16"
                | "int32"
                | "int64"
                | "uint"
                | "uint8"
                | "uint16"
                | "uint32"
                | "uint64"
                | "float32"
                | "float64"
                | "bool"
                | "string"
                | "byte"
                | "rune"
                | "error"
        ),
        _ => false,
    }
}

/// Extract IMPORTS edges from preamble content.
///
/// Parses import/use statements to create edges from the File node to
/// imported symbols. Uses simple line-by-line regex-free matching for
/// the most common patterns.
fn extract_imports_from_content(
    content: &str,
    language: &str,
    tenant_id: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let file_node_id =
        super::compute_node_id(tenant_id, file_path, file_path, NodeType::File);

    for line in content.lines() {
        let line = line.trim();
        let imports = parse_import_line(line, language);
        for symbol in imports {
            if symbol.is_empty() || symbol.len() < 2 {
                continue;
            }
            let stub = GraphNode::stub(tenant_id, &symbol, NodeType::Module);
            let edge = GraphEdge::new(
                tenant_id,
                &file_node_id,
                &stub.node_id,
                EdgeType::Imports,
                file_path,
            );
            result.nodes.push(stub);
            result.edges.push(edge);
        }
    }
}

/// Parse a single import/use line and return imported symbol names.
fn parse_import_line(line: &str, language: &str) -> Vec<String> {
    match language {
        "rust" => parse_rust_use(line),
        "python" => parse_python_import(line),
        "javascript" | "typescript" | "tsx" | "jsx" => parse_js_import(line),
        "go" => parse_go_import(line),
        _ => vec![],
    }
}

/// Parse Rust `use` statements.
///
/// Examples:
/// - `use std::collections::HashMap;` → ["HashMap"]
/// - `use crate::graph::{GraphNode, GraphEdge};` → ["GraphNode", "GraphEdge"]
/// - `use super::*;` → [] (wildcard, skip)
fn parse_rust_use(line: &str) -> Vec<String> {
    let line = line.trim().trim_end_matches(';');
    if !line.starts_with("use ") {
        return vec![];
    }
    let path = line[4..].trim();

    // Skip wildcard imports
    if path.ends_with("::*") {
        return vec![];
    }

    // Grouped imports: `use foo::{A, B, C};`
    if let Some(brace_start) = path.find('{') {
        if let Some(brace_end) = path.find('}') {
            let items = &path[brace_start + 1..brace_end];
            return items
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && *s != "self" && *s != "*")
                .collect();
        }
    }

    // Simple import: `use std::collections::HashMap`
    // Take the last path component
    if let Some(pos) = path.rfind("::") {
        let name = path[pos + 2..].trim();
        if !name.is_empty() && name != "self" {
            return vec![name.to_string()];
        }
    }

    // Single-segment use (rare): `use serde;`
    if !path.contains("::") && !path.is_empty() {
        return vec![path.to_string()];
    }

    vec![]
}

/// Parse Python import statements.
///
/// Examples:
/// - `import numpy` → ["numpy"]
/// - `from pathlib import Path` → ["Path"]
/// - `from typing import Dict, List, Optional` → ["Dict", "List", "Optional"]
fn parse_python_import(line: &str) -> Vec<String> {
    let line = line.trim();

    if line.starts_with("from ") {
        // `from X import Y, Z`
        if let Some(import_pos) = line.find(" import ") {
            let items = &line[import_pos + 8..];
            return items
                .split(',')
                .map(|s| {
                    // Handle `as` aliases: `import X as Y` → X
                    let s = s.trim();
                    if let Some(as_pos) = s.find(" as ") {
                        s[..as_pos].trim().to_string()
                    } else {
                        s.to_string()
                    }
                })
                .filter(|s| !s.is_empty() && *s != "*")
                .collect();
        }
    } else if line.starts_with("import ") {
        // `import X, Y`
        let items = &line[7..];
        return items
            .split(',')
            .map(|s| {
                let s = s.trim();
                if let Some(as_pos) = s.find(" as ") {
                    s[..as_pos].trim().to_string()
                } else {
                    s.to_string()
                }
            })
            .filter(|s| !s.is_empty())
            .collect();
    }

    vec![]
}

/// Parse JavaScript/TypeScript import statements.
///
/// Examples:
/// - `import { Component, useState } from 'react';` → ["Component", "useState"]
/// - `import React from 'react';` → ["React"]
/// - `import * as path from 'path';` → [] (namespace import, skip)
fn parse_js_import(line: &str) -> Vec<String> {
    let line = line.trim().trim_end_matches(';');

    if !line.starts_with("import ") {
        return vec![];
    }
    let rest = &line[7..].trim();

    // Skip `import * as X from ...`
    if rest.starts_with("* as") || rest.starts_with("* ") {
        return vec![];
    }

    // Named imports: `import { A, B } from '...'`
    if let Some(brace_start) = rest.find('{') {
        if let Some(brace_end) = rest.find('}') {
            let items = &rest[brace_start + 1..brace_end];
            return items
                .split(',')
                .map(|s| {
                    let s = s.trim();
                    // Handle `X as Y` → X
                    if let Some(as_pos) = s.find(" as ") {
                        s[..as_pos].trim().to_string()
                    } else {
                        s.to_string()
                    }
                })
                .filter(|s| !s.is_empty())
                .collect();
        }
    }

    // Default import: `import React from '...'`
    if let Some(from_pos) = rest.find(" from ") {
        let name = rest[..from_pos].trim();
        if !name.is_empty() && !name.contains('{') {
            return vec![name.to_string()];
        }
    }

    vec![]
}

/// Parse Go import statements (single line within import block).
///
/// Examples:
/// - `"fmt"` → ["fmt"]
/// - `"encoding/json"` → ["json"]
/// - `alias "some/package"` → ["package"]
fn parse_go_import(line: &str) -> Vec<String> {
    let line = line.trim();

    // Skip `import (` and `)` lines
    if line.starts_with("import") || line == "(" || line == ")" {
        return vec![];
    }

    // Extract the quoted path
    if let Some(start) = line.find('"') {
        if let Some(end) = line[start + 1..].find('"') {
            let path = &line[start + 1..start + 1 + end];
            // Use last path segment as the import name
            let name = path.rsplit('/').next().unwrap_or(path);
            if !name.is_empty() {
                return vec![name.to_string()];
            }
        }
    }

    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_qualified_name ─────────────────────────────────────────

    #[test]
    fn test_parse_qualified_rust() {
        let (q, name) = parse_qualified_name("std::collections::HashMap::new");
        assert_eq!(q.unwrap(), "std::collections::HashMap");
        assert_eq!(name, "new");
    }

    #[test]
    fn test_parse_qualified_dot() {
        let (q, name) = parse_qualified_name("self.process");
        assert_eq!(q.unwrap(), "self");
        assert_eq!(name, "process");
    }

    #[test]
    fn test_parse_unqualified() {
        let (q, name) = parse_qualified_name("println");
        assert!(q.is_none());
        assert_eq!(name, "println");
    }

    #[test]
    fn test_parse_empty() {
        let (q, name) = parse_qualified_name("");
        assert!(q.is_none());
        assert_eq!(name, "");
    }

    // ── extract_type_references ──────────────────────────────────────

    #[test]
    fn test_type_refs_rust_signature() {
        let sig = "fn process(data: Vec<String>) -> Result<(), Error>";
        let refs = extract_type_references(sig, "rust");
        assert!(refs.contains(&"Vec".to_string()));
        assert!(refs.contains(&"String".to_string()));
        assert!(refs.contains(&"Result".to_string()));
        assert!(refs.contains(&"Error".to_string()));
        // Should not contain keywords or primitives
        assert!(!refs.contains(&"fn".to_string()));
    }

    #[test]
    fn test_type_refs_typescript() {
        let sig = "function fetch(url: string): Promise<Response>";
        let refs = extract_type_references(sig, "typescript");
        assert!(refs.contains(&"Promise".to_string()));
        assert!(refs.contains(&"Response".to_string()));
        // "string" is primitive, should be excluded
        assert!(!refs.contains(&"string".to_string()));
    }

    #[test]
    fn test_type_refs_no_duplicates() {
        let sig = "fn merge(a: Vec<String>, b: Vec<String>) -> Vec<String>";
        let refs = extract_type_references(sig, "rust");
        let vec_count = refs.iter().filter(|r| *r == "Vec").count();
        assert_eq!(vec_count, 1);
    }

    // ── parse_rust_use ───────────────────────────────────────────────

    #[test]
    fn test_rust_use_simple() {
        let imports = parse_rust_use("use std::collections::HashMap;");
        assert_eq!(imports, vec!["HashMap"]);
    }

    #[test]
    fn test_rust_use_grouped() {
        let imports = parse_rust_use("use crate::graph::{GraphNode, GraphEdge};");
        assert_eq!(imports, vec!["GraphNode", "GraphEdge"]);
    }

    #[test]
    fn test_rust_use_wildcard_skipped() {
        let imports = parse_rust_use("use super::*;");
        assert!(imports.is_empty());
    }

    #[test]
    fn test_rust_use_single_segment() {
        let imports = parse_rust_use("use serde;");
        assert_eq!(imports, vec!["serde"]);
    }

    // ── parse_python_import ──────────────────────────────────────────

    #[test]
    fn test_python_import_simple() {
        let imports = parse_python_import("import numpy");
        assert_eq!(imports, vec!["numpy"]);
    }

    #[test]
    fn test_python_from_import() {
        let imports = parse_python_import("from pathlib import Path");
        assert_eq!(imports, vec!["Path"]);
    }

    #[test]
    fn test_python_from_import_multiple() {
        let imports = parse_python_import("from typing import Dict, List, Optional");
        assert_eq!(imports, vec!["Dict", "List", "Optional"]);
    }

    #[test]
    fn test_python_import_as() {
        let imports = parse_python_import("import numpy as np");
        assert_eq!(imports, vec!["numpy"]);
    }

    // ── parse_js_import ──────────────────────────────────────────────

    #[test]
    fn test_js_named_imports() {
        let imports = parse_js_import("import { Component, useState } from 'react';");
        assert_eq!(imports, vec!["Component", "useState"]);
    }

    #[test]
    fn test_js_default_import() {
        let imports = parse_js_import("import React from 'react';");
        assert_eq!(imports, vec!["React"]);
    }

    #[test]
    fn test_js_namespace_import_skipped() {
        let imports = parse_js_import("import * as path from 'path';");
        assert!(imports.is_empty());
    }

    #[test]
    fn test_js_import_with_alias() {
        let imports = parse_js_import("import { useState as state } from 'react';");
        assert_eq!(imports, vec!["useState"]);
    }

    // ── parse_go_import ──────────────────────────────────────────────

    #[test]
    fn test_go_import_simple() {
        let imports = parse_go_import("\"fmt\"");
        assert_eq!(imports, vec!["fmt"]);
    }

    #[test]
    fn test_go_import_path() {
        let imports = parse_go_import("\"encoding/json\"");
        assert_eq!(imports, vec!["json"]);
    }

    // ── extract_edges integration ────────────────────────────────────

    #[test]
    fn test_extract_edges_contains() {
        let chunks = vec![SemanticChunk::new(
            ChunkType::Method,
            "process",
            "fn process(&self) {}",
            10,
            15,
            "rust",
            "src/lib.rs",
        )
        .with_parent("MyStruct")];

        let result = extract_edges(&chunks, "t1", "src/lib.rs");

        // Should have: File node, method node, parent stub
        assert!(result.nodes.len() >= 3);

        // Should have a CONTAINS edge
        let contains_edges: Vec<_> = result
            .edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Contains)
            .collect();
        assert_eq!(contains_edges.len(), 1);
    }

    #[test]
    fn test_extract_edges_calls() {
        let mut chunk = SemanticChunk::new(
            ChunkType::Function,
            "main",
            "fn main() { foo(); bar(); }",
            1,
            5,
            "rust",
            "src/main.rs",
        );
        chunk.calls = vec!["foo".to_string(), "bar".to_string()];

        let result = extract_edges(&[chunk], "t1", "src/main.rs");

        let call_edges: Vec<_> = result
            .edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Calls)
            .collect();
        assert_eq!(call_edges.len(), 2);
    }

    #[test]
    fn test_extract_edges_uses_type() {
        let mut chunk = SemanticChunk::new(
            ChunkType::Function,
            "process",
            "fn process(data: Vec<String>) -> Result<(), Error> {}",
            1,
            5,
            "rust",
            "src/lib.rs",
        );
        chunk.signature = Some("fn process(data: Vec<String>) -> Result<(), Error>".to_string());

        let result = extract_edges(&[chunk], "t1", "src/lib.rs");

        let type_edges: Vec<_> = result
            .edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::UsesType)
            .collect();
        // Vec, String, Result, Error
        assert!(type_edges.len() >= 4);
    }

    #[test]
    fn test_extract_edges_imports_from_preamble() {
        let chunk = SemanticChunk::new(
            ChunkType::Preamble,
            "preamble",
            "use std::collections::HashMap;\nuse crate::graph::{GraphNode, GraphEdge};",
            1,
            3,
            "rust",
            "src/lib.rs",
        );

        let result = extract_edges(&[chunk], "t1", "src/lib.rs");

        let import_edges: Vec<_> = result
            .edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Imports)
            .collect();
        // HashMap, GraphNode, GraphEdge
        assert_eq!(import_edges.len(), 3);
    }
}

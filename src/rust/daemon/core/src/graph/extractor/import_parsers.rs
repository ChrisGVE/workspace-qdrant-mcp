//! Import statement parsers for Rust, Python, JavaScript/TypeScript, and Go.
//!
//! Each parser extracts imported symbol names from a single line of source code.

use super::super::{compute_node_id, EdgeType, GraphEdge, GraphNode, NodeType};
use super::ExtractionResult;

/// Extract IMPORTS edges from preamble content.
///
/// Parses import/use statements to create edges from the File node to
/// imported symbols. Uses simple line-by-line regex-free matching for
/// the most common patterns.
pub(crate) fn extract_imports_from_content(
    content: &str,
    language: &str,
    tenant_id: &str,
    file_path: &str,
    result: &mut ExtractionResult,
) {
    let file_node_id =
        compute_node_id(tenant_id, file_path, file_path, NodeType::File);

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
/// - `use std::collections::HashMap;` -> ["HashMap"]
/// - `use crate::graph::{GraphNode, GraphEdge};` -> ["GraphNode", "GraphEdge"]
/// - `use super::*;` -> [] (wildcard, skip)
pub(crate) fn parse_rust_use(line: &str) -> Vec<String> {
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
/// - `import numpy` -> ["numpy"]
/// - `from pathlib import Path` -> ["Path"]
/// - `from typing import Dict, List, Optional` -> ["Dict", "List", "Optional"]
pub(crate) fn parse_python_import(line: &str) -> Vec<String> {
    let line = line.trim();

    if line.starts_with("from ") {
        // `from X import Y, Z`
        if let Some(import_pos) = line.find(" import ") {
            let items = &line[import_pos + 8..];
            return items
                .split(',')
                .map(|s| {
                    // Handle `as` aliases: `import X as Y` -> X
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
/// - `import { Component, useState } from 'react';` -> ["Component", "useState"]
/// - `import React from 'react';` -> ["React"]
/// - `import * as path from 'path';` -> [] (namespace import, skip)
pub(crate) fn parse_js_import(line: &str) -> Vec<String> {
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
                    // Handle `X as Y` -> X
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
/// - `"fmt"` -> ["fmt"]
/// - `"encoding/json"` -> ["json"]
/// - `alias "some/package"` -> ["package"]
pub(crate) fn parse_go_import(line: &str) -> Vec<String> {
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

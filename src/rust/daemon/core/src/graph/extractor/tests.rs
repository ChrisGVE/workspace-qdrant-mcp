//! Tests for graph relationship extraction.

use super::*;
use crate::tree_sitter::types::{ChunkType, SemanticChunk};

// -- parse_qualified_name -------------------------------------------------

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

// -- extract_type_references ----------------------------------------------

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

// -- parse_rust_use -------------------------------------------------------

#[test]
fn test_rust_use_simple() {
    let imports = import_parsers::parse_rust_use("use std::collections::HashMap;");
    assert_eq!(imports, vec!["HashMap"]);
}

#[test]
fn test_rust_use_grouped() {
    let imports = import_parsers::parse_rust_use("use crate::graph::{GraphNode, GraphEdge};");
    assert_eq!(imports, vec!["GraphNode", "GraphEdge"]);
}

#[test]
fn test_rust_use_wildcard_skipped() {
    let imports = import_parsers::parse_rust_use("use super::*;");
    assert!(imports.is_empty());
}

#[test]
fn test_rust_use_single_segment() {
    let imports = import_parsers::parse_rust_use("use serde;");
    assert_eq!(imports, vec!["serde"]);
}

// -- parse_python_import --------------------------------------------------

#[test]
fn test_python_import_simple() {
    let imports = import_parsers::parse_python_import("import numpy");
    assert_eq!(imports, vec!["numpy"]);
}

#[test]
fn test_python_from_import() {
    let imports = import_parsers::parse_python_import("from pathlib import Path");
    assert_eq!(imports, vec!["Path"]);
}

#[test]
fn test_python_from_import_multiple() {
    let imports = import_parsers::parse_python_import("from typing import Dict, List, Optional");
    assert_eq!(imports, vec!["Dict", "List", "Optional"]);
}

#[test]
fn test_python_import_as() {
    let imports = import_parsers::parse_python_import("import numpy as np");
    assert_eq!(imports, vec!["numpy"]);
}

// -- parse_js_import ------------------------------------------------------

#[test]
fn test_js_named_imports() {
    let imports = import_parsers::parse_js_import("import { Component, useState } from 'react';");
    assert_eq!(imports, vec!["Component", "useState"]);
}

#[test]
fn test_js_default_import() {
    let imports = import_parsers::parse_js_import("import React from 'react';");
    assert_eq!(imports, vec!["React"]);
}

#[test]
fn test_js_namespace_import_skipped() {
    let imports = import_parsers::parse_js_import("import * as path from 'path';");
    assert!(imports.is_empty());
}

#[test]
fn test_js_import_with_alias() {
    let imports = import_parsers::parse_js_import("import { useState as state } from 'react';");
    assert_eq!(imports, vec!["useState"]);
}

// -- parse_go_import ------------------------------------------------------

#[test]
fn test_go_import_simple() {
    let imports = import_parsers::parse_go_import("\"fmt\"");
    assert_eq!(imports, vec!["fmt"]);
}

#[test]
fn test_go_import_path() {
    let imports = import_parsers::parse_go_import("\"encoding/json\"");
    assert_eq!(imports, vec!["json"]);
}

// -- extract_edges integration --------------------------------------------

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

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

// -- parse_java_import ----------------------------------------------------

#[test]
fn test_java_import_simple() {
    let imports = import_parsers::parse_java_import("import com.example.Foo;");
    assert_eq!(imports, vec!["Foo"]);
}

#[test]
fn test_java_import_static() {
    let imports = import_parsers::parse_java_import("import static com.example.Bar.baz;");
    assert_eq!(imports, vec!["baz"]);
}

#[test]
fn test_java_import_wildcard_skipped() {
    let imports = import_parsers::parse_java_import("import com.example.*;");
    assert!(imports.is_empty());
}

#[test]
fn test_kotlin_import_alias_uses_symbol() {
    // Kotlin `as` aliases the local binding; the imported symbol is the tail.
    let imports = import_parsers::parse_java_import("import com.example.Foo as Bar");
    assert_eq!(imports, vec!["Foo"]);
}

// -- parse_dart_import ----------------------------------------------------

#[test]
fn test_dart_import_package() {
    let imports = import_parsers::parse_dart_import("import 'package:flutter/material.dart';");
    assert_eq!(imports, vec!["material"]);
}

#[test]
fn test_dart_import_sdk() {
    let imports = import_parsers::parse_dart_import("import 'dart:async';");
    assert_eq!(imports, vec!["async"]);
}

#[test]
fn test_dart_import_alias_wins() {
    let imports = import_parsers::parse_dart_import("import 'widgets/foo.dart' as foo;");
    assert_eq!(imports, vec!["foo"]);
}

#[test]
fn test_dart_export_relative() {
    let imports = import_parsers::parse_dart_import("export 'src/bar.dart';");
    assert_eq!(imports, vec!["bar"]);
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
fn test_calls_skip_turbofish_artifacts() {
    use crate::TextChunk;
    use std::collections::HashMap;

    // A body with `foo()` and a turbofish call `bar::<String, _>()`. When the
    // call extractor splits the turbofish argument list, it leaks the fragments
    // `<String` and `_>` into the call list alongside the real `foo`. Only the
    // valid identifier `foo` should become a CALLS target.
    let mut meta = HashMap::new();
    meta.insert("chunk_type".to_string(), "function".to_string());
    meta.insert("symbol_name".to_string(), "caller".to_string());
    meta.insert("language".to_string(), "rust".to_string());
    meta.insert("calls".to_string(), "foo,<String, _>".to_string());

    let chunk = TextChunk {
        content: "fn caller() { foo(); bar::<String, _>(); }".to_string(),
        chunk_index: 0,
        start_char: 0,
        end_char: 0,
        metadata: meta,
    };

    let result = extract_edges_from_text_chunks(&[chunk], "t1", "src/lib.rs");

    let call_targets: Vec<&str> = result
        .edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::Calls)
        .filter_map(|e| {
            result
                .nodes
                .iter()
                .find(|n| n.node_id == e.target_node_id)
                .map(|n| n.symbol_name.as_str())
        })
        .collect();
    assert_eq!(
        call_targets,
        vec!["foo"],
        "only `foo` should be a CALLS target, not `<String` or `_>`"
    );

    // Belt-and-suspenders: no graph node should carry a generic-fragment artifact.
    assert!(
        result
            .nodes
            .iter()
            .all(|n| !n.symbol_name.contains('<') && !n.symbol_name.contains('>')),
        "no graph node should contain `<` or `>`"
    );
}

#[test]
fn test_semantic_calls_skip_turbofish_artifacts() {
    // Same guarantee on the SemanticChunk path: artifacts in `chunk.calls`
    // never become edge targets.
    let mut chunk = SemanticChunk::new(
        ChunkType::Function,
        "caller",
        "fn caller() { foo(); bar::<String, _>(); }",
        1,
        3,
        "rust",
        "src/main.rs",
    );
    chunk.calls = vec!["foo".to_string(), "<String".to_string(), "_>".to_string()];

    let result = extract_edges(&[chunk], "t1", "src/main.rs");

    let call_targets: Vec<&str> = result
        .edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::Calls)
        .filter_map(|e| {
            result
                .nodes
                .iter()
                .find(|n| n.node_id == e.target_node_id)
                .map(|n| n.symbol_name.as_str())
        })
        .collect();
    assert_eq!(call_targets, vec!["foo"]);
}

#[test]
fn test_is_valid_symbol_name_rejects_generic_fragments() {
    assert!(is_valid_symbol_name("foo"));
    assert!(is_valid_symbol_name("HashMap"));
    assert!(is_valid_symbol_name("_private"));
    assert!(is_valid_symbol_name("std::vec::Vec")); // `::`-qualified accepted

    assert!(!is_valid_symbol_name("<String"));
    assert!(!is_valid_symbol_name("_>"));
    assert!(!is_valid_symbol_name("<String, _>"));
    assert!(!is_valid_symbol_name("foo::")); // trailing empty segment
    assert!(!is_valid_symbol_name(""));
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

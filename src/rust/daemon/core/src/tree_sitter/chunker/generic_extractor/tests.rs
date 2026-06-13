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

fn typescript_patterns() -> SemanticPatterns {
    SemanticPatterns {
        preamble: PatternGroup {
            node_types: vec!["import_statement".into()],
        },
        root_wrappers: vec!["export_statement".into(), "lexical_declaration".into()],
        function: FunctionPatternGroup {
            node_types: vec![
                "function_declaration".into(),
                "generator_function_declaration".into(),
                "arrow_function".into(),
                "function".into(),
            ],
            async_node_types: vec![],
        },
        class: PatternGroup {
            node_types: vec!["class_declaration".into()],
        },
        method: MethodPatternGroup {
            node_types: vec!["method_definition".into(), "public_field_definition".into()],
            context: Some("inside_class".into()),
        },
        interface: PatternGroup {
            node_types: vec!["interface_declaration".into()],
        },
        constant: PatternGroup {
            node_types: vec!["variable_declarator".into()],
        },
        type_alias: PatternGroup {
            node_types: vec!["type_alias_declaration".into()],
        },
        name_node: Some("identifier".into()),
        body_node: Some("statement_block".into()),
        comment_nodes: vec!["comment".into()],
        docstring_style: DocstringStyle::Javadoc,
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

#[test]
fn test_typescript_exported_function_and_const_chunks() {
    let Some(lang) = get_language("typescript") else {
        return;
    };
    let source = r#"
import type { SearchMode } from './types';

const SPARSE_ONLY_WEIGHT = 0.5;

export function applyRRFFusion(results: string[], mode: SearchMode): string[] {
    return mode === 'hybrid' ? results : results;
}
"#;
    let extractor = GenericExtractor::new("typescript", lang, typescript_patterns());
    let chunks = extractor
        .extract_chunks(source, &PathBuf::from("search-qdrant.ts"))
        .unwrap();

    assert!(chunks.iter().any(|c| c.chunk_type == ChunkType::Preamble));
    assert!(chunks.iter().any(|c| {
        c.chunk_type == ChunkType::Constant && c.symbol_name == "SPARSE_ONLY_WEIGHT"
    }));
    assert!(chunks.iter().any(|c| {
        c.chunk_type == ChunkType::Function && c.symbol_name == "applyRRFFusion"
    }));
}

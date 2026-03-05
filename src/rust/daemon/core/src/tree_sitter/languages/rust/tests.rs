//! Unit tests for the Rust chunk extractor.

use super::*;
use crate::tree_sitter::parser::get_language;
use crate::tree_sitter::types::ChunkType;
use std::path::PathBuf;

#[test]
fn test_extract_function() {
    let Some(lang) = get_language("rust") else {
        return;
    };
    let source = r#"
/// A simple function.
fn hello() {
    println!("Hello!");
}
"#;
    let path = PathBuf::from("test.rs");
    let extractor = RustExtractor::with_language(lang);
    let chunks = extractor.extract_chunks(source, &path).unwrap();

    assert!(!chunks.is_empty());
    let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
    assert!(fn_chunk.is_some());
    let fn_chunk = fn_chunk.unwrap();
    assert_eq!(fn_chunk.symbol_name, "hello");
    assert!(fn_chunk.docstring.is_some());
    assert!(fn_chunk
        .docstring
        .as_ref()
        .unwrap()
        .contains("simple function"));
}

#[test]
fn test_extract_struct() {
    let Some(lang) = get_language("rust") else {
        return;
    };
    let source = r#"
/// A person struct.
pub struct Person {
    name: String,
    age: u32,
}
"#;
    let path = PathBuf::from("test.rs");
    let extractor = RustExtractor::with_language(lang);
    let chunks = extractor.extract_chunks(source, &path).unwrap();

    let struct_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Struct);
    assert!(struct_chunk.is_some());
    let struct_chunk = struct_chunk.unwrap();
    assert_eq!(struct_chunk.symbol_name, "Person");
}

#[test]
fn test_extract_impl() {
    let Some(lang) = get_language("rust") else {
        return;
    };
    let source = r#"
impl Person {
    fn new(name: String) -> Self {
        Self { name, age: 0 }
    }

    fn greet(&self) {
        println!("Hello, {}!", self.name);
    }
}
"#;
    let path = PathBuf::from("test.rs");
    let extractor = RustExtractor::with_language(lang);
    let chunks = extractor.extract_chunks(source, &path).unwrap();

    // Should have impl block + methods
    assert!(chunks.len() >= 3);
    let impl_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Impl);
    assert!(impl_chunk.is_some());

    let methods: Vec<_> = chunks
        .iter()
        .filter(|c| c.chunk_type == ChunkType::Method)
        .collect();
    assert_eq!(methods.len(), 2);
}

#[test]
fn test_extract_trait() {
    let Some(lang) = get_language("rust") else {
        return;
    };
    let source = r#"
/// A greeter trait.
pub trait Greeter {
    fn greet(&self);
}
"#;
    let path = PathBuf::from("test.rs");
    let extractor = RustExtractor::with_language(lang);
    let chunks = extractor.extract_chunks(source, &path).unwrap();

    let trait_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Trait);
    assert!(trait_chunk.is_some());
    assert_eq!(trait_chunk.unwrap().symbol_name, "Greeter");
}

#[test]
fn test_extract_preamble() {
    let Some(lang) = get_language("rust") else {
        return;
    };
    let source = r#"
use std::collections::HashMap;
use std::io::Result;

mod utils;

fn main() {}
"#;
    let path = PathBuf::from("test.rs");
    let extractor = RustExtractor::with_language(lang);
    let chunks = extractor.extract_chunks(source, &path).unwrap();

    let preamble = chunks.iter().find(|c| c.chunk_type == ChunkType::Preamble);
    assert!(preamble.is_some());
    let preamble = preamble.unwrap();
    assert!(preamble.content.contains("use std::collections::HashMap"));
    assert!(preamble.content.contains("mod utils"));
}

#[test]
fn test_extract_async_function() {
    let Some(lang) = get_language("rust") else {
        return;
    };
    let source = r#"
async fn fetch_data() -> Result<String> {
    Ok("data".to_string())
}
"#;
    let path = PathBuf::from("test.rs");
    let extractor = RustExtractor::with_language(lang);
    let chunks = extractor.extract_chunks(source, &path).unwrap();

    let async_chunk = chunks
        .iter()
        .find(|c| c.chunk_type == ChunkType::AsyncFunction);
    assert!(async_chunk.is_some());
    assert_eq!(async_chunk.unwrap().symbol_name, "fetch_data");
}

#[test]
fn test_extract_function_calls() {
    let Some(lang) = get_language("rust") else {
        return;
    };
    let source = r#"
fn process() {
    helper();
    validate(data);
    transform();
}
"#;
    let path = PathBuf::from("test.rs");
    let extractor = RustExtractor::with_language(lang);
    let chunks = extractor.extract_chunks(source, &path).unwrap();

    let fn_chunk = chunks.iter().find(|c| c.chunk_type == ChunkType::Function);
    assert!(fn_chunk.is_some());
    let calls = &fn_chunk.unwrap().calls;
    assert!(calls.contains(&"helper".to_string()));
    assert!(calls.contains(&"validate".to_string()));
    assert!(calls.contains(&"transform".to_string()));
}

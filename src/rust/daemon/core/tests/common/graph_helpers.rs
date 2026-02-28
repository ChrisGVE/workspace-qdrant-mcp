//! Shared helpers for graph integration tests.
//!
//! Provides reusable chunk builders and ingestion utilities used across
//! `graph_store_tests`, `graph_algorithm_tests`, `graph_extractor_tests`,
//! and `graph_migration_tests`.

use workspace_qdrant_core::graph::{
    create_sqlite_graph_store, extractor, SharedGraphStore, SqliteGraphStore,
};
use workspace_qdrant_core::tree_sitter::types::{ChunkType, SemanticChunk};

pub const TENANT: &str = "integration-test";

/// Build a realistic Rust file as SemanticChunks for extraction testing.
pub fn build_rust_file_chunks() -> Vec<SemanticChunk> {
    let preamble = SemanticChunk::new(
        ChunkType::Preamble,
        "_preamble",
        "use std::collections::HashMap;\nuse crate::config::Config;",
        1,
        2,
        "rust",
        "src/processor.rs",
    );

    let struct_chunk = SemanticChunk::new(
        ChunkType::Struct,
        "Processor",
        "pub struct Processor {\n    config: Config,\n    cache: HashMap<String, Vec<u8>>,\n}",
        4,
        7,
        "rust",
        "src/processor.rs",
    )
    .with_signature("pub struct Processor");

    let method_process = SemanticChunk::new(
        ChunkType::Method,
        "process",
        "pub fn process(&self, data: &[u8]) -> Result<Output, Error> {\n    self.validate(data)?;\n    self.transform(data)\n}",
        10,
        14,
        "rust",
        "src/processor.rs",
    )
    .with_parent("Processor")
    .with_signature("pub fn process(&self, data: &[u8]) -> Result<Output, Error>")
    .with_calls(vec![
        "self.validate".to_string(),
        "self.transform".to_string(),
    ]);

    let method_validate = SemanticChunk::new(
        ChunkType::Method,
        "validate",
        "fn validate(&self, data: &[u8]) -> Result<(), Error> { Ok(()) }",
        16,
        18,
        "rust",
        "src/processor.rs",
    )
    .with_parent("Processor")
    .with_signature("fn validate(&self, data: &[u8]) -> Result<(), Error>");

    let method_transform = SemanticChunk::new(
        ChunkType::Method,
        "transform",
        "fn transform(&self, data: &[u8]) -> Result<Output, Error> {\n    let output = Output::new(data);\n    Ok(output)\n}",
        20,
        24,
        "rust",
        "src/processor.rs",
    )
    .with_parent("Processor")
    .with_signature("fn transform(&self, data: &[u8]) -> Result<Output, Error>")
    .with_calls(vec!["Output::new".to_string()]);

    vec![
        preamble,
        struct_chunk,
        method_process,
        method_validate,
        method_transform,
    ]
}

/// Build a second Rust file that depends on the first (for cross-file tests).
pub fn build_rust_main_chunks() -> Vec<SemanticChunk> {
    let preamble = SemanticChunk::new(
        ChunkType::Preamble,
        "_preamble",
        "use crate::processor::Processor;",
        1,
        1,
        "rust",
        "src/main.rs",
    );

    let main_fn = SemanticChunk::new(
        ChunkType::Function,
        "main",
        "fn main() {\n    let p = Processor::new();\n    p.process(&data);\n}",
        3,
        6,
        "rust",
        "src/main.rs",
    )
    .with_signature("fn main()")
    .with_calls(vec![
        "Processor::new".to_string(),
        "p.process".to_string(),
    ]);

    let helper_fn = SemanticChunk::new(
        ChunkType::Function,
        "setup_logging",
        "fn setup_logging() {\n    tracing_subscriber::init();\n}",
        8,
        10,
        "rust",
        "src/main.rs",
    )
    .with_signature("fn setup_logging()")
    .with_calls(vec!["tracing_subscriber::init".to_string()]);

    vec![preamble, main_fn, helper_fn]
}

/// Build TypeScript chunks for multi-language extraction testing.
pub fn build_typescript_chunks() -> Vec<SemanticChunk> {
    let preamble = SemanticChunk::new(
        ChunkType::Preamble,
        "_preamble",
        "import { Component, useState } from 'react';",
        1,
        1,
        "typescript",
        "src/App.tsx",
    );

    let class_chunk = SemanticChunk::new(
        ChunkType::Class,
        "AppComponent",
        "class AppComponent extends Component {\n  render() { return <div />; }\n}",
        3,
        5,
        "typescript",
        "src/App.tsx",
    )
    .with_signature("class AppComponent extends Component");

    let method_render = SemanticChunk::new(
        ChunkType::Method,
        "render",
        "render() { return <div />; }",
        4,
        4,
        "typescript",
        "src/App.tsx",
    )
    .with_parent("AppComponent")
    .with_signature("render(): JSX.Element")
    .with_calls(vec!["useState".to_string()]);

    vec![preamble, class_chunk, method_render]
}

/// Create a graph store via factory (on-disk, with schema migration).
pub async fn create_factory_store(dir: &std::path::Path) -> SharedGraphStore<SqliteGraphStore> {
    create_sqlite_graph_store(dir)
        .await
        .expect("factory should create store")
}

/// Extract and ingest chunks into a store.
pub async fn ingest_file_chunks(
    store: &SharedGraphStore<SqliteGraphStore>,
    chunks: &[SemanticChunk],
    tenant_id: &str,
    file_path: &str,
) {
    let result = extractor::extract_edges(chunks, tenant_id, file_path);
    store.upsert_nodes(&result.nodes).await.unwrap();
    store.insert_edges(&result.edges).await.unwrap();
}

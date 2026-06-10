//! Integration test: verify .proto files produce semantic chunks.
//!
//! Regression test for protobuf falling back to plain-text chunking: the
//! registry entry had a grammar but no `semantic_patterns`, so every .proto
//! chunk landed in Qdrant as chunk_type="text" / symbol="_text" and queries
//! like "Where are the gRPC services defined?" never ranked the proto file.
//!
//! Requires network access on first run (downloads and compiles the
//! mitchellh/tree-sitter-proto grammar into the system cache).

use std::path::PathBuf;
use std::sync::Arc;
use workspace_qdrant_core::config::GrammarConfig;
use workspace_qdrant_core::tree_sitter::{
    extract_chunks_with_provider, ChunkType, GrammarManager, SemanticChunk,
};

const PROTO_SOURCE: &str = r#"// Widget definitions for the chunker test.
syntax = "proto3";

package acme.widgets.v1;

import "google/protobuf/timestamp.proto";

option java_package = "com.acme.widgets";


// Catalog of widget kinds.
enum WidgetKind {
  WIDGET_KIND_UNSPECIFIED = 0;
  WIDGET_KIND_GADGET = 1;
}

// A widget in the catalog.
message Widget {
  string id = 1;
  WidgetKind kind = 2;
}

message GetWidgetRequest {
  string id = 1;
}

// Widget lookup APIs.
service WidgetService {
  // Fetch a single widget.
  rpc GetWidget(GetWidgetRequest) returns (Widget) {}
  rpc ListWidgets(GetWidgetRequest) returns (stream Widget);
}
"#;

fn find<'a>(
    chunks: &'a [SemanticChunk],
    chunk_type: ChunkType,
    symbol: &str,
) -> &'a SemanticChunk {
    chunks
        .iter()
        .find(|c| c.chunk_type == chunk_type && c.symbol_name == symbol)
        .unwrap_or_else(|| {
            let inventory: Vec<String> = chunks
                .iter()
                .map(|c| format!("{:?}:{}", c.chunk_type, c.symbol_name))
                .collect();
            panic!("expected {chunk_type:?} chunk named {symbol}, got: {inventory:?}")
        })
}

#[tokio::test]
async fn test_proto_produces_semantic_chunks() {
    let cache_dir = dirs::home_dir().unwrap().join(".workspace-qdrant/grammars");
    let config = GrammarConfig {
        cache_dir,
        auto_download: true,
        ..Default::default()
    };

    let mut manager = GrammarManager::new(config);
    manager
        .get_grammar("protobuf")
        .await
        .expect("protobuf grammar should download and load (exports tree_sitter_proto)");

    let provider = manager.create_language_provider();
    let chunks = extract_chunks_with_provider(
        PROTO_SOURCE,
        &PathBuf::from("widgets.proto"),
        4096,
        Some(Arc::new(provider)),
    )
    .expect("chunking should succeed");

    // The whole point: no plain-text fallback chunks.
    assert!(
        chunks.iter().all(|c| c.chunk_type != ChunkType::Text),
        "proto must not fall back to text chunking: {:?}",
        chunks
            .iter()
            .map(|c| (c.chunk_type, c.symbol_name.clone()))
            .collect::<Vec<_>>()
    );

    // Preamble coalesces syntax/package/import/option.
    let preamble = chunks
        .iter()
        .find(|c| c.chunk_type == ChunkType::Preamble)
        .expect("should extract a preamble chunk");
    assert!(preamble.content.contains("syntax = \"proto3\""));
    assert!(preamble.content.contains("package acme.widgets.v1"));
    assert!(preamble.content.contains("import \"google/protobuf/timestamp.proto\""));

    // Top-level definitions carry real symbol names.
    find(&chunks, ChunkType::Enum, "WidgetKind");
    find(&chunks, ChunkType::Struct, "Widget");
    find(&chunks, ChunkType::Struct, "GetWidgetRequest");
    let service = find(&chunks, ChunkType::Class, "WidgetService");
    assert!(service.content.contains("rpc GetWidget"));

    // rpcs are direct children of `service` (no body wrapper node in the
    // grammar) and must still be extracted as methods of the service.
    let get_widget = find(&chunks, ChunkType::Method, "GetWidget");
    assert_eq!(get_widget.parent_symbol.as_deref(), Some("WidgetService"));
    let list_widgets = find(&chunks, ChunkType::Method, "ListWidgets");
    assert_eq!(list_widgets.parent_symbol.as_deref(), Some("WidgetService"));
}

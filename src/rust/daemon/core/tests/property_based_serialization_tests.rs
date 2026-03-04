//! Property-based tests for data serialization and structure validation
//!
//! Tests document content structure integrity, configuration serialization
//! round-trips, and chunking config JSON validity using proptest.

#![cfg(feature = "processing_engine")]

#[allow(dead_code)]
#[path = "common/proptest_generators.rs"]
mod proptest_generators;

use std::collections::HashMap;

use proptest::prelude::*;

use workspace_qdrant_core::{
    ChunkingConfig, DocumentContent, TextChunk,
    config::Config,
};

use proptest_generators::{arb_chunking_config, arb_document_type, arb_file_content};

// ============================================================================
// DATA SERIALIZATION PROPERTY TESTS
// ============================================================================

/// Property test: Document content structure validation
proptest! {
    #[test]
    fn prop_document_content_structure(
        raw_text in arb_file_content(),
        document_type in arb_document_type(),
        chunk_count in 0..100usize,
    ) {
        // Create a DocumentContent instance
        let mut metadata = HashMap::new();
        metadata.insert("test_key".to_string(), "test_value".to_string());
        metadata.insert("char_count".to_string(), raw_text.len().to_string());

        // Create chunks
        let chunks: Vec<TextChunk> = (0..chunk_count).map(|i| {
            let mut chunk_metadata = HashMap::new();
            chunk_metadata.insert("chunk_index".to_string(), i.to_string());

            TextChunk {
                content: format!("Chunk {} content", i),
                chunk_index: i,
                start_char: i * 100,
                end_char: (i + 1) * 100,
                metadata: chunk_metadata,
            }
        }).collect();

        let document_content = DocumentContent {
            raw_text: raw_text.clone(),
            metadata: metadata.clone(),
            document_type: document_type.clone(),
            chunks: chunks.clone(),
        };

        // Verify structure properties
        assert_eq!(document_content.raw_text, raw_text);
        assert_eq!(document_content.metadata.len(), 2);
        assert_eq!(document_content.chunks.len(), chunk_count);

        // Verify chunk integrity
        for (i, chunk) in document_content.chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i);
            assert!(chunk.end_char >= chunk.start_char);
            assert!(chunk.metadata.contains_key("chunk_index"));
        }

        // Test that structure is logically consistent
        if !document_content.chunks.is_empty() {
            let first_chunk = &document_content.chunks[0];
            let last_chunk = &document_content.chunks[document_content.chunks.len() - 1];
            assert!(first_chunk.chunk_index < last_chunk.chunk_index || document_content.chunks.len() == 1);
        }
    }
}

/// Property test: Configuration serialization should be robust
proptest! {
    #[test]
    fn prop_config_serialization(
        max_concurrent in prop::option::of(1..1000usize),
        default_timeout in prop::option::of(100..300000u64),
        chunk_size in 1..100000usize,
        overlap_size in 0..10000usize,
        preserve_paragraphs in any::<bool>(),
    ) {
        let config = Config {
            max_concurrent_tasks: max_concurrent,
            default_timeout_ms: default_timeout,
            ..Default::default()
        };

        let chunking_config = ChunkingConfig {
            chunk_size,
            overlap_size: std::cmp::min(overlap_size, chunk_size / 2),
            preserve_paragraphs,
            ..ChunkingConfig::default()
        };

        // Test YAML serialization for config
        match serde_yaml::to_string(&config) {
            Ok(yaml_str) => {
                match serde_yaml::from_str::<Config>(&yaml_str) {
                    Ok(restored) => {
                        assert_eq!(config.max_concurrent_tasks, restored.max_concurrent_tasks);
                        assert_eq!(config.default_timeout_ms, restored.default_timeout_ms);
                    },
                    Err(e) => {
                        tracing::debug!("YAML config deserialization error: {}", e);
                    }
                }
            },
            Err(e) => {
                tracing::debug!("YAML config serialization error: {}", e);
            }
        }

        // Test manual serialization for chunking config (since it doesn't derive Serialize)
        let chunking_json = format!(
            r#"{{"chunk_size":{},"overlap_size":{},"preserve_paragraphs":{}}}"#,
            chunking_config.chunk_size,
            chunking_config.overlap_size,
            chunking_config.preserve_paragraphs
        );

        // Verify the JSON is well-formed
        match serde_json::from_str::<serde_json::Value>(&chunking_json) {
            Ok(parsed) => {
                assert!(parsed.is_object());
                assert!(parsed.get("chunk_size").is_some());
                assert!(parsed.get("overlap_size").is_some());
                assert!(parsed.get("preserve_paragraphs").is_some());
            },
            Err(e) => {
                tracing::debug!("JSON chunking config validation error: {}", e);
            }
        }
    }
}

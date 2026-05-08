//! AST node walking and chunk-type dispatch for the generic extractor.
//!
//! Provides `walk_children` (the main recursive descent) plus handlers for
//! decorated nodes and Elixir-style `call` nodes. All functions take
//! `&SemanticPatterns` and extractor callbacks explicitly to remain
//! stateless with respect to `GenericExtractor` ownership.

use tree_sitter::Node;

use crate::language_registry::types::SemanticPatterns;
use crate::tree_sitter::chunker::helpers::node_text;
use crate::tree_sitter::types::{ChunkType, SemanticChunk};

/// Return `true` if `kind` is present in `types`.
pub(super) fn matches_any(kind: &str, types: &[String]) -> bool {
    types.iter().any(|t| t == kind)
}

/// Map an AST node kind to a `ChunkType` using the configured patterns.
///
/// Returns `None` if the kind is not in any pattern group.
pub(super) fn classify_node(patterns: &SemanticPatterns, kind: &str) -> Option<ChunkType> {
    if matches_any(kind, &patterns.function.node_types)
        || matches_any(kind, &patterns.function.async_node_types)
    {
        return Some(ChunkType::Function);
    }
    if matches_any(kind, &patterns.class.node_types) {
        return Some(ChunkType::Class);
    }
    if matches_any(kind, &patterns.struct_def.node_types) {
        return Some(ChunkType::Struct);
    }
    if matches_any(kind, &patterns.enum_def.node_types) {
        return Some(ChunkType::Enum);
    }
    if matches_any(kind, &patterns.trait_def.node_types) {
        return Some(ChunkType::Trait);
    }
    if matches_any(kind, &patterns.interface.node_types) {
        return Some(ChunkType::Interface);
    }
    if matches_any(kind, &patterns.module.node_types) {
        return Some(ChunkType::Module);
    }
    if matches_any(kind, &patterns.constant.node_types) {
        return Some(ChunkType::Constant);
    }
    if matches_any(kind, &patterns.macro_def.node_types) {
        return Some(ChunkType::Macro);
    }
    if matches_any(kind, &patterns.type_alias.node_types) {
        return Some(ChunkType::TypeAlias);
    }
    if matches_any(kind, &patterns.impl_block.node_types) {
        return Some(ChunkType::Impl);
    }
    None
}

/// Walk children of `parent`, classifying each node and collecting chunks.
///
/// `root_wrappers` nodes are unwrapped transparently (recursed into without
/// emitting a chunk). `decorated_wrapper` and Elixir `call` nodes get
/// dedicated handlers.
///
/// `extract_container_fn` and `extract_definition_fn` are closures that
/// delegate back to `GenericExtractor` methods so that `walker.rs` has no
/// direct dependency on the extractor struct.
pub(super) fn walk_children<F1, F2>(
    patterns: &SemanticPatterns,
    parent: &Node,
    source: &str,
    file_path: &str,
    chunks: &mut Vec<SemanticChunk>,
    extract_container_fn: &F1,
    extract_definition_fn: &F2,
) where
    F1: Fn(&Node, &str, &str, ChunkType) -> Vec<SemanticChunk>,
    F2: Fn(&Node, &str, &str, ChunkType, Option<&str>) -> SemanticChunk,
{
    let decorated_wrapper = patterns.decorated_wrapper.as_deref();
    let wrappers = &patterns.root_wrappers;
    let mut cursor = parent.walk();

    for child in parent.children(&mut cursor) {
        let kind = child.kind();

        if wrappers.iter().any(|w| w == kind) {
            walk_children(
                patterns,
                &child,
                source,
                file_path,
                chunks,
                extract_container_fn,
                extract_definition_fn,
            );
            continue;
        }

        if let Some(chunk_type) = classify_node(patterns, kind) {
            match chunk_type {
                ChunkType::Class
                | ChunkType::Struct
                | ChunkType::Trait
                | ChunkType::Interface
                | ChunkType::Module
                | ChunkType::Impl => {
                    chunks.extend(extract_container_fn(&child, source, file_path, chunk_type));
                }
                _ => {
                    chunks.push(extract_definition_fn(
                        &child, source, file_path, chunk_type, None,
                    ));
                }
            }
        } else if Some(kind) == decorated_wrapper {
            handle_decorated_node(
                patterns,
                &child,
                source,
                file_path,
                chunks,
                extract_container_fn,
                extract_definition_fn,
            );
        } else if kind == "call" {
            handle_call_node(
                patterns,
                &child,
                source,
                file_path,
                chunks,
                extract_container_fn,
                extract_definition_fn,
            );
        }
    }
}

fn handle_decorated_node<F1, F2>(
    patterns: &SemanticPatterns,
    node: &Node,
    source: &str,
    file_path: &str,
    chunks: &mut Vec<SemanticChunk>,
    extract_container_fn: &F1,
    extract_definition_fn: &F2,
) where
    F1: Fn(&Node, &str, &str, ChunkType) -> Vec<SemanticChunk>,
    F2: Fn(&Node, &str, &str, ChunkType, Option<&str>) -> SemanticChunk,
{
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        if let Some(chunk_type) = classify_node(patterns, kind) {
            match chunk_type {
                ChunkType::Class | ChunkType::Struct | ChunkType::Module => {
                    chunks.extend(extract_container_fn(&child, source, file_path, chunk_type));
                }
                _ => {
                    chunks.push(extract_definition_fn(
                        &child, source, file_path, chunk_type, None,
                    ));
                }
            }
            return;
        }
    }
}

fn handle_call_node<F1, F2>(
    patterns: &SemanticPatterns,
    node: &Node,
    source: &str,
    file_path: &str,
    chunks: &mut Vec<SemanticChunk>,
    extract_container_fn: &F1,
    extract_definition_fn: &F2,
) where
    F1: Fn(&Node, &str, &str, ChunkType) -> Vec<SemanticChunk>,
    F2: Fn(&Node, &str, &str, ChunkType, Option<&str>) -> SemanticChunk,
{
    let text = node_text(node, source);
    let first_word = text.split_whitespace().next().unwrap_or("");

    if matches_any("call", &patterns.module.node_types) && matches!(first_word, "defmodule") {
        chunks.extend(extract_container_fn(
            node,
            source,
            file_path,
            ChunkType::Module,
        ));
    } else if matches_any("call", &patterns.function.node_types)
        && matches!(first_word, "def" | "defp")
    {
        chunks.push(extract_definition_fn(
            node,
            source,
            file_path,
            ChunkType::Function,
            None,
        ));
    } else if matches_any("call", &patterns.macro_def.node_types)
        && matches!(first_word, "defmacro" | "defmacrop")
    {
        chunks.push(extract_definition_fn(
            node,
            source,
            file_path,
            ChunkType::Macro,
            None,
        ));
    }
}

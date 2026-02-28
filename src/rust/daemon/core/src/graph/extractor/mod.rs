//! Graph relationship extractor -- derives graph edges from SemanticChunk data.
//!
//! Takes tree-sitter `SemanticChunk` output and produces `GraphNode`/`GraphEdge`
//! pairs for CONTAINS, CALLS, IMPORTS, and USES_TYPE relationships.

pub(crate) mod import_parsers;
mod type_analysis;

#[cfg(test)]
mod tests;

use crate::tree_sitter::types::{ChunkType, SemanticChunk};
use crate::TextChunk;

use super::{EdgeType, GraphEdge, GraphNode, NodeType};

pub use type_analysis::{extract_type_references, parse_qualified_name};
use import_parsers::extract_imports_from_content;

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

        // CONTAINS edges: parent_symbol -> this chunk
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

        // CALLS edges: this chunk -> called functions
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

/// Extract graph nodes and edges from `TextChunk` metadata maps.
///
/// This is the pipeline-integrated entry point. The document processor
/// converts `SemanticChunk` data to `TextChunk` with metadata strings
/// (`chunk_type`, `symbol_name`, `parent_symbol`, `calls`, `signature`,
/// `language`, `start_line`, `end_line`). This function reconstructs
/// graph relationships from those metadata maps.
pub fn extract_edges_from_text_chunks(
    chunks: &[TextChunk],
    tenant_id: &str,
    file_path: &str,
) -> ExtractionResult {
    let mut result = ExtractionResult::default();

    // Create a File node for import edges
    let file_node = GraphNode::new(tenant_id, file_path, file_path, NodeType::File);
    result.nodes.push(file_node);

    for chunk in chunks {
        let meta = &chunk.metadata;

        // Determine node type from chunk_type metadata
        let chunk_type_str = match meta.get("chunk_type") {
            Some(s) => s.as_str(),
            None => continue, // No chunk_type means not a semantic chunk
        };
        let Some(node_type) = node_type_from_display_name(chunk_type_str) else {
            // Preamble and text chunks: extract imports from preamble content
            if chunk_type_str == "preamble" {
                let language = meta.get("language").map(|s| s.as_str()).unwrap_or("");
                extract_imports_from_content(
                    &chunk.content,
                    language,
                    tenant_id,
                    file_path,
                    &mut result,
                );
            }
            continue;
        };

        let symbol_name = match meta.get("symbol_name") {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        let language = meta.get("language").cloned().unwrap_or_default();

        // Create the node
        let mut node = GraphNode::new(tenant_id, file_path, symbol_name, node_type);
        node.start_line = meta.get("start_line").and_then(|s| s.parse::<u32>().ok());
        node.end_line = meta.get("end_line").and_then(|s| s.parse::<u32>().ok());
        node.signature = meta.get("signature").cloned();
        node.language = Some(language.clone());
        result.nodes.push(node.clone());

        // CONTAINS edges
        if let Some(parent) = meta.get("parent_symbol") {
            if !parent.is_empty() {
                let parent_type = infer_parent_node_type(parent, &language);
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
        }

        // CALLS edges (comma-separated in metadata)
        if let Some(calls_str) = meta.get("calls") {
            for call in calls_str.split(',') {
                let call = call.trim();
                if call.is_empty() {
                    continue;
                }
                let (_qualifier, callee_name) = parse_qualified_name(call);
                if callee_name.is_empty() {
                    continue;
                }
                let callee_stub =
                    GraphNode::stub(tenant_id, &callee_name, NodeType::Function);
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
        }

        // USES_TYPE edges
        if let Some(sig) = meta.get("signature") {
            let type_refs = extract_type_references(sig, &language);
            for type_name in type_refs {
                let type_stub =
                    GraphNode::stub(tenant_id, &type_name, NodeType::Struct);
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

/// Convert a `ChunkType::display_name()` string back to `NodeType`.
fn node_type_from_display_name(name: &str) -> Option<NodeType> {
    match name {
        "function" => Some(NodeType::Function),
        "async_function" => Some(NodeType::AsyncFunction),
        "class" => Some(NodeType::Class),
        "method" => Some(NodeType::Method),
        "struct" => Some(NodeType::Struct),
        "trait" => Some(NodeType::Trait),
        "interface" => Some(NodeType::Interface),
        "enum" => Some(NodeType::Enum),
        "impl" => Some(NodeType::Impl),
        "module" => Some(NodeType::Module),
        "constant" => Some(NodeType::Constant),
        "type_alias" => Some(NodeType::TypeAlias),
        "macro" => Some(NodeType::Macro),
        "preamble" | "text" => None,
        _ => None,
    }
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

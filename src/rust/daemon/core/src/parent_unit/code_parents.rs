//! Code-file parent mapping: block-level parents and chunk-to-parent assignment.

use std::collections::HashMap;

use crate::tree_sitter::types::{ChunkType, SemanticChunk};

use super::types::{code_block_parent, code_file_parent, ParentUnitRecord};

/// Result of creating parent records for a code file's semantic chunks.
///
/// Maps each chunk index to its nearest parent point ID (block parent if inside
/// a class/struct/impl, file parent otherwise).
#[derive(Debug, Clone)]
pub struct CodeParentMapping {
    /// The file-level parent record
    pub file_parent: ParentUnitRecord,
    /// Block-level parent records (class, struct, impl, etc.)
    pub block_parents: Vec<ParentUnitRecord>,
    /// For each chunk index, the parent point_id it should reference
    pub chunk_parent_ids: Vec<String>,
}

/// Container-level chunk types that warrant block-level parent records.
pub(crate) fn is_container_type(chunk_type: ChunkType) -> bool {
    matches!(
        chunk_type,
        ChunkType::Class
            | ChunkType::Struct
            | ChunkType::Trait
            | ChunkType::Interface
            | ChunkType::Impl
            | ChunkType::Module
            | ChunkType::Enum
    )
}

/// Create parent records for a code file and its semantic chunks.
///
/// Produces:
/// 1. A file-level parent (full file text, no vector)
/// 2. Block-level parents for each container (class/struct/impl/trait/module/enum)
/// 3. A mapping from each chunk index → its nearest parent point_id
///
/// Chunks whose `parent_symbol` matches a container block get the block's point_id;
/// all others get the file parent's point_id.
pub fn create_code_parents(
    doc_id: &str,
    doc_fingerprint: &str,
    file_path: &str,
    file_text: &str,
    chunks: &[SemanticChunk],
) -> CodeParentMapping {
    let file_parent = code_file_parent(doc_id, doc_fingerprint, file_path, file_text);

    // Collect container blocks and build name → point_id index
    let mut block_parents = Vec::new();
    let mut block_name_to_id: HashMap<String, String> = HashMap::new();

    for chunk in chunks {
        if is_container_type(chunk.chunk_type) {
            let block = code_block_parent(
                doc_id,
                doc_fingerprint,
                file_path,
                &chunk.symbol_name,
                chunk.chunk_type.display_name(),
                chunk.start_line,
                chunk.end_line,
                &chunk.content,
            );
            block_name_to_id.insert(chunk.symbol_name.clone(), block.point_id.clone());
            block_parents.push(block);
        }
    }

    // Map each chunk to its nearest parent
    let chunk_parent_ids = chunks
        .iter()
        .map(|chunk| {
            // If this chunk has a parent_symbol that matches a block parent, use it
            if let Some(ref parent_sym) = chunk.parent_symbol {
                if let Some(block_id) = block_name_to_id.get(parent_sym) {
                    return block_id.clone();
                }
            }
            // If this chunk IS a container, it references the file parent
            // (the container itself is stored as a block parent, its chunks are children)
            file_parent.point_id.clone()
        })
        .collect();

    CodeParentMapping {
        file_parent,
        block_parents,
        chunk_parent_ids,
    }
}


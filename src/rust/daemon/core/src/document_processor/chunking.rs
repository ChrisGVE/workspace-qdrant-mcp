//! Text chunking strategies for document processing.

use std::collections::HashMap;

use crate::tree_sitter::SemanticChunk;
use crate::{ChunkingConfig, TextChunk};

/// Find the largest byte index <= `index` that is a valid UTF-8 char boundary.
/// This prevents panics when slicing strings at byte offsets calculated from
/// `len()` arithmetic, which can land inside multi-byte characters.
pub fn floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        s.len()
    } else {
        let mut i = index;
        while i > 0 && !s.is_char_boundary(i) {
            i -= 1;
        }
        i
    }
}

/// Chunk text into smaller pieces with overlap
pub fn chunk_text(
    text: &str,
    base_metadata: &HashMap<String, String>,
    config: &ChunkingConfig,
) -> Vec<TextChunk> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();

    if config.preserve_paragraphs {
        chunk_by_paragraphs(text, base_metadata, &mut chunks, config);
    } else {
        chunk_by_characters(text, base_metadata, &mut chunks, config);
    }

    chunks
}

/// Chunk text preserving paragraph boundaries
pub fn chunk_by_paragraphs(
    text: &str,
    base_metadata: &HashMap<String, String>,
    chunks: &mut Vec<TextChunk>,
    config: &ChunkingConfig,
) {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();

    let mut current_chunk = String::new();
    let mut current_start = 0;
    let mut chunk_index = 0;

    for paragraph in paragraphs {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }

        if !current_chunk.is_empty()
            && current_chunk.len() + paragraph.len() + 2 > config.chunk_size
        {
            let mut chunk_metadata = base_metadata.clone();
            chunk_metadata.insert("chunk_index".to_string(), chunk_index.to_string());

            chunks.push(TextChunk {
                content: current_chunk.clone(),
                chunk_index,
                start_char: current_start,
                end_char: current_start + current_chunk.len(),
                metadata: chunk_metadata,
            });

            chunk_index += 1;

            let overlap_start = current_chunk.len().saturating_sub(config.overlap_size);
            let overlap_start = floor_char_boundary(&current_chunk, overlap_start);
            current_chunk = current_chunk[overlap_start..].to_string();
            current_start += overlap_start;
        }

        if !current_chunk.is_empty() {
            current_chunk.push_str("\n\n");
        }
        current_chunk.push_str(paragraph);
    }

    if !current_chunk.is_empty() {
        let mut chunk_metadata = base_metadata.clone();
        chunk_metadata.insert("chunk_index".to_string(), chunk_index.to_string());

        chunks.push(TextChunk {
            content: current_chunk.clone(),
            chunk_index,
            start_char: current_start,
            end_char: current_start + current_chunk.len(),
            metadata: chunk_metadata,
        });
    }
}

/// Simple character-based chunking
pub fn chunk_by_characters(
    text: &str,
    base_metadata: &HashMap<String, String>,
    chunks: &mut Vec<TextChunk>,
    config: &ChunkingConfig,
) {
    let total_chars = text.len();
    let mut start = 0;
    let mut chunk_index = 0;

    while start < total_chars {
        let end = floor_char_boundary(text, (start + config.chunk_size).min(total_chars));

        let actual_end = if end < total_chars {
            text[start..end]
                .rfind(char::is_whitespace)
                .map(|pos| start + pos)
                .filter(|&pos| pos > start) // Avoid zero-progress when whitespace is at start
                .unwrap_or(end)
        } else {
            end
        };

        let chunk_text = text[start..actual_end].trim().to_string();

        if !chunk_text.is_empty() {
            let mut chunk_metadata = base_metadata.clone();
            chunk_metadata.insert("chunk_index".to_string(), chunk_index.to_string());

            chunks.push(TextChunk {
                content: chunk_text,
                chunk_index,
                start_char: start,
                end_char: actual_end,
                metadata: chunk_metadata,
            });

            chunk_index += 1;
        }

        let new_start = floor_char_boundary(text, actual_end.saturating_sub(config.overlap_size));
        if new_start <= start {
            start = actual_end; // Guarantee forward progress
        } else {
            start = new_start;
        }
    }
}

/// Convert SemanticChunks to TextChunks with semantic metadata preserved
pub fn convert_semantic_chunks_to_text_chunks(
    semantic_chunks: Vec<SemanticChunk>,
    base_metadata: &HashMap<String, String>,
) -> Vec<TextChunk> {
    semantic_chunks
        .into_iter()
        .enumerate()
        .map(|(idx, chunk)| {
            let mut metadata = base_metadata.clone();

            // Add semantic chunk metadata (Task 4 requirements)
            metadata.insert(
                "chunk_type".to_string(),
                chunk.chunk_type.display_name().to_string(),
            );
            metadata.insert("symbol_name".to_string(), chunk.symbol_name.clone());
            metadata.insert("symbol_kind".to_string(), chunk.symbol_kind.clone());

            // Add parent symbol for methods (key for Task 4)
            if let Some(ref parent) = chunk.parent_symbol {
                metadata.insert("parent_symbol".to_string(), parent.clone());
            }

            // Add signature if available
            if let Some(ref sig) = chunk.signature {
                metadata.insert("signature".to_string(), sig.clone());
            }

            // Add docstring if available
            if let Some(ref doc) = chunk.docstring {
                metadata.insert("docstring".to_string(), doc.clone());
            }

            // Add function calls if any
            if !chunk.calls.is_empty() {
                metadata.insert("calls".to_string(), chunk.calls.join(","));
            }

            // Add line range
            metadata.insert("start_line".to_string(), chunk.start_line.to_string());
            metadata.insert("end_line".to_string(), chunk.end_line.to_string());
            metadata.insert("language".to_string(), chunk.language.clone());

            // Add fragment info if applicable
            if chunk.is_fragment {
                metadata.insert("is_fragment".to_string(), "true".to_string());
                if let Some(frag_idx) = chunk.fragment_index {
                    metadata.insert("fragment_index".to_string(), frag_idx.to_string());
                }
                if let Some(total) = chunk.total_fragments {
                    metadata.insert("total_fragments".to_string(), total.to_string());
                }
            }

            TextChunk {
                content: chunk.content,
                chunk_index: idx,
                start_char: 0, // Line-based, not char-based
                end_char: 0,   // Line-based, not char-based
                metadata,
            }
        })
        .collect()
}

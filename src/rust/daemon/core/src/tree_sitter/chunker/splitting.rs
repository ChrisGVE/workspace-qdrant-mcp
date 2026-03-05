//! Chunk splitting and boundary detection.
//!
//! Handles splitting oversized semantic chunks into smaller fragments with
//! overlap, and provides text-based fallback chunking for unsupported languages.

use std::path::Path;

use crate::tree_sitter::types::SemanticChunk;

/// Overlap size for fragmented chunks (in characters).
pub(super) const FRAGMENT_OVERLAP: usize = 500;

/// Round a byte index down to the nearest UTF-8 char boundary.
/// Prevents panics when slicing `&str` at byte offsets computed from
/// arithmetic on `content.len()` (which counts bytes, not chars).
pub(crate) fn safe_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Handle chunks that exceed the maximum size by splitting them.
pub(super) fn handle_oversized_chunks(
    chunks: Vec<SemanticChunk>,
    _source: &str,
    max_chunk_size: usize,
) -> Vec<SemanticChunk> {
    let mut result = Vec::new();

    for chunk in chunks {
        if chunk.estimated_tokens() > max_chunk_size {
            // Split into fragments with overlap
            let fragments = split_chunk_with_overlap(&chunk, max_chunk_size);
            result.extend(fragments);
        } else {
            result.push(chunk);
        }
    }

    result
}

/// Split a large chunk into smaller fragments with overlap.
///
/// All byte-offset arithmetic uses `safe_char_boundary` to avoid panics
/// on multi-byte UTF-8 characters (e.g., box-drawing chars, CJK, emoji).
fn split_chunk_with_overlap(chunk: &SemanticChunk, max_chunk_size: usize) -> Vec<SemanticChunk> {
    let content = &chunk.content;
    let target_size = max_chunk_size * 4; // Convert tokens to chars (approximate)

    if content.len() <= target_size {
        return vec![chunk.clone()];
    }

    let mut fragments = Vec::new();
    let mut start = 0;
    // Use saturating_sub to avoid overflow when target_size < FRAGMENT_OVERLAP
    let step_size = target_size.saturating_sub(FRAGMENT_OVERLAP).max(1);
    let total_fragments = (content.len() + step_size - 1) / step_size;

    let mut fragment_index = 0;
    while start < content.len() {
        let end = safe_char_boundary(content, (start + target_size).min(content.len()));

        // Try to break at a line boundary
        let actual_end = find_line_boundary(content, start, end);

        let fragment_content = &content[start..actual_end];

        // Calculate line numbers for this fragment
        let lines_before_start = content[..start].matches('\n').count();
        let lines_in_fragment = fragment_content.matches('\n').count();
        let fragment_start_line = chunk.start_line + lines_before_start;
        let fragment_end_line = fragment_start_line + lines_in_fragment;

        let mut fragment = SemanticChunk::new(
            chunk.chunk_type,
            &chunk.symbol_name,
            fragment_content,
            fragment_start_line,
            fragment_end_line,
            &chunk.language,
            &chunk.file_path,
        )
        .as_fragment(fragment_index, total_fragments);

        // Preserve metadata from original chunk
        if let Some(ref parent) = chunk.parent_symbol {
            fragment = fragment.with_parent(parent.clone());
        }
        if fragment_index == 0 {
            // Only first fragment gets docstring and signature
            if let Some(ref doc) = chunk.docstring {
                fragment = fragment.with_docstring(doc.clone());
            }
            if let Some(ref sig) = chunk.signature {
                fragment = fragment.with_signature(sig.clone());
            }
        }

        fragments.push(fragment);

        fragment_index += 1;

        // Move start forward, with overlap if there's more content
        if actual_end >= content.len() {
            break;
        }
        // Use saturating_sub to avoid overflow
        let overlap = if actual_end > FRAGMENT_OVERLAP {
            FRAGMENT_OVERLAP
        } else {
            0
        };
        start = safe_char_boundary(content, actual_end - overlap);

        // Safety check to prevent infinite loops
        if start >= content.len() || start >= actual_end {
            break;
        }
    }

    fragments
}

/// Find the best line boundary for splitting content.
///
/// If `end` is not at end-of-content, search backwards from `end` for a
/// newline within the `[start..end]` range, returning the position after it.
/// Falls back to `end` if no newline is found.
fn find_line_boundary(content: &str, start: usize, end: usize) -> usize {
    if end < content.len() {
        content[start..end]
            .rfind('\n')
            .map(|i| start + i + 1)
            .unwrap_or(end)
    } else {
        end
    }
}

/// Create text chunks as a fallback for unsupported languages or parse errors.
pub fn text_chunk_fallback(
    source: &str,
    file_path: &Path,
    max_chunk_size: usize,
) -> Vec<SemanticChunk> {
    let language = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("text");
    let file_path_str = file_path.to_string_lossy().to_string();
    let target_chars = max_chunk_size * 4; // Approximate chars per token

    if source.len() <= target_chars {
        // Single chunk for small files
        let end_line = source.matches('\n').count() + 1;
        return vec![SemanticChunk::text(
            source,
            1,
            end_line,
            language,
            file_path_str,
        )];
    }

    // Split into chunks by lines
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut chunk_start_line = 1;
    let mut current_line = 1;

    for line in source.lines() {
        let would_exceed = (current_chunk.len() + line.len() + 1) > target_chars;

        if would_exceed && !current_chunk.is_empty() {
            // Emit current chunk
            let end_line = current_line - 1;
            chunks.push(SemanticChunk::text(
                &current_chunk,
                chunk_start_line,
                end_line,
                language,
                &file_path_str,
            ));
            current_chunk.clear();
            chunk_start_line = current_line;
        }

        if !current_chunk.is_empty() {
            current_chunk.push('\n');
        }
        current_chunk.push_str(line);
        current_line += 1;
    }

    // Emit final chunk
    if !current_chunk.is_empty() {
        let end_line = current_line - 1;
        chunks.push(SemanticChunk::text(
            &current_chunk,
            chunk_start_line,
            end_line,
            language,
            &file_path_str,
        ));
    }

    chunks
}

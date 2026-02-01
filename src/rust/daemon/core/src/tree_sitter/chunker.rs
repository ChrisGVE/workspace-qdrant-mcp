//! Semantic chunker that extracts meaningful code units.

use std::path::Path;

use tree_sitter::Node;

use super::languages::{
    CExtractor, CppExtractor, GoExtractor, JavaExtractor, JavaScriptExtractor, PythonExtractor,
    RustExtractor, TypeScriptExtractor,
};
use super::types::{ChunkExtractor, SemanticChunk};
#[cfg(test)]
use super::types::ChunkType;
use crate::error::DaemonError;

/// Default maximum chunk size in estimated tokens.
const DEFAULT_MAX_CHUNK_SIZE: usize = 8000;

/// Overlap size for fragmented chunks (in characters).
const FRAGMENT_OVERLAP: usize = 500;

/// Semantic chunker that extracts code units from source files.
pub struct SemanticChunker {
    max_chunk_size: usize,
}

impl SemanticChunker {
    /// Create a new semantic chunker with the specified max chunk size.
    pub fn new(max_chunk_size: usize) -> Self {
        Self { max_chunk_size }
    }

    /// Create a chunker with default settings.
    pub fn default() -> Self {
        Self::new(DEFAULT_MAX_CHUNK_SIZE)
    }

    /// Chunk source code using the appropriate language extractor.
    pub fn chunk_source(
        &self,
        source: &str,
        file_path: &Path,
        language: &str,
    ) -> Result<Vec<SemanticChunk>, DaemonError> {
        // Get the appropriate extractor
        let extractor: Box<dyn ChunkExtractor> = match language {
            "rust" => Box::new(RustExtractor::new()),
            "python" => Box::new(PythonExtractor::new()),
            "javascript" | "jsx" => Box::new(JavaScriptExtractor::new()),
            "typescript" | "tsx" => Box::new(TypeScriptExtractor::new(language == "tsx")),
            "go" => Box::new(GoExtractor::new()),
            "java" => Box::new(JavaExtractor::new()),
            "c" => Box::new(CExtractor::new()),
            "cpp" => Box::new(CppExtractor::new()),
            _ => {
                // Fall back to text chunking
                return Ok(text_chunk_fallback(source, file_path, self.max_chunk_size));
            }
        };

        // Extract chunks using the language-specific extractor
        let mut chunks = extractor.extract_chunks(source, file_path)?;

        // Process chunks that exceed the size limit
        chunks = self.handle_oversized_chunks(chunks, source);

        Ok(chunks)
    }

    /// Handle chunks that exceed the maximum size by splitting them.
    fn handle_oversized_chunks(
        &self,
        chunks: Vec<SemanticChunk>,
        _source: &str,
    ) -> Vec<SemanticChunk> {
        let mut result = Vec::new();

        for chunk in chunks {
            if chunk.estimated_tokens() > self.max_chunk_size {
                // Split into fragments with overlap
                let fragments = self.split_chunk_with_overlap(&chunk);
                result.extend(fragments);
            } else {
                result.push(chunk);
            }
        }

        result
    }

    /// Split a large chunk into smaller fragments with overlap.
    fn split_chunk_with_overlap(&self, chunk: &SemanticChunk) -> Vec<SemanticChunk> {
        let content = &chunk.content;
        let target_size = self.max_chunk_size * 4; // Convert tokens to chars (approximate)

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
            let end = (start + target_size).min(content.len());

            // Try to break at a line boundary
            let actual_end = if end < content.len() {
                content[start..end]
                    .rfind('\n')
                    .map(|i| start + i + 1)
                    .unwrap_or(end)
            } else {
                end
            };

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
            start = actual_end - overlap;

            // Safety check to prevent infinite loops
            if start >= content.len() || start >= actual_end {
                break;
            }
        }

        fragments
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

    // Split into chunks by paragraphs (double newlines) or lines
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

/// Helper to extract text from a node.
pub fn node_text<'a>(node: &Node, source: &'a str) -> &'a str {
    let start = node.start_byte();
    let end = node.end_byte();
    &source[start..end]
}

/// Helper to find a child node by kind.
pub fn find_child_by_kind<'a>(node: &'a Node<'a>, kind: &str) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == kind {
            return Some(child);
        }
    }
    None
}

/// Helper to find all children of a specific kind.
pub fn find_children_by_kind<'a>(node: &'a Node<'a>, kind: &str) -> Vec<Node<'a>> {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .filter(|child| child.kind() == kind)
        .collect()
}

/// Helper to extract function calls from a node.
pub fn extract_function_calls(node: &Node, source: &str) -> Vec<String> {
    let mut calls = Vec::new();
    let mut cursor = node.walk();

    fn visit(node: &Node, source: &str, calls: &mut Vec<String>, cursor: &mut tree_sitter::TreeCursor) {
        match node.kind() {
            "call_expression" | "function_call" | "invocation_expression" | "call" => {
                // Try to get the function name
                if let Some(callee) = node.child_by_field_name("function")
                    .or_else(|| node.child_by_field_name("callee"))
                    .or_else(|| node.child(0))
                {
                    let name = node_text(&callee, source);
                    // Extract just the function name (not the full path)
                    let clean_name = name
                        .rsplit("::")
                        .next()
                        .unwrap_or(name)
                        .rsplit('.')
                        .next()
                        .unwrap_or(name);
                    if !clean_name.is_empty() && !calls.contains(&clean_name.to_string()) {
                        calls.push(clean_name.to_string());
                    }
                }
            }
            _ => {}
        }

        // Visit children
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i) {
                visit(&child, source, calls, cursor);
            }
        }
    }

    visit(node, source, &mut calls, &mut cursor);
    calls
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_text_chunk_fallback_small() {
        let source = "hello\nworld";
        let path = PathBuf::from("test.txt");
        let chunks = text_chunk_fallback(source, &path, 8000);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type, ChunkType::Text);
        assert_eq!(chunks[0].start_line, 1);
        // end_line is the last line processed
        assert!(chunks[0].end_line >= 1);
    }

    #[test]
    fn test_text_chunk_fallback_large() {
        let source = "line\n".repeat(10000);
        let path = PathBuf::from("test.txt");
        let chunks = text_chunk_fallback(&source, &path, 100); // Small max size

        assert!(chunks.len() > 1);
        // All chunks should be text type
        for chunk in &chunks {
            assert_eq!(chunk.chunk_type, ChunkType::Text);
        }
    }

    #[test]
    fn test_chunker_creation() {
        let chunker = SemanticChunker::new(4000);
        assert_eq!(chunker.max_chunk_size, 4000);

        let chunker = SemanticChunker::default();
        assert_eq!(chunker.max_chunk_size, DEFAULT_MAX_CHUNK_SIZE);
    }

    #[test]
    fn test_split_chunk_with_overlap() {
        // Use a chunk size that gives target_size > FRAGMENT_OVERLAP (500)
        // max_chunk_size * 4 = target_size, so we need max_chunk_size > 125
        let chunker = SemanticChunker::new(200); // target_size = 800 chars

        // Create content larger than target_size to trigger splitting
        // 31 chars * 100 = 3100 chars, which is > 800
        let chunk = SemanticChunk::new(
            ChunkType::Function,
            "big_fn",
            "line1\nline2\nline3\nline4\nline5\n".repeat(100),
            1,
            500,
            "rust",
            "test.rs",
        )
        .with_docstring("A big function")
        .with_signature("fn big_fn()");

        let fragments = chunker.split_chunk_with_overlap(&chunk);

        assert!(fragments.len() > 1, "Expected multiple fragments, got {}", fragments.len());
        // First fragment should have docstring and signature
        assert!(fragments[0].docstring.is_some());
        assert!(fragments[0].signature.is_some());
        // Subsequent fragments should not
        if fragments.len() > 1 {
            assert!(fragments[1].docstring.is_none());
        }
        // All should be marked as fragments
        for frag in &fragments {
            assert!(frag.is_fragment);
            assert!(frag.fragment_index.is_some());
            assert!(frag.total_fragments.is_some());
        }
    }
}

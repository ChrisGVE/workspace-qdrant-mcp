//! Core types for semantic code chunking.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::DaemonError;

/// Type of semantic chunk extracted from source code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkType {
    /// File preamble (imports, module docstrings, use statements)
    Preamble,
    /// Function definition
    Function,
    /// Async function definition
    AsyncFunction,
    /// Class definition
    Class,
    /// Method within a class/struct/impl
    Method,
    /// Struct definition (Rust, Go, C)
    Struct,
    /// Trait definition (Rust)
    Trait,
    /// Interface definition (Go, Java, TypeScript)
    Interface,
    /// Enum definition
    Enum,
    /// Impl block (Rust)
    Impl,
    /// Module/namespace definition
    Module,
    /// Constant definition
    Constant,
    /// Type alias definition
    TypeAlias,
    /// Macro definition (Rust)
    Macro,
    /// Plain text chunk (fallback)
    Text,
}

impl ChunkType {
    /// Get display name for the chunk type.
    pub fn display_name(&self) -> &'static str {
        match self {
            ChunkType::Preamble => "preamble",
            ChunkType::Function => "function",
            ChunkType::AsyncFunction => "async_function",
            ChunkType::Class => "class",
            ChunkType::Method => "method",
            ChunkType::Struct => "struct",
            ChunkType::Trait => "trait",
            ChunkType::Interface => "interface",
            ChunkType::Enum => "enum",
            ChunkType::Impl => "impl",
            ChunkType::Module => "module",
            ChunkType::Constant => "constant",
            ChunkType::TypeAlias => "type_alias",
            ChunkType::Macro => "macro",
            ChunkType::Text => "text",
        }
    }
}

/// A semantic chunk extracted from source code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticChunk {
    /// Type of the chunk
    pub chunk_type: ChunkType,

    /// Name of the symbol (function name, class name, etc.)
    pub symbol_name: String,

    /// Specific kind of symbol (e.g., "async_function", "static_method")
    pub symbol_kind: String,

    /// Parent symbol for nested items (e.g., class name for methods)
    pub parent_symbol: Option<String>,

    /// Starting line number (1-indexed)
    pub start_line: usize,

    /// Ending line number (1-indexed)
    pub end_line: usize,

    /// The actual content of the chunk
    pub content: String,

    /// Associated documentation string
    pub docstring: Option<String>,

    /// Function/method signature
    pub signature: Option<String>,

    /// List of function/method calls within this chunk
    pub calls: Vec<String>,

    /// Whether this is a fragment of a larger unit (due to size limits)
    pub is_fragment: bool,

    /// Fragment index if this is part of a larger unit
    pub fragment_index: Option<usize>,

    /// Total fragments if this is part of a larger unit
    pub total_fragments: Option<usize>,

    /// Language of the source code
    pub language: String,

    /// Original file path
    pub file_path: String,
}

impl SemanticChunk {
    /// Create a new semantic chunk with required fields.
    pub fn new(
        chunk_type: ChunkType,
        symbol_name: impl Into<String>,
        content: impl Into<String>,
        start_line: usize,
        end_line: usize,
        language: impl Into<String>,
        file_path: impl Into<String>,
    ) -> Self {
        Self {
            chunk_type,
            symbol_name: symbol_name.into(),
            symbol_kind: chunk_type.display_name().to_string(),
            parent_symbol: None,
            start_line,
            end_line,
            content: content.into(),
            docstring: None,
            signature: None,
            calls: Vec::new(),
            is_fragment: false,
            fragment_index: None,
            total_fragments: None,
            language: language.into(),
            file_path: file_path.into(),
        }
    }

    /// Create a preamble chunk.
    pub fn preamble(
        content: impl Into<String>,
        end_line: usize,
        language: impl Into<String>,
        file_path: impl Into<String>,
    ) -> Self {
        Self::new(
            ChunkType::Preamble,
            "_preamble",
            content,
            1,
            end_line,
            language,
            file_path,
        )
    }

    /// Create a text chunk (fallback).
    pub fn text(
        content: impl Into<String>,
        start_line: usize,
        end_line: usize,
        language: impl Into<String>,
        file_path: impl Into<String>,
    ) -> Self {
        Self::new(
            ChunkType::Text,
            "_text",
            content,
            start_line,
            end_line,
            language,
            file_path,
        )
    }

    /// Set the parent symbol.
    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent_symbol = Some(parent.into());
        self
    }

    /// Set the docstring.
    pub fn with_docstring(mut self, docstring: impl Into<String>) -> Self {
        self.docstring = Some(docstring.into());
        self
    }

    /// Set the signature.
    pub fn with_signature(mut self, signature: impl Into<String>) -> Self {
        self.signature = Some(signature.into());
        self
    }

    /// Set the symbol kind (overriding default).
    pub fn with_symbol_kind(mut self, kind: impl Into<String>) -> Self {
        self.symbol_kind = kind.into();
        self
    }

    /// Add function calls.
    pub fn with_calls(mut self, calls: Vec<String>) -> Self {
        self.calls = calls;
        self
    }

    /// Mark as a fragment.
    pub fn as_fragment(mut self, index: usize, total: usize) -> Self {
        self.is_fragment = true;
        self.fragment_index = Some(index);
        self.total_fragments = Some(total);
        self
    }

    /// Estimate the token count for this chunk.
    /// Uses a simple approximation: ~4 characters per token.
    pub fn estimated_tokens(&self) -> usize {
        self.content.len() / 4
    }
}

/// Trait for language-specific chunk extraction.
pub trait ChunkExtractor: Send + Sync {
    /// Extract semantic chunks from source code.
    fn extract_chunks(
        &self,
        source: &str,
        file_path: &Path,
    ) -> Result<Vec<SemanticChunk>, DaemonError>;

    /// Get the language name for this extractor.
    fn language(&self) -> &'static str;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_type_display() {
        assert_eq!(ChunkType::Function.display_name(), "function");
        assert_eq!(ChunkType::AsyncFunction.display_name(), "async_function");
        assert_eq!(ChunkType::Class.display_name(), "class");
        assert_eq!(ChunkType::Method.display_name(), "method");
        assert_eq!(ChunkType::Struct.display_name(), "struct");
        assert_eq!(ChunkType::Trait.display_name(), "trait");
    }

    #[test]
    fn test_semantic_chunk_creation() {
        let chunk = SemanticChunk::new(
            ChunkType::Function,
            "my_function",
            "fn my_function() {}",
            10,
            15,
            "rust",
            "src/lib.rs",
        );

        assert_eq!(chunk.chunk_type, ChunkType::Function);
        assert_eq!(chunk.symbol_name, "my_function");
        assert_eq!(chunk.start_line, 10);
        assert_eq!(chunk.end_line, 15);
        assert_eq!(chunk.language, "rust");
        assert!(!chunk.is_fragment);
    }

    #[test]
    fn test_chunk_builders() {
        let chunk = SemanticChunk::new(
            ChunkType::Method,
            "process",
            "def process(self): pass",
            5,
            6,
            "python",
            "app.py",
        )
        .with_parent("MyClass")
        .with_docstring("Process the data.")
        .with_signature("def process(self) -> None")
        .with_calls(vec!["helper".to_string(), "validate".to_string()]);

        assert_eq!(chunk.parent_symbol, Some("MyClass".to_string()));
        assert_eq!(chunk.docstring, Some("Process the data.".to_string()));
        assert_eq!(chunk.signature, Some("def process(self) -> None".to_string()));
        assert_eq!(chunk.calls.len(), 2);
    }

    #[test]
    fn test_fragment_marking() {
        let chunk = SemanticChunk::new(
            ChunkType::Function,
            "large_fn",
            "...",
            1,
            500,
            "rust",
            "big.rs",
        )
        .as_fragment(0, 3);

        assert!(chunk.is_fragment);
        assert_eq!(chunk.fragment_index, Some(0));
        assert_eq!(chunk.total_fragments, Some(3));
    }

    #[test]
    fn test_estimated_tokens() {
        let chunk = SemanticChunk::new(
            ChunkType::Text,
            "_text",
            "a".repeat(400), // 400 chars = ~100 tokens
            1,
            10,
            "text",
            "file.txt",
        );

        assert_eq!(chunk.estimated_tokens(), 100);
    }
}

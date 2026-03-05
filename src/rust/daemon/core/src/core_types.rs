//! Core domain types for document processing.
//!
//! Defines the fundamental types used across the processing pipeline:
//! document types, chunking configuration, processing results, and errors.

use std::collections::HashMap;
use thiserror::Error;

/// Core processing errors
#[derive(Error, Debug)]
pub enum ProcessingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parsing error: {0}")]
    Parse(String),

    #[error("Processing error: {0}")]
    Processing(String),

    #[error("Storage error: {0}")]
    Storage(String),
}

/// Document processing result
#[derive(Debug, Clone)]
pub struct DocumentResult {
    pub document_id: String,
    pub collection: String,
    pub chunks_created: Option<usize>,
    pub processing_time_ms: u64,
}

/// Processing statistics for monitoring
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    pub total_documents_processed: u64,
    pub total_chunks_created: u64,
    pub average_processing_time_ms: u64,
    pub chunking_config: ChunkingConfig,
}

/// Document type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum DocumentType {
    Pdf,
    Epub,
    Docx,
    Pptx,
    Odt,
    Odp,
    Ods,
    Rtf,
    Doc,
    Ppt,
    Xlsx,
    Xls,
    Numbers,
    Csv,
    Jupyter,
    Pages,
    Key,
    Text,
    Markdown,
    Code(String), // Language name
    Unknown,
}

impl DocumentType {
    /// Return a clean lowercase string for storage in Qdrant payloads.
    /// Unlike Debug format which produces `Code("rust")`, this returns `"code"`.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Pdf => "pdf",
            Self::Epub => "epub",
            Self::Docx => "docx",
            Self::Pptx => "pptx",
            Self::Odt => "odt",
            Self::Odp => "odp",
            Self::Ods => "ods",
            Self::Rtf => "rtf",
            Self::Doc => "doc",
            Self::Ppt => "ppt",
            Self::Xlsx => "xlsx",
            Self::Xls => "xls",
            Self::Numbers => "numbers",
            Self::Csv => "csv",
            Self::Jupyter => "jupyter",
            Self::Pages => "pages",
            Self::Key => "key",
            Self::Text => "text",
            Self::Markdown => "markdown",
            Self::Code(_) => "code",
            Self::Unknown => "unknown",
        }
    }

    /// Return true if this is a code document.
    pub fn is_code(&self) -> bool {
        matches!(self, Self::Code(_))
    }

    /// Return the language name for Code documents, None for other types.
    pub fn language(&self) -> Option<&str> {
        match self {
            Self::Code(lang) => Some(lang.as_str()),
            _ => None,
        }
    }
}

/// Text chunk with metadata
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub content: String,
    pub chunk_index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub metadata: HashMap<String, String>,
}

/// Document content representation
#[derive(Debug, Clone)]
pub struct DocumentContent {
    pub raw_text: String,
    pub metadata: HashMap<String, String>,
    pub document_type: DocumentType,
    pub chunks: Vec<TextChunk>,
}

/// Chunking configuration
///
/// Supports both character-based (legacy) and token-based chunking.
/// Token-based fields take precedence when set. Character-based fields
/// are used by the existing document_processor chunking paths.
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Character-based chunk size (used by existing document_processor paths)
    pub chunk_size: usize,
    /// Character-based overlap size
    pub overlap_size: usize,
    /// Whether to preserve paragraph boundaries in character-based chunking
    pub preserve_paragraphs: bool,
    /// Token-based target chunk size (used by token-aware chunking for library documents)
    /// Default: 105 tokens (fits within all-MiniLM-L6-v2's 128-token limit with header)
    pub chunk_target_tokens: usize,
    /// Token-based overlap between chunks
    /// Default: 12 tokens (~10-15% of target)
    pub chunk_overlap_tokens: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 384,
            overlap_size: 58,
            preserve_paragraphs: true,
            chunk_target_tokens: 105,
            chunk_overlap_tokens: 12,
        }
    }
}

impl ChunkingConfig {
    /// Create a token-only config for library document ingestion.
    ///
    /// Character-based fields are set to defaults but won't be used
    /// when the token-aware chunking path is active.
    pub fn for_library_document(target_tokens: usize, overlap_tokens: usize) -> Self {
        Self {
            chunk_target_tokens: target_tokens,
            chunk_overlap_tokens: overlap_tokens,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_type_as_str() {
        assert_eq!(DocumentType::Pdf.as_str(), "pdf");
        assert_eq!(DocumentType::Epub.as_str(), "epub");
        assert_eq!(DocumentType::Docx.as_str(), "docx");
        assert_eq!(DocumentType::Pptx.as_str(), "pptx");
        assert_eq!(DocumentType::Odt.as_str(), "odt");
        assert_eq!(DocumentType::Odp.as_str(), "odp");
        assert_eq!(DocumentType::Ods.as_str(), "ods");
        assert_eq!(DocumentType::Rtf.as_str(), "rtf");
        assert_eq!(DocumentType::Doc.as_str(), "doc");
        assert_eq!(DocumentType::Ppt.as_str(), "ppt");
        assert_eq!(DocumentType::Xlsx.as_str(), "xlsx");
        assert_eq!(DocumentType::Xls.as_str(), "xls");
        assert_eq!(DocumentType::Numbers.as_str(), "numbers");
        assert_eq!(DocumentType::Csv.as_str(), "csv");
        assert_eq!(DocumentType::Jupyter.as_str(), "jupyter");
        assert_eq!(DocumentType::Pages.as_str(), "pages");
        assert_eq!(DocumentType::Key.as_str(), "key");
        assert_eq!(DocumentType::Text.as_str(), "text");
        assert_eq!(DocumentType::Markdown.as_str(), "markdown");
        assert_eq!(DocumentType::Code("rust".to_string()).as_str(), "code");
        assert_eq!(DocumentType::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_document_type_language() {
        assert_eq!(
            DocumentType::Code("rust".to_string()).language(),
            Some("rust")
        );
        assert_eq!(
            DocumentType::Code("python".to_string()).language(),
            Some("python")
        );
        assert_eq!(DocumentType::Pdf.language(), None);
        assert_eq!(DocumentType::Text.language(), None);
        assert_eq!(DocumentType::Markdown.language(), None);
        assert_eq!(DocumentType::Unknown.language(), None);
    }

    #[test]
    fn test_chunking_config_default_has_token_fields() {
        let config = ChunkingConfig::default();
        assert_eq!(config.chunk_target_tokens, 105);
        assert_eq!(config.chunk_overlap_tokens, 12);
        assert_eq!(config.chunk_size, 384);
        assert_eq!(config.overlap_size, 58);
        assert!(config.preserve_paragraphs);
    }

    #[test]
    fn test_chunking_config_token_overlap_ratio() {
        let config = ChunkingConfig::default();
        let ratio = config.chunk_overlap_tokens as f64 / config.chunk_target_tokens as f64;
        assert!(
            ratio >= 0.10,
            "Overlap should be >= 10% of target: {:.1}%",
            ratio * 100.0
        );
        assert!(
            ratio <= 0.15,
            "Overlap should be <= 15% of target: {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_chunking_config_for_library_document() {
        let config = ChunkingConfig::for_library_document(100, 15);
        assert_eq!(config.chunk_target_tokens, 100);
        assert_eq!(config.chunk_overlap_tokens, 15);
        assert_eq!(config.chunk_size, 384);
        assert_eq!(config.overlap_size, 58);
    }

    #[test]
    fn test_chunking_config_target_fits_model_limit() {
        let config = ChunkingConfig::default();
        assert!(
            config.chunk_target_tokens <= 120,
            "Target tokens ({}) must leave room for header within 128-token model limit",
            config.chunk_target_tokens
        );
    }
}

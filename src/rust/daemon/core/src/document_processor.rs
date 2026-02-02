//! Document processing module for extracting text from various file formats (Task 462).
//!
//! This module provides the DocumentProcessor for extracting text content from:
//! - PDF documents (via pdf-extract)
//! - EPUB documents (via epub crate)
//! - DOCX documents (via zip + XML extraction)
//! - Code files (UTF-8 with language detection)
//! - Text, Markdown, HTML, XML, JSON, YAML, TOML files
//!
//! The processor handles encoding detection, chunking, and metadata generation.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use chardet::detect;
use encoding_rs::Encoding;
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::tree_sitter::{extract_chunks, is_language_supported, SemanticChunk};
use crate::{ChunkingConfig, DocumentContent, DocumentResult, DocumentType, TextChunk};

/// Document processing errors
#[derive(Error, Debug)]
pub enum DocumentProcessorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("PDF extraction error: {0}")]
    PdfExtraction(String),

    #[error("EPUB extraction error: {0}")]
    EpubExtraction(String),

    #[error("DOCX extraction error: {0}")]
    DocxExtraction(String),

    #[error("Encoding detection failed: {0}")]
    EncodingError(String),

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Processing task failed: {0}")]
    TaskError(String),
}

/// Result type for document processing operations
pub type DocumentProcessorResult<T> = Result<T, DocumentProcessorError>;

/// Document processor for extracting text from various file formats
#[derive(Debug)]
pub struct DocumentProcessor {
    chunking_config: ChunkingConfig,
    healthy: Arc<AtomicBool>,
}

impl Clone for DocumentProcessor {
    fn clone(&self) -> Self {
        Self {
            chunking_config: self.chunking_config.clone(),
            healthy: Arc::clone(&self.healthy),
        }
    }
}

impl Default for DocumentProcessor {
    fn default() -> Self {
        Self {
            chunking_config: ChunkingConfig::default(),
            healthy: Arc::new(AtomicBool::new(true)),
        }
    }
}

impl DocumentProcessor {
    /// Create a new DocumentProcessor with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new DocumentProcessor with custom chunking configuration
    pub fn with_config(chunking_config: ChunkingConfig) -> Self {
        Self {
            chunking_config,
            healthy: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Alias for with_config for compatibility with existing tests
    pub fn with_chunking_config(chunking_config: ChunkingConfig) -> Self {
        Self::with_config(chunking_config)
    }

    /// Check if the processor is healthy
    pub async fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::SeqCst)
    }

    /// Process a file and extract its content with chunks (async interface)
    /// Returns DocumentResult with document_id, collection, chunks_created, and processing_time_ms
    pub async fn process_file(
        &self,
        file_path: &Path,
        collection: &str,
    ) -> DocumentProcessorResult<DocumentResult> {
        let start_time = Instant::now();
        let path = file_path.to_path_buf();
        let collection_str = collection.to_string();
        let collection_result = collection.to_string();
        let chunking_config = self.chunking_config.clone();

        // Run blocking file processing in a separate thread
        let content = tokio::task::spawn_blocking(move || {
            process_file_sync(&path, &collection_str, &chunking_config)
        })
        .await
        .map_err(|e| DocumentProcessorError::TaskError(e.to_string()))??;

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(DocumentResult {
            document_id: uuid::Uuid::new_v4().to_string(),
            collection: collection_result,
            chunks_created: Some(content.chunks.len()),
            processing_time_ms,
        })
    }

    /// Process a file and return the full DocumentContent (for internal use)
    pub async fn process_file_content(
        &self,
        file_path: &Path,
        collection: &str,
    ) -> DocumentProcessorResult<DocumentContent> {
        let path = file_path.to_path_buf();
        let collection = collection.to_string();
        let chunking_config = self.chunking_config.clone();

        // Run blocking file processing in a separate thread
        let result = tokio::task::spawn_blocking(move || {
            process_file_sync(&path, &collection, &chunking_config)
        })
        .await
        .map_err(|e| DocumentProcessorError::TaskError(e.to_string()))?;

        result
    }

    /// Get the chunking configuration
    pub fn chunking_config(&self) -> &ChunkingConfig {
        &self.chunking_config
    }

    /// Detect document type from file path
    pub fn detect_document_type(&self, file_path: &Path) -> DocumentProcessorResult<DocumentType> {
        if !file_path.exists() {
            return Err(DocumentProcessorError::FileNotFound(
                file_path.display().to_string(),
            ));
        }
        Ok(detect_document_type(file_path))
    }
}

/// Synchronous file processing implementation (standalone function for spawn_blocking)
fn process_file_sync(
    file_path: &Path,
    collection: &str,
    chunking_config: &ChunkingConfig,
) -> DocumentProcessorResult<DocumentContent> {
    if !file_path.exists() {
        return Err(DocumentProcessorError::FileNotFound(
            file_path.display().to_string(),
        ));
    }

    let document_type = detect_document_type(file_path);
    debug!(
        "Processing file {:?} as {:?} for collection {}",
        file_path.file_name(),
        document_type,
        collection
    );

    let (raw_text, mut metadata) = match &document_type {
        DocumentType::Pdf => extract_pdf(file_path)?,
        DocumentType::Epub => extract_epub(file_path)?,
        DocumentType::Docx => extract_docx(file_path)?,
        DocumentType::Code(lang) => extract_code(file_path, lang)?,
        DocumentType::Markdown => extract_text_with_encoding(file_path)?,
        DocumentType::Text => extract_text_with_encoding(file_path)?,
        DocumentType::Unknown => extract_text_with_encoding(file_path)?,
    };

    // Add collection to metadata
    metadata.insert("collection".to_string(), collection.to_string());

    // Generate chunks - use semantic chunking for supported code files
    let chunks = match &document_type {
        DocumentType::Code(lang) if is_language_supported(lang) => {
            // Use tree-sitter semantic chunking for supported languages
            match extract_chunks(&raw_text, file_path, chunking_config.chunk_size) {
                Ok(semantic_chunks) => {
                    info!(
                        "Extracted {} semantic chunks from {:?} ({})",
                        semantic_chunks.len(),
                        file_path.file_name(),
                        lang
                    );
                    convert_semantic_chunks_to_text_chunks(semantic_chunks, &metadata)
                }
                Err(e) => {
                    warn!(
                        "Semantic chunking failed for {:?}: {}, falling back to text chunking",
                        file_path, e
                    );
                    chunk_text(&raw_text, &metadata, chunking_config)
                }
            }
        }
        _ => {
            // Use text-based chunking for non-code files or unsupported languages
            chunk_text(&raw_text, &metadata, chunking_config)
        }
    };

    Ok(DocumentContent {
        raw_text,
        metadata,
        document_type,
        chunks,
    })
}

/// Convert SemanticChunks to TextChunks with semantic metadata preserved
fn convert_semantic_chunks_to_text_chunks(
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

/// Detect document type from file extension
fn detect_document_type(file_path: &Path) -> DocumentType {
    let extension = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    match extension.as_str() {
        // Document formats
        "pdf" => DocumentType::Pdf,
        "epub" => DocumentType::Epub,
        "docx" => DocumentType::Docx,

        // Markup and text
        "md" | "markdown" => DocumentType::Markdown,
        "txt" | "text" => DocumentType::Text,
        "html" | "htm" => DocumentType::Text,
        "xml" => DocumentType::Text,
        "json" => DocumentType::Code("json".to_string()),
        "yaml" | "yml" => DocumentType::Code("yaml".to_string()),
        "toml" => DocumentType::Code("toml".to_string()),

        // Programming languages
        "rs" => DocumentType::Code("rust".to_string()),
        "py" | "pyw" => DocumentType::Code("python".to_string()),
        "js" | "mjs" | "cjs" => DocumentType::Code("javascript".to_string()),
        "ts" | "mts" | "cts" => DocumentType::Code("typescript".to_string()),
        "jsx" | "tsx" => DocumentType::Code("typescript".to_string()),
        "java" => DocumentType::Code("java".to_string()),
        "kt" | "kts" => DocumentType::Code("kotlin".to_string()),
        "swift" => DocumentType::Code("swift".to_string()),
        "go" => DocumentType::Code("go".to_string()),
        "c" | "h" => DocumentType::Code("c".to_string()),
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" => DocumentType::Code("cpp".to_string()),
        "cs" => DocumentType::Code("csharp".to_string()),
        "rb" => DocumentType::Code("ruby".to_string()),
        "php" => DocumentType::Code("php".to_string()),
        "sh" | "bash" | "zsh" => DocumentType::Code("shell".to_string()),
        "sql" => DocumentType::Code("sql".to_string()),
        "r" | "R" => DocumentType::Code("r".to_string()),
        "lua" => DocumentType::Code("lua".to_string()),
        "pl" | "pm" => DocumentType::Code("perl".to_string()),
        "scala" => DocumentType::Code("scala".to_string()),
        "hs" | "lhs" => DocumentType::Code("haskell".to_string()),
        "ex" | "exs" => DocumentType::Code("elixir".to_string()),
        "erl" | "hrl" => DocumentType::Code("erlang".to_string()),
        "clj" | "cljs" | "cljc" => DocumentType::Code("clojure".to_string()),
        "ml" | "mli" => DocumentType::Code("ocaml".to_string()),
        "fs" | "fsx" | "fsi" => DocumentType::Code("fsharp".to_string()),
        "vue" => DocumentType::Code("vue".to_string()),
        "svelte" => DocumentType::Code("svelte".to_string()),
        "css" | "scss" | "sass" | "less" => DocumentType::Code("css".to_string()),
        "dockerfile" => DocumentType::Code("dockerfile".to_string()),
        "makefile" => DocumentType::Code("makefile".to_string()),

        // Config files
        "ini" | "cfg" | "conf" => DocumentType::Code("ini".to_string()),
        "env" => DocumentType::Code("env".to_string()),

        _ => DocumentType::Unknown,
    }
}

/// Extract text from PDF using pdf-extract
fn extract_pdf(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "pdf".to_string());

    match pdf_extract::extract_text(file_path) {
        Ok(text) => {
            let cleaned_text = clean_extracted_text(&text);
            metadata.insert("page_count".to_string(), "unknown".to_string());
            Ok((cleaned_text, metadata))
        }
        Err(e) => Err(DocumentProcessorError::PdfExtraction(e.to_string())),
    }
}

/// Extract text from EPUB using epub crate
fn extract_epub(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "epub".to_string());

    let doc = epub::doc::EpubDoc::new(file_path)
        .map_err(|e| DocumentProcessorError::EpubExtraction(e.to_string()))?;

    // Extract metadata
    if let Some(title) = doc.mdata("title") {
        metadata.insert("title".to_string(), title.value.clone());
    }
    if let Some(author) = doc.mdata("creator") {
        metadata.insert("author".to_string(), author.value.clone());
    }

    // Extract text from all chapters
    let mut all_text = String::new();
    let mut chapter_count = 0;

    let mut doc = doc;
    loop {
        if let Some((content, _mime)) = doc.get_current_str() {
            let text = html2text::from_read(content.as_bytes(), 80);
            all_text.push_str(&text);
            all_text.push_str("\n\n");
            chapter_count += 1;
        }
        if !doc.go_next() {
            break;
        }
    }

    metadata.insert("chapter_count".to_string(), chapter_count.to_string());
    Ok((clean_extracted_text(&all_text), metadata))
}

/// Extract text from DOCX (ZIP file with XML content)
fn extract_docx(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "docx".to_string());

    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| DocumentProcessorError::DocxExtraction(e.to_string()))?;

    let mut text = String::new();

    if let Ok(mut document_file) = archive.by_name("word/document.xml") {
        let mut content = String::new();
        document_file.read_to_string(&mut content)?;
        text = extract_text_from_docx_xml(&content);
    }

    if text.is_empty() {
        return Err(DocumentProcessorError::DocxExtraction(
            "No text content found in DOCX".to_string(),
        ));
    }

    Ok((clean_extracted_text(&text), metadata))
}

/// Extract text from DOCX XML content
fn extract_text_from_docx_xml(xml_content: &str) -> String {
    let mut text = String::new();
    let mut in_text_tag = false;
    let mut current_text = String::new();

    for line in xml_content.lines() {
        for part in line.split('<') {
            if part.starts_with("w:t") {
                in_text_tag = true;
                if let Some(content_start) = part.find('>') {
                    current_text.push_str(&part[content_start + 1..]);
                }
            } else if part.starts_with("/w:t") {
                in_text_tag = false;
                if !current_text.is_empty() {
                    text.push_str(&current_text);
                    current_text.clear();
                }
            } else if part.starts_with("w:p") && !part.starts_with("w:pPr") {
                if !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
            } else if in_text_tag {
                if let Some(end_pos) = part.find('>') {
                    current_text.push_str(&part[end_pos + 1..]);
                } else {
                    current_text.push_str(part);
                }
            }
        }
    }

    text
}

/// Extract code file with language metadata
fn extract_code(
    file_path: &Path,
    language: &str,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "code".to_string());
    metadata.insert("language".to_string(), language.to_string());

    let (text, mut text_metadata) = extract_text_with_encoding(file_path)?;

    // Merge metadata
    for (k, v) in text_metadata.drain() {
        metadata.entry(k).or_insert(v);
    }

    let line_count = text.lines().count();
    metadata.insert("line_count".to_string(), line_count.to_string());

    Ok((text, metadata))
}

/// Extract text file with encoding detection
fn extract_text_with_encoding(
    file_path: &Path,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "text".to_string());

    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Try UTF-8 first
    if let Ok(text) = std::str::from_utf8(&buffer) {
        metadata.insert("encoding".to_string(), "utf-8".to_string());
        return Ok((text.to_string(), metadata));
    }

    // Detect encoding using chardet
    let detection = detect(&buffer);
    let encoding_name = detection.0.to_uppercase();
    metadata.insert("encoding".to_string(), encoding_name.clone());
    metadata.insert("encoding_confidence".to_string(), detection.1.to_string());

    // Try to decode using detected encoding
    if let Some(encoding) = Encoding::for_label(encoding_name.as_bytes()) {
        let (decoded, _, had_errors) = encoding.decode(&buffer);
        if !had_errors {
            return Ok((decoded.to_string(), metadata));
        }
    }

    // Fallback: decode as UTF-8 with lossy conversion
    warn!(
        "Encoding detection failed for {:?}, using lossy UTF-8",
        file_path
    );
    metadata.insert("encoding_fallback".to_string(), "true".to_string());
    Ok((String::from_utf8_lossy(&buffer).to_string(), metadata))
}

/// Clean up extracted text (normalize whitespace, remove control chars)
fn clean_extracted_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_whitespace = false;

    for ch in text.chars() {
        if ch.is_control() && ch != '\n' && ch != '\t' {
            continue;
        }

        if ch.is_whitespace() {
            if !prev_was_whitespace || ch == '\n' {
                result.push(if ch == '\n' { '\n' } else { ' ' });
            }
            prev_was_whitespace = true;
        } else {
            result.push(ch);
            prev_was_whitespace = false;
        }
    }

    result.trim().to_string()
}

/// Chunk text into smaller pieces with overlap
fn chunk_text(
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
fn chunk_by_paragraphs(
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
fn chunk_by_characters(
    text: &str,
    base_metadata: &HashMap<String, String>,
    chunks: &mut Vec<TextChunk>,
    config: &ChunkingConfig,
) {
    let total_chars = text.len();
    let mut start = 0;
    let mut chunk_index = 0;

    while start < total_chars {
        let end = (start + config.chunk_size).min(total_chars);

        let actual_end = if end < total_chars {
            text[start..end]
                .rfind(char::is_whitespace)
                .map(|pos| start + pos)
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

        start = actual_end.saturating_sub(config.overlap_size);
        if start <= chunks.last().map(|c| c.start_char).unwrap_or(0) {
            start = actual_end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_detect_document_type_pdf() {
        let path = Path::new("test.pdf");
        assert_eq!(detect_document_type(path), DocumentType::Pdf);
    }

    #[test]
    fn test_detect_document_type_code() {
        assert_eq!(
            detect_document_type(Path::new("main.rs")),
            DocumentType::Code("rust".to_string())
        );
        assert_eq!(
            detect_document_type(Path::new("app.py")),
            DocumentType::Code("python".to_string())
        );
        assert_eq!(
            detect_document_type(Path::new("index.js")),
            DocumentType::Code("javascript".to_string())
        );
    }

    #[test]
    fn test_detect_document_type_text() {
        assert_eq!(
            detect_document_type(Path::new("readme.md")),
            DocumentType::Markdown
        );
        assert_eq!(
            detect_document_type(Path::new("notes.txt")),
            DocumentType::Text
        );
    }

    #[tokio::test]
    async fn test_process_text_file() {
        let processor = DocumentProcessor::new();

        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Hello, World!\nThis is a test file.").unwrap();

        let result = processor.process_file(temp_file.path(), "test_collection").await;
        assert!(result.is_ok());

        let doc_result = result.unwrap();
        assert!(!doc_result.document_id.is_empty());
        assert_eq!(doc_result.collection, "test_collection");
        assert!(doc_result.chunks_created.unwrap_or(0) > 0);
        // processing_time_ms may be 0 for very fast operations (sub-millisecond)
        assert!(doc_result.processing_time_ms < 60000); // But should complete within a minute
    }

    #[tokio::test]
    async fn test_process_text_file_content() {
        let processor = DocumentProcessor::new();

        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "Hello, World!\nThis is a test file.").unwrap();

        let result = processor.process_file_content(temp_file.path(), "test_collection").await;
        assert!(result.is_ok());

        let content = result.unwrap();
        assert!(content.raw_text.contains("Hello, World!"));
        assert_eq!(content.metadata.get("collection"), Some(&"test_collection".to_string()));
    }

    #[tokio::test]
    async fn test_processor_is_healthy() {
        let processor = DocumentProcessor::new();
        assert!(processor.is_healthy().await);
    }

    #[test]
    fn test_chunk_text_simple() {
        let config = ChunkingConfig {
            chunk_size: 50,
            overlap_size: 10,
            preserve_paragraphs: false,
        };

        let text = "This is a test. It has multiple sentences. Each one should be processed.";
        let chunks = chunk_text(text, &HashMap::new(), &config);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_text_with_paragraphs() {
        let config = ChunkingConfig {
            chunk_size: 100,
            overlap_size: 10,
            preserve_paragraphs: true,
        };

        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
        let chunks = chunk_text(text, &HashMap::new(), &config);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.metadata.contains_key("chunk_index"));
        }
    }

    #[test]
    fn test_clean_extracted_text() {
        let text = "Hello    World\n\n\nTest\x00Control";
        let cleaned = clean_extracted_text(text);

        assert!(!cleaned.contains('\x00'));
    }

    #[tokio::test]
    async fn test_file_not_found() {
        let processor = DocumentProcessor::new();
        let result = processor.process_file(Path::new("/nonexistent/file.txt"), "test").await;

        assert!(result.is_err());
        matches!(result.unwrap_err(), DocumentProcessorError::FileNotFound(_));
    }
}

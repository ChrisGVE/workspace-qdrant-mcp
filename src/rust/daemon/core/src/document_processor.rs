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

    #[error("Spreadsheet extraction error: {0}")]
    SpreadsheetExtraction(String),

    #[error("CSV extraction error: {0}")]
    CsvExtraction(String),

    #[error("Jupyter extraction error: {0}")]
    JupyterExtraction(String),

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

        // Generate stable document_id from collection + file path
        let path_str = file_path.to_string_lossy();
        let document_id = crate::generate_document_id(&collection_result, &path_str);

        Ok(DocumentResult {
            document_id,
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
        DocumentType::Pptx => extract_pptx(file_path)?,
        DocumentType::Odt | DocumentType::Odp | DocumentType::Ods => {
            let fmt = match &document_type {
                DocumentType::Odt => "odt",
                DocumentType::Odp => "odp",
                DocumentType::Ods => "ods",
                _ => unreachable!(),
            };
            extract_opendocument(file_path, fmt)?
        }
        DocumentType::Rtf => extract_rtf(file_path)?,
        DocumentType::Xlsx | DocumentType::Xls => extract_spreadsheet(file_path)?,
        DocumentType::Csv => extract_csv(file_path)?,
        DocumentType::Jupyter => extract_jupyter(file_path)?,
        DocumentType::Ppt | DocumentType::Doc => {
            let fmt = match &document_type {
                DocumentType::Ppt => "PPT",
                DocumentType::Doc => "DOC",
                _ => unreachable!(),
            };
            warn!("Legacy binary format {} not supported, attempting text extraction: {:?}", fmt, file_path.file_name());
            extract_text_with_encoding(file_path)?
        }
        DocumentType::Pages | DocumentType::Key => {
            let fmt = match &document_type {
                DocumentType::Pages => "Pages",
                DocumentType::Key => "Keynote",
                _ => unreachable!(),
            };
            warn!("Apple iWork format {} has limited support, attempting extraction: {:?}", fmt, file_path.file_name());
            extract_iwork(file_path, fmt)?
        }
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
    // Check compound extensions first (before standard Path::extension())
    let filename = file_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let lower_filename = filename.to_lowercase();

    // Handle .d.ts / .d.mts / .d.cts (TypeScript declaration files)
    if lower_filename.ends_with(".d.ts")
        || lower_filename.ends_with(".d.mts")
        || lower_filename.ends_with(".d.cts")
    {
        return DocumentType::Code("typescript".to_string());
    }

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
        "pptx" => DocumentType::Pptx,
        "ppt" => DocumentType::Ppt,
        "odt" => DocumentType::Odt,
        "odp" => DocumentType::Odp,
        "ods" => DocumentType::Ods,
        "rtf" => DocumentType::Rtf,
        "doc" => DocumentType::Doc,
        "xlsx" => DocumentType::Xlsx,
        "xls" => DocumentType::Xls,
        "csv" | "tsv" => DocumentType::Csv,
        "ipynb" => DocumentType::Jupyter,
        "pages" => DocumentType::Pages,
        "key" => DocumentType::Key,

        // Markup and text
        "md" | "markdown" => DocumentType::Markdown,
        "txt" | "text" | "rst" | "org" | "adoc" => DocumentType::Text,

        // Web content (with language for metadata)
        "html" | "htm" | "xhtml" => DocumentType::Code("html".to_string()),
        "xml" | "xsl" | "xslt" | "svg" => DocumentType::Code("xml".to_string()),
        "css" | "scss" | "sass" | "less" => DocumentType::Code("css".to_string()),

        // Config/data formats
        "json" => DocumentType::Code("json".to_string()),
        "yaml" | "yml" => DocumentType::Code("yaml".to_string()),
        "toml" => DocumentType::Code("toml".to_string()),
        "ini" | "cfg" | "conf" => DocumentType::Code("ini".to_string()),
        "env" => DocumentType::Code("env".to_string()),

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
        "ps1" | "psm1" | "psd1" => DocumentType::Code("powershell".to_string()),
        "sql" => DocumentType::Code("sql".to_string()),
        "r" => DocumentType::Code("r".to_string()),
        "lua" => DocumentType::Code("lua".to_string()),
        "pl" | "pm" => DocumentType::Code("perl".to_string()),
        "scala" => DocumentType::Code("scala".to_string()),
        "hs" | "lhs" => DocumentType::Code("haskell".to_string()),
        "ex" | "exs" => DocumentType::Code("elixir".to_string()),
        "erl" | "hrl" => DocumentType::Code("erlang".to_string()),
        "clj" | "cljs" | "cljc" => DocumentType::Code("clojure".to_string()),
        "ml" | "mli" => DocumentType::Code("ocaml".to_string()),
        "fs" | "fsx" | "fsi" => DocumentType::Code("fsharp".to_string()),
        "d" => DocumentType::Code("d".to_string()),
        "zig" => DocumentType::Code("zig".to_string()),
        "dart" => DocumentType::Code("dart".to_string()),
        "nim" => DocumentType::Code("nim".to_string()),
        "v" => DocumentType::Code("v".to_string()),
        "proto" => DocumentType::Code("protobuf".to_string()),
        "graphql" | "gql" => DocumentType::Code("graphql".to_string()),
        "vue" => DocumentType::Code("vue".to_string()),
        "svelte" => DocumentType::Code("svelte".to_string()),
        "astro" => DocumentType::Code("astro".to_string()),
        "nix" => DocumentType::Code("nix".to_string()),
        "lean" => DocumentType::Code("lean".to_string()),
        "dockerfile" => DocumentType::Code("dockerfile".to_string()),
        "makefile" => DocumentType::Code("makefile".to_string()),
        "cmake" => DocumentType::Code("cmake".to_string()),

        _ => DocumentType::Unknown,
    }
}

/// Extract text from PDF using pdf-extract
///
/// Wrapped in `catch_unwind` because `pdf-extract` (via `type1-encoding-parser`)
/// panics on certain malformed Type 1 font encodings instead of returning an error.
fn extract_pdf(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "pdf".to_string());

    let path_buf = file_path.to_path_buf();
    let result = std::panic::catch_unwind(|| pdf_extract::extract_text(&path_buf));

    match result {
        Ok(Ok(text)) => {
            let cleaned_text = clean_extracted_text(&text);
            metadata.insert("page_count".to_string(), "unknown".to_string());
            Ok((cleaned_text, metadata))
        }
        Ok(Err(e)) => Err(DocumentProcessorError::PdfExtraction(e.to_string())),
        Err(_panic) => Err(DocumentProcessorError::PdfExtraction(
            format!("PDF parsing panicked (likely malformed font encoding): {}", file_path.display()),
        )),
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

/// Extract text from PowerPoint PPTX file (ZIP-based, slides in ppt/slides/slide*.xml)
fn extract_pptx(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "pptx".to_string());

    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| DocumentProcessorError::DocxExtraction(format!("PPTX: {}", e)))?;

    let mut all_text = String::new();
    let mut slide_count = 0u32;

    // Collect slide file names (they're numbered: slide1.xml, slide2.xml, etc.)
    let slide_names: Vec<String> = (0..archive.len())
        .filter_map(|i| {
            archive.by_index(i).ok().and_then(|f| {
                let name = f.name().to_string();
                if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
                    Some(name)
                } else {
                    None
                }
            })
        })
        .collect();

    for slide_name in &slide_names {
        if let Ok(mut slide_file) = archive.by_name(slide_name) {
            let mut content = String::new();
            slide_file.read_to_string(&mut content)?;
            let slide_text = extract_text_from_xml_tags(&content, "a:t");
            if !slide_text.is_empty() {
                slide_count += 1;
                if !all_text.is_empty() {
                    all_text.push('\n');
                }
                all_text.push_str(&slide_text);
            }
        }
    }

    metadata.insert("slide_count".to_string(), slide_count.to_string());

    if all_text.is_empty() {
        return Err(DocumentProcessorError::DocxExtraction(
            "No text content found in PPTX".to_string(),
        ));
    }

    Ok((clean_extracted_text(&all_text), metadata))
}

/// Extract text from OpenDocument formats (ODT/ODP/ODS) — all are ZIP-based with content.xml
fn extract_opendocument(file_path: &Path, format_name: &str) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), format_name.to_string());

    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| DocumentProcessorError::DocxExtraction(format!("{}: {}", format_name.to_uppercase(), e)))?;

    let mut text = String::new();

    if let Ok(mut content_file) = archive.by_name("content.xml") {
        let mut content = String::new();
        content_file.read_to_string(&mut content)?;
        text = extract_text_from_xml_tags(&content, "text:p");

        // Also extract from text:h (heading) and text:span tags
        if text.is_empty() {
            text = extract_text_from_xml_tags(&content, "text:span");
        }
    }

    if text.is_empty() {
        return Err(DocumentProcessorError::DocxExtraction(
            format!("No text content found in {} file", format_name.to_uppercase()),
        ));
    }

    Ok((clean_extracted_text(&text), metadata))
}

/// Extract text from RTF file by stripping RTF control codes
fn extract_rtf(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "rtf".to_string());

    let (raw_text, _) = extract_text_with_encoding(file_path)?;

    // Strip RTF control codes
    let mut result = String::with_capacity(raw_text.len());
    let mut skip_group_depth = 0i32; // Track depth of groups to skip (e.g., \fonttbl)
    let mut chars = raw_text.chars().peekable();

    // RTF groups that contain metadata, not text content
    let skip_groups = ["fonttbl", "colortbl", "stylesheet", "info", "pict", "object"];

    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                if skip_group_depth > 0 {
                    skip_group_depth += 1;
                }
            }
            '}' => {
                if skip_group_depth > 0 {
                    skip_group_depth -= 1;
                }
            }
            '\\' if skip_group_depth == 0 => {
                // RTF control word or symbol
                if let Some(&next) = chars.peek() {
                    if next == '\'' {
                        // Hex-encoded character: skip the \'XX
                        chars.next();
                        chars.next();
                        chars.next();
                    } else if next == '\\' || next == '{' || next == '}' {
                        result.push(chars.next().unwrap());
                    } else if next == '\n' || next == '\r' {
                        chars.next();
                        result.push('\n');
                    } else {
                        let mut control_word = String::new();
                        while let Some(&c) = chars.peek() {
                            if c.is_ascii_alphabetic() {
                                control_word.push(c);
                                chars.next();
                            } else {
                                if c == '-' || c.is_ascii_digit() {
                                    chars.next();
                                    while let Some(&d) = chars.peek() {
                                        if d.is_ascii_digit() { chars.next(); } else { break; }
                                    }
                                }
                                if chars.peek() == Some(&' ') {
                                    chars.next();
                                }
                                break;
                            }
                        }
                        if control_word == "par" || control_word == "line" {
                            result.push('\n');
                        } else if control_word == "tab" {
                            result.push('\t');
                        } else if skip_groups.contains(&control_word.as_str()) {
                            skip_group_depth = 1;
                        }
                    }
                }
            }
            _ if skip_group_depth == 0 => {
                result.push(ch);
            }
            _ => {}
        }
    }

    let text = clean_extracted_text(&result);
    if text.is_empty() {
        return Err(DocumentProcessorError::DocxExtraction(
            "No text content found in RTF file".to_string(),
        ));
    }

    Ok((text, metadata))
}

/// Extract text from Apple iWork formats (.pages, .key) — ZIP-based bundles
fn extract_iwork(file_path: &Path, format_name: &str) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), format_name.to_lowercase());

    let file = File::open(file_path)?;
    let archive_result = zip::ZipArchive::new(file);

    let mut archive = match archive_result {
        Ok(a) => a,
        Err(_) => {
            // Some iWork files are package bundles (directories), not ZIP
            return Err(DocumentProcessorError::DocxExtraction(
                format!("{} format: not a ZIP archive (may be a package bundle)", format_name),
            ));
        }
    };

    let mut text = String::new();

    // Try QuickLook preview text first (most reliable for iWork)
    if let Ok(mut preview) = archive.by_name("QuickLook/Preview.txt") {
        preview.read_to_string(&mut text)?;
    }

    // Try index.xml or Index/Document.iwa
    if text.is_empty() {
        // Try extracting from any XML files in the archive
        let xml_names: Vec<String> = (0..archive.len())
            .filter_map(|i| {
                archive.by_index(i).ok().and_then(|f| {
                    let name = f.name().to_string();
                    if name.ends_with(".xml") {
                        Some(name)
                    } else {
                        None
                    }
                })
            })
            .collect();

        for name in &xml_names {
            if let Ok(mut f) = archive.by_name(name) {
                let mut content = String::new();
                if f.read_to_string(&mut content).is_ok() {
                    let extracted = extract_text_from_xml_tags(&content, "sf:p");
                    if !extracted.is_empty() {
                        text.push_str(&extracted);
                        text.push('\n');
                    }
                }
            }
        }
    }

    if text.is_empty() {
        return Err(DocumentProcessorError::DocxExtraction(
            format!("No text content found in {} file. Consider exporting as PDF or DOCX.", format_name),
        ));
    }

    Ok((clean_extracted_text(&text), metadata))
}

/// Generic XML tag text extractor — extracts text content from all matching tags
fn extract_text_from_xml_tags(xml_content: &str, tag_name: &str) -> String {
    let mut text = String::new();
    let open_tag = format!("<{}", tag_name);
    let close_tag = format!("</{}", tag_name);

    let mut in_tag = false;
    let mut depth = 0i32;

    for part in xml_content.split('<') {
        if part.is_empty() {
            continue;
        }

        if part.starts_with(&tag_name[..]) || part.starts_with(&open_tag[1..]) {
            in_tag = true;
            depth += 1;
            if let Some(content_start) = part.find('>') {
                let content = &part[content_start + 1..];
                if !content.is_empty() {
                    text.push_str(content);
                }
            }
        } else if part.starts_with(&close_tag[1..]) {
            depth -= 1;
            if depth <= 0 {
                in_tag = false;
                depth = 0;
                text.push('\n');
            }
        } else if in_tag {
            // Nested tags inside — extract text after '>'
            if let Some(pos) = part.find('>') {
                let content = &part[pos + 1..];
                if !content.is_empty() {
                    text.push_str(content);
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

/// Extract text from Excel spreadsheet files (XLSX and XLS) using calamine
fn extract_spreadsheet(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    use calamine::{open_workbook_auto, Reader, Data};

    let mut metadata = HashMap::new();
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("xlsx")
        .to_lowercase();
    metadata.insert("source_format".to_string(), ext);

    let mut workbook = open_workbook_auto(file_path)
        .map_err(|e| DocumentProcessorError::SpreadsheetExtraction(e.to_string()))?;

    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
    metadata.insert("sheet_count".to_string(), sheet_names.len().to_string());

    let mut all_text = String::new();
    let mut total_rows = 0usize;

    for sheet_name in &sheet_names {
        if let Ok(range) = workbook.worksheet_range(sheet_name) {
            if !all_text.is_empty() {
                all_text.push('\n');
            }
            all_text.push_str(&format!("## {}\n", sheet_name));

            for row in range.rows() {
                total_rows += 1;
                let cells: Vec<String> = row
                    .iter()
                    .map(|cell| match cell {
                        Data::Empty => String::new(),
                        Data::String(s) => s.clone(),
                        Data::Int(i) => i.to_string(),
                        Data::Float(f) => f.to_string(),
                        Data::Bool(b) => b.to_string(),
                        Data::DateTime(dt) => dt.to_string(),
                        Data::Error(e) => format!("#ERR:{:?}", e),
                        Data::DateTimeIso(s) => s.clone(),
                        Data::DurationIso(s) => s.clone(),
                    })
                    .collect();

                // Skip entirely empty rows
                if cells.iter().all(|c| c.is_empty()) {
                    continue;
                }

                all_text.push_str(&cells.join("\t"));
                all_text.push('\n');
            }
        }
    }

    metadata.insert("row_count".to_string(), total_rows.to_string());

    if all_text.trim().is_empty() {
        return Err(DocumentProcessorError::SpreadsheetExtraction(
            "No data found in spreadsheet".to_string(),
        ));
    }

    Ok((clean_extracted_text(&all_text), metadata))
}

/// Extract text from CSV/TSV files
fn extract_csv(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("csv")
        .to_lowercase();
    metadata.insert("source_format".to_string(), ext.clone());

    let delimiter = if ext == "tsv" { b'\t' } else { b',' };

    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true) // tolerate rows with varying column counts
        .has_headers(true)
        .from_path(file_path)
        .map_err(|e| DocumentProcessorError::CsvExtraction(e.to_string()))?;

    let mut all_text = String::new();
    let mut row_count = 0usize;

    // Include headers
    let headers = reader.headers()
        .map_err(|e| DocumentProcessorError::CsvExtraction(e.to_string()))?
        .clone();
    let col_count = headers.len();
    metadata.insert("column_count".to_string(), col_count.to_string());

    if col_count > 0 {
        let header_line: Vec<&str> = headers.iter().collect();
        all_text.push_str(&header_line.join("\t"));
        all_text.push('\n');
    }

    for result in reader.records() {
        let record = result.map_err(|e| DocumentProcessorError::CsvExtraction(e.to_string()))?;
        row_count += 1;
        let fields: Vec<&str> = record.iter().collect();
        all_text.push_str(&fields.join("\t"));
        all_text.push('\n');
    }

    metadata.insert("row_count".to_string(), row_count.to_string());

    if all_text.trim().is_empty() {
        return Err(DocumentProcessorError::CsvExtraction(
            "No data found in CSV/TSV file".to_string(),
        ));
    }

    Ok((clean_extracted_text(&all_text), metadata))
}

/// Extract code and markdown from Jupyter notebook (.ipynb) files
fn extract_jupyter(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "jupyter".to_string());

    let mut file = File::open(file_path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;

    let notebook: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| DocumentProcessorError::JupyterExtraction(
            format!("Invalid notebook JSON: {}", e),
        ))?;

    // Detect kernel language from metadata
    let language = notebook
        .pointer("/metadata/kernelspec/language")
        .or_else(|| notebook.pointer("/metadata/language_info/name"))
        .and_then(|v| v.as_str())
        .unwrap_or("python")
        .to_string();
    metadata.insert("language".to_string(), language.clone());

    let cells = notebook
        .get("cells")
        .and_then(|v| v.as_array())
        .ok_or_else(|| DocumentProcessorError::JupyterExtraction(
            "No cells array found in notebook".to_string(),
        ))?;

    metadata.insert("cell_count".to_string(), cells.len().to_string());

    let mut all_text = String::new();
    let mut code_cells = 0usize;
    let mut markdown_cells = 0usize;

    for cell in cells {
        let cell_type = cell.get("cell_type").and_then(|v| v.as_str()).unwrap_or("unknown");
        let source = cell.get("source")
            .and_then(|v| v.as_array())
            .map(|lines| {
                lines.iter()
                    .filter_map(|l| l.as_str())
                    .collect::<Vec<&str>>()
                    .join("")
            })
            .or_else(|| cell.get("source").and_then(|v| v.as_str()).map(String::from))
            .unwrap_or_default();

        if source.trim().is_empty() {
            continue;
        }

        if !all_text.is_empty() {
            all_text.push('\n');
        }

        match cell_type {
            "code" => {
                code_cells += 1;
                all_text.push_str(&format!("```{}\n{}\n```", language, source.trim()));
            }
            "markdown" => {
                markdown_cells += 1;
                all_text.push_str(source.trim());
            }
            "raw" => {
                all_text.push_str(source.trim());
            }
            _ => {
                all_text.push_str(source.trim());
            }
        }
    }

    metadata.insert("code_cells".to_string(), code_cells.to_string());
    metadata.insert("markdown_cells".to_string(), markdown_cells.to_string());

    if all_text.trim().is_empty() {
        return Err(DocumentProcessorError::JupyterExtraction(
            "No content found in notebook".to_string(),
        ));
    }

    Ok((clean_extracted_text(&all_text), metadata))
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
/// Find the largest byte index <= `index` that is a valid UTF-8 char boundary.
/// This prevents panics when slicing strings at byte offsets calculated from
/// `len()` arithmetic, which can land inside multi-byte characters.
fn floor_char_boundary(s: &str, index: usize) -> usize {
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

    #[test]
    fn test_floor_char_boundary() {
        // ASCII-only string: all byte indices are char boundaries
        let ascii = "hello";
        assert_eq!(floor_char_boundary(ascii, 3), 3);
        assert_eq!(floor_char_boundary(ascii, 5), 5);
        assert_eq!(floor_char_boundary(ascii, 10), 5); // beyond end

        // Multi-byte: '─' is U+2500, encoded as 3 bytes (0xE2 0x94 0x80)
        let s = "ab─cd"; // bytes: a(0) b(1) ─(2,3,4) c(5) d(6)
        assert_eq!(floor_char_boundary(s, 2), 2); // start of ─
        assert_eq!(floor_char_boundary(s, 3), 2); // inside ─ → back to 2
        assert_eq!(floor_char_boundary(s, 4), 2); // inside ─ → back to 2
        assert_eq!(floor_char_boundary(s, 5), 5); // start of c
    }

    #[test]
    fn test_chunk_by_paragraphs_with_multibyte_overlap() {
        // Text with multi-byte box-drawing characters that caused the original crash
        let text = "First paragraph with content.\n\n\
                    ┌─────────────────────┐\n\
                    │   Box drawing test   │\n\
                    └─────────────────────┘\n\n\
                    Third paragraph after box.";

        let config = ChunkingConfig {
            chunk_size: 60,
            overlap_size: 20,
            ..ChunkingConfig::default()
        };

        let metadata = HashMap::new();
        let chunks = chunk_text(text, &metadata, &config);
        // Should not panic and should produce chunks
        assert!(!chunks.is_empty());
        // All chunks concatenated should cover the original text
        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
        }
    }

    #[test]
    fn test_chunk_by_characters_with_multibyte() {
        // Ensure character-based chunking handles multi-byte chars
        let text = "Hello ─── world ─── end";
        let config = ChunkingConfig {
            chunk_size: 10,
            overlap_size: 4,
            ..ChunkingConfig::default()
        };

        let metadata = HashMap::new();
        let mut chunks = Vec::new();
        chunk_by_characters(text, &metadata, &mut chunks, &config);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            // Each chunk should be valid UTF-8 (guaranteed by &str, but verify no panics)
            assert!(!chunk.content.is_empty());
        }
    }

    #[test]
    fn test_detect_document_type_new_formats() {
        assert_eq!(detect_document_type(Path::new("slides.pptx")), DocumentType::Pptx);
        assert_eq!(detect_document_type(Path::new("slides.ppt")), DocumentType::Ppt);
        assert_eq!(detect_document_type(Path::new("doc.odt")), DocumentType::Odt);
        assert_eq!(detect_document_type(Path::new("slides.odp")), DocumentType::Odp);
        assert_eq!(detect_document_type(Path::new("sheet.ods")), DocumentType::Ods);
        assert_eq!(detect_document_type(Path::new("doc.rtf")), DocumentType::Rtf);
        assert_eq!(detect_document_type(Path::new("legacy.doc")), DocumentType::Doc);
        assert_eq!(detect_document_type(Path::new("doc.pages")), DocumentType::Pages);
        assert_eq!(detect_document_type(Path::new("slides.key")), DocumentType::Key);
    }

    #[test]
    fn test_detect_document_type_case_insensitive() {
        assert_eq!(detect_document_type(Path::new("FILE.PPTX")), DocumentType::Pptx);
        assert_eq!(detect_document_type(Path::new("FILE.Rtf")), DocumentType::Rtf);
        assert_eq!(detect_document_type(Path::new("FILE.ODT")), DocumentType::Odt);
    }

    #[test]
    fn test_extract_text_from_xml_tags() {
        let xml = r#"<a:t>Hello</a:t><a:t>World</a:t>"#;
        let result = extract_text_from_xml_tags(xml, "a:t");
        assert!(result.contains("Hello"));
        assert!(result.contains("World"));
    }

    #[test]
    fn test_extract_text_from_xml_tags_nested() {
        let xml = r#"<text:p><text:span>Inner text</text:span></text:p>"#;
        let result = extract_text_from_xml_tags(xml, "text:p");
        assert!(result.contains("Inner text"));
    }

    #[test]
    fn test_extract_rtf_basic() {
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, r"{{\rtf1\ansi Hello World \par Second line}}").unwrap();
        let result = extract_rtf(tmp.path());
        assert!(result.is_ok());
        let (text, metadata) = result.unwrap();
        assert!(text.contains("Hello World"));
        assert!(text.contains("Second line"));
        assert_eq!(metadata.get("source_format").unwrap(), "rtf");
    }

    #[test]
    fn test_extract_rtf_with_formatting() {
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, r"{{\rtf1\ansi\deff0 {{\b Bold text}} normal text \par New para}}").unwrap();
        let result = extract_rtf(tmp.path());
        assert!(result.is_ok());
        let (text, _) = result.unwrap();
        assert!(text.contains("Bold text"));
        assert!(text.contains("normal text"));
    }

    // --- New format detection tests ---

    #[test]
    fn test_detect_document_type_spreadsheet_formats() {
        assert_eq!(detect_document_type(Path::new("data.xlsx")), DocumentType::Xlsx);
        assert_eq!(detect_document_type(Path::new("data.xls")), DocumentType::Xls);
        assert_eq!(detect_document_type(Path::new("DATA.XLSX")), DocumentType::Xlsx);
        assert_eq!(detect_document_type(Path::new("report.XLS")), DocumentType::Xls);
    }

    #[test]
    fn test_detect_document_type_csv() {
        assert_eq!(detect_document_type(Path::new("data.csv")), DocumentType::Csv);
        assert_eq!(detect_document_type(Path::new("data.tsv")), DocumentType::Csv);
        assert_eq!(detect_document_type(Path::new("DATA.CSV")), DocumentType::Csv);
    }

    #[test]
    fn test_detect_document_type_jupyter() {
        assert_eq!(detect_document_type(Path::new("notebook.ipynb")), DocumentType::Jupyter);
        assert_eq!(detect_document_type(Path::new("analysis.IPYNB")), DocumentType::Jupyter);
    }

    #[test]
    fn test_detect_document_type_web_content() {
        // HTML files now get language metadata instead of being treated as plain text
        assert_eq!(
            detect_document_type(Path::new("index.html")),
            DocumentType::Code("html".to_string())
        );
        assert_eq!(
            detect_document_type(Path::new("page.htm")),
            DocumentType::Code("html".to_string())
        );
        assert_eq!(
            detect_document_type(Path::new("doc.xhtml")),
            DocumentType::Code("html".to_string())
        );
        assert_eq!(
            detect_document_type(Path::new("data.xml")),
            DocumentType::Code("xml".to_string())
        );
        assert_eq!(
            detect_document_type(Path::new("icon.svg")),
            DocumentType::Code("xml".to_string())
        );
    }

    #[test]
    fn test_detect_document_type_new_languages() {
        // PowerShell
        assert_eq!(
            detect_document_type(Path::new("script.ps1")),
            DocumentType::Code("powershell".to_string())
        );
        assert_eq!(
            detect_document_type(Path::new("module.psm1")),
            DocumentType::Code("powershell".to_string())
        );
        // D language
        assert_eq!(
            detect_document_type(Path::new("main.d")),
            DocumentType::Code("d".to_string())
        );
        // Zig
        assert_eq!(
            detect_document_type(Path::new("build.zig")),
            DocumentType::Code("zig".to_string())
        );
        // Dart
        assert_eq!(
            detect_document_type(Path::new("app.dart")),
            DocumentType::Code("dart".to_string())
        );
        // Protocol Buffers
        assert_eq!(
            detect_document_type(Path::new("service.proto")),
            DocumentType::Code("protobuf".to_string())
        );
        // GraphQL
        assert_eq!(
            detect_document_type(Path::new("schema.graphql")),
            DocumentType::Code("graphql".to_string())
        );
        assert_eq!(
            detect_document_type(Path::new("query.gql")),
            DocumentType::Code("graphql".to_string())
        );
        // Astro
        assert_eq!(
            detect_document_type(Path::new("page.astro")),
            DocumentType::Code("astro".to_string())
        );
    }

    #[test]
    fn test_detect_document_type_compound_extensions() {
        // .d.ts should be TypeScript, not D language
        assert_eq!(
            detect_document_type(Path::new("types.d.ts")),
            DocumentType::Code("typescript".to_string())
        );
        assert_eq!(
            detect_document_type(Path::new("module.d.mts")),
            DocumentType::Code("typescript".to_string())
        );
        assert_eq!(
            detect_document_type(Path::new("common.d.cts")),
            DocumentType::Code("typescript".to_string())
        );
        // But plain .d should be D language
        assert_eq!(
            detect_document_type(Path::new("main.d")),
            DocumentType::Code("d".to_string())
        );
    }

    #[test]
    fn test_detect_document_type_text_extensions() {
        // rst, org, adoc are plain text
        assert_eq!(detect_document_type(Path::new("doc.rst")), DocumentType::Text);
        assert_eq!(detect_document_type(Path::new("notes.org")), DocumentType::Text);
        assert_eq!(detect_document_type(Path::new("guide.adoc")), DocumentType::Text);
    }

    // --- CSV extraction tests ---

    #[test]
    fn test_extract_csv_basic() {
        let mut tmp = NamedTempFile::with_suffix(".csv").unwrap();
        write!(tmp, "name,age,city\nAlice,30,New York\nBob,25,London\n").unwrap();
        let result = extract_csv(tmp.path());
        assert!(result.is_ok());
        let (text, metadata) = result.unwrap();
        assert!(text.contains("name"));
        assert!(text.contains("Alice"));
        assert!(text.contains("Bob"));
        assert_eq!(metadata.get("source_format").unwrap(), "csv");
        assert_eq!(metadata.get("row_count").unwrap(), "2");
        assert_eq!(metadata.get("column_count").unwrap(), "3");
    }

    #[test]
    fn test_extract_csv_tsv() {
        let mut tmp = NamedTempFile::with_suffix(".tsv").unwrap();
        write!(tmp, "col1\tcol2\nval1\tval2\n").unwrap();
        let result = extract_csv(tmp.path());
        assert!(result.is_ok());
        let (text, metadata) = result.unwrap();
        assert!(text.contains("col1"));
        assert!(text.contains("val1"));
        assert_eq!(metadata.get("source_format").unwrap(), "tsv");
    }

    #[test]
    fn test_extract_csv_empty() {
        let mut tmp = NamedTempFile::with_suffix(".csv").unwrap();
        write!(tmp, "").unwrap();
        let result = extract_csv(tmp.path());
        assert!(result.is_err());
    }

    // --- Jupyter extraction tests ---

    #[test]
    fn test_extract_jupyter_basic() {
        let notebook = serde_json::json!({
            "metadata": {
                "kernelspec": { "language": "python" }
            },
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Test Notebook\n", "This is a test."]
                },
                {
                    "cell_type": "code",
                    "source": ["import pandas as pd\n", "df = pd.read_csv('data.csv')"]
                }
            ],
            "nbformat": 4,
            "nbformat_minor": 2
        }).to_string();

        let mut tmp = NamedTempFile::with_suffix(".ipynb").unwrap();
        write!(tmp, "{}", &notebook).unwrap();
        let result = extract_jupyter(tmp.path());
        assert!(result.is_ok());
        let (text, metadata) = result.unwrap();
        assert!(text.contains("Test Notebook"));
        assert!(text.contains("import pandas"));
        assert_eq!(metadata.get("language").unwrap(), "python");
        assert_eq!(metadata.get("cell_count").unwrap(), "2");
        assert_eq!(metadata.get("code_cells").unwrap(), "1");
        assert_eq!(metadata.get("markdown_cells").unwrap(), "1");
    }

    #[test]
    fn test_extract_jupyter_source_as_string() {
        // Some notebooks have source as a single string instead of array
        let notebook = serde_json::json!({
            "metadata": { "language_info": { "name": "r" } },
            "cells": [
                { "cell_type": "code", "source": "x <- 1:10\nplot(x)" }
            ],
            "nbformat": 4,
            "nbformat_minor": 2
        }).to_string();

        let mut tmp = NamedTempFile::with_suffix(".ipynb").unwrap();
        write!(tmp, "{}", &notebook).unwrap();
        let result = extract_jupyter(tmp.path());
        assert!(result.is_ok());
        let (text, metadata) = result.unwrap();
        assert!(text.contains("x <- 1:10"));
        assert_eq!(metadata.get("language").unwrap(), "r");
    }

    #[test]
    fn test_extract_jupyter_invalid_json() {
        let mut tmp = NamedTempFile::with_suffix(".ipynb").unwrap();
        write!(tmp, "not valid json").unwrap();
        let result = extract_jupyter(tmp.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_jupyter_no_cells() {
        let notebook = serde_json::json!({ "metadata": {}, "nbformat": 4 }).to_string();
        let mut tmp = NamedTempFile::with_suffix(".ipynb").unwrap();
        write!(tmp, "{}", &notebook).unwrap();
        let result = extract_jupyter(tmp.path());
        assert!(result.is_err());
    }

    // --- Spreadsheet extraction tests ---

    #[test]
    fn test_extract_spreadsheet_invalid_file() {
        let mut tmp = NamedTempFile::with_suffix(".xlsx").unwrap();
        write!(tmp, "not a valid xlsx file").unwrap();
        let result = extract_spreadsheet(tmp.path());
        assert!(result.is_err());
    }

    // --- Allowed extensions tests ---

    #[test]
    fn test_allowed_extensions_new_formats() {
        use crate::allowed_extensions::AllowedExtensions;
        let ae = AllowedExtensions::default();
        // Project extensions
        assert!(ae.is_allowed("data.csv", "projects"));
        assert!(ae.is_allowed("data.tsv", "projects"));
        assert!(ae.is_allowed("notebook.ipynb", "projects"));
        // Library extensions
        assert!(ae.is_allowed("report.xlsx", "libraries"));
        assert!(ae.is_allowed("legacy.xls", "libraries"));
    }
}

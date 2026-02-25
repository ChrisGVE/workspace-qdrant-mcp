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

pub mod chunking;
pub mod extraction;
#[cfg(feature = "ocr")]
pub mod ocr;
pub mod types;

pub use self::types::{detect_document_type, DocumentProcessorError, DocumentProcessorResult};

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, info, warn};

#[cfg(feature = "ocr")]
use crate::ocr::OcrEngine;
use crate::tree_sitter::{extract_chunks, is_language_supported};
use crate::{ChunkingConfig, DocumentContent, DocumentResult, DocumentType};

use self::chunking::{chunk_text, convert_semantic_chunks_to_text_chunks};
use self::extraction::{
    extract_code, extract_csv, extract_docx, extract_epub, extract_iwork, extract_jupyter,
    extract_opendocument, extract_pdf, extract_pptx, extract_rtf, extract_spreadsheet,
    extract_text_with_encoding,
};

/// Document processor for extracting text from various file formats
#[derive(Debug)]
pub struct DocumentProcessor {
    chunking_config: ChunkingConfig,
    healthy: Arc<AtomicBool>,
    #[cfg(feature = "ocr")]
    ocr_engine: Option<Arc<OcrEngine>>,
}

impl Clone for DocumentProcessor {
    fn clone(&self) -> Self {
        Self {
            chunking_config: self.chunking_config.clone(),
            healthy: Arc::clone(&self.healthy),
            #[cfg(feature = "ocr")]
            ocr_engine: self.ocr_engine.clone(),
        }
    }
}

impl Default for DocumentProcessor {
    fn default() -> Self {
        Self {
            chunking_config: ChunkingConfig::default(),
            healthy: Arc::new(AtomicBool::new(true)),
            #[cfg(feature = "ocr")]
            ocr_engine: None,
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
            #[cfg(feature = "ocr")]
            ocr_engine: None,
        }
    }

    /// Alias for with_config for compatibility with existing tests
    pub fn with_chunking_config(chunking_config: ChunkingConfig) -> Self {
        Self::with_config(chunking_config)
    }

    /// Create a DocumentProcessor with OCR support.
    #[cfg(feature = "ocr")]
    pub fn with_ocr(chunking_config: ChunkingConfig, ocr_engine: OcrEngine) -> Self {
        Self {
            chunking_config,
            healthy: Arc::new(AtomicBool::new(true)),
            ocr_engine: Some(Arc::new(ocr_engine)),
        }
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

        #[cfg(feature = "ocr")]
        let ocr = self.ocr_engine.clone();

        // Run blocking file processing in a separate thread
        let content = tokio::task::spawn_blocking(move || {
            #[cfg(feature = "ocr")]
            {
                process_file_sync(&path, &collection_str, &chunking_config, ocr.as_deref())
            }
            #[cfg(not(feature = "ocr"))]
            {
                process_file_sync(&path, &collection_str, &chunking_config)
            }
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

        #[cfg(feature = "ocr")]
        let ocr = self.ocr_engine.clone();

        // Run blocking file processing in a separate thread
        let result = tokio::task::spawn_blocking(move || {
            #[cfg(feature = "ocr")]
            {
                process_file_sync(&path, &collection, &chunking_config, ocr.as_deref())
            }
            #[cfg(not(feature = "ocr"))]
            {
                process_file_sync(&path, &collection, &chunking_config)
            }
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
#[cfg(feature = "ocr")]
fn process_file_sync(
    file_path: &Path,
    collection: &str,
    chunking_config: &ChunkingConfig,
    ocr_engine: Option<&OcrEngine>,
) -> DocumentProcessorResult<DocumentContent> {
    process_file_sync_inner(file_path, collection, chunking_config, ocr_engine)
}

#[cfg(not(feature = "ocr"))]
fn process_file_sync(
    file_path: &Path,
    collection: &str,
    chunking_config: &ChunkingConfig,
) -> DocumentProcessorResult<DocumentContent> {
    process_file_sync_inner(file_path, collection, chunking_config)
}

/// Inner implementation of file processing, shared by OCR and non-OCR paths.
fn process_file_sync_inner(
    file_path: &Path,
    collection: &str,
    chunking_config: &ChunkingConfig,
    #[cfg(feature = "ocr")] ocr_engine: Option<&OcrEngine>,
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
            warn!(
                "Legacy binary format {} not supported, attempting text extraction: {:?}",
                fmt,
                file_path.file_name()
            );
            extract_text_with_encoding(file_path)?
        }
        DocumentType::Pages | DocumentType::Key => {
            let fmt = match &document_type {
                DocumentType::Pages => "Pages",
                DocumentType::Key => "Keynote",
                _ => unreachable!(),
            };
            warn!(
                "Apple iWork format {} has limited support, attempting extraction: {:?}",
                fmt,
                file_path.file_name()
            );
            extract_iwork(file_path, fmt)?
        }
        DocumentType::Code(lang) => extract_code(file_path, lang)?,
        DocumentType::Markdown => extract_text_with_encoding(file_path)?,
        DocumentType::Text => extract_text_with_encoding(file_path)?,
        DocumentType::Unknown => extract_text_with_encoding(file_path)?,
    };

    // Run OCR on embedded images for supported document types
    #[cfg(feature = "ocr")]
    let raw_text = ocr::enrich_text_with_ocr(file_path, raw_text, &mut metadata, ocr_engine);

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;
    use tempfile::NamedTempFile;

    use self::chunking::{chunk_by_characters, floor_char_boundary};
    use self::extraction::{
        clean_extracted_text, count_docx_images, count_pdf_images, extract_csv, extract_jupyter,
        extract_rtf, extract_spreadsheet, extract_text_from_xml_tags,
    };

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

        let result = processor
            .process_file(temp_file.path(), "test_collection")
            .await;
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

        let result = processor
            .process_file_content(temp_file.path(), "test_collection")
            .await;
        assert!(result.is_ok());

        let content = result.unwrap();
        assert!(content.raw_text.contains("Hello, World!"));
        assert_eq!(
            content.metadata.get("collection"),
            Some(&"test_collection".to_string())
        );
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
            ..ChunkingConfig::default()
        };

        let text =
            "This is a test. It has multiple sentences. Each one should be processed.";
        let chunks = chunk_text(text, &HashMap::new(), &config);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_text_with_paragraphs() {
        let config = ChunkingConfig {
            chunk_size: 100,
            overlap_size: 10,
            preserve_paragraphs: true,
            ..ChunkingConfig::default()
        };

        let text =
            "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
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
        let result = processor
            .process_file(Path::new("/nonexistent/file.txt"), "test")
            .await;

        assert!(result.is_err());
        matches!(
            result.unwrap_err(),
            DocumentProcessorError::FileNotFound(_)
        );
    }

    #[test]
    fn test_floor_char_boundary() {
        // ASCII-only string: all byte indices are char boundaries
        let ascii = "hello";
        assert_eq!(floor_char_boundary(ascii, 3), 3);
        assert_eq!(floor_char_boundary(ascii, 5), 5);
        assert_eq!(floor_char_boundary(ascii, 10), 5); // beyond end

        // Multi-byte: '-' is U+2500, encoded as 3 bytes (0xE2 0x94 0x80)
        let s = "ab\u{2500}cd"; // bytes: a(0) b(1) -(2,3,4) c(5) d(6)
        assert_eq!(floor_char_boundary(s, 2), 2); // start of -
        assert_eq!(floor_char_boundary(s, 3), 2); // inside - -> back to 2
        assert_eq!(floor_char_boundary(s, 4), 2); // inside - -> back to 2
        assert_eq!(floor_char_boundary(s, 5), 5); // start of c
    }

    #[test]
    fn test_chunk_by_paragraphs_with_multibyte_overlap() {
        // Text with multi-byte box-drawing characters that caused the original crash
        let text = "First paragraph with content.\n\n\
                    \u{250c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}\n\
                    \u{2502}   Box drawing test   \u{2502}\n\
                    \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}\n\n\
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
        let text = "Hello \u{2500}\u{2500}\u{2500} world \u{2500}\u{2500}\u{2500} end";
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
        assert_eq!(
            detect_document_type(Path::new("slides.pptx")),
            DocumentType::Pptx
        );
        assert_eq!(
            detect_document_type(Path::new("slides.ppt")),
            DocumentType::Ppt
        );
        assert_eq!(
            detect_document_type(Path::new("doc.odt")),
            DocumentType::Odt
        );
        assert_eq!(
            detect_document_type(Path::new("slides.odp")),
            DocumentType::Odp
        );
        assert_eq!(
            detect_document_type(Path::new("sheet.ods")),
            DocumentType::Ods
        );
        assert_eq!(
            detect_document_type(Path::new("doc.rtf")),
            DocumentType::Rtf
        );
        assert_eq!(
            detect_document_type(Path::new("legacy.doc")),
            DocumentType::Doc
        );
        assert_eq!(
            detect_document_type(Path::new("doc.pages")),
            DocumentType::Pages
        );
        assert_eq!(
            detect_document_type(Path::new("slides.key")),
            DocumentType::Key
        );
    }

    #[test]
    fn test_detect_document_type_case_insensitive() {
        assert_eq!(
            detect_document_type(Path::new("FILE.PPTX")),
            DocumentType::Pptx
        );
        assert_eq!(
            detect_document_type(Path::new("FILE.Rtf")),
            DocumentType::Rtf
        );
        assert_eq!(
            detect_document_type(Path::new("FILE.ODT")),
            DocumentType::Odt
        );
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
        write!(
            tmp,
            r"{{\rtf1\ansi\deff0 {{\b Bold text}} normal text \par New para}}"
        )
        .unwrap();
        let result = extract_rtf(tmp.path());
        assert!(result.is_ok());
        let (text, _) = result.unwrap();
        assert!(text.contains("Bold text"));
        assert!(text.contains("normal text"));
    }

    // --- New format detection tests ---

    #[test]
    fn test_detect_document_type_spreadsheet_formats() {
        assert_eq!(
            detect_document_type(Path::new("data.xlsx")),
            DocumentType::Xlsx
        );
        assert_eq!(
            detect_document_type(Path::new("data.xls")),
            DocumentType::Xls
        );
        assert_eq!(
            detect_document_type(Path::new("DATA.XLSX")),
            DocumentType::Xlsx
        );
        assert_eq!(
            detect_document_type(Path::new("report.XLS")),
            DocumentType::Xls
        );
    }

    #[test]
    fn test_detect_document_type_csv() {
        assert_eq!(
            detect_document_type(Path::new("data.csv")),
            DocumentType::Csv
        );
        assert_eq!(
            detect_document_type(Path::new("data.tsv")),
            DocumentType::Csv
        );
        assert_eq!(
            detect_document_type(Path::new("DATA.CSV")),
            DocumentType::Csv
        );
    }

    #[test]
    fn test_detect_document_type_jupyter() {
        assert_eq!(
            detect_document_type(Path::new("notebook.ipynb")),
            DocumentType::Jupyter
        );
        assert_eq!(
            detect_document_type(Path::new("analysis.IPYNB")),
            DocumentType::Jupyter
        );
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
        assert_eq!(
            detect_document_type(Path::new("doc.rst")),
            DocumentType::Text
        );
        assert_eq!(
            detect_document_type(Path::new("notes.org")),
            DocumentType::Text
        );
        assert_eq!(
            detect_document_type(Path::new("guide.adoc")),
            DocumentType::Text
        );
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
        })
        .to_string();

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
        })
        .to_string();

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
        let notebook =
            serde_json::json!({ "metadata": {}, "nbformat": 4 }).to_string();
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

    // --- Image counting tests ---

    #[test]
    fn test_count_docx_images_with_media() {
        use std::fs::File;
        // Build a minimal DOCX ZIP with images in word/media/
        let temp = NamedTempFile::new().unwrap();
        {
            let mut zip = zip::ZipWriter::new(&temp);
            let options = zip::write::SimpleFileOptions::default();
            zip.start_file("word/document.xml", options).unwrap();
            zip.write_all(b"<w:document/>").unwrap();
            zip.start_file("word/media/image1.png", options).unwrap();
            zip.write_all(b"PNG_DATA").unwrap();
            zip.start_file("word/media/image2.jpeg", options).unwrap();
            zip.write_all(b"JPEG_DATA").unwrap();
            zip.start_file("word/media/chart1.xml", options).unwrap();
            zip.write_all(b"<chart/>").unwrap();
            zip.finish().unwrap();
        }

        let file = File::open(temp.path()).unwrap();
        let archive = zip::ZipArchive::new(file).unwrap();
        assert_eq!(count_docx_images(&archive), 2);
    }

    #[test]
    fn test_count_docx_images_no_media() {
        use std::fs::File;
        let temp = NamedTempFile::new().unwrap();
        {
            let mut zip = zip::ZipWriter::new(&temp);
            let options = zip::write::SimpleFileOptions::default();
            zip.start_file("word/document.xml", options).unwrap();
            zip.write_all(b"<w:document/>").unwrap();
            zip.finish().unwrap();
        }

        let file = File::open(temp.path()).unwrap();
        let archive = zip::ZipArchive::new(file).unwrap();
        assert_eq!(count_docx_images(&archive), 0);
    }

    #[test]
    fn test_count_pdf_images_nonexistent_file() {
        // Gracefully returns 0 for non-existent files
        assert_eq!(count_pdf_images(Path::new("/tmp/nonexistent.pdf")), 0);
    }

    #[test]
    fn test_count_pdf_images_invalid_file() {
        // Gracefully returns 0 for invalid PDF files
        let temp = NamedTempFile::with_suffix(".pdf").unwrap();
        std::fs::write(temp.path(), b"not a pdf").unwrap();
        assert_eq!(count_pdf_images(temp.path()), 0);
    }

    #[cfg(feature = "ocr")]
    mod ocr_integration_tests {
        use super::*;
        use crate::ocr::{OcrConfig, OcrEngine};

        #[test]
        fn test_enrich_text_with_ocr_no_engine() {
            let temp = NamedTempFile::with_suffix(".txt").unwrap();
            std::fs::write(temp.path(), "hello").unwrap();
            let mut metadata = HashMap::new();
            let result = ocr::enrich_text_with_ocr(
                temp.path(),
                "original text".to_string(),
                &mut metadata,
                None,
            );
            assert_eq!(result, "original text");
        }

        #[test]
        fn test_enrich_text_with_ocr_no_images() {
            let temp = NamedTempFile::with_suffix(".txt").unwrap();
            std::fs::write(temp.path(), "hello world").unwrap();

            let config = OcrConfig::default();
            if !config.tessdata_path.exists() {
                return; // Skip without Tesseract
            }
            let engine = match OcrEngine::new(&config) {
                Ok(e) => e,
                Err(_) => return,
            };

            let mut metadata = HashMap::new();
            let result = ocr::enrich_text_with_ocr(
                temp.path(),
                "original text".to_string(),
                &mut metadata,
                Some(&engine),
            );
            assert_eq!(result, "original text");
            assert_eq!(metadata.get("images_detected").unwrap(), "0");
        }

        #[test]
        fn test_with_ocr_constructor() {
            let config = OcrConfig::default();
            if !config.tessdata_path.exists() {
                return;
            }
            let engine = match OcrEngine::new(&config) {
                Ok(e) => e,
                Err(_) => return,
            };

            let processor =
                DocumentProcessor::with_ocr(ChunkingConfig::default(), engine);
            assert!(processor.ocr_engine.is_some());
        }

        #[test]
        fn test_processor_without_ocr_has_none() {
            let processor = DocumentProcessor::new();
            assert!(processor.ocr_engine.is_none());
        }
    }
}

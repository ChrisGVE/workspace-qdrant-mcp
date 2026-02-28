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

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tracing::{debug, info, warn};

#[cfg(feature = "ocr")]
use crate::ocr::OcrEngine;
use crate::tree_sitter::{extract_chunks, is_language_supported};
use crate::{ChunkingConfig, DocumentContent, DocumentResult, DocumentType, TextChunk};

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

/// Extract raw text and metadata from a file based on its document type.
fn extract_by_document_type(
    file_path: &Path,
    document_type: &DocumentType,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    match document_type {
        DocumentType::Pdf => extract_pdf(file_path),
        DocumentType::Epub => extract_epub(file_path),
        DocumentType::Docx => extract_docx(file_path),
        DocumentType::Pptx => extract_pptx(file_path),
        DocumentType::Odt | DocumentType::Odp | DocumentType::Ods => {
            let fmt = match document_type {
                DocumentType::Odt => "odt",
                DocumentType::Odp => "odp",
                DocumentType::Ods => "ods",
                _ => unreachable!(),
            };
            extract_opendocument(file_path, fmt)
        }
        DocumentType::Rtf => extract_rtf(file_path),
        DocumentType::Xlsx | DocumentType::Xls => extract_spreadsheet(file_path),
        DocumentType::Csv => extract_csv(file_path),
        DocumentType::Jupyter => extract_jupyter(file_path),
        DocumentType::Ppt | DocumentType::Doc => {
            let fmt = match document_type {
                DocumentType::Ppt => "PPT",
                DocumentType::Doc => "DOC",
                _ => unreachable!(),
            };
            warn!(
                "Legacy binary format {} not supported, attempting text extraction: {:?}",
                fmt,
                file_path.file_name()
            );
            extract_text_with_encoding(file_path)
        }
        DocumentType::Pages | DocumentType::Key => {
            let fmt = match document_type {
                DocumentType::Pages => "Pages",
                DocumentType::Key => "Keynote",
                _ => unreachable!(),
            };
            warn!(
                "Apple iWork format {} has limited support, attempting extraction: {:?}",
                fmt,
                file_path.file_name()
            );
            extract_iwork(file_path, fmt)
        }
        DocumentType::Code(lang) => extract_code(file_path, lang),
        DocumentType::Markdown | DocumentType::Text | DocumentType::Unknown => {
            extract_text_with_encoding(file_path)
        }
    }
}

/// Generate chunks from extracted text, using semantic chunking for supported code files.
fn generate_chunks(
    raw_text: &str,
    file_path: &Path,
    document_type: &DocumentType,
    metadata: &HashMap<String, String>,
    chunking_config: &ChunkingConfig,
) -> Vec<TextChunk> {
    match document_type {
        DocumentType::Code(lang) if is_language_supported(lang) => {
            match extract_chunks(raw_text, file_path, chunking_config.chunk_size) {
                Ok(semantic_chunks) => {
                    info!(
                        "Extracted {} semantic chunks from {:?} ({})",
                        semantic_chunks.len(),
                        file_path.file_name(),
                        lang
                    );
                    convert_semantic_chunks_to_text_chunks(semantic_chunks, metadata)
                }
                Err(e) => {
                    warn!(
                        "Semantic chunking failed for {:?}: {}, falling back to text chunking",
                        file_path, e
                    );
                    chunk_text(raw_text, metadata, chunking_config)
                }
            }
        }
        _ => chunk_text(raw_text, metadata, chunking_config),
    }
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

    let (raw_text, mut metadata) = extract_by_document_type(file_path, &document_type)?;

    // Run OCR on embedded images for supported document types
    #[cfg(feature = "ocr")]
    let raw_text = ocr::enrich_text_with_ocr(file_path, raw_text, &mut metadata, ocr_engine);

    // Add collection to metadata
    metadata.insert("collection".to_string(), collection.to_string());

    // Generate chunks
    let chunks =
        generate_chunks(&raw_text, file_path, &document_type, &metadata, chunking_config);

    Ok(DocumentContent {
        raw_text,
        metadata,
        document_type,
        chunks,
    })
}

#[cfg(test)]
mod tests;

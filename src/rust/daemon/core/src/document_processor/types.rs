//! Document processor error types and document type detection.

use std::path::Path;

use thiserror::Error;

use crate::DocumentType;

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

    #[error("OCR extraction error: {0}")]
    OcrError(String),

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Processing task failed: {0}")]
    TaskError(String),
}

/// Result type for document processing operations
pub type DocumentProcessorResult<T> = Result<T, DocumentProcessorError>;

/// Detect document type from file extension
pub fn detect_document_type(file_path: &Path) -> DocumentType {
    use wqm_common::classification;

    // Check compound extensions first (before standard Path::extension())
    let filename = file_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let lower_filename = filename.to_lowercase();

    // Handle compound extensions (.d.ts, .d.mts, .d.cts)
    for (suffix, _) in classification::compound_extensions() {
        if lower_filename.ends_with(&format!(".{suffix}")) {
            if let Some(lang) = classification::compound_extension_language(suffix) {
                return DocumentType::Code(lang.to_string());
            }
        }
    }

    let extension = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    // Check for document_type override first (pdf, markdown, csv, etc.)
    if let Some(doc_type) = classification::extension_to_document_type(&extension) {
        return match doc_type {
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
            "numbers" => DocumentType::Numbers,
            "csv" => DocumentType::Csv,
            "jupyter" => DocumentType::Jupyter,
            "pages" => DocumentType::Pages,
            "key" => DocumentType::Key,
            "markdown" => DocumentType::Markdown,
            "text" => DocumentType::Text,
            "unknown" => DocumentType::Unknown,
            _ => DocumentType::Unknown,
        };
    }

    // Check for language mapping (-> DocumentType::Code)
    if let Some(lang) = classification::extension_to_language(&extension) {
        return DocumentType::Code(lang.to_string());
    }

    DocumentType::Unknown
}

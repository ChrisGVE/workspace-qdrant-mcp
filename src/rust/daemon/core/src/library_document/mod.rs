//! Library document structural splitting and token-based chunking.
//!
//! Splits documents into structural units (pages, slides, sections, chapters)
//! and chunks each unit into token-budgeted pieces. Creates parent records
//! (no vectors) for each structural unit and child chunk records that reference
//! their parent.
//!
//! Two document families:
//! - **Page-based**: PDF, DOCX, PPTX, ODP, ODT, ODS, RTF
//! - **Stream-based**: EPUB, HTML, Markdown, plain text

use std::collections::HashMap;
use std::path::Path;

use thiserror::Error;
use tracing::debug;

use crate::parent_unit::{self, ParentUnitRecord};
use crate::tokenizer::{ModelTokenizer, TokenizerError};

mod format_splitters;
mod text_processing;
mod xml_helpers;

// Re-export all public splitter functions
pub use format_splitters::{
    split_docx, split_epub, split_html, split_markdown, split_odp, split_ods, split_odt, split_pdf,
    split_plain_text, split_pptx, split_rtf,
};

/// Errors from library document processing
#[derive(Error, Debug)]
pub enum LibraryDocumentError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Extraction error ({format}): {message}")]
    Extraction { format: String, message: String },

    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
}

/// A structural unit extracted from a document (page, slide, section, chapter).
#[derive(Debug, Clone)]
pub struct StructuralUnit {
    /// Unit type for Qdrant payload (pdf_page, epub_section, etc.)
    pub unit_type: String,
    /// Locator JSON for this unit within the document
    pub unit_locator: serde_json::Value,
    /// Full text content of this unit
    pub text: String,
    /// Optional title for this unit (slide title, chapter title, heading)
    pub title: Option<String>,
}

/// A child chunk with its parent reference.
#[derive(Debug, Clone)]
pub struct LibraryChunk {
    /// The chunk text (raw, without header)
    pub text_raw: String,
    /// The chunk text with header prepended (for indexing/embedding)
    pub text_indexed: String,
    /// Number of tokens in text_indexed
    pub token_count: usize,
    /// Character start offset within the parent unit's text
    pub char_start: usize,
    /// Character end offset within the parent unit's text
    pub char_end: usize,
    /// Chunk index within this parent unit
    pub chunk_index: usize,
    /// Point ID of the parent unit record
    pub parent_unit_id: String,
}

/// Result of processing a single structural unit.
#[derive(Debug, Clone)]
pub struct ProcessedUnit {
    /// The parent record (no vectors)
    pub parent: ParentUnitRecord,
    /// Token-based child chunks
    pub chunks: Vec<LibraryChunk>,
}

/// Result of processing an entire library document.
#[derive(Debug, Clone)]
pub struct LibraryDocumentResult {
    /// All processed units (parents + their chunks)
    pub units: Vec<ProcessedUnit>,
    /// Document metadata extracted during processing
    pub metadata: HashMap<String, String>,
    /// Total chunks across all units
    pub total_chunks: usize,
}

/// Split a document into structural units based on its format.
///
/// Dispatches to the appropriate format-specific splitter.
pub fn split_document(
    file_path: &Path,
    source_format: &str,
) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    match source_format {
        "pdf" => split_pdf(file_path),
        "pptx" => split_pptx(file_path),
        "docx" => split_docx(file_path),
        "odp" => split_odp(file_path),
        "odt" => split_odt(file_path),
        "ods" => split_ods(file_path),
        "rtf" => split_rtf(file_path),
        "epub" => split_epub(file_path),
        "html" => split_html(file_path),
        "markdown" | "md" => split_markdown(file_path),
        "text" | "txt" => split_plain_text(file_path),
        other => Err(LibraryDocumentError::UnsupportedFormat(other.into())),
    }
}

/// Process a library document: split into structural units, create parents,
/// and chunk each unit with token-based budgets.
///
/// Returns a `LibraryDocumentResult` containing all parent records and their
/// child chunks ready for embedding and storage.
pub fn process_library_document(
    file_path: &Path,
    source_format: &str,
    doc_id: &str,
    doc_fingerprint: &str,
    doc_title: Option<&str>,
    tokenizer: &ModelTokenizer,
    target_tokens: usize,
    overlap_tokens: usize,
) -> Result<LibraryDocumentResult, LibraryDocumentError> {
    let units = split_document(file_path, source_format)?;

    if units.is_empty() {
        return Ok(LibraryDocumentResult {
            units: vec![],
            metadata: HashMap::new(),
            total_chunks: 0,
        });
    }

    let mut processed_units = Vec::with_capacity(units.len());
    let mut total_chunks = 0;

    for unit in &units {
        // Create parent record
        let parent = create_parent_for_unit(doc_id, doc_fingerprint, unit);

        // Build header for indexed text
        let header = build_chunk_header(doc_title, &unit.title, &unit.unit_locator);

        // Chunk the unit text
        let token_chunks = tokenizer.chunk_by_tokens(&unit.text, target_tokens, overlap_tokens)?;

        let mut chunks = Vec::with_capacity(token_chunks.len());
        for (i, tc) in token_chunks.iter().enumerate() {
            let text_indexed = if header.is_empty() {
                tc.text.clone()
            } else {
                format!("{}\n{}", header, tc.text)
            };

            chunks.push(LibraryChunk {
                text_raw: tc.text.clone(),
                text_indexed,
                token_count: tc.token_count,
                char_start: tc.char_start,
                char_end: tc.char_end,
                chunk_index: i,
                parent_unit_id: parent.point_id.clone(),
            });
        }

        total_chunks += chunks.len();
        processed_units.push(ProcessedUnit { parent, chunks });
    }

    let mut metadata = HashMap::new();
    metadata.insert("unit_count".to_string(), processed_units.len().to_string());
    metadata.insert("total_chunks".to_string(), total_chunks.to_string());
    metadata.insert("source_format".to_string(), source_format.to_string());

    debug!(
        "Processed library document: {} units, {} total chunks",
        processed_units.len(),
        total_chunks
    );

    Ok(LibraryDocumentResult {
        units: processed_units,
        metadata,
        total_chunks,
    })
}

/// Create a parent record for a structural unit.
fn create_parent_for_unit(
    doc_id: &str,
    doc_fingerprint: &str,
    unit: &StructuralUnit,
) -> ParentUnitRecord {
    let point_id = parent_unit::parent_point_id(doc_id, &unit.unit_type, &unit.unit_locator);
    ParentUnitRecord {
        point_id,
        doc_id: doc_id.to_string(),
        doc_fingerprint: doc_fingerprint.to_string(),
        unit_type: unit.unit_type.clone(),
        unit_locator: unit.unit_locator.clone(),
        unit_text: unit.text.clone(),
        unit_char_len: unit.text.len(),
        unit_hash: parent_unit::sha256_hex(&unit.text),
    }
}

/// Build a header string for chunk indexing.
///
/// Format: `{doc_title} - {unit_label}` where unit_label is derived from
/// the unit's title or locator (e.g., "Page 5", "Slide 3", "Chapter: Introduction").
fn build_chunk_header(
    doc_title: Option<&str>,
    unit_title: &Option<String>,
    unit_locator: &serde_json::Value,
) -> String {
    let unit_label = if let Some(title) = unit_title {
        title.clone()
    } else if let Some(page) = unit_locator.get("page").and_then(|v| v.as_u64()) {
        format!("Page {}", page)
    } else if let Some(slide) = unit_locator.get("slide").and_then(|v| v.as_u64()) {
        format!("Slide {}", slide)
    } else if let Some(section) = unit_locator.get("section").and_then(|v| v.as_u64()) {
        format!("Section {}", section)
    } else if let Some(sheet) = unit_locator.get("sheet").and_then(|v| v.as_str()) {
        format!("Sheet: {}", sheet)
    } else if let Some(idx) = unit_locator.get("section_index").and_then(|v| v.as_u64()) {
        format!("Section {}", idx + 1)
    } else {
        return doc_title.unwrap_or("").to_string();
    };

    match doc_title {
        Some(title) if !title.is_empty() => format!("{} - {}", title, unit_label),
        _ => unit_label,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use crate::parent_unit::{
        UNIT_TYPE_DOCX_SECTION, UNIT_TYPE_EPUB_SECTION, UNIT_TYPE_PDF_PAGE, UNIT_TYPE_TEXT_SECTION,
    };

    #[test]
    fn test_build_chunk_header_page() {
        let locator = serde_json::json!({"page": 5});
        let header = build_chunk_header(Some("My PDF"), &None, &locator);
        assert_eq!(header, "My PDF - Page 5");
    }

    #[test]
    fn test_build_chunk_header_slide() {
        let locator = serde_json::json!({"slide": 3});
        let header = build_chunk_header(Some("Presentation"), &None, &locator);
        assert_eq!(header, "Presentation - Slide 3");
    }

    #[test]
    fn test_build_chunk_header_with_unit_title() {
        let locator = serde_json::json!({"spine_id": "ch1"});
        let title = Some("Introduction".to_string());
        let header = build_chunk_header(Some("My Book"), &title, &locator);
        assert_eq!(header, "My Book - Introduction");
    }

    #[test]
    fn test_build_chunk_header_no_doc_title() {
        let locator = serde_json::json!({"page": 1});
        let header = build_chunk_header(None, &None, &locator);
        assert_eq!(header, "Page 1");
    }

    #[test]
    fn test_build_chunk_header_sheet() {
        let locator = serde_json::json!({"sheet": "Revenue"});
        let header = build_chunk_header(Some("Budget"), &None, &locator);
        assert_eq!(header, "Budget - Sheet: Revenue");
    }

    #[test]
    fn test_split_document_unsupported_format() {
        let result = split_document(Path::new("test.xyz"), "xyz");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Unsupported format"));
    }

    #[test]
    fn test_structural_unit_types() {
        assert_eq!(UNIT_TYPE_PDF_PAGE, "pdf_page");
        assert_eq!(UNIT_TYPE_EPUB_SECTION, "epub_section");
        assert_eq!(UNIT_TYPE_DOCX_SECTION, "docx_section");
        assert_eq!(UNIT_TYPE_TEXT_SECTION, "text_section");
    }

    fn get_test_tokenizer() -> Option<ModelTokenizer> {
        ModelTokenizer::from_model_cache(None).ok()
    }

    #[test]
    fn test_process_markdown_document() {
        let tokenizer = match get_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        let dir = std::env::temp_dir().join("wqm_test_lib_doc");
        let _ = std::fs::create_dir_all(&dir);
        let md_path = dir.join("test.md");
        std::fs::write(
            &md_path,
            "# Chapter 1\n\nThis is the first chapter with enough text to be meaningful.\n\n## Section 1.1\n\nDetailed content goes here with more words to fill it out.",
        )
        .unwrap();

        let result = process_library_document(
            &md_path,
            "markdown",
            "doc-test-1",
            "fp-test-1",
            Some("Test Document"),
            &tokenizer,
            105,
            12,
        )
        .unwrap();

        assert!(!result.units.is_empty(), "Should have at least one unit");
        assert!(result.total_chunks > 0, "Should have at least one chunk");

        for unit in &result.units {
            assert_eq!(unit.parent.doc_id, "doc-test-1");
            for chunk in &unit.chunks {
                assert_eq!(chunk.parent_unit_id, unit.parent.point_id);
                assert!(!chunk.text_raw.is_empty());
                assert!(chunk.text_indexed.contains("Test Document"));
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_process_plain_text_document() {
        let tokenizer = match get_test_tokenizer() {
            Some(t) => t,
            None => return,
        };

        let dir = std::env::temp_dir().join("wqm_test_lib_doc_txt");
        let _ = std::fs::create_dir_all(&dir);
        let txt_path = dir.join("test.txt");
        std::fs::write(
            &txt_path,
            "A simple paragraph of text for testing purposes.",
        )
        .unwrap();

        let result = process_library_document(
            &txt_path,
            "text",
            "doc-txt-1",
            "fp-txt-1",
            None,
            &tokenizer,
            105,
            12,
        )
        .unwrap();

        assert_eq!(result.units.len(), 1);
        assert!(result.total_chunks >= 1);
        let chunk = &result.units[0].chunks[0];
        assert!(!chunk.text_raw.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}

//! MOBI text extraction via the mobi crate (pure Rust).

use std::collections::HashMap;
use std::path::Path;

use super::xml_utils::clean_extracted_text;
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract text from a MOBI file.
///
/// The mobi crate handles PalmDOC, MOBI, and basic AZW files.
/// Content is returned as HTML-like markup; html2text converts it to plain text.
pub fn extract_mobi(
    file_path: &Path,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "mobi".to_string());

    let book = mobi::Mobi::from_path(file_path)
        .map_err(|e| DocumentProcessorError::MobiExtraction(e.to_string()))?;

    metadata.insert("title".to_string(), book.title());
    if let Some(author) = book.author() {
        metadata.insert("author".to_string(), author);
    }
    if let Some(publisher) = book.publisher() {
        metadata.insert("publisher".to_string(), publisher);
    }

    // content_as_string_lossy never fails — best-effort decoding
    let raw = book.content_as_string_lossy();
    // MOBI content is HTML-like; strip tags to get plain text
    let text = html2text::from_read(raw.as_bytes(), 80);

    Ok((clean_extracted_text(&text), metadata))
}

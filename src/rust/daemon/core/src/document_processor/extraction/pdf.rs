//! PDF text extraction.
//!
//! Uses pdf-extract as the primary extractor, with lopdf as a fallback.
//! Both are pure-Rust and statically linked (Principle 8).
//!
//! TODO: Replace with a statically-linked Poppler C++ integration for ~4x
//! throughput improvement. The pdftotext crate's static-poppler feature has
//! a build.rs include-path bug that prevents use today; a proper wrapper
//! crate is needed (see docs/specs/14-future-development.md).

use std::collections::HashMap;
use std::path::Path;

use super::xml_utils::clean_extracted_text;
use crate::document_processor::types::DocumentProcessorResult;

/// Extract text from PDF using pdf-extract, with lopdf fallback.
///
/// Wrapped in `catch_unwind` because `pdf-extract` (via `type1-encoding-parser`)
/// panics on certain malformed Type 1 font encodings instead of returning an error.
/// Old scanned PDFs (e.g. 1960s documents) often trigger this — they render fine
/// in PDF viewers (which paint glyphs from images) but have broken encoding tables
/// that text extraction must interpret. The lopdf fallback handles many such cases;
/// if it also yields nothing the document is treated as image-only (empty text,
/// no error) so it is still indexed by filename and metadata.
pub fn extract_pdf(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
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
        Ok(Err(_)) | Err(_) => {
            // pdf-extract failed or panicked — try lopdf as a best-effort fallback.
            let text = try_lopdf_text(file_path);
            if text.is_empty() {
                metadata.insert("image_only".to_string(), "true".to_string());
            } else {
                metadata.insert("extraction_mode".to_string(), "lopdf_fallback".to_string());
            }
            Ok((text, metadata))
        }
    }
}

/// Attempt text extraction via lopdf for PDFs that pdf-extract cannot handle.
/// Returns an empty string on any failure (including truly image-only PDFs).
fn try_lopdf_text(file_path: &Path) -> String {
    let doc = match lopdf::Document::load(file_path) {
        Ok(d) => d,
        Err(_) => return String::new(),
    };
    let page_nums: Vec<u32> = doc.get_pages().keys().cloned().collect();
    if page_nums.is_empty() {
        return String::new();
    }
    match doc.extract_text(&page_nums) {
        Ok(text) => clean_extracted_text(&text),
        Err(_) => String::new(),
    }
}

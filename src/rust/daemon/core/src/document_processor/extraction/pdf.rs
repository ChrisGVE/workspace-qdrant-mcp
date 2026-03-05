//! PDF text extraction.

use std::collections::HashMap;
use std::path::Path;

use super::xml_utils::clean_extracted_text;
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract text from PDF using pdf-extract
///
/// Wrapped in `catch_unwind` because `pdf-extract` (via `type1-encoding-parser`)
/// panics on certain malformed Type 1 font encodings instead of returning an error.
pub fn extract_pdf(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "pdf".to_string());

    // Count images (Tier 1: metadata only)
    let image_count = count_pdf_images(file_path);
    metadata.insert("images_detected".to_string(), image_count.to_string());

    let path_buf = file_path.to_path_buf();
    let result = std::panic::catch_unwind(|| pdf_extract::extract_text(&path_buf));

    match result {
        Ok(Ok(text)) => {
            let cleaned_text = clean_extracted_text(&text);
            metadata.insert("page_count".to_string(), "unknown".to_string());
            Ok((cleaned_text, metadata))
        }
        Ok(Err(e)) => Err(DocumentProcessorError::PdfExtraction(e.to_string())),
        Err(_panic) => Err(DocumentProcessorError::PdfExtraction(format!(
            "PDF parsing panicked (likely malformed font encoding): {}",
            file_path.display()
        ))),
    }
}

/// Count images in a PDF by scanning for image XObjects in page resources.
pub fn count_pdf_images(file_path: &Path) -> usize {
    let doc = match lopdf::Document::load(file_path) {
        Ok(d) => d,
        Err(_) => return 0,
    };

    let mut count = 0;
    for (_page_num, page_id) in doc.get_pages() {
        if let Ok(page) = doc.get_object(page_id) {
            if let Ok(resources) = page
                .as_dict()
                .and_then(|d| d.get(b"Resources"))
                .and_then(|r| doc.dereference(r).map(|(_, obj)| obj))
                .and_then(|obj| obj.as_dict())
            {
                if let Ok(xobjects) = resources
                    .get(b"XObject")
                    .and_then(|x| doc.dereference(x).map(|(_, obj)| obj))
                    .and_then(|obj| obj.as_dict())
                {
                    for (_name, xobj_ref) in xobjects.iter() {
                        if let Ok((_, xobj)) = doc.dereference(xobj_ref) {
                            if let Ok(stream) = xobj.as_stream() {
                                if stream
                                    .dict
                                    .get(b"Subtype")
                                    .ok()
                                    .and_then(|v| v.as_name_str().ok())
                                    == Some("Image")
                                {
                                    count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    count
}

//! OpenDocument format (ODT/ODP/ODS) text extraction.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use super::xml_utils::{clean_extracted_text, extract_text_from_xml_tags};
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract text from OpenDocument formats (ODT/ODP/ODS) -- all are ZIP-based with content.xml
pub fn extract_opendocument(
    file_path: &Path,
    format_name: &str,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), format_name.to_string());

    let file = std::fs::File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| {
        DocumentProcessorError::DocxExtraction(format!(
            "{}: {}",
            format_name.to_uppercase(),
            e
        ))
    })?;

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
        return Err(DocumentProcessorError::DocxExtraction(format!(
            "No text content found in {} file",
            format_name.to_uppercase()
        )));
    }

    Ok((clean_extracted_text(&text), metadata))
}

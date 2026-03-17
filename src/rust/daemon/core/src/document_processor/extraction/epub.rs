//! EPUB text extraction via rbook.

use std::collections::HashMap;
use std::path::Path;

use super::xml_utils::clean_extracted_text;
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract text from an EPUB using rbook.
pub fn extract_epub(
    file_path: &Path,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "epub".to_string());

    let epub = rbook::Epub::open(file_path)
        .map_err(|e| DocumentProcessorError::EpubExtraction(e.to_string()))?;

    // Extract metadata
    if let Some(entry) = epub.metadata().title() {
        metadata.insert("title".to_string(), entry.value().to_string());
    }
    if let Some(entry) = epub.metadata().creators().next() {
        metadata.insert("author".to_string(), entry.value().to_string());
    }

    // Count images from the manifest (Manifest::images() filters by image MIME types)
    let image_count = epub.manifest().images().count();
    metadata.insert("images_detected".to_string(), image_count.to_string());

    // Extract text from all spine items in reading order
    let mut all_text = String::new();
    let mut chapter_count: usize = 0;

    for result in epub.reader() {
        let data = result.map_err(|e| DocumentProcessorError::EpubExtraction(e.to_string()))?;
        let html = data.content();
        let text = html2text::from_read(html.as_bytes(), 80);
        all_text.push_str(&text);
        all_text.push_str("\n\n");
        chapter_count += 1;
    }

    metadata.insert("chapter_count".to_string(), chapter_count.to_string());
    Ok((clean_extracted_text(&all_text), metadata))
}

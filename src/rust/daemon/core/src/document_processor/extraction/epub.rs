//! EPUB text extraction.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use super::xml_utils::clean_extracted_text;
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract text from EPUB using epub crate
pub fn extract_epub(
    file_path: &Path,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
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

    // Count images (Tier 1: metadata only)
    let image_count = count_epub_images(&doc);
    metadata.insert("images_detected".to_string(), image_count.to_string());

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

/// Count images in an EPUB by scanning resources with image MIME types.
pub fn count_epub_images(doc: &epub::doc::EpubDoc<std::io::BufReader<File>>) -> usize {
    doc.resources
        .values()
        .filter(|item| item.mime.starts_with("image/"))
        .count()
}

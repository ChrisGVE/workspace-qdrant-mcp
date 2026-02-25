//! DOCX text extraction.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use super::xml_utils::clean_extracted_text;
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract text from DOCX (ZIP file with XML content)
pub fn extract_docx(
    file_path: &Path,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "docx".to_string());

    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| DocumentProcessorError::DocxExtraction(e.to_string()))?;

    // Count images (Tier 1: metadata only)
    let image_count = count_docx_images(&archive);
    metadata.insert("images_detected".to_string(), image_count.to_string());

    let mut text = String::new();

    if let Ok(mut document_file) = archive.by_name("word/document.xml") {
        let mut content = String::new();
        document_file.read_to_string(&mut content)?;
        text = extract_text_from_docx_xml(&content);
    }

    if text.is_empty() {
        return Err(DocumentProcessorError::DocxExtraction(
            "No text content found in DOCX".to_string(),
        ));
    }

    Ok((clean_extracted_text(&text), metadata))
}

/// Extract text from DOCX XML content
pub fn extract_text_from_docx_xml(xml_content: &str) -> String {
    let mut text = String::new();
    let mut in_text_tag = false;
    let mut current_text = String::new();

    for line in xml_content.lines() {
        for part in line.split('<') {
            if part.starts_with("w:t") {
                in_text_tag = true;
                if let Some(content_start) = part.find('>') {
                    current_text.push_str(&part[content_start + 1..]);
                }
            } else if part.starts_with("/w:t") {
                in_text_tag = false;
                if !current_text.is_empty() {
                    text.push_str(&current_text);
                    current_text.clear();
                }
            } else if part.starts_with("w:p") && !part.starts_with("w:pPr") {
                if !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
            } else if in_text_tag {
                if let Some(end_pos) = part.find('>') {
                    current_text.push_str(&part[end_pos + 1..]);
                } else {
                    current_text.push_str(part);
                }
            }
        }
    }

    text
}

/// Count images in a DOCX by scanning the word/media/ directory in the ZIP.
pub fn count_docx_images(archive: &zip::ZipArchive<File>) -> usize {
    let image_extensions = [
        "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "svg", "emf", "wmf",
    ];

    (0..archive.len())
        .filter_map(|i| archive.name_for_index(i))
        .filter(|name| {
            name.starts_with("word/media/")
                && image_extensions
                    .iter()
                    .any(|ext| name.to_lowercase().ends_with(ext))
        })
        .count()
}

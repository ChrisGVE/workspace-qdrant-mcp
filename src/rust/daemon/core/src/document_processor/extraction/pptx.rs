//! PowerPoint PPTX text extraction.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use super::xml_utils::{clean_extracted_text, extract_text_from_xml_tags};
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract text from PowerPoint PPTX file (ZIP-based, slides in ppt/slides/slide*.xml)
pub fn extract_pptx(
    file_path: &Path,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "pptx".to_string());

    let file = std::fs::File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| DocumentProcessorError::DocxExtraction(format!("PPTX: {}", e)))?;

    let mut all_text = String::new();
    let mut slide_count = 0u32;

    // Collect slide file names (they're numbered: slide1.xml, slide2.xml, etc.)
    let slide_names: Vec<String> = (0..archive.len())
        .filter_map(|i| {
            archive.by_index(i).ok().and_then(|f| {
                let name = f.name().to_string();
                if name.starts_with("ppt/slides/slide") && name.ends_with(".xml") {
                    Some(name)
                } else {
                    None
                }
            })
        })
        .collect();

    for slide_name in &slide_names {
        if let Ok(mut slide_file) = archive.by_name(slide_name) {
            let mut content = String::new();
            slide_file.read_to_string(&mut content)?;
            let slide_text = extract_text_from_xml_tags(&content, "a:t");
            if !slide_text.is_empty() {
                slide_count += 1;
                if !all_text.is_empty() {
                    all_text.push('\n');
                }
                all_text.push_str(&slide_text);
            }
        }
    }

    metadata.insert("slide_count".to_string(), slide_count.to_string());

    if all_text.is_empty() {
        return Err(DocumentProcessorError::DocxExtraction(
            "No text content found in PPTX".to_string(),
        ));
    }

    Ok((clean_extracted_text(&all_text), metadata))
}

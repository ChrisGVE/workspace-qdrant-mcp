//! Apple iWork format (.pages, .key) text extraction.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use super::xml_utils::{clean_extracted_text, extract_text_from_xml_tags};
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract text from Apple iWork formats (.pages, .key) -- ZIP-based bundles
pub fn extract_iwork(
    file_path: &Path,
    format_name: &str,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert(
        "source_format".to_string(),
        format_name.to_lowercase(),
    );

    let file = std::fs::File::open(file_path)?;
    let archive_result = zip::ZipArchive::new(file);

    let mut archive = match archive_result {
        Ok(a) => a,
        Err(_) => {
            // Some iWork files are package bundles (directories), not ZIP
            return Err(DocumentProcessorError::DocxExtraction(format!(
                "{} format: not a ZIP archive (may be a package bundle)",
                format_name
            )));
        }
    };

    let mut text = String::new();

    // Try QuickLook preview text first (most reliable for iWork)
    if let Ok(mut preview) = archive.by_name("QuickLook/Preview.txt") {
        preview.read_to_string(&mut text)?;
    }

    // Try index.xml or Index/Document.iwa
    if text.is_empty() {
        // Try extracting from any XML files in the archive
        let xml_names: Vec<String> = (0..archive.len())
            .filter_map(|i| {
                archive.by_index(i).ok().and_then(|f| {
                    let name = f.name().to_string();
                    if name.ends_with(".xml") {
                        Some(name)
                    } else {
                        None
                    }
                })
            })
            .collect();

        for name in &xml_names {
            if let Ok(mut f) = archive.by_name(name) {
                let mut content = String::new();
                if f.read_to_string(&mut content).is_ok() {
                    let extracted = extract_text_from_xml_tags(&content, "sf:p");
                    if !extracted.is_empty() {
                        text.push_str(&extracted);
                        text.push('\n');
                    }
                }
            }
        }
    }

    if text.is_empty() {
        return Err(DocumentProcessorError::DocxExtraction(format!(
            "No text content found in {} file. Consider exporting as PDF or DOCX.",
            format_name
        )));
    }

    Ok((clean_extracted_text(&text), metadata))
}

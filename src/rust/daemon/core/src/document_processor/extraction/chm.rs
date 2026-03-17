//! CHM (Microsoft HTML Help) text extraction via chmlib.
//!
//! CHMLib C source is vendored inside the chmlib crate and compiled statically,
//! so the binary remains self-contained (Principle 8).

use std::collections::HashMap;
use std::path::Path;

use chmlib::{ChmFile, Filter};

use super::xml_utils::clean_extracted_text;
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract text from a CHM file by iterating all HTML pages.
pub fn extract_chm(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "chm".to_string());

    let mut chm = ChmFile::open(file_path)
        .map_err(|e| DocumentProcessorError::ChmExtraction(e.to_string()))?;

    let mut html_parts: Vec<String> = Vec::new();
    let mut page_count: usize = 0;

    // Collect HTML page paths first (cannot borrow chm mutably twice at once).
    // Enumeration errors (e.g. malformed CHM index) are non-fatal: we proceed
    // with whatever paths were collected before the error.
    let mut html_paths: Vec<(String, u64)> = Vec::new();
    let _ = chm.for_each(Filter::FILES | Filter::NORMAL, |_chm, unit| {
        if let Some(p) = unit.path() {
            let path_lower = p.to_string_lossy().to_lowercase();
            if path_lower.ends_with(".html") || path_lower.ends_with(".htm") {
                html_paths.push((p.to_string_lossy().to_string(), unit.length()));
            }
        }
        chmlib::Continuation::Continue
    });

    // Read each HTML page
    for (path_str, length) in html_paths {
        if length == 0 {
            continue;
        }
        let path = std::path::Path::new(&path_str);
        if let Some(unit) = chm.find(path) {
            let mut buffer = vec![0u8; length as usize];
            if chm.read(&unit, 0, &mut buffer).is_ok() {
                if let Ok(html) = String::from_utf8(buffer) {
                    let text = html2text::from_read(html.as_bytes(), 80);
                    if !text.trim().is_empty() {
                        html_parts.push(text);
                        page_count += 1;
                    }
                }
            }
        }
    }

    metadata.insert("page_count".to_string(), page_count.to_string());
    let combined = html_parts.join("\n\n");
    Ok((clean_extracted_text(&combined), metadata))
}

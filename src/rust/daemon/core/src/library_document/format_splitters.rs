//! Format-specific document splitters.
//!
//! Contains splitter functions for each supported document format:
//! - Page-based: PDF, PPTX, DOCX, ODP, ODT, ODS, RTF
//! - Stream-based: EPUB, HTML, Markdown, plain text

use std::fs::File;
use std::io::Read;
use std::path::Path;

use tracing::debug;

use crate::parent_unit::{
    UNIT_TYPE_DOCX_SECTION, UNIT_TYPE_EPUB_SECTION, UNIT_TYPE_PDF_PAGE, UNIT_TYPE_TEXT_SECTION,
};

use super::text_processing::{
    clean_text, split_markdown_text, split_text_by_headings, split_text_by_paragraphs,
    strip_rtf_control_codes,
};
use super::xml_helpers::{
    extract_docx_text_from_xml, extract_html_heading, extract_text_from_xml_tags, extract_xml_attr,
};
use super::{LibraryDocumentError, StructuralUnit};

// ─── Page-based splitters ───────────────────────────────────────────────────

/// Split a PDF into per-page structural units.
///
/// Primary: `pdf-extract`, which inserts form feed (`\x0C`) between pages.
/// Fallback: `lopdf` for PDFs with malformed Type 1 font encodings that cause
/// `pdf-extract` to panic (common in old scanned documents). If neither can
/// extract text the function returns an empty vec (image-only PDF, not an error).
pub fn split_pdf(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let path_buf = file_path.to_path_buf();
    let result = std::panic::catch_unwind(|| pdf_extract::extract_text(&path_buf));

    match result {
        Ok(Ok(text)) => split_pdf_text_by_formfeed(&text),
        Ok(Err(_)) | Err(_) => split_pdf_lopdf_fallback(file_path),
    }
}

/// Split pre-extracted PDF text at form feed boundaries (one unit per page).
fn split_pdf_text_by_formfeed(
    full_text: &str,
) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let pages: Vec<&str> = full_text.split('\x0C').collect();

    if pages.len() <= 1 {
        let cleaned = clean_text(full_text);
        if cleaned.is_empty() {
            return Ok(vec![]);
        }
        return Ok(vec![StructuralUnit {
            unit_type: UNIT_TYPE_PDF_PAGE.to_string(),
            unit_locator: serde_json::json!({"page": 1}),
            text: cleaned,
            title: None,
        }]);
    }

    let mut units = Vec::new();
    for (i, page_text) in pages.iter().enumerate() {
        let cleaned = clean_text(page_text);
        if cleaned.is_empty() {
            continue;
        }
        units.push(StructuralUnit {
            unit_type: UNIT_TYPE_PDF_PAGE.to_string(),
            unit_locator: serde_json::json!({"page": i + 1}),
            text: cleaned,
            title: None,
        });
    }

    debug!("PDF split into {} pages", units.len());
    Ok(units)
}

/// Fallback PDF splitter using lopdf for PDFs that pdf-extract cannot handle.
/// Returns an empty vec (not an error) if no text can be extracted.
fn split_pdf_lopdf_fallback(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let doc = match lopdf::Document::load(file_path) {
        Ok(d) => d,
        Err(_) => return Ok(vec![]),
    };

    let mut units = Vec::new();
    for (&page_num, _) in &doc.get_pages() {
        if let Ok(text) = doc.extract_text(&[page_num]) {
            let cleaned = clean_text(&text);
            if !cleaned.is_empty() {
                units.push(StructuralUnit {
                    unit_type: UNIT_TYPE_PDF_PAGE.to_string(),
                    unit_locator: serde_json::json!({"page": page_num}),
                    text: cleaned,
                    title: None,
                });
            }
        }
    }

    debug!("PDF (lopdf fallback) split into {} pages", units.len());
    Ok(units)
}

/// Split a PPTX into per-slide structural units.
pub fn split_pptx(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| LibraryDocumentError::Extraction {
        format: "pptx".into(),
        message: e.to_string(),
    })?;

    // Collect and sort slide file names
    let mut slide_names: Vec<String> = (0..archive.len())
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
    slide_names.sort();

    let mut units = Vec::new();
    for (i, slide_name) in slide_names.iter().enumerate() {
        if let Ok(mut slide_file) = archive.by_name(slide_name) {
            let mut content = String::new();
            slide_file.read_to_string(&mut content)?;
            let slide_text = extract_text_from_xml_tags(&content, "a:t");
            let cleaned = clean_text(&slide_text);
            if cleaned.is_empty() {
                continue;
            }
            units.push(StructuralUnit {
                unit_type: "pptx_slide".to_string(),
                unit_locator: serde_json::json!({"slide": i + 1}),
                text: cleaned,
                title: None,
            });
        }
    }

    debug!("PPTX split into {} slides", units.len());
    Ok(units)
}

/// Split a DOCX into section-level structural units.
///
/// Splits at `w:sectPr` (section properties) boundaries in the XML.
/// If no section boundaries found, treats entire document as one section.
pub fn split_docx(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| LibraryDocumentError::Extraction {
        format: "docx".into(),
        message: e.to_string(),
    })?;

    let mut xml_content = String::new();
    if let Ok(mut doc_file) = archive.by_name("word/document.xml") {
        doc_file.read_to_string(&mut xml_content)?;
    } else {
        return Err(LibraryDocumentError::Extraction {
            format: "docx".into(),
            message: "No word/document.xml found".into(),
        });
    }

    // Split XML by section properties (w:sectPr marks end of a section)
    let sections: Vec<&str> = xml_content.split("w:sectPr").collect();

    let mut units = Vec::new();
    for (i, section_xml) in sections.iter().enumerate() {
        // Last fragment after final sectPr is usually empty/closing tags
        if i == sections.len() - 1 && sections.len() > 1 {
            continue;
        }
        let text = extract_docx_text_from_xml(section_xml);
        let cleaned = clean_text(&text);
        if cleaned.is_empty() {
            continue;
        }
        units.push(StructuralUnit {
            unit_type: UNIT_TYPE_DOCX_SECTION.to_string(),
            unit_locator: serde_json::json!({"section": i + 1}),
            text: cleaned,
            title: None,
        });
    }

    // If no sections found, treat entire document as one section
    if units.is_empty() {
        let full_text = extract_docx_text_from_xml(&xml_content);
        let cleaned = clean_text(&full_text);
        if !cleaned.is_empty() {
            units.push(StructuralUnit {
                unit_type: UNIT_TYPE_DOCX_SECTION.to_string(),
                unit_locator: serde_json::json!({"section": 1}),
                text: cleaned,
                title: None,
            });
        }
    }

    debug!("DOCX split into {} sections", units.len());
    Ok(units)
}

/// Split an ODP (OpenDocument Presentation) into per-slide structural units.
pub fn split_odp(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    split_opendocument_pages(file_path, "odp", "draw:page", "odp_slide", "slide")
}

/// Split an ODT (OpenDocument Text) into section-level structural units.
///
/// Splits by `text:section` elements, or falls back to treating the whole
/// document as one section.
pub fn split_odt(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| LibraryDocumentError::Extraction {
        format: "odt".into(),
        message: e.to_string(),
    })?;

    let mut xml_content = String::new();
    if let Ok(mut content_file) = archive.by_name("content.xml") {
        content_file.read_to_string(&mut xml_content)?;
    } else {
        return Err(LibraryDocumentError::Extraction {
            format: "odt".into(),
            message: "No content.xml found".into(),
        });
    }

    // Try splitting by text:section elements
    let sections: Vec<&str> = xml_content.split("<text:section").collect();

    let mut units = Vec::new();
    if sections.len() > 1 {
        // Skip first element (before the first section)
        for (i, section_xml) in sections.iter().skip(1).enumerate() {
            let text = extract_text_from_xml_tags(section_xml, "text:p");
            let cleaned = clean_text(&text);
            if cleaned.is_empty() {
                continue;
            }
            units.push(StructuralUnit {
                unit_type: "odt_section".to_string(),
                unit_locator: serde_json::json!({"section": i + 1}),
                text: cleaned,
                title: None,
            });
        }
    }

    // If no sections, treat entire document as one section
    if units.is_empty() {
        let text = extract_text_from_xml_tags(&xml_content, "text:p");
        let cleaned = clean_text(&text);
        if !cleaned.is_empty() {
            units.push(StructuralUnit {
                unit_type: "odt_section".to_string(),
                unit_locator: serde_json::json!({"section": 1}),
                text: cleaned,
                title: None,
            });
        }
    }

    debug!("ODT split into {} sections", units.len());
    Ok(units)
}

/// Split an ODS (OpenDocument Spreadsheet) into per-sheet structural units.
pub fn split_ods(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| LibraryDocumentError::Extraction {
        format: "ods".into(),
        message: e.to_string(),
    })?;

    let mut xml_content = String::new();
    if let Ok(mut content_file) = archive.by_name("content.xml") {
        content_file.read_to_string(&mut xml_content)?;
    } else {
        return Err(LibraryDocumentError::Extraction {
            format: "ods".into(),
            message: "No content.xml found".into(),
        });
    }

    // Split by table:table elements
    let tables: Vec<&str> = xml_content.split("<table:table ").collect();

    let mut units = Vec::new();
    for (i, table_xml) in tables.iter().skip(1).enumerate() {
        // Extract sheet name from table:name attribute
        let sheet_name =
            extract_xml_attr(table_xml, "table:name").unwrap_or_else(|| format!("Sheet{}", i + 1));

        // Extract text from table cells
        let text = extract_text_from_xml_tags(table_xml, "text:p");
        let cleaned = clean_text(&text);
        if cleaned.is_empty() {
            continue;
        }
        units.push(StructuralUnit {
            unit_type: "ods_sheet".to_string(),
            unit_locator: serde_json::json!({"sheet": sheet_name}),
            text: cleaned,
            title: Some(sheet_name),
        });
    }

    debug!("ODS split into {} sheets", units.len());
    Ok(units)
}

/// Split an RTF file into a single structural unit.
///
/// RTF has no page/section structure accessible without a full layout engine,
/// so the entire content is treated as a single section.
pub fn split_rtf(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let mut file = File::open(file_path)?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)?;

    let text = strip_rtf_control_codes(&raw);
    let cleaned = clean_text(&text);
    if cleaned.is_empty() {
        return Ok(vec![]);
    }

    Ok(vec![StructuralUnit {
        unit_type: UNIT_TYPE_TEXT_SECTION.to_string(),
        unit_locator: serde_json::json!({"section": 1}),
        text: cleaned,
        title: None,
    }])
}

// ─── Stream-based splitters ─────────────────────────────────────────────────

/// Split an EPUB into per-chapter structural units.
pub fn split_epub(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let mut doc =
        epub::doc::EpubDoc::new(file_path).map_err(|e| LibraryDocumentError::Extraction {
            format: "epub".into(),
            message: e.to_string(),
        })?;

    let mut units = Vec::new();
    let mut spine_index = 0usize;

    loop {
        let spine_id = doc
            .get_current_id()
            .unwrap_or_else(|| format!("spine_{}", spine_index));

        if let Some((content, _mime)) = doc.get_current_str() {
            let text = html2text::from_read(content.as_bytes(), 80);
            let cleaned = clean_text(&text);

            if !cleaned.is_empty() {
                // Try to extract chapter title from HTML content
                let chapter_title = extract_html_heading(&content);

                let mut locator = serde_json::json!({"spine_id": spine_id});
                if let Some(ref title) = chapter_title {
                    locator["chapter_title"] = serde_json::Value::String(title.clone());
                }

                units.push(StructuralUnit {
                    unit_type: UNIT_TYPE_EPUB_SECTION.to_string(),
                    unit_locator: locator,
                    text: cleaned,
                    title: chapter_title,
                });
            }
        }

        spine_index += 1;
        if !doc.go_next() {
            break;
        }
    }

    debug!("EPUB split into {} chapters", units.len());
    Ok(units)
}

/// Split an HTML document into heading-based structural units.
pub fn split_html(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let mut file = File::open(file_path)?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)?;

    // Convert HTML to plain text first
    let full_text = html2text::from_read(raw.as_bytes(), 120);

    // Split by heading-like patterns in the text output
    split_text_by_headings(&full_text, "html_section")
}

/// Split a Markdown document into heading-based structural units.
pub fn split_markdown(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let mut file = File::open(file_path)?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)?;

    split_markdown_text(&raw)
}

/// Split a plain text file into paragraph-group structural units.
///
/// Groups paragraphs (separated by double newlines) into sections of
/// reasonable size. Very short documents become a single section.
pub fn split_plain_text(file_path: &Path) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let mut file = File::open(file_path)?;
    let mut raw = String::new();
    file.read_to_string(&mut raw)?;

    split_text_by_paragraphs(&raw)
}

// ─── Internal helpers ───────────────────────────────────────────────────────

/// Helper for splitting OpenDocument presentations/drawings by page elements.
fn split_opendocument_pages(
    file_path: &Path,
    format_name: &str,
    page_tag: &str,
    unit_type: &str,
    locator_key: &str,
) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let file = File::open(file_path)?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| LibraryDocumentError::Extraction {
        format: format_name.into(),
        message: e.to_string(),
    })?;

    let mut xml_content = String::new();
    if let Ok(mut content_file) = archive.by_name("content.xml") {
        content_file.read_to_string(&mut xml_content)?;
    } else {
        return Err(LibraryDocumentError::Extraction {
            format: format_name.into(),
            message: "No content.xml found".into(),
        });
    }

    let open_tag = format!("<{} ", page_tag);
    let pages: Vec<&str> = xml_content.split(&open_tag).collect();

    let mut units = Vec::new();
    for (i, page_xml) in pages.iter().skip(1).enumerate() {
        // Extract page name attribute
        let page_name = extract_xml_attr(page_xml, "draw:name")
            .unwrap_or_else(|| format!("{}{}", locator_key, i + 1));

        let text = extract_text_from_xml_tags(page_xml, "text:p");
        let cleaned = clean_text(&text);
        if cleaned.is_empty() {
            continue;
        }
        units.push(StructuralUnit {
            unit_type: unit_type.to_string(),
            unit_locator: serde_json::json!({locator_key: i + 1}),
            text: cleaned,
            title: Some(page_name),
        });
    }

    debug!(
        "{} split into {} pages",
        format_name.to_uppercase(),
        units.len()
    );
    Ok(units)
}

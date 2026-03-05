//! Format-specific metadata title extraction.
//!
//! Handles extraction of document titles from embedded metadata in
//! PDF, DOCX, PPTX, ODT, and RTF files.

use regex::Regex;
use std::path::Path;

/// Extract metadata title directly from file (format-specific).
pub fn extract_metadata_title_from_file(file_path: &Path, source_format: &str) -> Option<String> {
    match source_format {
        "pdf" => extract_pdf_metadata_title(file_path),
        "docx" => extract_office_xml_title(file_path, "docProps/core.xml"),
        "pptx" => extract_office_xml_title(file_path, "docProps/core.xml"),
        "odt" | "odp" | "ods" => extract_opendocument_title(file_path),
        "rtf" => extract_rtf_title(file_path),
        "html" | "htm" => None, // HTML title extracted from content
        "epub" => None,         // EPUB metadata already extracted by epub crate
        _ => None,
    }
}

/// Extract title from PDF Info dictionary using lopdf.
fn extract_pdf_metadata_title(file_path: &Path) -> Option<String> {
    let doc = match lopdf::Document::load(file_path) {
        Ok(d) => d,
        Err(_) => return None,
    };

    // Try to get trailer -> Info dict
    let info_ref = doc.trailer.get(b"Info").ok()?;
    let info_ref = match info_ref {
        lopdf::Object::Reference(r) => *r,
        _ => return None,
    };

    let info_dict = match doc.get_object(info_ref) {
        Ok(lopdf::Object::Dictionary(d)) => d,
        _ => return None,
    };

    // Extract /Title
    let title_obj = info_dict.get(b"Title").ok()?;
    let title = match title_obj {
        lopdf::Object::String(bytes, _) => String::from_utf8_lossy(bytes).trim().to_string(),
        _ => return None,
    };

    if title.is_empty() {
        None
    } else {
        Some(title)
    }
}

/// Extract dc:title from Office XML (DOCX/PPTX) docProps/core.xml.
fn extract_office_xml_title(file_path: &Path, core_xml_path: &str) -> Option<String> {
    let file = std::fs::File::open(file_path).ok()?;
    let mut archive = zip::ZipArchive::new(file).ok()?;

    let mut core_xml = archive.by_name(core_xml_path).ok()?;
    let mut content = String::new();
    std::io::Read::read_to_string(&mut core_xml, &mut content).ok()?;

    // Simple XML extraction for dc:title
    extract_xml_element_text(&content, "dc:title")
        .or_else(|| extract_xml_element_text(&content, "cp:title"))
}

/// Extract title from OpenDocument meta.xml.
fn extract_opendocument_title(file_path: &Path) -> Option<String> {
    let file = std::fs::File::open(file_path).ok()?;
    let mut archive = zip::ZipArchive::new(file).ok()?;

    let mut meta_xml = archive.by_name("meta.xml").ok()?;
    let mut content = String::new();
    std::io::Read::read_to_string(&mut meta_xml, &mut content).ok()?;

    extract_xml_element_text(&content, "dc:title")
}

/// Extract title from RTF {\info{\title ...}} block.
fn extract_rtf_title(file_path: &Path) -> Option<String> {
    let bytes = std::fs::read(file_path).ok()?;
    let content = String::from_utf8_lossy(&bytes);

    // Look for {\info{...{\title ...}...}}
    let re = Regex::new(r"(?s)\{\\title\s+([^}]+)\}").ok()?;
    if let Some(caps) = re.captures(&content) {
        let title = caps[1].trim().to_string();
        if !title.is_empty() {
            return Some(title);
        }
    }
    None
}

/// Extract text content of an XML element by tag name.
/// Simple regex-based extraction (avoids full XML parser dependency).
pub fn extract_xml_element_text(xml: &str, tag: &str) -> Option<String> {
    let escaped_tag = regex::escape(tag);
    let pattern = format!(r"<{0}[^>]*>(.*?)</{0}>", escaped_tag);
    let re = Regex::new(&pattern).ok()?;
    if let Some(caps) = re.captures(xml) {
        let text = caps[1].trim().to_string();
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_xml_element() {
        let xml = r#"<dc:title>My Document Title</dc:title>"#;
        assert_eq!(
            extract_xml_element_text(xml, "dc:title"),
            Some("My Document Title".to_string())
        );
    }

    #[test]
    fn test_extract_xml_element_empty() {
        let xml = r#"<dc:title></dc:title>"#;
        assert_eq!(extract_xml_element_text(xml, "dc:title"), None);
    }
}

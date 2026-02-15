//! Document title extraction with priority cascade.
//!
//! Extracts titles from documents using a three-level priority cascade:
//! 1. **Embedded metadata** (format-specific: PDF Info dict, DOCX core.xml, EPUB OPF, etc.)
//! 2. **Content heuristics** (first heading, first prominent line)
//! 3. **Filename fallback** (cleaned, title-cased filename)
//!
//! Supports both page-based documents (PDF, DOCX, PPTX, ODT, etc.) and
//! stream-based documents (EPUB, HTML, Markdown, plain text).

use std::path::Path;
use regex::Regex;
use tracing::debug;

/// Result of title extraction, including source attribution.
#[derive(Debug, Clone)]
pub struct TitleResult {
    /// The extracted title
    pub title: String,
    /// How the title was obtained
    pub source: TitleSource,
    /// Extracted authors, if available
    pub authors: Vec<String>,
}

/// How the title was obtained
#[derive(Debug, Clone, PartialEq)]
pub enum TitleSource {
    /// Extracted from embedded document metadata
    Metadata,
    /// Extracted from content heuristics (first heading, etc.)
    ContentHeuristic,
    /// Derived from filename
    FilenameFallback,
}

/// Extract title from a document using the priority cascade.
///
/// Takes pre-extracted metadata (from document_processor) and raw text.
/// Returns the best available title.
pub fn extract_title(
    file_path: &Path,
    metadata: &std::collections::HashMap<String, String>,
    raw_text: &str,
    source_format: &str,
) -> TitleResult {
    let mut authors = Vec::new();

    // Extract authors from metadata if present
    if let Some(author) = metadata.get("author") {
        if !author.is_empty() {
            authors = parse_authors(author);
        }
    }

    // Priority 1: Embedded metadata title
    if let Some(title) = metadata.get("title").or_else(|| metadata.get("doc_title")) {
        let title = title.trim().to_string();
        if !title.is_empty() && !is_placeholder_title(&title) {
            debug!("Title from metadata: {:?}", title);
            return TitleResult { title, source: TitleSource::Metadata, authors };
        }
    }

    // Priority 1b: Format-specific metadata extraction from file
    if let Some(title) = extract_metadata_title_from_file(file_path, source_format) {
        if !is_placeholder_title(&title) {
            debug!("Title from file metadata: {:?}", title);
            return TitleResult { title, source: TitleSource::Metadata, authors };
        }
    }

    // Priority 2: Content heuristics
    if let Some(title) = extract_title_from_content(raw_text, source_format) {
        debug!("Title from content heuristic: {:?}", title);
        return TitleResult { title, source: TitleSource::ContentHeuristic, authors };
    }

    // Priority 3: Filename fallback
    let title = title_from_filename(file_path);
    debug!("Title from filename fallback: {:?}", title);
    TitleResult { title, source: TitleSource::FilenameFallback, authors }
}

/// Check if a title is a known placeholder pattern.
pub fn is_placeholder_title(title: &str) -> bool {
    let title_lower = title.to_lowercase();

    // Common auto-generated titles
    let placeholder_patterns = [
        "untitled", "document", "presentation", "slide",
        "book", "new document", "noname",
    ];
    for pattern in &placeholder_patterns {
        if title_lower == *pattern {
            return true;
        }
    }

    // Numbered placeholders: "Document1", "Presentation2", "Slide 3"
    let re = Regex::new(r"(?i)^(microsoft\s+word\s*[-–—]\s*|document|presentation|slide|book|untitled)\s*\d*$").unwrap();
    if re.is_match(title.trim()) {
        return true;
    }

    // Microsoft Word header pattern: "Microsoft Word - filename.docx"
    if title_lower.starts_with("microsoft word") {
        return true;
    }

    false
}

/// Extract metadata title directly from file (format-specific).
fn extract_metadata_title_from_file(file_path: &Path, source_format: &str) -> Option<String> {
    match source_format {
        "pdf" => extract_pdf_metadata_title(file_path),
        "docx" => extract_office_xml_title(file_path, "docProps/core.xml"),
        "pptx" => extract_office_xml_title(file_path, "docProps/core.xml"),
        "odt" | "odp" | "ods" => extract_opendocument_title(file_path),
        "rtf" => extract_rtf_title(file_path),
        "html" | "htm" => None, // HTML title extracted from content
        "epub" => None, // EPUB metadata already extracted by epub crate
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
        lopdf::Object::String(bytes, _) => {
            String::from_utf8_lossy(bytes).trim().to_string()
        }
        _ => return None,
    };

    if title.is_empty() { None } else { Some(title) }
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
fn extract_xml_element_text(xml: &str, tag: &str) -> Option<String> {
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

/// Extract title from document content using heuristics.
fn extract_title_from_content(text: &str, source_format: &str) -> Option<String> {
    if text.trim().is_empty() {
        return None;
    }

    match source_format {
        "html" | "htm" => extract_html_title(text),
        "markdown" | "md" => extract_markdown_title(text),
        _ => extract_first_line_title(text),
    }
}

/// Extract title from HTML content.
fn extract_html_title(text: &str) -> Option<String> {
    // Try <title> tag
    if let Some(title) = extract_xml_element_text(text, "title") {
        if !is_placeholder_title(&title) {
            return Some(title);
        }
    }

    // Try og:title meta tag
    let re = Regex::new(r#"<meta\s+(?:property|name)=["']og:title["']\s+content=["']([^"']+)["']"#).ok()?;
    if let Some(caps) = re.captures(text) {
        let title = caps[1].trim().to_string();
        if !title.is_empty() && !is_placeholder_title(&title) {
            return Some(title);
        }
    }

    // Try first <h1>
    if let Some(title) = extract_xml_element_text(text, "h1") {
        // Strip any inner HTML tags
        let re = Regex::new(r"<[^>]+>").unwrap();
        let clean = re.replace_all(&title, "").trim().to_string();
        if !clean.is_empty() {
            return Some(clean);
        }
    }

    None
}

/// Extract title from Markdown content.
fn extract_markdown_title(text: &str) -> Option<String> {
    // Try YAML frontmatter
    if text.starts_with("---") {
        if let Some(end) = text[3..].find("---") {
            let frontmatter = &text[3..3 + end];
            // Simple key: value extraction for title
            for line in frontmatter.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("title:") {
                    let title = rest.trim().trim_matches('"').trim_matches('\'').to_string();
                    if !title.is_empty() {
                        return Some(title);
                    }
                }
            }
        }
    }

    // Try first # heading
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(heading) = trimmed.strip_prefix("# ") {
            let title = heading.trim().to_string();
            if !title.is_empty() {
                return Some(title);
            }
        }
        // Skip empty lines and frontmatter delimiters
        if trimmed.is_empty() || trimmed == "---" {
            continue;
        }
        // Stop looking after first non-empty, non-heading line
        if !trimmed.starts_with('#') && !trimmed.starts_with("---") {
            break;
        }
    }

    None
}

/// Extract title from first line of text (for plain text and fallback).
fn extract_first_line_title(text: &str) -> Option<String> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // A title-like first line: not too long, no trailing period, not all lowercase
        if trimmed.len() <= 200
            && !trimmed.ends_with('.')
            && !trimmed.ends_with(',')
            && !trimmed.ends_with(';')
            && trimmed.chars().any(|c| c.is_uppercase())
        {
            return Some(trimmed.to_string());
        }
        // First non-empty line doesn't look like a title
        return None;
    }
    None
}

/// Generate a title from a filename.
pub fn title_from_filename(file_path: &Path) -> String {
    let stem = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Untitled");

    // Replace underscores and hyphens with spaces
    let cleaned = stem
        .replace('_', " ")
        .replace('-', " ");

    // Simple title case: capitalize first letter of each word
    cleaned
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    let upper = first.to_uppercase().to_string();
                    upper + &chars.as_str().to_string()
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Parse author string into a list of authors.
/// Handles comma-separated, semicolon-separated, and "and"-separated lists.
fn parse_authors(author_str: &str) -> Vec<String> {
    if author_str.is_empty() {
        return Vec::new();
    }

    // Split on semicolons first, then commas, then " and "
    let authors: Vec<String> = if author_str.contains(';') {
        author_str.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    } else if author_str.contains(',') {
        author_str.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    } else if author_str.contains(" and ") {
        author_str.split(" and ").map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    } else {
        vec![author_str.trim().to_string()]
    };

    authors
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ===== Placeholder detection =====

    #[test]
    fn test_placeholder_untitled() {
        assert!(is_placeholder_title("Untitled"));
        assert!(is_placeholder_title("untitled"));
        assert!(is_placeholder_title("UNTITLED"));
    }

    #[test]
    fn test_placeholder_numbered() {
        assert!(is_placeholder_title("Document1"));
        assert!(is_placeholder_title("Presentation2"));
        assert!(is_placeholder_title("Slide 3"));
    }

    #[test]
    fn test_placeholder_microsoft_word() {
        assert!(is_placeholder_title("Microsoft Word - Document1.docx"));
        assert!(is_placeholder_title("Microsoft Word - report.docx"));
    }

    #[test]
    fn test_non_placeholder_titles() {
        assert!(!is_placeholder_title("Annual Report 2024"));
        assert!(!is_placeholder_title("My Presentation"));
        assert!(!is_placeholder_title("Introduction to Rust"));
    }

    // ===== Filename fallback =====

    #[test]
    fn test_title_from_filename_underscores() {
        let path = Path::new("/docs/2024_annual_report.pdf");
        assert_eq!(title_from_filename(path), "2024 Annual Report");
    }

    #[test]
    fn test_title_from_filename_hyphens() {
        let path = Path::new("/docs/project-proposal-v2.docx");
        assert_eq!(title_from_filename(path), "Project Proposal V2");
    }

    #[test]
    fn test_title_from_filename_mixed() {
        let path = Path::new("Q4_results-final.pptx");
        assert_eq!(title_from_filename(path), "Q4 Results Final");
    }

    #[test]
    fn test_title_from_filename_simple() {
        let path = Path::new("README.md");
        assert_eq!(title_from_filename(path), "README");
    }

    // ===== Markdown title extraction =====

    #[test]
    fn test_markdown_frontmatter_title() {
        let text = "---\ntitle: My Document\nauthor: John\n---\n\n# Content\nSome text.";
        let result = extract_markdown_title(text);
        assert_eq!(result, Some("My Document".to_string()));
    }

    #[test]
    fn test_markdown_frontmatter_quoted_title() {
        let text = "---\ntitle: \"Quoted Title\"\n---\n\n# Content";
        let result = extract_markdown_title(text);
        assert_eq!(result, Some("Quoted Title".to_string()));
    }

    #[test]
    fn test_markdown_heading_title() {
        let text = "# My Heading\n\nSome content here.";
        let result = extract_markdown_title(text);
        assert_eq!(result, Some("My Heading".to_string()));
    }

    #[test]
    fn test_markdown_no_title() {
        let text = "Just some plain text without any headings or frontmatter.";
        let result = extract_markdown_title(text);
        assert_eq!(result, None);
    }

    // ===== HTML title extraction =====

    #[test]
    fn test_html_title_tag() {
        let text = "<html><head><title>My Page</title></head><body><h1>Content</h1></body></html>";
        let result = extract_html_title(text);
        assert_eq!(result, Some("My Page".to_string()));
    }

    #[test]
    fn test_html_h1_fallback() {
        let text = "<html><head></head><body><h1>First Heading</h1><p>Content</p></body></html>";
        let result = extract_html_title(text);
        assert_eq!(result, Some("First Heading".to_string()));
    }

    #[test]
    fn test_html_og_title() {
        let text = r#"<html><head><meta property="og:title" content="Open Graph Title"></head></html>"#;
        let result = extract_html_title(text);
        assert_eq!(result, Some("Open Graph Title".to_string()));
    }

    // ===== First-line heuristic =====

    #[test]
    fn test_first_line_title_valid() {
        let text = "Introduction to Machine Learning\n\nThis chapter covers...";
        let result = extract_first_line_title(text);
        assert_eq!(result, Some("Introduction to Machine Learning".to_string()));
    }

    #[test]
    fn test_first_line_title_too_long() {
        let long_line = "a".repeat(250);
        let text = format!("{}\n\nMore content.", long_line);
        let result = extract_first_line_title(&text);
        assert_eq!(result, None);
    }

    #[test]
    fn test_first_line_ends_with_period() {
        let text = "This is a sentence.\n\nMore content.";
        let result = extract_first_line_title(text);
        assert_eq!(result, None);
    }

    // ===== Author parsing =====

    #[test]
    fn test_parse_single_author() {
        assert_eq!(parse_authors("John Doe"), vec!["John Doe"]);
    }

    #[test]
    fn test_parse_comma_authors() {
        assert_eq!(
            parse_authors("John Doe, Jane Smith"),
            vec!["John Doe", "Jane Smith"]
        );
    }

    #[test]
    fn test_parse_semicolon_authors() {
        assert_eq!(
            parse_authors("John Doe; Jane Smith; Bob"),
            vec!["John Doe", "Jane Smith", "Bob"]
        );
    }

    #[test]
    fn test_parse_and_authors() {
        assert_eq!(
            parse_authors("John Doe and Jane Smith"),
            vec!["John Doe", "Jane Smith"]
        );
    }

    // ===== XML element extraction =====

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

    // ===== Full cascade =====

    #[test]
    fn test_cascade_metadata_wins() {
        let mut metadata = HashMap::new();
        metadata.insert("title".to_string(), "Metadata Title".to_string());
        let result = extract_title(
            Path::new("fallback.pdf"),
            &metadata,
            "# Content Heading",
            "pdf",
        );
        assert_eq!(result.title, "Metadata Title");
        assert_eq!(result.source, TitleSource::Metadata);
    }

    #[test]
    fn test_cascade_content_when_no_metadata() {
        let metadata = HashMap::new();
        let result = extract_title(
            Path::new("fallback.md"),
            &metadata,
            "# My Document\n\nSome content.",
            "markdown",
        );
        assert_eq!(result.title, "My Document");
        assert_eq!(result.source, TitleSource::ContentHeuristic);
    }

    #[test]
    fn test_cascade_filename_fallback() {
        let metadata = HashMap::new();
        let result = extract_title(
            Path::new("my_great_doc.pdf"),
            &metadata,
            "just some content with no clear title.",
            "pdf",
        );
        assert_eq!(result.title, "My Great Doc");
        assert_eq!(result.source, TitleSource::FilenameFallback);
    }

    #[test]
    fn test_cascade_skips_placeholder_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("title".to_string(), "Document1".to_string());
        let result = extract_title(
            Path::new("real_name.pdf"),
            &metadata,
            "no clear heading either.",
            "pdf",
        );
        // Should skip placeholder and use filename
        assert_eq!(result.title, "Real Name");
        assert_eq!(result.source, TitleSource::FilenameFallback);
    }

    #[test]
    fn test_cascade_with_authors() {
        let mut metadata = HashMap::new();
        metadata.insert("title".to_string(), "A Paper".to_string());
        metadata.insert("author".to_string(), "Alice; Bob".to_string());
        let result = extract_title(
            Path::new("paper.pdf"),
            &metadata,
            "",
            "pdf",
        );
        assert_eq!(result.title, "A Paper");
        assert_eq!(result.authors, vec!["Alice", "Bob"]);
    }
}

//! Document title extraction with priority cascade.
//!
//! Extracts titles from documents using a three-level priority cascade:
//! 1. **Embedded metadata** (format-specific: PDF Info dict, DOCX core.xml, EPUB OPF, etc.)
//! 2. **Content heuristics** (first heading, first prominent line)
//! 3. **Filename fallback** (cleaned, title-cased filename)
//!
//! Supports both page-based documents (PDF, DOCX, PPTX, ODT, etc.) and
//! stream-based documents (EPUB, HTML, Markdown, plain text).

mod content;
mod metadata;
mod types;

pub use types::{TitleResult, TitleSource};

use std::path::Path;
use regex::Regex;
use tracing::debug;

use content::extract_title_from_content;
use metadata::extract_metadata_title_from_file;

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
    if author_str.contains(';') {
        author_str.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    } else if author_str.contains(',') {
        author_str.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    } else if author_str.contains(" and ") {
        author_str.split(" and ").map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    } else {
        vec![author_str.trim().to_string()]
    }
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

//! Content heuristic title extraction.
//!
//! Extracts titles from document content using format-specific heuristics:
//! HTML `<title>` / `<h1>` tags, Markdown YAML frontmatter / `#` headings,
//! and first-line detection for plain text.

use regex::Regex;
use super::metadata::extract_xml_element_text;
use super::is_placeholder_title;

/// Extract title from document content using heuristics.
pub fn extract_title_from_content(text: &str, source_format: &str) -> Option<String> {
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
pub fn extract_html_title(text: &str) -> Option<String> {
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
pub fn extract_markdown_title(text: &str) -> Option<String> {
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
pub fn extract_first_line_title(text: &str) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}

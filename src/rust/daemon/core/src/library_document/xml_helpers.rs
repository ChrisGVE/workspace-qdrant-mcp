//! XML extraction helpers for library document splitting.
//!
//! Contains functions for extracting text from DOCX, ODP, ODS, and
//! other XML-based document formats, plus HTML heading extraction.

/// Extract text from DOCX XML paragraphs (w:t tags within w:p elements).
pub(super) fn extract_docx_text_from_xml(xml: &str) -> String {
    let mut text = String::new();
    let mut in_text_tag = false;
    let mut current_text = String::new();

    for part in xml.split('<') {
        if part.is_empty() {
            continue;
        }
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

    text
}

/// Extract text content from XML tags (same algorithm as document_processor).
pub(super) fn extract_text_from_xml_tags(xml_content: &str, tag_name: &str) -> String {
    let mut text = String::new();
    let mut in_tag = false;
    let mut depth = 0i32;

    for part in xml_content.split('<') {
        if part.is_empty() {
            continue;
        }

        let close_prefix = format!("/{}", tag_name);

        if part.starts_with(tag_name) {
            in_tag = true;
            depth += 1;
            if let Some(content_start) = part.find('>') {
                let content = &part[content_start + 1..];
                if !content.is_empty() {
                    text.push_str(content);
                }
            }
        } else if part.starts_with(&close_prefix) {
            depth -= 1;
            if depth <= 0 {
                in_tag = false;
                depth = 0;
                text.push('\n');
            }
        } else if in_tag {
            if let Some(pos) = part.find('>') {
                let content = &part[pos + 1..];
                if !content.is_empty() {
                    text.push_str(content);
                }
            }
        }
    }

    text
}

/// Extract the first XML attribute value from a tag fragment.
pub(super) fn extract_xml_attr(xml_fragment: &str, attr_name: &str) -> Option<String> {
    let search = format!("{}=\"", attr_name);
    if let Some(start) = xml_fragment.find(&search) {
        let value_start = start + search.len();
        if let Some(end) = xml_fragment[value_start..].find('"') {
            return Some(xml_fragment[value_start..value_start + end].to_string());
        }
    }
    None
}

/// Extract first heading (h1-h3) from HTML content.
pub(super) fn extract_html_heading(html: &str) -> Option<String> {
    for tag in &["h1", "h2", "h3"] {
        let open = format!("<{}", tag);
        if let Some(start) = html.to_lowercase().find(&open) {
            let rest = &html[start..];
            if let Some(gt) = rest.find('>') {
                let after_tag = &rest[gt + 1..];
                if let Some(close) = after_tag.find('<') {
                    let heading_text = after_tag[..close].trim();
                    if !heading_text.is_empty() {
                        return Some(heading_text.to_string());
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_xml_attr() {
        assert_eq!(
            extract_xml_attr(r#"table:name="Sheet1" table:style-name="ta1""#, "table:name"),
            Some("Sheet1".to_string())
        );
        assert_eq!(
            extract_xml_attr(r#"draw:name="Slide 1""#, "draw:name"),
            Some("Slide 1".to_string())
        );
        assert_eq!(extract_xml_attr("no-attr-here", "table:name"), None);
    }

    #[test]
    fn test_extract_text_from_xml_tags() {
        let xml = r#"<text:p text:style-name="P1">Hello world</text:p>"#;
        let text = extract_text_from_xml_tags(xml, "text:p");
        assert!(text.contains("Hello world"));
    }

    #[test]
    fn test_extract_docx_text_from_xml() {
        let xml = r#"<w:p><w:r><w:t>Hello</w:t></w:r></w:p><w:p><w:r><w:t>World</w:t></w:r></w:p>"#;
        let text = extract_docx_text_from_xml(xml);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[test]
    fn test_extract_html_heading() {
        assert_eq!(
            extract_html_heading("<h1>Title Here</h1><p>Content</p>"),
            Some("Title Here".to_string())
        );
        assert_eq!(extract_html_heading("<p>No heading</p>"), None);
        assert_eq!(
            extract_html_heading("<H2>Mixed Case</H2>"),
            Some("Mixed Case".to_string())
        );
    }
}

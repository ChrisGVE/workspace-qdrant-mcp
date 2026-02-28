//! XML text extraction and text cleaning utilities.

/// Generic XML tag text extractor -- extracts text content from all matching tags
pub fn extract_text_from_xml_tags(xml_content: &str, tag_name: &str) -> String {
    let mut text = String::new();
    let open_tag = format!("<{}", tag_name);
    let close_tag = format!("</{}", tag_name);

    let mut in_tag = false;
    let mut depth = 0i32;

    for part in xml_content.split('<') {
        if part.is_empty() {
            continue;
        }

        if part.starts_with(&tag_name[..]) || part.starts_with(&open_tag[1..]) {
            in_tag = true;
            depth += 1;
            if let Some(content_start) = part.find('>') {
                let content = &part[content_start + 1..];
                if !content.is_empty() {
                    text.push_str(content);
                }
            }
        } else if part.starts_with(&close_tag[1..]) {
            depth -= 1;
            if depth <= 0 {
                in_tag = false;
                depth = 0;
                text.push('\n');
            }
        } else if in_tag {
            // Nested tags inside -- extract text after '>'
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

/// Clean up extracted text (normalize whitespace, remove control chars)
pub fn clean_extracted_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_whitespace = false;

    for ch in text.chars() {
        if ch.is_control() && ch != '\n' && ch != '\t' {
            continue;
        }

        if ch.is_whitespace() {
            if !prev_was_whitespace || ch == '\n' {
                result.push(if ch == '\n' { '\n' } else { ' ' });
            }
            prev_was_whitespace = true;
        } else {
            result.push(ch);
            prev_was_whitespace = false;
        }
    }

    result.trim().to_string()
}

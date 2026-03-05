//! Text processing utilities for library document splitting.
//!
//! Contains text cleaning, markdown parsing, paragraph grouping,
//! and heading detection functions.

use tracing::debug;

use crate::parent_unit::UNIT_TYPE_TEXT_SECTION;

use super::{LibraryDocumentError, StructuralUnit};

/// Clean extracted text: normalize whitespace, remove control characters.
pub(super) fn clean_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_whitespace = false;

    for ch in text.chars() {
        if ch == '\x0C' {
            continue; // Skip form feeds
        }
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

/// Parse an ATX-style markdown heading (# through ######).
pub(super) struct HeadingInfo {
    pub level: usize,
    pub title: String,
}

pub(super) fn parse_atx_heading(line: &str) -> Option<HeadingInfo> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }

    let level = trimmed.chars().take_while(|&c| c == '#').count();
    if level == 0 || level > 6 {
        return None;
    }

    let rest = &trimmed[level..];
    if !rest.is_empty() && !rest.starts_with(' ') && !rest.starts_with('\t') {
        return None; // No space after # — not a heading (e.g., #hashtag)
    }

    let title = rest.trim().trim_end_matches('#').trim().to_string();
    if title.is_empty() {
        return None;
    }

    Some(HeadingInfo { level, title })
}

/// Split markdown text into heading-based sections.
///
/// Recognizes ATX-style headings (# through ######).
/// YAML frontmatter is skipped.
pub(super) fn split_markdown_text(text: &str) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let mut units = Vec::new();
    let mut current_title: Option<String> = None;
    let mut current_level = 0usize;
    let mut current_text = String::new();
    let mut section_index = 0usize;
    let mut in_frontmatter = false;
    for (line_idx, line) in text.lines().enumerate() {
        // Handle YAML frontmatter
        if line_idx == 0 && line.trim() == "---" {
            in_frontmatter = true;
            continue;
        }
        if in_frontmatter {
            if line.trim() == "---" || line.trim() == "..." {
                in_frontmatter = false;
            }
            continue;
        }

        // Check for ATX headings
        if let Some(heading) = parse_atx_heading(line) {
            // Flush previous section
            if !current_text.is_empty() || current_title.is_some() {
                let cleaned = clean_text(&current_text);
                if !cleaned.is_empty() {
                    let mut locator = serde_json::json!({
                        "section_index": section_index,
                        "heading_level": current_level,
                    });
                    if let Some(ref title) = current_title {
                        locator["title"] = serde_json::Value::String(title.clone());
                    }
                    units.push(StructuralUnit {
                        unit_type: "markdown_section".to_string(),
                        unit_locator: locator,
                        text: cleaned,
                        title: current_title.take(),
                    });
                    section_index += 1;
                }
            }
            current_title = Some(heading.title);
            current_level = heading.level;
            current_text.clear();
        } else {
            if !current_text.is_empty() || !line.trim().is_empty() {
                if !current_text.is_empty() {
                    current_text.push('\n');
                }
                current_text.push_str(line);
            }
        }
    }

    // Flush final section
    let cleaned = clean_text(&current_text);
    if !cleaned.is_empty() || current_title.is_some() {
        let actual_text = if cleaned.is_empty() {
            current_title.clone().unwrap_or_default()
        } else {
            cleaned
        };
        if !actual_text.is_empty() {
            // If we never encountered any headings, use text_section type
            let unit_type = if section_index == 0 && current_title.is_none() {
                UNIT_TYPE_TEXT_SECTION.to_string()
            } else {
                "markdown_section".to_string()
            };
            let mut locator = serde_json::json!({
                "section_index": section_index,
                "heading_level": current_level,
            });
            if let Some(ref title) = current_title {
                locator["title"] = serde_json::Value::String(title.clone());
            }
            units.push(StructuralUnit {
                unit_type,
                unit_locator: locator,
                text: actual_text,
                title: current_title,
            });
        }
    }

    debug!("Markdown split into {} sections", units.len());
    Ok(units)
}

/// Split text into paragraph-group sections.
///
/// Groups consecutive paragraphs into sections. A new section starts
/// when accumulated text exceeds ~2000 characters.
pub(super) fn split_text_by_paragraphs(
    text: &str,
) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();

    if paragraphs.len() <= 1 {
        let cleaned = clean_text(text);
        if cleaned.is_empty() {
            return Ok(vec![]);
        }
        return Ok(vec![StructuralUnit {
            unit_type: UNIT_TYPE_TEXT_SECTION.to_string(),
            unit_locator: serde_json::json!({"section_index": 0}),
            text: cleaned,
            title: None,
        }]);
    }

    let mut units = Vec::new();
    let mut current = String::new();
    let mut section_index = 0usize;
    let section_target_chars = 2000;

    for para in &paragraphs {
        let trimmed = para.trim();
        if trimmed.is_empty() {
            continue;
        }

        if !current.is_empty() && current.len() + trimmed.len() > section_target_chars {
            let cleaned = clean_text(&current);
            if !cleaned.is_empty() {
                units.push(StructuralUnit {
                    unit_type: UNIT_TYPE_TEXT_SECTION.to_string(),
                    unit_locator: serde_json::json!({"section_index": section_index}),
                    text: cleaned,
                    title: Some(format!("Section {}", section_index + 1)),
                });
                section_index += 1;
            }
            current.clear();
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(trimmed);
    }

    // Flush remaining
    let cleaned = clean_text(&current);
    if !cleaned.is_empty() {
        units.push(StructuralUnit {
            unit_type: UNIT_TYPE_TEXT_SECTION.to_string(),
            unit_locator: serde_json::json!({"section_index": section_index}),
            text: cleaned,
            title: if section_index > 0 {
                Some(format!("Section {}", section_index + 1))
            } else {
                None
            },
        });
    }

    debug!("Text split into {} sections", units.len());
    Ok(units)
}

/// Split text output into heading-based sections.
///
/// Looks for lines that appear to be headings: short, no trailing punctuation,
/// followed by content. Used for HTML-converted text.
pub(super) fn split_text_by_headings(
    text: &str,
    unit_type: &str,
) -> Result<Vec<StructuralUnit>, LibraryDocumentError> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Ok(vec![]);
    }

    // Try markdown-style heading detection first
    let has_md_headings = lines.iter().any(|l| parse_atx_heading(l).is_some());
    if has_md_headings {
        return split_markdown_text(text);
    }

    // Fallback: treat entire text as single section
    let cleaned = clean_text(text);
    if cleaned.is_empty() {
        return Ok(vec![]);
    }

    Ok(vec![StructuralUnit {
        unit_type: unit_type.to_string(),
        unit_locator: serde_json::json!({"section_index": 0}),
        text: cleaned,
        title: None,
    }])
}

/// Strip RTF control codes, extracting plain text content.
pub(super) fn strip_rtf_control_codes(rtf: &str) -> String {
    let mut text = String::new();
    let mut i = 0;
    let chars: Vec<char> = rtf.chars().collect();
    let mut brace_depth = 0i32;
    let mut skip_group = false;

    while i < chars.len() {
        match chars[i] {
            '{' => {
                brace_depth += 1;
                // Check if this is a special group to skip
                let rest: String = chars[i..].iter().take(20).collect();
                if rest.contains("\\fonttbl")
                    || rest.contains("\\colortbl")
                    || rest.contains("\\stylesheet")
                    || rest.contains("\\info")
                {
                    skip_group = true;
                }
                i += 1;
            }
            '}' => {
                brace_depth -= 1;
                if brace_depth <= 0 {
                    skip_group = false;
                }
                i += 1;
            }
            '\\' if !skip_group => {
                i += 1;
                if i >= chars.len() {
                    break;
                }
                match chars[i] {
                    '\n' | '\r' => {
                        i += 1;
                    }
                    '\'' => {
                        // Hex escape: \'XX
                        i += 1;
                        if i + 1 < chars.len() {
                            let hex: String = chars[i..i + 2].iter().collect();
                            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                                text.push(byte as char);
                            }
                            i += 2;
                        }
                    }
                    _ => {
                        // Control word — skip until space or non-alpha
                        let mut word = String::new();
                        while i < chars.len() && chars[i].is_ascii_alphabetic() {
                            word.push(chars[i]);
                            i += 1;
                        }
                        // Skip optional numeric parameter
                        while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '-') {
                            i += 1;
                        }
                        // Skip trailing space delimiter
                        if i < chars.len() && chars[i] == ' ' {
                            i += 1;
                        }
                        // Translate known control words
                        match word.as_str() {
                            "par" | "line" => text.push('\n'),
                            "tab" => text.push('\t'),
                            _ => {}
                        }
                    }
                }
            }
            _ if !skip_group => {
                text.push(chars[i]);
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parent_unit::UNIT_TYPE_TEXT_SECTION;

    // ─── clean_text tests ───────────────────────────────────────────

    #[test]
    fn test_clean_text_basic() {
        assert_eq!(clean_text("  hello   world  "), "hello world");
        assert_eq!(clean_text("line1\n\nline2"), "line1\n\nline2");
        assert_eq!(clean_text(""), "");
    }

    #[test]
    fn test_clean_text_form_feeds() {
        assert_eq!(clean_text("page1\x0Cpage2"), "page1page2");
    }

    #[test]
    fn test_clean_text_control_chars() {
        assert_eq!(clean_text("hello\x01world"), "helloworld");
        assert_eq!(clean_text("keep\ttabs"), "keep tabs");
        assert_eq!(clean_text("keep\nnewlines"), "keep\nnewlines");
    }

    // ─── ATX heading parsing ────────────────────────────────────────

    #[test]
    fn test_parse_atx_heading_levels() {
        let h1 = parse_atx_heading("# Title").unwrap();
        assert_eq!(h1.level, 1);
        assert_eq!(h1.title, "Title");

        let h2 = parse_atx_heading("## Subtitle").unwrap();
        assert_eq!(h2.level, 2);
        assert_eq!(h2.title, "Subtitle");

        let h6 = parse_atx_heading("###### Deep").unwrap();
        assert_eq!(h6.level, 6);
        assert_eq!(h6.title, "Deep");
    }

    #[test]
    fn test_parse_atx_heading_not_heading() {
        assert!(parse_atx_heading("Not a heading").is_none());
        assert!(parse_atx_heading("#hashtag").is_none());
        assert!(parse_atx_heading("####### TooDeep").is_none());
        assert!(parse_atx_heading("# ").is_none());
    }

    #[test]
    fn test_parse_atx_heading_trailing_hashes() {
        let h = parse_atx_heading("## Title ##").unwrap();
        assert_eq!(h.title, "Title");
    }

    // ─── Markdown splitting ─────────────────────────────────────────

    #[test]
    fn test_split_markdown_basic() {
        let md = "# Intro\n\nSome intro text.\n\n## Details\n\nMore details here.";
        let units = split_markdown_text(md).unwrap();
        assert_eq!(units.len(), 2);
        assert_eq!(units[0].title, Some("Intro".to_string()));
        assert!(units[0].text.contains("intro text"));
        assert_eq!(units[1].title, Some("Details".to_string()));
        assert!(units[1].text.contains("More details"));
    }

    #[test]
    fn test_split_markdown_frontmatter_skipped() {
        let md = "---\ntitle: Test\nauthor: Me\n---\n\n# Chapter 1\n\nContent.";
        let units = split_markdown_text(md).unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].title, Some("Chapter 1".to_string()));
        assert!(!units[0].text.contains("title: Test"));
    }

    #[test]
    fn test_split_markdown_no_headings() {
        let md = "Just some plain text\nwith no headings at all.";
        let units = split_markdown_text(md).unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].unit_type, UNIT_TYPE_TEXT_SECTION);
    }

    // ─── Plain text splitting ───────────────────────────────────────

    #[test]
    fn test_split_text_by_paragraphs_single() {
        let text = "Short text.";
        let units = split_text_by_paragraphs(text).unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].text, "Short text.");
    }

    #[test]
    fn test_split_text_by_paragraphs_multiple() {
        let long_para = "A".repeat(1500);
        let text = format!("{}\n\n{}\n\n{}", long_para, long_para, long_para);
        let units = split_text_by_paragraphs(&text).unwrap();
        assert!(
            units.len() >= 2,
            "Long text should split into multiple sections"
        );
    }

    #[test]
    fn test_split_text_empty() {
        let units = split_text_by_paragraphs("").unwrap();
        assert!(units.is_empty());
    }

    // ─── RTF stripping ──────────────────────────────────────────────

    #[test]
    fn test_strip_rtf_basic() {
        let rtf = r"{\rtf1\ansi Hello World}";
        let text = strip_rtf_control_codes(rtf);
        assert!(text.contains("Hello World"), "Got: {}", text);
    }

    #[test]
    fn test_strip_rtf_paragraphs() {
        let rtf = r"{\rtf1 Line one\par Line two}";
        let text = strip_rtf_control_codes(rtf);
        assert!(text.contains("Line one"));
        assert!(text.contains("Line two"));
    }
}

//! RTF text extraction.

use std::collections::HashMap;
use std::path::Path;

use super::text::extract_text_with_encoding;
use super::xml_utils::clean_extracted_text;
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// RTF groups that contain metadata rather than text content.
const RTF_SKIP_GROUPS: &[&str] = &[
    "fonttbl",
    "colortbl",
    "stylesheet",
    "info",
    "pict",
    "object",
];

/// Extract text from RTF file by stripping RTF control codes
pub fn extract_rtf(file_path: &Path) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "rtf".to_string());

    let (raw_text, _) = extract_text_with_encoding(file_path)?;
    let stripped = strip_rtf_control_codes(&raw_text);

    let text = clean_extracted_text(&stripped);
    if text.is_empty() {
        return Err(DocumentProcessorError::DocxExtraction(
            "No text content found in RTF file".to_string(),
        ));
    }

    Ok((text, metadata))
}

/// Strip RTF control codes from a raw RTF string, returning plain text.
fn strip_rtf_control_codes(raw_text: &str) -> String {
    let mut result = String::with_capacity(raw_text.len());
    let mut skip_group_depth = 0i32;
    let mut chars = raw_text.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                if skip_group_depth > 0 {
                    skip_group_depth += 1;
                }
            }
            '}' => {
                if skip_group_depth > 0 {
                    skip_group_depth -= 1;
                }
            }
            '\\' if skip_group_depth == 0 => {
                if let Some(&next) = chars.peek() {
                    process_rtf_backslash(next, &mut chars, &mut result, &mut skip_group_depth);
                }
            }
            _ if skip_group_depth == 0 => {
                result.push(ch);
            }
            _ => {}
        }
    }

    result
}

/// Process one RTF backslash sequence, updating result and skip depth.
fn process_rtf_backslash(
    next: char,
    chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    result: &mut String,
    skip_group_depth: &mut i32,
) {
    if next == '\'' {
        // Hex-encoded character \'XX — skip 3 chars
        chars.next();
        chars.next();
        chars.next();
    } else if next == '\\' || next == '{' || next == '}' {
        result.push(chars.next().unwrap());
    } else if next == '\n' || next == '\r' {
        chars.next();
        result.push('\n');
    } else {
        let control_word = consume_rtf_control_word(chars);
        if control_word == "par" || control_word == "line" {
            result.push('\n');
        } else if control_word == "tab" {
            result.push('\t');
        } else if RTF_SKIP_GROUPS.contains(&control_word.as_str()) {
            *skip_group_depth = 1;
        }
    }
}

/// Consume one RTF control word (letters, optional numeric param) from the iterator.
fn consume_rtf_control_word(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut word = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_alphabetic() {
            word.push(c);
            chars.next();
        } else {
            // Consume optional numeric parameter
            if c == '-' || c.is_ascii_digit() {
                chars.next();
                while let Some(&d) = chars.peek() {
                    if d.is_ascii_digit() {
                        chars.next();
                    } else {
                        break;
                    }
                }
            }
            // Consume optional trailing space delimiter
            if chars.peek() == Some(&' ') {
                chars.next();
            }
            break;
        }
    }
    word
}

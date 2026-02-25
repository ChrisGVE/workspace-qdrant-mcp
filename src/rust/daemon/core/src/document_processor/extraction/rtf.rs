//! RTF text extraction.

use std::collections::HashMap;
use std::path::Path;

use super::text::extract_text_with_encoding;
use super::xml_utils::clean_extracted_text;
use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract text from RTF file by stripping RTF control codes
pub fn extract_rtf(
    file_path: &Path,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "rtf".to_string());

    let (raw_text, _) = extract_text_with_encoding(file_path)?;

    // Strip RTF control codes
    let mut result = String::with_capacity(raw_text.len());
    let mut skip_group_depth = 0i32; // Track depth of groups to skip (e.g., \fonttbl)
    let mut chars = raw_text.chars().peekable();

    // RTF groups that contain metadata, not text content
    let skip_groups = [
        "fonttbl",
        "colortbl",
        "stylesheet",
        "info",
        "pict",
        "object",
    ];

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
                // RTF control word or symbol
                if let Some(&next) = chars.peek() {
                    if next == '\'' {
                        // Hex-encoded character: skip the \'XX
                        chars.next();
                        chars.next();
                        chars.next();
                    } else if next == '\\' || next == '{' || next == '}' {
                        result.push(chars.next().unwrap());
                    } else if next == '\n' || next == '\r' {
                        chars.next();
                        result.push('\n');
                    } else {
                        let mut control_word = String::new();
                        while let Some(&c) = chars.peek() {
                            if c.is_ascii_alphabetic() {
                                control_word.push(c);
                                chars.next();
                            } else {
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
                                if chars.peek() == Some(&' ') {
                                    chars.next();
                                }
                                break;
                            }
                        }
                        if control_word == "par" || control_word == "line" {
                            result.push('\n');
                        } else if control_word == "tab" {
                            result.push('\t');
                        } else if skip_groups.contains(&control_word.as_str()) {
                            skip_group_depth = 1;
                        }
                    }
                }
            }
            _ if skip_group_depth == 0 => {
                result.push(ch);
            }
            _ => {}
        }
    }

    let text = clean_extracted_text(&result);
    if text.is_empty() {
        return Err(DocumentProcessorError::DocxExtraction(
            "No text content found in RTF file".to_string(),
        ));
    }

    Ok((text, metadata))
}

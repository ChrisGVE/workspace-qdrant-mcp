//! Plain text and code file extraction with encoding detection.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use chardet::detect;
use encoding_rs::Encoding;
use tracing::warn;

use crate::document_processor::types::{DocumentProcessorError, DocumentProcessorResult};

/// Extract text file with encoding detection
pub fn extract_text_with_encoding(
    file_path: &Path,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "text".to_string());

    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Try UTF-8 first (the overwhelming majority of source/text files).
    if let Ok(text) = std::str::from_utf8(&buffer) {
        metadata.insert("encoding".to_string(), "utf-8".to_string());
        return Ok((text.to_string(), metadata));
    }

    // Reject binary content BEFORE the legacy/lossy decodes below. chardet will
    // happily map an executable/image to a single-byte charset (e.g. latin-1)
    // that "decodes without errors", and the lossy fallback would turn it into
    // garbage "text" — which then feeds the chunker a multi-hundred-KB blob
    // with no line breaks, the trigger for the unbounded chunk-splitting loop
    // (#103). Heuristic (same as git): a NUL byte in the first 8 KiB means
    // binary. UTF-16/32 text legitimately contains NUL bytes, so skip the gate
    // when a UTF BOM is present — the encoding path below decodes those
    // correctly.
    const BINARY_SNIFF_LEN: usize = 8192;
    let has_utf_bom = buffer.starts_with(&[0xFF, 0xFE]) // UTF-16 LE / UTF-32 LE
        || buffer.starts_with(&[0xFE, 0xFF]) // UTF-16 BE
        || buffer.starts_with(&[0x00, 0x00, 0xFE, 0xFF]); // UTF-32 BE
    if !has_utf_bom {
        let sniff = &buffer[..buffer.len().min(BINARY_SNIFF_LEN)];
        if sniff.contains(&0) {
            return Err(DocumentProcessorError::BinaryFile(
                file_path.display().to_string(),
            ));
        }
    }

    // Detect encoding using chardet
    let detection = detect(&buffer);
    let encoding_name = detection.0.to_uppercase();
    metadata.insert("encoding".to_string(), encoding_name.clone());
    metadata.insert("encoding_confidence".to_string(), detection.1.to_string());

    // Try to decode using detected encoding
    if let Some(encoding) = Encoding::for_label(encoding_name.as_bytes()) {
        let (decoded, _, had_errors) = encoding.decode(&buffer);
        if !had_errors {
            return Ok((decoded.to_string(), metadata));
        }
    }

    // Fallback: decode as UTF-8 with lossy conversion
    warn!(
        "Encoding detection failed for {:?}, using lossy UTF-8",
        file_path
    );
    metadata.insert("encoding_fallback".to_string(), "true".to_string());
    Ok((String::from_utf8_lossy(&buffer).to_string(), metadata))
}

/// Extract code file with language metadata
pub fn extract_code(
    file_path: &Path,
    language: &str,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "code".to_string());
    metadata.insert("language".to_string(), language.to_string());

    let (text, mut text_metadata) = extract_text_with_encoding(file_path)?;

    // Merge metadata
    for (k, v) in text_metadata.drain() {
        metadata.entry(k).or_insert(v);
    }

    let line_count = text.lines().count();
    metadata.insert("line_count".to_string(), line_count.to_string());

    Ok((text, metadata))
}

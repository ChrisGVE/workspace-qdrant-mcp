//! Plain text and code file extraction with encoding detection.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use chardet::detect;
use encoding_rs::Encoding;
use tracing::warn;

use crate::document_processor::types::DocumentProcessorResult;

/// Extract text file with encoding detection
pub fn extract_text_with_encoding(
    file_path: &Path,
) -> DocumentProcessorResult<(String, HashMap<String, String>)> {
    let mut metadata = HashMap::new();
    metadata.insert("source_format".to_string(), "text".to_string());

    let mut file = File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Try UTF-8 first
    if let Ok(text) = std::str::from_utf8(&buffer) {
        metadata.insert("encoding".to_string(), "utf-8".to_string());
        return Ok((text.to_string(), metadata));
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

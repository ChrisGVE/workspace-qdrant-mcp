//! OCR enrichment for document processing.
//!
//! This module is entirely gated behind `#[cfg(feature = "ocr")]`.

use std::collections::HashMap;
use std::path::Path;

use tracing::{debug, warn};

use crate::image_extraction::extract_images;
use crate::ocr::OcrEngine;

/// Enrich document text with OCR output from embedded images.
///
/// Extracts images from the document, runs OCR on each, and appends
/// the recognized text to the document's raw text. Updates metadata
/// with OCR-related fields. OCR failures are logged as warnings and
/// do not block document processing.
pub fn enrich_text_with_ocr(
    file_path: &Path,
    mut raw_text: String,
    metadata: &mut HashMap<String, String>,
    ocr_engine: Option<&OcrEngine>,
) -> String {
    let engine = match ocr_engine {
        Some(e) => e,
        None => {
            // No OCR engine configured -- still record image count metadata
            let images = extract_images(file_path);
            if !images.is_empty() {
                metadata.insert("images_detected".to_string(), images.len().to_string());
            }
            return raw_text;
        }
    };

    let images = extract_images(file_path);
    if images.is_empty() {
        metadata.insert("images_detected".to_string(), "0".to_string());
        return raw_text;
    }

    metadata.insert("images_detected".to_string(), images.len().to_string());

    let mut ocr_texts: Vec<String> = Vec::new();
    let mut total_confidence: f32 = 0.0;
    let mut ocr_count: usize = 0;

    for (idx, image) in images.iter().enumerate() {
        match engine.extract_text(&image.bytes) {
            Ok(result) if !result.text.is_empty() => {
                debug!(
                    image_idx = idx,
                    confidence = result.confidence,
                    text_len = result.text.len(),
                    "OCR extracted text from image"
                );
                ocr_texts.push(result.text);
                total_confidence += result.confidence;
                ocr_count += 1;
            }
            Ok(_) => {
                debug!(image_idx = idx, "OCR produced empty text, skipping");
            }
            Err(e) => {
                warn!(
                    image_idx = idx,
                    error = %e,
                    "OCR failed for image, continuing"
                );
            }
        }
    }

    if !ocr_texts.is_empty() {
        append_ocr_texts(
            file_path,
            &mut raw_text,
            metadata,
            &ocr_texts,
            ocr_count,
            total_confidence,
            images.len(),
        );
    }

    raw_text
}

/// Append collected OCR texts to the document and record OCR metadata.
fn append_ocr_texts(
    file_path: &std::path::Path,
    raw_text: &mut String,
    metadata: &mut HashMap<String, String>,
    ocr_texts: &[String],
    ocr_count: usize,
    total_confidence: f32,
    image_count: usize,
) {
    raw_text.push_str("\n\n--- OCR Content ---\n\n");
    for (idx, text) in ocr_texts.iter().enumerate() {
        raw_text.push_str(&format!("[Image {}]\n{}\n\n", idx + 1, text));
    }

    metadata.insert("has_ocr_content".to_string(), "true".to_string());
    metadata.insert("ocr_images_processed".to_string(), ocr_count.to_string());
    let avg_confidence = total_confidence / ocr_count as f32;
    metadata.insert(
        "ocr_confidence".to_string(),
        format!("{:.2}", avg_confidence),
    );

    debug!(
        path = %file_path.display(),
        images = image_count,
        ocr_successful = ocr_count,
        "Enriched document with OCR text"
    );
}

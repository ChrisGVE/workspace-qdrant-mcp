/// OCR module — Tesseract-based text extraction from images.
///
/// Gated behind the `ocr` feature flag. Requires Tesseract and Leptonica
/// system libraries. Install via:
///   macOS: `brew install tesseract`
///   Linux: `apt-get install tesseract-ocr libtesseract-dev`
pub mod errors;

#[cfg(feature = "ocr")]
mod engine;

#[cfg(feature = "ocr")]
pub use engine::{OcrConfig, OcrEngine, OcrResult};

pub use errors::OcrError;

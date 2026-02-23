/// OCR-specific error types.
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OcrError {
    #[error("Tesseract initialization failed: {0}")]
    InitFailed(String),

    #[error("OCR extraction failed: {0}")]
    ExtractionFailed(String),

    #[error("Unsupported image format: {0}")]
    UnsupportedFormat(String),

    #[error("Tessdata not found at {0}")]
    TessdataNotFound(String),

    #[error("Image too small ({width}x{height}), minimum is {min_width}x{min_height}")]
    ImageTooSmall {
        width: u32,
        height: u32,
        min_width: u32,
        min_height: u32,
    },
}

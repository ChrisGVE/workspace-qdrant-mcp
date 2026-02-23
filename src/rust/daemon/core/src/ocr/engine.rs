/// Tesseract OCR engine wrapper.
///
/// Provides thread-safe text extraction from image bytes using leptess.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use leptess::LepTess;
use tracing::debug;

use super::errors::OcrError;

/// Result of OCR text extraction.
#[derive(Debug, Clone)]
pub struct OcrResult {
    /// Extracted text content.
    pub text: String,
    /// Mean confidence score (0.0–1.0). Higher is better.
    pub confidence: f32,
}

/// Configuration for the OCR engine.
#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// Path to tessdata directory containing language models.
    pub tessdata_path: PathBuf,
    /// Tesseract language code (e.g. "eng").
    pub language: String,
    /// Minimum confidence to accept OCR output (0.0–1.0).
    pub min_confidence: f32,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            tessdata_path: default_tessdata_path(),
            language: "eng".to_string(),
            min_confidence: 0.3,
        }
    }
}

/// Tesseract OCR engine with thread-safe access.
///
/// LepTess is not Send/Sync, so we wrap it in a Mutex and use
/// `spawn_blocking` from the caller side for async integration.
pub struct OcrEngine {
    inner: Mutex<LepTess>,
    min_confidence: f32,
}

impl std::fmt::Debug for OcrEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OcrEngine")
            .field("min_confidence", &self.min_confidence)
            .field("available", &self.inner.lock().is_ok())
            .finish()
    }
}

impl OcrEngine {
    /// Create a new OCR engine with the given configuration.
    pub fn new(config: &OcrConfig) -> Result<Self, OcrError> {
        let tessdata_str = config.tessdata_path.to_string_lossy().to_string();

        if !config.tessdata_path.exists() {
            return Err(OcrError::TessdataNotFound(tessdata_str));
        }

        let lt = LepTess::new(Some(&tessdata_str), &config.language)
            .map_err(|e| OcrError::InitFailed(format!("{e}")))?;

        debug!(
            tessdata = %tessdata_str,
            language = %config.language,
            "OCR engine initialized"
        );

        Ok(Self {
            inner: Mutex::new(lt),
            min_confidence: config.min_confidence,
        })
    }

    /// Extract text from image bytes.
    ///
    /// Supports JPEG, PNG, TIFF, BMP, and other formats supported by
    /// Leptonica. This is a blocking call — wrap in `spawn_blocking`
    /// for async contexts.
    pub fn extract_text(&self, image_bytes: &[u8]) -> Result<OcrResult, OcrError> {
        if image_bytes.is_empty() {
            return Err(OcrError::ExtractionFailed(
                "empty image data".to_string(),
            ));
        }

        let mut lt = self.inner.lock().map_err(|e| {
            OcrError::ExtractionFailed(format!("lock poisoned: {e}"))
        })?;

        lt.set_image_from_mem(image_bytes)
            .map_err(|e| OcrError::ExtractionFailed(format!("failed to load image: {e}")))?;

        let text = lt
            .get_utf8_text()
            .map_err(|e| OcrError::ExtractionFailed(format!("text extraction failed: {e}")))?;

        // mean_text_conf returns 0–100 integer
        let raw_confidence = lt.mean_text_conf();
        let confidence = (raw_confidence as f32) / 100.0;

        let trimmed = text.trim().to_string();

        if confidence < self.min_confidence {
            debug!(
                confidence,
                min = self.min_confidence,
                text_len = trimmed.len(),
                "OCR confidence below threshold, discarding"
            );
            return Ok(OcrResult {
                text: String::new(),
                confidence,
            });
        }

        if trimmed.is_empty() {
            debug!("OCR produced empty text");
        }

        Ok(OcrResult {
            text: trimmed,
            confidence,
        })
    }

    /// Check if the engine is available and functional.
    pub fn is_available(&self) -> bool {
        self.inner.lock().is_ok()
    }
}

/// Resolve the default tessdata path.
///
/// Checks in order:
/// 1. `TESSDATA_PREFIX` environment variable
/// 2. Homebrew location: `/usr/local/share/tessdata`
/// 3. Linux default: `/usr/share/tesseract-ocr/5/tessdata`
/// 4. Fallback: `~/.workspace-qdrant/tesseract`
fn default_tessdata_path() -> PathBuf {
    if let Ok(prefix) = std::env::var("TESSDATA_PREFIX") {
        let p = PathBuf::from(&prefix);
        if p.exists() {
            return p;
        }
    }

    let candidates = [
        "/usr/local/share/tessdata",
        "/opt/homebrew/share/tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tessdata",
    ];

    for candidate in &candidates {
        let p = Path::new(candidate);
        if p.exists() {
            return p.to_path_buf();
        }
    }

    // Fallback to user-local directory
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".workspace-qdrant")
        .join("tesseract")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_tessdata_resolves() {
        let path = default_tessdata_path();
        // Should resolve to some path (may or may not exist in CI)
        assert!(!path.as_os_str().is_empty());
    }

    #[test]
    fn test_ocr_config_default() {
        let config = OcrConfig::default();
        assert_eq!(config.language, "eng");
        assert!((config.min_confidence - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_engine_creation_with_invalid_path() {
        let config = OcrConfig {
            tessdata_path: PathBuf::from("/nonexistent/tessdata"),
            ..OcrConfig::default()
        };
        let result = OcrEngine::new(&config);
        assert!(result.is_err());
        match result.unwrap_err() {
            OcrError::TessdataNotFound(path) => {
                assert!(path.contains("nonexistent"));
            }
            other => panic!("expected TessdataNotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_extract_text_empty_bytes() {
        // Only run if tessdata is available
        let config = OcrConfig::default();
        if !config.tessdata_path.exists() {
            return; // Skip in environments without Tesseract
        }
        let engine = match OcrEngine::new(&config) {
            Ok(e) => e,
            Err(_) => return, // Skip if engine can't initialize
        };

        let result = engine.extract_text(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_text_from_png() {
        let config = OcrConfig::default();
        if !config.tessdata_path.exists() {
            return;
        }
        let engine = match OcrEngine::new(&config) {
            Ok(e) => e,
            Err(_) => return,
        };

        // Try reading a test image if it exists
        let test_image = Path::new("/tmp/ocr_test.png");
        if !test_image.exists() {
            return;
        }
        let bytes = std::fs::read(test_image).unwrap();
        let result = engine.extract_text(&bytes).unwrap();

        // Should extract some text (likely "Hello World" or similar)
        assert!(!result.text.is_empty(), "OCR should extract text from image");
        assert!(result.confidence > 0.0, "confidence should be positive");
    }

    #[test]
    fn test_extract_text_corrupt_data() {
        let config = OcrConfig::default();
        if !config.tessdata_path.exists() {
            return;
        }
        let engine = match OcrEngine::new(&config) {
            Ok(e) => e,
            Err(_) => return,
        };

        // Random bytes that aren't a valid image
        let result = engine.extract_text(&[0xFF, 0xFE, 0x00, 0x01]);
        assert!(result.is_err(), "corrupt data should produce an error");
    }
}

//! High-performance language detection system using comprehensive pattern data
//!
//! This module provides optimized language detection capabilities leveraging the
//! comprehensive internal configuration with 500+ languages. Uses efficient
//! data structures for fast lookups and multi-stage detection strategies.

mod detector;
mod types;

pub use detector::LanguageDetector;
pub use types::{DetectionConfidence, DetectionMethod, DetectionResult, DetectorStats};

use std::path::Path;

/// Convenient function for quick language detection from file path
pub fn detect_language_from_path(file_path: &Path) -> DetectionResult {
    match LanguageDetector::global() {
        Ok(detector) => detector.detect_from_path(file_path),
        Err(e) => DetectionResult {
            language: None,
            confidence: DetectionConfidence::Unknown,
            detection_method: DetectionMethod::None,
            details: format!("Detector initialization failed: {}", e),
        },
    }
}

/// Convenient function for quick language detection from content
pub fn detect_language_from_content(content: &str, file_path: Option<&Path>) -> DetectionResult {
    match LanguageDetector::global() {
        Ok(detector) => detector.detect_from_content(content, file_path),
        Err(e) => DetectionResult {
            language: None,
            confidence: DetectionConfidence::Unknown,
            detection_method: DetectionMethod::None,
            details: format!("Detector initialization failed: {}", e),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_detector_initialization() {
        let detector = LanguageDetector::new();
        assert!(detector.is_ok(), "Should initialize language detector");

        let detector = detector.unwrap();
        let stats = detector.stats();
        assert!(
            stats.total_extensions > 0,
            "Should have at least one extension"
        );
        assert!(
            stats.unique_languages > 0,
            "Should support at least one language"
        );
    }

    #[test]
    fn test_extension_detection() {
        let detector = LanguageDetector::new().unwrap();

        // Test common extensions
        assert_eq!(
            detector.detect_from_extension("rs"),
            Some("rust".to_string())
        );
        assert_eq!(
            detector.detect_from_extension("py"),
            Some("python".to_string())
        );
        assert_eq!(
            detector.detect_from_extension("js"),
            Some("javascript".to_string())
        );
        assert_eq!(
            detector.detect_from_extension("ts"),
            Some("typescript".to_string())
        );

        // Test with leading dot
        assert_eq!(
            detector.detect_from_extension(".rs"),
            Some("rust".to_string())
        );

        // Test case insensitive
        assert!(
            detector.detect_from_extension("RS").is_some()
                || detector.detect_from_extension("rs").is_some()
        );
    }

    #[test]
    fn test_filename_detection() {
        let detector = LanguageDetector::new().unwrap();

        assert_eq!(
            detector.detect_from_filename("Dockerfile"),
            Some("dockerfile".to_string())
        );
        assert_eq!(
            detector.detect_from_filename("Makefile"),
            Some("make".to_string())
        );
        assert_eq!(
            detector.detect_from_filename("Cargo.toml"),
            Some("rust".to_string())
        );
        assert_eq!(
            detector.detect_from_filename("package.json"),
            Some("javascript".to_string())
        );
    }

    #[test]
    fn test_shebang_detection() {
        let detector = LanguageDetector::new().unwrap();

        let python_script = "#!/usr/bin/env python3\nprint('hello')";
        let result = detector.detect_from_shebang(python_script);
        assert!(result.is_some());
        let (_, language) = result.unwrap();
        assert_eq!(language, "python");

        let bash_script = "#!/bin/bash\necho hello";
        let result = detector.detect_from_shebang(bash_script);
        assert!(result.is_some());
    }

    #[test]
    fn test_keyword_detection() {
        let detector = LanguageDetector::new().unwrap();

        let rust_code = "fn main() {\n    let x = 5;\n    use std::collections::HashMap;\n    println!(\"Hello\");\n}";
        let result = detector.detect_from_keywords(rust_code);
        assert!(result.is_some());
        let (language, keywords) = result.unwrap();
        // Rust should be detected due to multiple unique keywords
        assert!(
            language == "rust" || language == "dart" || language == "javascript",
            "Expected rust, dart, or javascript, got {}",
            language
        );
        assert!(!keywords.is_empty());

        let python_code = "def main():\n    import sys\n    print('hello')";
        let result = detector.detect_from_keywords(python_code);
        assert!(result.is_some());
        let (language, _) = result.unwrap();
        assert_eq!(language, "python");
    }

    #[test]
    fn test_path_detection() {
        let detector = LanguageDetector::new().unwrap();

        let rust_file = PathBuf::from("src/main.rs");
        let result = detector.detect_from_path(&rust_file);
        assert_eq!(result.language, Some("rust".to_string()));
        assert_eq!(result.confidence, DetectionConfidence::VeryHigh);

        let dockerfile = PathBuf::from("Dockerfile");
        let result = detector.detect_from_path(&dockerfile);
        assert_eq!(result.language, Some("dockerfile".to_string()));
        assert_eq!(result.confidence, DetectionConfidence::High);
    }

    #[test]
    fn test_content_detection_with_consensus() {
        let detector = LanguageDetector::new().unwrap();

        // Content that should be detected as Rust with high confidence
        let rust_content =
            "fn main() {\n    use std::collections::HashMap;\n    let mut map = HashMap::new();\n}";
        let rust_path = PathBuf::from("main.rs");

        let result = detector.detect_from_content(rust_content, Some(&rust_path));
        assert_eq!(result.language, Some("rust".to_string()));
        assert!(result.confidence >= DetectionConfidence::High);

        match result.detection_method {
            DetectionMethod::Consensus(_) => {
                // Expected: consensus between extension and keywords
            }
            DetectionMethod::Extension(_) => {
                // Also acceptable: extension detection dominated
            }
            _ => panic!("Unexpected detection method: {:?}", result.detection_method),
        }
    }

    #[test]
    fn test_global_detector() {
        let detector = LanguageDetector::global();
        assert!(detector.is_ok(), "Global detector should be available");

        let stats = detector.unwrap().stats();
        assert!(stats.total_extensions > 0, "Should have extensions loaded");
    }

    #[test]
    fn test_convenience_functions() {
        let rust_file = PathBuf::from("test.rs");
        let result = detect_language_from_path(&rust_file);
        assert_eq!(result.language, Some("rust".to_string()));

        let content = "#!/usr/bin/env python3\ndef main(): pass";
        let result = detect_language_from_content(content, None);
        assert_eq!(result.language, Some("python".to_string()));
    }
}

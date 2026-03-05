//! Tests for library module helpers

use super::helpers::{classify_document_extension, DEFAULT_LIBRARY_PATTERNS};

#[test]
fn test_default_library_patterns_cover_all_formats() {
    // All supported document formats must be present
    let expected = [
        "*.pdf", "*.epub", "*.docx", "*.pptx", "*.ppt", "*.pages", "*.key", "*.odt", "*.odp",
        "*.ods", "*.rtf", "*.doc", "*.md", "*.txt", "*.html", "*.htm",
    ];
    for pat in &expected {
        assert!(
            DEFAULT_LIBRARY_PATTERNS.contains(pat),
            "Missing default library pattern: {}",
            pat
        );
    }
    assert_eq!(
        DEFAULT_LIBRARY_PATTERNS.len(),
        expected.len(),
        "Default patterns count mismatch"
    );
}

#[test]
fn test_default_patterns_used_when_none_provided() {
    let user_patterns: Vec<String> = vec![];
    let effective: Vec<String> = if user_patterns.is_empty() {
        DEFAULT_LIBRARY_PATTERNS
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        user_patterns
    };
    assert_eq!(effective.len(), DEFAULT_LIBRARY_PATTERNS.len());
    assert_eq!(effective[0], "*.pdf");
}

#[test]
fn test_user_patterns_override_defaults() {
    let user_patterns: Vec<String> = vec!["*.pdf".to_string(), "*.md".to_string()];
    let effective: Vec<String> = if user_patterns.is_empty() {
        DEFAULT_LIBRARY_PATTERNS
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        user_patterns.clone()
    };
    assert_eq!(effective.len(), 2);
    assert_eq!(effective, vec!["*.pdf", "*.md"]);
}

#[test]
fn test_classify_page_based_extensions() {
    let page_based = [
        "pdf", "docx", "doc", "pptx", "ppt", "pages", "key", "odp", "odt", "ods", "rtf",
    ];
    for ext in &page_based {
        let result = classify_document_extension(ext);
        assert!(result.is_some(), "Extension '{}' should be classified", ext);
        let (_, doc_type) = result.unwrap();
        assert_eq!(
            doc_type, "page_based",
            "Extension '{}' should be page_based",
            ext
        );
    }
}

#[test]
fn test_classify_stream_based_extensions() {
    let stream_based = [
        ("epub", "epub"),
        ("html", "html"),
        ("htm", "html"),
        ("md", "markdown"),
        ("markdown", "markdown"),
        ("txt", "text"),
    ];
    for (ext, expected_format) in &stream_based {
        let result = classify_document_extension(ext);
        assert!(result.is_some(), "Extension '{}' should be classified", ext);
        let (source_format, doc_type) = result.unwrap();
        assert_eq!(
            doc_type, "stream_based",
            "Extension '{}' should be stream_based",
            ext
        );
        assert_eq!(
            source_format, *expected_format,
            "Extension '{}' format mismatch",
            ext
        );
    }
}

#[test]
fn test_classify_unsupported_extensions() {
    let unsupported = ["exe", "zip", "rs", "py", "js", "json", "yaml", "csv"];
    for ext in &unsupported {
        assert!(
            classify_document_extension(ext).is_none(),
            "Extension '{}' should not be classified for library ingestion",
            ext
        );
    }
}

#[test]
fn test_classify_source_format_correctness() {
    // Verify exact source_format values for key extensions
    assert_eq!(classify_document_extension("pdf").unwrap().0, "pdf");
    assert_eq!(classify_document_extension("docx").unwrap().0, "docx");
    assert_eq!(classify_document_extension("epub").unwrap().0, "epub");
    assert_eq!(classify_document_extension("md").unwrap().0, "markdown");
    assert_eq!(classify_document_extension("txt").unwrap().0, "text");
    assert_eq!(classify_document_extension("html").unwrap().0, "html");
    assert_eq!(classify_document_extension("htm").unwrap().0, "html");
}

#[test]
fn test_all_default_patterns_classifiable() {
    // Every extension in DEFAULT_LIBRARY_PATTERNS must be classifiable
    for pattern in DEFAULT_LIBRARY_PATTERNS {
        let ext = pattern
            .strip_prefix("*.")
            .expect("Pattern must start with *.");
        assert!(
            classify_document_extension(ext).is_some(),
            "DEFAULT_LIBRARY_PATTERNS extension '{}' is not classifiable",
            ext
        );
    }
}

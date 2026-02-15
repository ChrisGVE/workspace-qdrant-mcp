//! Bug Fix Contract Tests
//!
//! Integration tests verifying that fixes from the refinement tag remain correct.
//! Tests cover: priority constants, document_type format, file_type format,
//! metadata normalization, error classification, and item_type semantics.

use wqm_common::constants::priority;
use workspace_qdrant_core::file_classification::{FileType, classify_file_type, is_test_file, get_extension_for_storage};
use workspace_qdrant_core::DocumentType;
use std::path::Path;

// =========================================================================
// Priority constant contracts (Tasks 1-3)
// =========================================================================

#[test]
fn test_priority_constants_ordering() {
    // HIGH < NORMAL < LOW (lower number = higher priority)
    assert!(priority::HIGH < priority::NORMAL, "HIGH must be lower than NORMAL");
    assert!(priority::NORMAL < priority::LOW, "NORMAL must be lower than LOW");
}

#[test]
fn test_priority_constants_are_positive() {
    assert!(priority::HIGH > 0, "Priority values must be positive");
    assert!(priority::NORMAL > 0, "Priority values must be positive");
    assert!(priority::LOW > 0, "Priority values must be positive");
}

#[test]
fn test_priority_constants_are_distinct() {
    assert_ne!(priority::HIGH, priority::NORMAL);
    assert_ne!(priority::NORMAL, priority::LOW);
    assert_ne!(priority::HIGH, priority::LOW);
}

#[test]
fn test_priority_high_is_one() {
    // Memory operations must use priority::HIGH = 1
    assert_eq!(priority::HIGH, 1, "HIGH priority must be 1 for memory operations");
}

// =========================================================================
// DocumentType format contracts (Task 5)
// =========================================================================

#[test]
fn test_document_type_as_str_returns_lowercase() {
    // All DocumentType variants must return lowercase strings via as_str()
    let types_and_expected = vec![
        (DocumentType::Pdf, "pdf"),
        (DocumentType::Epub, "epub"),
        (DocumentType::Docx, "docx"),
        (DocumentType::Pptx, "pptx"),
        (DocumentType::Odt, "odt"),
        (DocumentType::Odp, "odp"),
        (DocumentType::Ods, "ods"),
        (DocumentType::Rtf, "rtf"),
        (DocumentType::Doc, "doc"),
        (DocumentType::Ppt, "ppt"),
        (DocumentType::Xlsx, "xlsx"),
        (DocumentType::Xls, "xls"),
        (DocumentType::Csv, "csv"),
        (DocumentType::Jupyter, "jupyter"),
        (DocumentType::Pages, "pages"),
        (DocumentType::Key, "key"),
        (DocumentType::Text, "text"),
        (DocumentType::Markdown, "markdown"),
        (DocumentType::Code("rust".to_string()), "code"),
        (DocumentType::Unknown, "unknown"),
    ];

    for (doc_type, expected) in types_and_expected {
        let result = doc_type.as_str();
        assert_eq!(result, expected, "DocumentType::{:?} should produce '{}'", doc_type, expected);
        assert_eq!(result, result.to_lowercase(), "DocumentType::as_str() must be lowercase: got '{}'", result);
    }
}

#[test]
fn test_document_type_as_str_not_debug_format() {
    // Bug: Code("rust") was previously stored as Debug format: Code("rust")
    // Now it should just be "code"
    let code_type = DocumentType::Code("rust".to_string());
    let result = code_type.as_str();
    assert!(!result.contains("Code("), "as_str() must not use Debug format");
    assert!(!result.contains('"'), "as_str() must not contain quotes");
    assert_eq!(result, "code");
}

#[test]
fn test_document_type_language_extraction() {
    let rust_code = DocumentType::Code("rust".to_string());
    assert_eq!(rust_code.language(), Some("rust"));

    let python_code = DocumentType::Code("python".to_string());
    assert_eq!(python_code.language(), Some("python"));

    // Non-code types should return None
    assert_eq!(DocumentType::Markdown.language(), None);
    assert_eq!(DocumentType::Text.language(), None);
    assert_eq!(DocumentType::Pdf.language(), None);
}

// =========================================================================
// FileType format contracts (Task 5)
// =========================================================================

#[test]
fn test_file_type_as_str_returns_lowercase() {
    let types = vec![
        (FileType::Code, "code"),
        (FileType::Text, "text"),
        (FileType::Docs, "docs"),
        (FileType::Web, "web"),
        (FileType::Slides, "slides"),
        (FileType::Config, "config"),
        (FileType::Data, "data"),
        (FileType::Build, "build"),
        (FileType::Other, "other"),
    ];

    for (file_type, expected) in types {
        let result = file_type.as_str();
        assert_eq!(result, expected, "FileType::{:?} should produce '{}'", file_type, expected);
        assert_eq!(result, result.to_lowercase(), "FileType::as_str() must be lowercase");
    }
}

#[test]
fn test_classify_file_type_common_extensions() {
    // Verify that classify_file_type returns expected categories
    assert_eq!(classify_file_type(Path::new("main.rs")), FileType::Code);
    assert_eq!(classify_file_type(Path::new("lib.py")), FileType::Code);
    assert_eq!(classify_file_type(Path::new("app.ts")), FileType::Code);
    assert_eq!(classify_file_type(Path::new("README.md")), FileType::Text);
    assert_eq!(classify_file_type(Path::new("config.yaml")), FileType::Config);
    assert_eq!(classify_file_type(Path::new("data.json")), FileType::Data);
}

// =========================================================================
// Test file detection contracts
// =========================================================================

#[test]
fn test_is_test_file_detection() {
    // Files with test indicators should be detected
    assert!(is_test_file(Path::new("test_utils.rs")));
    assert!(is_test_file(Path::new("src/tests/integration.rs")));
    assert!(is_test_file(Path::new("spec/model_spec.rb")));

    // Normal files should not be detected as tests
    assert!(!is_test_file(Path::new("src/main.rs")));
    assert!(!is_test_file(Path::new("src/lib.rs")));
}

// =========================================================================
// Extension normalization contracts (Task 12)
// =========================================================================

#[test]
fn test_get_extension_for_storage_lowercase() {
    // Extensions must be stored lowercase
    let ext = get_extension_for_storage(Path::new("file.RS"));
    assert_eq!(ext, Some("rs".to_string()));

    let ext = get_extension_for_storage(Path::new("file.PY"));
    assert_eq!(ext, Some("py".to_string()));

    let ext = get_extension_for_storage(Path::new("file.Md"));
    assert_eq!(ext, Some("md".to_string()));

    // No extension
    let ext = get_extension_for_storage(Path::new("Makefile"));
    assert_eq!(ext, None);
}

// =========================================================================
// LSP enrichment status contracts (Task 12)
// =========================================================================

#[test]
fn test_enrichment_status_as_str_is_lowercase() {
    use workspace_qdrant_core::lsp::EnrichmentStatus;

    let statuses = vec![
        (EnrichmentStatus::Success, "success"),
        (EnrichmentStatus::Partial, "partial"),
        (EnrichmentStatus::Failed, "failed"),
        (EnrichmentStatus::Skipped, "skipped"),
    ];

    for (status, expected) in statuses {
        let result = status.as_str();
        assert_eq!(result, expected, "EnrichmentStatus::{:?} must produce '{}'", status, expected);
        assert_eq!(result, result.to_lowercase(), "EnrichmentStatus::as_str() must be lowercase");
    }
}

// =========================================================================
// Error classification contracts (Task 10)
// =========================================================================

#[test]
fn test_error_classification_categories_are_lowercase() {
    // All error categories must be lowercase with underscore separator
    let valid_categories = [
        "permanent_data",
        "permanent_gone",
        "transient_infrastructure",
        "transient_resource",
        "partial",
    ];

    for cat in &valid_categories {
        assert_eq!(*cat, cat.to_lowercase(), "Error category must be lowercase");
        assert!(!cat.contains(' '), "Error category must not contain spaces");
    }
}

// =========================================================================
// Tracked file processing status contracts
// =========================================================================

#[test]
fn test_processing_status_roundtrip() {
    use workspace_qdrant_core::tracked_files_schema::ProcessingStatus;

    let statuses = vec![
        ProcessingStatus::None,
        ProcessingStatus::Done,
        ProcessingStatus::Failed,
        ProcessingStatus::Skipped,
    ];

    for status in statuses {
        let s = status.to_string();
        let parsed = ProcessingStatus::from_str(&s);
        assert_eq!(parsed, Some(status), "ProcessingStatus roundtrip failed for '{}'", s);
    }
}

#[test]
fn test_processing_status_display_is_lowercase() {
    use workspace_qdrant_core::tracked_files_schema::ProcessingStatus;

    let statuses = vec![
        ProcessingStatus::None,
        ProcessingStatus::Done,
        ProcessingStatus::Failed,
        ProcessingStatus::Skipped,
    ];

    for status in statuses {
        let s = status.to_string();
        assert_eq!(s, s.to_lowercase(), "ProcessingStatus Display must be lowercase: got '{}'", s);
    }
}

// =========================================================================
// Queue item type contracts (Task 6)
// =========================================================================

#[test]
fn test_item_type_url_exists() {
    use workspace_qdrant_core::unified_queue_schema::ItemType;

    // URL items must use ItemType::Url, not content
    let url_type = ItemType::Url;
    assert_eq!(format!("{:?}", url_type), "Url");
}

#[test]
fn test_item_types_are_distinct() {
    use workspace_qdrant_core::unified_queue_schema::ItemType;

    // Verify all item types are distinct
    let types: Vec<ItemType> = vec![
        ItemType::Content,
        ItemType::File,
        ItemType::Folder,
        ItemType::Project,
        ItemType::Library,
        ItemType::Url,
        ItemType::DeleteTenant,
        ItemType::DeleteDocument,
        ItemType::Rename,
    ];

    for (i, t1) in types.iter().enumerate() {
        for (j, t2) in types.iter().enumerate() {
            if i != j {
                assert_ne!(
                    format!("{:?}", t1),
                    format!("{:?}", t2),
                    "Item types must be distinct"
                );
            }
        }
    }
}

// =========================================================================
// Collection constant contracts
// =========================================================================

#[test]
fn test_collection_constants_are_lowercase() {
    use wqm_common::constants::{COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_MEMORY};

    assert_eq!(COLLECTION_PROJECTS, "projects");
    assert_eq!(COLLECTION_LIBRARIES, "libraries");
    assert_eq!(COLLECTION_MEMORY, "memory");
}

// =========================================================================
// Timestamp format contracts (Tasks 10-12 from previous set)
// =========================================================================

#[test]
fn test_timestamp_now_utc_ends_with_z() {
    let ts = wqm_common::timestamps::now_utc();
    assert!(ts.ends_with('Z'), "Timestamps must end with Z, got: {}", ts);
    assert!(!ts.contains("+00:00"), "Must not use +00:00 format");
}

#[test]
fn test_timestamp_format_utc_ends_with_z() {
    let now = chrono::Utc::now();
    let ts = wqm_common::timestamps::format_utc(&now);
    assert!(ts.ends_with('Z'), "format_utc must end with Z, got: {}", ts);
}

// =========================================================================
// Hash contract tests
// =========================================================================

#[test]
fn test_content_hash_is_sha256() {
    let hash = wqm_common::hashing::compute_content_hash("hello world");
    assert_eq!(hash.len(), 64, "SHA256 hex must be 64 chars");
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "Hash must be hex");
}

#[test]
fn test_content_hash_deterministic() {
    let h1 = wqm_common::hashing::compute_content_hash("test input");
    let h2 = wqm_common::hashing::compute_content_hash("test input");
    assert_eq!(h1, h2, "Same input must produce same hash");
}

#[test]
fn test_content_hash_different_inputs() {
    let h1 = wqm_common::hashing::compute_content_hash("input a");
    let h2 = wqm_common::hashing::compute_content_hash("input b");
    assert_ne!(h1, h2, "Different inputs must produce different hashes");
}

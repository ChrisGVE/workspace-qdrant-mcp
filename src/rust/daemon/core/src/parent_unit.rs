//! Parent-unit record structure for Qdrant.
//!
//! Parent records store full structural units (pages, chapters, code files)
//! without vectors. They serve as expansion targets: when a search returns
//! a chunk, the parent record provides the full surrounding context.
//!
//! Parent records live in the same collection as chunks, discriminated by
//! `record_type = "parent"` vs `record_type = "chunk"`.

use sha2::{Sha256, Digest};
use std::collections::HashMap;
use uuid::Uuid;

/// Record type discriminator values
pub const RECORD_TYPE_PARENT: &str = "parent";
pub const RECORD_TYPE_CHUNK: &str = "chunk";

/// Unit type values for parent records
pub const UNIT_TYPE_PDF_PAGE: &str = "pdf_page";
pub const UNIT_TYPE_EPUB_SECTION: &str = "epub_section";
pub const UNIT_TYPE_CODE_FILE: &str = "code_file";
pub const UNIT_TYPE_CODE_BLOCK: &str = "code_block";
pub const UNIT_TYPE_TEXT_SECTION: &str = "text_section";
pub const UNIT_TYPE_DOCX_SECTION: &str = "docx_section";

/// A parent-unit record for Qdrant.
///
/// Parent records store full structural units without vectors.
/// Child chunks reference their parent via `parent_unit_id`.
#[derive(Debug, Clone)]
pub struct ParentUnitRecord {
    /// Unique point ID for this parent record (deterministic, hex string)
    pub point_id: String,
    /// Document identifier (UUID v5)
    pub doc_id: String,
    /// SHA256 fingerprint of the source file
    pub doc_fingerprint: String,
    /// Structural unit type (pdf_page, epub_section, code_file, etc.)
    pub unit_type: String,
    /// Locator for this unit within the document (JSON value)
    /// Examples: {"page": 12}, {"spine_id": "ch1", "chapter_title": "Introduction"}
    pub unit_locator: serde_json::Value,
    /// Full extracted text of this unit
    pub unit_text: String,
    /// Character length of unit_text
    pub unit_char_len: usize,
    /// SHA256 hash of unit_text for change detection
    pub unit_hash: String,
}

impl ParentUnitRecord {
    /// Generate the Qdrant payload for this parent record.
    ///
    /// Includes all provenance and parent-child fields but NO vectors.
    pub fn to_payload(
        &self,
        library_name: &str,
        doc_title: Option<&str>,
        doc_type: &str,
        source_format: &str,
    ) -> HashMap<String, serde_json::Value> {
        use wqm_common::schema::qdrant::libraries;

        let mut payload = HashMap::new();

        // Core identification
        payload.insert(
            libraries::LIBRARY_NAME.name.to_string(),
            serde_json::Value::String(library_name.to_string()),
        );

        // Record type discriminator
        payload.insert(
            libraries::RECORD_TYPE.name.to_string(),
            serde_json::Value::String(RECORD_TYPE_PARENT.to_string()),
        );

        // Document provenance
        payload.insert(
            libraries::DOC_ID.name.to_string(),
            serde_json::Value::String(self.doc_id.clone()),
        );
        payload.insert(
            libraries::DOC_FINGERPRINT.name.to_string(),
            serde_json::Value::String(self.doc_fingerprint.clone()),
        );
        payload.insert(
            libraries::DOC_TYPE.name.to_string(),
            serde_json::Value::String(doc_type.to_string()),
        );
        payload.insert(
            libraries::SOURCE_FORMAT.name.to_string(),
            serde_json::Value::String(source_format.to_string()),
        );
        if let Some(title) = doc_title {
            payload.insert(
                libraries::DOC_TITLE.name.to_string(),
                serde_json::Value::String(title.to_string()),
            );
        }

        // Unit metadata
        payload.insert(
            libraries::UNIT_TYPE.name.to_string(),
            serde_json::Value::String(self.unit_type.clone()),
        );
        payload.insert(
            libraries::UNIT_LOCATOR.name.to_string(),
            self.unit_locator.clone(),
        );
        payload.insert(
            libraries::UNIT_HASH.name.to_string(),
            serde_json::Value::String(self.unit_hash.clone()),
        );

        // Full text stored for expansion retrieval
        payload.insert(
            libraries::CHUNK_TEXT_RAW.name.to_string(),
            serde_json::Value::String(self.unit_text.clone()),
        );

        payload
    }
}

/// Compute SHA256 hash of text, returning hex string.
pub fn sha256_hex(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Generate a deterministic point ID for a parent unit.
///
/// Uses UUID v5 with the document ID and unit locator as input.
pub fn parent_point_id(doc_id: &str, unit_type: &str, unit_locator: &serde_json::Value) -> String {
    let namespace = Uuid::NAMESPACE_URL;
    let input = format!("parent:{}:{}:{}", doc_id, unit_type, unit_locator);
    let uuid = Uuid::new_v5(&namespace, input.as_bytes());
    uuid.to_string().replace('-', "")
}

/// Create a parent record for a PDF page.
pub fn pdf_page_parent(
    doc_id: &str,
    doc_fingerprint: &str,
    page_number: usize,
    page_text: &str,
) -> ParentUnitRecord {
    let unit_locator = serde_json::json!({ "page": page_number });
    let point_id = parent_point_id(doc_id, UNIT_TYPE_PDF_PAGE, &unit_locator);
    ParentUnitRecord {
        point_id,
        doc_id: doc_id.to_string(),
        doc_fingerprint: doc_fingerprint.to_string(),
        unit_type: UNIT_TYPE_PDF_PAGE.to_string(),
        unit_locator,
        unit_text: page_text.to_string(),
        unit_char_len: page_text.len(),
        unit_hash: sha256_hex(page_text),
    }
}

/// Create a parent record for an EPUB section/chapter.
pub fn epub_section_parent(
    doc_id: &str,
    doc_fingerprint: &str,
    spine_id: &str,
    chapter_title: Option<&str>,
    section_text: &str,
) -> ParentUnitRecord {
    let mut locator = serde_json::json!({ "spine_id": spine_id });
    if let Some(title) = chapter_title {
        locator["chapter_title"] = serde_json::Value::String(title.to_string());
    }
    let point_id = parent_point_id(doc_id, UNIT_TYPE_EPUB_SECTION, &locator);
    ParentUnitRecord {
        point_id,
        doc_id: doc_id.to_string(),
        doc_fingerprint: doc_fingerprint.to_string(),
        unit_type: UNIT_TYPE_EPUB_SECTION.to_string(),
        unit_locator: locator,
        unit_text: section_text.to_string(),
        unit_char_len: section_text.len(),
        unit_hash: sha256_hex(section_text),
    }
}

/// Create a parent record for a code file.
pub fn code_file_parent(
    doc_id: &str,
    doc_fingerprint: &str,
    file_path: &str,
    file_text: &str,
) -> ParentUnitRecord {
    let unit_locator = serde_json::json!({ "file_path": file_path });
    let point_id = parent_point_id(doc_id, UNIT_TYPE_CODE_FILE, &unit_locator);
    ParentUnitRecord {
        point_id,
        doc_id: doc_id.to_string(),
        doc_fingerprint: doc_fingerprint.to_string(),
        unit_type: UNIT_TYPE_CODE_FILE.to_string(),
        unit_locator,
        unit_text: file_text.to_string(),
        unit_char_len: file_text.len(),
        unit_hash: sha256_hex(file_text),
    }
}

/// Create a parent record for a text section (markdown heading, etc.).
pub fn text_section_parent(
    doc_id: &str,
    doc_fingerprint: &str,
    section_title: &str,
    section_index: usize,
    section_text: &str,
) -> ParentUnitRecord {
    let unit_locator = serde_json::json!({
        "section_title": section_title,
        "section_index": section_index,
    });
    let point_id = parent_point_id(doc_id, UNIT_TYPE_TEXT_SECTION, &unit_locator);
    ParentUnitRecord {
        point_id,
        doc_id: doc_id.to_string(),
        doc_fingerprint: doc_fingerprint.to_string(),
        unit_type: UNIT_TYPE_TEXT_SECTION.to_string(),
        unit_locator,
        unit_text: section_text.to_string(),
        unit_char_len: section_text.len(),
        unit_hash: sha256_hex(section_text),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_page_parent_creation() {
        let parent = pdf_page_parent(
            "doc-123",
            "fingerprint-abc",
            5,
            "Page five content here.",
        );
        assert_eq!(parent.doc_id, "doc-123");
        assert_eq!(parent.doc_fingerprint, "fingerprint-abc");
        assert_eq!(parent.unit_type, UNIT_TYPE_PDF_PAGE);
        assert_eq!(parent.unit_locator["page"], 5);
        assert_eq!(parent.unit_text, "Page five content here.");
        assert_eq!(parent.unit_char_len, 23);
        assert!(!parent.unit_hash.is_empty());
        assert!(!parent.point_id.is_empty());
    }

    #[test]
    fn test_epub_section_parent_creation() {
        let parent = epub_section_parent(
            "doc-456",
            "fp-xyz",
            "ch3",
            Some("Chapter Three"),
            "The content of chapter three.",
        );
        assert_eq!(parent.unit_type, UNIT_TYPE_EPUB_SECTION);
        assert_eq!(parent.unit_locator["spine_id"], "ch3");
        assert_eq!(parent.unit_locator["chapter_title"], "Chapter Three");
    }

    #[test]
    fn test_epub_section_no_title() {
        let parent = epub_section_parent(
            "doc-456",
            "fp-xyz",
            "ch1",
            None,
            "Untitled chapter content.",
        );
        assert_eq!(parent.unit_locator["spine_id"], "ch1");
        assert!(parent.unit_locator.get("chapter_title").is_none());
    }

    #[test]
    fn test_code_file_parent_creation() {
        let parent = code_file_parent(
            "doc-789",
            "fp-code",
            "src/main.rs",
            "fn main() { println!(\"hello\"); }",
        );
        assert_eq!(parent.unit_type, UNIT_TYPE_CODE_FILE);
        assert_eq!(parent.unit_locator["file_path"], "src/main.rs");
    }

    #[test]
    fn test_text_section_parent_creation() {
        let parent = text_section_parent(
            "doc-txt",
            "fp-text",
            "Introduction",
            0,
            "This is the introduction.",
        );
        assert_eq!(parent.unit_type, UNIT_TYPE_TEXT_SECTION);
        assert_eq!(parent.unit_locator["section_title"], "Introduction");
        assert_eq!(parent.unit_locator["section_index"], 0);
    }

    #[test]
    fn test_unit_hash_deterministic() {
        let text = "Deterministic hashing test.";
        let hash1 = sha256_hex(text);
        let hash2 = sha256_hex(text);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA256 hex = 64 chars
    }

    #[test]
    fn test_unit_hash_changes_with_content() {
        let hash1 = sha256_hex("Version 1");
        let hash2 = sha256_hex("Version 2");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_parent_point_id_deterministic() {
        let locator = serde_json::json!({"page": 1});
        let id1 = parent_point_id("doc-1", UNIT_TYPE_PDF_PAGE, &locator);
        let id2 = parent_point_id("doc-1", UNIT_TYPE_PDF_PAGE, &locator);
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 32); // UUID hex without dashes
    }

    #[test]
    fn test_parent_point_id_unique_across_pages() {
        let loc1 = serde_json::json!({"page": 1});
        let loc2 = serde_json::json!({"page": 2});
        let id1 = parent_point_id("doc-1", UNIT_TYPE_PDF_PAGE, &loc1);
        let id2 = parent_point_id("doc-1", UNIT_TYPE_PDF_PAGE, &loc2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_parent_point_id_unique_across_unit_types() {
        let locator = serde_json::json!({"page": 1});
        let id1 = parent_point_id("doc-1", UNIT_TYPE_PDF_PAGE, &locator);
        let id2 = parent_point_id("doc-1", UNIT_TYPE_EPUB_SECTION, &locator);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_to_payload_fields() {
        let parent = pdf_page_parent("doc-1", "fp-1", 3, "Page three.");
        let payload = parent.to_payload("my-lib", Some("My Doc"), "page_based", "pdf");

        assert_eq!(payload["library_name"], "my-lib");
        assert_eq!(payload["record_type"], RECORD_TYPE_PARENT);
        assert_eq!(payload["doc_id"], "doc-1");
        assert_eq!(payload["doc_fingerprint"], "fp-1");
        assert_eq!(payload["doc_type"], "page_based");
        assert_eq!(payload["source_format"], "pdf");
        assert_eq!(payload["doc_title"], "My Doc");
        assert_eq!(payload["unit_type"], UNIT_TYPE_PDF_PAGE);
        assert_eq!(payload["chunk_text_raw"], "Page three.");
        assert!(payload.contains_key("unit_hash"));
        assert!(payload.contains_key("unit_locator"));
    }

    #[test]
    fn test_to_payload_without_title() {
        let parent = pdf_page_parent("doc-1", "fp-1", 1, "Content.");
        let payload = parent.to_payload("lib", None, "page_based", "pdf");
        assert!(!payload.contains_key("doc_title"));
    }
}

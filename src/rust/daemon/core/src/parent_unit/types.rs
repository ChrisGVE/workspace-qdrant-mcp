//! Core types, constants, and factory functions for parent-unit records.

use sha2::{Digest, Sha256};
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
pub fn parent_point_id(
    doc_id: &str,
    unit_type: &str,
    unit_locator: &serde_json::Value,
) -> String {
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

/// Create a parent record for a code block (class, struct, impl, trait, module, etc.).
///
/// Block-level parents sit between the file parent and individual method/function chunks.
/// Methods within a class reference the block parent; the block parent references the file parent.
pub fn code_block_parent(
    doc_id: &str,
    doc_fingerprint: &str,
    file_path: &str,
    block_name: &str,
    block_kind: &str,
    start_line: usize,
    end_line: usize,
    block_text: &str,
) -> ParentUnitRecord {
    let unit_locator = serde_json::json!({
        "file_path": file_path,
        "block_name": block_name,
        "block_kind": block_kind,
        "start_line": start_line,
        "end_line": end_line,
    });
    let point_id = parent_point_id(doc_id, UNIT_TYPE_CODE_BLOCK, &unit_locator);
    ParentUnitRecord {
        point_id,
        doc_id: doc_id.to_string(),
        doc_fingerprint: doc_fingerprint.to_string(),
        unit_type: UNIT_TYPE_CODE_BLOCK.to_string(),
        unit_locator,
        unit_text: block_text.to_string(),
        unit_char_len: block_text.len(),
        unit_hash: sha256_hex(block_text),
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

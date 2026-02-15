//! Payload field definitions for the `libraries` Qdrant collection.
//!
//! The libraries collection stores reference documentation and library content,
//! isolated by `library_name`.

use crate::schema::FieldDef;

// Core identification (library_name is the tenant key)
pub const LIBRARY_NAME: FieldDef = FieldDef::categorical("library_name");
pub const DOCUMENT_ID: FieldDef = FieldDef::categorical("document_id");
pub const ITEM_TYPE: FieldDef = FieldDef::categorical("item_type");

// Content
pub const CONTENT: FieldDef = FieldDef::content("content");
pub const SOURCE_TYPE: FieldDef = FieldDef::categorical("source_type");
pub const MAIN_TAG: FieldDef = FieldDef::categorical("main_tag");
pub const FULL_TAG: FieldDef = FieldDef::categorical("full_tag");

// Library metadata
pub const BRANCH: FieldDef = FieldDef::categorical("branch");

// Document provenance fields
pub const DOC_ID: FieldDef = FieldDef::categorical("doc_id");
pub const DOC_TITLE: FieldDef = FieldDef::content("doc_title");
pub const DOC_AUTHORS: FieldDef = FieldDef::content("doc_authors");
pub const DOC_SOURCE: FieldDef = FieldDef::content("doc_source");
pub const DOC_FINGERPRINT: FieldDef = FieldDef::categorical("doc_fingerprint");
pub const DOC_VERSION: FieldDef = FieldDef::content("doc_version");
pub const DOC_TYPE: FieldDef = FieldDef::categorical("doc_type");
pub const SOURCE_FORMAT: FieldDef = FieldDef::categorical("source_format");
pub const EXTRACTOR: FieldDef = FieldDef::content("extractor");

// Chunk position within document
pub const LOCATOR: FieldDef = FieldDef::content("locator");
pub const CHAR_START: FieldDef = FieldDef::content("char_start");
pub const CHAR_END: FieldDef = FieldDef::content("char_end");

// Parent-child architecture
pub const RECORD_TYPE: FieldDef = FieldDef::categorical("record_type");
pub const PARENT_UNIT_ID: FieldDef = FieldDef::categorical("parent_unit_id");
pub const UNIT_TYPE: FieldDef = FieldDef::categorical("unit_type");
pub const UNIT_LOCATOR: FieldDef = FieldDef::content("unit_locator");
pub const UNIT_HASH: FieldDef = FieldDef::categorical("unit_hash");
pub const CHUNK_TEXT_RAW: FieldDef = FieldDef::content("chunk_text_raw");
pub const CHUNK_TEXT_INDEXED: FieldDef = FieldDef::content("chunk_text_indexed");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_names_match_daemon() {
        assert_eq!(LIBRARY_NAME.name, "library_name");
        assert_eq!(CONTENT.name, "content");
        assert_eq!(DOCUMENT_ID.name, "document_id");
    }

    #[test]
    fn provenance_field_names() {
        assert_eq!(DOC_ID.name, "doc_id");
        assert_eq!(DOC_TITLE.name, "doc_title");
        assert_eq!(DOC_AUTHORS.name, "doc_authors");
        assert_eq!(DOC_SOURCE.name, "doc_source");
        assert_eq!(DOC_FINGERPRINT.name, "doc_fingerprint");
        assert_eq!(DOC_VERSION.name, "doc_version");
        assert_eq!(DOC_TYPE.name, "doc_type");
        assert_eq!(SOURCE_FORMAT.name, "source_format");
        assert_eq!(EXTRACTOR.name, "extractor");
        assert_eq!(LOCATOR.name, "locator");
        assert_eq!(CHAR_START.name, "char_start");
        assert_eq!(CHAR_END.name, "char_end");
    }

    #[test]
    fn parent_child_field_names() {
        assert_eq!(RECORD_TYPE.name, "record_type");
        assert_eq!(PARENT_UNIT_ID.name, "parent_unit_id");
        assert_eq!(UNIT_TYPE.name, "unit_type");
        assert_eq!(UNIT_LOCATOR.name, "unit_locator");
        assert_eq!(UNIT_HASH.name, "unit_hash");
        assert_eq!(CHUNK_TEXT_RAW.name, "chunk_text_raw");
        assert_eq!(CHUNK_TEXT_INDEXED.name, "chunk_text_indexed");
    }
}

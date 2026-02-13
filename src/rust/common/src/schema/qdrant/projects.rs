//! Payload field definitions for the `projects` Qdrant collection.
//!
//! The projects collection stores code chunks from all registered projects,
//! isolated by `tenant_id`.

use crate::schema::FieldDef;

// Core identification
pub const TENANT_ID: FieldDef = FieldDef::categorical("tenant_id");
pub const DOCUMENT_ID: FieldDef = FieldDef::categorical("document_id");
pub const ITEM_TYPE: FieldDef = FieldDef::categorical("item_type");
pub const DOCUMENT_TYPE: FieldDef = FieldDef::categorical("document_type");

// File metadata
pub const FILE_PATH: FieldDef = FieldDef::content("file_path");
pub const FILE_TYPE: FieldDef = FieldDef::categorical("file_type");
pub const BRANCH: FieldDef = FieldDef::categorical("branch");

// Chunk data
pub const CONTENT: FieldDef = FieldDef::content("content");
pub const CHUNK_INDEX: FieldDef = FieldDef::categorical("chunk_index");

// Tree-sitter chunk metadata (prefixed with "chunk_" in payload)
pub const CHUNK_CHUNK_TYPE: FieldDef = FieldDef::categorical("chunk_chunk_type");
pub const CHUNK_SYMBOL_NAME: FieldDef = FieldDef::content("chunk_symbol_name");
pub const CHUNK_START_LINE: FieldDef = FieldDef::categorical("chunk_start_line");
pub const CHUNK_END_LINE: FieldDef = FieldDef::categorical("chunk_end_line");

// Content ingestion (non-file)
pub const SOURCE_TYPE: FieldDef = FieldDef::categorical("source_type");
pub const MAIN_TAG: FieldDef = FieldDef::categorical("main_tag");
pub const FULL_TAG: FieldDef = FieldDef::categorical("full_tag");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_names_match_daemon() {
        assert_eq!(TENANT_ID.name, "tenant_id");
        assert_eq!(FILE_PATH.name, "file_path");
        assert_eq!(CONTENT.name, "content");
        assert_eq!(CHUNK_SYMBOL_NAME.name, "chunk_symbol_name");
    }
}

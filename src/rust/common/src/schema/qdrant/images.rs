//! Payload field definitions for the `images` Qdrant collection.
//!
//! The images collection stores CLIP-embedded images extracted from documents,
//! using 512-dimensional dense vectors (ViT-B-32) with no sparse vectors.
//! Images are linked back to their source documents via `source_document_id`.

use crate::schema::FieldDef;

// Core identification
pub const TENANT_ID: FieldDef = FieldDef::categorical("tenant_id");
pub const DOCUMENT_ID: FieldDef = FieldDef::categorical("document_id");

// Source provenance
pub const SOURCE_DOCUMENT_ID: FieldDef = FieldDef::categorical("source_document_id");
pub const SOURCE_COLLECTION: FieldDef = FieldDef::categorical("source_collection");
pub const FILE_PATH: FieldDef = FieldDef::content("file_path");

// Position within source document
pub const PAGE_NUMBER: FieldDef = FieldDef::categorical("page_number");
pub const SECTION: FieldDef = FieldDef::content("section");
pub const IMAGE_INDEX: FieldDef = FieldDef::categorical("image_index");

// Image metadata
pub const IMAGE_WIDTH: FieldDef = FieldDef::categorical("image_width");
pub const IMAGE_HEIGHT: FieldDef = FieldDef::categorical("image_height");
pub const IMAGE_FORMAT: FieldDef = FieldDef::categorical("image_format");

// Content
pub const THUMBNAIL_B64: FieldDef = FieldDef::content("thumbnail_b64");
pub const OCR_TEXT: FieldDef = FieldDef::content("ocr_text");
pub const ALT_TEXT: FieldDef = FieldDef::content("alt_text");

// Timestamps
pub const INGESTION_TIMESTAMP: FieldDef = FieldDef::categorical("ingestion_timestamp");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_names_match_constants() {
        assert_eq!(TENANT_ID.name, "tenant_id");
        assert_eq!(DOCUMENT_ID.name, "document_id");
        assert_eq!(SOURCE_DOCUMENT_ID.name, "source_document_id");
        assert_eq!(SOURCE_COLLECTION.name, "source_collection");
        assert_eq!(FILE_PATH.name, "file_path");
    }

    #[test]
    fn image_metadata_field_names() {
        assert_eq!(PAGE_NUMBER.name, "page_number");
        assert_eq!(SECTION.name, "section");
        assert_eq!(IMAGE_INDEX.name, "image_index");
        assert_eq!(IMAGE_WIDTH.name, "image_width");
        assert_eq!(IMAGE_HEIGHT.name, "image_height");
        assert_eq!(IMAGE_FORMAT.name, "image_format");
    }

    #[test]
    fn content_field_names() {
        assert_eq!(THUMBNAIL_B64.name, "thumbnail_b64");
        assert_eq!(OCR_TEXT.name, "ocr_text");
        assert_eq!(ALT_TEXT.name, "alt_text");
        assert_eq!(INGESTION_TIMESTAMP.name, "ingestion_timestamp");
    }
}

//! Shared constants used across daemon and CLI
//!
//! Single source of truth for collection names, default URLs, ports, branches,
//! item type strings, and operation strings.

/// Projects collection - stores code and documents from all projects
/// Filtered by project_id payload field
pub const COLLECTION_PROJECTS: &str = "projects";

/// Libraries collection - stores library documentation
/// Filtered by library_name payload field
pub const COLLECTION_LIBRARIES: &str = "libraries";

/// Memory collection - stores agent memory and cross-project notes
pub const COLLECTION_MEMORY: &str = "memory";

/// Scratchpad collection - persistent LLM scratch space
/// Filtered by tenant_id payload field (_global_ or project_id)
pub const COLLECTION_SCRATCHPAD: &str = "scratchpad";

/// Images collection - stores CLIP-embedded images from documents
/// 512-dimensional vectors (CLIP ViT-B-32), dense-only (no sparse)
pub const COLLECTION_IMAGES: &str = "images";

/// Default Qdrant server URL
pub const DEFAULT_QDRANT_URL: &str = "http://localhost:6333";

/// Default gRPC port for the daemon
pub const DEFAULT_GRPC_PORT: u16 = 50051;

/// Default Git branch name
pub const DEFAULT_BRANCH: &str = "main";

/// Queue priority constants
/// Lower number = higher priority in processing order
pub mod priority {
    /// HIGH priority: Active agent sessions - items processed first
    pub const HIGH: i32 = 1;
    /// NORMAL priority: Registered projects without active sessions
    pub const NORMAL: i32 = 3;
    /// LOW priority: Background/inactive projects
    pub const LOW: i32 = 5;
}

/// String constants for ItemType enum values.
///
/// Every consumer must import from here rather than using string literals.
pub mod item_type {
    /// Direct text content (scratchbook, notes, memory rules)
    pub const TEXT: &str = "text";
    /// Single file with path reference
    pub const FILE: &str = "file";
    /// URL fetch (individual web page)
    pub const URL: &str = "url";
    /// Entire website (multi-page crawl)
    pub const WEBSITE: &str = "website";
    /// Generalized document reference (for delete-by-ID, uplift-by-ID)
    pub const DOC: &str = "doc";
    /// Directory
    pub const FOLDER: &str = "folder";
    /// Project or library tenant (collection field disambiguates)
    pub const TENANT: &str = "tenant";
    /// A Qdrant collection itself
    pub const COLLECTION: &str = "collection";
}

/// Qdrant payload field name constants.
///
/// Single source of truth for field names used in Qdrant point payloads.
/// Every consumer must import from here rather than using string literals.
pub mod field {
    /// Multi-tenant isolation key
    pub const TENANT_ID: &str = "tenant_id";
    /// Project scoping within memory collection
    pub const PROJECT_ID: &str = "project_id";
    /// Library scoping key
    pub const LIBRARY_NAME: &str = "library_name";
    /// Library hierarchical path (e.g., "cs/design_patterns")
    pub const LIBRARY_PATH: &str = "library_path";
    /// Source project ID for format-routed documents
    pub const SOURCE_PROJECT_ID: &str = "source_project_id";
    /// Routing reason (e.g., "format_based")
    pub const ROUTING_REASON: &str = "routing_reason";
    /// Instance-aware filtering (base point IDs)
    pub const BASE_POINT: &str = "base_point";
    /// Git branch filter
    pub const BRANCH: &str = "branch";
    /// File extension discriminator
    pub const FILE_TYPE: &str = "file_type";
    /// File path for glob matching
    pub const FILE_PATH: &str = "file_path";
    /// Concept tags for tag-based filtering
    pub const CONCEPT_TAGS: &str = "concept_tags";
    /// Soft-delete flag
    pub const DELETED: &str = "deleted";
    /// Document text content
    pub const CONTENT: &str = "content";
    /// Document title
    pub const TITLE: &str = "title";
    /// Content source classification
    pub const SOURCE_TYPE: &str = "source_type";
    /// Document identifier
    pub const DOCUMENT_ID: &str = "document_id";
    /// File/content type discriminator
    pub const ITEM_TYPE: &str = "item_type";
    /// Parent unit link for chunked documents
    pub const PARENT_UNIT_ID: &str = "parent_unit_id";

    // Image-specific fields (images collection)
    /// Source document identifier for the image
    pub const SOURCE_DOCUMENT_ID: &str = "source_document_id";
    /// Source collection (projects or libraries)
    pub const SOURCE_COLLECTION: &str = "source_collection";
    /// Page number within document (PDFs)
    pub const PAGE_NUMBER: &str = "page_number";
    /// Section or chapter name (EPUBs)
    pub const SECTION: &str = "section";
    /// Image index within page/section
    pub const IMAGE_INDEX: &str = "image_index";
    /// Image width in pixels
    pub const IMAGE_WIDTH: &str = "image_width";
    /// Image height in pixels
    pub const IMAGE_HEIGHT: &str = "image_height";
    /// Image format (JPEG, PNG, etc.)
    pub const IMAGE_FORMAT: &str = "image_format";
    /// Base64-encoded 64x64 thumbnail
    pub const THUMBNAIL_B64: &str = "thumbnail_b64";
    /// OCR-extracted text from image
    pub const OCR_TEXT: &str = "ocr_text";
    /// Alt text from HTML/EPUB source
    pub const ALT_TEXT: &str = "alt_text";
    /// Ingestion timestamp
    pub const INGESTION_TIMESTAMP: &str = "ingestion_timestamp";
}

/// String constants for QueueOperation enum values.
///
/// Every consumer must import from here rather than using string literals.
pub mod operation {
    /// Create/add new content
    pub const ADD: &str = "add";
    /// Modify existing content; tenant re-sync
    pub const UPDATE: &str = "update";
    /// Remove content
    pub const DELETE: &str = "delete";
    /// Enumerate contents (folders, tenants, websites)
    pub const SCAN: &str = "scan";
    /// Move/rename path or tenant_id
    pub const RENAME: &str = "rename";
    /// Metadata enrichment cascade (collection → tenant → doc)
    pub const UPLIFT: &str = "uplift";
    /// Clear all content in a collection (not delete the collection)
    pub const RESET: &str = "reset";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_names() {
        assert_eq!(COLLECTION_PROJECTS, "projects");
        assert_eq!(COLLECTION_LIBRARIES, "libraries");
        assert_eq!(COLLECTION_MEMORY, "memory");
        assert_eq!(COLLECTION_SCRATCHPAD, "scratchpad");
        assert_eq!(COLLECTION_IMAGES, "images");
    }

    #[test]
    fn test_defaults() {
        assert_eq!(DEFAULT_QDRANT_URL, "http://localhost:6333");
        assert_eq!(DEFAULT_GRPC_PORT, 50051);
        assert_eq!(DEFAULT_BRANCH, "main");
    }

    #[test]
    fn test_priority_ordering() {
        assert!(priority::HIGH < priority::NORMAL);
        assert!(priority::NORMAL < priority::LOW);
    }

    #[test]
    fn test_item_type_constants() {
        assert_eq!(item_type::TEXT, "text");
        assert_eq!(item_type::FILE, "file");
        assert_eq!(item_type::URL, "url");
        assert_eq!(item_type::WEBSITE, "website");
        assert_eq!(item_type::DOC, "doc");
        assert_eq!(item_type::FOLDER, "folder");
        assert_eq!(item_type::TENANT, "tenant");
        assert_eq!(item_type::COLLECTION, "collection");
    }

    #[test]
    fn test_field_constants() {
        assert_eq!(field::TENANT_ID, "tenant_id");
        assert_eq!(field::PROJECT_ID, "project_id");
        assert_eq!(field::LIBRARY_NAME, "library_name");
        assert_eq!(field::BASE_POINT, "base_point");
        assert_eq!(field::BRANCH, "branch");
        assert_eq!(field::FILE_TYPE, "file_type");
        assert_eq!(field::FILE_PATH, "file_path");
        assert_eq!(field::CONCEPT_TAGS, "concept_tags");
        assert_eq!(field::DELETED, "deleted");
        assert_eq!(field::CONTENT, "content");
        assert_eq!(field::TITLE, "title");
        assert_eq!(field::SOURCE_TYPE, "source_type");
        assert_eq!(field::DOCUMENT_ID, "document_id");
        assert_eq!(field::ITEM_TYPE, "item_type");
        assert_eq!(field::PARENT_UNIT_ID, "parent_unit_id");
    }

    #[test]
    fn test_image_field_constants() {
        assert_eq!(field::SOURCE_DOCUMENT_ID, "source_document_id");
        assert_eq!(field::SOURCE_COLLECTION, "source_collection");
        assert_eq!(field::PAGE_NUMBER, "page_number");
        assert_eq!(field::SECTION, "section");
        assert_eq!(field::IMAGE_INDEX, "image_index");
        assert_eq!(field::IMAGE_WIDTH, "image_width");
        assert_eq!(field::IMAGE_HEIGHT, "image_height");
        assert_eq!(field::IMAGE_FORMAT, "image_format");
        assert_eq!(field::THUMBNAIL_B64, "thumbnail_b64");
        assert_eq!(field::OCR_TEXT, "ocr_text");
        assert_eq!(field::ALT_TEXT, "alt_text");
        assert_eq!(field::INGESTION_TIMESTAMP, "ingestion_timestamp");
    }

    #[test]
    fn test_operation_constants() {
        assert_eq!(operation::ADD, "add");
        assert_eq!(operation::UPDATE, "update");
        assert_eq!(operation::DELETE, "delete");
        assert_eq!(operation::SCAN, "scan");
        assert_eq!(operation::RENAME, "rename");
        assert_eq!(operation::UPLIFT, "uplift");
        assert_eq!(operation::RESET, "reset");
    }
}

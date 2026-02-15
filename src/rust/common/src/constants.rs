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

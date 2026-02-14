//! Shared constants used across daemon and CLI
//!
//! Single source of truth for collection names, default URLs, ports, and branches.

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
}

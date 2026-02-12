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

/// Default Qdrant server URL
pub const DEFAULT_QDRANT_URL: &str = "http://localhost:6333";

/// Default gRPC port for the daemon
pub const DEFAULT_GRPC_PORT: u16 = 50051;

/// Default Git branch name
pub const DEFAULT_BRANCH: &str = "main";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_names() {
        assert_eq!(COLLECTION_PROJECTS, "projects");
        assert_eq!(COLLECTION_LIBRARIES, "libraries");
        assert_eq!(COLLECTION_MEMORY, "memory");
    }

    #[test]
    fn test_defaults() {
        assert_eq!(DEFAULT_QDRANT_URL, "http://localhost:6333");
        assert_eq!(DEFAULT_GRPC_PORT, 50051);
        assert_eq!(DEFAULT_BRANCH, "main");
    }
}

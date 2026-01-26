//! Metadata Enrichment Module
//!
//! Implements metadata enrichment rules for different collection types according to
//! architectural decisions in Task 375. Enriches document metadata based on collection
//! type: PROJECT, LIBRARY, USER, or MEMORY.
//!
//! Collection Type Detection:
//! - PROJECT: _{project_id} where project_id is 12-char hex hash
//! - LIBRARY: _{library_name} where library_name is alphanumeric with hyphens
//! - USER: {basename}-{type} format
//! - MEMORY: exact match "memory"
//!
//! Metadata Enrichment Rules:
//! - PROJECT: project_id, branch, file_type
//! - USER: project_id only (no branch)
//! - LIBRARY: library_name (no project_id or branch)
//! - MEMORY: global metadata only (no project_id or branch)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use git2::Repository;
use tracing::{debug, warn};

use crate::file_classification::{classify_file_type};
use crate::watching_queue::{calculate_tenant_id, get_current_branch};

/// Collection type enumeration for metadata enrichment
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectionType {
    /// Project collection (_{project_id})
    Project {
        /// 12-character hex project ID
        project_id: String,
    },
    /// Library collection (_{library_name})
    Library {
        /// Library name (e.g., "fastapi", "pandas")
        library_name: String,
    },
    /// User collection ({basename}-{type})
    User {
        /// Collection basename
        basename: String,
        /// Collection type suffix
        collection_type: String,
    },
    /// Memory collection (exact match "memory")
    Memory,
}

impl CollectionType {
    /// Parse collection name to determine collection type
    ///
    /// # Arguments
    /// * `collection_name` - Name of the collection
    ///
    /// # Returns
    /// CollectionType enum variant
    ///
    /// # Examples
    /// ```
    /// use workspace_qdrant_daemon_core::metadata_enrichment::CollectionType;
    ///
    /// let ctype = CollectionType::from_name("_0f72d776622e");
    /// assert!(matches!(ctype, CollectionType::Project { .. }));
    ///
    /// let ctype = CollectionType::from_name("_fastapi");
    /// assert!(matches!(ctype, CollectionType::Library { .. }));
    ///
    /// let ctype = CollectionType::from_name("myapp-notes");
    /// assert!(matches!(ctype, CollectionType::User { .. }));
    ///
    /// let ctype = CollectionType::from_name("memory");
    /// assert!(matches!(ctype, CollectionType::Memory));
    /// ```
    pub fn from_name(collection_name: &str) -> Self {
        // Check for exact "memory" match
        if collection_name == "memory" {
            return CollectionType::Memory;
        }

        // Check for underscore prefix (PROJECT or LIBRARY)
        if let Some(name_without_underscore) = collection_name.strip_prefix('_') {
            // PROJECT collections are 12-character hex hashes
            if name_without_underscore.len() == 12
                && name_without_underscore.chars().all(|c| c.is_ascii_hexdigit()) {
                return CollectionType::Project {
                    project_id: name_without_underscore.to_string(),
                };
            }

            // LIBRARY collections are alphanumeric with hyphens/underscores
            return CollectionType::Library {
                library_name: name_without_underscore.to_string(),
            };
        }

        // USER collections have {basename}-{type} format
        if let Some(dash_pos) = collection_name.rfind('-') {
            let basename = collection_name[..dash_pos].to_string();
            let collection_type = collection_name[dash_pos + 1..].to_string();
            return CollectionType::User {
                basename,
                collection_type,
            };
        }

        // Fallback: treat as USER collection with no type suffix
        CollectionType::User {
            basename: collection_name.to_string(),
            collection_type: String::new(),
        }
    }
}

/// Find project root by traversing up from file path
///
/// Uses git2 Repository::discover() to find the nearest Git repository.
/// Falls back to file's parent directory if not in a Git repository.
///
/// # Arguments
/// * `file_path` - Path to a file within the project
///
/// # Returns
/// PathBuf pointing to project root directory
fn find_project_root(file_path: &Path) -> PathBuf {
    // Try to find Git repository root
    match Repository::discover(file_path) {
        Ok(repo) => {
            if let Some(workdir) = repo.workdir() {
                debug!(
                    "Found Git repository root for {}: {}",
                    file_path.display(),
                    workdir.display()
                );
                return workdir.to_path_buf();
            }
        }
        Err(e) => {
            debug!(
                "No Git repository found for {}: {}",
                file_path.display(),
                e
            );
        }
    }

    // Fallback: use file's parent directory or current directory
    file_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| {
            warn!(
                "Could not determine project root for {}, using current directory",
                file_path.display()
            );
            PathBuf::from(".")
        })
}

/// Enrich metadata according to collection type rules
///
/// # Arguments
/// * `collection_name` - Name of the target collection
/// * `file_path` - Path to the file being processed
/// * `base_metadata` - Base metadata to enrich (optional)
/// * `task_source` - Optional task source info for MCP vs CLI distinction
///
/// # Returns
/// HashMap<String, String> with enriched metadata
///
/// # Examples
/// ```no_run
/// use std::path::Path;
/// use std::collections::HashMap;
/// use workspace_qdrant_daemon_core::metadata_enrichment::enrich_metadata;
///
/// let metadata = enrich_metadata(
///     "_0f72d776622e",
///     Path::new("/project/src/main.rs"),
///     None,
///     None,
/// );
/// assert!(metadata.contains_key("project_id"));
/// assert!(metadata.contains_key("branch"));
/// assert!(metadata.contains_key("file_type"));
/// ```
pub fn enrich_metadata(
    collection_name: &str,
    file_path: &Path,
    base_metadata: Option<HashMap<String, String>>,
    task_source: Option<&str>,
) -> HashMap<String, String> {
    let mut metadata = base_metadata.unwrap_or_default();
    let collection_type = CollectionType::from_name(collection_name);

    debug!(
        "Enriching metadata for collection '{}' (type: {:?}), file: {}",
        collection_name,
        collection_type,
        file_path.display()
    );

    match collection_type {
        CollectionType::Project { project_id } => {
            // PROJECT collections get: project_id, branch, file_type
            let project_root = find_project_root(file_path);

            // Add project_id (from collection name)
            metadata.insert("project_id".to_string(), project_id);

            // Add branch
            let branch = get_current_branch(&project_root);
            metadata.insert("branch".to_string(), branch);

            // Add file_type
            let file_type = classify_file_type(file_path);
            metadata.insert("file_type".to_string(), file_type.as_str().to_string());

            debug!(
                "PROJECT collection metadata: project_id={}, branch={}, file_type={}",
                metadata.get("project_id").unwrap(),
                metadata.get("branch").unwrap(),
                metadata.get("file_type").unwrap()
            );
        }

        CollectionType::User { .. } => {
            // USER collections get: project_id ONLY (no branch)
            // MCP: auto-detect from current project
            // CLI: require explicit --project flag (passed in base_metadata)

            let should_auto_detect = task_source
                .map(|s| s.contains("Mcp") || s.contains("ProjectWatcher"))
                .unwrap_or(true); // Default to auto-detect if source unknown

            if should_auto_detect {
                // Auto-detect project_id from file path
                let project_root = find_project_root(file_path);
                let project_id = calculate_tenant_id(&project_root);
                metadata.insert("project_id".to_string(), project_id);

                debug!(
                    "USER collection (auto-detect): project_id={}",
                    metadata.get("project_id").unwrap()
                );
                        } else {
                            // CLI mode: project_id should have been passed in base_metadata
                            // If it's there, it's already in metadata from unwrap_or_default above
                            if metadata.contains_key("project_id") {
                                debug!(
                                    "USER collection (CLI with --project): project_id={}",
                                    metadata.get("project_id").unwrap()
                                );
                            } else {
                debug!("USER collection (CLI without --project): no project_id added");
                }
            }

            // Note: NO branch metadata for USER collections
        }

        CollectionType::Library { library_name } => {
            // LIBRARY collections get: library_name (no project_id or branch)
            metadata.insert("library_name".to_string(), library_name.clone());

            debug!(
                "LIBRARY collection metadata: library_name={}",
                library_name
            );

            // Note: NO project_id or branch for LIBRARY collections
        }

        CollectionType::Memory => {
            // MEMORY collection: global metadata only (no project_id or branch)
            metadata.insert("scope".to_string(), "global".to_string());

            debug!("MEMORY collection: global scope, no project metadata");

            // Note: NO project_id or branch for MEMORY collection
        }
    }

    metadata
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_parse_project_collection() {
        let ctype = CollectionType::from_name("_0f72d776622e");
        match ctype {
            CollectionType::Project { project_id } => {
                assert_eq!(project_id, "0f72d776622e");
            }
            _ => panic!("Expected Project collection type"),
        }
    }

    #[test]
    fn test_parse_library_collection() {
        let ctype = CollectionType::from_name("_fastapi");
        match ctype {
            CollectionType::Library { library_name } => {
                assert_eq!(library_name, "fastapi");
            }
            _ => panic!("Expected Library collection type"),
        }

        let ctype = CollectionType::from_name("_pandas-stubs");
        match ctype {
            CollectionType::Library { library_name } => {
                assert_eq!(library_name, "pandas-stubs");
            }
            _ => panic!("Expected Library collection type"),
        }
    }

    #[test]
    fn test_parse_user_collection() {
        let ctype = CollectionType::from_name("myapp-notes");
        match ctype {
            CollectionType::User { basename, collection_type } => {
                assert_eq!(basename, "myapp");
                assert_eq!(collection_type, "notes");
            }
            _ => panic!("Expected User collection type"),
        }

        let ctype = CollectionType::from_name("workspace-qdrant-docs");
        match ctype {
            CollectionType::User { basename, collection_type } => {
                assert_eq!(basename, "workspace-qdrant");
                assert_eq!(collection_type, "docs");
            }
            _ => panic!("Expected User collection type"),
        }
    }

    #[test]
    fn test_parse_memory_collection() {
        let ctype = CollectionType::from_name("memory");
        assert!(matches!(ctype, CollectionType::Memory));
    }

    #[test]
    fn test_parse_edge_cases() {
        // Single underscore (too short for project ID)
        let ctype = CollectionType::from_name("_");
        assert!(matches!(ctype, CollectionType::Library { .. }));

        // Underscore with non-hex characters (library)
        let ctype = CollectionType::from_name("_numpy123");
        assert!(matches!(ctype, CollectionType::Library { .. }));

        // No dash (user collection with no type)
        let ctype = CollectionType::from_name("mycollection");
        match ctype {
            CollectionType::User { basename, collection_type } => {
                assert_eq!(basename, "mycollection");
                assert_eq!(collection_type, "");
            }
            _ => panic!("Expected User collection type"),
        }
    }

    #[test]
    fn test_enrich_project_collection_metadata() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("main.py");
        fs::write(&test_file, "print('hello')").unwrap();

        let metadata = enrich_metadata(
            "_0f72d776622e",
            &test_file,
            None,
            None,
        );

        assert_eq!(metadata.get("project_id"), Some(&"0f72d776622e".to_string()));
        assert!(metadata.contains_key("branch"));
        assert_eq!(metadata.get("branch"), Some(&"main".to_string())); // No git repo
        assert_eq!(metadata.get("file_type"), Some(&"code".to_string()));
    }

    #[test]
    fn test_enrich_user_collection_auto_detect() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("notes.md");
        fs::write(&test_file, "# Notes").unwrap();

        // Simulate MCP source (auto-detect)
        let metadata = enrich_metadata(
            "myapp-notes",
            &test_file,
            None,
            Some("McpServer"),
        );

        assert!(metadata.contains_key("project_id"));
        assert!(!metadata.contains_key("branch")); // USER collections don't get branch
    }

    #[test]
    fn test_enrich_user_collection_cli_with_project() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("notes.md");
        fs::write(&test_file, "# Notes").unwrap();

        // Simulate CLI with explicit project_id
        let mut base = HashMap::new();
        base.insert("project_id".to_string(), "abc123".to_string());

        let metadata = enrich_metadata(
            "myapp-notes",
            &test_file,
            Some(base),
            Some("CliCommand"),
        );

        assert_eq!(metadata.get("project_id"), Some(&"abc123".to_string()));
        assert!(!metadata.contains_key("branch"));
    }

    #[test]
    fn test_enrich_user_collection_cli_without_project() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("notes.md");
        fs::write(&test_file, "# Notes").unwrap();

        // Simulate CLI without project_id
        let metadata = enrich_metadata(
            "myapp-notes",
            &test_file,
            None,
            Some("CliCommand"),
        );

        assert!(!metadata.contains_key("project_id"));
        assert!(!metadata.contains_key("branch"));
    }

    #[test]
    fn test_enrich_library_collection_metadata() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("fastapi_doc.md");
        fs::write(&test_file, "# FastAPI").unwrap();

        let metadata = enrich_metadata(
            "_fastapi",
            &test_file,
            None,
            None,
        );

        assert_eq!(metadata.get("library_name"), Some(&"fastapi".to_string()));
        assert!(!metadata.contains_key("project_id"));
        assert!(!metadata.contains_key("branch"));
        assert!(!metadata.contains_key("file_type"));
    }

    #[test]
    fn test_enrich_memory_collection_metadata() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("note.txt");
        fs::write(&test_file, "Note").unwrap();

        let metadata = enrich_metadata(
            "memory",
            &test_file,
            None,
            None,
        );

        assert_eq!(metadata.get("scope"), Some(&"global".to_string()));
        assert!(!metadata.contains_key("project_id"));
        assert!(!metadata.contains_key("branch"));
    }

    #[test]
    fn test_file_type_classification() {
        let temp_dir = tempdir().unwrap();

        // Test code file
        let py_file = temp_dir.path().join("main.py");
        fs::write(&py_file, "").unwrap();
        let metadata = enrich_metadata("_abc123def456", &py_file, None, None);
        assert_eq!(metadata.get("file_type"), Some(&"code".to_string()));

        // Test test file
        let test_file = temp_dir.path().join("test_main.py");
        fs::write(&test_file, "").unwrap();
        let metadata = enrich_metadata("_abc123def456", &test_file, None, None);
        assert_eq!(metadata.get("file_type"), Some(&"test".to_string()));

        // Test docs file
        let md_file = temp_dir.path().join("README.md");
        fs::write(&md_file, "").unwrap();
        let metadata = enrich_metadata("_abc123def456", &md_file, None, None);
        assert_eq!(metadata.get("file_type"), Some(&"docs".to_string()));
    }

    #[test]
    fn test_base_metadata_preservation() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "test").unwrap();

        let mut base = HashMap::new();
        base.insert("custom_field".to_string(), "custom_value".to_string());

        let metadata = enrich_metadata(
            "memory",
            &test_file,
            Some(base),
            None,
        );

        // Base metadata should be preserved
        assert_eq!(metadata.get("custom_field"), Some(&"custom_value".to_string()));
        // Memory-specific metadata should be added
        assert_eq!(metadata.get("scope"), Some(&"global".to_string()));
    }
}

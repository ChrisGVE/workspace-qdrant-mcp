//! Metadata enrichment logic for each collection type.
//!
//! Metadata Enrichment Rules:
//! - PROJECT: project_id, branch, file_type, extension, is_test
//! - USER: project_id only (no branch)
//! - LIBRARY: library_name (no project_id or branch)
//! - RULES: global metadata only (no project_id or branch)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use git2::Repository;
use tracing::{debug, warn};

use crate::file_classification::{classify_file_type, get_extension_for_storage, is_test_file};
use crate::watching_queue::get_current_branch;

use super::collection_type::CollectionType;

/// Find project root by traversing up from file path.
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
            debug!("No Git repository found for {}: {}", file_path.display(), e);
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

/// Enrich metadata according to collection type rules.
///
/// # Arguments
/// * `collection_name` - Name of the target collection
/// * `file_path` - Path to the file being processed
/// * `base_metadata` - Base metadata to enrich (optional)
/// * `task_source` - Optional task source info for MCP vs CLI distinction
/// * `tenant_id` - Pre-computed tenant_id from watch_folders (single source of truth).
///   When provided, used directly as `project_id` instead of re-deriving from filesystem.
///
/// # Returns
/// HashMap<String, String> with enriched metadata
///
/// # Examples
/// ```no_run
/// use std::path::Path;
/// use std::collections::HashMap;
/// use workspace_qdrant_core::metadata_enrichment::enrich_metadata;
///
/// let metadata = enrich_metadata(
///     "_0f72d776622e",
///     Path::new("/project/src/main.rs"),
///     None,
///     None,
///     Some("0f72d776622e"),
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
    tenant_id: Option<&str>,
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
            enrich_project(&mut metadata, file_path, project_id);
        }

        CollectionType::User { .. } => {
            enrich_user(&mut metadata, file_path, task_source, tenant_id);
        }

        CollectionType::Library { library_name } => {
            enrich_library(&mut metadata, library_name);
        }

        CollectionType::Rules => {
            enrich_rules(&mut metadata);
        }
    }

    metadata
}

/// Apply PROJECT collection enrichment: project_id, branch, file_type, extension, is_test.
fn enrich_project(metadata: &mut HashMap<String, String>, file_path: &Path, project_id: String) {
    let project_root = find_project_root(file_path);

    metadata.insert("project_id".to_string(), project_id);

    let branch = get_current_branch(&project_root);
    metadata.insert("branch".to_string(), branch);

    let file_type = classify_file_type(file_path);
    metadata.insert("file_type".to_string(), file_type.as_str().to_string());

    if let Some(ext) = get_extension_for_storage(file_path) {
        metadata.insert("extension".to_string(), ext);
    }

    metadata.insert("is_test".to_string(), is_test_file(file_path).to_string());

    debug!(
        "PROJECT collection metadata: project_id={}, branch={}, file_type={}, is_test={}",
        metadata.get("project_id").unwrap(),
        metadata.get("branch").unwrap(),
        metadata.get("file_type").unwrap(),
        metadata.get("is_test").unwrap()
    );
}

/// Apply USER collection enrichment: project_id only (no branch).
fn enrich_user(
    metadata: &mut HashMap<String, String>,
    file_path: &Path,
    task_source: Option<&str>,
    tenant_id: Option<&str>,
) {
    let should_auto_detect = task_source
        .map(|s| s.contains("Mcp") || s.contains("ProjectWatcher"))
        .unwrap_or(true); // Default to auto-detect if source unknown

    if should_auto_detect {
        // Use pre-computed tenant_id if available (single source of truth),
        // otherwise fall back to filesystem detection
        let project_id = if let Some(tid) = tenant_id {
            tid.to_string()
        } else {
            let project_root = find_project_root(file_path);
            wqm_common::project_id::calculate_tenant_id(&project_root)
        };
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

/// Apply LIBRARY collection enrichment: library_name only.
fn enrich_library(metadata: &mut HashMap<String, String>, library_name: String) {
    metadata.insert("library_name".to_string(), library_name.clone());

    debug!("LIBRARY collection metadata: library_name={}", library_name);

    // Note: NO project_id or branch for LIBRARY collections
}

/// Apply RULES collection enrichment: global scope only.
fn enrich_rules(metadata: &mut HashMap<String, String>) {
    metadata.insert("scope".to_string(), "global".to_string());

    debug!("RULES collection: global scope, no project metadata");

    // Note: NO project_id or branch for RULES collection
}

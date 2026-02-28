//! Multi-tenant collection routing and input validation
//!
//! Routes content to canonical collections based on tenant ID format:
//! - `memory`, `agent_memory` -> Direct `memory` collection
//! - Project ID patterns (path hashes, git URLs) -> `projects` collection
//! - Human-readable names -> `libraries` collection

use tonic::Status;
use uuid::Uuid;
use wqm_common::constants::{COLLECTION_PROJECTS, COLLECTION_LIBRARIES};

/// Validate collection name format.
/// Rules: 3-255 chars, alphanumeric + underscore/hyphen, no leading numbers.
pub(crate) fn validate_collection_name(name: &str) -> Result<(), Status> {
    if name.is_empty() {
        return Err(Status::invalid_argument("Collection name cannot be empty"));
    }

    if name.len() < 3 {
        return Err(Status::invalid_argument(
            "Collection name must be at least 3 characters"
        ));
    }

    if name.len() > 255 {
        return Err(Status::invalid_argument(
            "Collection name must not exceed 255 characters"
        ));
    }

    if name.chars().next().map(|c| c.is_numeric()).unwrap_or(false) {
        return Err(Status::invalid_argument(
            "Collection name cannot start with a number"
        ));
    }

    if !name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
        return Err(Status::invalid_argument(
            "Collection name can only contain alphanumeric characters, underscores, and hyphens"
        ));
    }

    Ok(())
}

/// Check if tenant_id is a project ID format.
///
/// Project IDs are generated from:
/// 1. Git remote URLs (sanitized): e.g., "github_com_user_repo"
/// 2. Path hashes: e.g., "path_abc123def456789a" (21 chars: "path_" + 16 hex)
///
/// Library names are human-readable without these patterns: e.g., "react", "numpy"
pub(crate) fn is_project_id(tenant_id: &str) -> bool {
    // Path hash format: "path_" + 16 hex characters = 21 chars total
    if tenant_id.starts_with("path_") && tenant_id.len() == 21 {
        let hash_part = &tenant_id[5..];
        if hash_part.chars().all(|c| c.is_ascii_hexdigit()) {
            return true;
        }
    }

    // Sanitized git remote URLs contain domain patterns
    let domain_patterns = [
        "github_com_",
        "gitlab_com_",
        "bitbucket_org_",
        "codeberg_org_",
        "sr_ht_",
        "git_",
    ];

    for pattern in domain_patterns {
        if tenant_id.starts_with(pattern) {
            return true;
        }
    }

    // Also match pattern: domain_tld_user_repo (contains at least 3 underscores)
    let underscore_count = tenant_id.chars().filter(|c| *c == '_').count();
    if underscore_count >= 3 && tenant_id.contains("_com_") {
        return true;
    }

    false
}

/// Validate collection basename format (same rules as collection name).
fn validate_collection_basename(basename: &str) -> Result<(), Status> {
    if basename.is_empty() {
        return Err(Status::invalid_argument("Collection basename cannot be empty"));
    }
    if basename.len() < 3 {
        return Err(Status::invalid_argument(
            "Collection basename must be at least 3 characters",
        ));
    }
    if basename.len() > 255 {
        return Err(Status::invalid_argument(
            "Collection basename must not exceed 255 characters",
        ));
    }
    if basename.chars().next().map(|c| c.is_numeric()).unwrap_or(false) {
        return Err(Status::invalid_argument(
            "Collection basename cannot start with a number",
        ));
    }
    if !basename.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
        return Err(Status::invalid_argument(
            "Collection basename can only contain alphanumeric characters, underscores, and hyphens",
        ));
    }
    Ok(())
}

/// Determine the target collection and tenant metadata for multi-tenant routing.
///
/// Returns: (collection_name, tenant_type, tenant_value)
/// - tenant_type: "project_id" or "library_name"
pub(crate) fn determine_collection_routing(
    basename: &str,
    tenant_id: &str,
) -> Result<(String, String, String), Status> {
    validate_collection_basename(basename)?;

    if tenant_id.is_empty() {
        return Err(Status::invalid_argument("Tenant ID cannot be empty"));
    }

    // Memory collection uses single canonical name with tenant isolation via metadata
    if basename == "memory" || basename == "agent_memory" {
        return Ok(("memory".to_string(), "project_id".to_string(), tenant_id.to_string()));
    }

    if is_project_id(tenant_id) {
        Ok((
            COLLECTION_PROJECTS.to_string(),
            "project_id".to_string(),
            tenant_id.to_string(),
        ))
    } else {
        Ok((
            COLLECTION_LIBRARIES.to_string(),
            "library_name".to_string(),
            tenant_id.to_string(),
        ))
    }
}

/// Validate document ID format (must be a valid UUID).
pub(crate) fn validate_document_id(id: &str) -> Result<(), Status> {
    if id.is_empty() {
        return Err(Status::invalid_argument("Document ID cannot be empty"));
    }

    Uuid::parse_str(id).map_err(|_| {
        Status::invalid_argument("Document ID must be a valid UUID")
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_collection_name() {
        assert!(validate_collection_name("memory_tenant1").is_ok());
        assert!(validate_collection_name("scratchbook_user123").is_ok());
        assert!(validate_collection_name("notes-project").is_ok());
        assert!(validate_collection_name("_projects").is_ok());
        assert!(validate_collection_name("_libraries").is_ok());

        assert!(validate_collection_name("projects").is_ok());
        assert!(validate_collection_name("libraries").is_ok());
        assert!(validate_collection_name("memory").is_ok());

        assert!(validate_collection_name("ab").is_err());
        assert!(validate_collection_name("1memory").is_err());
        assert!(validate_collection_name("memory@tenant").is_err());
        assert!(validate_collection_name("memory.tenant").is_err());
        assert!(validate_collection_name("").is_err());
    }

    #[test]
    fn test_is_project_id() {
        assert!(is_project_id("path_a1b2c3d4e5f6789a"));
        assert!(is_project_id("path_0000000000000000"));
        assert!(is_project_id("path_ffffffffffffffff"));

        assert!(is_project_id("github_com_user_repo"));
        assert!(is_project_id("github_com_anthropics_claude_code"));
        assert!(is_project_id("gitlab_com_org_project"));
        assert!(is_project_id("bitbucket_org_team_repo"));
        assert!(is_project_id("codeberg_org_user_project"));
        assert!(is_project_id("sr_ht_user_repo"));
        assert!(is_project_id("git_myserver_com_repo"));

        assert!(is_project_id("mycompany_com_team_project"));

        assert!(!is_project_id("path_a1b2c3d4e5f6789"));
        assert!(!is_project_id("path_a1b2c3d4e5f6789ab"));
        assert!(!is_project_id("path_ghijklmnopqrstuv"));
        assert!(!is_project_id("paths_a1b2c3d4e5f6789a"));

        assert!(!is_project_id("langchain"));
        assert!(!is_project_id("react"));
        assert!(!is_project_id("react-docs"));
        assert!(!is_project_id("numpy"));
        assert!(!is_project_id("lodash"));
        assert!(!is_project_id("tensorflow_keras"));

        assert!(!is_project_id(""));
        assert!(!is_project_id("path_"));
    }

    #[test]
    fn test_determine_collection_routing_memory() {
        let result = determine_collection_routing("memory", "github_com_user_repo");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "memory");
        assert_eq!(tenant_type, "project_id");
        assert_eq!(tenant_value, "github_com_user_repo");

        let result = determine_collection_routing("agent_memory", "path_a1b2c3d4e5f6789a");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "memory");
        assert_eq!(tenant_type, "project_id");
        assert_eq!(tenant_value, "path_a1b2c3d4e5f6789a");
    }

    #[test]
    fn test_determine_collection_routing_projects() {
        let result = determine_collection_routing("notes", "path_a1b2c3d4e5f6789a");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "projects");
        assert_eq!(tenant_type, "project_id");
        assert_eq!(tenant_value, "path_a1b2c3d4e5f6789a");

        let result = determine_collection_routing("code", "github_com_anthropics_claude_code");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "projects");
        assert_eq!(tenant_type, "project_id");
        assert_eq!(tenant_value, "github_com_anthropics_claude_code");

        let result = determine_collection_routing("src", "gitlab_com_org_project");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "projects");
        assert_eq!(tenant_type, "project_id");
        assert_eq!(tenant_value, "gitlab_com_org_project");
    }

    #[test]
    fn test_determine_collection_routing_libraries() {
        let result = determine_collection_routing("docs", "langchain");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "libraries");
        assert_eq!(tenant_type, "library_name");
        assert_eq!(tenant_value, "langchain");

        let result = determine_collection_routing("reference", "react");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "libraries");
        assert_eq!(tenant_type, "library_name");
        assert_eq!(tenant_value, "react");

        let result = determine_collection_routing("api", "react-native");
        assert!(result.is_ok());
        let (collection, tenant_type, tenant_value) = result.unwrap();
        assert_eq!(collection, "libraries");
        assert_eq!(tenant_type, "library_name");
        assert_eq!(tenant_value, "react-native");
    }

    #[test]
    fn test_determine_collection_routing_validation() {
        assert!(determine_collection_routing("", "tenant123").is_err());
        assert!(determine_collection_routing("notes", "").is_err());
        assert!(determine_collection_routing("", "").is_err());
    }

    #[test]
    fn test_validate_document_id() {
        let valid_uuid = Uuid::new_v4().to_string();
        assert!(validate_document_id(&valid_uuid).is_ok());

        assert!(validate_document_id("not-a-uuid").is_err());
        assert!(validate_document_id("12345").is_err());
        assert!(validate_document_id("").is_err());
    }
}

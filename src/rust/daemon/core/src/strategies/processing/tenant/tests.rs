//! Tests for the tenant processing strategy.

use std::path::Path;

use crate::patterns::exclusion::should_exclude_file;
use crate::strategies::ProcessingStrategy;
use crate::unified_queue_schema::{ItemType, QueueOperation};

use super::TenantStrategy;

#[test]
fn test_tenant_strategy_handles_tenant_items() {
    let strategy = TenantStrategy;
    assert!(strategy.handles(&ItemType::Tenant, &QueueOperation::Add));
    assert!(strategy.handles(&ItemType::Tenant, &QueueOperation::Scan));
    assert!(strategy.handles(&ItemType::Tenant, &QueueOperation::Delete));
    assert!(strategy.handles(&ItemType::Tenant, &QueueOperation::Rename));
}

#[test]
fn test_tenant_strategy_handles_doc_items() {
    let strategy = TenantStrategy;
    assert!(strategy.handles(&ItemType::Doc, &QueueOperation::Delete));
    assert!(strategy.handles(&ItemType::Doc, &QueueOperation::Uplift));
}

#[test]
fn test_tenant_strategy_rejects_non_tenant_items() {
    let strategy = TenantStrategy;
    assert!(!strategy.handles(&ItemType::File, &QueueOperation::Add));
    assert!(!strategy.handles(&ItemType::Text, &QueueOperation::Add));
    assert!(!strategy.handles(&ItemType::Folder, &QueueOperation::Scan));
}

#[test]
fn test_tenant_strategy_name() {
    let strategy = TenantStrategy;
    assert_eq!(strategy.name(), "tenant");
}

/// Test that the exclusion check logic correctly identifies files that should be cleaned up.
/// This tests the core decision logic used by cleanup_excluded_files without needing
/// Qdrant or SQLite connections.
#[test]
fn test_cleanup_exclusion_logic_identifies_hidden_files() {
    let project_root = Path::new("/home/user/project");

    // Simulate file paths as they would be stored in Qdrant (absolute paths)
    let qdrant_paths = vec![
        "/home/user/project/src/main.rs",
        "/home/user/project/.hidden_file",
        "/home/user/project/src/.secret",
        "/home/user/project/.git/config",
        "/home/user/project/src/lib.rs",
        "/home/user/project/node_modules/package/index.js",
        "/home/user/project/.env",
        "/home/user/project/README.md",
        "/home/user/project/src/.cache/data",
        "/home/user/project/.github/workflows/ci.yml",
    ];

    let mut should_delete = Vec::new();
    let mut should_keep = Vec::new();

    for qdrant_file in &qdrant_paths {
        let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
            Ok(stripped) => stripped.to_string_lossy().to_string(),
            Err(_) => qdrant_file.to_string(),
        };

        if should_exclude_file(&rel_path) {
            should_delete.push(qdrant_file.to_string());
        } else {
            should_keep.push(qdrant_file.to_string());
        }
    }

    // Hidden files should be marked for deletion
    assert!(
        should_delete.contains(&"/home/user/project/.hidden_file".to_string()),
        "Expected .hidden_file to be excluded"
    );
    assert!(
        should_delete.contains(&"/home/user/project/src/.secret".to_string()),
        "Expected src/.secret to be excluded"
    );
    assert!(
        should_delete.contains(&"/home/user/project/.git/config".to_string()),
        "Expected .git/config to be excluded"
    );
    assert!(
        should_delete.contains(&"/home/user/project/.env".to_string()),
        "Expected .env to be excluded"
    );
    assert!(
        should_delete.contains(&"/home/user/project/src/.cache/data".to_string()),
        "Expected src/.cache/data to be excluded"
    );
    assert!(
        should_delete.contains(&"/home/user/project/node_modules/package/index.js".to_string()),
        "Expected node_modules content to be excluded"
    );

    // Normal files should NOT be deleted
    assert!(
        should_keep.contains(&"/home/user/project/src/main.rs".to_string()),
        "Expected src/main.rs to be kept"
    );
    assert!(
        should_keep.contains(&"/home/user/project/src/lib.rs".to_string()),
        "Expected src/lib.rs to be kept"
    );
    assert!(
        should_keep.contains(&"/home/user/project/README.md".to_string()),
        "Expected README.md to be kept"
    );

    // .github/ should be whitelisted (not excluded)
    assert!(
        should_keep.contains(&"/home/user/project/.github/workflows/ci.yml".to_string()),
        "Expected .github/workflows/ci.yml to be kept (whitelisted)"
    );
}

#[test]
fn test_cleanup_exclusion_logic_with_non_strippable_paths() {
    // Test when Qdrant paths don't share the project root prefix
    let project_root = Path::new("/home/user/project");
    let qdrant_file = "/different/root/src/.hidden";

    let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
        Ok(stripped) => stripped.to_string_lossy().to_string(),
        Err(_) => qdrant_file.to_string(),
    };

    // Should still detect hidden component even with full path fallback
    assert!(
        should_exclude_file(&rel_path),
        "Expected .hidden to be excluded even when path can't be stripped"
    );
}

#[test]
fn test_cleanup_exclusion_logic_empty_paths() {
    // Verify no panic with edge cases
    let project_root = Path::new("/home/user/project");
    let qdrant_paths: Vec<String> = vec![];

    let mut count = 0u64;
    for qdrant_file in &qdrant_paths {
        let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
            Ok(stripped) => stripped.to_string_lossy().to_string(),
            Err(_) => qdrant_file.clone(),
        };

        if should_exclude_file(&rel_path) {
            count += 1;
        }
    }

    assert_eq!(count, 0, "Empty path list should produce zero deletions");
}

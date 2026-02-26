//! Unit tests for project command module

use super::check::{CheckSummary, FileCheckEntry};
use super::resolver::{resolve_project_id, resolve_project_id_or_cwd_quiet};

// --- resolve_project_id tests ---

#[test]
fn test_resolve_project_id_plain_id_passthrough() {
    // A string without '/' or '.' is treated as a direct project ID
    let result = resolve_project_id("my-project-abc123");
    assert_eq!(result, "my-project-abc123");
}

#[test]
fn test_resolve_project_id_with_dot_triggers_path_resolution() {
    // A string containing '.' is treated as a path; if canonicalize fails,
    // it falls back to the original string
    let result = resolve_project_id("nonexistent.path");
    // canonicalize will fail -> falls back to original
    assert_eq!(result, "nonexistent.path");
}

#[test]
fn test_resolve_project_id_with_slash_triggers_path_resolution() {
    // A string containing '/' is treated as a path
    let result = resolve_project_id("/nonexistent/path");
    // canonicalize will fail -> falls back to original
    assert_eq!(result, "/nonexistent/path");
}

#[test]
fn test_resolve_project_id_tilde_triggers_path_resolution() {
    // "~" alone is treated as a path
    let result = resolve_project_id("~");
    // "~" won't canonicalize through std::path -> falls back
    assert_eq!(result, "~");
}

#[test]
fn test_resolve_project_id_real_path() {
    // Use a path that actually exists: /tmp
    let result = resolve_project_id("/tmp");
    // /tmp on macOS is a symlink to /private/tmp, so canonicalize resolves it
    // The result should be a valid project_id (SHA256 hash)
    assert!(!result.is_empty());
    assert_ne!(result, "/tmp"); // canonicalize + calculate_project_id transforms it
}

#[test]
fn test_resolve_project_id_current_dir() {
    // "." contains a dot, so it triggers path resolution
    let result = resolve_project_id(".");
    // Should resolve CWD and compute a project ID
    assert!(!result.is_empty());
    assert_ne!(result, ".");
}

// --- resolve_project_id_or_cwd_quiet tests ---

#[test]
fn test_resolve_project_id_or_cwd_quiet_with_explicit_id() {
    let (id, auto) = resolve_project_id_or_cwd_quiet(Some("explicit-id")).unwrap();
    assert_eq!(id, "explicit-id");
    assert!(!auto);
}

#[test]
fn test_resolve_project_id_or_cwd_quiet_with_explicit_path() {
    let (id, auto) = resolve_project_id_or_cwd_quiet(Some("/tmp")).unwrap();
    // Path triggers calculate_project_id
    assert!(!id.is_empty());
    assert!(!auto);
}

// --- CheckSummary serialization tests ---

#[test]
fn test_check_summary_json_serialization() {
    let summary = CheckSummary {
        project_id: "test-project".to_string(),
        project_root: "/home/user/project".to_string(),
        up_to_date: 50,
        to_add: 3,
        to_update: 2,
        to_delete: 1,
        total_tracked: 53,
        total_on_disk: 55,
        files: vec![
            FileCheckEntry {
                path: "src/new.rs".to_string(),
                status: "add",
            },
            FileCheckEntry {
                path: "src/changed.rs".to_string(),
                status: "update",
            },
            FileCheckEntry {
                path: "src/deleted.rs".to_string(),
                status: "delete",
            },
        ],
    };
    let serialized = serde_json::to_string(&summary).unwrap();
    assert!(serialized.contains("\"project_id\":\"test-project\""));
    assert!(serialized.contains("\"up_to_date\":50"));
    assert!(serialized.contains("\"to_add\":3"));
    assert!(serialized.contains("\"to_update\":2"));
    assert!(serialized.contains("\"to_delete\":1"));
    assert!(serialized.contains("\"total_tracked\":53"));
    assert!(serialized.contains("\"total_on_disk\":55"));

    // Verify file entries
    let value: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    let files = value["files"].as_array().unwrap();
    assert_eq!(files.len(), 3);
    assert_eq!(files[0]["status"], "add");
    assert_eq!(files[1]["status"], "update");
    assert_eq!(files[2]["status"], "delete");
}

#[test]
fn test_check_summary_empty_files_omitted() {
    let summary = CheckSummary {
        project_id: "test".to_string(),
        project_root: "/tmp".to_string(),
        up_to_date: 10,
        to_add: 0,
        to_update: 0,
        to_delete: 0,
        total_tracked: 10,
        total_on_disk: 10,
        files: Vec::new(),
    };
    let serialized = serde_json::to_string(&summary).unwrap();
    // files field is skipped when empty due to skip_serializing_if
    assert!(!serialized.contains("\"files\""));
}

#[test]
fn test_check_summary_roundtrip() {
    let summary = CheckSummary {
        project_id: "proj-123".to_string(),
        project_root: "/path/to/project".to_string(),
        up_to_date: 100,
        to_add: 5,
        to_update: 3,
        to_delete: 1,
        total_tracked: 104,
        total_on_disk: 108,
        files: Vec::new(),
    };
    let json_str = serde_json::to_string(&summary).unwrap();
    let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(value["up_to_date"], 100);
    assert_eq!(value["to_add"], 5);
    assert_eq!(value["total_tracked"], 104);
    assert_eq!(value["total_on_disk"], 108);
}

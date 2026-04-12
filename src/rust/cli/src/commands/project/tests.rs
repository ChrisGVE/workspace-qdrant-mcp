//! Unit tests for project command module

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

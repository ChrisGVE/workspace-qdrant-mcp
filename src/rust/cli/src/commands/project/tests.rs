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
    // A string containing '.' is treated as a path. Under syntactic
    // normalization (spec §3.1, no fs canonicalize) a relative input
    // is absolutized against CWD purely syntactically — no `fs::metadata`
    // existence check happens. The result is therefore a project_id
    // hash, NOT the original literal.
    let result = resolve_project_id("nonexistent.path");
    assert!(!result.is_empty());
    assert_ne!(result, "nonexistent.path");
}

#[test]
fn test_resolve_project_id_with_slash_triggers_path_resolution() {
    // A string containing '/' is treated as a path. Under syntactic
    // normalization an absolute path always produces a project_id —
    // existence on disk is not checked.
    let result = resolve_project_id("/nonexistent/path");
    assert!(!result.is_empty());
    assert_ne!(result, "/nonexistent/path");
}

#[test]
fn test_resolve_project_id_tilde_triggers_path_resolution() {
    // "~" alone is treated as a path. Tilde expansion is part of the
    // canonical normalization rules (spec §3.1 rule 2), so it now
    // produces a valid canonical path and therefore a project_id hash.
    let result = resolve_project_id("~");
    assert!(!result.is_empty());
    assert_ne!(result, "~");
}

#[test]
fn test_resolve_project_id_real_path() {
    // Use a path that actually exists: /tmp
    let result = resolve_project_id("/tmp");
    // Under syntactic normalization /tmp stays /tmp (no symlink follow);
    // calculate_project_id hashes the syntactic-canonical form.
    assert!(!result.is_empty());
    assert_ne!(result, "/tmp");
}

#[test]
fn test_resolve_project_id_current_dir() {
    // "." contains a dot, so it triggers path resolution. Under
    // syntactic normalization it is absolutized against CWD and a
    // hash is produced.
    let result = resolve_project_id(".");
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

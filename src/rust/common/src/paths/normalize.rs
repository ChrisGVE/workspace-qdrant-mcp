//! Pure syntactic path normalization.
//!
//! Implements the nine canonical-form rules from spec §3.1:
//!
//! 1. Reject relative input.
//! 2. Expand leading `~` via `$HOME`.
//! 3. Remove `.` segments.
//! 4. Reject any `..` segment (rationale: §3.2.1).
//! 5. Collapse duplicate `/` separators.
//! 6. Preserve case exactly.
//! 7. Do NOT resolve symbolic links.
//! 8. Do NOT touch the filesystem.
//! 9. Require UTF-8 validity.
//!
//! No filesystem access, no symlink resolution, no case folding.

use super::PathError;

/// Apply the nine normalization rules to `input` and return the canonical
/// string form.
///
/// This is the single authoritative implementation of canonical normalization
/// in the Rust workspace. Both [`super::CanonicalPath::from_user_input`] and
/// (in debug builds) [`super::CanonicalPath::from_validated`] call it.
///
/// # Errors
///
/// Returns a [`PathError`] when any rule fails: relative input,
/// `..` present, non-UTF-8 segment, empty result, or NUL byte.
pub(super) fn normalize_path(input: &str) -> Result<String, PathError> {
    if input.is_empty() {
        return Err(PathError::EmptyPath);
    }

    // Embedded NUL bytes are never valid in canonical paths and would
    // produce surprising behavior at any fs/SQLite/gRPC boundary.
    if input.contains('\0') {
        return Err(PathError::InvalidNormalization(
            "path contains embedded NUL byte".to_string(),
        ));
    }

    // Rule 2: `~` expansion. shellexpand::tilde is a borrowing op when no
    // expansion is needed, so this allocates only when the input begins with
    // `~`.
    let expanded = shellexpand::tilde(input);
    let expanded_str: &str = &expanded;

    // Normalize Windows separators first so the canonical form stays POSIX-like
    // even when the input originated from a Windows shell.
    let as_posix = expanded_str.replace('\\', "/");

    // Rule 1: must be absolute after tilde expansion and separator
    // normalization. This accepts `/mnt/c/...`-style inputs while still
    // rejecting relative paths and drive-prefixed Windows paths.
    if !as_posix.starts_with('/') {
        return Err(PathError::RelativeInput(input.to_string()));
    }

    let mut parts = Vec::new();

    for component in as_posix.split('/') {
        // Rule 3: drop `.` segments.
        if component.is_empty() || component == "." {
            continue;
        }

        // Rule 4: reject `..` entirely. §3.2.1 explains why we do NOT
        // resolve syntactically — combined with rule 7's no-symlink
        // resolution it produces paths that don't correspond to a real
        // filesystem location.
        if component == ".." {
            return Err(PathError::ContainsParentDir(input.to_string()));
        }

        // Rule 6 + Rule 9: preserve case, require UTF-8.
        parts.push(component);
    }

    if parts.is_empty() {
        return Err(PathError::EmptyPath);
    }

    Ok(format!("/{}", parts.join("/")))
}

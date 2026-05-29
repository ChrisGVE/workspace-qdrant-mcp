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
//!
//! # Accepted absolute forms
//!
//! - **POSIX**: `/foo/bar/baz` — leading `/`.
//! - **Windows drive**: `C:/foo/bar/baz` — drive letter + `:` + `/`.
//!   Backslashes in input (`C:\foo\bar`) are normalized to forward
//!   slashes before the absolute-form check.
//! - **UNC / `\\?\` paths**: not yet supported; treated as relative.
//!
//! On Windows hosts, `C:/Users/...` is the natural canonical form. On
//! POSIX hosts, Windows-drive inputs are still accepted so a config
//! authored on Windows can be parsed identically on macOS/Linux for
//! tooling and validation.

use super::PathError;

/// Returns `Some(drive_prefix_len)` if `s` starts with a Windows drive
/// prefix like `C:/`. Otherwise `None`.
///
/// `s` is assumed to already have backslashes converted to forward
/// slashes. The drive letter must be ASCII alphabetic; `1:/` is rejected.
fn windows_drive_prefix_len(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && bytes[2] == b'/' {
        // "C:" — caller skips the trailing slash separately when splitting.
        Some(2)
    } else {
        None
    }
}

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

    // Normalize Windows separators first so the canonical form uses
    // forward slashes regardless of whether the input came from a
    // Windows shell or a POSIX one.
    let as_posix = expanded_str.replace('\\', "/");

    // Rule 1: must be absolute after tilde expansion and separator
    // normalization. Accept POSIX (`/...`) and Windows drive (`C:/...`).
    let (prefix, rest) = if let Some(drive_len) = windows_drive_prefix_len(&as_posix) {
        // `drive_len` = 2 ("C:"). Skip the drive AND the separator slash.
        let drive = &as_posix[..drive_len];
        let after_drive = &as_posix[drive_len + 1..];
        (drive.to_string(), after_drive)
    } else if let Some(rest) = as_posix.strip_prefix('/') {
        (String::new(), rest)
    } else {
        return Err(PathError::RelativeInput(input.to_string()));
    };

    let mut parts = Vec::new();

    for component in rest.split('/') {
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

    if prefix.is_empty() {
        Ok(format!("/{}", parts.join("/")))
    } else {
        Ok(format!("{prefix}/{}", parts.join("/")))
    }
}

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

use std::path::{Component, Path};

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

    // Rule 1: must be absolute after tilde expansion. We check the expanded
    // form because `~/foo` becomes `/Users/username/foo` and IS absolute, but
    // `foo/bar` stays relative.
    let path = Path::new(expanded_str);
    if !path.is_absolute() {
        return Err(PathError::RelativeInput(input.to_string()));
    }

    let mut normalized = String::with_capacity(expanded_str.len());

    for component in path.components() {
        match component {
            // Rule 4: reject `..` entirely. §3.2.1 explains why we do NOT
            // resolve syntactically — combined with rule 7's no-symlink
            // resolution it produces paths that don't correspond to a real
            // filesystem location.
            Component::ParentDir => {
                return Err(PathError::ContainsParentDir(input.to_string()));
            }
            // Rule 3: drop `.` segments.
            Component::CurDir => continue,
            // Rule 5 (start): leading `/`.
            Component::RootDir => normalized.push('/'),
            // Rule 6 + Rule 9: preserve case, require UTF-8.
            Component::Normal(s) => {
                if !normalized.ends_with('/') && !normalized.is_empty() {
                    normalized.push('/');
                }
                let segment = s.to_str().ok_or(PathError::NonUtf8)?;
                normalized.push_str(segment);
            }
            // Windows-only components (Prefix). Out of scope per spec §13;
            // reject defensively so a Windows-style path doesn't sneak through.
            Component::Prefix(_) => {
                return Err(PathError::InvalidNormalization(
                    "windows path prefix not supported".to_string(),
                ));
            }
        }
    }

    if normalized.is_empty() {
        return Err(PathError::EmptyPath);
    }

    Ok(normalized)
}

/// Fold a Windows-host WSL UNC path to the daemon's native POSIX path.
///
/// On a Windows host editing a repo that lives inside WSL, the client reports
/// `cwd` as a UNC path such as `\\wsl.localhost\Ubuntu\home\user\repo` (or the
/// legacy `\\wsl$\Ubuntu\...`). The daemon runs inside WSL and stored the
/// project under `/home/user/repo`, so the UNC form never matches a registered
/// project. This strips the `\\wsl(.localhost|$)\<distro>` prefix and converts
/// the remaining backslashes to forward slashes, yielding the POSIX path the
/// daemon knows. Any other input — already-POSIX, a native Windows path, a
/// non-WSL UNC share, or a relative path — is returned unchanged.
///
/// Pure and syntactic: no filesystem access, no symlink resolution. The host
/// token (`wsl.localhost` / `wsl$`) is matched case-insensitively, as Windows
/// UNC hosts are. Salvaged from alkmimm PR #134 (`5e7497759` item 1).
pub fn canonicalize_host_path(path: &str) -> String {
    // Only `\\…` or `//…` UNC-style inputs can be WSL paths — everything else
    // (POSIX, drive-letter Windows, relative) is returned untouched.
    if !(path.starts_with("\\\\") || path.starts_with("//")) {
        return path.to_string();
    }

    // Accept both `\\wsl$\…` and an already-forward-slashed `//wsl$/…`.
    let slashed = path.replace('\\', "/");
    let rest = slashed.trim_start_matches('/');
    let mut segments = rest.split('/');

    let host = segments.next().unwrap_or("").to_ascii_lowercase();
    if host != "wsl.localhost" && host != "wsl$" {
        // Some other UNC share (e.g. `\\server\share`) — leave it alone.
        return path.to_string();
    }

    // The next segment is the distro name; everything after it is the POSIX
    // path rooted at `/`.
    let _distro = segments.next();
    let tail: Vec<&str> = segments.filter(|s| !s.is_empty()).collect();
    format!("/{}", tail.join("/"))
}

#[cfg(test)]
mod host_path_tests {
    use super::canonicalize_host_path;

    #[test]
    fn folds_wsl_localhost_unc() {
        assert_eq!(
            canonicalize_host_path(r"\\wsl.localhost\Ubuntu\home\user\repo"),
            "/home/user/repo"
        );
    }

    #[test]
    fn folds_legacy_wsl_dollar_unc() {
        assert_eq!(
            canonicalize_host_path(r"\\wsl$\Debian\srv\code\proj"),
            "/srv/code/proj"
        );
    }

    #[test]
    fn folds_forward_slashed_and_mixed_case_host() {
        assert_eq!(
            canonicalize_host_path("//WSL.LocalHost/Ubuntu/home/x"),
            "/home/x"
        );
    }

    #[test]
    fn distro_root_folds_to_slash() {
        assert_eq!(canonicalize_host_path(r"\\wsl.localhost\Ubuntu"), "/");
    }

    #[test]
    fn posix_path_unchanged() {
        assert_eq!(canonicalize_host_path("/home/user/repo"), "/home/user/repo");
    }

    #[test]
    fn windows_drive_path_unchanged() {
        assert_eq!(
            canonicalize_host_path(r"C:\Users\x\repo"),
            r"C:\Users\x\repo"
        );
    }

    #[test]
    fn non_wsl_unc_share_unchanged() {
        assert_eq!(
            canonicalize_host_path(r"\\fileserver\share\dir"),
            r"\\fileserver\share\dir"
        );
    }
}

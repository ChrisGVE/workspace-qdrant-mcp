//! Canonicalize well-known Docker Desktop mount aliases back to host form.
//!
//! On Windows, the same physical file under `C:\Users\alice\repo` can appear
//! to a process under several different paths depending on where the process
//! is running:
//!
//! * `/mnt/c/Users/alice/repo` — the canonical WSL2 host view (what `wsl.exe`
//!   shows and what host-native processes see).
//! * `/run/desktop/mnt/host/c/Users/alice/repo` — the path as observed from a
//!   Docker Desktop container that has bind-mounted the WSL2 root.
//! * `/host_mnt/c/Users/alice/repo` — the older Docker Desktop mount-point
//!   convention used by some images and earlier engine versions.
//!
//! For purposes of tenant ID derivation we want all three forms to hash to
//! the same canonical value so a project registered from the WSL2 host and
//! the same project registered from a Docker container produce a single
//! [`crate::project_id::ProjectIdCalculator`] tenant_id instead of two.
//!
//! This module provides a pure string rewrite that maps the two container
//! aliases back to the canonical `/mnt/<drive>/...` form. It does not consult
//! the filesystem and never resolves symlinks — the rewrite is syntactic per
//! spec §3.1's no-fs-access guarantee.
//!
//! # Scope
//!
//! The rewrite is intentionally narrow:
//!
//! * Only the two well-known Docker Desktop prefix families are recognized.
//!   User-declared mounts (whose source of truth is
//!   [`crate::paths::MountMap`]) are out of scope and unaffected.
//! * Only single-letter drive components (`a`..`z`, ASCII) are recognized.
//!   This matches the Windows drive-letter convention; non-conforming
//!   components are left as-is.
//! * The drive letter is **lowercased** in the output. Docker Desktop emits
//!   `/run/desktop/mnt/host/c/...` (lowercase) and `/mnt/c/...` is also
//!   lowercase by WSL2 convention, so this is a no-op in practice — but we
//!   normalize defensively so a stray `/host_mnt/C/...` still collapses to
//!   `/mnt/c/...`.
//! * Component boundaries are respected: `/run/desktop/mnt/hostfoo` does
//!   **not** match the `/run/desktop/mnt/host/` prefix.
//!
//! # Examples
//!
//! ```
//! use wqm_common::paths::docker_alias::canonicalize_docker_mount_alias;
//!
//! // Docker Desktop WSL2 view collapses to canonical WSL2 host form.
//! assert_eq!(
//!     canonicalize_docker_mount_alias("/run/desktop/mnt/host/c/Users/alice/repo"),
//!     Some("/mnt/c/Users/alice/repo".to_string()),
//! );
//!
//! // Older Docker Desktop alias also collapses.
//! assert_eq!(
//!     canonicalize_docker_mount_alias("/host_mnt/d/data"),
//!     Some("/mnt/d/data".to_string()),
//! );
//!
//! // Already-canonical paths are not rewritten.
//! assert_eq!(canonicalize_docker_mount_alias("/mnt/c/Users/alice/repo"), None);
//!
//! // Paths under unrelated roots are not rewritten.
//! assert_eq!(canonicalize_docker_mount_alias("/Users/chris/dev/project"), None);
//! ```

/// Well-known Docker Desktop alias prefixes, each followed by a single-letter
/// drive component. Listed longest-first so the matcher picks the most
/// specific prefix when more than one would syntactically match (in practice
/// these prefixes are disjoint, but the ordering is robust regardless).
const DOCKER_ALIAS_PREFIXES: &[&str] = &["/run/desktop/mnt/host/", "/host_mnt/"];

/// Rewrite a well-known Docker Desktop mount alias to the canonical host
/// `/mnt/<drive>/...` form, or return `None` if no alias matches.
///
/// Returning `None` means the input either is already canonical or refers
/// to a path outside the known Docker Desktop alias families — the caller
/// should leave it untouched.
///
/// # Examples
///
/// ```
/// use wqm_common::paths::docker_alias::canonicalize_docker_mount_alias;
///
/// // The bare prefix (just the drive component, no trailing segments) maps
/// // to the canonical drive root.
/// assert_eq!(
///     canonicalize_docker_mount_alias("/run/desktop/mnt/host/c"),
///     Some("/mnt/c".to_string()),
/// );
///
/// // Trailing slash is preserved by the rewrite (it stays a trailing slash
/// // in the output for the bare-drive case).
/// assert_eq!(
///     canonicalize_docker_mount_alias("/host_mnt/c/"),
///     Some("/mnt/c".to_string()),
/// );
///
/// // Non-drive component after the prefix is rejected — `hostfoo` is not
/// // a single ASCII letter so the input is left as-is.
/// assert_eq!(
///     canonicalize_docker_mount_alias("/run/desktop/mnt/host/foo/bar"),
///     None,
/// );
/// ```
pub fn canonicalize_docker_mount_alias(input: &str) -> Option<String> {
    for prefix in DOCKER_ALIAS_PREFIXES {
        if let Some(rest) = input.strip_prefix(prefix) {
            // `rest` is everything after the trailing `/` of the alias
            // prefix. We need to peel off the next component and verify it
            // is a single ASCII letter (the Windows drive letter).
            let (drive_raw, suffix) = match rest.find('/') {
                Some(idx) => (&rest[..idx], &rest[idx + 1..]),
                None => (rest, ""),
            };

            if !is_drive_letter(drive_raw) {
                // The prefix matched but the next component is not a valid
                // drive letter (e.g. `/run/desktop/mnt/host/foo/...`). Leave
                // the input untouched — we do not know how to canonicalize it.
                return None;
            }

            // Lowercase the drive letter defensively; production Docker
            // Desktop always emits lowercase, but a stray uppercase from
            // user input should still collapse to the canonical form.
            let drive_lower = drive_raw.to_ascii_lowercase();

            // Strip a trailing `/` from the suffix so the output never has a
            // dangling separator. The downstream `normalize_path` strips
            // duplicates but we keep the helper's contract clean.
            let suffix = suffix.trim_end_matches('/');

            return Some(if suffix.is_empty() {
                format!("/mnt/{drive_lower}")
            } else {
                format!("/mnt/{drive_lower}/{suffix}")
            });
        }
    }
    None
}

/// Returns `true` when `s` is exactly one ASCII letter — a Windows drive
/// letter component.
fn is_drive_letter(s: &str) -> bool {
    s.len() == 1 && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ────────────────────────────────────────────────────────────────────
    // Docker Desktop WSL2 alias (/run/desktop/mnt/host/...)
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn run_desktop_alias_with_subpath() {
        assert_eq!(
            canonicalize_docker_mount_alias(
                "/run/desktop/mnt/host/c/Users/alice/repo"
            ),
            Some("/mnt/c/Users/alice/repo".to_string()),
        );
    }

    #[test]
    fn run_desktop_alias_bare_drive() {
        assert_eq!(
            canonicalize_docker_mount_alias("/run/desktop/mnt/host/c"),
            Some("/mnt/c".to_string()),
        );
    }

    #[test]
    fn run_desktop_alias_drive_d() {
        // Drives other than C must work too.
        assert_eq!(
            canonicalize_docker_mount_alias("/run/desktop/mnt/host/d/data/foo"),
            Some("/mnt/d/data/foo".to_string()),
        );
    }

    #[test]
    fn run_desktop_alias_uppercase_drive_collapses() {
        // Defensively lowercase the drive letter so a stray uppercase still
        // hashes to the same canonical form as the WSL2 host view.
        assert_eq!(
            canonicalize_docker_mount_alias("/run/desktop/mnt/host/C/Users/alice"),
            Some("/mnt/c/Users/alice".to_string()),
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Older Docker Desktop alias (/host_mnt/...)
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn host_mnt_alias_with_subpath() {
        assert_eq!(
            canonicalize_docker_mount_alias("/host_mnt/c/Users/alice/repo"),
            Some("/mnt/c/Users/alice/repo".to_string()),
        );
    }

    #[test]
    fn host_mnt_alias_bare_drive() {
        assert_eq!(
            canonicalize_docker_mount_alias("/host_mnt/c"),
            Some("/mnt/c".to_string()),
        );
    }

    #[test]
    fn host_mnt_alias_with_trailing_slash() {
        // Trailing slash on the bare-drive form is normalized away — the
        // result is the canonical drive root with no dangling separator.
        assert_eq!(
            canonicalize_docker_mount_alias("/host_mnt/c/"),
            Some("/mnt/c".to_string()),
        );
    }

    // ────────────────────────────────────────────────────────────────────
    // Non-matching inputs
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn already_canonical_returns_none() {
        // `/mnt/<drive>/...` is already the canonical host form and must
        // never be rewritten — that would be lossy / introduce a fixed point
        // that does not converge.
        assert_eq!(
            canonicalize_docker_mount_alias("/mnt/c/Users/alice/repo"),
            None,
        );
    }

    #[test]
    fn unrelated_host_path_returns_none() {
        // macOS / Linux native paths must pass through untouched.
        assert_eq!(
            canonicalize_docker_mount_alias("/Users/chris/dev/project"),
            None,
        );
        assert_eq!(
            canonicalize_docker_mount_alias("/home/alice/repo"),
            None,
        );
        assert_eq!(canonicalize_docker_mount_alias("/"), None);
    }

    #[test]
    fn component_boundary_required() {
        // `/run/desktop/mnt/hostfoo/...` shares a prefix string with
        // `/run/desktop/mnt/host/` but is NOT under the alias root. The
        // strip-prefix match would still succeed (`hostfoo` after the
        // shared chars), but the next component check (`foo` after the
        // shared `/run/desktop/mnt/`) is not what we strip — we strip on
        // the trailing-slash form so this scenario does not match at all.
        assert_eq!(
            canonicalize_docker_mount_alias("/run/desktop/mnt/hostfoo/bar"),
            None,
        );
    }

    #[test]
    fn non_drive_component_after_prefix_returns_none() {
        // The prefix matches but the next component is not a single ASCII
        // letter — leave the input untouched rather than producing a
        // malformed `/mnt/foo/...` (which is meaningless under WSL2 conv).
        assert_eq!(
            canonicalize_docker_mount_alias("/run/desktop/mnt/host/foo/bar"),
            None,
        );
        assert_eq!(
            canonicalize_docker_mount_alias("/host_mnt/12/foo"),
            None,
        );
        assert_eq!(canonicalize_docker_mount_alias("/host_mnt/"), None);
    }

    // ────────────────────────────────────────────────────────────────────
    // Fixed-point and idempotency
    // ────────────────────────────────────────────────────────────────────

    #[test]
    fn rewrite_is_a_fixed_point_on_canonical_form() {
        // Applying the rewrite twice must equal applying it once. The first
        // call produces `/mnt/c/...`, the second sees that and returns None,
        // which the caller interprets as "no change".
        let alias = "/run/desktop/mnt/host/c/Users/alice/repo";
        let once = canonicalize_docker_mount_alias(alias).unwrap();
        assert_eq!(canonicalize_docker_mount_alias(&once), None);
    }

    #[test]
    fn both_alias_families_converge_to_same_output() {
        // Both Docker Desktop conventions must collapse to the same canonical
        // form — that is the entire point of this helper.
        let wsl2_view = "/run/desktop/mnt/host/c/Users/alice/repo";
        let host_mnt_view = "/host_mnt/c/Users/alice/repo";

        let wsl2_canonical = canonicalize_docker_mount_alias(wsl2_view).unwrap();
        let host_mnt_canonical =
            canonicalize_docker_mount_alias(host_mnt_view).unwrap();

        assert_eq!(wsl2_canonical, host_mnt_canonical);
        assert_eq!(wsl2_canonical, "/mnt/c/Users/alice/repo");
    }
}

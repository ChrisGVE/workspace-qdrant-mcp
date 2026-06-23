//! Branch-name validation for the branch-storage registry (AC-F4.6 / SEC-N04).
//!
//! File: src/rust/daemon/core/src/branch_name_validation.rs
//! Context: daemon-core owns the registration path for `project_locations`
//! (the write side). Every `branch_name` accepted into `project_locations` — or
//! rendered in `wqm project branches` output — must pass validation here BEFORE
//! it is stored or displayed. This is a defence-in-depth guard against:
//!
//! 1. Injected terminal escape sequences via a crafted ref name surfaced in
//!    table output (SEC-N04 terminal injection).
//! 2. Silent acceptance of git-invalid names that `git2` operations would later
//!    reject or misinterpret.
//!
//! ## Validation rules (AC-F4.6)
//!
//! - Length: reject if byte length exceeds 255 (git's practical ref-name limit).
//! - Control characters: reject if any byte is ASCII < 0x20 (covers `\0`, `\n`,
//!   all C0 controls — the primary terminal-injection vector).
//! - Git ref-name rules: delegate to `git2::Reference::is_valid_name`, prefixing
//!   `"refs/heads/"` to convert a branch short-name into a fully-qualified ref
//!   path that git2 can validate. This covers git-forbidden sequences:
//!     `..`, leading `/`, trailing `.lock`, `@{`, embedded whitespace, and
//!     any other git ref-name restriction (DR GP-9 — reuse, not a hand-rolled regex).
//!
//! ## Placement constraint (AC-F4.6 enforcement)
//!
//! `git2` is a LEAF-FORBIDDEN dependency in `wqm-common` (the leaf principle):
//! `wqm-common` must stay git2-free so it can be linked into MCP-server and
//! wqm-cli without pulling in the git C library. `daemon-core` already depends
//! on `git2` (Cargo.toml line 48), so this module lives here.
//!
//! The error type is `wqm_common::StorageError` so callers on the registration
//! path (which already use `StorageError`) need no new error conversions.

use wqm_common::error::StorageError;

/// Maximum permitted byte length for a branch name (git's practical limit).
///
/// Git itself does not enforce a byte limit in all code paths, but `git check-ref-format`
/// documents 255 as the safe upper bound for interoperability. Names beyond this
/// are rejected pre-storage to prevent oversized values in `project_locations.branch_name`.
pub const MAX_BRANCH_NAME_BYTES: usize = 255;

/// Validate `branch_name` against git ref-name rules before storing or rendering.
///
/// Returns `Ok(())` when the name is safe to store in `project_locations.branch_name`
/// and display in terminal output. Returns `Err(StorageError::Validation)` with a
/// descriptive message for any violation.
///
/// Validation sequence:
/// 1. Empty check — an empty name is always invalid.
/// 2. Length check — reject if byte length exceeds [`MAX_BRANCH_NAME_BYTES`].
/// 3. Control-char check — reject any byte < 0x20 (ASCII C0 controls, covers
///    `\0`, `\n`, `\r`, ESC, and the full control range).
/// 4. Git ref-name check — delegate to `git2::Reference::is_valid_name` with the
///    `"refs/heads/<name>"` prefix, which applies git's full ref-name ruleset
///    without a hand-rolled regex (DR GP-9).
///
/// The caller is responsible for calling this BEFORE invoking
/// `wqm_common::hashing::branch_id` or any INSERT into `project_locations`.
pub fn validate_branch_name(branch_name: &str) -> Result<(), StorageError> {
    // 1. Empty.
    if branch_name.is_empty() {
        return Err(StorageError::Validation(
            "branch_name must not be empty".into(),
        ));
    }

    // 2. Length — byte count, not char count (git operates on bytes).
    if branch_name.len() > MAX_BRANCH_NAME_BYTES {
        return Err(StorageError::Validation(format!(
            "branch_name exceeds maximum length of {} bytes (got {})",
            MAX_BRANCH_NAME_BYTES,
            branch_name.len()
        )));
    }

    // 3. Control characters (ASCII < 0x20).
    //    This is a pre-filter BEFORE the git2 call: git2's `is_valid_name` does
    //    catch many control chars, but an explicit check here makes the rejection
    //    reason clear and guards against any future git2 version that might relax
    //    its check. Rejecting the full C0 range (not just \0 and \n) also ensures
    //    terminal output cannot be manipulated by escape-sequence injection.
    if branch_name.bytes().any(|b| b < 0x20) {
        return Err(StorageError::Validation(
            "branch_name must not contain control characters (ASCII < 0x20)".into(),
        ));
    }

    // 4. Git ref-name rules via git2 (DR GP-9 reuse).
    //    We validate the SHORT name by prefixing "refs/heads/" — the canonical
    //    fully-qualified form for a local branch. `git2::Reference::is_valid_name`
    //    applies git's ref-name checker (the same logic as `git check-ref-format`),
    //    covering: leading `/`, trailing `.lock`, `..`, `@{`, trailing `.`,
    //    embedded whitespace, `~`, `^`, `:`, `?`, `*`, `[`, and `\`.
    let qualified = format!("refs/heads/{branch_name}");
    if !git2::Reference::is_valid_name(&qualified) {
        return Err(StorageError::Validation(format!(
            "branch_name '{}' is not a valid git ref name \
             (failed git2::Reference::is_valid_name for 'refs/heads/{}')",
            branch_name, branch_name
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- passing cases ---------------------------------------------------------

    /// T-F4-val-normal: a conventional branch name passes validation.
    #[test]
    fn t_f4_val_normal() {
        assert!(validate_branch_name("main").is_ok(), "'main' must pass");
        assert!(
            validate_branch_name("feat/new-feature").is_ok(),
            "'feat/new-feature' must pass"
        );
        assert!(
            validate_branch_name("fix/issue-123").is_ok(),
            "'fix/issue-123' must pass"
        );
        assert!(
            validate_branch_name("release/v2.0.0").is_ok(),
            "'release/v2.0.0' must pass"
        );
    }

    // ---- rejection cases -------------------------------------------------------

    /// T-F4-val-empty: an empty name is rejected.
    #[test]
    fn t_f4_val_empty() {
        let err = validate_branch_name("").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("empty"), "empty rejection: {msg}");
    }

    /// T-F4-val-over-length: a name exceeding 255 bytes is rejected (AC-F4.6).
    #[test]
    fn t_f4_val_over_length() {
        let long_name = "a".repeat(MAX_BRANCH_NAME_BYTES + 1);
        let err = validate_branch_name(&long_name).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("maximum length"),
            "over-length rejection must mention length: {msg}"
        );
    }

    /// T-F4-val-255-bytes-ok: a name of exactly 255 bytes passes the length check.
    #[test]
    fn t_f4_val_255_bytes_ok() {
        // Build a 255-byte valid-char name: "a" repeated 255 times.
        let at_limit = "a".repeat(MAX_BRANCH_NAME_BYTES);
        // This must not fail on the length check (git2 may or may not accept it,
        // but the length guard is what AC-F4.6 specifies).
        let result = validate_branch_name(&at_limit);
        // If git2 rejects it for other reasons that is fine; what must NOT happen
        // is a length-related rejection for a 255-byte name.
        if let Err(e) = &result {
            assert!(
                !e.to_string().contains("maximum length"),
                "255-byte name must not be rejected for length: {e}"
            );
        }
    }

    /// T-F4-val-control-chars: names containing control characters are rejected.
    #[test]
    fn t_f4_val_control_chars() {
        // Null byte.
        let err = validate_branch_name("feat/\x00null").unwrap_err();
        assert!(
            err.to_string().contains("control"),
            "null-byte rejection: {err}"
        );

        // Newline.
        let err = validate_branch_name("feat/\nnewline").unwrap_err();
        assert!(
            err.to_string().contains("control"),
            "newline rejection: {err}"
        );

        // Carriage return (0x0D, a C0 control).
        let err = validate_branch_name("feat/\r").unwrap_err();
        assert!(err.to_string().contains("control"), "CR rejection: {err}");

        // ESC (0x1B, the classic terminal-injection entry point).
        let err = validate_branch_name("\x1b[31mhello\x1b[0m").unwrap_err();
        assert!(
            err.to_string().contains("control"),
            "ESC-sequence rejection: {err}"
        );
    }

    /// T-F4-val-git-invalid: git-forbidden sequences are rejected via git2.
    #[test]
    fn t_f4_val_git_invalid() {
        // Double dot — rejected by git check-ref-format.
        let err = validate_branch_name("foo..bar").unwrap_err();
        assert!(
            err.to_string().contains("valid git ref"),
            "double-dot rejection: {err}"
        );

        // Trailing .lock — rejected by git check-ref-format.
        let err = validate_branch_name("feat.lock").unwrap_err();
        assert!(
            err.to_string().contains("valid git ref"),
            ".lock-suffix rejection: {err}"
        );

        // @{ sequence — rejected by git check-ref-format.
        let err = validate_branch_name("branch@{0}").unwrap_err();
        assert!(
            err.to_string().contains("valid git ref"),
            "@{{}} rejection: {err}"
        );

        // Leading slash — rejected by git check-ref-format.
        let err = validate_branch_name("/leading-slash").unwrap_err();
        assert!(
            err.to_string().contains("valid git ref"),
            "leading-slash rejection: {err}"
        );
    }

    /// T-F4-val-error-type: validation errors are StorageError::Validation variants.
    #[test]
    fn t_f4_val_error_is_storage_error() {
        let err = validate_branch_name("foo..bar").unwrap_err();
        assert!(
            matches!(err, StorageError::Validation(_)),
            "expected StorageError::Validation, got {err:?}"
        );
    }
}

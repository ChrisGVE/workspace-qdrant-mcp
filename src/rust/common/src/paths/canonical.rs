//! [`CanonicalPath`] — host-absolute, syntactically normalized, UTF-8 path.
//!
//! See `docs/specs/16-path-abstraction.md` §3 and §4.1.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::normalize::normalize_path;
use super::PathError;

/// Host-absolute, syntactically normalized, UTF-8 path.
///
/// Stable across deployment modes. This is the form persisted to SQLite,
/// transmitted over gRPC, and returned in MCP responses. Two processes
/// running on the same host — one natively, one in Docker — agree on the
/// canonical form even when the local filesystem view differs.
///
/// Stored as [`String`] internally because gRPC `string` fields and SQLite
/// `TEXT` columns both require UTF-8. [`std::path::PathBuf`] on Linux can
/// hold arbitrary bytes that would fail at serialization boundaries; storing
/// `String` shifts the validation cost to construction time where it
/// belongs.
///
/// Construct via [`CanonicalPath::from_user_input`] for untrusted input
/// (CLI args, gRPC payloads, config fields) or
/// [`CanonicalPath::from_validated`] for values already known to be
/// canonical (DB row decode, deserialization).
///
/// # Examples
///
/// ```
/// use wqm_common::paths::CanonicalPath;
///
/// let path = CanonicalPath::from_user_input("/Users/username/dev/project").unwrap();
/// assert_eq!(path.as_str(), "/Users/username/dev/project");
///
/// // Relative paths are rejected.
/// assert!(CanonicalPath::from_user_input("relative/path").is_err());
///
/// // `..` segments are rejected (spec §3.2.1).
/// assert!(CanonicalPath::from_user_input("/Users/username/../other").is_err());
///
/// // `.` segments are removed.
/// let path = CanonicalPath::from_user_input("/Users/username/./project").unwrap();
/// assert_eq!(path.as_str(), "/Users/username/project");
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CanonicalPath(String);

impl CanonicalPath {
    /// Build a [`CanonicalPath`] from raw user input.
    ///
    /// This is the primary entrypoint for untrusted input: CLI arguments,
    /// gRPC request fields, configuration values. Applies all nine
    /// normalization rules from spec §3.1.
    ///
    /// # Errors
    ///
    /// Returns [`PathError::RelativeInput`] if the path is not absolute
    /// (after `~` expansion), [`PathError::ContainsParentDir`] if any `..`
    /// segment is present, [`PathError::NonUtf8`] for non-UTF-8 segments,
    /// [`PathError::EmptyPath`] for empty input, or
    /// [`PathError::InvalidNormalization`] for embedded NUL bytes and
    /// Windows-style prefixes.
    ///
    /// # Examples
    ///
    /// ```
    /// use wqm_common::paths::CanonicalPath;
    ///
    /// let p = CanonicalPath::from_user_input("/Users/username//dev/./project").unwrap();
    /// assert_eq!(p.as_str(), "/Users/username/dev/project");
    /// ```
    pub fn from_user_input(s: &str) -> Result<Self, PathError> {
        let normalized = normalize_path(s)?;
        Ok(CanonicalPath(normalized))
    }

    /// Build a [`CanonicalPath`] from a value already known to be canonical.
    ///
    /// Intended for deserialization paths: SQLite row decode, prost message
    /// decode, JSON payload decode. Still runs the same validation as
    /// [`Self::from_user_input`] — never silently accepts a malformed string.
    /// In debug builds, additionally asserts that the input is already in
    /// fully-normalized canonical form (no `.` segments, no duplicate `/`,
    /// no tilde to expand).
    ///
    /// This constructor is `pub` rather than crate-private because
    /// `CanonicalPath` lives in `wqm-common` while persistence
    /// (`daemon/core`) and gRPC (`daemon/grpc`) layers are separate crates
    /// that must call it. The "deserialization-only" discipline is enforced
    /// by the multi-layer defense in spec §4.3 (CI grep + code review +
    /// type system), not by Rust visibility.
    ///
    /// # Errors
    ///
    /// Returns a [`PathError`] for any of the same failure modes as
    /// [`Self::from_user_input`].
    pub fn from_validated(s: String) -> Result<Self, PathError> {
        let normalized = normalize_path(&s)?;
        // Debug-only: catch persistence layers writing unnormalized strings.
        // Mismatch means the producer should have used from_user_input or
        // re-normalized before storing.
        debug_assert_eq!(
            normalized, s,
            "from_validated called with non-canonical path: input did not equal its normalized form"
        );
        Ok(CanonicalPath(normalized))
    }

    /// View the canonical path as a borrowed `&str`.
    ///
    /// Use this for read-only inspection (logging, comparison, gRPC
    /// serialization). The returned slice is guaranteed to be in canonical
    /// form per spec §3.1.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the [`CanonicalPath`] and return the inner owned [`String`].
    ///
    /// Useful at serialization boundaries where the canonical form is
    /// about to leave the type system (e.g., into a `prost`-generated
    /// `String` field).
    pub fn into_string(self) -> String {
        self.0
    }
}

impl fmt::Display for CanonicalPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for CanonicalPath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

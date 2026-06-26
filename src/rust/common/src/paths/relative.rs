//! [`RelativePath`] — content path anchored to a watch-folder or library root.
//!
//! See `docs/specs/16-path-abstraction.md` §3.3 and §4.1.
//!
//! A relative path names content **inside** a project or library. It is
//! deployment-independent by construction: the same relative path is valid
//! in every clone of a project sharing the same `tenant_id`, and on both
//! sides of a host/container boundary.
//!
//! Reconstruction of the absolute path at read time is a JOIN:
//!
//! ```text
//! absolute = watch_folders.path + "/" + relative_path
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::CanonicalPath;

/// Content-relative path anchored to a watch-folder or library root.
///
/// Not absolute, no `..` segments, no `.` segments after normalization,
/// no leading `/`, no duplicate `/`, UTF-8.
///
/// Construct via [`RelativePath::from_user_input`] for untrusted input
/// (CLI arguments, gRPC payloads, watcher events) or
/// [`RelativePath::from_validated`] for values already known to be valid
/// (DB row decode, deserialization, internal construction after upstream
/// validation).
///
/// Mount-map translation is NOT needed — relative paths carry no
/// host/container distinction.
///
/// # Examples
///
/// ```
/// use wqm_common::paths::RelativePath;
///
/// let p = RelativePath::from_user_input("src/main.rs").unwrap();
/// assert_eq!(p.as_str(), "src/main.rs");
///
/// // Leading `/` is rejected (absolute input).
/// assert!(RelativePath::from_user_input("/etc/passwd").is_err());
///
/// // `..` segments are rejected (traversal risk).
/// assert!(RelativePath::from_user_input("src/../escape").is_err());
///
/// // `.` segments are stripped, duplicate `/` collapsed.
/// let p = RelativePath::from_user_input("src/./foo//bar.rs").unwrap();
/// assert_eq!(p.as_str(), "src/foo/bar.rs");
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RelativePath(String);

/// Failure modes for relative-path construction.
///
/// Each variant carries the offending input verbatim for diagnostics.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum RelativePathError {
    /// Input was absolute (starts with `/` on Unix or contains a Windows
    /// drive-letter prefix). Spec §3.3 rule 1.
    #[error("relative path must not be absolute, got: {0:?}")]
    Absolute(String),

    /// Input contained a `..` segment. Spec §3.3 rule 2.
    #[error("relative path must not contain '..' segments, got: {0:?}")]
    ParentTraversal(String),

    /// Input contained an unnormalized `.` segment after a leading-position
    /// stripping pass. Spec §3.3 rule 3 (these are removed during
    /// normalization; this error only fires when normalization fails to
    /// remove a curious construct, never in normal use).
    #[error("relative path contains '.' segment after normalization, got: {0:?}")]
    CurrentDir(String),

    /// Input was empty or normalized to an empty string. Spec §3.3 rule 7.
    #[error("relative path must not be empty")]
    Empty,

    /// Input contained a NUL byte or other illegal character. Spec §3.3.
    #[error("relative path is not normalized: {reason}; got: {input:?}")]
    NotNormalized {
        /// Reason for the normalization failure.
        reason: String,
        /// The offending input verbatim.
        input: String,
    },

    /// Absolute input was not under the supplied root. Returned by
    /// [`RelativePath::from_absolute_and_root`] when `path.strip_prefix(root)`
    /// fails — the producer asked to anchor a path to a root the path does
    /// not actually live under.
    #[error("absolute path {path:?} is not under root {root:?}")]
    NotUnderRoot {
        /// The absolute path the caller tried to anchor.
        path: String,
        /// The root the caller tried to anchor it to.
        root: String,
    },
}

impl RelativePath {
    /// Build a [`RelativePath`] by stripping a canonical root prefix from an
    /// absolute path.
    ///
    /// Used at producer sites that have both the absolute filesystem path
    /// being enqueued and the owning `CanonicalPath` root (watch_folder
    /// root, library root). The resulting `RelativePath` is the same string
    /// that would later be reconstructed via `to_absolute(root)`.
    ///
    /// # Errors
    ///
    /// - [`RelativePathError::NotUnderRoot`] when `path` is not under `root`.
    /// - Any [`RelativePathError`] variant from normalization when the
    ///   stripped suffix is invalid (e.g. empty, contains NUL, contains `..`).
    ///
    /// # Examples
    ///
    /// ```
    /// use wqm_common::paths::{CanonicalPath, RelativePath};
    ///
    /// let root = CanonicalPath::from_user_input("/Users/username/lib").unwrap();
    /// let abs = CanonicalPath::from_user_input("/Users/username/lib/cs/book.pdf").unwrap();
    /// let rel = RelativePath::from_absolute_and_root(&abs, &root).unwrap();
    /// assert_eq!(rel.as_str(), "cs/book.pdf");
    /// ```
    pub fn from_absolute_and_root(
        path: &CanonicalPath,
        root: &CanonicalPath,
    ) -> Result<Self, RelativePathError> {
        use std::path::Path;
        let suffix = Path::new(path.as_str())
            .strip_prefix(root.as_str())
            .map_err(|_| RelativePathError::NotUnderRoot {
                path: path.as_str().to_string(),
                root: root.as_str().to_string(),
            })?;
        let suffix_str = suffix
            .to_str()
            .ok_or_else(|| RelativePathError::NotNormalized {
                reason: "stripped suffix is not valid UTF-8".to_string(),
                input: path.as_str().to_string(),
            })?;
        Self::from_user_input(suffix_str)
    }

    /// Build a [`RelativePath`] from raw user input.
    ///
    /// Applies §3.3 normalization rules: rejects absolute input, rejects
    /// `..`, strips `.` segments, collapses duplicate `/`, rejects empty.
    ///
    /// # Errors
    ///
    /// Returns a [`RelativePathError`] variant matching the failed rule.
    pub fn from_user_input(s: &str) -> Result<Self, RelativePathError> {
        let normalized = normalize_relative(s)?;
        Ok(RelativePath(normalized))
    }

    /// Build a [`RelativePath`] from a value already known to be valid.
    ///
    /// Intended for DB row decode and prost message decode where the value
    /// originated from a `from_user_input` upstream. Still validates — never
    /// silently accepts a malformed string. In debug builds, additionally
    /// asserts that the input is already in fully-normalized form.
    ///
    /// This constructor is `pub` rather than crate-private because
    /// `RelativePath` lives in `wqm-common` while persistence
    /// (`daemon/core`) and gRPC (`daemon/grpc`) layers are separate crates
    /// that must call it. The "deserialization-only" discipline is enforced
    /// by spec §4.3's multi-layer defense (CI grep + code review + type
    /// system), not by Rust visibility.
    ///
    /// # Errors
    ///
    /// Same as [`Self::from_user_input`].
    pub fn from_validated(s: String) -> Result<Self, RelativePathError> {
        let normalized = normalize_relative(&s)?;
        debug_assert_eq!(
            normalized, s,
            "from_validated called with non-normalized relative path: input did not equal its normalized form"
        );
        Ok(RelativePath(normalized))
    }

    /// View the relative path as a borrowed `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume and return the inner [`String`].
    pub fn into_string(self) -> String {
        self.0
    }

    /// Reconstruct an absolute [`CanonicalPath`] by joining to a root.
    ///
    /// Used at the I/O boundary in the daemon when the caller has both the
    /// owning watch-folder root (`CanonicalPath`) and the relative content
    /// path (`RelativePath`).
    pub fn to_absolute(&self, root: &CanonicalPath) -> CanonicalPath {
        // `CanonicalPath` is guaranteed to be absolute and normalized;
        // `RelativePath` is guaranteed to not start with `/` and not
        // contain `..`. Concatenation cannot escape, and we only need
        // the syntactic-canonical form for storage.
        let mut joined = String::with_capacity(root.as_str().len() + 1 + self.0.len());
        joined.push_str(root.as_str());
        if !joined.ends_with('/') {
            joined.push('/');
        }
        joined.push_str(&self.0);
        // The result is guaranteed to be normalized canonical form: the
        // root is canonical, the relative has no `.`/`..` and no duplicate
        // `/`, and the join inserts exactly one `/`. Bypass re-validation
        // by constructing directly through `from_validated` which only
        // debug-asserts.
        CanonicalPath::from_validated(joined).expect(
            "absolute path joined from canonical root and validated relative must be canonical",
        )
    }
}

impl fmt::Display for RelativePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for RelativePath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Apply the §3.3 normalization rules to `input` and return the relative
/// string form.
///
/// Rules:
///
/// 1. Reject absolute (leading `/` on Unix, Windows drive-letter prefix).
/// 2. Reject any `..` segment.
/// 3. Strip `.` segments.
/// 4. Collapse duplicate `/`.
/// 5. Reject empty input/result.
/// 6. UTF-8 is guaranteed by `&str` typing.
/// 7. Reject NUL bytes.
fn normalize_relative(input: &str) -> Result<String, RelativePathError> {
    if input.is_empty() {
        return Err(RelativePathError::Empty);
    }
    if input.contains('\0') {
        return Err(RelativePathError::NotNormalized {
            reason: "contains embedded NUL byte".to_string(),
            input: input.to_string(),
        });
    }

    // Rule 1: reject absolute input.
    if input.starts_with('/') {
        return Err(RelativePathError::Absolute(input.to_string()));
    }
    // Reject Windows-style drive-letter prefix defensively (e.g. `C:\foo`).
    // Two leading characters: alpha + colon.
    let bytes = input.as_bytes();
    if bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic() {
        return Err(RelativePathError::Absolute(input.to_string()));
    }

    let mut out = String::with_capacity(input.len());
    for segment in input.split('/') {
        if segment.is_empty() {
            // Duplicate `/` or trailing `/`: skip.
            continue;
        }
        if segment == "." {
            // Rule 3: drop `.` segments.
            continue;
        }
        if segment == ".." {
            // Rule 2: reject `..`.
            return Err(RelativePathError::ParentTraversal(input.to_string()));
        }
        if !out.is_empty() {
            out.push('/');
        }
        out.push_str(segment);
    }

    if out.is_empty() {
        return Err(RelativePathError::Empty);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Positive cases — each §3.3 rule.
    // -----------------------------------------------------------------

    #[test]
    fn simple_relative_path_survives() {
        let p = RelativePath::from_user_input("src/main.rs").unwrap();
        assert_eq!(p.as_str(), "src/main.rs");
    }

    #[test]
    fn deep_relative_path_survives() {
        let p = RelativePath::from_user_input("a/b/c/d/e/f.rs").unwrap();
        assert_eq!(p.as_str(), "a/b/c/d/e/f.rs");
    }

    #[test]
    fn dot_segments_are_stripped() {
        let p = RelativePath::from_user_input("src/./foo/./bar.rs").unwrap();
        assert_eq!(p.as_str(), "src/foo/bar.rs");
    }

    #[test]
    fn duplicate_slashes_collapsed() {
        let p = RelativePath::from_user_input("src//foo///bar.rs").unwrap();
        assert_eq!(p.as_str(), "src/foo/bar.rs");
    }

    #[test]
    fn trailing_slash_stripped() {
        let p = RelativePath::from_user_input("src/foo/").unwrap();
        assert_eq!(p.as_str(), "src/foo");
    }

    #[test]
    fn case_is_preserved() {
        let p = RelativePath::from_user_input("Src/Foo/Bar.rs").unwrap();
        assert_eq!(p.as_str(), "Src/Foo/Bar.rs");
    }

    // -----------------------------------------------------------------
    // Negative cases — every error variant.
    // -----------------------------------------------------------------

    #[test]
    fn absolute_input_rejected() {
        let err = RelativePath::from_user_input("/etc/passwd").unwrap_err();
        assert!(matches!(err, RelativePathError::Absolute(_)));
    }

    #[test]
    fn windows_drive_letter_rejected() {
        let err = RelativePath::from_user_input("C:/Users/foo").unwrap_err();
        assert!(matches!(err, RelativePathError::Absolute(_)));
    }

    #[test]
    fn parent_traversal_rejected() {
        let err = RelativePath::from_user_input("src/../escape").unwrap_err();
        assert!(matches!(err, RelativePathError::ParentTraversal(_)));
    }

    #[test]
    fn leading_parent_traversal_rejected() {
        let err = RelativePath::from_user_input("../escape").unwrap_err();
        assert!(matches!(err, RelativePathError::ParentTraversal(_)));
    }

    #[test]
    fn empty_input_rejected() {
        let err = RelativePath::from_user_input("").unwrap_err();
        assert!(matches!(err, RelativePathError::Empty));
    }

    #[test]
    fn only_dot_segments_rejected_as_empty() {
        // After stripping `.` segments the result is empty.
        let err = RelativePath::from_user_input("./.").unwrap_err();
        assert!(matches!(err, RelativePathError::Empty));
    }

    #[test]
    fn only_slashes_rejected_as_absolute() {
        // Leading `/` is the first check.
        let err = RelativePath::from_user_input("//").unwrap_err();
        assert!(matches!(err, RelativePathError::Absolute(_)));
    }

    #[test]
    fn nul_byte_rejected() {
        let err = RelativePath::from_user_input("src/\0/main.rs").unwrap_err();
        assert!(matches!(err, RelativePathError::NotNormalized { .. }));
    }

    // -----------------------------------------------------------------
    // Constructor variants.
    // -----------------------------------------------------------------

    #[test]
    fn from_validated_accepts_normalized_input() {
        let p = RelativePath::from_validated("src/main.rs".to_string()).unwrap();
        assert_eq!(p.as_str(), "src/main.rs");
    }

    #[test]
    #[should_panic(expected = "from_validated called with non-normalized")]
    #[cfg(debug_assertions)]
    fn from_validated_debug_asserts_normalized() {
        // Has a duplicate `/` — would normalize but debug-assert fires.
        let _ = RelativePath::from_validated("src//main.rs".to_string());
    }

    #[test]
    fn from_validated_rejects_absolute() {
        let err = RelativePath::from_validated("/abs".to_string()).unwrap_err();
        assert!(matches!(err, RelativePathError::Absolute(_)));
    }

    // -----------------------------------------------------------------
    // Serialize / deserialize round-trip.
    // -----------------------------------------------------------------

    #[test]
    fn serde_transparent_serializes_as_string() {
        let p = RelativePath::from_user_input("src/main.rs").unwrap();
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, "\"src/main.rs\"");
    }

    #[test]
    fn serde_round_trip() {
        let p = RelativePath::from_user_input("a/b/c").unwrap();
        let json = serde_json::to_string(&p).unwrap();
        let back: RelativePath = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    // -----------------------------------------------------------------
    // Display / AsRef.
    // -----------------------------------------------------------------

    #[test]
    fn display_writes_inner_verbatim() {
        let p = RelativePath::from_user_input("src/foo.rs").unwrap();
        assert_eq!(format!("{}", p), "src/foo.rs");
    }

    #[test]
    fn as_ref_borrows_inner() {
        let p = RelativePath::from_user_input("x/y").unwrap();
        let r: &str = p.as_ref();
        assert_eq!(r, "x/y");
    }

    #[test]
    fn into_string_consumes_and_returns_inner() {
        let p = RelativePath::from_user_input("x/y").unwrap();
        assert_eq!(p.into_string(), "x/y");
    }

    // -----------------------------------------------------------------
    // to_absolute — join with canonical root.
    // -----------------------------------------------------------------

    #[test]
    fn to_absolute_joins_with_root() {
        let root = CanonicalPath::from_user_input("/Users/username/dev/project").unwrap();
        let rel = RelativePath::from_user_input("src/main.rs").unwrap();
        let abs = rel.to_absolute(&root);
        assert_eq!(abs.as_str(), "/Users/username/dev/project/src/main.rs");
    }

    #[test]
    fn to_absolute_handles_root_with_trailing_slash_normalization() {
        // CanonicalPath construction strips trailing slashes via its own
        // normalization, but we still defensively handle the case here.
        let root = CanonicalPath::from_user_input("/r").unwrap();
        let rel = RelativePath::from_user_input("a/b.rs").unwrap();
        let abs = rel.to_absolute(&root);
        assert_eq!(abs.as_str(), "/r/a/b.rs");
    }

    // -----------------------------------------------------------------
    // from_absolute_and_root — strip canonical-root prefix.
    // -----------------------------------------------------------------

    #[test]
    fn from_absolute_and_root_strips_root_prefix() {
        let root = CanonicalPath::from_user_input("/Users/username/lib").unwrap();
        let abs = CanonicalPath::from_user_input("/Users/username/lib/cs/book.pdf").unwrap();
        let rel = RelativePath::from_absolute_and_root(&abs, &root).unwrap();
        assert_eq!(rel.as_str(), "cs/book.pdf");
    }

    #[test]
    fn from_absolute_and_root_handles_immediate_child() {
        let root = CanonicalPath::from_user_input("/r").unwrap();
        let abs = CanonicalPath::from_user_input("/r/file.txt").unwrap();
        let rel = RelativePath::from_absolute_and_root(&abs, &root).unwrap();
        assert_eq!(rel.as_str(), "file.txt");
    }

    #[test]
    fn from_absolute_and_root_rejects_path_outside_root() {
        let root = CanonicalPath::from_user_input("/Users/username/lib").unwrap();
        let abs = CanonicalPath::from_user_input("/Users/username/other/doc.pdf").unwrap();
        let err = RelativePath::from_absolute_and_root(&abs, &root).unwrap_err();
        assert!(matches!(err, RelativePathError::NotUnderRoot { .. }));
    }

    #[test]
    fn from_absolute_and_root_rejects_root_equals_path() {
        // Root and absolute path are identical; stripped suffix is empty.
        let root = CanonicalPath::from_user_input("/Users/username/lib").unwrap();
        let abs = CanonicalPath::from_user_input("/Users/username/lib").unwrap();
        let err = RelativePath::from_absolute_and_root(&abs, &root).unwrap_err();
        assert!(matches!(err, RelativePathError::Empty));
    }

    #[test]
    fn from_absolute_and_root_round_trip_through_to_absolute() {
        let root = CanonicalPath::from_user_input("/r/lib").unwrap();
        let original = CanonicalPath::from_user_input("/r/lib/a/b/c.txt").unwrap();
        let rel = RelativePath::from_absolute_and_root(&original, &root).unwrap();
        let reconstructed = rel.to_absolute(&root);
        assert_eq!(reconstructed.as_str(), original.as_str());
    }
}

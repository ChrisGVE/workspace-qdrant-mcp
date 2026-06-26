//! Path validation macros and helpers for gRPC handler boundaries.
//!
//! Provides three extraction macros that validate raw `String` path fields
//! from proto messages and convert them into validated newtypes. Invalid
//! inputs are mapped to `tonic::Status::invalid_argument` with a
//! descriptive message.
//!
//! # Macro catalogue
//!
//! | Macro | Input | Output | Use for |
//! |---|---|---|---|
//! | [`extract_canonical_path!`] | `String` | `CanonicalPath` | Root/watch paths |
//! | [`extract_relative_path!`] | `String` | `RelativePath` | Single content path |
//! | [`extract_relative_paths!`] | `Vec<String>` | `Vec<RelativePath>` | Repeated content paths |
//!
//! # Design notes
//!
//! `CanonicalPath` is re-used from `wqm_common::paths`. `RelativePath` is
//! defined locally in this module because the `wqm_common` crate does not
//! yet export one (deferred to a future common-layer task). The local type
//! enforces the same invariants specified in `docs/specs/16-path-abstraction.md`
//! section 6.3: no leading `/`, no `..` segments, no embedded NUL, non-empty,
//! valid UTF-8.

use std::fmt;

/// A validated relative content path.
///
/// Invariants (enforced at construction):
/// - Non-empty.
/// - Not absolute (no leading `/` or `~`).
/// - No `..` path segments.
/// - No embedded NUL bytes.
/// - Valid UTF-8 (guaranteed by accepting `&str`).
///
/// Construct via [`RelativePath::from_user_input`].
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct RelativePath(String);

impl RelativePath {
    /// Build a [`RelativePath`] from untrusted user input (gRPC request field).
    ///
    /// # Errors
    ///
    /// Returns a human-readable error string when any invariant is violated.
    pub fn from_user_input(s: &str) -> Result<Self, String> {
        if s.is_empty() {
            return Err("path is empty".to_string());
        }
        if s.contains('\0') {
            return Err("path contains embedded NUL byte".to_string());
        }
        if s.starts_with('/') {
            return Err(format!("path must be relative, got absolute: {s:?}"));
        }
        if s.starts_with('~') {
            return Err(format!("path must be relative, got tilde-prefixed: {s:?}"));
        }
        // Check for `..` segments: split on `/` and reject any segment that
        // is exactly `..`.
        for segment in s.split('/') {
            if segment == ".." {
                return Err(format!("path must not contain '..' segments, got: {s:?}"));
            }
        }
        Ok(RelativePath(s.to_string()))
    }

    /// View the relative path as a borrowed `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the [`RelativePath`] and return the inner owned [`String`].
    pub fn into_string(self) -> String {
        self.0
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

// ---------------------------------------------------------------------------
// Extraction macros
// ---------------------------------------------------------------------------

/// Validate a `String` field as a [`wqm_common::paths::CanonicalPath`],
/// returning `tonic::Status::invalid_argument` on failure.
///
/// # Usage
///
/// ```ignore
/// let canonical = extract_canonical_path!(req.path, "path")?;
/// ```
///
/// The second argument is the field name used in the error message.
macro_rules! extract_canonical_path {
    ($field:expr, $name:expr) => {{
        use wqm_common::paths::CanonicalPath;
        CanonicalPath::from_user_input(&$field)
            .map_err(|e| tonic::Status::invalid_argument(format!("{}: {e}", $name)))
    }};
}

/// Validate a `String` field as a [`RelativePath`], returning
/// `tonic::Status::invalid_argument` on failure.
///
/// # Usage
///
/// ```ignore
/// let rel = extract_relative_path!(req.file_path, "file_path")?;
/// ```
macro_rules! extract_relative_path {
    ($field:expr, $name:expr) => {{
        use $crate::validation::RelativePath;
        RelativePath::from_user_input(&$field)
            .map_err(|e| tonic::Status::invalid_argument(format!("{}: {e}", $name)))
    }};
}

/// Validate each element of a `Vec<String>` as a [`RelativePath`].
/// Returns `tonic::Status::invalid_argument` on the first failure,
/// including the zero-based element index in the error message.
///
/// # Usage
///
/// ```ignore
/// let paths = extract_relative_paths!(req.file_paths, "file_paths")?;
/// ```
macro_rules! extract_relative_paths {
    ($vec:expr, $name:expr) => {{
        use $crate::validation::RelativePath;
        (|| -> Result<Vec<RelativePath>, tonic::Status> {
            let mut out = Vec::with_capacity($vec.len());
            for (i, s) in $vec.iter().enumerate() {
                let rp = RelativePath::from_user_input(s).map_err(|e| {
                    tonic::Status::invalid_argument(format!("{}[{}]: {e}", $name, i))
                })?;
                out.push(rp);
            }
            Ok(out)
        })()
    }};
}

pub(crate) use extract_canonical_path;
pub(crate) use extract_relative_path;
pub(crate) use extract_relative_paths;

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- RelativePath positive cases --

    #[test]
    fn relative_path_simple() {
        let rp = RelativePath::from_user_input("src/main.rs").unwrap();
        assert_eq!(rp.as_str(), "src/main.rs");
    }

    #[test]
    fn relative_path_single_segment() {
        let rp = RelativePath::from_user_input("file.txt").unwrap();
        assert_eq!(rp.as_str(), "file.txt");
    }

    #[test]
    fn relative_path_with_dot_segment() {
        // Single `.` segments are allowed (they are harmless in relative paths).
        let rp = RelativePath::from_user_input("src/./file.rs").unwrap();
        assert_eq!(rp.as_str(), "src/./file.rs");
    }

    #[test]
    fn relative_path_display_matches_as_str() {
        let rp = RelativePath::from_user_input("a/b/c").unwrap();
        assert_eq!(format!("{rp}"), "a/b/c");
        assert_eq!(rp.as_ref(), "a/b/c");
    }

    #[test]
    fn relative_path_into_string() {
        let rp = RelativePath::from_user_input("a/b").unwrap();
        let s: String = rp.into_string();
        assert_eq!(s, "a/b");
    }

    // -- RelativePath negative cases --

    #[test]
    fn relative_path_empty_rejected() {
        let err = RelativePath::from_user_input("").unwrap_err();
        assert!(err.contains("empty"));
    }

    #[test]
    fn relative_path_absolute_rejected() {
        let err = RelativePath::from_user_input("/absolute/path").unwrap_err();
        assert!(err.contains("absolute"));
    }

    #[test]
    fn relative_path_tilde_rejected() {
        let err = RelativePath::from_user_input("~/home/file").unwrap_err();
        assert!(err.contains("tilde"));
    }

    #[test]
    fn relative_path_parent_dir_rejected() {
        let err = RelativePath::from_user_input("src/../secret").unwrap_err();
        assert!(err.contains(".."));
    }

    #[test]
    fn relative_path_parent_dir_at_start_rejected() {
        let err = RelativePath::from_user_input("../escape").unwrap_err();
        assert!(err.contains(".."));
    }

    #[test]
    fn relative_path_nul_byte_rejected() {
        let err = RelativePath::from_user_input("src/\0file").unwrap_err();
        assert!(err.contains("NUL"));
    }

    // -- Macro smoke tests (require tonic on the path) --

    #[test]
    fn macro_extract_canonical_path_valid() {
        let field = "/Users/username/dev".to_string();
        let result = extract_canonical_path!(field, "path");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_str(), "/Users/username/dev");
    }

    #[test]
    fn macro_extract_canonical_path_relative_rejected() {
        let field = "relative/path".to_string();
        let result = extract_canonical_path!(field, "path");
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("path:"));
    }

    #[test]
    fn macro_extract_relative_path_valid() {
        let field = "src/main.rs".to_string();
        let result = extract_relative_path!(field, "file_path");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_str(), "src/main.rs");
    }

    #[test]
    fn macro_extract_relative_path_absolute_rejected() {
        let field = "/absolute/path.rs".to_string();
        let result = extract_relative_path!(field, "file_path");
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("file_path:"));
    }

    #[test]
    fn macro_extract_relative_paths_all_valid() {
        let fields = vec!["src/a.rs".to_string(), "src/b.rs".to_string()];
        let result = extract_relative_paths!(fields, "file_paths");
        assert!(result.is_ok());
        let paths = result.unwrap();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].as_str(), "src/a.rs");
        assert_eq!(paths[1].as_str(), "src/b.rs");
    }

    #[test]
    fn macro_extract_relative_paths_second_invalid() {
        let fields = vec![
            "src/ok.rs".to_string(),
            "/absolute/bad.rs".to_string(),
            "src/also_ok.rs".to_string(),
        ];
        let result = extract_relative_paths!(fields, "file_paths");
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        // Error message must include the index [1]
        assert!(
            status.message().contains("file_paths[1]"),
            "expected index in error, got: {}",
            status.message()
        );
    }

    #[test]
    fn macro_extract_relative_paths_empty_vec_ok() {
        let fields: Vec<String> = vec![];
        let result = extract_relative_paths!(fields, "file_paths");
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn macro_extract_relative_paths_parent_dir_rejected() {
        let fields = vec!["../escape".to_string()];
        let result = extract_relative_paths!(fields, "file_paths");
        assert!(result.is_err());
        let status = result.unwrap_err();
        assert!(status.message().contains("file_paths[0]"));
        assert!(status.message().contains(".."));
    }
}

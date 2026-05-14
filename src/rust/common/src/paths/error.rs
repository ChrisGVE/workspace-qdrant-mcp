//! Error types for the canonical path abstraction.
//!
//! See `docs/specs/16-path-abstraction.md` §3.1 for the normalization rules
//! whose violations these errors describe.

use thiserror::Error;

/// Failure modes for canonical-path construction and mount-map translation.
///
/// Each variant carries enough context to produce an actionable error message
/// for the user. Variants map to spec §3.1 normalization rules and §5
/// mount-map semantics.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum PathError {
    /// Input was not absolute (§3.1 rule 1).
    #[error("path must be absolute, got: {0:?}")]
    RelativeInput(String),

    /// Input contained `..` segments, which canonical paths forbid
    /// (§3.1 rule 4, §3.2.1 rationale: `..` cannot be safely resolved
    /// without filesystem access because of symlinks).
    #[error("path must not contain '..' segments, got: {0:?}")]
    ContainsParentDir(String),

    /// Path contained non-UTF-8 byte sequences (§3.1 rule 9).
    #[error("path contains non-UTF-8 sequences")]
    NonUtf8,

    /// Empty input.
    #[error("path is empty")]
    EmptyPath,

    /// No mount-map entry covered the given canonical path (§5.2).
    #[error("no mount entry covers canonical path: {canonical:?}")]
    NoMountCoverage {
        /// The canonical path that no mount-map entry covered.
        canonical: String,
    },

    /// Other normalization failures (NUL bytes, malformed inputs the
    /// other variants don't capture).
    #[error("invalid path: {0}")]
    InvalidNormalization(String),

    /// Mount-map configuration error (duplicate host or container prefix,
    /// per §5.3).
    #[error("mount map error: {0}")]
    MountMapError(String),
}

//! Grammar version compatibility checking.
//!
//! This module provides functionality to verify that loaded tree-sitter grammars
//! are compatible with the current tree-sitter runtime version.
//!
//! # ABI Versioning
//!
//! Tree-sitter uses ABI (Application Binary Interface) versions to track
//! compatibility between grammars and the runtime. When tree-sitter's internal
//! data structures change in ways that would break compatibility, the ABI version
//! is incremented.
//!
//! - Grammars compiled against an older ABI may crash or produce incorrect results
//! - Grammars compiled against a newer ABI may not load at all
//!
//! The ABI version is an integer embedded in each Language struct.

use super::grammar_cache::GrammarMetadata;
use thiserror::Error;
use tree_sitter::Language;

/// The current tree-sitter ABI version from the tree-sitter crate.
///
/// This constant is used to check if loaded grammars are compatible.
pub const CURRENT_ABI_VERSION: u32 = tree_sitter::LANGUAGE_VERSION as u32;

/// The minimum ABI version that is compatible with the current runtime.
///
/// Tree-sitter maintains backwards compatibility within a range of ABI versions.
pub const MIN_COMPATIBLE_ABI_VERSION: u32 = tree_sitter::MIN_COMPATIBLE_LANGUAGE_VERSION as u32;

/// Errors that can occur during version compatibility checking.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum VersionError {
    #[error("Grammar ABI version {grammar_abi} is older than minimum compatible version {min_compatible}")]
    AbiTooOld {
        grammar_abi: u32,
        min_compatible: u32,
        runtime_version: u32,
    },

    #[error("Grammar ABI version {grammar_abi} is newer than current version {current}")]
    AbiTooNew { grammar_abi: u32, current: u32 },

    #[error("Invalid version string: {0}")]
    InvalidVersionString(String),
}

/// Result of a version compatibility check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompatibilityStatus {
    /// Grammar is fully compatible with the runtime.
    Compatible,
    /// Grammar uses an older but still compatible ABI version.
    CompatibleOldAbi {
        grammar_abi: u32,
        current_abi: u32,
    },
    /// Grammar is incompatible with the runtime.
    Incompatible(VersionError),
}

impl CompatibilityStatus {
    /// Returns true if the grammar is compatible (either fully or with older ABI).
    pub fn is_compatible(&self) -> bool {
        matches!(
            self,
            CompatibilityStatus::Compatible | CompatibilityStatus::CompatibleOldAbi { .. }
        )
    }

    /// Returns true if the grammar is fully compatible with current ABI.
    pub fn is_fully_compatible(&self) -> bool {
        matches!(self, CompatibilityStatus::Compatible)
    }

    /// Returns the error if incompatible.
    pub fn into_error(self) -> Option<VersionError> {
        match self {
            CompatibilityStatus::Incompatible(e) => Some(e),
            _ => None,
        }
    }
}

/// Check if a loaded grammar's ABI version is compatible with the runtime.
///
/// # Arguments
///
/// * `language` - The loaded tree-sitter Language to check
///
/// # Returns
///
/// A `CompatibilityStatus` indicating whether the grammar is compatible.
pub fn check_grammar_compatibility(language: &Language) -> CompatibilityStatus {
    let grammar_abi = language.version() as u32;

    if grammar_abi < MIN_COMPATIBLE_ABI_VERSION {
        return CompatibilityStatus::Incompatible(VersionError::AbiTooOld {
            grammar_abi,
            min_compatible: MIN_COMPATIBLE_ABI_VERSION,
            runtime_version: CURRENT_ABI_VERSION,
        });
    }

    if grammar_abi > CURRENT_ABI_VERSION {
        return CompatibilityStatus::Incompatible(VersionError::AbiTooNew {
            grammar_abi,
            current: CURRENT_ABI_VERSION,
        });
    }

    if grammar_abi < CURRENT_ABI_VERSION {
        return CompatibilityStatus::CompatibleOldAbi {
            grammar_abi,
            current_abi: CURRENT_ABI_VERSION,
        };
    }

    CompatibilityStatus::Compatible
}

/// Check if a grammar's metadata indicates compatibility with the runtime.
///
/// This checks the tree-sitter version string from metadata against the current runtime.
///
/// # Arguments
///
/// * `metadata` - The grammar metadata containing version information
///
/// # Returns
///
/// A `CompatibilityStatus` based on the version string comparison.
pub fn check_metadata_compatibility(metadata: &GrammarMetadata) -> Result<bool, VersionError> {
    let expected_version = parse_version_string(&metadata.tree_sitter_version)?;
    let current_version = parse_version_string(tree_sitter_version_string())?;

    // For now, we just check if the major.minor version matches
    // A more sophisticated check could allow compatible ranges
    Ok(expected_version.0 == current_version.0 && expected_version.1 == current_version.1)
}

/// Parse a version string like "0.24" or "0.24.0" into major/minor/patch.
fn parse_version_string(version: &str) -> Result<(u32, u32, Option<u32>), VersionError> {
    let parts: Vec<&str> = version.split('.').collect();

    if parts.is_empty() || parts.len() > 3 {
        return Err(VersionError::InvalidVersionString(version.to_string()));
    }

    let major = parts[0]
        .parse::<u32>()
        .map_err(|_| VersionError::InvalidVersionString(version.to_string()))?;

    let minor = parts
        .get(1)
        .map(|s| {
            s.parse::<u32>()
                .map_err(|_| VersionError::InvalidVersionString(version.to_string()))
        })
        .transpose()?
        .unwrap_or(0);

    let patch = parts
        .get(2)
        .map(|s| {
            s.parse::<u32>()
                .map_err(|_| VersionError::InvalidVersionString(version.to_string()))
        })
        .transpose()?;

    Ok((major, minor, patch))
}

/// Get the tree-sitter version string from the crate.
///
/// Derived at compile time from Cargo.lock via build.rs — no manual
/// synchronization needed when updating the tree-sitter dependency.
pub fn tree_sitter_version_string() -> &'static str {
    env!("TREE_SITTER_VERSION")
}

/// Get the current tree-sitter ABI version.
pub fn current_abi_version() -> u32 {
    CURRENT_ABI_VERSION
}

/// Get the minimum compatible ABI version.
pub fn min_compatible_abi_version() -> u32 {
    MIN_COMPATIBLE_ABI_VERSION
}

/// Information about the tree-sitter runtime.
#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    /// The tree-sitter version string (e.g., "0.24")
    pub version_string: &'static str,
    /// Current ABI version
    pub abi_version: u32,
    /// Minimum compatible ABI version
    pub min_compatible_abi: u32,
}

impl RuntimeInfo {
    /// Get information about the current tree-sitter runtime.
    pub fn current() -> Self {
        Self {
            version_string: tree_sitter_version_string(),
            abi_version: CURRENT_ABI_VERSION,
            min_compatible_abi: MIN_COMPATIBLE_ABI_VERSION,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_abi_version() {
        // ABI version should be a reasonable number (14 for tree-sitter 0.24)
        assert!(CURRENT_ABI_VERSION >= 13);
        assert!(CURRENT_ABI_VERSION < 100);
    }

    #[test]
    fn test_min_compatible_version() {
        // Min compatible should be less than or equal to current
        assert!(MIN_COMPATIBLE_ABI_VERSION <= CURRENT_ABI_VERSION);
        // Min compatible should be positive
        assert!(MIN_COMPATIBLE_ABI_VERSION > 0);
    }

    #[test]
    fn test_parse_version_string() {
        assert_eq!(parse_version_string("0.24").unwrap(), (0, 24, None));
        assert_eq!(parse_version_string("0.24.0").unwrap(), (0, 24, Some(0)));
        assert_eq!(parse_version_string("1.0.5").unwrap(), (1, 0, Some(5)));
        assert_eq!(parse_version_string("0").unwrap(), (0, 0, None));
    }

    #[test]
    fn test_parse_version_string_invalid() {
        assert!(parse_version_string("").is_err());
        assert!(parse_version_string("abc").is_err());
        assert!(parse_version_string("0.a.1").is_err());
        assert!(parse_version_string("1.2.3.4").is_err());
    }

    #[test]
    fn test_tree_sitter_version_string() {
        let version = tree_sitter_version_string();
        // Must not be the fallback value
        assert_ne!(version, "unknown", "build.rs should extract version from Cargo.lock");
        // Must start with expected major version
        assert!(version.starts_with("0."), "Expected 0.x.y, got: {}", version);
        // Must be a full semver triple (e.g., "0.24.7"), not just "0.24"
        let (major, minor, patch) = parse_version_string(version).unwrap();
        assert_eq!(major, 0);
        assert!(minor > 0, "Minor version should be positive");
        assert!(patch.is_some(), "Expected full semver triple from Cargo.lock");
    }

    #[test]
    fn test_compatibility_status_is_compatible() {
        assert!(CompatibilityStatus::Compatible.is_compatible());
        assert!(
            CompatibilityStatus::CompatibleOldAbi {
                grammar_abi: 13,
                current_abi: 14
            }
            .is_compatible()
        );
        assert!(!CompatibilityStatus::Incompatible(VersionError::AbiTooOld {
            grammar_abi: 10,
            min_compatible: 13,
            runtime_version: 14,
        })
        .is_compatible());
    }

    #[test]
    fn test_compatibility_status_is_fully_compatible() {
        assert!(CompatibilityStatus::Compatible.is_fully_compatible());
        assert!(
            !CompatibilityStatus::CompatibleOldAbi {
                grammar_abi: 13,
                current_abi: 14
            }
            .is_fully_compatible()
        );
    }

    #[test]
    fn test_compatibility_status_into_error() {
        assert!(CompatibilityStatus::Compatible.into_error().is_none());
        assert!(CompatibilityStatus::CompatibleOldAbi {
            grammar_abi: 13,
            current_abi: 14
        }
        .into_error()
        .is_none());

        let error = CompatibilityStatus::Incompatible(VersionError::AbiTooOld {
            grammar_abi: 10,
            min_compatible: 13,
            runtime_version: 14,
        })
        .into_error();
        assert!(error.is_some());
    }

    #[test]
    fn test_runtime_info() {
        let info = RuntimeInfo::current();
        assert_eq!(info.abi_version, CURRENT_ABI_VERSION);
        assert_eq!(info.min_compatible_abi, MIN_COMPATIBLE_ABI_VERSION);
        assert!(!info.version_string.is_empty());
    }

    // Note: Testing check_grammar_compatibility requires a real loaded grammar.
    // Those tests should be in integration tests where we have access to compiled grammars.
}

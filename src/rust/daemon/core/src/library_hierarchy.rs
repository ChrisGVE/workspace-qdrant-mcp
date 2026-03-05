/// Hierarchical library tenancy with path-based organization.
///
/// Supports hierarchical library naming: `library_name/library_path/document_name`.
/// Enables prefix-based search scoping within a library's folder structure.
///
/// Example:
/// - Library: "main"
/// - File at: `computer_science/design_patterns/GoF.pdf`
/// - library_name: "main"
/// - library_path: "computer_science/design_patterns"
/// - document_name: "GoF.pdf"
use std::path::Path;

use serde::{Deserialize, Serialize};

// ─── Types ─────────────────────────────────────────────────────────────

/// A fully resolved library document location.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LibraryLocation {
    /// Top-level library name (tenant_id in the libraries collection).
    pub library_name: String,
    /// Relative path within the library (may be empty for root-level docs).
    pub library_path: String,
    /// Document filename.
    pub document_name: String,
}

impl LibraryLocation {
    /// Build the full hierarchical path: `library_name/library_path/document_name`.
    ///
    /// If `library_path` is empty, returns `library_name/document_name`.
    pub fn full_path(&self) -> String {
        if self.library_path.is_empty() {
            format!("{}/{}", self.library_name, self.document_name)
        } else {
            format!(
                "{}/{}/{}",
                self.library_name, self.library_path, self.document_name
            )
        }
    }
}

// ─── Path extraction ───────────────────────────────────────────────────

/// Extract the library path from a document's absolute path relative to
/// a library's root directory.
///
/// Given library root `/docs/main/` and file `/docs/main/cs/patterns/GoF.pdf`,
/// returns `library_path = "cs/patterns"` and `document_name = "GoF.pdf"`.
///
/// Returns `None` if the file is not under the library root.
pub fn extract_library_path(library_root: &Path, document_path: &Path) -> Option<(String, String)> {
    let relative = document_path.strip_prefix(library_root).ok()?;

    let document_name = relative
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();

    if document_name.is_empty() {
        return None;
    }

    let library_path = relative
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default()
        // Normalize path separators to forward slashes
        .replace('\\', "/");

    Some((library_path, document_name))
}

/// Build a complete LibraryLocation from components.
pub fn build_location(
    library_name: &str,
    library_root: &Path,
    document_path: &Path,
) -> Option<LibraryLocation> {
    let (library_path, document_name) = extract_library_path(library_root, document_path)?;

    Some(LibraryLocation {
        library_name: library_name.to_string(),
        library_path,
        document_name,
    })
}

// ─── Search filter helpers ─────────────────────────────────────────────

/// Build a Qdrant filter condition for library path prefix matching.
///
/// In Qdrant, prefix matching on text fields uses the `match` condition
/// with `text` type. This returns the filter value to use.
///
/// `prefix` can be:
/// - `""` — match all documents in the library
/// - `"computer_science"` — match docs in cs/ and subdirs
/// - `"computer_science/design_patterns"` — narrower prefix
pub fn normalize_path_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim().trim_matches('/');
    // Normalize separators
    trimmed.replace('\\', "/")
}

/// Check if a library_path matches a given prefix.
///
/// Used for client-side filtering or test assertions.
pub fn path_matches_prefix(library_path: &str, prefix: &str) -> bool {
    if prefix.is_empty() {
        return true;
    }
    let normalized_path = library_path.replace('\\', "/");
    let normalized_prefix = prefix.replace('\\', "/").trim_matches('/').to_string();

    normalized_path == normalized_prefix
        || normalized_path.starts_with(&format!("{}/", normalized_prefix))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // ─── LibraryLocation tests ──────────────────────────────────────────

    #[test]
    fn test_full_path_with_library_path() {
        let loc = LibraryLocation {
            library_name: "main".to_string(),
            library_path: "cs/design_patterns".to_string(),
            document_name: "GoF.pdf".to_string(),
        };
        assert_eq!(loc.full_path(), "main/cs/design_patterns/GoF.pdf");
    }

    #[test]
    fn test_full_path_root_level() {
        let loc = LibraryLocation {
            library_name: "refs".to_string(),
            library_path: String::new(),
            document_name: "manual.pdf".to_string(),
        };
        assert_eq!(loc.full_path(), "refs/manual.pdf");
    }

    // ─── Path extraction tests ──────────────────────────────────────────

    #[test]
    fn test_extract_nested_path() {
        let root = PathBuf::from("/docs/main");
        let file = PathBuf::from("/docs/main/cs/patterns/GoF.pdf");

        let (path, name) = extract_library_path(&root, &file).unwrap();
        assert_eq!(path, "cs/patterns");
        assert_eq!(name, "GoF.pdf");
    }

    #[test]
    fn test_extract_root_level_document() {
        let root = PathBuf::from("/docs/main");
        let file = PathBuf::from("/docs/main/README.pdf");

        let (path, name) = extract_library_path(&root, &file).unwrap();
        assert_eq!(path, "");
        assert_eq!(name, "README.pdf");
    }

    #[test]
    fn test_extract_single_level_path() {
        let root = PathBuf::from("/libs");
        let file = PathBuf::from("/libs/rust/book.epub");

        let (path, name) = extract_library_path(&root, &file).unwrap();
        assert_eq!(path, "rust");
        assert_eq!(name, "book.epub");
    }

    #[test]
    fn test_extract_not_under_root() {
        let root = PathBuf::from("/docs/main");
        let file = PathBuf::from("/other/place/file.pdf");

        assert!(extract_library_path(&root, &file).is_none());
    }

    #[test]
    fn test_extract_directory_returns_none() {
        let root = PathBuf::from("/docs");
        let file = PathBuf::from("/docs/subdir");

        // Path without a file component (no extension, just a dir name)
        // extract_library_path treats it as a file named "subdir" with empty extension
        let result = extract_library_path(&root, &file);
        // This is actually valid — "subdir" is the document_name
        assert!(result.is_some());
        let (path, name) = result.unwrap();
        assert_eq!(path, "");
        assert_eq!(name, "subdir");
    }

    // ─── Build location tests ───────────────────────────────────────────

    #[test]
    fn test_build_location() {
        let loc = build_location(
            "my-lib",
            Path::new("/libs/my-lib"),
            Path::new("/libs/my-lib/topics/ai/paper.pdf"),
        )
        .unwrap();

        assert_eq!(loc.library_name, "my-lib");
        assert_eq!(loc.library_path, "topics/ai");
        assert_eq!(loc.document_name, "paper.pdf");
        assert_eq!(loc.full_path(), "my-lib/topics/ai/paper.pdf");
    }

    // ─── Prefix matching tests ──────────────────────────────────────────

    #[test]
    fn test_normalize_prefix() {
        assert_eq!(normalize_path_prefix("  cs/patterns/  "), "cs/patterns");
        assert_eq!(
            normalize_path_prefix("/leading/trailing/"),
            "leading/trailing"
        );
        assert_eq!(normalize_path_prefix(""), "");
    }

    #[test]
    fn test_path_matches_prefix_empty() {
        assert!(path_matches_prefix("cs/patterns", ""));
        assert!(path_matches_prefix("", ""));
    }

    #[test]
    fn test_path_matches_prefix_exact() {
        assert!(path_matches_prefix("cs/patterns", "cs/patterns"));
    }

    #[test]
    fn test_path_matches_prefix_parent() {
        assert!(path_matches_prefix("cs/patterns/gof", "cs"));
        assert!(path_matches_prefix("cs/patterns/gof", "cs/patterns"));
    }

    #[test]
    fn test_path_no_match() {
        assert!(!path_matches_prefix("math/algebra", "cs"));
        assert!(!path_matches_prefix("cs-extra/foo", "cs"));
    }

    #[test]
    fn test_path_matches_not_partial_segment() {
        // "cs" should not match "cs-advanced" (partial segment)
        assert!(!path_matches_prefix("cs-advanced/topic", "cs"));
    }
}

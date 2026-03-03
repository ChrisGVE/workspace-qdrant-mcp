//! File type classification and extension extraction utilities.

use std::path::Path;
use wqm_common::classification;

use super::types::FileType;

/// Determine file type for metadata classification.
///
/// Classification priority:
/// 1. Configuration dotfiles (by filename)
/// 2. Extension-based lookup from unified YAML reference
/// 3. Special handling for JSON (config vs data based on path)
/// 4. Tarball compound extensions
/// 5. Other (fallback)
///
/// Test detection is **not** part of file_type — use [`super::is_test_file`] separately.
///
/// # Examples
/// ```
/// use std::path::Path;
/// use workspace_qdrant_core::classify_file_type;
///
/// let file_type = classify_file_type(Path::new("README.md"));
/// assert_eq!(file_type.as_str(), "text");
///
/// let file_type = classify_file_type(Path::new("main.py"));
/// assert_eq!(file_type.as_str(), "code");
///
/// let file_type = classify_file_type(Path::new("index.html"));
/// assert_eq!(file_type.as_str(), "web");
/// ```
pub fn classify_file_type(file_path: &Path) -> FileType {
    let extension = get_extension(file_path);

    let filename = file_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Priority 1: Configuration dotfiles without extensions
    if classification::is_config_filename(&filename) {
        return FileType::Config;
    }

    // Priority 2: Extension-based lookup from YAML reference
    if let Some(file_type_str) = classification::extension_to_file_type(&extension) {
        // Special handling for JSON: context-aware (config path → config, else → data)
        if extension == ".json" {
            if classification::is_config_path(&file_path.to_string_lossy().to_lowercase()) {
                return FileType::Config;
            }
            return FileType::Data;
        }

        if let Some(ft) = FileType::from_str(file_type_str) {
            return ft;
        }
    }

    // Priority 3: Tarball compound extensions
    let path_str = file_path.to_string_lossy().to_lowercase();
    if classification::is_tarball(&path_str) {
        return FileType::Build;
    }

    // Fallback
    FileType::Other
}

/// Extract the file extension, normalized to lowercase with a leading dot.
///
/// For compound extensions like `.d.ts`, returns `.d.ts` if the stem ends with `.d`.
pub(super) fn get_extension(file_path: &Path) -> String {
    // Check for compound extensions first
    let filename = file_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");

    let lower = filename.to_lowercase();

    // Handle known compound extensions
    if lower.ends_with(".d.ts") || lower.ends_with(".d.mts") || lower.ends_with(".d.cts") {
        // Return the compound extension including the .d part
        let suffix_len = if lower.ends_with(".d.ts") {
            5
        } else if lower.ends_with(".d.mts") {
            6
        } else {
            6 // .d.cts
        };
        return lower[lower.len() - suffix_len..].to_string();
    }

    file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| format!(".{}", ext.to_lowercase()))
        .unwrap_or_default()
}

/// Extract the file extension for storage (lowercase, no leading dot).
///
/// For compound extensions like `.d.ts`, returns `d.ts`.
pub fn get_extension_for_storage(file_path: &Path) -> Option<String> {
    let ext = get_extension(file_path);
    if ext.is_empty() {
        None
    } else {
        // Strip leading dot
        Some(ext[1..].to_string())
    }
}

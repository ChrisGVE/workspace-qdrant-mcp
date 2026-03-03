//! Test file and test directory detection utilities.

use std::path::Path;
use wqm_common::classification;

use super::classify::get_extension;

/// Check if a file is a test file based on naming conventions and path.
///
/// Test detection is independent of file_type — a test file is always also code.
/// Non-code files (e.g., `test_data.txt`) are NOT classified as test files.
///
/// Detects:
/// - Filename patterns: `test_*`, `*_test.*`, `*.test.*`, `*.spec.*`, `conftest.*`
/// - Files under test directories: `tests/`, `test/`, `__tests__/`, `spec/`, `__spec__/`
///
/// Returns true only if the file has a code extension AND matches a test pattern.
pub fn is_test_file(file_path: &Path) -> bool {
    let extension = get_extension(file_path);

    // Must be a code file to be a test
    if !classification::is_file_type(&extension, "code") {
        return false;
    }

    let filename = file_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Check filename patterns
    if has_test_filename_pattern(&filename) {
        return true;
    }

    // Check if under a test directory
    is_in_test_directory(file_path)
}

/// Check if a directory is a test directory.
///
/// Common test directory names:
/// - tests, test, __tests__
/// - spec, specs
/// - integration, e2e, unit
pub fn is_test_directory(directory_path: &Path) -> bool {
    let dir_name = directory_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_lowercase();

    classification::is_test_directory_name(&dir_name)
}

/// Check if a filename matches test file patterns.
fn has_test_filename_pattern(filename: &str) -> bool {
    // Common test file prefixes
    if filename.starts_with("test_") {
        return true;
    }

    // Get filename without extension
    let name_without_ext = if let Some(pos) = filename.rfind('.') {
        &filename[..pos]
    } else {
        filename
    };

    // Common test file suffixes
    if name_without_ext.ends_with("_test") {
        return true;
    }

    // .test. and .spec. patterns (JS/TS ecosystem)
    if filename.contains(".test.") || filename.contains(".spec.") {
        return true;
    }

    // Dot-separated suffixes
    if name_without_ext.ends_with(".test") || name_without_ext.ends_with(".spec") {
        return true;
    }

    // Special test file names (only with code extensions)
    if name_without_ext == "conftest" || name_without_ext == "test" || name_without_ext == "tests" {
        return true;
    }

    false
}

/// Check if a file is under a test directory.
fn is_in_test_directory(file_path: &Path) -> bool {
    for ancestor in file_path.ancestors() {
        let dir_name = ancestor
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("")
            .to_lowercase();

        if classification::is_test_directory_name(&dir_name) {
            return true;
        }
    }
    false
}

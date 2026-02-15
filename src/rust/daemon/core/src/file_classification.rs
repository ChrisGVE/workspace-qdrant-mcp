//! File type classification for metadata-based routing
//!
//! This module provides file type classification to replace collection-based routing
//! with metadata-based differentiation. All extension mappings are derived from the
//! unified YAML reference in `wqm_common::classification`.
//!
//! File Types:
//! - code: Source code files (.py, .rs, .js, etc.)
//! - text: Plain text and lightweight markup (.txt, .md, .rst, .org, .adoc, .tex)
//! - docs: Binary/rich document formats (.pdf, .docx, .odt, .epub, .rtf)
//! - web: Web content files (.html, .css, .scss, .xml)
//! - slides: Presentation formats (.ppt, .pptx, .key, .odp)
//! - config: Configuration files (.yaml, .json, .toml, .ini)
//! - data: Data files (.csv, .parquet, .xlsx, .ipynb)
//! - build: Build artifacts (.whl, .tar.gz, .zip, .jar)
//! - other: Unclassified files
//!
//! Test detection is separate: `is_test_file()` returns a bool independent of file_type.
//! A file can be both `code` and a test file.

use std::path::Path;
use wqm_common::classification;

/// File type classification result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileType {
    /// Source code files
    Code,
    /// Plain text and lightweight markup (.txt, .md, .rst, .org, .adoc, .tex)
    Text,
    /// Binary/rich document formats (.pdf, .docx, .epub, .odt, .rtf)
    Docs,
    /// Web content files (.html, .css, .xml)
    Web,
    /// Presentation formats (.ppt, .pptx, .key, .odp)
    Slides,
    /// Configuration files (.yaml, .json, .toml, .ini)
    Config,
    /// Data files (.csv, .parquet, .xlsx, .ipynb)
    Data,
    /// Build artifacts and build system files
    Build,
    /// Unclassified files
    Other,
}

impl FileType {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            FileType::Code => "code",
            FileType::Text => "text",
            FileType::Docs => "docs",
            FileType::Web => "web",
            FileType::Slides => "slides",
            FileType::Config => "config",
            FileType::Data => "data",
            FileType::Build => "build",
            FileType::Other => "other",
        }
    }

    /// Parse from string representation
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "code" => Some(FileType::Code),
            "text" => Some(FileType::Text),
            "docs" => Some(FileType::Docs),
            "web" => Some(FileType::Web),
            "slides" => Some(FileType::Slides),
            "config" => Some(FileType::Config),
            "data" => Some(FileType::Data),
            "build" => Some(FileType::Build),
            "other" => Some(FileType::Other),
            _ => None,
        }
    }
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Determine file type for metadata classification.
///
/// Classification priority:
/// 1. Configuration dotfiles (by filename)
/// 2. Extension-based lookup from unified YAML reference
/// 3. Special handling for JSON (config vs data based on path)
/// 4. Tarball compound extensions
/// 5. Other (fallback)
///
/// Test detection is **not** part of file_type — use [`is_test_file`] separately.
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

/// Extract the file extension, normalized to lowercase with a leading dot.
///
/// For compound extensions like `.d.ts`, returns `.d.ts` if the stem ends with `.d`.
fn get_extension(file_path: &Path) -> String {
    // Check for compound extensions first
    let filename = file_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");

    let lower = filename.to_lowercase();

    // Handle known compound extensions
    if lower.ends_with(".d.ts") || lower.ends_with(".d.mts") || lower.ends_with(".d.cts") {
        // Return the compound extension including the .d part
        let suffix_len = if lower.ends_with(".d.ts") { 5 }
            else if lower.ends_with(".d.mts") { 6 }
            else { 6 }; // .d.cts
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

/// Check if a filename matches test file patterns
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

/// Check if a file is under a test directory
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

/// Check if a directory is a test directory
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_code_files() {
        assert_eq!(classify_file_type(Path::new("main.py")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("lib.rs")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("app.js")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("component.tsx")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("handler.go")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("script.ps1")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("module.d")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("app.vue")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("page.svelte")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("layout.astro")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("schema.proto")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("main.zig")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("app.dart")), FileType::Code);
    }

    #[test]
    fn test_text_files() {
        assert_eq!(classify_file_type(Path::new("README.md")), FileType::Text);
        assert_eq!(classify_file_type(Path::new("guide.rst")), FileType::Text);
        assert_eq!(classify_file_type(Path::new("notes.txt")), FileType::Text);
        assert_eq!(classify_file_type(Path::new("doc.adoc")), FileType::Text);
        assert_eq!(classify_file_type(Path::new("notes.org")), FileType::Text);
        assert_eq!(classify_file_type(Path::new("paper.tex")), FileType::Text);
    }

    #[test]
    fn test_docs_files() {
        assert_eq!(classify_file_type(Path::new("manual.pdf")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("book.epub")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("report.docx")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("legacy.doc")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("document.odt")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("formatted.rtf")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("notes.pages")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("book.mobi")), FileType::Docs);
    }

    #[test]
    fn test_web_files() {
        assert_eq!(classify_file_type(Path::new("index.html")), FileType::Web);
        assert_eq!(classify_file_type(Path::new("page.htm")), FileType::Web);
        assert_eq!(classify_file_type(Path::new("doc.xhtml")), FileType::Web);
        assert_eq!(classify_file_type(Path::new("styles.css")), FileType::Web);
        assert_eq!(classify_file_type(Path::new("styles.scss")), FileType::Web);
        assert_eq!(classify_file_type(Path::new("styles.less")), FileType::Web);
        assert_eq!(classify_file_type(Path::new("data.xml")), FileType::Web);
    }

    #[test]
    fn test_slides_files() {
        assert_eq!(classify_file_type(Path::new("deck.pptx")), FileType::Slides);
        assert_eq!(classify_file_type(Path::new("legacy.ppt")), FileType::Slides);
        assert_eq!(classify_file_type(Path::new("presentation.key")), FileType::Slides);
        assert_eq!(classify_file_type(Path::new("slides.odp")), FileType::Slides);
    }

    #[test]
    fn test_config_files() {
        assert_eq!(classify_file_type(Path::new("config.yaml")), FileType::Config);
        assert_eq!(classify_file_type(Path::new("settings.toml")), FileType::Config);
        assert_eq!(classify_file_type(Path::new(".env")), FileType::Config);
        assert_eq!(classify_file_type(Path::new("app.ini")), FileType::Config);
    }

    #[test]
    fn test_json_xml_context_aware() {
        // JSON in config location → config
        let config_json = PathBuf::from("/project/config/app.json");
        assert_eq!(classify_file_type(&config_json), FileType::Config);

        // JSON in data location → data
        let data_json = PathBuf::from("/project/data/records.json");
        assert_eq!(classify_file_type(&data_json), FileType::Data);

        // XML → always web (moved from config-dependent)
        let xml = PathBuf::from("/project/exports/data.xml");
        assert_eq!(classify_file_type(&xml), FileType::Web);
    }

    #[test]
    fn test_data_files() {
        assert_eq!(classify_file_type(Path::new("data.csv")), FileType::Data);
        assert_eq!(classify_file_type(Path::new("export.parquet")), FileType::Data);
        assert_eq!(classify_file_type(Path::new("db.sqlite")), FileType::Data);
        assert_eq!(classify_file_type(Path::new("array.npy")), FileType::Data);
    }

    #[test]
    fn test_build_files() {
        assert_eq!(classify_file_type(Path::new("package.whl")), FileType::Build);
        assert_eq!(classify_file_type(Path::new("app.zip")), FileType::Build);
        assert_eq!(classify_file_type(Path::new("lib.so")), FileType::Build);
        assert_eq!(classify_file_type(Path::new("archive.tar.gz")), FileType::Build);
        assert_eq!(classify_file_type(Path::new("backup.tar.bz2")), FileType::Build);
    }

    #[test]
    fn test_other_files() {
        assert_eq!(classify_file_type(Path::new("unknown.xyz")), FileType::Other);
        assert_eq!(classify_file_type(Path::new("data")), FileType::Other);
    }

    // --- Test file detection (separate from file_type) ---

    #[test]
    fn test_is_test_file_by_filename() {
        assert!(is_test_file(Path::new("test_auth.py")));
        assert!(is_test_file(Path::new("main_test.go")));
        assert!(is_test_file(Path::new("app.test.js")));
        assert!(is_test_file(Path::new("component.spec.ts")));
        assert!(is_test_file(Path::new("conftest.py")));
        assert!(is_test_file(Path::new("test_utils.rs")));
    }

    #[test]
    fn test_is_test_file_non_code_excluded() {
        // Non-code files should NOT be classified as test even with test patterns
        assert!(!is_test_file(Path::new("test_data.txt")));
        assert!(!is_test_file(Path::new("test_fixture.json")));
        assert!(!is_test_file(Path::new("test_input.md")));
        assert!(!is_test_file(Path::new("test_config.yaml")));
    }

    #[test]
    fn test_is_test_file_by_directory() {
        assert!(is_test_file(Path::new("/project/tests/helper.py")));
        assert!(is_test_file(Path::new("/project/__tests__/utils.js")));
        assert!(is_test_file(Path::new("/project/spec/models.rb")));

        // Non-code files in test dirs are NOT tests
        assert!(!is_test_file(Path::new("/project/tests/fixture.txt")));
        assert!(!is_test_file(Path::new("/project/__tests__/mock_data.json")));
    }

    #[test]
    fn test_is_test_file_regular_code_not_test() {
        assert!(!is_test_file(Path::new("main.py")));
        assert!(!is_test_file(Path::new("utils.rs")));
        assert!(!is_test_file(Path::new("index.js")));
    }

    #[test]
    fn test_test_files_are_still_code() {
        // A test file should be classified as "code", not "test"
        assert_eq!(classify_file_type(Path::new("test_main.py")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("app.test.js")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("component.spec.ts")), FileType::Code);

        // But they should be detected by is_test_file
        assert!(is_test_file(Path::new("test_main.py")));
        assert!(is_test_file(Path::new("app.test.js")));
        assert!(is_test_file(Path::new("component.spec.ts")));
    }

    // --- Extension extraction ---

    #[test]
    fn test_get_extension_simple() {
        assert_eq!(get_extension(Path::new("main.py")), ".py");
        assert_eq!(get_extension(Path::new("lib.rs")), ".rs");
        assert_eq!(get_extension(Path::new("FILE.HTML")), ".html");
    }

    #[test]
    fn test_get_extension_compound() {
        assert_eq!(get_extension(Path::new("types.d.ts")), ".d.ts");
        assert_eq!(get_extension(Path::new("global.d.mts")), ".d.mts");
        assert_eq!(get_extension(Path::new("index.d.cts")), ".d.cts");
    }

    #[test]
    fn test_get_extension_for_storage() {
        assert_eq!(get_extension_for_storage(Path::new("main.py")), Some("py".to_string()));
        assert_eq!(get_extension_for_storage(Path::new("types.d.ts")), Some("d.ts".to_string()));
        assert_eq!(get_extension_for_storage(Path::new("noext")), None);
    }

    // --- Other ---

    #[test]
    fn test_test_directory_detection() {
        assert!(is_test_directory(Path::new("/project/tests")));
        assert!(is_test_directory(Path::new("/project/test")));
        assert!(is_test_directory(Path::new("/project/__tests__")));
        assert!(is_test_directory(Path::new("/project/spec")));
        assert!(is_test_directory(Path::new("/project/e2e")));
        assert!(is_test_directory(Path::new("/project/integration")));

        assert!(!is_test_directory(Path::new("/project/src")));
        assert!(!is_test_directory(Path::new("/project/lib")));
    }

    #[test]
    fn test_file_type_as_str() {
        assert_eq!(FileType::Code.as_str(), "code");
        assert_eq!(FileType::Text.as_str(), "text");
        assert_eq!(FileType::Docs.as_str(), "docs");
        assert_eq!(FileType::Web.as_str(), "web");
        assert_eq!(FileType::Slides.as_str(), "slides");
        assert_eq!(FileType::Config.as_str(), "config");
        assert_eq!(FileType::Data.as_str(), "data");
        assert_eq!(FileType::Build.as_str(), "build");
        assert_eq!(FileType::Other.as_str(), "other");
    }

    #[test]
    fn test_xml_is_web_not_config() {
        // XML is now always classified as web, not config
        assert_eq!(classify_file_type(Path::new("data.xml")), FileType::Web);
        // Even in config location, XML → web (it's a markup language)
        assert_eq!(classify_file_type(&PathBuf::from("/project/.github/workflow.xml")), FileType::Web);
    }

    #[test]
    fn test_compound_extension_classification() {
        // .d.ts files should be classified as code (TypeScript declarations)
        assert_eq!(classify_file_type(Path::new("types.d.ts")), FileType::Code);
        assert_eq!(classify_file_type(Path::new("global.d.mts")), FileType::Code);
    }
}

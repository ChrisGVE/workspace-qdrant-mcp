//! File type classification for metadata-based routing
//!
//! This module provides file type classification to replace collection-based routing
//! with metadata-based differentiation. Mirrors the Python implementation in
//! `src/python/common/utils/file_type_classifier.py`.
//!
//! File Types:
//! - code: Source code files (.py, .rs, .js, etc.)
//! - test: Test files (test_*, *_test.*, spec.*)
//! - docs: Documentation (.md, .rst, .txt, .adoc)
//! - config: Configuration files (.yaml, .json, .toml, .ini)
//! - data: Data files (.csv, .parquet, .json, .xml)
//! - build: Build artifacts (.whl, .tar.gz, .zip, .jar)
//! - other: Unclassified files

use std::path::Path;
use phf::phf_set;

/// Code file extensions
static CODE_EXTENSIONS: phf::Set<&'static str> = phf_set! {
    ".py", ".pyx", ".pyi",  // Python
    ".rs",  // Rust
    ".js", ".jsx", ".mjs", ".cjs",  // JavaScript
    ".ts", ".tsx", ".d.ts",  // TypeScript
    ".go",  // Go
    ".java", ".kt", ".scala",  // JVM languages
    ".c", ".cpp", ".cxx", ".cc", ".h", ".hpp", ".hxx",  // C/C++
    ".cs", ".fs", ".vb",  // .NET languages
    ".rb", ".erb",  // Ruby
    ".php", ".phtml",  // PHP
    ".swift",  // Swift
    ".m", ".mm",  // Objective-C
    ".sh", ".bash", ".zsh", ".fish",  // Shell scripts
    ".sql", ".ddl", ".dml",  // SQL
    ".r", ".R",  // R
    ".jl",  // Julia
    ".hs",  // Haskell
    ".erl", ".ex", ".exs",  // Erlang/Elixir
    ".clj", ".cljs",  // Clojure
    ".ml", ".mli",  // OCaml
    ".lua",  // Lua
    ".vim",  // Vimscript
    ".el",  // Emacs Lisp
};

/// Documentation file extensions
static DOCS_EXTENSIONS: phf::Set<&'static str> = phf_set! {
    ".md", ".markdown",  // Markdown
    ".rst", ".rest",  // reStructuredText
    ".txt", ".text",  // Plain text
    ".adoc", ".asciidoc",  // AsciiDoc
    ".org",  // Org mode
    ".tex",  // LaTeX
    ".pdf",  // PDF
    ".epub",  // EPUB
    ".docx", ".doc",  // Word
    ".odt",  // OpenDocument Text
    ".rtf",  // Rich Text Format
};

/// Configuration file extensions
static CONFIG_EXTENSIONS: phf::Set<&'static str> = phf_set! {
    ".yaml", ".yml",  // YAML
    ".json", ".jsonc", ".json5",  // JSON
    ".toml",  // TOML
    ".ini",  // INI
    ".conf", ".cfg", ".config",  // Generic config
    ".env",  // Environment files
    ".properties",  // Java properties
    ".xml",  // XML (can be config)
    ".plist",  // macOS property list
    ".editorconfig",  // EditorConfig
    ".gitconfig", ".gitignore", ".gitattributes",  // Git config
};

/// Configuration file names without extensions
static CONFIG_FILENAMES: phf::Set<&'static str> = phf_set! {
    ".env", ".env.local", ".env.example",
    ".editorconfig",
    ".gitconfig", ".gitignore", ".gitattributes",
    ".npmrc", ".dockerignore",
};

/// Data file extensions
static DATA_EXTENSIONS: phf::Set<&'static str> = phf_set! {
    ".csv", ".tsv",  // CSV/TSV
    ".parquet",  // Parquet
    ".json", ".jsonl", ".ndjson",  // JSON data (overlaps with config, intent-based)
    ".xml",  // XML data (overlaps with config)
    ".arrow",  // Apache Arrow
    ".feather",  // Feather format
    ".hdf5", ".h5",  // HDF5
    ".db", ".sqlite", ".sqlite3",  // SQLite
    ".pkl", ".pickle",  // Python pickle
    ".npy", ".npz",  // NumPy arrays
    ".mat",  // MATLAB data
    ".rds", ".rdata",  // R data
};

/// Build artifact extensions
static BUILD_EXTENSIONS: phf::Set<&'static str> = phf_set! {
    ".whl",  // Python wheel
    ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz",  // Tarballs
    ".zip",  // ZIP
    ".jar", ".war", ".ear",  // Java archives
    ".so", ".dylib", ".dll",  // Shared libraries
    ".a", ".lib",  // Static libraries
    ".o", ".obj",  // Object files
    ".exe", ".app",  // Executables
    ".deb", ".rpm",  // Package formats
    ".dmg",  // macOS disk image
    ".iso",  // ISO image
};

/// File type classification result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileType {
    /// Source code files
    Code,
    /// Test files
    Test,
    /// Documentation files
    Docs,
    /// Configuration files
    Config,
    /// Data files
    Data,
    /// Build artifacts
    Build,
    /// Unclassified files
    Other,
}

impl FileType {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            FileType::Code => "code",
            FileType::Test => "test",
            FileType::Docs => "docs",
            FileType::Config => "config",
            FileType::Data => "data",
            FileType::Build => "build",
            FileType::Other => "other",
        }
    }
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Determine file type for metadata classification
///
/// Classification priority:
/// 1. Test files (checked first to avoid misclassification as code)
/// 2. Documentation files
/// 3. Configuration files
/// 4. Code files
/// 5. Data files
/// 6. Build artifacts
/// 7. Other (fallback)
///
/// # Arguments
/// * `file_path` - Path to the file to classify
///
/// # Returns
/// FileType enum indicating the file category
///
/// # Examples
/// ```
/// use std::path::Path;
/// use workspace_qdrant_daemon_core::classify_file_type;
///
/// let file_type = classify_file_type(Path::new("test_auth.py"));
/// assert_eq!(file_type.as_str(), "test");
///
/// let file_type = classify_file_type(Path::new("README.md"));
/// assert_eq!(file_type.as_str(), "docs");
///
/// let file_type = classify_file_type(Path::new("main.py"));
/// assert_eq!(file_type.as_str(), "code");
/// ```
pub fn classify_file_type(file_path: &Path) -> FileType {
    // Get file extension and name
    let extension = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| format!(".{}", ext.to_lowercase()))
        .unwrap_or_default();

    let filename = file_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_lowercase();

    // Priority 1: Test files (must check before code classification)
    if is_test_file(&filename, &extension) {
        return FileType::Test;
    }

    // Priority 2: Configuration dotfiles without extensions
    if CONFIG_FILENAMES.contains(filename.as_str()) {
        return FileType::Config;
    }

    // Priority 3: Documentation
    if DOCS_EXTENSIONS.contains(extension.as_str()) {
        return FileType::Docs;
    }

    // Priority 4: Configuration
    // Special handling for JSON/XML which can be data or config
    if CONFIG_EXTENSIONS.contains(extension.as_str()) {
        // If it's JSON/XML in typical config locations, classify as config
        if extension == ".json" || extension == ".xml" {
            // Check parent directory names for config indicators
            if is_config_location(file_path) {
                return FileType::Config;
            }
            // Otherwise, treat as data
            return FileType::Data;
        }
        return FileType::Config;
    }

    // Priority 5: Code files
    if CODE_EXTENSIONS.contains(extension.as_str()) {
        return FileType::Code;
    }

    // Priority 6: Data files
    if DATA_EXTENSIONS.contains(extension.as_str()) {
        return FileType::Data;
    }

    // Priority 7: Build artifacts
    if BUILD_EXTENSIONS.contains(extension.as_str()) {
        return FileType::Build;
    }

    // Handle compound extensions like .tar.gz
    let path_str = file_path.to_string_lossy().to_lowercase();
    if path_str.ends_with(".tar.gz") || path_str.ends_with(".tar.bz2") || path_str.ends_with(".tar.xz") {
        return FileType::Build;
    }

    // Priority 8: Fallback to "other"
    FileType::Other
}

/// Check if file is a test file based on naming conventions
///
/// Detects common test file patterns:
/// - test_*.py, test_*.rs, etc.
/// - *_test.py, *_test.rs, etc.
/// - *.test.js, *.spec.ts, etc.
/// - Tests in __tests__ or test directories are handled by path logic
fn is_test_file(filename: &str, extension: &str) -> bool {
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

    if name_without_ext.ends_with(".test") || name_without_ext.ends_with(".spec") {
        return true;
    }

    // Spec files (common in JS/TS ecosystems)
    if filename.contains(".spec.") || filename.contains(".test.") {
        return true;
    }

    // Special test file names
    if CODE_EXTENSIONS.contains(extension)
        && (name_without_ext == "conftest" || name_without_ext == "test" || name_without_ext == "tests") {
            return true;
        }

    false
}

/// Check if a path is in a typical configuration location
fn is_config_location(file_path: &Path) -> bool {
    let path_str = file_path.to_string_lossy().to_lowercase();

    // Config directory indicators
    let config_indicators = [
        "/config/",
        "/conf/",
        "/settings/",
        "/.github/",
        "/.vscode/",
        "/etc/",
    ];

    for indicator in &config_indicators {
        if path_str.contains(indicator) {
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
///
/// # Arguments
/// * `directory_path` - Path to directory
///
/// # Returns
/// True if directory name suggests it contains tests, False otherwise
pub fn is_test_directory(directory_path: &Path) -> bool {
    let dir_name = directory_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_lowercase();

    matches!(
        dir_name.as_str(),
        "tests" | "test" | "__tests__" | "spec" | "specs" | "integration" | "e2e" | "unit" | "functional" | "acceptance"
    )
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
    }

    #[test]
    fn test_test_files() {
        assert_eq!(classify_file_type(Path::new("test_auth.py")), FileType::Test);
        assert_eq!(classify_file_type(Path::new("main_test.go")), FileType::Test);
        assert_eq!(classify_file_type(Path::new("app.test.js")), FileType::Test);
        assert_eq!(classify_file_type(Path::new("component.spec.ts")), FileType::Test);
        assert_eq!(classify_file_type(Path::new("conftest.py")), FileType::Test);
    }

    #[test]
    fn test_docs_files() {
        assert_eq!(classify_file_type(Path::new("README.md")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("guide.rst")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("notes.txt")), FileType::Docs);
        assert_eq!(classify_file_type(Path::new("manual.pdf")), FileType::Docs);
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
        // JSON in config location should be config
        let config_json = PathBuf::from("/project/config/app.json");
        assert_eq!(classify_file_type(&config_json), FileType::Config);

        // JSON in data location should be data
        let data_json = PathBuf::from("/project/data/records.json");
        assert_eq!(classify_file_type(&data_json), FileType::Data);

        // XML in config location should be config
        let config_xml = PathBuf::from("/project/.github/workflow.xml");
        assert_eq!(classify_file_type(&config_xml), FileType::Config);

        // XML in other location should be data
        let data_xml = PathBuf::from("/project/exports/data.xml");
        assert_eq!(classify_file_type(&data_xml), FileType::Data);
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
    fn test_priority_test_over_code() {
        // test_*.py should be classified as test, not code
        assert_eq!(classify_file_type(Path::new("test_main.py")), FileType::Test);
        assert_eq!(classify_file_type(Path::new("test_utils.rs")), FileType::Test);

        // But main.py should be code
        assert_eq!(classify_file_type(Path::new("main.py")), FileType::Code);
    }

    #[test]
    fn test_file_type_as_str() {
        assert_eq!(FileType::Code.as_str(), "code");
        assert_eq!(FileType::Test.as_str(), "test");
        assert_eq!(FileType::Docs.as_str(), "docs");
        assert_eq!(FileType::Config.as_str(), "config");
        assert_eq!(FileType::Data.as_str(), "data");
        assert_eq!(FileType::Build.as_str(), "build");
        assert_eq!(FileType::Other.as_str(), "other");
    }
}

//! Unified content classification from embedded YAML reference
//!
//! Provides extension→language, extension→file_type, and related lookups
//! derived from a single YAML source of truth (`content_classification.yaml`).
//! The YAML is embedded at compile time and parsed lazily on first access.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

/// Raw YAML string, embedded at compile time.
pub static CLASSIFICATION_YAML: &str =
    include_str!("content_classification.yaml");

/// Parsed classification data, initialized once on first access.
static CLASSIFICATION: LazyLock<ClassificationData> = LazyLock::new(|| {
    let raw: RawClassification =
        serde_yml::from_str(CLASSIFICATION_YAML).expect("content_classification.yaml must parse");
    ClassificationData::from_raw(raw)
});

// ─── Raw YAML deserialization structs ────────────────────────────────────────

#[derive(Deserialize)]
struct RawClassification {
    compound_extensions: HashMap<String, String>,
    extensions: HashMap<String, ExtensionEntry>,
    config_filenames: Vec<String>,
    config_path_indicators: Vec<String>,
    test_directories: Vec<String>,
    tarball_suffixes: Vec<String>,
}

#[derive(Deserialize)]
struct ExtensionEntry {
    language: Option<String>,
    file_type: String,
    document_type: Option<String>,
}

// ─── Processed lookup data ──────────────────────────────────────────────────

/// Pre-processed classification data with fast lookup maps.
struct ClassificationData {
    /// Extension (without dot, lowercase) → language name
    ext_to_language: HashMap<String, String>,
    /// Extension (without dot, lowercase) → file_type string
    ext_to_file_type: HashMap<String, String>,
    /// Extension (without dot, lowercase) → document_type override string
    ext_to_document_type: HashMap<String, String>,
    /// Compound extension suffix → language
    compound_ext_to_language: HashMap<String, String>,
    /// Set of extensions per file_type category
    file_type_sets: HashMap<String, HashSet<String>>,
    /// Config filenames (lowercase)
    config_filenames: HashSet<String>,
    /// Config path indicators
    config_path_indicators: Vec<String>,
    /// Test directory names (lowercase)
    test_directories: HashSet<String>,
    /// Tarball suffixes
    tarball_suffixes: Vec<String>,
}

impl ClassificationData {
    fn from_raw(raw: RawClassification) -> Self {
        let mut ext_to_language = HashMap::new();
        let mut ext_to_file_type = HashMap::new();
        let mut ext_to_document_type = HashMap::new();
        let mut file_type_sets: HashMap<String, HashSet<String>> = HashMap::new();

        for (ext, entry) in &raw.extensions {
            let ext_lower = ext.to_lowercase();
            let dotted = format!(".{}", ext_lower);

            if let Some(lang) = &entry.language {
                ext_to_language.insert(ext_lower.clone(), lang.clone());
                ext_to_language.insert(dotted.clone(), lang.clone());
            }

            ext_to_file_type.insert(ext_lower.clone(), entry.file_type.clone());
            ext_to_file_type.insert(dotted.clone(), entry.file_type.clone());

            if let Some(dt) = &entry.document_type {
                ext_to_document_type.insert(ext_lower.clone(), dt.clone());
                ext_to_document_type.insert(dotted.clone(), dt.clone());
            }

            file_type_sets
                .entry(entry.file_type.clone())
                .or_default()
                .insert(dotted);
        }

        let config_filenames: HashSet<String> =
            raw.config_filenames.into_iter().map(|s| s.to_lowercase()).collect();

        let test_directories: HashSet<String> =
            raw.test_directories.into_iter().map(|s| s.to_lowercase()).collect();

        Self {
            ext_to_language,
            ext_to_file_type,
            ext_to_document_type,
            compound_ext_to_language: raw.compound_extensions,
            file_type_sets,
            config_filenames,
            config_path_indicators: raw.config_path_indicators,
            test_directories,
            tarball_suffixes: raw.tarball_suffixes,
        }
    }
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Look up the programming language for a file extension.
///
/// Accepts extensions with or without leading dot (e.g., `"rs"` or `".rs"`).
/// Returns `None` for non-code extensions (text, docs, data, etc.).
pub fn extension_to_language(ext: &str) -> Option<&str> {
    CLASSIFICATION.ext_to_language.get(&ext.to_lowercase()).map(|s| s.as_str())
}

/// Look up the file_type category for a file extension.
///
/// Accepts extensions with or without leading dot.
/// Returns `None` if the extension is not in the classification data.
pub fn extension_to_file_type(ext: &str) -> Option<&str> {
    CLASSIFICATION.ext_to_file_type.get(&ext.to_lowercase()).map(|s| s.as_str())
}

/// Look up the document_type override for a file extension.
///
/// Returns the document_type string (e.g., "pdf", "markdown", "csv") if one
/// is explicitly defined. For code extensions without a document_type override,
/// returns `None` (caller should use `DocumentType::Code(language)`).
pub fn extension_to_document_type(ext: &str) -> Option<&str> {
    CLASSIFICATION.ext_to_document_type.get(&ext.to_lowercase()).map(|s| s.as_str())
}

/// Check if a compound extension suffix maps to a language.
///
/// The suffix should NOT include a leading dot (e.g., `"d.ts"` not `".d.ts"`).
pub fn compound_extension_language(suffix: &str) -> Option<&str> {
    CLASSIFICATION.compound_ext_to_language.get(suffix).map(|s| s.as_str())
}

/// Check if a dotted extension belongs to the given file_type category.
///
/// The extension should include a leading dot (e.g., `".rs"`).
pub fn is_file_type(dotted_ext: &str, file_type: &str) -> bool {
    CLASSIFICATION
        .file_type_sets
        .get(file_type)
        .map(|set| set.contains(&dotted_ext.to_lowercase()))
        .unwrap_or(false)
}

/// Check if a filename (lowercase) is a known configuration filename.
pub fn is_config_filename(filename: &str) -> bool {
    CLASSIFICATION.config_filenames.contains(&filename.to_lowercase())
}

/// Check if a path string contains a configuration path indicator.
pub fn is_config_path(path_lower: &str) -> bool {
    CLASSIFICATION
        .config_path_indicators
        .iter()
        .any(|indicator| path_lower.contains(indicator))
}

/// Check if a directory name is a known test directory.
pub fn is_test_directory_name(dir_name: &str) -> bool {
    CLASSIFICATION.test_directories.contains(&dir_name.to_lowercase())
}

/// Check if a lowercased path ends with a tarball suffix.
pub fn is_tarball(path_lower: &str) -> bool {
    CLASSIFICATION
        .tarball_suffixes
        .iter()
        .any(|suffix| path_lower.ends_with(suffix))
}

/// Get all extensions for a given file_type category.
///
/// Returns dotted extensions (e.g., `".rs"`, `".py"`).
pub fn extensions_for_file_type(file_type: &str) -> Vec<&str> {
    CLASSIFICATION
        .file_type_sets
        .get(file_type)
        .map(|set| set.iter().map(|s| s.as_str()).collect())
        .unwrap_or_default()
}

/// Get all known compound extension suffixes.
pub fn compound_extensions() -> Vec<(&'static str, &'static str)> {
    CLASSIFICATION
        .compound_ext_to_language
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yaml_parses_successfully() {
        // Force initialization to verify YAML is valid
        let _ = &*CLASSIFICATION;
    }

    #[test]
    fn test_extension_to_language_code() {
        assert_eq!(extension_to_language("rs"), Some("rust"));
        assert_eq!(extension_to_language(".rs"), Some("rust"));
        assert_eq!(extension_to_language("py"), Some("python"));
        assert_eq!(extension_to_language("js"), Some("javascript"));
        assert_eq!(extension_to_language("ts"), Some("typescript"));
        assert_eq!(extension_to_language("go"), Some("go"));
        assert_eq!(extension_to_language("java"), Some("java"));
    }

    #[test]
    fn test_extension_to_language_case_insensitive() {
        assert_eq!(extension_to_language("RS"), Some("rust"));
        assert_eq!(extension_to_language(".PY"), Some("python"));
    }

    #[test]
    fn test_extension_to_language_non_code() {
        assert_eq!(extension_to_language("txt"), None);
        assert_eq!(extension_to_language("pdf"), None);
        assert_eq!(extension_to_language("csv"), None);
    }

    #[test]
    fn test_extension_to_file_type() {
        assert_eq!(extension_to_file_type("rs"), Some("code"));
        assert_eq!(extension_to_file_type("md"), Some("text"));
        assert_eq!(extension_to_file_type("pdf"), Some("docs"));
        assert_eq!(extension_to_file_type("html"), Some("web"));
        assert_eq!(extension_to_file_type("pptx"), Some("slides"));
        assert_eq!(extension_to_file_type("yaml"), Some("config"));
        assert_eq!(extension_to_file_type("csv"), Some("data"));
        assert_eq!(extension_to_file_type("whl"), Some("build"));
    }

    #[test]
    fn test_extension_to_document_type() {
        assert_eq!(extension_to_document_type("pdf"), Some("pdf"));
        assert_eq!(extension_to_document_type("md"), Some("markdown"));
        assert_eq!(extension_to_document_type("txt"), Some("text"));
        assert_eq!(extension_to_document_type("csv"), Some("csv"));
        assert_eq!(extension_to_document_type("xlsx"), Some("xlsx"));
        // Code extensions don't have document_type overrides
        assert_eq!(extension_to_document_type("rs"), None);
        assert_eq!(extension_to_document_type("py"), None);
    }

    #[test]
    fn test_compound_extension_language() {
        assert_eq!(compound_extension_language("d.ts"), Some("typescript"));
        assert_eq!(compound_extension_language("d.mts"), Some("typescript"));
        assert_eq!(compound_extension_language("d.cts"), Some("typescript"));
        assert_eq!(compound_extension_language("ts"), None);
    }

    #[test]
    fn test_is_file_type() {
        assert!(is_file_type(".rs", "code"));
        assert!(is_file_type(".py", "code"));
        assert!(!is_file_type(".rs", "text"));
        assert!(is_file_type(".md", "text"));
        assert!(is_file_type(".pdf", "docs"));
    }

    #[test]
    fn test_is_config_filename() {
        assert!(is_config_filename(".env"));
        assert!(is_config_filename(".gitignore"));
        assert!(is_config_filename(".npmrc"));
        assert!(!is_config_filename("main.rs"));
    }

    #[test]
    fn test_is_config_path() {
        assert!(is_config_path("/project/config/app.json"));
        assert!(is_config_path("/project/.github/workflow.yml"));
        assert!(!is_config_path("/project/src/main.rs"));
    }

    #[test]
    fn test_is_test_directory_name() {
        assert!(is_test_directory_name("tests"));
        assert!(is_test_directory_name("__tests__"));
        assert!(is_test_directory_name("spec"));
        assert!(is_test_directory_name("e2e"));
        assert!(!is_test_directory_name("src"));
    }

    #[test]
    fn test_is_tarball() {
        assert!(is_tarball("archive.tar.gz"));
        assert!(is_tarball("package.tar.bz2"));
        assert!(is_tarball("release.tar.xz"));
        assert!(!is_tarball("file.zip"));
    }

    #[test]
    fn test_web_extensions_have_language() {
        assert_eq!(extension_to_language("html"), Some("html"));
        assert_eq!(extension_to_language("css"), Some("css"));
        assert_eq!(extension_to_language("xml"), Some("xml"));
    }

    #[test]
    fn test_config_extensions_with_language() {
        assert_eq!(extension_to_language("yaml"), Some("yaml"));
        assert_eq!(extension_to_language("json"), Some("json"));
        assert_eq!(extension_to_language("toml"), Some("toml"));
    }

    #[test]
    fn test_all_code_extensions_have_language() {
        // Every extension with file_type=code should have a language
        let code_exts = extensions_for_file_type("code");
        for ext in code_exts {
            assert!(
                extension_to_language(ext).is_some(),
                "Code extension {} should have a language mapping",
                ext
            );
        }
    }

    #[test]
    fn test_extensions_for_file_type() {
        let code_exts = extensions_for_file_type("code");
        assert!(code_exts.contains(&".rs"));
        assert!(code_exts.contains(&".py"));
        assert!(!code_exts.contains(&".pdf"));
    }
}

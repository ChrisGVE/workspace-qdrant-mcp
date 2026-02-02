//! Grammar cache directory structure and metadata management.
//!
//! This module manages the directory structure for dynamically loaded tree-sitter grammars:
//!
//! ```text
//! ~/.workspace-qdrant/grammars/
//! └── <platform>/                         # e.g., x86_64-apple-darwin
//!     └── <tree_sitter_version>/          # e.g., 0.24.0
//!         └── <language>/                 # e.g., rust
//!             ├── grammar.so              # The grammar library (or .dylib on macOS)
//!             └── metadata.json           # Version and checksum metadata
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Metadata for a cached grammar file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GrammarMetadata {
    /// Language identifier (e.g., "rust", "python")
    pub language: String,
    /// Tree-sitter runtime version this grammar was built for
    pub tree_sitter_version: String,
    /// Version of the grammar crate (e.g., "0.23.0")
    pub grammar_version: String,
    /// Platform triple (e.g., "x86_64-apple-darwin")
    pub platform: String,
    /// SHA256 checksum of the grammar library file
    pub checksum: String,
    /// Download URL if available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download_url: Option<String>,
    /// Timestamp when grammar was cached (ISO 8601)
    pub cached_at: String,
}

impl GrammarMetadata {
    /// Create new metadata for a grammar.
    pub fn new(
        language: impl Into<String>,
        tree_sitter_version: impl Into<String>,
        grammar_version: impl Into<String>,
        platform: impl Into<String>,
        checksum: impl Into<String>,
    ) -> Self {
        Self {
            language: language.into(),
            tree_sitter_version: tree_sitter_version.into(),
            grammar_version: grammar_version.into(),
            platform: platform.into(),
            checksum: checksum.into(),
            download_url: None,
            cached_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Directory structure for grammar cache.
#[derive(Debug, Clone)]
pub struct GrammarCachePaths {
    /// Root directory for grammar cache (default: ~/.workspace-qdrant/grammars)
    pub root: PathBuf,
    /// Current platform triple
    pub platform: String,
    /// Current tree-sitter runtime version
    pub tree_sitter_version: String,
}

impl GrammarCachePaths {
    /// Create paths with default root directory.
    pub fn new(tree_sitter_version: impl Into<String>) -> Self {
        let root = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".workspace-qdrant")
            .join("grammars");

        Self {
            root,
            platform: current_platform(),
            tree_sitter_version: tree_sitter_version.into(),
        }
    }

    /// Create paths with custom root directory.
    pub fn with_root(root: impl Into<PathBuf>, tree_sitter_version: impl Into<String>) -> Self {
        Self {
            root: root.into(),
            platform: current_platform(),
            tree_sitter_version: tree_sitter_version.into(),
        }
    }

    /// Get the platform-specific directory.
    ///
    /// Returns: `<root>/<platform>/`
    pub fn platform_dir(&self) -> PathBuf {
        self.root.join(&self.platform)
    }

    /// Get the version-specific directory.
    ///
    /// Returns: `<root>/<platform>/<tree_sitter_version>/`
    pub fn version_dir(&self) -> PathBuf {
        self.platform_dir().join(&self.tree_sitter_version)
    }

    /// Get the language-specific directory.
    ///
    /// Returns: `<root>/<platform>/<tree_sitter_version>/<language>/`
    pub fn language_dir(&self, language: &str) -> PathBuf {
        self.version_dir().join(language)
    }

    /// Get the path to the grammar library file.
    ///
    /// Returns: `<root>/<platform>/<tree_sitter_version>/<language>/grammar.<ext>`
    pub fn grammar_path(&self, language: &str) -> PathBuf {
        self.language_dir(language).join(grammar_filename())
    }

    /// Get the path to the metadata file.
    ///
    /// Returns: `<root>/<platform>/<tree_sitter_version>/<language>/metadata.json`
    pub fn metadata_path(&self, language: &str) -> PathBuf {
        self.language_dir(language).join("metadata.json")
    }

    /// Check if a grammar exists in the cache.
    pub fn grammar_exists(&self, language: &str) -> bool {
        self.grammar_path(language).exists()
    }

    /// List all cached languages for current platform and tree-sitter version.
    pub fn list_cached_languages(&self) -> std::io::Result<Vec<String>> {
        let version_dir = self.version_dir();
        if !version_dir.exists() {
            return Ok(vec![]);
        }

        let mut languages = Vec::new();
        for entry in std::fs::read_dir(version_dir)? {
            let entry = entry?;
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    // Verify it has a grammar file
                    let grammar_path = entry.path().join(grammar_filename());
                    if grammar_path.exists() {
                        languages.push(name.to_string());
                    }
                }
            }
        }
        Ok(languages)
    }

    /// Load metadata for a cached grammar.
    pub fn load_metadata(&self, language: &str) -> std::io::Result<Option<GrammarMetadata>> {
        let metadata_path = self.metadata_path(language);
        if !metadata_path.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(&metadata_path)?;
        let metadata: GrammarMetadata = serde_json::from_str(&content).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;
        Ok(Some(metadata))
    }

    /// Save metadata for a grammar.
    pub fn save_metadata(&self, language: &str, metadata: &GrammarMetadata) -> std::io::Result<()> {
        let language_dir = self.language_dir(language);
        std::fs::create_dir_all(&language_dir)?;

        let metadata_path = self.metadata_path(language);
        let content = serde_json::to_string_pretty(metadata).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;
        std::fs::write(&metadata_path, content)
    }

    /// Create all necessary directories for a language grammar.
    pub fn create_directories(&self, language: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(self.language_dir(language))
    }
}

/// Get the current platform triple.
fn current_platform() -> String {
    // Use Rust's built-in target triple
    // This matches what cargo uses for building
    format!("{}-{}-{}", std::env::consts::ARCH, os_family(), std::env::consts::OS)
}

/// Get the OS family component for platform triple.
fn os_family() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "apple"
    }
    #[cfg(target_os = "linux")]
    {
        "unknown-linux"
    }
    #[cfg(target_os = "windows")]
    {
        "pc-windows"
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "unknown"
    }
}

/// Get the grammar library filename for the current platform.
pub fn grammar_filename() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "grammar.dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "grammar.so"
    }
    #[cfg(target_os = "windows")]
    {
        "grammar.dll"
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "grammar.so"
    }
}

/// Get the tree-sitter runtime version.
///
/// This reads from the tree-sitter crate version at compile time.
pub fn tree_sitter_runtime_version() -> &'static str {
    // The tree-sitter crate version we're compiled against
    // This should match what's in Cargo.toml
    "0.24"
}

/// Compute SHA256 checksum of a file.
///
/// Returns the hex-encoded SHA256 hash of the file contents.
pub fn compute_checksum(path: &std::path::Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};

    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();

    std::io::copy(&mut reader, &mut hasher)?;

    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

/// Verify that a grammar file matches the expected checksum.
///
/// Returns true if the checksum matches, false otherwise.
/// Returns an error if the file cannot be read.
pub fn verify_checksum(
    grammar_path: &std::path::Path,
    expected_checksum: &str,
) -> std::io::Result<bool> {
    let actual = compute_checksum(grammar_path)?;
    Ok(actual == expected_checksum)
}

/// Verify a grammar against its metadata checksum.
///
/// Loads the metadata file from the grammar's directory and verifies
/// that the grammar file's checksum matches the stored checksum.
pub fn verify_grammar_integrity(
    paths: &GrammarCachePaths,
    language: &str,
) -> std::io::Result<bool> {
    let metadata = paths
        .load_metadata(language)?
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Metadata not found"))?;

    let grammar_path = paths.grammar_path(language);
    verify_checksum(&grammar_path, &metadata.checksum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_grammar_metadata_new() {
        let metadata = GrammarMetadata::new(
            "rust",
            "0.24.0",
            "0.23.0",
            "x86_64-apple-darwin",
            "abc123",
        );
        assert_eq!(metadata.language, "rust");
        assert_eq!(metadata.tree_sitter_version, "0.24.0");
        assert_eq!(metadata.grammar_version, "0.23.0");
        assert_eq!(metadata.platform, "x86_64-apple-darwin");
        assert_eq!(metadata.checksum, "abc123");
        assert!(metadata.download_url.is_none());
    }

    #[test]
    fn test_grammar_metadata_serialization() {
        let metadata = GrammarMetadata::new(
            "python",
            "0.24.0",
            "0.23.0",
            "x86_64-unknown-linux-gnu",
            "def456",
        );
        let json = serde_json::to_string(&metadata).unwrap();
        let deserialized: GrammarMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(metadata, deserialized);
    }

    #[test]
    fn test_grammar_cache_paths_structure() {
        let temp_dir = TempDir::new().unwrap();
        let paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24.0");

        // Verify path structure
        let rust_path = paths.grammar_path("rust");
        assert!(rust_path.to_str().unwrap().contains("0.24.0"));
        assert!(rust_path.to_str().unwrap().contains("rust"));
        assert!(rust_path.to_str().unwrap().ends_with(grammar_filename()));

        let metadata_path = paths.metadata_path("rust");
        assert!(metadata_path.to_str().unwrap().ends_with("metadata.json"));
    }

    #[test]
    fn test_grammar_cache_create_directories() {
        let temp_dir = TempDir::new().unwrap();
        let paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24.0");

        paths.create_directories("rust").unwrap();
        assert!(paths.language_dir("rust").exists());
    }

    #[test]
    fn test_grammar_cache_metadata_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24.0");

        let metadata = GrammarMetadata::new(
            "rust",
            "0.24.0",
            "0.23.0",
            &paths.platform,
            "checksum123",
        );

        paths.save_metadata("rust", &metadata).unwrap();
        let loaded = paths.load_metadata("rust").unwrap().unwrap();

        assert_eq!(metadata.language, loaded.language);
        assert_eq!(metadata.tree_sitter_version, loaded.tree_sitter_version);
        assert_eq!(metadata.grammar_version, loaded.grammar_version);
        assert_eq!(metadata.checksum, loaded.checksum);
    }

    #[test]
    fn test_grammar_cache_list_languages() {
        let temp_dir = TempDir::new().unwrap();
        let paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24.0");

        // Create some fake grammar directories
        for lang in &["rust", "python", "go"] {
            paths.create_directories(lang).unwrap();
            // Create a fake grammar file
            std::fs::write(paths.grammar_path(lang), "fake grammar").unwrap();
        }

        let languages = paths.list_cached_languages().unwrap();
        assert_eq!(languages.len(), 3);
        assert!(languages.contains(&"rust".to_string()));
        assert!(languages.contains(&"python".to_string()));
        assert!(languages.contains(&"go".to_string()));
    }

    #[test]
    fn test_grammar_exists() {
        let temp_dir = TempDir::new().unwrap();
        let paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24.0");

        assert!(!paths.grammar_exists("rust"));

        paths.create_directories("rust").unwrap();
        std::fs::write(paths.grammar_path("rust"), "fake grammar").unwrap();

        assert!(paths.grammar_exists("rust"));
    }

    #[test]
    fn test_current_platform_format() {
        let platform = current_platform();
        // Should have format: arch-family-os
        let parts: Vec<&str> = platform.split('-').collect();
        assert!(parts.len() >= 2);
        // First part should be architecture
        assert!(!parts[0].is_empty());
    }

    #[test]
    fn test_tree_sitter_runtime_version() {
        let version = tree_sitter_runtime_version();
        assert!(version.starts_with("0."));
    }

    #[test]
    fn test_compute_checksum() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.txt");
        std::fs::write(&test_file, "test content").unwrap();

        let checksum = compute_checksum(&test_file).unwrap();
        // SHA256 of "test content"
        assert_eq!(checksum.len(), 64); // SHA256 produces 64 hex chars
        assert!(checksum.chars().all(|c| c.is_ascii_hexdigit()));

        // Same content should produce same checksum
        let checksum2 = compute_checksum(&test_file).unwrap();
        assert_eq!(checksum, checksum2);
    }

    #[test]
    fn test_compute_checksum_different_content() {
        let temp_dir = TempDir::new().unwrap();

        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.txt");

        std::fs::write(&file1, "content one").unwrap();
        std::fs::write(&file2, "content two").unwrap();

        let checksum1 = compute_checksum(&file1).unwrap();
        let checksum2 = compute_checksum(&file2).unwrap();

        // Different content should produce different checksums
        assert_ne!(checksum1, checksum2);
    }

    #[test]
    fn test_verify_checksum_valid() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("grammar.so");
        std::fs::write(&test_file, "fake grammar content").unwrap();

        let expected = compute_checksum(&test_file).unwrap();
        assert!(verify_checksum(&test_file, &expected).unwrap());
    }

    #[test]
    fn test_verify_checksum_invalid() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("grammar.so");
        std::fs::write(&test_file, "fake grammar content").unwrap();

        assert!(!verify_checksum(&test_file, "wrong_checksum").unwrap());
    }

    #[test]
    fn test_verify_grammar_integrity() {
        let temp_dir = TempDir::new().unwrap();
        let paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24.0");

        // Create grammar and compute its checksum
        paths.create_directories("rust").unwrap();
        let grammar_path = paths.grammar_path("rust");
        std::fs::write(&grammar_path, "fake rust grammar binary").unwrap();
        let checksum = compute_checksum(&grammar_path).unwrap();

        // Save metadata with correct checksum
        let metadata = GrammarMetadata::new("rust", "0.24.0", "0.23.0", &paths.platform, &checksum);
        paths.save_metadata("rust", &metadata).unwrap();

        // Verify should succeed
        assert!(verify_grammar_integrity(&paths, "rust").unwrap());

        // Modify the grammar file
        std::fs::write(&grammar_path, "modified grammar").unwrap();

        // Verify should now fail
        assert!(!verify_grammar_integrity(&paths, "rust").unwrap());
    }
}

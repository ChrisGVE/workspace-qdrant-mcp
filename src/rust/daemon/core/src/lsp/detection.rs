//! LSP Server Detection Module
//!
//! This module handles automatic detection of LSP servers available on the system
//! through PATH scanning and capability discovery.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use which::which;

use crate::lsp::{Language, LspResult};

/// Information about a detected LSP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedServer {
    /// Server executable name
    pub name: String,
    /// Full path to the executable
    pub path: PathBuf,
    /// Languages supported by this server
    pub languages: Vec<Language>,
    /// Server version if detectable
    pub version: Option<String>,
    /// Server capabilities
    pub capabilities: ServerCapabilities,
    /// Priority ranking for this server type
    pub priority: u8,
}

/// LSP server capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Supports text document synchronization
    pub text_document_sync: bool,
    /// Supports completion
    pub completion: bool,
    /// Supports hover information
    pub hover: bool,
    /// Supports signature help
    pub signature_help: bool,
    /// Supports go to definition
    pub definition: bool,
    /// Supports find references
    pub references: bool,
    /// Supports document highlighting
    pub document_highlight: bool,
    /// Supports document symbols
    pub document_symbol: bool,
    /// Supports workspace symbols
    pub workspace_symbol: bool,
    /// Supports code actions
    pub code_action: bool,
    /// Supports code lens
    pub code_lens: bool,
    /// Supports document formatting
    pub document_formatting: bool,
    /// Supports document range formatting
    pub document_range_formatting: bool,
    /// Supports document on type formatting
    pub document_on_type_formatting: bool,
    /// Supports renaming
    pub rename: bool,
    /// Supports folding ranges
    pub folding_range: bool,
    /// Supports selection ranges
    pub selection_range: bool,
    /// Supports semantic tokens
    pub semantic_tokens: bool,
    /// Supports diagnostics
    pub diagnostics: bool,
}

impl Default for ServerCapabilities {
    fn default() -> Self {
        Self {
            text_document_sync: true,
            completion: true,
            hover: true,
            signature_help: false,
            definition: true,
            references: true,
            document_highlight: false,
            document_symbol: true,
            workspace_symbol: true,
            code_action: false,
            code_lens: false,
            document_formatting: false,
            document_range_formatting: false,
            document_on_type_formatting: false,
            rename: false,
            folding_range: false,
            selection_range: false,
            semantic_tokens: false,
            diagnostics: true,
        }
    }
}

/// LSP server detector that scans the system for available servers
pub struct LspServerDetector {
    /// Known LSP server configurations
    known_servers: HashMap<&'static str, ServerTemplate>,
}

/// Template for known LSP server configurations
#[derive(Debug, Clone)]
struct ServerTemplate {
    /// Executable name to look for
    executable: &'static str,
    /// Languages this server supports
    languages: &'static [Language],
    /// Default capabilities
    capabilities: ServerCapabilities,
    /// Priority (lower = higher priority)
    priority: u8,
    /// Version detection command arguments
    version_args: &'static [&'static str],
}

impl LspServerDetector {
    /// Create a new LSP server detector
    pub fn new() -> Self {
        let mut known_servers = HashMap::new();

        // Python LSP servers
        known_servers.insert("ruff-lsp", ServerTemplate {
            executable: "ruff-lsp",
            languages: &[Language::Python],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                definition: true,
                references: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                document_formatting: true,
                document_range_formatting: true,
                document_on_type_formatting: false,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1, // High priority for Python
            version_args: &["--version"],
        });

        known_servers.insert("pylsp", ServerTemplate {
            executable: "pylsp",
            languages: &[Language::Python],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                document_formatting: true,
                rename: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 2,
            version_args: &["--version"],
        });

        known_servers.insert("pyright-langserver", ServerTemplate {
            executable: "pyright-langserver",
            languages: &[Language::Python],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                rename: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 3,
            version_args: &["--version"],
        });

        // Rust LSP server
        known_servers.insert("rust-analyzer", ServerTemplate {
            executable: "rust-analyzer",
            languages: &[Language::Rust],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                code_lens: true,
                document_formatting: true,
                document_range_formatting: false,
                document_on_type_formatting: false,
                rename: true,
                folding_range: true,
                selection_range: true,
                semantic_tokens: true,
                diagnostics: true,
            },
            priority: 1,
            version_args: &["--version"],
        });

        // TypeScript/JavaScript LSP servers
        known_servers.insert("typescript-language-server", ServerTemplate {
            executable: "typescript-language-server",
            languages: &[Language::TypeScript, Language::JavaScript],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                document_formatting: true,
                rename: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1,
            version_args: &["--version"],
        });

        known_servers.insert("vscode-json-languageserver", ServerTemplate {
            executable: "vscode-json-languageserver",
            languages: &[Language::Json],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                document_symbol: true,
                document_formatting: true,
                document_range_formatting: true,
                document_on_type_formatting: false,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1,
            version_args: &["--version"],
        });

        // C/C++ LSP servers
        known_servers.insert("clangd", ServerTemplate {
            executable: "clangd",
            languages: &[Language::C, Language::Cpp],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                document_formatting: true,
                rename: true,
                semantic_tokens: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1,
            version_args: &["--version"],
        });

        known_servers.insert("ccls", ServerTemplate {
            executable: "ccls",
            languages: &[Language::C, Language::Cpp],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                definition: true,
                references: true,
                document_symbol: true,
                workspace_symbol: true,
                selection_range: false,
                diagnostics: true,
                ..Default::default()
            },
            priority: 2,
            version_args: &["--version"],
        });

        // Go LSP server
        known_servers.insert("gopls", ServerTemplate {
            executable: "gopls",
            languages: &[Language::Go],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                code_lens: true,
                document_formatting: true,
                document_range_formatting: false,
                document_on_type_formatting: false,
                rename: true,
                folding_range: true,
                selection_range: false,
                semantic_tokens: true,
                diagnostics: true,
            },
            priority: 1,
            version_args: &["version"],
        });

        Self { known_servers }
    }

    /// Detect all available LSP servers on the system
    pub async fn detect_servers(&self) -> LspResult<Vec<DetectedServer>> {
        info!("Starting LSP server detection");
        let mut detected = Vec::new();

        for (name, template) in &self.known_servers {
            debug!("Looking for LSP server: {}", name);
            
            match self.detect_server(name, template).await {
                Ok(Some(server)) => {
                    info!("Detected LSP server: {} at {}", server.name, server.path.display());
                    detected.push(server);
                }
                Ok(None) => {
                    debug!("LSP server not found: {}", name);
                }
                Err(e) => {
                    warn!("Error detecting LSP server {}: {}", name, e);
                }
            }
        }

        // Sort by priority and language coverage
        detected.sort_by(|a, b| {
            a.priority.cmp(&b.priority)
                .then_with(|| b.languages.len().cmp(&a.languages.len()))
        });

        info!("Detected {} LSP servers", detected.len());
        Ok(detected)
    }

    /// Detect a specific LSP server
    async fn detect_server(
        &self,
        name: &str,
        template: &ServerTemplate,
    ) -> LspResult<Option<DetectedServer>> {
        // Try to find the executable in PATH
        let path = match which(template.executable) {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };

        // Verify the executable is actually executable
        if !self.is_executable(&path).await? {
            return Ok(None);
        }

        // Try to get version information
        let version = self.get_server_version(&path, template.version_args).await;

        let detected = DetectedServer {
            name: name.to_string(),
            path,
            languages: template.languages.to_vec(),
            version,
            capabilities: template.capabilities.clone(),
            priority: template.priority,
        };

        Ok(Some(detected))
    }

    /// Check if a path is an executable file
    async fn is_executable(&self, path: &Path) -> LspResult<bool> {
        let metadata = tokio::fs::metadata(path).await?;
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = metadata.permissions();
            Ok(metadata.is_file() && (permissions.mode() & 0o111) != 0)
        }
        
        #[cfg(windows)]
        {
            Ok(metadata.is_file())
        }
    }

    /// Try to get version information from an LSP server
    async fn get_server_version(&self, path: &Path, version_args: &[&str]) -> Option<String> {
        let output = Command::new(path)
            .args(version_args)
            .output()
            .ok()?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            
            // Try stdout first, then stderr
            let version_text = if !stdout.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            };

            if !version_text.is_empty() {
                Some(self.extract_version_number(version_text))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Extract version number from version output
    fn extract_version_number(&self, text: &str) -> String {
        // Look for common version patterns
        let version_patterns = [
            regex::Regex::new(r"(\d+\.\d+(?:\.\d+)?)").unwrap(),
            regex::Regex::new(r"version\s+(\d+\.\d+(?:\.\d+)?)").unwrap(),
            regex::Regex::new(r"v(\d+\.\d+(?:\.\d+)?)").unwrap(),
        ];

        for pattern in &version_patterns {
            if let Some(captures) = pattern.captures(text) {
                if let Some(version) = captures.get(1) {
                    return version.as_str().to_string();
                }
            }
        }

        // If no pattern matches, return the first line cleaned up
        text.lines()
            .next()
            .unwrap_or(text)
            .trim()
            .to_string()
    }

    /// Get servers for a specific language, sorted by priority
    pub fn get_servers_for_language(&self, language: &Language) -> Vec<&str> {
        let mut servers: Vec<(&str, u8)> = self.known_servers
            .iter()
            .filter_map(|(name, template)| {
                if template.languages.contains(language) {
                    Some((name.as_ref(), template.priority))
                } else {
                    None
                }
            })
            .collect();

        servers.sort_by_key(|(_, priority)| *priority);
        servers.into_iter().map(|(name, _)| name).collect()
    }

    /// Check if a specific server is known
    pub fn is_known_server(&self, name: &str) -> bool {
        self.known_servers.contains_key(name)
    }

    /// Get template for a known server
    fn get_server_template(&self, name: &str) -> Option<&ServerTemplate> {
        self.known_servers.get(name)
    }
}

impl Default for LspServerDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Project language marker - a file that indicates a specific language is used
#[derive(Debug, Clone)]
pub struct LanguageMarker {
    /// File name to look for (e.g., "Cargo.toml")
    pub filename: &'static str,
    /// Language this marker indicates
    pub language: Language,
    /// Priority - lower is checked first
    pub priority: u8,
}

/// Result of project language detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectLanguageResult {
    /// Languages detected from marker files
    pub marker_languages: Vec<Language>,
    /// Languages detected from file extensions
    pub extension_languages: Vec<Language>,
    /// Combined deduplicated list
    pub all_languages: Vec<Language>,
    /// Whether the result was from cache
    pub from_cache: bool,
}

/// Project language detector - detects programming languages used in a project
///
/// Detection priority:
/// 1. Project marker files (Cargo.toml, package.json, etc.)
/// 2. File extension scanning as fallback
///
/// Results are cached per project_id to avoid repeated scans.
pub struct ProjectLanguageDetector {
    /// Cache of detected languages per project
    cache: RwLock<HashMap<String, ProjectLanguageResult>>,
    /// Known language markers
    markers: Vec<LanguageMarker>,
}

impl ProjectLanguageDetector {
    /// Create a new project language detector
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            markers: vec![
                // Rust
                LanguageMarker {
                    filename: "Cargo.toml",
                    language: Language::Rust,
                    priority: 1,
                },
                // Python
                LanguageMarker {
                    filename: "pyproject.toml",
                    language: Language::Python,
                    priority: 1,
                },
                LanguageMarker {
                    filename: "setup.py",
                    language: Language::Python,
                    priority: 2,
                },
                LanguageMarker {
                    filename: "requirements.txt",
                    language: Language::Python,
                    priority: 3,
                },
                // TypeScript
                LanguageMarker {
                    filename: "tsconfig.json",
                    language: Language::TypeScript,
                    priority: 1,
                },
                // JavaScript/TypeScript (package.json)
                LanguageMarker {
                    filename: "package.json",
                    language: Language::JavaScript,
                    priority: 2,
                },
                // Go
                LanguageMarker {
                    filename: "go.mod",
                    language: Language::Go,
                    priority: 1,
                },
                // Java
                LanguageMarker {
                    filename: "pom.xml",
                    language: Language::Java,
                    priority: 1,
                },
                LanguageMarker {
                    filename: "build.gradle",
                    language: Language::Java,
                    priority: 1,
                },
                LanguageMarker {
                    filename: "build.gradle.kts",
                    language: Language::Java,
                    priority: 1,
                },
                // Ruby
                LanguageMarker {
                    filename: "Gemfile",
                    language: Language::Ruby,
                    priority: 1,
                },
                // PHP
                LanguageMarker {
                    filename: "composer.json",
                    language: Language::Php,
                    priority: 1,
                },
                // C/C++
                LanguageMarker {
                    filename: "CMakeLists.txt",
                    language: Language::Cpp,
                    priority: 1,
                },
                LanguageMarker {
                    filename: "Makefile",
                    language: Language::C,
                    priority: 3,
                },
            ],
        }
    }

    /// Detect languages in a project directory
    ///
    /// First checks for marker files, then scans file extensions as fallback.
    /// Results are cached per project_id.
    pub async fn detect(
        &self,
        project_id: &str,
        project_root: &Path,
    ) -> LspResult<ProjectLanguageResult> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(project_id) {
                debug!(
                    project_id = project_id,
                    "Returning cached language detection result"
                );
                return Ok(ProjectLanguageResult {
                    marker_languages: cached.marker_languages.clone(),
                    extension_languages: cached.extension_languages.clone(),
                    all_languages: cached.all_languages.clone(),
                    from_cache: true,
                });
            }
        }

        // Detect from marker files
        let marker_languages = self.detect_from_markers(project_root).await;

        // Detect from file extensions
        let extension_languages = self.detect_from_extensions(project_root).await;

        // Combine and deduplicate
        let mut all_languages = marker_languages.clone();
        for lang in &extension_languages {
            if !all_languages.iter().any(|l| std::mem::discriminant(l) == std::mem::discriminant(lang)) {
                all_languages.push(lang.clone());
            }
        }

        let result = ProjectLanguageResult {
            marker_languages,
            extension_languages,
            all_languages,
            from_cache: false,
        };

        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(project_id.to_string(), result.clone());
            debug!(
                project_id = project_id,
                languages = ?result.all_languages,
                "Cached language detection result"
            );
        }

        Ok(result)
    }

    /// Detect languages from marker files
    async fn detect_from_markers(&self, project_root: &Path) -> Vec<Language> {
        let mut detected = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Sort markers by priority
        let mut markers = self.markers.clone();
        markers.sort_by_key(|m| m.priority);

        for marker in &markers {
            let marker_path = project_root.join(marker.filename);
            if marker_path.exists() {
                let discriminant = std::mem::discriminant(&marker.language);
                if !seen.contains(&discriminant) {
                    seen.insert(discriminant);

                    // Special handling for package.json - check for TypeScript dep
                    if marker.filename == "package.json" {
                        if let Some(lang) = self.check_package_json(&marker_path).await {
                            detected.push(lang);
                        } else {
                            detected.push(marker.language.clone());
                        }
                    } else {
                        detected.push(marker.language.clone());
                    }

                    info!(
                        marker_file = marker.filename,
                        language = ?marker.language,
                        "Detected language from marker file"
                    );
                }
            }
        }

        detected
    }

    /// Check package.json for TypeScript dependency
    async fn check_package_json(&self, path: &Path) -> Option<Language> {
        let content = tokio::fs::read_to_string(path).await.ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;

        // Check dependencies and devDependencies for typescript
        let has_typescript = ["dependencies", "devDependencies"]
            .iter()
            .filter_map(|key| json.get(key))
            .any(|deps| deps.get("typescript").is_some());

        if has_typescript {
            Some(Language::TypeScript)
        } else {
            None
        }
    }

    /// Detect languages from file extensions
    async fn detect_from_extensions(&self, project_root: &Path) -> Vec<Language> {
        let mut languages = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Use walkdir for synchronous directory walking
        let walker = walkdir::WalkDir::new(project_root)
            .max_depth(5)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                !name.starts_with('.') &&
                !matches!(name.as_ref(),
                    "node_modules" | "target" | "build" | "__pycache__" |
                    "venv" | ".venv" | "dist" | ".git" | "vendor")
            });

        for entry in walker.filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                if let Some(ext) = entry.path().extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    let lang = Language::from_extension(&ext_str);

                    // Only add supported languages (not Other)
                    if !matches!(lang, Language::Other(_)) && !seen.contains(&ext_str) {
                        seen.insert(ext_str);
                        languages.push(lang);
                    }
                }
            }

            // Limit to avoid scanning huge projects
            if seen.len() >= 50 {
                break;
            }
        }

        languages
    }

    /// Clear cache for a specific project
    pub async fn invalidate_cache(&self, project_id: &str) {
        let mut cache = self.cache.write().await;
        cache.remove(project_id);
        debug!(project_id = project_id, "Invalidated language detection cache");
    }

    /// Clear all cached results
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        debug!("Cleared all language detection cache");
    }

    /// Get cached result for a project (if available)
    pub async fn get_cached(&self, project_id: &str) -> Option<ProjectLanguageResult> {
        let cache = self.cache.read().await;
        cache.get(project_id).cloned()
    }
}

impl Default for ProjectLanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_detector_creation() {
        let detector = LspServerDetector::new();
        assert!(!detector.known_servers.is_empty());
    }

    #[test]
    fn test_get_servers_for_language() {
        let detector = LspServerDetector::new();
        let python_servers = detector.get_servers_for_language(&Language::Python);
        assert!(!python_servers.is_empty());
        
        // Should be sorted by priority
        let priorities: Vec<u8> = python_servers.iter()
            .map(|name| detector.get_server_template(name).unwrap().priority)
            .collect();
        
        for i in 1..priorities.len() {
            assert!(priorities[i-1] <= priorities[i]);
        }
    }

    #[test]
    fn test_extract_version_number() {
        let detector = LspServerDetector::new();
        
        assert_eq!(detector.extract_version_number("rust-analyzer 0.3.1"), "0.3.1");
        assert_eq!(detector.extract_version_number("version 2.1.0"), "2.1.0");
        assert_eq!(detector.extract_version_number("v1.2.3"), "1.2.3");
        assert_eq!(detector.extract_version_number("Some program 1.0"), "1.0");
    }

    #[test]
    fn test_is_known_server() {
        let detector = LspServerDetector::new();
        assert!(detector.is_known_server("rust-analyzer"));
        assert!(detector.is_known_server("ruff-lsp"));
        assert!(!detector.is_known_server("unknown-server"));
    }

    #[tokio::test]
    async fn test_detect_servers() {
        let detector = LspServerDetector::new();
        // This test will depend on what's installed on the system
        let result = detector.detect_servers().await;
        assert!(result.is_ok());
    }

    // ProjectLanguageDetector tests

    #[test]
    fn test_project_language_detector_creation() {
        let detector = ProjectLanguageDetector::new();
        assert!(!detector.markers.is_empty());
    }

    #[test]
    fn test_language_markers_include_common_files() {
        let detector = ProjectLanguageDetector::new();
        let marker_filenames: Vec<&str> = detector.markers.iter()
            .map(|m| m.filename)
            .collect();

        // Verify common marker files are present
        assert!(marker_filenames.contains(&"Cargo.toml"));
        assert!(marker_filenames.contains(&"pyproject.toml"));
        assert!(marker_filenames.contains(&"package.json"));
        assert!(marker_filenames.contains(&"go.mod"));
        assert!(marker_filenames.contains(&"pom.xml"));
    }

    #[tokio::test]
    async fn test_project_language_detector_caching() {
        let detector = ProjectLanguageDetector::new();

        // Initially cache should be empty
        assert!(detector.get_cached("test-project").await.is_none());

        // After detection, result should be cached
        let temp_dir = std::env::temp_dir().join("test_project_lang_detect");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Create a Cargo.toml to detect Rust
        let cargo_path = temp_dir.join("Cargo.toml");
        std::fs::write(&cargo_path, "[package]\nname = \"test\"").unwrap();

        // Detect languages
        let result = detector.detect("test-project", &temp_dir).await;
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.from_cache);

        // Second detection should be from cache
        let cached = detector.detect("test-project", &temp_dir).await;
        assert!(cached.is_ok());
        let cached = cached.unwrap();
        assert!(cached.from_cache);

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_project_language_detector_cache_invalidation() {
        let detector = ProjectLanguageDetector::new();

        let temp_dir = std::env::temp_dir().join("test_cache_invalidation");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Detect once to populate cache
        let _ = detector.detect("test-project-inv", &temp_dir).await;
        assert!(detector.get_cached("test-project-inv").await.is_some());

        // Invalidate cache
        detector.invalidate_cache("test-project-inv").await;
        assert!(detector.get_cached("test-project-inv").await.is_none());

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_project_language_detector_rust_project() {
        let detector = ProjectLanguageDetector::new();

        let temp_dir = std::env::temp_dir().join("test_rust_project");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Create a Cargo.toml
        let cargo_path = temp_dir.join("Cargo.toml");
        std::fs::write(&cargo_path, "[package]\nname = \"test\"\nversion = \"0.1.0\"").unwrap();

        // Create a src/main.rs
        let src_dir = temp_dir.join("src");
        let _ = std::fs::create_dir_all(&src_dir);
        std::fs::write(src_dir.join("main.rs"), "fn main() {}").unwrap();

        // Detect languages
        let result = detector.detect("rust-test", &temp_dir).await.unwrap();

        // Should detect Rust from marker
        assert!(result.marker_languages.iter().any(|l| matches!(l, Language::Rust)));

        // All languages should include Rust
        assert!(result.all_languages.iter().any(|l| matches!(l, Language::Rust)));

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_project_language_detector_python_project() {
        let detector = ProjectLanguageDetector::new();

        let temp_dir = std::env::temp_dir().join("test_python_project");
        let _ = std::fs::create_dir_all(&temp_dir);

        // Create a pyproject.toml
        let pyproject_path = temp_dir.join("pyproject.toml");
        std::fs::write(&pyproject_path, "[project]\nname = \"test\"\nversion = \"0.1.0\"").unwrap();

        // Create a main.py
        std::fs::write(temp_dir.join("main.py"), "def main(): pass").unwrap();

        // Detect languages
        let result = detector.detect("python-test", &temp_dir).await.unwrap();

        // Should detect Python from marker
        assert!(result.marker_languages.iter().any(|l| matches!(l, Language::Python)));

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_project_language_result_serialization() {
        let result = ProjectLanguageResult {
            marker_languages: vec![Language::Rust],
            extension_languages: vec![Language::Python, Language::JavaScript],
            all_languages: vec![Language::Rust, Language::Python, Language::JavaScript],
            from_cache: false,
        };

        // Test serialization
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Rust"));

        // Test deserialization
        let deserialized: ProjectLanguageResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.marker_languages.len(), 1);
        assert_eq!(deserialized.all_languages.len(), 3);
    }
}
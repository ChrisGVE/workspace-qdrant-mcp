//! Project Language Detection
//!
//! Detects programming languages used in a project through marker file analysis
//! (e.g., Cargo.toml, package.json) and file extension scanning. Results are
//! cached per project to avoid repeated filesystem scans.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::lsp::{Language, LspResult};

/// Project language marker -- a file that indicates a specific language is used
#[derive(Debug, Clone)]
pub struct LanguageMarker {
    /// File name to look for (e.g., "Cargo.toml")
    pub filename: &'static str,
    /// Language this marker indicates
    pub language: Language,
    /// Priority -- lower is checked first
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

/// Project language detector -- detects programming languages used in a project
///
/// Detection priority:
/// 1. Project marker files (Cargo.toml, package.json, etc.)
/// 2. File extension scanning as fallback
///
/// Results are cached per `project_id` to avoid repeated scans.
pub struct ProjectLanguageDetector {
    /// Cache of detected languages per project
    cache: RwLock<HashMap<String, ProjectLanguageResult>>,
    /// Known language markers
    pub(crate) markers: Vec<LanguageMarker>,
}

impl ProjectLanguageDetector {
    /// Create a new project language detector
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            markers: build_language_markers(),
        }
    }

    /// Detect languages in a project directory
    ///
    /// First checks for marker files, then scans file extensions as fallback.
    /// Results are cached per `project_id`.
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
        let extension_languages = detect_from_extensions(project_root);

        // Combine and deduplicate
        let mut all_languages = marker_languages.clone();
        for lang in &extension_languages {
            if !all_languages
                .iter()
                .any(|l| std::mem::discriminant(l) == std::mem::discriminant(lang))
            {
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

                    // Special handling for package.json -- check for TypeScript dep
                    if marker.filename == "package.json" {
                        if let Some(lang) = check_package_json(&marker_path).await {
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

    /// Clear cache for a specific project
    pub async fn invalidate_cache(&self, project_id: &str) {
        let mut cache = self.cache.write().await;
        cache.remove(project_id);
        debug!(
            project_id = project_id,
            "Invalidated language detection cache"
        );
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

/// Build the list of known language marker files
fn build_language_markers() -> Vec<LanguageMarker> {
    vec![
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
    ]
}

/// Check package.json for TypeScript dependency
async fn check_package_json(path: &Path) -> Option<Language> {
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

/// Detect languages from file extensions in a project directory
fn detect_from_extensions(project_root: &Path) -> Vec<Language> {
    let mut languages = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Use walkdir for synchronous directory walking
    let walker = walkdir::WalkDir::new(project_root)
        .max_depth(5)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !name.starts_with('.')
                && !matches!(
                    name.as_ref(),
                    "node_modules"
                        | "target"
                        | "build"
                        | "__pycache__"
                        | "venv"
                        | ".venv"
                        | "dist"
                        | ".git"
                        | "vendor"
                )
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

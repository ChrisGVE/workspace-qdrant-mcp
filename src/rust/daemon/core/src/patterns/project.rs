//! Project detection system for identifying build systems and project ecosystems
//!
//! This module provides comprehensive project detection capabilities using the
//! embedded configuration data. Detects languages, build systems, frameworks,
//! and project characteristics from file patterns and directory structures.

use super::comprehensive::{ComprehensivePatternManager, ComprehensiveResult};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use once_cell::sync::Lazy;

/// Project detection confidence levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProjectConfidence {
    /// Very high confidence (multiple strong indicators)
    VeryHigh = 100,
    /// High confidence (strong primary indicator)
    High = 80,
    /// Medium confidence (some indicators present)
    Medium = 60,
    /// Low confidence (weak or ambiguous indicators)
    Low = 40,
    /// Unknown (no clear indicators)
    Unknown = 0,
}

/// Detected project information
#[derive(Debug, Clone)]
pub struct ProjectInfo {
    /// Primary language detected
    pub primary_language: Option<String>,
    /// All languages found in the project
    pub languages: Vec<String>,
    /// Build system(s) detected
    pub build_systems: Vec<BuildSystemInfo>,
    /// Frameworks detected
    pub frameworks: Vec<String>,
    /// Project type (e.g., "library", "application", "monorepo")
    pub project_type: ProjectType,
    /// Overall detection confidence
    pub confidence: ProjectConfidence,
    /// Detailed detection reasoning
    pub detection_details: DetectionDetails,
}

/// Build system information
#[derive(Debug, Clone)]
pub struct BuildSystemInfo {
    pub name: String,
    pub language: String,
    pub config_files: Vec<String>,
    pub commands: Vec<String>,
    pub confidence: ProjectConfidence,
}

/// Project type classification
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectType {
    /// Single language application
    Application,
    /// Reusable library or package
    Library,
    /// Multiple projects in one repository
    Monorepo,
    /// Documentation project
    Documentation,
    /// Configuration or infrastructure
    Configuration,
    /// Unknown or mixed type
    Unknown,
}

/// Detailed detection information
#[derive(Debug, Clone)]
pub struct DetectionDetails {
    /// Files analyzed
    pub files_analyzed: usize,
    /// Pattern matches found
    pub pattern_matches: Vec<PatternMatch>,
    /// Detection methods used
    pub methods_used: Vec<String>,
    /// Reasoning for final decision
    pub reasoning: String,
}

/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern: String,
    pub matched_files: Vec<String>,
    pub confidence: ProjectConfidence,
    pub category: String,
}

/// High-performance project detector
#[derive(Debug)]
pub struct ProjectDetector {
    /// Build system patterns for fast lookup
    build_system_patterns: HashMap<String, BuildSystemPattern>,
    /// Project indicator patterns
    _project_indicators: ProjectIndicators,
    /// Language ecosystem patterns
    _language_ecosystems: HashMap<String, EcosystemPattern>,
}

#[derive(Debug, Clone)]
struct BuildSystemPattern {
    _name: String,
    language: String,
    files: Vec<String>,
    commands: Vec<String>,
    weight: u32,
}

#[derive(Debug, Clone)]
struct ProjectIndicators {
    _version_control: Vec<String>,
    _ci_cd: Vec<String>,
    _containerization: Vec<String>,
    _config_management: Vec<String>,
}

#[derive(Debug, Clone)]
struct EcosystemPattern {
    _language: String,
    _indicators: Vec<String>,
    _frameworks: HashMap<String, Vec<String>>,
}

/// Global project detector instance
static PROJECT_DETECTOR: Lazy<Result<ProjectDetector, String>> = Lazy::new(|| {
    ProjectDetector::new().map_err(|e| format!("Failed to initialize project detector: {}", e))
});

impl ProjectDetector {
    /// Create a new project detector
    pub fn new() -> ComprehensiveResult<Self> {
        let comprehensive = ComprehensivePatternManager::new()?;
        let config = comprehensive.config();

        // Build optimized build system patterns
        let mut build_system_patterns = HashMap::new();
        for (name, build_config) in &config.build_systems {
            let pattern = BuildSystemPattern {
                _name: name.clone(),
                language: build_config.language.clone(),
                files: build_config.files.clone(),
                commands: build_config.commands.clone(),
                weight: calculate_build_system_weight(name),
            };
            build_system_patterns.insert(name.clone(), pattern);
        }

        // Build project indicators
        let project_indicators = ProjectIndicators {
            _version_control: config.project_indicators.version_control.clone(),
            _ci_cd: config.project_indicators.ci_cd.clone(),
            _containerization: config.project_indicators.containerization.clone(),
            _config_management: config.project_indicators.config_management.clone(),
        };

        // Build language ecosystem patterns
        let mut language_ecosystems = HashMap::new();

        // Create ecosystem patterns for major languages
        for (_ext, language) in &config.file_extensions {
            if !language_ecosystems.contains_key(language) {
                let indicators = config.project_indicators.language_ecosystems
                    .iter()
                    .filter(|indicator| {
                        // Match indicators that likely belong to this language
                        indicator.contains(&language.to_lowercase()) ||
                        indicator.to_lowercase().contains(&language.to_lowercase())
                    })
                    .cloned()
                    .collect();

                let ecosystem = EcosystemPattern {
                    _language: language.clone(),
                    _indicators: indicators,
                    _frameworks: detect_frameworks_for_language(language),
                };
                language_ecosystems.insert(language.clone(), ecosystem);
            }
        }

        tracing::debug!(
            "Project detector initialized: {} build systems, {} ecosystems",
            build_system_patterns.len(),
            language_ecosystems.len()
        );

        Ok(Self {
            build_system_patterns,
            _project_indicators: project_indicators,
            _language_ecosystems: language_ecosystems,
        })
    }

    /// Get the global project detector instance
    pub fn global() -> Result<&'static ProjectDetector, &'static str> {
        PROJECT_DETECTOR.as_ref().map_err(|e| e.as_str())
    }

    /// Analyze a project directory and detect its characteristics
    pub fn analyze_project(&self, files: &[String]) -> ProjectInfo {
        let mut detection_details = DetectionDetails {
            files_analyzed: files.len(),
            pattern_matches: Vec::new(),
            methods_used: Vec::new(),
            reasoning: String::new(),
        };

        // 1. Detect build systems
        let build_systems = self.detect_build_systems(files, &mut detection_details);
        detection_details.methods_used.push("build_system_detection".to_string());

        // 2. Detect languages from file extensions
        let languages = self.detect_languages(files, &mut detection_details);
        detection_details.methods_used.push("language_detection".to_string());

        // 3. Detect frameworks
        let frameworks = self.detect_frameworks(files, &languages, &mut detection_details);
        detection_details.methods_used.push("framework_detection".to_string());

        // 4. Determine primary language
        let primary_language = self.determine_primary_language(&languages, &build_systems);

        // 5. Classify project type
        let project_type = self.classify_project_type(files, &build_systems, &languages);

        // 6. Calculate overall confidence
        let confidence = self.calculate_overall_confidence(&build_systems, &languages, &frameworks);

        // 7. Generate reasoning
        detection_details.reasoning = self.generate_reasoning(
            &primary_language,
            &languages,
            &build_systems,
            &frameworks,
            &project_type,
        );

        ProjectInfo {
            primary_language,
            languages,
            build_systems,
            frameworks,
            project_type,
            confidence,
            detection_details,
        }
    }

    /// Detect build systems from file patterns
    fn detect_build_systems(&self, files: &[String], details: &mut DetectionDetails) -> Vec<BuildSystemInfo> {
        let mut detected_systems = Vec::new();
        let _file_set: HashSet<&str> = files.iter().map(|f| f.as_str()).collect();

        for (name, pattern) in &self.build_system_patterns {
            let mut matched_files = Vec::new();
            let mut score = 0;

            for file_pattern in &pattern.files {
                for file in files {
                    if file_matches_pattern(file, file_pattern) {
                        matched_files.push(file.clone());
                        score += 1;
                    }
                }
            }

            if !matched_files.is_empty() {
                let confidence = if score >= 2 {
                    ProjectConfidence::VeryHigh
                } else if score == 1 && pattern.weight > 80 {
                    ProjectConfidence::High
                } else {
                    ProjectConfidence::Medium
                };

                let build_system = BuildSystemInfo {
                    name: name.clone(),
                    language: pattern.language.clone(),
                    config_files: matched_files.clone(),
                    commands: pattern.commands.clone(),
                    confidence,
                };

                detected_systems.push(build_system);

                details.pattern_matches.push(PatternMatch {
                    pattern: format!("build_system:{}", name),
                    matched_files,
                    confidence,
                    category: "build_system".to_string(),
                });
            }
        }

        // Sort by confidence and weight
        detected_systems.sort_by(|a, b| {
            b.confidence.cmp(&a.confidence)
                .then_with(|| {
                    let weight_a = self.build_system_patterns.get(&a.name).map(|p| p.weight).unwrap_or(0);
                    let weight_b = self.build_system_patterns.get(&b.name).map(|p| p.weight).unwrap_or(0);
                    weight_b.cmp(&weight_a)
                })
        });

        detected_systems
    }

    /// Detect languages from file extensions
    fn detect_languages(&self, files: &[String], details: &mut DetectionDetails) -> Vec<String> {
        let mut language_counts: HashMap<String, usize> = HashMap::new();

        for file in files {
            if let Some(extension) = Path::new(file).extension().and_then(|e| e.to_str()) {
                // Use the language detector we built earlier
                if let Ok(detector) = super::detection::LanguageDetector::global() {
                    if let Some(language) = detector.detect_from_extension(extension) {
                        *language_counts.entry(language).or_insert(0) += 1;
                    }
                }
            }
        }

        // Convert to sorted list
        let mut languages: Vec<(String, usize)> = language_counts.into_iter().collect();
        languages.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count descending

        let language_names: Vec<String> = languages.iter().map(|(lang, _)| lang.clone()).collect();

        // Add pattern match details
        for (language, count) in languages {
            details.pattern_matches.push(PatternMatch {
                pattern: format!("language:{}", language),
                matched_files: vec![format!("{} files", count)],
                confidence: if count > 10 {
                    ProjectConfidence::VeryHigh
                } else if count > 3 {
                    ProjectConfidence::High
                } else {
                    ProjectConfidence::Medium
                },
                category: "language".to_string(),
            });
        }

        language_names
    }

    /// Detect frameworks based on file patterns and languages
    fn detect_frameworks(&self, files: &[String], _languages: &[String], details: &mut DetectionDetails) -> Vec<String> {
        let mut frameworks = Vec::new();

        // Framework detection patterns
        let framework_patterns = get_framework_patterns();

        for (framework, patterns) in framework_patterns {
            let mut matched = false;
            let mut matched_files = Vec::new();

            for pattern in patterns {
                for file in files {
                    if file_matches_pattern(file, &pattern) {
                        matched = true;
                        matched_files.push(file.clone());
                    }
                }
            }

            if matched {
                frameworks.push(framework.clone());

                details.pattern_matches.push(PatternMatch {
                    pattern: format!("framework:{}", framework),
                    matched_files,
                    confidence: ProjectConfidence::High,
                    category: "framework".to_string(),
                });
            }
        }

        frameworks
    }

    /// Determine the primary language of the project
    fn determine_primary_language(&self, languages: &[String], build_systems: &[BuildSystemInfo]) -> Option<String> {
        // If we have build systems, prefer their language
        if let Some(build_system) = build_systems.first() {
            if build_system.confidence >= ProjectConfidence::High {
                return Some(build_system.language.clone());
            }
        }

        // Otherwise, use the most common language
        languages.first().cloned()
    }

    /// Classify the project type based on detected patterns
    fn classify_project_type(&self, files: &[String], build_systems: &[BuildSystemInfo], languages: &[String]) -> ProjectType {
        // Check for documentation projects
        let doc_files = files.iter().filter(|f| {
            f.ends_with(".md") || f.ends_with(".rst") || f.ends_with(".adoc") ||
            f.contains("docs/") || f.contains("documentation/")
        }).count();

        if doc_files > files.len() / 2 {
            return ProjectType::Documentation;
        }

        // Check for configuration projects
        let config_files = files.iter().filter(|f| {
            f.ends_with(".yml") || f.ends_with(".yaml") || f.ends_with(".toml") ||
            f.ends_with(".json") || f.ends_with(".ini") || f.ends_with(".conf") ||
            f.contains("ansible") || f.contains("terraform") || f.contains("k8s") ||
            f.contains("kubernetes")
        }).count();

        if config_files > files.len() / 3 {
            return ProjectType::Configuration;
        }

        // Check for monorepo (multiple build systems or languages)
        if build_systems.len() > 1 || languages.len() > 2 {
            return ProjectType::Monorepo;
        }

        // Check for library patterns
        let has_lib_indicators = files.iter().any(|f| {
            f.contains("lib/") || f.contains("src/lib") ||
            f.ends_with(".gemspec") || f.contains("setup.py") ||
            f.contains("package.json") && files.iter().any(|other| other.contains("index.js") || other.contains("lib/"))
        });

        if has_lib_indicators {
            return ProjectType::Library;
        }

        // Default to application
        ProjectType::Application
    }

    /// Calculate overall confidence based on detection results
    fn calculate_overall_confidence(&self, build_systems: &[BuildSystemInfo], languages: &[String], frameworks: &[String]) -> ProjectConfidence {
        let mut confidence_score = 0;

        // Build system confidence
        if let Some(build_system) = build_systems.first() {
            confidence_score += build_system.confidence as i32;
        }

        // Language confidence
        if !languages.is_empty() {
            confidence_score += 40; // Base score for language detection
            if languages.len() == 1 {
                confidence_score += 20; // Bonus for single clear language
            }
        }

        // Framework confidence
        if !frameworks.is_empty() {
            confidence_score += 20;
        }

        match confidence_score {
            160.. => ProjectConfidence::VeryHigh,
            120..160 => ProjectConfidence::High,
            80..120 => ProjectConfidence::Medium,
            40..80 => ProjectConfidence::Low,
            _ => ProjectConfidence::Unknown,
        }
    }

    /// Generate human-readable reasoning for the detection
    fn generate_reasoning(&self,
        primary_language: &Option<String>,
        languages: &[String],
        build_systems: &[BuildSystemInfo],
        frameworks: &[String],
        project_type: &ProjectType,
    ) -> String {
        let mut reasoning = Vec::new();

        if let Some(lang) = primary_language {
            reasoning.push(format!("Primary language: {}", lang));
        }

        if languages.len() > 1 {
            reasoning.push(format!("Multi-language project: {}", languages.join(", ")));
        }

        for build_system in build_systems {
            reasoning.push(format!("Build system: {} ({})", build_system.name, build_system.language));
        }

        if !frameworks.is_empty() {
            reasoning.push(format!("Frameworks: {}", frameworks.join(", ")));
        }

        reasoning.push(format!("Project type: {:?}", project_type));

        reasoning.join("; ")
    }
}

/// Calculate weight for build system importance
fn calculate_build_system_weight(name: &str) -> u32 {
    match name {
        "cargo" => 95,  // Rust
        "npm" | "yarn" | "pnpm" => 90,  // JavaScript/Node.js
        "maven" | "gradle" => 85,  // Java
        "poetry" | "pip" => 80,  // Python
        "go" => 95,  // Go modules
        "composer" => 75,  // PHP
        "gem" | "bundle" => 75,  // Ruby
        "mix" => 80,  // Elixir
        "stack" | "cabal" => 70,  // Haskell
        "dune" => 70,  // OCaml
        "leiningen" => 70,  // Clojure
        "cmake" | "make" => 60,  // C/C++
        "meson" => 65,  // Alternative C/C++
        _ => 50,
    }
}

/// Detect frameworks for a specific language
fn detect_frameworks_for_language(language: &str) -> HashMap<String, Vec<String>> {
    let mut frameworks = HashMap::new();

    match language {
        "javascript" | "typescript" => {
            frameworks.insert("react".to_string(), vec!["package.json".to_string()]);
            frameworks.insert("vue".to_string(), vec!["vue.config.js".to_string()]);
            frameworks.insert("angular".to_string(), vec!["angular.json".to_string()]);
            frameworks.insert("next".to_string(), vec!["next.config.js".to_string()]);
            frameworks.insert("nuxt".to_string(), vec!["nuxt.config.js".to_string()]);
        },
        "python" => {
            frameworks.insert("django".to_string(), vec!["manage.py".to_string(), "settings.py".to_string()]);
            frameworks.insert("flask".to_string(), vec!["app.py".to_string()]);
            frameworks.insert("fastapi".to_string(), vec!["main.py".to_string()]);
        },
        "rust" => {
            frameworks.insert("actix".to_string(), vec!["Cargo.toml".to_string()]);
            frameworks.insert("rocket".to_string(), vec!["Cargo.toml".to_string()]);
            frameworks.insert("warp".to_string(), vec!["Cargo.toml".to_string()]);
        },
        "go" => {
            frameworks.insert("gin".to_string(), vec!["go.mod".to_string()]);
            frameworks.insert("echo".to_string(), vec!["go.mod".to_string()]);
            frameworks.insert("fiber".to_string(), vec!["go.mod".to_string()]);
        },
        _ => {},
    }

    frameworks
}

/// Get framework detection patterns
fn get_framework_patterns() -> HashMap<String, Vec<String>> {
    let mut patterns = HashMap::new();

    // Web frameworks
    patterns.insert("react".to_string(), vec![
        "package.json".to_string(),
        "src/App.jsx".to_string(),
        "src/App.js".to_string(),
        "public/index.html".to_string(),
    ]);

    patterns.insert("vue".to_string(), vec![
        "vue.config.js".to_string(),
        "src/App.vue".to_string(),
        "src/main.js".to_string(),
    ]);

    patterns.insert("angular".to_string(), vec![
        "angular.json".to_string(),
        "src/app/app.module.ts".to_string(),
        "src/main.ts".to_string(),
    ]);

    patterns.insert("django".to_string(), vec![
        "manage.py".to_string(),
        "*/settings.py".to_string(),
        "*/urls.py".to_string(),
    ]);

    patterns.insert("flask".to_string(), vec![
        "app.py".to_string(),
        "application.py".to_string(),
        "run.py".to_string(),
    ]);

    patterns.insert("rails".to_string(), vec![
        "Gemfile".to_string(),
        "config/application.rb".to_string(),
        "app/controllers/application_controller.rb".to_string(),
    ]);

    patterns.insert("spring".to_string(), vec![
        "pom.xml".to_string(),
        "src/main/java/**/Application.java".to_string(),
        "application.properties".to_string(),
    ]);

    patterns
}

/// Check if a file matches a pattern (with glob-like support)
fn file_matches_pattern(file: &str, pattern: &str) -> bool {
    // Exact match
    if file == pattern {
        return true;
    }

    // Ends with pattern
    if pattern.starts_with('*') && file.ends_with(&pattern[1..]) {
        return true;
    }

    // Contains pattern
    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            return file.starts_with(parts[0]) && file.ends_with(parts[1]);
        }
    }

    // Directory-based matching
    if pattern.contains('/') && file.contains(pattern) {
        return true;
    }

    // Filename only matching
    if let Some(filename) = Path::new(file).file_name().and_then(|f| f.to_str()) {
        return filename == pattern;
    }

    false
}

/// Convenient function for quick project analysis
pub fn analyze_project_from_files(files: &[String]) -> ProjectInfo {
    match ProjectDetector::global() {
        Ok(detector) => detector.analyze_project(files),
        Err(e) => ProjectInfo {
            primary_language: None,
            languages: Vec::new(),
            build_systems: Vec::new(),
            frameworks: Vec::new(),
            project_type: ProjectType::Unknown,
            confidence: ProjectConfidence::Unknown,
            detection_details: DetectionDetails {
                files_analyzed: files.len(),
                pattern_matches: Vec::new(),
                methods_used: Vec::new(),
                reasoning: format!("Detector initialization failed: {}", e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_initialization() {
        let detector = ProjectDetector::new();
        assert!(detector.is_ok(), "Should initialize project detector");
    }

    #[test]
    fn test_rust_project_detection() {
        let detector = ProjectDetector::new().unwrap();
        let files = vec![
            "Cargo.toml".to_string(),
            "Cargo.lock".to_string(),
            "src/main.rs".to_string(),
            "src/lib.rs".to_string(),
            "tests/integration_test.rs".to_string(),
        ];

        let project_info = detector.analyze_project(&files);

        assert_eq!(project_info.primary_language, Some("rust".to_string()));
        assert!(project_info.languages.contains(&"rust".to_string()));
        assert!(!project_info.build_systems.is_empty());
        assert_eq!(project_info.build_systems[0].name, "cargo");
        assert!(project_info.confidence >= ProjectConfidence::High);
    }

    #[test]
    fn test_javascript_project_detection() {
        let detector = ProjectDetector::new().unwrap();
        let files = vec![
            "package.json".to_string(),
            "package-lock.json".to_string(),
            "src/index.js".to_string(),
            "src/App.jsx".to_string(),
            "public/index.html".to_string(),
        ];

        let project_info = detector.analyze_project(&files);

        assert!(project_info.primary_language == Some("javascript".to_string()) ||
                project_info.primary_language == Some("typescript".to_string()));
        assert!(!project_info.build_systems.is_empty());
        assert!(project_info.frameworks.contains(&"react".to_string()));
    }

    #[test]
    fn test_python_project_detection() {
        let detector = ProjectDetector::new().unwrap();
        let files = vec![
            "pyproject.toml".to_string(),
            "setup.py".to_string(),
            "src/main.py".to_string(),
            "tests/test_main.py".to_string(),
            "requirements.txt".to_string(),
        ];

        let project_info = detector.analyze_project(&files);

        assert_eq!(project_info.primary_language, Some("python".to_string()));
        assert!(project_info.languages.contains(&"python".to_string()));
        assert!(project_info.confidence >= ProjectConfidence::Medium);
    }

    #[test]
    fn test_monorepo_detection() {
        let detector = ProjectDetector::new().unwrap();
        let files = vec![
            "Cargo.toml".to_string(),
            "package.json".to_string(),
            "go.mod".to_string(),
            "rust-service/src/main.rs".to_string(),
            "js-frontend/src/index.js".to_string(),
            "go-api/main.go".to_string(),
        ];

        let project_info = detector.analyze_project(&files);

        assert_eq!(project_info.project_type, ProjectType::Monorepo);
        assert!(project_info.build_systems.len() > 1);
        assert!(project_info.languages.len() > 1);
    }

    #[test]
    fn test_documentation_project_detection() {
        let detector = ProjectDetector::new().unwrap();
        let files = vec![
            "README.md".to_string(),
            "docs/guide.md".to_string(),
            "docs/api.md".to_string(),
            "docs/tutorial.md".to_string(),
            "mkdocs.yml".to_string(),
        ];

        let project_info = detector.analyze_project(&files);

        assert_eq!(project_info.project_type, ProjectType::Documentation);
    }

    #[test]
    fn test_file_pattern_matching() {
        assert!(file_matches_pattern("package.json", "package.json"));
        assert!(file_matches_pattern("src/main.rs", "*.rs"));
        assert!(file_matches_pattern("test.txt", "test.txt"));
        assert!(file_matches_pattern("src/App.jsx", "*App.jsx"));
        assert!(!file_matches_pattern("main.py", "*.rs"));
    }

    #[test]
    fn test_global_detector() {
        let detector = ProjectDetector::global();
        assert!(detector.is_ok(), "Global detector should be available");
    }

    #[test]
    fn test_convenience_function() {
        let files = vec![
            "Cargo.toml".to_string(),
            "src/main.rs".to_string(),
        ];

        let project_info = analyze_project_from_files(&files);
        assert_eq!(project_info.primary_language, Some("rust".to_string()));
    }
}
//! Core `LanguageDetector` implementation: construction, global instance, and all
//! detection methods (extension, filename, shebang, keyword, path, content).

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::path::Path;

use super::types::{DetectionConfidence, DetectionMethod, DetectionResult, DetectorStats};
use crate::patterns::comprehensive::{ComprehensivePatternManager, ComprehensiveResult};

/// Optimized language detector with comprehensive pattern support
#[derive(Debug)]
pub struct LanguageDetector {
    /// Fast lookup map for extensions (optimized with ahash)
    pub(super) extension_map: HashMap<String, String>,
    /// Preprocessed shebang patterns for fast matching
    pub(super) shebang_patterns: Vec<(String, String)>,
    /// Keyword patterns for content-based detection
    pub(super) keyword_patterns: HashMap<String, Vec<String>>,
    /// Case-insensitive extension lookups
    pub(super) case_insensitive_extensions: HashMap<String, String>,
}

/// Global optimized language detector instance
static LANGUAGE_DETECTOR: Lazy<Result<LanguageDetector, String>> = Lazy::new(|| {
    LanguageDetector::new().map_err(|e| format!("Failed to initialize language detector: {}", e))
});

impl LanguageDetector {
    /// Create a new optimized language detector
    pub fn new() -> ComprehensiveResult<Self> {
        let comprehensive = ComprehensivePatternManager::new()?;
        let config = comprehensive.config();

        // Build optimized extension map
        let mut extension_map = HashMap::with_capacity(config.file_extensions.len());
        let mut case_insensitive_extensions = HashMap::new();

        for (ext, language) in &config.file_extensions {
            let clean_ext = ext.trim_start_matches('.');
            extension_map.insert(clean_ext.to_string(), language.clone());

            // Add case-insensitive variant for common patterns
            if clean_ext != clean_ext.to_lowercase() {
                case_insensitive_extensions.insert(clean_ext.to_lowercase(), language.clone());
            }
        }

        // Build preprocessed shebang patterns (sorted by length descending for better matching)
        let mut shebang_patterns: Vec<(String, String)> = config
            .content_signatures
            .shebangs
            .iter()
            .map(|(shebang, lang)| (shebang.clone(), lang.clone()))
            .collect();

        // Sort by shebang length (descending) for more specific matches first
        shebang_patterns.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // Clone keyword patterns
        let keyword_patterns = config.content_signatures.keyword_patterns.clone();

        tracing::debug!(
            "Language detector initialized: {} extensions, {} shebangs, {} keyword patterns",
            extension_map.len(),
            shebang_patterns.len(),
            keyword_patterns.len()
        );

        Ok(Self {
            extension_map,
            shebang_patterns,
            keyword_patterns,
            case_insensitive_extensions,
        })
    }

    /// Get the global language detector instance
    pub fn global() -> Result<&'static LanguageDetector, &'static str> {
        LANGUAGE_DETECTOR.as_ref().map_err(|e| e.as_str())
    }

    /// Detect language from file path with comprehensive analysis
    pub fn detect_from_path(&self, file_path: &Path) -> DetectionResult {
        let mut detection_methods = Vec::new();
        let mut best_confidence = DetectionConfidence::Unknown;
        let mut detected_language: Option<String> = None;

        // 1. Try extension-based detection (fastest and most reliable)
        if let Some(extension) = file_path.extension().and_then(|e| e.to_str()) {
            if let Some(result) = self.detect_from_extension(extension) {
                detection_methods.push(DetectionMethod::Extension(extension.to_string()));
                best_confidence = DetectionConfidence::VeryHigh;
                detected_language = Some(result);
            }
        }

        // 2. Try filename-based detection for special files (Dockerfile, Makefile, etc.)
        if detected_language.is_none() {
            if let Some(filename) = file_path.file_name().and_then(|f| f.to_str()) {
                if let Some(result) = self.detect_from_filename(filename) {
                    detection_methods.push(DetectionMethod::Filename(filename.to_string()));
                    best_confidence = DetectionConfidence::High;
                    detected_language = Some(result);
                }
            }
        }

        let detection_method = if detection_methods.len() > 1 {
            DetectionMethod::Consensus(detection_methods)
        } else if detection_methods.len() == 1 {
            detection_methods.into_iter().next().unwrap()
        } else {
            DetectionMethod::None
        };

        DetectionResult {
            language: detected_language,
            confidence: best_confidence,
            detection_method,
            details: format!("File path analysis: {:?}", file_path),
        }
    }

    /// Detect language from file content using multiple strategies
    pub fn detect_from_content(&self, content: &str, file_path: Option<&Path>) -> DetectionResult {
        let mut detection_methods = Vec::new();
        let mut confidence_scores = Vec::new();
        let mut candidate_languages = Vec::new();

        if let Some((shebang, language)) = self.detect_from_shebang(content) {
            detection_methods.push(DetectionMethod::Shebang(shebang));
            confidence_scores.push(DetectionConfidence::VeryHigh);
            candidate_languages.push(language);
        }

        if let Some((language, keywords)) = self.detect_from_keywords(content) {
            detection_methods.push(DetectionMethod::Keywords(keywords));
            confidence_scores.push(DetectionConfidence::Medium);
            candidate_languages.push(language);
        }

        if let Some(path) = file_path {
            let path_result = self.detect_from_path(path);
            if let Some(path_language) = path_result.language {
                if path_result.confidence >= DetectionConfidence::High {
                    detection_methods.push(path_result.detection_method);
                    confidence_scores.push(path_result.confidence);
                    candidate_languages.push(path_language);
                }
            }
        }

        let num_candidates = candidate_languages.len();
        let num_methods = detection_methods.len();

        let (final_language, final_confidence, final_method) =
            resolve_detection_candidates(candidate_languages, confidence_scores, detection_methods);

        DetectionResult {
            language: final_language,
            confidence: final_confidence,
            detection_method: final_method,
            details: format!(
                "Content analysis: {} candidates, {} methods",
                num_candidates, num_methods
            ),
        }
    }

    /// Detect language from file extension (optimized lookup)
    pub fn detect_from_extension(&self, extension: &str) -> Option<String> {
        let clean_ext = extension.trim_start_matches('.');

        // Try exact match first
        if let Some(language) = self.extension_map.get(clean_ext) {
            return Some(language.clone());
        }

        // Try case-insensitive match
        if let Some(language) = self
            .case_insensitive_extensions
            .get(&clean_ext.to_lowercase())
        {
            return Some(language.clone());
        }

        None
    }

    /// Detect language from filename patterns (for special files)
    pub fn detect_from_filename(&self, filename: &str) -> Option<String> {
        // Special filename patterns that don't have extensions
        match filename.to_lowercase().as_str() {
            "dockerfile" | "dockerfile.dev" | "dockerfile.prod" => Some("dockerfile".to_string()),
            "makefile" | "gnumakefile" => Some("make".to_string()),
            "rakefile" => Some("ruby".to_string()),
            "gemfile" | "gemfile.lock" => Some("ruby".to_string()),
            "justfile" => Some("just".to_string()),
            "pipfile" | "pipfile.lock" => Some("python".to_string()),
            "cargo.toml" | "cargo.lock" => Some("rust".to_string()),
            "go.mod" | "go.sum" | "go.work" => Some("go".to_string()),
            "package.json" | "package-lock.json" => Some("javascript".to_string()),
            "tsconfig.json" | "jsconfig.json" => Some("typescript".to_string()),
            "pom.xml" => Some("java".to_string()),
            "build.gradle" | "build.gradle.kts" => Some("groovy".to_string()),
            "composer.json" | "composer.lock" => Some("php".to_string()),
            "mix.exs" | "mix.lock" => Some("elixir".to_string()),
            "stack.yaml" => Some("haskell".to_string()),
            "dune-project" => Some("ocaml".to_string()),
            "project.clj" => Some("clojure".to_string()),
            "pubspec.yaml" | "pubspec.lock" => Some("dart".to_string()),
            "package.swift" => Some("swift".to_string()),
            "build.sbt" => Some("scala".to_string()),
            _ => None,
        }
    }

    /// Detect language from shebang line (optimized pattern matching)
    pub fn detect_from_shebang(&self, content: &str) -> Option<(String, String)> {
        // Check first few lines for shebang
        for line in content.lines().take(3) {
            let line = line.trim();
            if line.starts_with("#!") {
                // Try to match against precompiled patterns (sorted by specificity)
                for (shebang_pattern, language) in &self.shebang_patterns {
                    if line.contains(shebang_pattern) {
                        return Some((shebang_pattern.clone(), language.clone()));
                    }
                }
            }
        }
        None
    }

    /// Detect language from content keywords (weighted scoring)
    pub fn detect_from_keywords(&self, content: &str) -> Option<(String, Vec<String>)> {
        let mut language_scores: HashMap<String, (usize, Vec<String>)> = HashMap::new();

        // Score languages based on keyword frequency
        for (language, keywords) in &self.keyword_patterns {
            let mut score = 0;
            let mut found_keywords = Vec::new();

            for keyword in keywords {
                // Count occurrences of each keyword
                let keyword_count = content.matches(keyword).count();
                if keyword_count > 0 {
                    score += keyword_count;
                    found_keywords.push(keyword.clone());
                }
            }

            if score > 0 {
                language_scores.insert(language.clone(), (score, found_keywords));
            }
        }

        // Return language with highest score
        language_scores
            .into_iter()
            .max_by_key(|(_, (score, _))| *score)
            .map(|(language, (_, keywords))| (language, keywords))
    }

    /// Get all supported extensions
    pub fn supported_extensions(&self) -> Vec<&String> {
        self.extension_map.keys().collect()
    }

    /// Get all supported languages
    pub fn supported_languages(&self) -> Vec<&String> {
        self.extension_map
            .values()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// Check if an extension is supported
    pub fn is_extension_supported(&self, extension: &str) -> bool {
        let clean_ext = extension.trim_start_matches('.');
        self.extension_map.contains_key(clean_ext)
            || self
                .case_insensitive_extensions
                .contains_key(&clean_ext.to_lowercase())
    }

    /// Get statistics about the detector
    pub fn stats(&self) -> DetectorStats {
        DetectorStats {
            total_extensions: self.extension_map.len(),
            case_insensitive_extensions: self.case_insensitive_extensions.len(),
            shebang_patterns: self.shebang_patterns.len(),
            keyword_patterns: self.keyword_patterns.len(),
            unique_languages: self
                .extension_map
                .values()
                .collect::<std::collections::HashSet<_>>()
                .len(),
        }
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new().expect("Failed to initialize LanguageDetector")
    }
}

/// Resolve a list of candidate detections to a single (language, confidence, method) result.
fn resolve_detection_candidates(
    candidate_languages: Vec<String>,
    confidence_scores: Vec<DetectionConfidence>,
    detection_methods: Vec<DetectionMethod>,
) -> (Option<String>, DetectionConfidence, DetectionMethod) {
    if candidate_languages.is_empty() {
        return (None, DetectionConfidence::Unknown, DetectionMethod::None);
    }
    if candidate_languages.len() == 1 {
        return (
            Some(candidate_languages[0].clone()),
            confidence_scores[0].clone(),
            detection_methods
                .into_iter()
                .next()
                .unwrap_or(DetectionMethod::None),
        );
    }

    // Multiple candidates — vote for consensus
    let mut language_votes: HashMap<String, (usize, DetectionConfidence)> = HashMap::new();
    for (i, lang) in candidate_languages.iter().enumerate() {
        let entry = language_votes
            .entry(lang.clone())
            .or_insert((0, DetectionConfidence::Unknown));
        entry.0 += 1;
        if confidence_scores[i] > entry.1 {
            entry.1 = confidence_scores[i].clone();
        }
    }

    let (best_language, (votes, best_confidence)) = language_votes
        .into_iter()
        .max_by(|(_, (v1, c1)), (_, (v2, c2))| v1.cmp(v2).then(c1.cmp(c2)))
        .unwrap();

    let final_confidence = if votes > 1 {
        match best_confidence {
            DetectionConfidence::VeryHigh => DetectionConfidence::VeryHigh,
            DetectionConfidence::High => DetectionConfidence::VeryHigh,
            DetectionConfidence::Medium => DetectionConfidence::High,
            DetectionConfidence::Low => DetectionConfidence::Medium,
            DetectionConfidence::Unknown => DetectionConfidence::Low,
        }
    } else {
        best_confidence
    };

    (
        Some(best_language),
        final_confidence,
        DetectionMethod::Consensus(detection_methods),
    )
}

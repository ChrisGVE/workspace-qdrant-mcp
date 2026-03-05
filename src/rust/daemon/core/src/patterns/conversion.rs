//! Conversion from comprehensive configuration to legacy AllPatterns structure.
//!
//! Converts the YAML-based ComprehensivePatternManager config into the
//! strongly-typed AllPatterns struct used throughout the codebase.

use std::collections::HashMap;

use super::comprehensive::ComprehensivePatternManager;
use super::{
    AllPatterns, ConfidenceLevel, Ecosystem, ExcludePatterns, IncludePatterns, LanguageExtensions,
    LanguageGroup, PatternError, PatternResult, PatternWithMetadata, ProjectIndicator,
    ProjectIndicators,
};

/// Convert comprehensive configuration to the AllPatterns structure.
pub(super) fn convert_comprehensive_to_patterns(
    comprehensive: &ComprehensivePatternManager,
) -> PatternResult<AllPatterns> {
    let config = comprehensive.config();

    let project_indicators = build_project_indicators(config);
    let exclude_patterns = build_exclude_patterns(config);
    let include_patterns = build_include_patterns(config);
    let language_extensions = build_language_extensions(config);

    Ok(AllPatterns {
        project_indicators,
        exclude_patterns,
        include_patterns,
        language_extensions,
    })
}

/// Validate that all patterns are well-formed and consistent.
pub(super) fn validate_patterns(patterns: &AllPatterns) -> PatternResult<()> {
    validate_ecosystems(patterns)?;
    validate_exclude_patterns(patterns)?;
    validate_include_patterns(patterns)?;
    validate_language_groups(patterns)?;

    tracing::debug!(
        "Pattern validation complete: {} ecosystems, {} exclude patterns, {} include patterns, {} languages",
        patterns.project_indicators.ecosystems.len(),
        patterns.all_exclude_patterns().len(),
        patterns.all_include_patterns().len(),
        patterns.supported_extensions().len()
    );

    Ok(())
}

// --- Project indicators ---

fn build_project_indicators(
    config: &super::comprehensive::InternalConfiguration,
) -> ProjectIndicators {
    let mut ecosystems = HashMap::new();

    // Rust ecosystem from cargo build system
    if let Some(rust_build) = config.build_systems.get("cargo") {
        let indicators = rust_build
            .files
            .iter()
            .map(|file| ProjectIndicator {
                pattern: file.clone(),
                confidence: ConfidenceLevel::High,
                rationale: "Cargo build system indicator".to_string(),
                weight: 90,
            })
            .collect();

        ecosystems.insert(
            "rust".to_string(),
            Ecosystem {
                name: "Rust".to_string(),
                description: "Rust programming ecosystem".to_string(),
                confidence_levels: HashMap::new(),
                indicators,
            },
        );
    }

    // Other language ecosystems from build systems
    for (name, build_config) in &config.build_systems {
        if !ecosystems.contains_key(&build_config.language) {
            let indicators = build_config
                .files
                .iter()
                .map(|file| ProjectIndicator {
                    pattern: file.clone(),
                    confidence: ConfidenceLevel::High,
                    rationale: format!("{} build system indicator", name),
                    weight: 80,
                })
                .collect();

            ecosystems.insert(
                build_config.language.clone(),
                Ecosystem {
                    name: build_config.language.clone(),
                    description: format!("{} programming ecosystem", build_config.language),
                    confidence_levels: HashMap::new(),
                    indicators,
                },
            );
        }
    }

    ProjectIndicators {
        version: "1.0.0".to_string(),
        last_updated: wqm_common::timestamps::now_utc(),
        research_coverage: "500+ languages from comprehensive A-Z research".to_string(),
        ecosystems,
    }
}

// --- Exclude patterns ---

fn build_exclude_patterns(config: &super::comprehensive::InternalConfiguration) -> ExcludePatterns {
    let exclusions = &config.exclusion_patterns;

    let all_exclude_patterns: Vec<PatternWithMetadata> = [
        &exclusions.version_control,
        &exclusions.build_outputs,
        &exclusions.cache_directories,
        &exclusions.virtual_environments,
        &exclusions.ide_files,
        &exclusions.temporary_files,
        &exclusions.binary_files,
        &exclusions.media_files,
        &exclusions.archive_files,
        &exclusions.package_files,
    ]
    .iter()
    .flat_map(|patterns| {
        patterns.iter().map(|pattern| PatternWithMetadata {
            pattern: normalize_pattern(pattern),
            description: format!("Auto-generated from comprehensive config: {}", pattern),
            ecosystems: vec!["all".to_string()],
        })
    })
    .collect();

    ExcludePatterns {
        build_artifacts: all_exclude_patterns,
        compiled_code: Vec::new(),
        environments: Vec::new(),
        caches: Vec::new(),
        version_control: Vec::new(),
        editor_files: Vec::new(),
        system_files: Vec::new(),
        logs_and_temp: Vec::new(),
        media_files: Vec::new(),
        large_binaries: Vec::new(),
        security: Vec::new(),
        test_artifacts: Vec::new(),
    }
}

// --- Include patterns ---

fn build_include_patterns(config: &super::comprehensive::InternalConfiguration) -> IncludePatterns {
    let source_code_patterns: Vec<PatternWithMetadata> = config
        .file_extensions
        .keys()
        .map(|ext| PatternWithMetadata {
            pattern: format!("*.{}", ext.trim_start_matches('.')),
            description: format!("Source code pattern for {}", ext),
            ecosystems: vec!["all".to_string()],
        })
        .collect();

    let config_patterns = build_config_include_patterns(config);

    IncludePatterns {
        version: "1.0.0".to_string(),
        last_updated: wqm_common::timestamps::now_utc(),
        research_coverage: "500+ languages from comprehensive A-Z research".to_string(),
        source_code: source_code_patterns,
        documentation: Vec::new(),
        configuration: config_patterns,
        schema_and_data: Vec::new(),
        templates_and_resources: Vec::new(),
        project_management: Vec::new(),
        special_patterns: Vec::new(),
    }
}

fn build_config_include_patterns(
    config: &super::comprehensive::InternalConfiguration,
) -> Vec<PatternWithMetadata> {
    let mut config_patterns = Vec::new();

    for patterns in [
        &config.project_indicators.version_control,
        &config.project_indicators.language_ecosystems,
        &config.project_indicators.build_systems,
        &config.project_indicators.ci_cd,
        &config.project_indicators.containerization,
        &config.project_indicators.config_management,
    ] {
        for pattern in patterns {
            config_patterns.push(PatternWithMetadata {
                pattern: normalize_pattern(pattern),
                description: format!("Config indicator: {}", pattern),
                ecosystems: vec!["all".to_string()],
            });
        }
    }

    for build in config.build_systems.values() {
        for pattern in &build.files {
            config_patterns.push(PatternWithMetadata {
                pattern: normalize_pattern(pattern),
                description: format!("Build system file: {}", pattern),
                ecosystems: vec!["all".to_string()],
            });
        }
    }

    config_patterns
}

// --- Language extensions ---

fn build_language_extensions(
    config: &super::comprehensive::InternalConfiguration,
) -> LanguageExtensions {
    let mut programming_languages = HashMap::new();
    let mut web_technologies = HashMap::new();
    let mut markup_languages = HashMap::new();
    let mut configuration_files = HashMap::new();
    let mut shell_scripting = HashMap::new();
    let mut data_formats = HashMap::new();
    let mut specialized_formats = HashMap::new();
    let mut extensions_to_languages = HashMap::new();

    for (ext, language) in &config.file_extensions {
        extensions_to_languages.insert(ext.clone(), language.clone());

        let lsp_id = config
            .lsp_servers
            .get(language)
            .map(|lsp| lsp.primary.clone())
            .unwrap_or_else(|| format!("{}-lsp", language));

        let category = categorize_language(language);
        let lang_group = LanguageGroup {
            extensions: vec![ext.clone()],
            filenames: Vec::new(),
            lsp_id,
            category: category.clone(),
        };

        let target = match category.as_str() {
            "programming" => &mut programming_languages,
            "web" => &mut web_technologies,
            "markup" => &mut markup_languages,
            "config" => &mut configuration_files,
            "shell" => &mut shell_scripting,
            "data" => &mut data_formats,
            _ => &mut specialized_formats,
        };
        target.insert(language.clone(), lang_group);
    }

    LanguageExtensions {
        programming_languages,
        web_technologies,
        markup_languages,
        configuration_files,
        shell_scripting,
        data_formats,
        specialized_formats,
        extensions_to_languages,
        filenames_to_languages: HashMap::new(),
        metadata: HashMap::new(),
    }
}

// --- Helpers ---

fn normalize_pattern(pattern: &str) -> String {
    if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
        pattern.to_string()
    } else {
        format!("*{}*", pattern)
    }
}

/// Categorize a language into a general category.
fn categorize_language(language: &str) -> String {
    match language {
        "rust" | "go" | "python" | "java" | "cpp" | "c" | "csharp" | "swift" | "kotlin"
        | "scala" | "clojure" | "haskell" | "ocaml" | "julia" | "dart" | "crystal" | "nim"
        | "zig" | "fortran" | "cobol" | "ada" | "d" | "elixir" | "erlang" | "fsharp" | "groovy"
        | "lua" | "perl" | "php" | "r" | "ruby" => "programming".to_string(),
        "javascript" | "typescript" | "css" | "scss" | "sass" | "less" | "html" | "jsx" | "tsx" => {
            "web".to_string()
        }
        "markdown" | "asciidoc" | "restructuredtext" | "latex" | "xml" => "markup".to_string(),
        "json" | "yaml" | "toml" | "ini" | "cfg" | "conf" => "config".to_string(),
        "bash" | "zsh" | "fish" | "powershell" | "bat" | "cmd" => "shell".to_string(),
        "csv" | "tsv" | "sql" | "graphql" => "data".to_string(),
        _ => "specialized".to_string(),
    }
}

// --- Validation helpers ---

fn validate_ecosystems(patterns: &AllPatterns) -> PatternResult<()> {
    for (ecosystem_name, ecosystem) in &patterns.project_indicators.ecosystems {
        if ecosystem.name.is_empty() {
            return Err(PatternError::Validation(format!(
                "Ecosystem '{}' has empty name",
                ecosystem_name
            )));
        }

        for indicator in &ecosystem.indicators {
            if indicator.pattern.is_empty() {
                return Err(PatternError::Validation(format!(
                    "Ecosystem '{}' has empty pattern",
                    ecosystem_name
                )));
            }
            if indicator.weight == 0 || indicator.weight > 100 {
                return Err(PatternError::Validation(format!(
                    "Ecosystem '{}' indicator '{}' has invalid weight: {}",
                    ecosystem_name, indicator.pattern, indicator.weight
                )));
            }
        }
    }
    Ok(())
}

fn validate_exclude_patterns(patterns: &AllPatterns) -> PatternResult<()> {
    for pattern_meta in &patterns.all_exclude_patterns() {
        if pattern_meta.pattern.is_empty() {
            return Err(PatternError::Validation(
                "Found empty exclude pattern".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_include_patterns(patterns: &AllPatterns) -> PatternResult<()> {
    for pattern_meta in &patterns.all_include_patterns() {
        if pattern_meta.pattern.is_empty() {
            return Err(PatternError::Validation(
                "Found empty include pattern".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_language_groups(patterns: &AllPatterns) -> PatternResult<()> {
    let all_lang_groups = [
        &patterns.language_extensions.programming_languages,
        &patterns.language_extensions.web_technologies,
        &patterns.language_extensions.markup_languages,
        &patterns.language_extensions.configuration_files,
        &patterns.language_extensions.shell_scripting,
        &patterns.language_extensions.data_formats,
        &patterns.language_extensions.specialized_formats,
    ];

    for category in all_lang_groups {
        for (lang_name, lang_group) in category {
            if lang_group.lsp_id.is_empty() {
                return Err(PatternError::Validation(format!(
                    "Language '{}' has empty lsp_id",
                    lang_name
                )));
            }
            if lang_group.extensions.is_empty() && lang_group.filenames.is_empty() {
                return Err(PatternError::Validation(format!(
                    "Language '{}' has no extensions or filenames",
                    lang_name
                )));
            }
        }
    }
    Ok(())
}

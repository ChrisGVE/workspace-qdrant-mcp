//! Registry YAML validation.
//!
//! Validates generated `language_registry.yaml` output for schema correctness,
//! completeness, and consistency. Used both in unit tests and as a CLI
//! post-generation step.

use workspace_qdrant_core::language_registry::types::LanguageDefinition;

/// Validation result for a single language definition.
#[derive(Debug)]
pub struct ValidationIssue {
    pub language: String,
    pub severity: Severity,
    pub message: String,
}

/// Issue severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// Validate a set of language definitions for schema correctness.
pub fn validate_definitions(definitions: &[LanguageDefinition]) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    // Check for duplicate language IDs
    let mut seen_ids = std::collections::HashSet::new();
    for def in definitions {
        let id = def.id();
        if !seen_ids.insert(id.clone()) {
            issues.push(ValidationIssue {
                language: id,
                severity: Severity::Error,
                message: "Duplicate language ID".to_string(),
            });
        }
    }

    // Per-language validation
    for def in definitions {
        let id = def.id();

        // Language name must not be empty
        if def.language.trim().is_empty() {
            issues.push(ValidationIssue {
                language: id.clone(),
                severity: Severity::Error,
                message: "Empty language name".to_string(),
            });
        }

        // Grammar sources: repos must look like "owner/repo"
        for source in &def.grammar.sources {
            if !source.repo.contains('/') {
                issues.push(ValidationIssue {
                    language: id.clone(),
                    severity: Severity::Error,
                    message: format!("Invalid grammar repo format: '{}'", source.repo),
                });
            }
        }

        // LSP servers: binary must not be empty
        for server in &def.lsp_servers {
            if server.binary.trim().is_empty() {
                issues.push(ValidationIssue {
                    language: id.clone(),
                    severity: Severity::Error,
                    message: format!("LSP server '{}' has empty binary name", server.name),
                });
            }
        }

        // Warning: language has no extensions
        if def.extensions.is_empty() {
            issues.push(ValidationIssue {
                language: id.clone(),
                severity: Severity::Warning,
                message: "No file extensions defined".to_string(),
            });
        }

        // Warning: programming language with no grammar
        if def.language_type
            == workspace_qdrant_core::language_registry::types::LanguageType::Programming
            && !def.has_grammar()
        {
            issues.push(ValidationIssue {
                language: id.clone(),
                severity: Severity::Warning,
                message: "Programming language with no grammar sources".to_string(),
            });
        }
    }

    issues
}

/// Validate that a YAML string can be parsed as a valid registry.
pub fn validate_yaml(yaml: &str) -> Result<Vec<LanguageDefinition>, String> {
    serde_yaml_ng::from_str::<Vec<LanguageDefinition>>(yaml)
        .map_err(|e| format!("YAML parse error: {e}"))
}

/// Print a validation report.
pub fn print_report(issues: &[ValidationIssue]) {
    let errors: Vec<_> = issues
        .iter()
        .filter(|i| i.severity == Severity::Error)
        .collect();
    let warnings: Vec<_> = issues
        .iter()
        .filter(|i| i.severity == Severity::Warning)
        .collect();

    if errors.is_empty() && warnings.is_empty() {
        println!("Validation: PASS (no issues)");
        return;
    }

    if !errors.is_empty() {
        println!("ERRORS ({}):", errors.len());
        for issue in &errors {
            println!("  [{}] {}", issue.language, issue.message);
        }
    }

    if !warnings.is_empty() {
        println!("WARNINGS ({}):", warnings.len());
        for issue in &warnings {
            println!("  [{}] {}", issue.language, issue.message);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use workspace_qdrant_core::language_registry::types::{
        GrammarConfig, GrammarQuality, GrammarSourceEntry, LanguageType, LspServerEntry,
        SourceMetadata,
    };

    fn valid_definition() -> LanguageDefinition {
        LanguageDefinition {
            language: "Rust".to_string(),
            aliases: vec!["rs".to_string()],
            extensions: vec![".rs".to_string()],
            language_type: LanguageType::Programming,
            grammar: GrammarConfig {
                sources: vec![GrammarSourceEntry {
                    repo: "tree-sitter/tree-sitter-rust".to_string(),
                    origin: None,
                    quality: GrammarQuality::Curated,
                }],
                ..Default::default()
            },
            semantic_patterns: None,
            lsp_servers: vec![LspServerEntry {
                name: "rust-analyzer".to_string(),
                binary: "rust-analyzer".to_string(),
                args: Vec::new(),
                priority: 1,
                install_methods: Vec::new(),
            }],
            sources: SourceMetadata::default(),
        }
    }

    #[test]
    fn test_valid_definition_passes() {
        let defs = vec![valid_definition()];
        let issues = validate_definitions(&defs);
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty(), "Expected no errors: {:?}", errors);
    }

    #[test]
    fn test_duplicate_id_detected() {
        let defs = vec![valid_definition(), valid_definition()];
        let issues = validate_definitions(&defs);
        assert!(issues.iter().any(|i| i.message.contains("Duplicate")));
    }

    #[test]
    fn test_invalid_grammar_repo_detected() {
        let mut def = valid_definition();
        def.grammar.sources[0].repo = "no-slash".to_string();
        let issues = validate_definitions(&[def]);
        assert!(issues
            .iter()
            .any(|i| i.message.contains("Invalid grammar repo")));
    }

    #[test]
    fn test_empty_binary_detected() {
        let mut def = valid_definition();
        def.lsp_servers[0].binary = "".to_string();
        let issues = validate_definitions(&[def]);
        assert!(issues.iter().any(|i| i.message.contains("empty binary")));
    }

    #[test]
    fn test_no_extensions_warning() {
        let mut def = valid_definition();
        def.extensions.clear();
        let issues = validate_definitions(&[def]);
        assert!(issues.iter().any(|i| {
            i.severity == Severity::Warning && i.message.contains("No file extensions")
        }));
    }

    #[test]
    fn test_programming_no_grammar_warning() {
        let mut def = valid_definition();
        def.grammar.sources.clear();
        let issues = validate_definitions(&[def]);
        assert!(issues.iter().any(|i| {
            i.severity == Severity::Warning && i.message.contains("no grammar sources")
        }));
    }

    #[test]
    fn test_validate_yaml_roundtrip() {
        let defs = vec![valid_definition()];
        let yaml = serde_yaml_ng::to_string(&defs).unwrap();
        let parsed = validate_yaml(&yaml).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].language, "Rust");
    }

    #[test]
    fn test_validate_yaml_invalid() {
        let result = validate_yaml("not: valid: yaml: [[[");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_bundled_registry() {
        // Validate the actual bundled language_registry.yaml
        let provider =
            workspace_qdrant_core::language_registry::providers::registry::RegistryProvider::new()
                .unwrap();
        let defs: Vec<LanguageDefinition> = provider.definitions().to_vec();

        let issues = validate_definitions(&defs);
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .collect();

        assert!(
            errors.is_empty(),
            "Bundled registry has validation errors: {:?}",
            errors
        );

        // Should have at least 40 languages
        assert!(
            defs.len() >= 40,
            "Expected 40+ bundled languages, got {}",
            defs.len()
        );
    }
}

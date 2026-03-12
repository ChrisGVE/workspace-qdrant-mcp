//! Multi-provider merge logic.
//!
//! Merges scraped data from all upstream sources into unified
//! `LanguageDefinition` entries following the priority-based merge
//! strategy from `docs/specs/15-language-registry.md`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use workspace_qdrant_core::language_registry::types::{
    GrammarConfig, GrammarSourceEntry, LanguageDefinition, LanguageType, SourceMetadata,
};

use crate::scraper::ScrapedData;

/// Diff statistics between current and generated registries.
#[derive(Debug, Default)]
pub struct RegistryDiff {
    pub added: usize,
    pub removed: usize,
    pub modified: usize,
}

/// Merge all scraped data into unified language definitions.
///
/// Merge strategy:
/// 1. Start with language identity from Linguist (backbone).
/// 2. Add grammar sources, deduplicated by repo. Higher-tier sources
///    (ts-grammars-org → Curated) take precedence in sort order.
/// 3. Add LSP servers, deduplicated by binary name.
/// 4. Apply semantic patterns from registry provider (if available).
pub fn merge_all(scraped: &ScrapedData) -> Result<Vec<LanguageDefinition>> {
    let mut map: HashMap<String, LanguageDefinition> = HashMap::new();

    // Step 1: Language identity from Linguist
    for lang in &scraped.languages {
        let id = lang.id.to_lowercase();
        map.insert(
            id.clone(),
            LanguageDefinition {
                language: lang.name.clone(),
                aliases: lang.aliases.clone(),
                extensions: lang.extensions.clone(),
                language_type: lang.language_type,
                grammar: GrammarConfig::default(),
                semantic_patterns: None,
                lsp_servers: Vec::new(),
                sources: SourceMetadata {
                    language: Some("linguist".to_string()),
                    ..Default::default()
                },
            },
        );
    }

    // Step 2: Add grammar sources
    for grammar in &scraped.grammars {
        let id = grammar.language.to_lowercase();
        let def = map.entry(id.clone()).or_insert_with(|| LanguageDefinition {
            language: grammar.language.clone(),
            aliases: Vec::new(),
            extensions: Vec::new(),
            language_type: LanguageType::Programming,
            grammar: GrammarConfig::default(),
            semantic_patterns: None,
            lsp_servers: Vec::new(),
            sources: SourceMetadata::default(),
        });

        // Deduplicate by repo
        let already_has = def.grammar.sources.iter().any(|s| s.repo == grammar.repo);
        if !already_has {
            def.grammar.sources.push(GrammarSourceEntry {
                repo: grammar.repo.clone(),
                origin: None,
                quality: grammar.quality,
            });
        }

        // Set build metadata (last writer wins, higher-quality sources preferred)
        if grammar.has_cpp_scanner {
            def.grammar.has_cpp_scanner = true;
        }
        if grammar.src_subdir.is_some() {
            def.grammar.src_subdir.clone_from(&grammar.src_subdir);
        }
        if grammar.symbol_name.is_some() {
            def.grammar.symbol_name.clone_from(&grammar.symbol_name);
        }
        if grammar.archive_branch.is_some() {
            def.grammar
                .archive_branch
                .clone_from(&grammar.archive_branch);
        }

        def.sources.grammar = Some("scraped".to_string());
    }

    // Step 3: Add LSP servers
    for lsp in &scraped.lsp_servers {
        let id = lsp.language.to_lowercase();
        let def = map.entry(id.clone()).or_insert_with(|| LanguageDefinition {
            language: lsp.language.clone(),
            aliases: Vec::new(),
            extensions: Vec::new(),
            language_type: LanguageType::Programming,
            grammar: GrammarConfig::default(),
            semantic_patterns: None,
            lsp_servers: Vec::new(),
            sources: SourceMetadata::default(),
        });

        // Deduplicate by binary name
        let already_has = def
            .lsp_servers
            .iter()
            .any(|s| s.binary == lsp.server.binary);
        if !already_has {
            def.lsp_servers.push(lsp.server.clone());
        }

        def.sources.lsp = Some("scraped".to_string());
    }

    // Step 4: Sort grammar sources by quality (Curated < Official < Community)
    // and LSP servers by priority (lower = preferred)
    for def in map.values_mut() {
        def.grammar.sources.sort_by_key(|s| s.quality);
        def.lsp_servers.sort_by_key(|s| s.priority);
    }

    // Collect and sort by language name
    let mut result: Vec<LanguageDefinition> = map.into_values().collect();
    result.sort_by(|a, b| a.language.to_lowercase().cmp(&b.language.to_lowercase()));

    Ok(result)
}

/// Compare generated definitions against the current registry YAML file.
pub fn diff_with_current(
    current_path: &Path,
    generated: &[LanguageDefinition],
) -> Result<RegistryDiff> {
    let content = std::fs::read_to_string(current_path)
        .with_context(|| format!("Reading current registry: {}", current_path.display()))?;

    let current: Vec<LanguageDefinition> =
        serde_yaml_ng::from_str(&content).with_context(|| "Parsing current registry YAML")?;

    let current_ids: std::collections::HashSet<String> = current.iter().map(|d| d.id()).collect();
    let generated_ids: std::collections::HashSet<String> =
        generated.iter().map(|d| d.id()).collect();

    let added = generated_ids.difference(&current_ids).count();
    let removed = current_ids.difference(&generated_ids).count();

    // Count modified: same ID but different grammar count or LSP count
    let current_map: HashMap<String, &LanguageDefinition> =
        current.iter().map(|d| (d.id(), d)).collect();
    let mut modified = 0;
    for def in generated {
        if let Some(cur) = current_map.get(&def.id()) {
            if def.grammar.sources.len() != cur.grammar.sources.len()
                || def.lsp_servers.len() != cur.lsp_servers.len()
                || def.extensions.len() != cur.extensions.len()
            {
                modified += 1;
            }
        }
    }

    Ok(RegistryDiff {
        added,
        removed,
        modified,
    })
}

/// Serialize language definitions to YAML.
pub fn serialize_definitions(definitions: &[LanguageDefinition]) -> Result<String> {
    let yaml = serde_yaml_ng::to_string(definitions).context("Serializing definitions to YAML")?;
    Ok(yaml)
}

/// Merge generated definitions with the existing bundled registry,
/// preserving hand-crafted semantic_patterns from the bundled YAML.
pub fn merge_with_bundled(
    bundled_path: &Path,
    generated: &mut Vec<LanguageDefinition>,
) -> Result<()> {
    let content = std::fs::read_to_string(bundled_path)
        .with_context(|| format!("Reading bundled registry: {}", bundled_path.display()))?;

    let bundled: Vec<LanguageDefinition> =
        serde_yaml_ng::from_str(&content).with_context(|| "Parsing bundled registry YAML")?;

    let bundled_map: HashMap<String, LanguageDefinition> =
        bundled.into_iter().map(|d| (d.id(), d)).collect();

    // Preserve semantic_patterns and LSP servers from bundled if not scraped
    for def in generated.iter_mut() {
        if let Some(bundled_def) = bundled_map.get(&def.id()) {
            // Keep hand-crafted semantic patterns
            if def.semantic_patterns.is_none() && bundled_def.semantic_patterns.is_some() {
                def.semantic_patterns
                    .clone_from(&bundled_def.semantic_patterns);
            }

            // Keep hand-crafted LSP server entries (with args, priority, install methods)
            // if the scraped version only has placeholder install methods
            for bundled_lsp in &bundled_def.lsp_servers {
                if let Some(gen_lsp) = def
                    .lsp_servers
                    .iter_mut()
                    .find(|s| s.binary == bundled_lsp.binary)
                {
                    // Prefer bundled LSP config (has hand-tuned args, priority, install)
                    if !bundled_lsp.args.is_empty() && gen_lsp.args.is_empty() {
                        gen_lsp.args.clone_from(&bundled_lsp.args);
                    }
                    if bundled_lsp.priority < gen_lsp.priority {
                        gen_lsp.priority = bundled_lsp.priority;
                    }
                    if !bundled_lsp.install_methods.is_empty()
                        && gen_lsp.install_methods.iter().all(|m| m.manager == "see")
                    {
                        gen_lsp
                            .install_methods
                            .clone_from(&bundled_lsp.install_methods);
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use workspace_qdrant_core::language_registry::types::{
        GrammarEntry, GrammarQuality, LanguageEntry, LspEntry, LspServerEntry,
    };

    fn sample_scraped() -> ScrapedData {
        ScrapedData {
            languages: vec![
                LanguageEntry {
                    name: "Rust".to_string(),
                    id: "rust".to_string(),
                    aliases: vec!["rs".to_string()],
                    extensions: vec![".rs".to_string()],
                    language_type: LanguageType::Programming,
                },
                LanguageEntry {
                    name: "Python".to_string(),
                    id: "python".to_string(),
                    aliases: vec!["py".to_string()],
                    extensions: vec![".py".to_string()],
                    language_type: LanguageType::Programming,
                },
            ],
            grammars: vec![
                GrammarEntry {
                    language: "rust".to_string(),
                    repo: "tree-sitter/tree-sitter-rust".to_string(),
                    quality: GrammarQuality::Curated,
                    has_cpp_scanner: false,
                    src_subdir: None,
                    symbol_name: None,
                    archive_branch: None,
                },
                GrammarEntry {
                    language: "rust".to_string(),
                    repo: "some-other/tree-sitter-rust".to_string(),
                    quality: GrammarQuality::Community,
                    has_cpp_scanner: false,
                    src_subdir: None,
                    symbol_name: None,
                    archive_branch: None,
                },
                GrammarEntry {
                    language: "python".to_string(),
                    repo: "tree-sitter/tree-sitter-python".to_string(),
                    quality: GrammarQuality::Official,
                    has_cpp_scanner: false,
                    src_subdir: None,
                    symbol_name: None,
                    archive_branch: None,
                },
            ],
            lsp_servers: vec![LspEntry {
                language: "rust".to_string(),
                server: LspServerEntry {
                    name: "rust-analyzer".to_string(),
                    binary: "rust-analyzer".to_string(),
                    args: Vec::new(),
                    priority: 1,
                    install_methods: Vec::new(),
                },
            }],
            warnings: Vec::new(),
        }
    }

    #[test]
    fn test_merge_all_basic() {
        let scraped = sample_scraped();
        let result = merge_all(&scraped).unwrap();

        assert_eq!(result.len(), 2);

        let rust = result.iter().find(|d| d.id() == "rust").unwrap();
        assert_eq!(rust.language, "Rust");
        assert_eq!(rust.grammar.sources.len(), 2);
        // Curated should sort before Community
        assert_eq!(rust.grammar.sources[0].quality, GrammarQuality::Curated);
        assert_eq!(rust.grammar.sources[1].quality, GrammarQuality::Community);
        assert_eq!(rust.lsp_servers.len(), 1);
        assert_eq!(rust.lsp_servers[0].name, "rust-analyzer");
    }

    #[test]
    fn test_merge_deduplicates_grammars() {
        let scraped = ScrapedData {
            languages: Vec::new(),
            grammars: vec![
                GrammarEntry {
                    language: "go".to_string(),
                    repo: "tree-sitter/tree-sitter-go".to_string(),
                    quality: GrammarQuality::Curated,
                    has_cpp_scanner: false,
                    src_subdir: None,
                    symbol_name: None,
                    archive_branch: None,
                },
                GrammarEntry {
                    language: "go".to_string(),
                    repo: "tree-sitter/tree-sitter-go".to_string(),
                    quality: GrammarQuality::Community,
                    has_cpp_scanner: false,
                    src_subdir: None,
                    symbol_name: None,
                    archive_branch: None,
                },
            ],
            lsp_servers: Vec::new(),
            warnings: Vec::new(),
        };

        let result = merge_all(&scraped).unwrap();
        let go = result.iter().find(|d| d.id() == "go").unwrap();
        // Same repo should be deduplicated
        assert_eq!(go.grammar.sources.len(), 1);
    }

    #[test]
    fn test_merge_deduplicates_lsp() {
        let scraped = ScrapedData {
            languages: Vec::new(),
            grammars: Vec::new(),
            lsp_servers: vec![
                LspEntry {
                    language: "python".to_string(),
                    server: LspServerEntry {
                        name: "pylsp".to_string(),
                        binary: "pylsp".to_string(),
                        args: Vec::new(),
                        priority: 1,
                        install_methods: Vec::new(),
                    },
                },
                LspEntry {
                    language: "python".to_string(),
                    server: LspServerEntry {
                        name: "pylsp-alt".to_string(),
                        binary: "pylsp".to_string(), // Same binary
                        args: Vec::new(),
                        priority: 2,
                        install_methods: Vec::new(),
                    },
                },
            ],
            warnings: Vec::new(),
        };

        let result = merge_all(&scraped).unwrap();
        let py = result.iter().find(|d| d.id() == "python").unwrap();
        assert_eq!(py.lsp_servers.len(), 1);
    }

    #[test]
    fn test_serialize_roundtrip() {
        let scraped = sample_scraped();
        let result = merge_all(&scraped).unwrap();
        let yaml = serialize_definitions(&result).unwrap();
        assert!(yaml.contains("Rust"));
        assert!(yaml.contains("Python"));
        assert!(yaml.contains("tree-sitter/tree-sitter-rust"));
    }
}

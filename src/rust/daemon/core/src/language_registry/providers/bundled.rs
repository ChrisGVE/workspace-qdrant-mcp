//! Bundled default language registry provider.
//!
//! Embeds language definitions at compile time as a YAML resource.
//! This provider serves as the offline fallback when no network
//! providers are available.

use async_trait::async_trait;

use crate::error::DaemonError;
use crate::language_registry::provider::LanguageSourceProvider;
use crate::language_registry::types::{
    GrammarEntry, LanguageEntry, LspEntry, LspServerEntry, ProviderData,
};
use crate::language_registry::LanguageDefinition;

const BUNDLED_YAML: &str = include_str!("../bundled_languages.yaml");

/// Provider that loads language definitions from the bundled YAML resource.
///
/// This provider has the lowest priority (255) and serves as the offline
/// fallback when no network-based providers are available. All 44 known
/// language grammars are included, with semantic patterns for the 25
/// languages that have dedicated extractors.
pub struct BundledProvider {
    definitions: Vec<LanguageDefinition>,
}

impl BundledProvider {
    /// Create a new bundled provider by parsing the embedded YAML.
    ///
    /// # Errors
    ///
    /// Returns `DaemonError::Config` if the embedded YAML fails to parse.
    /// This should never happen in practice since the YAML is validated
    /// at test time.
    pub fn new() -> Result<Self, DaemonError> {
        let definitions: Vec<LanguageDefinition> =
            serde_yaml_ng::from_str(BUNDLED_YAML).map_err(|e| {
                DaemonError::Other(format!("Failed to parse bundled languages: {e}"))
            })?;
        Ok(Self { definitions })
    }
}

#[async_trait]
impl LanguageSourceProvider for BundledProvider {
    fn name(&self) -> &str {
        "bundled"
    }

    fn priority(&self) -> u8 {
        255
    }

    fn last_updated(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        None
    }

    async fn fetch_languages(&self) -> Result<Vec<LanguageEntry>, DaemonError> {
        Ok(self
            .definitions
            .iter()
            .map(|d| LanguageEntry {
                name: d.language.clone(),
                id: d.id(),
                aliases: d.aliases.clone(),
                extensions: d.extensions.clone(),
                language_type: d.language_type,
            })
            .collect())
    }

    async fn fetch_grammars(&self) -> Result<Vec<GrammarEntry>, DaemonError> {
        let mut entries = Vec::new();
        for def in &self.definitions {
            for src in &def.grammar.sources {
                entries.push(GrammarEntry {
                    language: def.id(),
                    repo: src.repo.clone(),
                    quality: src.quality,
                    has_cpp_scanner: def.grammar.has_cpp_scanner,
                    src_subdir: def.grammar.src_subdir.clone(),
                    symbol_name: def.grammar.symbol_name.clone(),
                    archive_branch: def.grammar.archive_branch.clone(),
                });
            }
        }
        Ok(entries)
    }

    async fn fetch_lsp_servers(&self) -> Result<Vec<LspEntry>, DaemonError> {
        let mut entries = Vec::new();
        for def in &self.definitions {
            for server in &def.lsp_servers {
                entries.push(LspEntry {
                    language: def.id(),
                    server: LspServerEntry {
                        name: server.name.clone(),
                        binary: server.binary.clone(),
                        args: server.args.clone(),
                        priority: server.priority,
                        install_methods: server.install_methods.clone(),
                    },
                });
            }
        }
        Ok(entries)
    }

    async fn refresh(&self) -> Result<ProviderData, DaemonError> {
        Ok(ProviderData {
            languages: self.fetch_languages().await?,
            grammars: self.fetch_grammars().await?,
            lsp_servers: self.fetch_lsp_servers().await?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundled_yaml_parses() {
        let provider = BundledProvider::new().expect("bundled YAML should parse");
        assert!(
            provider.definitions.len() > 40,
            "expected > 40 languages, got {}",
            provider.definitions.len()
        );
    }

    #[test]
    fn test_all_44_languages_present() {
        let provider = BundledProvider::new().unwrap();
        let ids: Vec<String> = provider.definitions.iter().map(|d| d.id()).collect();

        let expected = [
            "ada",
            "bash",
            "c",
            "c-sharp",
            "clojure",
            "cpp",
            "css",
            "dart",
            "elixir",
            "elm",
            "erlang",
            "fortran",
            "go",
            "haskell",
            "html",
            "java",
            "javascript",
            "json",
            "julia",
            "kotlin",
            "latex",
            "lisp",
            "lua",
            "markdown",
            "nix",
            "ocaml",
            "odin",
            "pascal",
            "perl",
            "php",
            "python",
            "r",
            "ruby",
            "rust",
            "scala",
            "scheme",
            "sql",
            "swift",
            "toml",
            "tsx",
            "typescript",
            "vala",
            "vue",
            "yaml",
            "zig",
        ];

        for lang in &expected {
            assert!(
                ids.contains(&(*lang).to_string()),
                "missing language: {lang}"
            );
        }
    }

    #[tokio::test]
    async fn test_bundled_languages_cover_known_grammars() {
        let provider = BundledProvider::new().unwrap();
        let languages = provider.fetch_languages().await.unwrap();
        let ids: Vec<&str> = languages.iter().map(|l| l.id.as_str()).collect();
        assert!(ids.contains(&"rust"));
        assert!(ids.contains(&"python"));
        assert!(ids.contains(&"javascript"));
    }

    #[tokio::test]
    async fn test_bundled_grammars() {
        let provider = BundledProvider::new().unwrap();
        let grammars = provider.fetch_grammars().await.unwrap();
        assert!(
            grammars.len() >= 44,
            "expected >= 44 grammars, got {}",
            grammars.len()
        );
        let rust_grammar = grammars.iter().find(|g| g.language == "rust").unwrap();
        assert_eq!(rust_grammar.repo, "tree-sitter/tree-sitter-rust");
    }

    #[tokio::test]
    async fn test_bundled_lsp_servers() {
        let provider = BundledProvider::new().unwrap();
        let servers = provider.fetch_lsp_servers().await.unwrap();

        // We have LSP entries for: python, rust, javascript, typescript, tsx,
        // json, c, cpp, go, java, ruby, php, bash, html
        assert!(
            servers.len() >= 10,
            "expected >= 10 LSP entries, got {}",
            servers.len()
        );

        let rust_servers: Vec<_> = servers.iter().filter(|s| s.language == "rust").collect();
        assert_eq!(rust_servers.len(), 1);
        assert_eq!(rust_servers[0].server.name, "rust-analyzer");

        let python_servers: Vec<_> = servers.iter().filter(|s| s.language == "python").collect();
        assert_eq!(python_servers.len(), 3);
    }

    #[tokio::test]
    async fn test_bundled_refresh() {
        let provider = BundledProvider::new().unwrap();
        let data = provider.refresh().await.unwrap();

        assert!(!data.languages.is_empty());
        assert!(!data.grammars.is_empty());
        assert!(!data.lsp_servers.is_empty());
    }

    #[test]
    fn test_provider_metadata() {
        let provider = BundledProvider::new().unwrap();
        assert_eq!(provider.name(), "bundled");
        assert_eq!(provider.priority(), 255);
        assert!(provider.last_updated().is_none());
        assert!(provider.is_enabled());
    }

    #[test]
    fn test_semantic_patterns_present_for_extractors() {
        let provider = BundledProvider::new().unwrap();

        let languages_with_patterns = [
            "python", "rust", "go", "java", "javascript", "typescript", "tsx", "c", "cpp",
            "ruby", "swift", "bash", "lua", "elixir", "erlang", "scala", "haskell", "zig",
            "odin", "clojure", "ocaml", "fortran", "ada", "perl", "pascal", "lisp",
        ];

        for lang_id in &languages_with_patterns {
            let def = provider
                .definitions
                .iter()
                .find(|d| d.id() == *lang_id)
                .unwrap_or_else(|| panic!("missing language: {lang_id}"));
            assert!(
                def.has_semantic_patterns(),
                "{lang_id} should have semantic patterns"
            );
        }
    }

    #[test]
    fn test_grammar_quality_tiers() {
        let provider = BundledProvider::new().unwrap();

        // Official grammars (tree-sitter org)
        let rust_def = provider.definitions.iter().find(|d| d.id() == "rust").unwrap();
        assert_eq!(
            rust_def.grammar.sources[0].quality,
            crate::language_registry::types::GrammarQuality::Official
        );

        // Curated grammars (tree-sitter-grammars org)
        let lua_def = provider.definitions.iter().find(|d| d.id() == "lua").unwrap();
        assert_eq!(
            lua_def.grammar.sources[0].quality,
            crate::language_registry::types::GrammarQuality::Curated
        );

        // Community grammars
        let ada_def = provider.definitions.iter().find(|d| d.id() == "ada").unwrap();
        assert_eq!(
            ada_def.grammar.sources[0].quality,
            crate::language_registry::types::GrammarQuality::Community
        );
    }
}

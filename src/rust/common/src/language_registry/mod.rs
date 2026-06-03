//! Shared language-registry data types and a pure, synchronous YAML reader
//! (WI-b1).
//!
//! The bundled `language_registry.yaml` (44 languages) is embedded at compile
//! time and parsed into [`LanguageDefinition`] structs by [`RegistryReader`].
//! This module is deliberately free of async, `async_trait`, and daemon error
//! types — the daemon's `LanguageSourceProvider` async wrapper lives in
//! `workspace-qdrant-core` and delegates to this reader.

pub mod types;

pub use types::{
    DocstringStyle, FunctionPatternGroup, GrammarConfig, GrammarEntry, GrammarQuality,
    GrammarSourceEntry, InstallMethod, LanguageDefinition, LanguageEntry, LanguageMap,
    LanguageType, LspEntry, LspServerEntry, MethodPatternGroup, PatternGroup, ProviderData,
    SemanticPatterns, SourceMetadata,
};

/// Errors produced while reading/parsing the bundled language registry.
///
/// Deliberately a leaf error type (no `DaemonError`, no `anyhow`) so the
/// component stays daemon-agnostic; callers map it into their own error enums.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    /// The registry YAML failed to parse into language definitions.
    #[error("failed to parse language registry: {0}")]
    Parse(String),
}

/// Bundled registry YAML, embedded at compile time.
const REGISTRY_YAML: &str = include_str!("language_registry.yaml");

/// Pure, synchronous reader for the bundled language registry.
///
/// Parsing is a stateless operation; the reader is a zero-sized namespace for
/// the parse functions rather than a stateful object.
pub struct RegistryReader;

impl RegistryReader {
    /// Parse the embedded bundled registry into language definitions.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::Parse`] if the embedded YAML is malformed.
    /// This should never happen in practice since the YAML is validated at
    /// test time.
    pub fn bundled() -> Result<Vec<LanguageDefinition>, RegistryError> {
        Self::from_yaml_str(REGISTRY_YAML)
    }

    /// Parse an arbitrary registry YAML string into language definitions.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::Parse`] if `yaml` is not a valid registry
    /// document.
    pub fn from_yaml_str(yaml: &str) -> Result<Vec<LanguageDefinition>, RegistryError> {
        serde_yaml_ng::from_str(yaml).map_err(|e| RegistryError::Parse(e.to_string()))
    }
}

/// Sorted, de-duplicated list of language IDs that have at least one grammar
/// source in the bundled registry — i.e. the grammars available for download.
///
/// Pure sync; the parsed result is cached for the process lifetime so the
/// returned `&'static str` slices are valid. On a (compile-time-impossible)
/// parse failure the list is empty.
pub fn known_grammar_languages() -> Vec<&'static str> {
    static LANGS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    LANGS
        .get_or_init(|| {
            let mut ids: Vec<String> = RegistryReader::bundled()
                .unwrap_or_default()
                .iter()
                .filter(|d| d.has_grammar())
                .map(|d| d.id())
                .collect();
            ids.sort();
            ids.dedup();
            ids
        })
        .iter()
        .map(|s| s.as_str())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_yaml_parses() {
        let defs = RegistryReader::bundled().expect("bundled YAML should parse");
        assert!(
            defs.len() > 40,
            "expected > 40 languages, got {}",
            defs.len()
        );
    }

    #[test]
    fn all_44_languages_present() {
        let defs = RegistryReader::bundled().unwrap();
        let ids: Vec<String> = defs.iter().map(|d| d.id()).collect();

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

    #[test]
    fn semantic_patterns_present_for_extractors() {
        let defs = RegistryReader::bundled().unwrap();

        let languages_with_patterns = [
            "python",
            "rust",
            "go",
            "java",
            "javascript",
            "typescript",
            "tsx",
            "c",
            "cpp",
            "ruby",
            "swift",
            "bash",
            "lua",
            "elixir",
            "erlang",
            "scala",
            "haskell",
            "zig",
            "odin",
            "clojure",
            "ocaml",
            "fortran",
            "ada",
            "perl",
            "pascal",
            "lisp",
        ];

        for lang_id in &languages_with_patterns {
            let def = defs
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
    fn grammar_quality_tiers() {
        let defs = RegistryReader::bundled().unwrap();

        // Official grammars (tree-sitter org).
        let rust_def = defs.iter().find(|d| d.id() == "rust").unwrap();
        assert_eq!(
            rust_def.grammar.sources[0].quality,
            GrammarQuality::Official
        );

        // Curated grammars (tree-sitter-grammars org).
        let lua_def = defs.iter().find(|d| d.id() == "lua").unwrap();
        assert_eq!(lua_def.grammar.sources[0].quality, GrammarQuality::Curated);

        // Community grammars.
        let ada_def = defs.iter().find(|d| d.id() == "ada").unwrap();
        assert_eq!(
            ada_def.grammar.sources[0].quality,
            GrammarQuality::Community
        );
    }

    #[test]
    fn from_yaml_str_rejects_malformed() {
        let err = RegistryReader::from_yaml_str("not: valid: yaml: [[[").unwrap_err();
        assert!(matches!(err, RegistryError::Parse(_)));
    }
}

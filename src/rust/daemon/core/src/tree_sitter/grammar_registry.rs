//! Registry of known tree-sitter grammar sources.
//!
//! Maps language names to their GitHub repository and build metadata.
//! Derives all grammar source data from the language registry YAML.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::language_registry::providers::registry::RegistryProvider;

/// Information about a tree-sitter grammar source.
#[derive(Debug, Clone)]
pub struct GrammarSource {
    /// GitHub owner/org (e.g., "tree-sitter")
    pub owner: String,
    /// GitHub repository name (e.g., "tree-sitter-rust")
    pub repo: String,
    /// The symbol name suffix exported by the grammar.
    /// Usually matches the language name, but some grammars differ
    /// (e.g., "commonlisp" grammar exports `tree_sitter_commonlisp`).
    /// If None, derived from the language name.
    pub symbol_name: Option<String>,
    /// Whether the grammar includes a C++ scanner (scanner.cc).
    /// Requires a C++ compiler instead of just a C compiler.
    pub has_cpp_scanner: bool,
    /// Subdirectory within the repo that contains the grammar src/.
    /// Most grammars have src/ at the root. Some monorepos (like
    /// tree-sitter-typescript) have it under a subdirectory.
    pub src_subdir: Option<String>,
    /// Non-standard branch to use for archive fallback.
    /// Defaults to trying "main" then "master". Some repos keep
    /// generated parser.c only on a "release" branch.
    pub archive_branch: Option<String>,
}

impl GrammarSource {
    /// Get the C symbol name for this grammar.
    /// Returns the explicit symbol_name if set, otherwise derives from repo name.
    pub fn c_symbol_name(&self, language: &str) -> String {
        if let Some(ref sym) = self.symbol_name {
            format!("tree_sitter_{sym}")
        } else {
            format!("tree_sitter_{}", language.replace('-', "_"))
        }
    }

    /// Get the GitHub release tarball URL for a specific tag.
    pub fn release_tarball_url(&self, tag: &str) -> String {
        format!(
            "https://github.com/{}/{}/releases/download/{}/{}.tar.gz",
            self.owner, self.repo, tag, self.repo
        )
    }

    /// Get the GitHub archive URL for a branch/tag (fallback when no releases exist).
    pub fn archive_tarball_url(&self, git_ref: &str) -> String {
        format!(
            "https://github.com/{}/{}/archive/refs/heads/{}.tar.gz",
            self.owner, self.repo, git_ref
        )
    }
}

/// Lazily loaded grammar registry data.
struct GrammarRegistryData {
    /// language_id → GrammarSource
    sources: HashMap<String, GrammarSource>,
    /// alias → canonical language_id
    aliases: HashMap<String, String>,
}

fn registry_data() -> &'static GrammarRegistryData {
    static DATA: OnceLock<GrammarRegistryData> = OnceLock::new();
    DATA.get_or_init(|| {
        let provider = match RegistryProvider::new() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("Failed to load language registry for grammar sources: {e}");
                return GrammarRegistryData {
                    sources: HashMap::new(),
                    aliases: HashMap::new(),
                };
            }
        };

        let mut sources = HashMap::new();
        let mut aliases = HashMap::new();

        for def in provider.definitions() {
            let lang_id = def.id();

            // Build alias map
            for alias in &def.aliases {
                aliases.insert(alias.to_lowercase(), lang_id.clone());
            }

            // Use the first (highest-quality) grammar source
            if let Some(src) = def.grammar.sources.first() {
                let (owner, repo) = parse_repo(&src.repo);
                sources.insert(
                    lang_id,
                    GrammarSource {
                        owner,
                        repo,
                        symbol_name: def.grammar.symbol_name.clone(),
                        has_cpp_scanner: def.grammar.has_cpp_scanner,
                        src_subdir: def.grammar.src_subdir.clone(),
                        archive_branch: def.grammar.archive_branch.clone(),
                    },
                );
            }
        }

        GrammarRegistryData { sources, aliases }
    })
}

/// Parse "owner/repo" format into (owner, repo).
fn parse_repo(full_repo: &str) -> (String, String) {
    match full_repo.split_once('/') {
        Some((owner, repo)) => (owner.to_string(), repo.to_string()),
        None => ("unknown".to_string(), full_repo.to_string()),
    }
}

/// Look up the grammar source for a language.
///
/// Returns `None` for unknown languages — the caller should fall back
/// to guessing `tree-sitter/tree-sitter-{language}`.
pub fn lookup(language: &str) -> Option<GrammarSource> {
    let data = registry_data();
    let normalized = language.to_lowercase();

    // Direct lookup
    if let Some(src) = data.sources.get(&normalized) {
        return Some(src.clone());
    }

    // Alias lookup
    if let Some(canonical) = data.aliases.get(&normalized) {
        return data.sources.get(canonical).cloned();
    }

    None
}

/// List all languages with known grammar sources.
pub fn known_languages() -> Vec<String> {
    let data = registry_data();
    let mut langs: Vec<String> = data.sources.keys().cloned().collect();
    langs.sort();
    langs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_known_language() {
        let rust = lookup("rust").unwrap();
        assert_eq!(rust.owner, "tree-sitter");
        assert_eq!(rust.repo, "tree-sitter-rust");
        assert!(!rust.has_cpp_scanner);
        assert!(rust.src_subdir.is_none());
    }

    #[test]
    fn test_lookup_with_cpp_scanner() {
        let cpp = lookup("cpp").unwrap();
        assert!(cpp.has_cpp_scanner);
    }

    #[test]
    fn test_lookup_with_subdir() {
        let ts = lookup("typescript").unwrap();
        assert_eq!(ts.src_subdir.as_deref(), Some("typescript"));
    }

    #[test]
    fn test_lookup_alias() {
        let lisp = lookup("commonlisp").unwrap();
        assert_eq!(lisp.repo, "tree-sitter-commonlisp");

        let csharp = lookup("csharp").unwrap();
        assert_eq!(csharp.repo, "tree-sitter-c-sharp");
    }

    #[test]
    fn test_lookup_unknown() {
        assert!(lookup("brainfuck").is_none());
    }

    #[test]
    fn test_c_symbol_name() {
        let rust = lookup("rust").unwrap();
        assert_eq!(rust.c_symbol_name("rust"), "tree_sitter_rust");

        let lisp = lookup("lisp").unwrap();
        assert_eq!(lisp.c_symbol_name("lisp"), "tree_sitter_commonlisp");

        let csharp = lookup("c-sharp").unwrap();
        assert_eq!(csharp.c_symbol_name("c-sharp"), "tree_sitter_c_sharp");
    }

    #[test]
    fn test_release_tarball_url() {
        let rust = lookup("rust").unwrap();
        assert_eq!(
            rust.release_tarball_url("v0.24.0"),
            "https://github.com/tree-sitter/tree-sitter-rust/releases/download/v0.24.0/tree-sitter-rust.tar.gz"
        );
    }

    #[test]
    fn test_known_languages_not_empty() {
        let langs = known_languages();
        assert!(langs.len() > 40);
        assert!(langs.contains(&"rust".to_string()));
        assert!(langs.contains(&"python".to_string()));
    }

    #[test]
    fn test_parse_repo() {
        let (owner, repo) = parse_repo("tree-sitter/tree-sitter-rust");
        assert_eq!(owner, "tree-sitter");
        assert_eq!(repo, "tree-sitter-rust");
    }
}

//! nvim-treesitter grammar metadata provider.
//!
//! Fetches grammar repository mappings from the nvim-treesitter lockfile.
//! This is the primary source for language-to-grammar-repo mappings.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::DaemonError;
use crate::language_registry::provider::LanguageSourceProvider;
use crate::language_registry::types::{
    GrammarEntry, GrammarQuality, LanguageEntry, LspEntry, ProviderData,
};

const LOCKFILE_URL: &str =
    "https://raw.githubusercontent.com/nvim-treesitter/nvim-treesitter/master/lockfile.json";

/// nvim-treesitter lockfile entry.
#[derive(Debug, Deserialize)]
struct LockfileEntry {
    /// Git revision hash.
    #[allow(dead_code)]
    revision: String,
}

/// Provider that fetches grammar metadata from nvim-treesitter.
///
/// Priority 20 — authoritative for grammar-to-repo mappings, but the
/// lockfile doesn't include build metadata (C++ scanner, subdirectory).
pub struct NvimTreesitterProvider {
    url: String,
    cached_data: Mutex<Option<Vec<GrammarEntry>>>,
}

impl NvimTreesitterProvider {
    /// Create a new provider with the default lockfile URL.
    pub fn new() -> Self {
        Self {
            url: LOCKFILE_URL.to_string(),
            cached_data: Mutex::new(None),
        }
    }

    /// Create a provider with a custom URL (for testing).
    pub fn with_url(url: String) -> Self {
        Self {
            url,
            cached_data: Mutex::new(None),
        }
    }

    /// Parse the lockfile JSON into grammar entries.
    fn parse_lockfile(json: &str) -> Result<Vec<GrammarEntry>, DaemonError> {
        let raw: HashMap<String, LockfileEntry> = serde_json::from_str(json)
            .map_err(|e| DaemonError::Other(format!("Failed to parse lockfile: {e}")))?;

        let mut entries = Vec::with_capacity(raw.len());

        for (language, _entry) in &raw {
            // nvim-treesitter uses a convention: tree-sitter-{language}
            // The lockfile keys are language names, repos follow the naming convention
            let repo = infer_repo(language);

            entries.push(GrammarEntry {
                language: language.clone(),
                repo,
                quality: GrammarQuality::Community, // nvim-treesitter curates but doesn't tier
                has_cpp_scanner: false,              // Not available from lockfile
                src_subdir: None,
                symbol_name: None,
                archive_branch: None,
            });
        }

        entries.sort_by(|a, b| a.language.cmp(&b.language));
        Ok(entries)
    }
}

/// Infer the GitHub repo name from a language identifier.
///
/// Most grammars follow the `tree-sitter/tree-sitter-{lang}` pattern.
/// Known exceptions are mapped explicitly.
fn infer_repo(language: &str) -> String {
    // Known repo overrides (language → owner/repo)
    match language {
        "c_sharp" => "tree-sitter/tree-sitter-c-sharp".to_string(),
        "commonlisp" => "theHamsta/tree-sitter-commonlisp".to_string(),
        "cuda" => "theHamsta/tree-sitter-cuda".to_string(),
        "dockerfile" => "camdencheek/tree-sitter-dockerfile".to_string(),
        "fish" => "ram02z/tree-sitter-fish".to_string(),
        "gitcommit" => "the-mikedavis/tree-sitter-git-commit".to_string(),
        "git_rebase" => "the-mikedavis/tree-sitter-git-rebase".to_string(),
        "glsl" => "theHamsta/tree-sitter-glsl".to_string(),
        "godot_resource" => "PrestonKnoworthy/tree-sitter-godot-resource".to_string(),
        "hlsl" => "theHamsta/tree-sitter-hlsl".to_string(),
        "http" => "rest-nvim/tree-sitter-http".to_string(),
        "ini" => "justinmk/tree-sitter-ini".to_string(),
        "jsdoc" => "tree-sitter/tree-sitter-jsdoc".to_string(),
        "json5" => "Joakker/tree-sitter-json5".to_string(),
        "latex" => "latex-lsp/tree-sitter-latex".to_string(),
        "llvm" => "benwilliamgraham/tree-sitter-llvm".to_string(),
        "make" => "alemuller/tree-sitter-make".to_string(),
        "markdown" => "tree-sitter-grammars/tree-sitter-markdown".to_string(),
        "nix" => "cstrahan/tree-sitter-nix".to_string(),
        "org" => "milisims/tree-sitter-org".to_string(),
        "pascal" => "Isopod/tree-sitter-pascal".to_string(),
        "proto" => "treywood/tree-sitter-proto".to_string(),
        "regex" => "tree-sitter/tree-sitter-regex".to_string(),
        "rst" => "stsewd/tree-sitter-rst".to_string(),
        "sql" => "derekstride/tree-sitter-sql".to_string(),
        "svelte" => "tree-sitter-grammars/tree-sitter-svelte".to_string(),
        "terraform" => "MichaHoffmann/tree-sitter-hcl".to_string(),
        "toml" => "tree-sitter-grammars/tree-sitter-toml".to_string(),
        "vim" => "neovim/tree-sitter-vim".to_string(),
        "vimdoc" => "neovim/tree-sitter-vimdoc".to_string(),
        "yaml" => "tree-sitter-grammars/tree-sitter-yaml".to_string(),
        _ => format!("tree-sitter/tree-sitter-{language}"),
    }
}

#[async_trait]
impl LanguageSourceProvider for NvimTreesitterProvider {
    fn name(&self) -> &str {
        "nvim-treesitter"
    }

    fn priority(&self) -> u8 {
        20
    }

    fn last_updated(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        None
    }

    async fn fetch_languages(&self) -> Result<Vec<LanguageEntry>, DaemonError> {
        // nvim-treesitter doesn't provide language identity data
        Ok(Vec::new())
    }

    async fn fetch_grammars(&self) -> Result<Vec<GrammarEntry>, DaemonError> {
        if let Some(ref cached) = *self.cached_data.lock().unwrap() {
            return Ok(cached.clone());
        }

        let response = reqwest::get(&self.url).await.map_err(|e| {
            DaemonError::Other(format!("Failed to fetch nvim-treesitter lockfile: {e}"))
        })?;

        if !response.status().is_success() {
            return Err(DaemonError::Other(format!(
                "nvim-treesitter fetch failed: {}",
                response.status()
            )));
        }

        let body = response.text().await.map_err(|e| {
            DaemonError::Other(format!("Failed to read lockfile response: {e}"))
        })?;

        let entries = Self::parse_lockfile(&body)?;
        *self.cached_data.lock().unwrap() = Some(entries.clone());
        Ok(entries)
    }

    async fn fetch_lsp_servers(&self) -> Result<Vec<LspEntry>, DaemonError> {
        Ok(Vec::new())
    }

    async fn refresh(&self) -> Result<ProviderData, DaemonError> {
        *self.cached_data.lock().unwrap() = None;
        Ok(ProviderData {
            languages: Vec::new(),
            grammars: self.fetch_grammars().await?,
            lsp_servers: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_LOCKFILE: &str = r#"{
  "bash": {
    "revision": "5caf530d7fc4baa8c33d6e795a2d0b72ccbdc11c"
  },
  "c": {
    "revision": "daf1e1ff9b5b47a7a7a43aa5b0f9ca1e28ce97f4"
  },
  "python": {
    "revision": "4bfdd9033a2225cc95032ce2e02f91700e6c6582"
  },
  "rust": {
    "revision": "9c3ae5ebc638c019c tried"
  }
}"#;

    #[test]
    fn test_parse_lockfile() {
        let entries = NvimTreesitterProvider::parse_lockfile(SAMPLE_LOCKFILE).unwrap();
        assert_eq!(entries.len(), 4);

        let rust = entries.iter().find(|e| e.language == "rust").unwrap();
        assert_eq!(rust.repo, "tree-sitter/tree-sitter-rust");
        assert_eq!(rust.quality, GrammarQuality::Community);

        let python = entries.iter().find(|e| e.language == "python").unwrap();
        assert_eq!(python.repo, "tree-sitter/tree-sitter-python");
    }

    #[test]
    fn test_infer_repo_overrides() {
        assert_eq!(
            infer_repo("c_sharp"),
            "tree-sitter/tree-sitter-c-sharp"
        );
        assert_eq!(
            infer_repo("markdown"),
            "tree-sitter-grammars/tree-sitter-markdown"
        );
        assert_eq!(
            infer_repo("toml"),
            "tree-sitter-grammars/tree-sitter-toml"
        );
        assert_eq!(
            infer_repo("latex"),
            "latex-lsp/tree-sitter-latex"
        );
    }

    #[test]
    fn test_infer_repo_default() {
        assert_eq!(infer_repo("rust"), "tree-sitter/tree-sitter-rust");
        assert_eq!(infer_repo("python"), "tree-sitter/tree-sitter-python");
        assert_eq!(infer_repo("go"), "tree-sitter/tree-sitter-go");
    }

    #[test]
    fn test_provider_metadata() {
        let provider = NvimTreesitterProvider::new();
        assert_eq!(provider.name(), "nvim-treesitter");
        assert_eq!(provider.priority(), 20);
        assert!(provider.is_enabled());
    }
}

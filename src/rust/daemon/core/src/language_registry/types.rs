//! Core data types for the dynamic language registry.
//!
//! These types represent the unified language metadata schema used throughout the
//! registry system. They are populated by providers and stored as YAML files.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

// ── Language identity ─────────────────────────────────────────────────

/// The type/category of a language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LanguageType {
    Programming,
    Data,
    Markup,
    Prose,
}

impl fmt::Display for LanguageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Programming => write!(f, "programming"),
            Self::Data => write!(f, "data"),
            Self::Markup => write!(f, "markup"),
            Self::Prose => write!(f, "prose"),
        }
    }
}

/// A language's identity metadata (from Linguist or equivalent).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageEntry {
    /// Canonical language name (e.g., "Python", "Rust").
    pub name: String,
    /// Internal identifier (lowercase, e.g., "python", "rust").
    pub id: String,
    /// Alternate names (e.g., ["python3", "py"]).
    #[serde(default)]
    pub aliases: Vec<String>,
    /// File extensions including dot (e.g., [".py", ".pyw"]).
    #[serde(default)]
    pub extensions: Vec<String>,
    /// Language type.
    #[serde(rename = "type")]
    pub language_type: LanguageType,
}

// ── Grammar metadata ──────────────────────────────────────────────────

/// Quality tier for a grammar source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GrammarQuality {
    /// Maintained by tree-sitter-grammars org — highest quality.
    Curated,
    /// From the official tree-sitter org or language maintainers.
    Official,
    /// Community-maintained grammar.
    Community,
}

impl fmt::Display for GrammarQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Curated => write!(f, "curated"),
            Self::Official => write!(f, "official"),
            Self::Community => write!(f, "community"),
        }
    }
}

/// A single grammar source repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarSourceEntry {
    /// GitHub owner/repo (e.g., "tree-sitter/tree-sitter-python").
    pub repo: String,
    /// Which provider discovered this source.
    #[serde(default)]
    pub origin: Option<String>,
    /// Quality tier.
    #[serde(default = "default_grammar_quality")]
    pub quality: GrammarQuality,
}

fn default_grammar_quality() -> GrammarQuality {
    GrammarQuality::Community
}

/// Grammar configuration for a language.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarConfig {
    /// Available grammar sources, ordered by preference.
    #[serde(default)]
    pub sources: Vec<GrammarSourceEntry>,
    /// Whether the grammar requires a C++ compiler (scanner.cc).
    #[serde(default)]
    pub has_cpp_scanner: bool,
    /// Subdirectory within the repo containing the grammar source.
    #[serde(default)]
    pub src_subdir: Option<String>,
    /// Override the C symbol name for `tree_sitter_<language>`.
    #[serde(default)]
    pub symbol_name: Option<String>,
    /// Override the branch used for archive tarball download.
    #[serde(default)]
    pub archive_branch: Option<String>,
}

impl Default for GrammarConfig {
    fn default() -> Self {
        Self {
            sources: Vec::new(),
            has_cpp_scanner: false,
            src_subdir: None,
            symbol_name: None,
            archive_branch: None,
        }
    }
}

/// A grammar entry as returned by a provider (before merging).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarEntry {
    /// Language identifier this grammar is for.
    pub language: String,
    /// Grammar repository (owner/repo).
    pub repo: String,
    /// Quality tier.
    pub quality: GrammarQuality,
    /// Whether a C++ scanner is required.
    #[serde(default)]
    pub has_cpp_scanner: bool,
    /// Subdirectory for grammar source within the repo.
    #[serde(default)]
    pub src_subdir: Option<String>,
    /// Override C symbol name.
    #[serde(default)]
    pub symbol_name: Option<String>,
    /// Override archive branch.
    #[serde(default)]
    pub archive_branch: Option<String>,
}

// ── Semantic patterns ─────────────────────────────────────────────────

/// Semantic patterns for AST-driven chunk extraction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticPatterns {
    /// Preamble patterns (imports, includes, requires).
    #[serde(default)]
    pub preamble: PatternGroup,
    /// Function definition patterns.
    #[serde(default)]
    pub function: FunctionPatternGroup,
    /// Class definition patterns.
    #[serde(default)]
    pub class: PatternGroup,
    /// Method patterns (functions inside class scope).
    #[serde(default)]
    pub method: MethodPatternGroup,
    /// Struct/record definition patterns.
    #[serde(default, rename = "struct")]
    pub struct_def: PatternGroup,
    /// Enum definition patterns.
    #[serde(default, rename = "enum")]
    pub enum_def: PatternGroup,
    /// Trait definition patterns.
    #[serde(default, rename = "trait")]
    pub trait_def: PatternGroup,
    /// Interface/protocol definition patterns.
    #[serde(default)]
    pub interface: PatternGroup,
    /// Module/namespace definition patterns.
    #[serde(default)]
    pub module: PatternGroup,
    /// Constant definition patterns.
    #[serde(default)]
    pub constant: PatternGroup,
    /// Macro definition patterns.
    #[serde(default, rename = "macro")]
    pub macro_def: PatternGroup,
    /// Type alias patterns.
    #[serde(default)]
    pub type_alias: PatternGroup,
    /// Impl block patterns (Rust-specific).
    #[serde(default, rename = "impl")]
    pub impl_block: PatternGroup,
    /// AST node type name used for identifiers.
    #[serde(default)]
    pub name_node: Option<String>,
    /// AST node type name for function/class bodies.
    #[serde(default)]
    pub body_node: Option<String>,
    /// AST node types for comments.
    #[serde(default)]
    pub comment_nodes: Vec<String>,
    /// Docstring extraction style.
    #[serde(default)]
    pub docstring_style: DocstringStyle,
    /// Node type wrapping decorated definitions (e.g., Python decorators).
    #[serde(default)]
    pub decorated_wrapper: Option<String>,
}

/// A group of AST node types matching a semantic category.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternGroup {
    /// AST node type names to match.
    #[serde(default)]
    pub node_types: Vec<String>,
}

/// Function pattern group with async variant support.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionPatternGroup {
    /// AST node types for function definitions.
    #[serde(default)]
    pub node_types: Vec<String>,
    /// AST node types specifically for async functions.
    #[serde(default)]
    pub async_node_types: Vec<String>,
}

/// Method pattern group (functions inside class scope).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MethodPatternGroup {
    /// AST node types for method definitions.
    #[serde(default)]
    pub node_types: Vec<String>,
    /// Context requirement (e.g., "inside_class").
    #[serde(default)]
    pub context: Option<String>,
}

/// Docstring extraction style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocstringStyle {
    /// Collect comment nodes preceding the definition (C, C++, Rust, Go, Ruby).
    PrecedingComments,
    /// First string expression in the function/class body (Python).
    FirstStringInBody,
    /// `/** ... */` block comment before definition (Java, JS, TS, Scala).
    Javadoc,
    /// `-- |` or `-- ^` Haddock comments (Haskell).
    Haddock,
    /// POD blocks (Perl).
    Pod,
    /// `@doc` attribute (Elixir).
    ElixirAttr,
    /// `(** ... *)` OCaml doc comments.
    OcamlDoc,
    /// No docstring extraction.
    None,
}

impl Default for DocstringStyle {
    fn default() -> Self {
        Self::PrecedingComments
    }
}

// ── LSP server metadata ───────────────────────────────────────────────

/// An LSP server installation method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallMethod {
    /// Package manager binary name (e.g., "brew", "pip", "cargo").
    pub manager: String,
    /// Full install command (e.g., "brew install rust-analyzer").
    pub command: String,
}

/// An LSP server entry for a language.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspServerEntry {
    /// Human-readable server name (e.g., "ruff", "rust-analyzer").
    pub name: String,
    /// Binary executable name.
    pub binary: String,
    /// Command-line arguments to start the server.
    #[serde(default)]
    pub args: Vec<String>,
    /// Priority (lower = preferred).
    #[serde(default = "default_lsp_priority")]
    pub priority: u8,
    /// Available installation methods.
    #[serde(default)]
    pub install_methods: Vec<InstallMethod>,
}

fn default_lsp_priority() -> u8 {
    10
}

/// An LSP entry as returned by a provider (before merging).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspEntry {
    /// Language identifier this LSP server supports.
    pub language: String,
    /// Server details.
    #[serde(flatten)]
    pub server: LspServerEntry,
}

// ── Source tracking metadata ──────────────────────────────────────────

/// Tracks which provider contributed each section of a language definition.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceMetadata {
    /// Provider that contributed the language identity.
    #[serde(default)]
    pub language: Option<String>,
    /// Provider that contributed the grammar configuration.
    #[serde(default)]
    pub grammar: Option<String>,
    /// Provider that contributed the LSP server list.
    #[serde(default)]
    pub lsp: Option<String>,
    /// When the data was last refreshed from providers.
    #[serde(default)]
    pub last_refreshed: Option<String>,
}

// ── Complete language definition ──────────────────────────────────────

/// A complete language definition combining all metadata.
///
/// This is the canonical type stored in YAML files and used at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDefinition {
    /// Canonical language name.
    pub language: String,
    /// Alternate names.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// File extensions including dot.
    #[serde(default)]
    pub extensions: Vec<String>,
    /// Language type.
    #[serde(rename = "type", default = "default_language_type")]
    pub language_type: LanguageType,
    /// Grammar configuration.
    #[serde(default)]
    pub grammar: GrammarConfig,
    /// Semantic extraction patterns for the generic AST walker.
    #[serde(default)]
    pub semantic_patterns: Option<SemanticPatterns>,
    /// Available LSP servers.
    #[serde(default)]
    pub lsp_servers: Vec<LspServerEntry>,
    /// Source provenance metadata.
    #[serde(default, rename = "_sources")]
    pub sources: SourceMetadata,
}

fn default_language_type() -> LanguageType {
    LanguageType::Programming
}

impl LanguageDefinition {
    /// Get the internal identifier (lowercase language name).
    pub fn id(&self) -> String {
        self.language.to_lowercase()
    }

    /// Check if this language has grammar sources defined.
    pub fn has_grammar(&self) -> bool {
        !self.grammar.sources.is_empty()
    }

    /// Check if this language has semantic patterns defined.
    pub fn has_semantic_patterns(&self) -> bool {
        self.semantic_patterns.is_some()
    }

    /// Check if this language has LSP servers defined.
    pub fn has_lsp(&self) -> bool {
        !self.lsp_servers.is_empty()
    }

    /// Get the preferred grammar source (first in list, highest quality).
    pub fn preferred_grammar(&self) -> Option<&GrammarSourceEntry> {
        self.grammar.sources.first()
    }

    /// Get the preferred LSP server (lowest priority number).
    pub fn preferred_lsp(&self) -> Option<&LspServerEntry> {
        self.lsp_servers.iter().min_by_key(|s| s.priority)
    }
}

// ── Provider result type ──────────────────────────────────────────────

/// Aggregated data from a provider refresh.
#[derive(Debug, Clone, Default)]
pub struct ProviderData {
    /// Language identity entries.
    pub languages: Vec<LanguageEntry>,
    /// Grammar metadata entries.
    pub grammars: Vec<GrammarEntry>,
    /// LSP server entries.
    pub lsp_servers: Vec<LspEntry>,
}

/// Map of language ID → LanguageDefinition.
pub type LanguageMap = HashMap<String, LanguageDefinition>;

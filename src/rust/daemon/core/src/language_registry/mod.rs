//! Dynamic language registry system.
//!
//! Provides a data-driven, YAML-based language registry that replaces all
//! hardcoded language support. The registry is populated by pluggable
//! upstream providers (GitHub Linguist, nvim-treesitter, mason.nvim, etc.)
//! and drives grammar download, semantic chunking patterns, and LSP
//! server discovery at runtime.
//!
//! # Architecture
//!
//! ```text
//! LanguageSourceProvider (trait)
//!   ├── LinguistProvider      (language identity)
//!   ├── NvimTreesitterProvider (grammar repos)
//!   ├── TreeSitterGrammarsOrgProvider (curated grammars)
//!   ├── MasonProvider         (LSP servers)
//!   └── BundledProvider       (offline fallback)
//!         │
//!         v
//!   LanguageRegistry (merge + cache)
//!         │
//!         v
//!   LanguageDefinition (YAML per language)
//!         │
//!     ┌───┴───┐
//!     v       v
//!  Generic   LSP
//!  AST       Discovery
//!  Walker
//! ```

pub mod provider;
pub mod providers;
pub mod types;

// Re-export key types for convenience.
pub use provider::{LanguageSourceProvider, ProviderConfig};
pub use types::{
    DocstringStyle, FunctionPatternGroup, GrammarConfig, GrammarEntry, GrammarQuality,
    GrammarSourceEntry, InstallMethod, LanguageDefinition, LanguageEntry, LanguageMap,
    LanguageType, LspEntry, LspServerEntry, MethodPatternGroup, PatternGroup, ProviderData,
    SemanticPatterns, SourceMetadata,
};

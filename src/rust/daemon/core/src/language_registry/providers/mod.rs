//! Upstream data source provider implementations.
//!
//! Each provider implements `LanguageSourceProvider` to fetch language
//! metadata from a specific upstream source of truth.

pub mod registry;
pub mod linguist;
pub mod mason;
pub mod nvim_treesitter;
pub mod ts_grammars_org;

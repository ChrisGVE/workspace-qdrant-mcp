//! Exact substring search over code_lines using FTS5 trigram index.
//!
//! Uses a two-phase approach:
//! 1. FTS5 trigram MATCH for fast candidate selection
//! 2. INSTR verification for exact substring match

pub mod context;
mod query_builder;
mod search;

#[cfg(test)]
mod tests;

// Public API — same surface as the original single-file module.
pub use search::search_exact;

// Crate-internal API — used by other text_search submodules.
// Re-export for convenience within the crate.
pub use context::attach_context_lines;

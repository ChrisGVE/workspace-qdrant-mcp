//! Regex search over code_lines with trigram acceleration (Task 54).
//!
//! Extracts literal substrings from regex patterns, uses FTS5 trigram MATCH
//! for candidate pre-filtering, then verifies with Rust's regex engine.

mod query;
mod search;

#[cfg(test)]
mod tests;

// Public API
pub use search::search_regex;

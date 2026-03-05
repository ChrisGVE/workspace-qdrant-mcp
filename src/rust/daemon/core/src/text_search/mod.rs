//! Text Search on FTS5 (Tasks 53, 54)
//!
//! Provides exact substring search and regex search over code_lines using the
//! FTS5 trigram index for candidate pre-filtering.
//!
//! ## Architecture
//!
//! ### Exact search (`search_exact`)
//! 1. **FTS5 trigram MATCH** — pre-filter using the trigram index (fast, ~O(1) per term)
//! 2. **INSTR verification** — exact match filter on `code_lines.content`
//! 3. **Materialized `line_number`** — reads pre-computed 1-based line numbers directly
//! 4. **file_metadata JOIN** — scopes results by project/branch/path
//!
//! ### Regex search (`search_regex`)
//! 1. **Literal extraction** — extract literal substrings (>=3 chars) from regex
//! 2. **FTS5 trigram MATCH** — pre-filter using OR query of extracted literals
//! 3. **Rust regex verification** — `regex::Regex::is_match()` on each candidate
//! 4. Falls back to full table scan when no literals can be extracted
//!
//! ## FTS5 Trigram Pattern Escaping
//!
//! FTS5 trigram tokenizer treats `"` as special. All patterns must be double-quote
//! wrapped for exact phrase matching. Internal double quotes are escaped as `""`.
//! Patterns shorter than 3 characters cannot use the trigram index and fall back
//! to a full table scan with LIKE only.

mod escaping;
mod exact_search;
mod regex_parser;
mod regex_search;
mod types;

// Public API
pub use escaping::{escape_fts5_pattern, escape_like_pattern};
pub use exact_search::search_exact;
pub use regex_parser::extract_literals_from_regex;
pub use regex_search::search_regex;
pub use types::{RegexLiterals, SearchMatch, SearchOptions, SearchResults};

// Crate-internal API (used by grep_search)
pub(crate) use escaping::{compile_glob_matcher, resolve_path_filter};

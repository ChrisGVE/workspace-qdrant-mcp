/// A single search match in a code file.
#[derive(Debug, Clone)]
pub struct SearchMatch {
    /// line_id from code_lines (primary key).
    pub line_id: i64,
    /// file_id reference to tracked_files.
    pub file_id: i64,
    /// 1-based line number within the file.
    pub line_number: i64,
    /// Full content of the matching line.
    pub content: String,
    /// File path from file_metadata.
    pub file_path: String,
    /// Tenant ID from file_metadata.
    pub tenant_id: String,
    /// Branch from file_metadata (may be empty).
    pub branch: Option<String>,
    /// Lines before the match (populated when context_lines > 0).
    pub context_before: Vec<String>,
    /// Lines after the match (populated when context_lines > 0).
    pub context_after: Vec<String>,
}

/// Search options for scoping and filtering.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Scope to a specific project (tenant_id).
    pub tenant_id: Option<String>,
    /// Scope to a specific branch.
    pub branch: Option<String>,
    /// Filter by file path prefix (e.g., "src/").
    pub path_prefix: Option<String>,
    /// Filter by file path glob pattern (e.g., "**/*.rs", "src/**/*.ts").
    ///
    /// Supports `?`, `*`, `**`, `[...]` patterns. When set, takes precedence
    /// over `path_prefix`. A SQL prefix is extracted from the glob for
    /// pre-filtering, then `glob::Pattern` verifies in Rust.
    pub path_glob: Option<String>,
    /// Case-insensitive search (default: false = case-sensitive).
    pub case_insensitive: bool,
    /// Maximum number of results to return (0 = unlimited).
    pub max_results: usize,
    /// Number of context lines before and after each match (0 = none).
    pub context_lines: usize,
}

/// Aggregated search results.
#[derive(Debug, Clone)]
pub struct SearchResults {
    /// The search pattern used.
    pub pattern: String,
    /// Matching lines.
    pub matches: Vec<SearchMatch>,
    /// Whether results were truncated by max_results.
    pub truncated: bool,
    /// Time spent in the FTS5 query (milliseconds).
    pub query_time_ms: u64,
    /// Which search engine produced these results ("fts5" or "grep").
    pub search_engine: String,
}

/// Structured representation of literals extracted from a regex pattern.
///
/// Separates mandatory literals (must all appear) from alternation groups
/// (at least one branch must appear). This allows building AND/OR FTS5 queries
/// instead of flat OR queries, dramatically reducing candidate counts.
#[derive(Debug, Clone, PartialEq)]
pub struct RegexLiterals {
    /// Literals that must all appear in a matching line (AND semantics).
    pub mandatory: Vec<String>,
    /// Groups of literals from alternation (`|`). Within each group, any one
    /// branch matching is sufficient (OR semantics). Groups themselves are
    /// AND'd with the mandatory literals.
    pub alternations: Vec<Vec<String>>,
}

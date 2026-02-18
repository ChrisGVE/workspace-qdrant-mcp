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
//! 3. **ROW_NUMBER()** — derives 1-based line numbers from gap-based `seq` ordering
//! 4. **file_metadata JOIN** — scopes results by project/branch/path
//!
//! ### Regex search (`search_regex`)
//! 1. **Literal extraction** — extract literal substrings (≥3 chars) from regex
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

use sqlx::Row;
use tracing::debug;

use crate::search_db::{SearchDbManager, SearchDbError};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

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
}

// ---------------------------------------------------------------------------
// FTS5 pattern escaping
// ---------------------------------------------------------------------------

/// Escape a search pattern for FTS5 trigram MATCH.
///
/// FTS5 trigram tokenizer requires patterns to be double-quote wrapped.
/// Internal double quotes are escaped as `""`.
///
/// Returns `None` if the pattern is shorter than 3 characters (trigram minimum).
pub fn escape_fts5_pattern(pattern: &str) -> Option<String> {
    if pattern.len() < 3 {
        return None;
    }
    let escaped = pattern.replace('"', "\"\"");
    Some(format!("\"{}\"", escaped))
}

/// Escape a LIKE pattern — escape `%`, `_`, and `\` for exact substring match.
pub fn escape_like_pattern(pattern: &str) -> String {
    pattern
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

// ---------------------------------------------------------------------------
// Path glob filtering (Task 55)
// ---------------------------------------------------------------------------

/// Extract a deterministic prefix from a glob pattern for SQL pre-filtering.
///
/// Returns the longest prefix before any glob metacharacter (`*`, `?`, `[`).
/// For example, `src/**/*.rs` → `Some("src/")`, `**/*.rs` → `None`.
fn extract_glob_prefix(glob: &str) -> Option<String> {
    let end = glob.find(|c: char| c == '*' || c == '?' || c == '[');
    match end {
        Some(0) | None if glob.contains('*') || glob.contains('?') || glob.contains('[') => None,
        Some(pos) => {
            let prefix = &glob[..pos];
            if prefix.is_empty() {
                None
            } else {
                Some(prefix.to_string())
            }
        }
        None => {
            // No metacharacters — treat as exact path match prefix
            Some(glob.to_string())
        }
    }
}

/// Expand brace expressions in a glob pattern.
///
/// Handles a single level of `{a,b,c}` expansion. For example:
/// `*.{rs,toml}` → `["*.rs", "*.toml"]`
///
/// If no braces are present, returns the original pattern as a single-element vec.
fn expand_braces(glob: &str) -> Vec<String> {
    if let Some(open) = glob.find('{') {
        if let Some(close) = glob[open..].find('}') {
            let close = open + close;
            let prefix = &glob[..open];
            let suffix = &glob[close + 1..];
            let alternatives = &glob[open + 1..close];

            return alternatives
                .split(',')
                .map(|alt| format!("{}{}{}", prefix, alt.trim(), suffix))
                .collect();
        }
    }
    vec![glob.to_string()]
}

/// Compile a glob pattern (with optional brace expansion) into a matcher.
///
/// Returns a closure that tests whether a file path matches the glob.
fn compile_glob_matcher(glob_pattern: &str) -> Result<Box<dyn Fn(&str) -> bool + Send + Sync>, SearchDbError> {
    let patterns = expand_braces(glob_pattern);
    let compiled: Vec<glob::Pattern> = patterns
        .iter()
        .map(|p| glob::Pattern::new(p))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| SearchDbError::InvalidPattern(format!("Invalid glob pattern: {}", e)))?;

    let opts = glob::MatchOptions {
        case_sensitive: true,
        require_literal_separator: false,
        require_literal_leading_dot: false,
    };

    Ok(Box::new(move |path: &str| {
        compiled.iter().any(|p| p.matches_with(path, opts))
    }))
}

// ---------------------------------------------------------------------------
// Search implementation
// ---------------------------------------------------------------------------

/// Resolve the effective path prefix for SQL pre-filtering.
///
/// If `path_glob` is set, extracts a prefix from the glob. Otherwise uses `path_prefix`.
/// The glob matcher (if any) is applied in Rust after SQL results are fetched.
fn resolve_path_filter(options: &SearchOptions) -> (Option<String>, SearchOptions) {
    if let Some(ref glob) = options.path_glob {
        let prefix = extract_glob_prefix(glob);
        let mut effective = options.clone();
        // Replace path_prefix with the extracted glob prefix for SQL pre-filtering
        effective.path_prefix = prefix;
        // Clear path_glob in effective options so query builder uses path_prefix
        effective.path_glob = None;
        (Some(glob.clone()), effective)
    } else {
        (None, options.clone())
    }
}

/// Search code_lines for an exact substring pattern.
///
/// Uses a two-phase approach:
/// 1. FTS5 trigram MATCH for fast candidate selection
/// 2. INSTR verification for exact substring match
///
/// For patterns shorter than 3 characters, falls back to INSTR-only scan.
/// When `path_glob` is set, applies glob filtering in Rust after SQL results.
pub async fn search_exact(
    search_db: &SearchDbManager,
    pattern: &str,
    options: &SearchOptions,
) -> Result<SearchResults, SearchDbError> {
    let start = std::time::Instant::now();

    if pattern.is_empty() {
        return Ok(SearchResults {
            pattern: pattern.to_string(),
            matches: vec![],
            truncated: false,
            query_time_ms: 0,
        });
    }

    // Resolve path_glob → SQL prefix + glob matcher
    let (glob_pattern, effective_options) = resolve_path_filter(options);
    let glob_matcher = glob_pattern
        .as_deref()
        .map(compile_glob_matcher)
        .transpose()?;

    let fts5_pattern = escape_fts5_pattern(pattern);

    // Build the SQL query dynamically based on options
    let (sql, use_fts) = build_search_query(&fts5_pattern, &effective_options);

    debug!(
        "FTS5 search: pattern={:?}, fts5={:?}, use_fts={}, tenant={:?}, branch={:?}, path_prefix={:?}, path_glob={:?}",
        pattern, fts5_pattern, use_fts,
        effective_options.tenant_id, effective_options.branch, effective_options.path_prefix,
        options.path_glob,
    );

    let pool = search_db.pool();
    let mut query = sqlx::query(&sql);

    // Bind parameters in order
    if use_fts {
        query = query.bind(fts5_pattern.as_ref().unwrap());
    }

    // INSTR pattern — case insensitive uses LOWER() in SQL so bind lowercase
    if options.case_insensitive {
        query = query.bind(pattern.to_lowercase());
    } else {
        query = query.bind(pattern);
    }

    // Bind optional scope filters
    if let Some(ref tid) = effective_options.tenant_id {
        query = query.bind(tid);
    }
    if let Some(ref branch) = effective_options.branch {
        query = query.bind(branch);
    }
    if let Some(ref prefix) = effective_options.path_prefix {
        query = query.bind(format!("{}%", prefix));
    }

    let rows = query.fetch_all(pool).await?;

    let max_results = if options.max_results > 0 {
        options.max_results
    } else {
        usize::MAX
    };

    // Apply glob filter and collect matches
    let mut matches = Vec::new();
    for row in &rows {
        let file_path: String = row.get("file_path");

        // Apply glob filter if set
        if let Some(ref matcher) = glob_matcher {
            if !matcher(&file_path) {
                continue;
            }
        }

        matches.push(SearchMatch {
            line_id: row.get("line_id"),
            file_id: row.get("file_id"),
            line_number: row.get("line_number"),
            content: row.get("content"),
            file_path,
            tenant_id: row.get("tenant_id"),
            branch: row.get("branch"),
        });

        if matches.len() > max_results {
            break;
        }
    }

    let truncated = matches.len() > max_results;
    if truncated {
        matches.truncate(max_results);
    }

    let query_time_ms = start.elapsed().as_millis() as u64;

    debug!(
        "FTS5 search complete: {} matches in {}ms (pattern={:?}, truncated={})",
        matches.len(), query_time_ms, pattern, truncated
    );

    Ok(SearchResults {
        pattern: pattern.to_string(),
        matches,
        truncated,
        query_time_ms,
    })
}

/// Build the search SQL query based on options.
///
/// Returns (sql_string, uses_fts5). When patterns are < 3 chars, FTS5
/// trigram index cannot be used and we fall back to LIKE-only scan.
///
/// ## Query structure
///
/// Uses a two-CTE approach to correctly derive absolute line numbers:
/// 1. `all_lines` — numbers ALL lines in each file using `ROW_NUMBER()`
/// 2. `matching` — filters to lines matching the search pattern (FTS5 + LIKE)
///
/// This ensures line numbers reflect the true position in the file, not the
/// position among matching lines.
fn build_search_query(
    fts5_pattern: &Option<String>,
    options: &SearchOptions,
) -> (String, bool) {
    let use_fts = fts5_pattern.is_some();

    // CTE 1: Number ALL lines in each file
    let all_lines_cte = r#"
WITH all_lines AS (
    SELECT
        cl.line_id,
        cl.file_id,
        ROW_NUMBER() OVER (PARTITION BY cl.file_id ORDER BY cl.seq) AS line_number,
        cl.content
    FROM code_lines cl
)"#;

    // CTE 2: Filter to matching lines using FTS5 + exact match verification
    //
    // Case-sensitive: use INSTR() for exact substring (SQLite LIKE is case-insensitive by default)
    // Case-insensitive: use INSTR(LOWER(), LOWER()) for case-folded comparison
    let matching_cte = if use_fts {
        if options.case_insensitive {
            r#",
matching AS (
    SELECT al.line_id, al.file_id, al.line_number, al.content
    FROM all_lines al
    JOIN code_lines_fts fts ON al.line_id = fts.rowid
    WHERE fts.content MATCH ?1 AND INSTR(LOWER(al.content), ?2) > 0
)"#
        } else {
            r#",
matching AS (
    SELECT al.line_id, al.file_id, al.line_number, al.content
    FROM all_lines al
    JOIN code_lines_fts fts ON al.line_id = fts.rowid
    WHERE fts.content MATCH ?1 AND INSTR(al.content, ?2) > 0
)"#
        }
    } else {
        // INSTR-only fallback for short patterns (< 3 chars)
        if options.case_insensitive {
            r#",
matching AS (
    SELECT al.line_id, al.file_id, al.line_number, al.content
    FROM all_lines al
    WHERE INSTR(LOWER(al.content), ?1) > 0
)"#
        } else {
            r#",
matching AS (
    SELECT al.line_id, al.file_id, al.line_number, al.content
    FROM all_lines al
    WHERE INSTR(al.content, ?1) > 0
)"#
        }
    };

    // Main query joins with file_metadata for scoping
    let mut sql = format!(
        "{}{}
SELECT m.line_id, m.file_id, m.line_number, m.content,
       fm.file_path, fm.tenant_id, fm.branch
FROM matching m
JOIN file_metadata fm ON m.file_id = fm.file_id
WHERE 1=1",
        all_lines_cte, matching_cte
    );

    // The bind index depends on whether FTS5 is used
    // FTS5 mode: ?1 = fts5_pattern, ?2 = like_pattern, ?3+ = scope filters
    // LIKE-only mode: ?1 = like_pattern, ?2+ = scope filters
    let mut next_param = if use_fts { 3 } else { 2 };

    if options.tenant_id.is_some() {
        sql.push_str(&format!(" AND fm.tenant_id = ?{}", next_param));
        next_param += 1;
    }
    if options.branch.is_some() {
        sql.push_str(&format!(" AND fm.branch = ?{}", next_param));
        next_param += 1;
    }
    if options.path_prefix.is_some() {
        sql.push_str(&format!(" AND fm.file_path LIKE ?{} ESCAPE '\\'", next_param));
        // next_param += 1; // not needed — last parameter
    }

    sql.push_str("\nORDER BY m.file_id, m.line_number");

    (sql, use_fts)
}

// ---------------------------------------------------------------------------
// Regex search with trigram acceleration (Task 54)
// ---------------------------------------------------------------------------

/// Extract literal substrings (≥3 characters) from a regex pattern.
///
/// Walks the regex string character by character, collecting runs of literal
/// characters. When a metacharacter is encountered, the current run is flushed.
/// Escaped literals (e.g., `\.`, `\\`) are treated as literal characters.
///
/// Returns a list of literal strings suitable for FTS5 trigram pre-filtering.
pub fn extract_literals_from_regex(pattern: &str) -> Vec<String> {
    let mut literals = Vec::new();
    let mut current = String::new();
    let mut chars = pattern.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '\\' => {
                // Escape sequence
                if let Some(&next) = chars.peek() {
                    match next {
                        // Metacharacter classes — end the current literal run
                        'd' | 'D' | 'w' | 'W' | 's' | 'S' | 'b' | 'B'
                        | 'A' | 'z' | 'Z' | 'G' => {
                            flush_literal(&mut current, &mut literals);
                            chars.next(); // consume the class character
                        }
                        // Escaped literals — add the literal character
                        _ => {
                            chars.next();
                            current.push(next);
                        }
                    }
                }
                // Trailing backslash — ignore
            }
            // Character class — skip everything until closing `]`
            '[' => {
                flush_literal(&mut current, &mut literals);
                // Skip contents of character class
                while let Some(inner) = chars.next() {
                    if inner == '\\' {
                        chars.next(); // skip escaped char inside class
                    } else if inner == ']' {
                        break;
                    }
                }
            }
            // Other metacharacters that end a literal run
            '.' | '*' | '+' | '?' | ']' | '(' | ')' | '{' | '}' | '|' | '^' | '$' => {
                flush_literal(&mut current, &mut literals);
            }
            // Literal character
            _ => {
                current.push(ch);
            }
        }
    }

    // Flush any remaining literal
    flush_literal(&mut current, &mut literals);

    literals
}

/// Flush the current literal buffer into the literals list if ≥ 3 chars.
fn flush_literal(current: &mut String, literals: &mut Vec<String>) {
    if current.len() >= 3 {
        literals.push(current.clone());
    }
    current.clear();
}

/// Build an FTS5 OR query from extracted literals.
///
/// Each literal is double-quote escaped for FTS5 trigram matching.
/// Returns `None` if no literals have ≥ 3 characters.
fn build_fts5_or_query(literals: &[String]) -> Option<String> {
    let terms: Vec<String> = literals
        .iter()
        .filter_map(|lit| escape_fts5_pattern(lit))
        .collect();

    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

/// Search code_lines using a regex pattern with trigram acceleration.
///
/// ## Strategy
///
/// 1. Extract literal substrings from the regex for FTS5 pre-filtering
/// 2. If literals found: query FTS5 with OR query, verify with regex in Rust
/// 3. If no literals: scan all code_lines (with scope filters), verify with regex
///
/// Case-insensitive mode uses `regex::RegexBuilder::case_insensitive(true)`.
/// When `path_glob` is set, applies glob filtering in Rust after SQL results.
pub async fn search_regex(
    search_db: &SearchDbManager,
    pattern: &str,
    options: &SearchOptions,
) -> Result<SearchResults, SearchDbError> {
    let start = std::time::Instant::now();

    if pattern.is_empty() {
        return Ok(SearchResults {
            pattern: pattern.to_string(),
            matches: vec![],
            truncated: false,
            query_time_ms: 0,
        });
    }

    // Resolve path_glob → SQL prefix + glob matcher
    let (glob_pattern, effective_options) = resolve_path_filter(options);
    let glob_matcher = glob_pattern
        .as_deref()
        .map(compile_glob_matcher)
        .transpose()?;

    // Compile the regex (case-insensitive if requested)
    let re = regex::RegexBuilder::new(pattern)
        .case_insensitive(options.case_insensitive)
        .build()
        .map_err(|e| SearchDbError::InvalidPattern(format!("{}", e)))?;

    // Extract literals for FTS5 acceleration
    let literals = extract_literals_from_regex(pattern);
    let fts5_query = build_fts5_or_query(&literals);

    debug!(
        "Regex search: pattern={:?}, literals={:?}, fts5_query={:?}, tenant={:?}, path_glob={:?}",
        pattern, literals, fts5_query, effective_options.tenant_id, options.path_glob,
    );

    // Build and execute SQL query
    let (sql, use_fts) = build_regex_search_query(&fts5_query, &effective_options);
    let pool = search_db.pool();
    let mut query = sqlx::query(&sql);

    // Bind parameters
    if use_fts {
        query = query.bind(fts5_query.as_ref().unwrap());
    }

    // Bind scope filters
    let mut _param_idx = if use_fts { 2 } else { 1 };
    if let Some(ref tid) = effective_options.tenant_id {
        query = query.bind(tid);
        _param_idx += 1;
    }
    if let Some(ref branch) = effective_options.branch {
        query = query.bind(branch);
        _param_idx += 1;
    }
    if let Some(ref prefix) = effective_options.path_prefix {
        query = query.bind(format!("{}%", prefix));
    }

    let rows = query.fetch_all(pool).await?;

    // Apply regex + glob filters in Rust
    let max_results = if options.max_results > 0 {
        options.max_results
    } else {
        usize::MAX
    };

    let mut matches = Vec::new();
    for row in &rows {
        let file_path: String = row.get("file_path");

        // Apply glob filter if set
        if let Some(ref matcher) = glob_matcher {
            if !matcher(&file_path) {
                continue;
            }
        }

        let content: String = row.get("content");
        if re.is_match(&content) {
            matches.push(SearchMatch {
                line_id: row.get("line_id"),
                file_id: row.get("file_id"),
                line_number: row.get("line_number"),
                content,
                file_path,
                tenant_id: row.get("tenant_id"),
                branch: row.get("branch"),
            });
            if matches.len() > max_results {
                break;
            }
        }
    }

    let truncated = matches.len() > max_results;
    if truncated {
        matches.truncate(max_results);
    }

    let query_time_ms = start.elapsed().as_millis() as u64;

    debug!(
        "Regex search complete: {} matches in {}ms (pattern={:?}, fts_candidates={}, truncated={})",
        matches.len(), query_time_ms, pattern, rows.len(), truncated
    );

    Ok(SearchResults {
        pattern: pattern.to_string(),
        matches,
        truncated,
        query_time_ms,
    })
}

/// Build SQL query for regex search.
///
/// When FTS5 literals are available, uses a two-CTE approach:
/// 1. `all_lines` — numbers ALL lines per file using ROW_NUMBER()
/// 2. `candidates` — pre-filters using FTS5 MATCH (OR query of literals)
///
/// When no FTS5 literals are available, scans all lines with scope filters only.
/// Regex matching is always done in Rust after fetching candidates.
fn build_regex_search_query(
    fts5_query: &Option<String>,
    options: &SearchOptions,
) -> (String, bool) {
    let use_fts = fts5_query.is_some();

    // CTE 1: Number ALL lines in each file
    let all_lines_cte = r#"
WITH all_lines AS (
    SELECT
        cl.line_id,
        cl.file_id,
        ROW_NUMBER() OVER (PARTITION BY cl.file_id ORDER BY cl.seq) AS line_number,
        cl.content
    FROM code_lines cl
)"#;

    // CTE 2: FTS5 candidate pre-filter or full scan
    let candidates_cte = if use_fts {
        r#",
candidates AS (
    SELECT al.line_id, al.file_id, al.line_number, al.content
    FROM all_lines al
    JOIN code_lines_fts fts ON al.line_id = fts.rowid
    WHERE fts.content MATCH ?1
)"#
    } else {
        // No FTS5 — return all lines (regex filtering in Rust)
        r#",
candidates AS (
    SELECT al.line_id, al.file_id, al.line_number, al.content
    FROM all_lines al
)"#
    };

    // Main query joins with file_metadata for scoping
    let mut sql = format!(
        "{}{}
SELECT c.line_id, c.file_id, c.line_number, c.content,
       fm.file_path, fm.tenant_id, fm.branch
FROM candidates c
JOIN file_metadata fm ON c.file_id = fm.file_id
WHERE 1=1",
        all_lines_cte, candidates_cte
    );

    // Bind indices: FTS5 mode starts scope filters at ?2, non-FTS at ?1
    let mut next_param = if use_fts { 2 } else { 1 };

    if options.tenant_id.is_some() {
        sql.push_str(&format!(" AND fm.tenant_id = ?{}", next_param));
        next_param += 1;
    }
    if options.branch.is_some() {
        sql.push_str(&format!(" AND fm.branch = ?{}", next_param));
        next_param += 1;
    }
    if options.path_prefix.is_some() {
        sql.push_str(&format!(" AND fm.file_path LIKE ?{} ESCAPE '\\'", next_param));
    }

    sql.push_str("\nORDER BY c.file_id, c.line_number");

    (sql, use_fts)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_lines_schema::{initial_seq, UPSERT_FILE_METADATA_SQL};

    async fn setup_search_db() -> (tempfile::TempDir, SearchDbManager) {
        let tmp = tempfile::tempdir().unwrap();
        let db_path = tmp.path().join("test_search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();
        (tmp, manager)
    }

    async fn insert_file_content(
        db: &SearchDbManager,
        file_id: i64,
        lines: &[&str],
        tenant_id: &str,
        branch: Option<&str>,
        file_path: &str,
    ) {
        let pool = db.pool();

        // Insert lines
        for (i, line) in lines.iter().enumerate() {
            let seq = initial_seq(i);
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (?1, ?2, ?3)")
                .bind(file_id)
                .bind(seq)
                .bind(*line)
                .execute(pool)
                .await
                .unwrap();
        }

        // Insert file_metadata
        sqlx::query(UPSERT_FILE_METADATA_SQL)
            .bind(file_id)
            .bind(tenant_id)
            .bind(branch)
            .bind(file_path)
            .execute(pool)
            .await
            .unwrap();

        // Rebuild FTS
        db.rebuild_fts().await.unwrap();
    }

    // ── Pattern escaping tests ──

    #[test]
    fn test_escape_fts5_pattern_basic() {
        assert_eq!(
            escape_fts5_pattern("println"),
            Some("\"println\"".to_string())
        );
    }

    #[test]
    fn test_escape_fts5_pattern_with_quotes() {
        assert_eq!(
            escape_fts5_pattern("say \"hello\""),
            Some("\"say \"\"hello\"\"\"".to_string())
        );
    }

    #[test]
    fn test_escape_fts5_pattern_short() {
        assert_eq!(escape_fts5_pattern("fn"), None);
        assert_eq!(escape_fts5_pattern("a"), None);
        assert_eq!(escape_fts5_pattern(""), None);
    }

    #[test]
    fn test_escape_fts5_pattern_exactly_3() {
        assert_eq!(
            escape_fts5_pattern("abc"),
            Some("\"abc\"".to_string())
        );
    }

    #[test]
    fn test_escape_like_pattern() {
        assert_eq!(escape_like_pattern("hello"), "hello");
        assert_eq!(escape_like_pattern("100%"), "100\\%");
        assert_eq!(escape_like_pattern("under_score"), "under\\_score");
        assert_eq!(escape_like_pattern("back\\slash"), "back\\\\slash");
    }

    // ── Search query building tests ──

    #[test]
    fn test_build_query_with_fts5() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions::default();
        let (sql, use_fts) = build_search_query(&pattern, &options);
        assert!(use_fts);
        assert!(sql.contains("MATCH ?1"));
        assert!(sql.contains("INSTR(al.content, ?2)"));
        assert!(sql.contains("all_lines"));
        assert!(sql.contains("matching"));
    }

    #[test]
    fn test_build_query_without_fts5() {
        let options = SearchOptions::default();
        let (sql, use_fts) = build_search_query(&None, &options);
        assert!(!use_fts);
        assert!(!sql.contains("MATCH"));
        assert!(sql.contains("INSTR(al.content, ?1)"));
    }

    #[test]
    fn test_build_query_with_tenant_filter() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions {
            tenant_id: Some("proj1".to_string()),
            ..Default::default()
        };
        let (sql, _) = build_search_query(&pattern, &options);
        assert!(sql.contains("fm.tenant_id = ?3"));
    }

    #[test]
    fn test_build_query_case_insensitive() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions {
            case_insensitive: true,
            ..Default::default()
        };
        let (sql, _) = build_search_query(&pattern, &options);
        assert!(sql.contains("INSTR(LOWER(al.content), ?2)"));
    }

    #[test]
    fn test_build_query_with_all_filters() {
        let pattern = Some("\"test\"".to_string());
        let options = SearchOptions {
            tenant_id: Some("proj1".to_string()),
            branch: Some("main".to_string()),
            path_prefix: Some("src/".to_string()),
            ..Default::default()
        };
        let (sql, _) = build_search_query(&pattern, &options);
        assert!(sql.contains("fm.tenant_id = ?3"));
        assert!(sql.contains("fm.branch = ?4"));
        assert!(sql.contains("fm.file_path LIKE ?5"));
    }

    // ── Integration tests (FTS5 with real SQLite) ──

    #[tokio::test]
    async fn test_search_exact_basic() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn main() {", "    println!(\"hello\");", "}"],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        let results = search_exact(&db, "println", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 2);
        assert_eq!(results.matches[0].file_path, "src/main.rs");
        assert!(results.matches[0].content.contains("println"));
    }

    #[tokio::test]
    async fn test_search_exact_multiple_matches() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "use std::io;",
                "fn read() { io::stdin() }",
                "fn write() { io::stdout() }",
                "fn other() {}",
            ],
            "proj1",
            Some("main"),
            "src/io.rs",
        )
        .await;

        let results = search_exact(&db, "io::", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 2);
        assert_eq!(results.matches[0].line_number, 2);
        assert_eq!(results.matches[1].line_number, 3);
    }

    #[tokio::test]
    async fn test_search_exact_no_match() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn main() {}", "let x = 42;"],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        let results = search_exact(&db, "nonexistent_function", &SearchOptions::default())
            .await
            .unwrap();

        assert!(results.matches.is_empty());
    }

    #[tokio::test]
    async fn test_search_exact_case_sensitive() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn Main() {}", "fn main() {}"],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        // Case-sensitive: only "Main" matches
        let results = search_exact(
            &db,
            "Main",
            &SearchOptions::default(),
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 1);
    }

    #[tokio::test]
    async fn test_search_exact_case_insensitive() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn Main() {}", "fn main() {}"],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        // Case-insensitive: both match
        let results = search_exact(
            &db,
            "main",
            &SearchOptions {
                case_insensitive: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 2);
    }

    #[tokio::test]
    async fn test_search_exact_scoped_by_tenant() {
        let (_tmp, db) = setup_search_db().await;

        // Two different projects
        insert_file_content(
            &db,
            1,
            &["fn hello() {}", "fn world() {}"],
            "proj1",
            Some("main"),
            "src/a.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["fn hello() {}", "fn goodbye() {}"],
            "proj2",
            Some("main"),
            "src/b.rs",
        )
        .await;

        // Search only in proj1
        let results = search_exact(
            &db,
            "hello",
            &SearchOptions {
                tenant_id: Some("proj1".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].tenant_id, "proj1");
    }

    #[tokio::test]
    async fn test_search_exact_scoped_by_path_prefix() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db,
            1,
            &["fn func_a() {}"],
            "proj1",
            Some("main"),
            "src/module/a.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["fn func_a() {}"],
            "proj1",
            Some("main"),
            "tests/test_a.rs",
        )
        .await;

        // Search only in src/
        let results = search_exact(
            &db,
            "func_a",
            &SearchOptions {
                path_prefix: Some("src/".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].file_path, "src/module/a.rs");
    }

    #[tokio::test]
    async fn test_search_exact_max_results() {
        let (_tmp, db) = setup_search_db().await;

        let lines: Vec<String> = (0..20)
            .map(|i| format!("let item_{} = process();", i))
            .collect();
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();

        insert_file_content(&db, 1, &line_refs, "proj1", Some("main"), "src/many.rs")
            .await;

        let results = search_exact(
            &db,
            "process",
            &SearchOptions {
                max_results: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 5);
        assert!(results.truncated);
    }

    #[tokio::test]
    async fn test_search_exact_empty_pattern() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn main() {}"],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        let results = search_exact(&db, "", &SearchOptions::default())
            .await
            .unwrap();

        assert!(results.matches.is_empty());
    }

    #[tokio::test]
    async fn test_search_exact_special_characters() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "let pct = 100%;",
                "let _under = true;",
                "let path = \"C:\\\\Windows\";",
            ],
            "proj1",
            Some("main"),
            "src/special.rs",
        )
        .await;

        // Search for literal %
        let results = search_exact(&db, "100%", &SearchOptions::default())
            .await
            .unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 1);
    }

    #[tokio::test]
    async fn test_search_exact_short_pattern_fallback() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn a() {}", "fn b() {}", "fn ab() {}"],
            "proj1",
            Some("main"),
            "src/short.rs",
        )
        .await;

        // "fn" is only 2 chars — falls back to LIKE-only scan
        let results = search_exact(&db, "fn", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 3);
    }

    #[tokio::test]
    async fn test_search_exact_across_files() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db,
            1,
            &["fn handler() {}", "  // process request"],
            "proj1",
            Some("main"),
            "src/api.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["fn worker() {}", "  // process job"],
            "proj1",
            Some("main"),
            "src/worker.rs",
        )
        .await;

        let results = search_exact(&db, "process", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 2);
        // Results should be ordered by file_id, then line_number
        assert_eq!(results.matches[0].file_path, "src/api.rs");
        assert_eq!(results.matches[1].file_path, "src/worker.rs");
    }

    #[tokio::test]
    async fn test_search_exact_line_numbers_correct() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "// line 1",
                "// line 2",
                "fn target() {}",
                "// line 4",
                "fn target_two() {}",
            ],
            "proj1",
            Some("main"),
            "src/lines.rs",
        )
        .await;

        let results = search_exact(&db, "target", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 2);
        assert_eq!(results.matches[0].line_number, 3);
        assert_eq!(results.matches[1].line_number, 5);
    }

    #[tokio::test]
    async fn test_search_exact_branch_filter() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db,
            1,
            &["fn feature() {}"],
            "proj1",
            Some("main"),
            "src/a.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["fn feature() {}"],
            "proj1",
            Some("dev"),
            "src/a.rs",
        )
        .await;

        let results = search_exact(
            &db,
            "feature",
            &SearchOptions {
                branch: Some("dev".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].branch, Some("dev".to_string()));
    }

    #[tokio::test]
    async fn test_search_exact_pattern_with_double_quotes() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "let msg = \"hello world\";",
                "let other = 42;",
            ],
            "proj1",
            Some("main"),
            "src/quotes.rs",
        )
        .await;

        // Search for a string containing double quotes
        let results = search_exact(&db, "\"hello world\"", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert!(results.matches[0].content.contains("\"hello world\""));
    }

    // ── Regex literal extraction tests ──

    #[test]
    fn test_extract_literals_basic() {
        let lits = extract_literals_from_regex("async.*fn");
        assert_eq!(lits, vec!["async"]);
        // "fn" is only 2 chars, not extracted
    }

    #[test]
    fn test_extract_literals_multiple() {
        let lits = extract_literals_from_regex("pub fn \\w+\\(\\)");
        // "pub fn " has 7 chars (including trailing space)
        assert_eq!(lits, vec!["pub fn "]);
    }

    #[test]
    fn test_extract_literals_escaped_chars() {
        // \. is a literal dot, \( is a literal paren
        let lits = extract_literals_from_regex("log\\.info\\(");
        assert_eq!(lits, vec!["log.info("]);
    }

    #[test]
    fn test_extract_literals_no_literals() {
        // All metacharacters, no extractable literals
        let lits = extract_literals_from_regex("^.$");
        assert!(lits.is_empty());

        let lits = extract_literals_from_regex("[a-z]+");
        assert!(lits.is_empty());

        let lits = extract_literals_from_regex("\\d+\\.\\d+");
        assert!(lits.is_empty()); // "." is only 1 char between \d groups
    }

    #[test]
    fn test_extract_literals_word_boundary() {
        // \b is a metacharacter class, ends the literal run
        let lits = extract_literals_from_regex("\\bclass\\b");
        assert_eq!(lits, vec!["class"]);
    }

    #[test]
    fn test_extract_literals_alternation() {
        let lits = extract_literals_from_regex("async|await");
        assert_eq!(lits, vec!["async", "await"]);
    }

    #[test]
    fn test_extract_literals_mixed() {
        let lits = extract_literals_from_regex("fn\\s+main\\(");
        // "fn" (2 chars, too short), "main(" (5 chars, good)
        assert_eq!(lits, vec!["main("]);
    }

    #[test]
    fn test_extract_literals_escaped_backslash() {
        let lits = extract_literals_from_regex("C:\\\\Windows\\\\system32");
        assert_eq!(lits, vec!["C:\\Windows\\system32"]);
    }

    // ── FTS5 OR query building tests ──

    #[test]
    fn test_build_fts5_or_query_basic() {
        let lits = vec!["async".to_string(), "await".to_string()];
        let query = build_fts5_or_query(&lits);
        assert_eq!(query, Some("\"async\" OR \"await\"".to_string()));
    }

    #[test]
    fn test_build_fts5_or_query_empty() {
        let lits: Vec<String> = vec![];
        assert_eq!(build_fts5_or_query(&lits), None);
    }

    #[test]
    fn test_build_fts5_or_query_short_filtered() {
        // "fn" is < 3 chars, filtered out by escape_fts5_pattern
        let lits = vec!["fn".to_string()];
        assert_eq!(build_fts5_or_query(&lits), None);
    }

    #[test]
    fn test_build_fts5_or_query_single() {
        let lits = vec!["println".to_string()];
        let query = build_fts5_or_query(&lits);
        assert_eq!(query, Some("\"println\"".to_string()));
    }

    // ── Regex search query building tests ──

    #[test]
    fn test_build_regex_query_with_fts() {
        let fts_query = Some("\"async\" OR \"await\"".to_string());
        let options = SearchOptions::default();
        let (sql, use_fts) = build_regex_search_query(&fts_query, &options);
        assert!(use_fts);
        assert!(sql.contains("MATCH ?1"));
        assert!(sql.contains("all_lines"));
        assert!(sql.contains("candidates"));
    }

    #[test]
    fn test_build_regex_query_without_fts() {
        let options = SearchOptions::default();
        let (sql, use_fts) = build_regex_search_query(&None, &options);
        assert!(!use_fts);
        assert!(!sql.contains("MATCH"));
        assert!(sql.contains("all_lines"));
        assert!(sql.contains("candidates"));
    }

    #[test]
    fn test_build_regex_query_with_scope_filters() {
        let fts_query = Some("\"test\"".to_string());
        let options = SearchOptions {
            tenant_id: Some("proj1".to_string()),
            branch: Some("main".to_string()),
            path_prefix: Some("src/".to_string()),
            ..Default::default()
        };
        let (sql, _) = build_regex_search_query(&fts_query, &options);
        assert!(sql.contains("fm.tenant_id = ?2"));
        assert!(sql.contains("fm.branch = ?3"));
        assert!(sql.contains("fm.file_path LIKE ?4"));
    }

    // ── Regex search integration tests ──

    #[tokio::test]
    async fn test_search_regex_basic() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "fn main() {",
                "    let x = 42;",
                "    println!(\"hello\");",
                "}",
            ],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        let results = search_regex(&db, "fn\\s+main", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 1);
        assert!(results.matches[0].content.contains("fn main"));
    }

    #[tokio::test]
    async fn test_search_regex_wildcard() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "async fn process_request() {}",
                "fn process_response() {}",
                "fn handle_error() {}",
            ],
            "proj1",
            Some("main"),
            "src/handler.rs",
        )
        .await;

        // Match any "process_" function
        let results = search_regex(&db, "fn process_\\w+", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 2);
        assert!(results.matches[0].content.contains("process_request"));
        assert!(results.matches[1].content.contains("process_response"));
    }

    #[tokio::test]
    async fn test_search_regex_no_trigrams_fallback() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["a", "ab", "abc", "abcd", "x"],
            "proj1",
            Some("main"),
            "src/short.rs",
        )
        .await;

        // Pattern "^.{3}$" has no extractable literals — full scan
        let results = search_regex(&db, "^.{3}$", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].content, "abc");
    }

    #[tokio::test]
    async fn test_search_regex_case_insensitive() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn Main() {}", "fn main() {}", "fn MAIN() {}"],
            "proj1",
            Some("main"),
            "src/case.rs",
        )
        .await;

        let results = search_regex(
            &db,
            "fn main",
            &SearchOptions {
                case_insensitive: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 3);
    }

    #[tokio::test]
    async fn test_search_regex_scoped_by_tenant() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db,
            1,
            &["pub fn hello() {}"],
            "proj1",
            Some("main"),
            "src/a.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["pub fn hello() {}"],
            "proj2",
            Some("main"),
            "src/b.rs",
        )
        .await;

        let results = search_regex(
            &db,
            "pub fn \\w+",
            &SearchOptions {
                tenant_id: Some("proj1".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].tenant_id, "proj1");
    }

    #[tokio::test]
    async fn test_search_regex_alternation() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "let future = async { 42 };",
                "let result = await!(future);",
                "let sync_val = 10;",
            ],
            "proj1",
            Some("main"),
            "src/async.rs",
        )
        .await;

        let results = search_regex(&db, "async|await", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 2);
    }

    #[tokio::test]
    async fn test_search_regex_max_results() {
        let (_tmp, db) = setup_search_db().await;

        let lines: Vec<String> = (0..20)
            .map(|i| format!("let item_{} = process();", i))
            .collect();
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();

        insert_file_content(&db, 1, &line_refs, "proj1", Some("main"), "src/many.rs")
            .await;

        let results = search_regex(
            &db,
            "item_\\d+",
            &SearchOptions {
                max_results: 5,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 5);
        assert!(results.truncated);
    }

    #[tokio::test]
    async fn test_search_regex_empty_pattern() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn main() {}"],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        let results = search_regex(&db, "", &SearchOptions::default())
            .await
            .unwrap();

        assert!(results.matches.is_empty());
    }

    #[tokio::test]
    async fn test_search_regex_invalid_pattern() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &["fn main() {}"],
            "proj1",
            Some("main"),
            "src/main.rs",
        )
        .await;

        let result = search_regex(&db, "[invalid", &SearchOptions::default()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_regex_line_numbers_correct() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "// comment",
                "use std::io;",
                "// comment",
                "fn read() -> io::Result<()> {",
                "// comment",
            ],
            "proj1",
            Some("main"),
            "src/io.rs",
        )
        .await;

        let results = search_regex(&db, "fn \\w+\\(\\)", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 4);
    }

    #[tokio::test]
    async fn test_search_regex_word_boundary() {
        let (_tmp, db) = setup_search_db().await;
        insert_file_content(
            &db,
            1,
            &[
                "class MyClass {}",
                "subclass OtherClass {}",
                "let classified = true;",
            ],
            "proj1",
            Some("main"),
            "src/class.rs",
        )
        .await;

        // \bclass\b matches only "class" as a whole word
        let results = search_regex(&db, "\\bclass\\b", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line_number, 1);
    }

    #[tokio::test]
    async fn test_search_regex_across_files() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db,
            1,
            &["pub struct Config {}", "impl Config {}"],
            "proj1",
            Some("main"),
            "src/config.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["pub struct Handler {}", "impl Handler {}"],
            "proj1",
            Some("main"),
            "src/handler.rs",
        )
        .await;

        let results = search_regex(&db, "pub struct \\w+", &SearchOptions::default())
            .await
            .unwrap();

        assert_eq!(results.matches.len(), 2);
        assert_eq!(results.matches[0].file_path, "src/config.rs");
        assert_eq!(results.matches[1].file_path, "src/handler.rs");
    }

    #[tokio::test]
    async fn test_search_regex_path_prefix_filter() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db,
            1,
            &["fn test_func() {}"],
            "proj1",
            Some("main"),
            "src/lib.rs",
        )
        .await;
        insert_file_content(
            &db,
            2,
            &["fn test_func() {}"],
            "proj1",
            Some("main"),
            "tests/test.rs",
        )
        .await;

        let results = search_regex(
            &db,
            "fn test_\\w+",
            &SearchOptions {
                path_prefix: Some("src/".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].file_path, "src/lib.rs");
    }

    // ── Glob utility tests (Task 55) ──

    #[test]
    fn test_extract_glob_prefix_with_prefix() {
        assert_eq!(extract_glob_prefix("src/**/*.rs"), Some("src/".to_string()));
        assert_eq!(extract_glob_prefix("src/rust/*.rs"), Some("src/rust/".to_string()));
    }

    #[test]
    fn test_extract_glob_prefix_no_prefix() {
        assert_eq!(extract_glob_prefix("**/*.rs"), None);
        assert_eq!(extract_glob_prefix("*.rs"), None);
        assert_eq!(extract_glob_prefix("?abc"), None);
    }

    #[test]
    fn test_extract_glob_prefix_no_metacharacters() {
        // No metacharacters — treat as exact prefix
        assert_eq!(extract_glob_prefix("src/main.rs"), Some("src/main.rs".to_string()));
    }

    #[test]
    fn test_expand_braces_basic() {
        let expanded = expand_braces("*.{rs,toml}");
        assert_eq!(expanded, vec!["*.rs", "*.toml"]);
    }

    #[test]
    fn test_expand_braces_three_alternatives() {
        let expanded = expand_braces("src/**/*.{rs,ts,js}");
        assert_eq!(expanded, vec!["src/**/*.rs", "src/**/*.ts", "src/**/*.js"]);
    }

    #[test]
    fn test_expand_braces_no_braces() {
        let expanded = expand_braces("**/*.rs");
        assert_eq!(expanded, vec!["**/*.rs"]);
    }

    #[test]
    fn test_compile_glob_matcher_star_star() {
        let matcher = compile_glob_matcher("**/*.rs").unwrap();
        assert!(matcher("src/main.rs"));
        assert!(matcher("src/deep/nested/lib.rs"));
        assert!(!matcher("src/main.ts"));
        assert!(matcher("lib.rs"));
    }

    #[test]
    fn test_compile_glob_matcher_with_prefix() {
        let matcher = compile_glob_matcher("src/**/*.rs").unwrap();
        assert!(matcher("src/main.rs"));
        assert!(matcher("src/deep/lib.rs"));
        assert!(!matcher("tests/test.rs"));
    }

    #[test]
    fn test_compile_glob_matcher_braces() {
        let matcher = compile_glob_matcher("**/*.{rs,toml}").unwrap();
        assert!(matcher("src/main.rs"));
        assert!(matcher("Cargo.toml"));
        assert!(!matcher("src/main.ts"));
    }

    #[test]
    fn test_compile_glob_matcher_invalid() {
        let result = compile_glob_matcher("[invalid");
        assert!(result.is_err());
    }

    // ── Glob search integration tests (Task 55) ──

    #[tokio::test]
    async fn test_search_exact_with_path_glob() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db, 1,
            &["fn hello() {}"],
            "proj1", Some("main"), "src/main.rs",
        ).await;
        insert_file_content(
            &db, 2,
            &["fn hello() {}"],
            "proj1", Some("main"), "src/utils.ts",
        ).await;
        insert_file_content(
            &db, 3,
            &["fn hello() {}"],
            "proj1", Some("main"), "tests/test_main.rs",
        ).await;

        // Glob: only .rs files under src/
        let results = search_exact(
            &db,
            "hello",
            &SearchOptions {
                path_glob: Some("src/**/*.rs".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].file_path, "src/main.rs");
    }

    #[tokio::test]
    async fn test_search_exact_with_glob_star_star() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db, 1,
            &["fn target() {}"],
            "proj1", Some("main"), "src/lib.rs",
        ).await;
        insert_file_content(
            &db, 2,
            &["fn target() {}"],
            "proj1", Some("main"), "src/deep/nested/mod.rs",
        ).await;
        insert_file_content(
            &db, 3,
            &["fn target() {}"],
            "proj1", Some("main"), "docs/guide.md",
        ).await;

        // Glob: any .rs file anywhere
        let results = search_exact(
            &db,
            "target",
            &SearchOptions {
                path_glob: Some("**/*.rs".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 2);
    }

    #[tokio::test]
    async fn test_search_exact_with_glob_braces() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db, 1,
            &["fn target() {}"],
            "proj1", Some("main"), "src/main.rs",
        ).await;
        insert_file_content(
            &db, 2,
            &["fn target() {}"],
            "proj1", Some("main"), "Cargo.toml",
        ).await;
        insert_file_content(
            &db, 3,
            &["fn target() {}"],
            "proj1", Some("main"), "src/script.js",
        ).await;

        // Glob: .rs or .toml files
        let results = search_exact(
            &db,
            "target",
            &SearchOptions {
                path_glob: Some("**/*.{rs,toml}".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 2);
    }

    #[tokio::test]
    async fn test_search_exact_glob_overrides_path_prefix() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db, 1,
            &["fn target() {}"],
            "proj1", Some("main"), "src/main.rs",
        ).await;
        insert_file_content(
            &db, 2,
            &["fn target() {}"],
            "proj1", Some("main"), "tests/test.rs",
        ).await;

        // path_glob set → path_prefix should be ignored
        let results = search_exact(
            &db,
            "target",
            &SearchOptions {
                path_prefix: Some("tests/".to_string()),
                path_glob: Some("src/**/*.rs".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        // path_glob wins: only src/*.rs
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].file_path, "src/main.rs");
    }

    #[tokio::test]
    async fn test_search_regex_with_path_glob() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db, 1,
            &["pub fn handler() {}"],
            "proj1", Some("main"), "src/api.rs",
        ).await;
        insert_file_content(
            &db, 2,
            &["pub fn handler() {}"],
            "proj1", Some("main"), "src/api.ts",
        ).await;
        insert_file_content(
            &db, 3,
            &["pub fn handler() {}"],
            "proj1", Some("main"), "tests/test_api.rs",
        ).await;

        let results = search_regex(
            &db,
            "pub fn \\w+",
            &SearchOptions {
                path_glob: Some("src/**/*.rs".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].file_path, "src/api.rs");
    }

    #[tokio::test]
    async fn test_search_exact_glob_no_matches() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db, 1,
            &["fn hello() {}"],
            "proj1", Some("main"), "src/main.rs",
        ).await;

        // Glob matches no files
        let results = search_exact(
            &db,
            "hello",
            &SearchOptions {
                path_glob: Some("**/*.py".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert!(results.matches.is_empty());
    }

    #[tokio::test]
    async fn test_search_exact_glob_with_tenant() {
        let (_tmp, db) = setup_search_db().await;

        insert_file_content(
            &db, 1,
            &["fn shared() {}"],
            "proj1", Some("main"), "src/lib.rs",
        ).await;
        insert_file_content(
            &db, 2,
            &["fn shared() {}"],
            "proj2", Some("main"), "src/lib.rs",
        ).await;

        // Glob + tenant scoping
        let results = search_exact(
            &db,
            "shared",
            &SearchOptions {
                tenant_id: Some("proj1".to_string()),
                path_glob: Some("**/*.rs".to_string()),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].tenant_id, "proj1");
    }
}

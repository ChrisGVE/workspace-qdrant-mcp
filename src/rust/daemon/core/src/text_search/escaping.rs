//! FTS5 pattern escaping, LIKE pattern escaping, and glob handling.

use crate::search_db::SearchDbError;
use super::types::SearchOptions;

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
/// For example, `src/**/*.rs` -> `Some("src/")`, `**/*.rs` -> `None`.
pub(crate) fn extract_glob_prefix(glob: &str) -> Option<String> {
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
/// `*.{rs,toml}` -> `["*.rs", "*.toml"]`
///
/// If no braces are present, returns the original pattern as a single-element vec.
pub(crate) fn expand_braces(glob: &str) -> Vec<String> {
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
pub(crate) fn compile_glob_matcher(glob_pattern: &str) -> Result<Box<dyn Fn(&str) -> bool + Send + Sync>, SearchDbError> {
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

/// Resolve the effective path prefix for SQL pre-filtering.
///
/// If `path_glob` is set, extracts a prefix from the glob. Otherwise uses `path_prefix`.
/// The glob matcher (if any) is applied in Rust after SQL results are fetched.
pub(crate) fn resolve_path_filter(options: &SearchOptions) -> (Option<String>, SearchOptions) {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}

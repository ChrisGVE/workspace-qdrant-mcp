//! Repo-relative path normalization and glob matching for the quality eval.
//!
//! Located at: `src/rust/cli/src/commands/benchmark/quality/path_match.rs`
//!
//! An `expectedFile` in the gold dataset is either a literal repo-relative path
//! (`src/rust/client/src/search/flow.rs`) or a glob
//! (`**/proto/*.proto`, `.../exact_search/*.rs`). A search result's path is a hit
//! when it equals a literal expectation or matches an expectation glob. Both the
//! result paths and the expectations are first normalized to a canonical
//! forward-slash, prefix-stripped form so the comparison is host-independent.
//!
//! The glob grammar mirrors the deleted TS harness (`globToRegExp` in
//! `semantic-search.ts`): `**/` matches any number of leading path segments,
//! `**` matches across separators, `*` matches within a single segment, and `?`
//! matches a single non-separator character. Everything else is literal.
//!
//! Neighbors: `metrics.rs` (consumes `expected_matcher` to score hits),
//! `dataset.rs` (the gold dataset whose `expected_files` these match against).

/// Normalize a path to canonical comparison form: forward slashes, no leading
/// `./`, no surrounding slashes, with an optional workspace-root prefix removed.
///
/// Mirrors `normalizeBenchmarkPath` in the TS harness. The workspace-root strip
/// is a plain string operation (not `std::path`) so behavior is identical on
/// every host regardless of separator conventions.
pub fn normalize_path(input: &str, workspace_root: &str) -> String {
    let candidate = strip_ends(&to_posix(input.trim()));
    let root = strip_ends(&to_posix(workspace_root.trim()));

    let stripped = if root.is_empty() {
        candidate.clone()
    } else if candidate == root {
        String::new()
    } else if let Some(rest) = candidate.strip_prefix(&format!("{root}/")) {
        rest.to_string()
    } else {
        candidate
    };

    // Defensive: drop any residual leading slashes / `./` the strip left behind.
    stripped
        .trim_start_matches('/')
        .trim_start_matches("./")
        .to_string()
}

/// Convert backslashes to forward slashes.
fn to_posix(path: &str) -> String {
    path.replace('\\', "/")
}

/// Strip a leading `./` and any trailing slashes.
fn strip_ends(path: &str) -> String {
    path.trim_start_matches("./")
        .trim_end_matches('/')
        .to_string()
}

/// A compiled expectation: either an exact path or a glob pattern.
///
/// Built once per expected file so a query with many results does not recompile
/// the same pattern for every candidate path.
#[derive(Debug, Clone)]
pub enum ExpectedMatcher {
    /// Match only this exact normalized path.
    Exact(String),
    /// Match any path accepted by this glob (stored as its source pattern).
    Glob(GlobPattern),
}

impl ExpectedMatcher {
    /// Build a matcher from a normalized expectation string. A pattern that
    /// contains any glob metacharacter (`* ? [ {`) becomes a [`GlobPattern`];
    /// everything else is an exact match. Mirrors TS `expectedMatcher`.
    pub fn new(expected: &str) -> Self {
        if expected.contains(['*', '?', '[', '{']) {
            ExpectedMatcher::Glob(GlobPattern::new(expected))
        } else {
            ExpectedMatcher::Exact(expected.to_string())
        }
    }

    /// Does the (already normalized) candidate path satisfy this expectation?
    pub fn matches(&self, candidate: &str) -> bool {
        match self {
            ExpectedMatcher::Exact(expected) => candidate == expected,
            ExpectedMatcher::Glob(glob) => glob.matches(candidate),
        }
    }
}

/// A glob compiled to an anchored regex-free state machine.
///
/// Rather than pull in a regex engine, the glob is translated to a small set of
/// segment-aware rules and matched directly. Translation grammar (TS parity):
/// - `**/` → any number of leading segments (including none)
/// - `**`  → any characters, including `/`
/// - `*`   → any run of non-`/` characters
/// - `?`   → exactly one non-`/` character
/// - other → literal
#[derive(Debug, Clone)]
pub struct GlobPattern {
    /// The regex-equivalent pattern, kept for readability/debug; matching is done
    /// by [`glob_match`] against the original source.
    source: String,
}

impl GlobPattern {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.to_string(),
        }
    }

    pub fn matches(&self, candidate: &str) -> bool {
        glob_match(self.source.as_bytes(), candidate.as_bytes())
    }
}

/// Recursive glob matcher over byte slices.
///
/// This is the classic backtracking glob algorithm (as used by shells and
/// described in Wikipedia, "glob (programming)"), extended with the `**`
/// cross-separator wildcard. `pattern` and `text` are both ASCII path bytes.
fn glob_match(pattern: &[u8], text: &[u8]) -> bool {
    // Empty pattern matches only empty text.
    if pattern.is_empty() {
        return text.is_empty();
    }

    match pattern[0] {
        b'*' => match_star(pattern, text),
        b'?' => {
            // `?` consumes exactly one non-separator byte.
            !text.is_empty() && text[0] != b'/' && glob_match(&pattern[1..], &text[1..])
        }
        b'[' => match_class(pattern, text),
        literal => !text.is_empty() && text[0] == literal && glob_match(&pattern[1..], &text[1..]),
    }
}

/// Handle a `*` or `**[/]` at the head of the pattern.
fn match_star(pattern: &[u8], text: &[u8]) -> bool {
    // `**/` — skip any number of whole leading segments (including none).
    if pattern.starts_with(b"**/") {
        let rest = &pattern[3..];
        // Zero segments: the `**/` matched nothing.
        if glob_match(rest, text) {
            return true;
        }
        // One-or-more segments: consume up to and including each `/`.
        for (idx, &byte) in text.iter().enumerate() {
            if byte == b'/' && glob_match(rest, &text[idx + 1..]) {
                return true;
            }
        }
        return false;
    }

    // `**` (not followed by `/`) — match any characters, including separators.
    if pattern.starts_with(b"**") {
        let rest = &pattern[2..];
        for split in 0..=text.len() {
            if glob_match(rest, &text[split..]) {
                return true;
            }
        }
        return false;
    }

    // Single `*` — match any run of non-separator characters.
    let rest = &pattern[1..];
    let mut split = 0;
    loop {
        if glob_match(rest, &text[split..]) {
            return true;
        }
        if split >= text.len() || text[split] == b'/' {
            return false;
        }
        split += 1;
    }
}

/// Handle a `[...]` character class at the head of the pattern.
///
/// Supports a literal set and a leading `!` or `^` negation, matching exactly one
/// non-separator byte. Unterminated classes (no closing `]`) are treated as a
/// literal `[`, mirroring shell behavior.
fn match_class(pattern: &[u8], text: &[u8]) -> bool {
    let Some(close) = pattern.iter().position(|&b| b == b']') else {
        // No closing bracket — treat `[` as a literal character.
        return !text.is_empty() && text[0] == b'[' && glob_match(&pattern[1..], &text[1..]);
    };
    if text.is_empty() || text[0] == b'/' {
        return false;
    }

    let mut class = &pattern[1..close];
    let negated = matches!(class.first(), Some(b'!') | Some(b'^'));
    if negated {
        class = &class[1..];
    }
    let in_class = class.contains(&text[0]);
    let accept = in_class != negated;
    accept && glob_match(&pattern[close + 1..], &text[1..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_leading_dot_slash_and_backslashes() {
        assert_eq!(normalize_path("./src/main.rs", ""), "src/main.rs");
        assert_eq!(normalize_path("src\\rust\\lib.rs", ""), "src/rust/lib.rs");
        assert_eq!(normalize_path("  src/a.rs  ", ""), "src/a.rs");
    }

    #[test]
    fn normalize_strips_workspace_root_prefix() {
        assert_eq!(
            normalize_path("/repo/src/a.rs", "/repo"),
            "src/a.rs",
            "root prefix is removed"
        );
        assert_eq!(normalize_path("/repo", "/repo"), "", "exact root → empty");
        assert_eq!(
            normalize_path("/other/a.rs", "/repo"),
            "other/a.rs",
            "non-matching root is left intact"
        );
    }

    #[test]
    fn exact_matcher_requires_full_equality() {
        let m = ExpectedMatcher::new("src/a.rs");
        assert!(m.matches("src/a.rs"));
        assert!(!m.matches("src/a.rs.bak"));
        assert!(!m.matches("other/a.rs"));
    }

    #[test]
    fn single_star_stays_within_a_segment() {
        let m = ExpectedMatcher::new("src/exact_search/*.rs");
        assert!(m.matches("src/exact_search/search.rs"));
        assert!(m.matches("src/exact_search/mod.rs"));
        assert!(
            !m.matches("src/exact_search/sub/search.rs"),
            "* must not cross a '/'"
        );
    }

    #[test]
    fn double_star_slash_skips_leading_segments() {
        let m = ExpectedMatcher::new("**/proto/workspace_daemon.proto");
        assert!(m.matches("proto/workspace_daemon.proto"), "zero segments");
        assert!(m.matches("src/rust/daemon/proto/workspace_daemon.proto"));
        assert!(!m.matches("src/rust/daemon/proto/other.proto"));
    }

    #[test]
    fn double_star_proto_glob_from_gold() {
        let m = ExpectedMatcher::new("**/proto/*.proto");
        assert!(m.matches("src/rust/daemon/proto/workspace_daemon.proto"));
        assert!(m.matches("proto/a.proto"));
        assert!(!m.matches("src/rust/daemon/proto/nested/a.proto"));
    }

    #[test]
    fn double_star_matches_across_separators() {
        let m = ExpectedMatcher::new("src/**/payload.rs");
        assert!(m.matches("src/a/b/c/payload.rs"));
        assert!(m.matches("src/payload.rs"));
    }

    #[test]
    fn question_mark_matches_single_non_separator() {
        let m = ExpectedMatcher::new("v4?.rs");
        assert!(m.matches("v47.rs"));
        assert!(!m.matches("v4.rs"), "? requires exactly one char");
        assert!(!m.matches("v4/.rs"), "? does not match '/'");
    }

    #[test]
    fn char_class_matches_and_negates() {
        assert!(ExpectedMatcher::new("v4[57].rs").matches("v47.rs"));
        assert!(!ExpectedMatcher::new("v4[57].rs").matches("v46.rs"));
        assert!(ExpectedMatcher::new("v4[!6].rs").matches("v47.rs"));
        assert!(!ExpectedMatcher::new("v4[!6].rs").matches("v46.rs"));
    }

    #[test]
    fn literal_classifies_as_exact_not_glob() {
        assert!(matches!(
            ExpectedMatcher::new("src/a.rs"),
            ExpectedMatcher::Exact(_)
        ));
        assert!(matches!(
            ExpectedMatcher::new("src/*.rs"),
            ExpectedMatcher::Glob(_)
        ));
    }
}

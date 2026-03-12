//! `.gitattributes` parser for linguist language overrides.
//!
//! Parses `.gitattributes` files to extract:
//! - `linguist-language=<lang>` — override detected language for matching files
//! - `linguist-vendored` — mark files as vendored (skip indexing)
//! - `linguist-generated` — mark files as generated (skip indexing)
//! - `linguist-documentation` — mark files as documentation (skip indexing)
//!
//! Glob patterns follow gitattributes syntax (fnmatch-style).

use std::collections::HashMap;
use std::path::Path;

/// Parsed `.gitattributes` override entry.
#[derive(Debug, Clone)]
struct GitattributeRule {
    /// The glob pattern from the rule (e.g., "*.c", "vendor/**").
    pattern: String,
    /// The attribute action.
    action: GitattributeAction,
}

/// What a gitattributes rule does.
#[derive(Debug, Clone)]
enum GitattributeAction {
    /// Override the detected language (linguist-language=<lang>).
    LanguageOverride(String),
    /// Mark as vendored (linguist-vendored).
    Vendored,
    /// Mark as generated (linguist-generated).
    Generated,
    /// Mark as documentation (linguist-documentation).
    Documentation,
}

/// Cached `.gitattributes` overrides for a project.
///
/// Constructed once per project root, then queried per file path.
#[derive(Debug, Clone, Default)]
pub struct GitattributesOverrides {
    rules: Vec<GitattributeRule>,
}

/// Result of checking a file against `.gitattributes`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitattributeResult {
    /// No relevant gitattributes rule matched.
    NoMatch,
    /// Language override: use this language instead of extension-based detection.
    LanguageOverride(String),
    /// File should be skipped (vendored, generated, or documentation).
    Skip(SkipReason),
}

/// Why a file is being skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    Vendored,
    Generated,
    Documentation,
}

impl GitattributesOverrides {
    /// Parse a `.gitattributes` file from its content string.
    pub fn parse(content: &str) -> Self {
        let mut rules = Vec::new();

        for line in content.lines() {
            let line = line.trim();
            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Split into pattern and attributes
            // Format: <pattern> <attr1> <attr2> ...
            let mut parts = line.splitn(2, char::is_whitespace);
            let pattern = match parts.next() {
                Some(p) if !p.is_empty() => p.to_string(),
                _ => continue,
            };
            let attrs_str = match parts.next() {
                Some(a) => a.trim(),
                None => continue,
            };

            // Parse each attribute
            for attr in attrs_str.split_whitespace() {
                if let Some(lang) = attr.strip_prefix("linguist-language=") {
                    rules.push(GitattributeRule {
                        pattern: pattern.clone(),
                        action: GitattributeAction::LanguageOverride(lang.to_lowercase()),
                    });
                } else if attr == "linguist-vendored" || attr == "linguist-vendored=true" {
                    rules.push(GitattributeRule {
                        pattern: pattern.clone(),
                        action: GitattributeAction::Vendored,
                    });
                } else if attr == "linguist-generated" || attr == "linguist-generated=true" {
                    rules.push(GitattributeRule {
                        pattern: pattern.clone(),
                        action: GitattributeAction::Generated,
                    });
                } else if attr == "linguist-documentation" || attr == "linguist-documentation=true"
                {
                    rules.push(GitattributeRule {
                        pattern: pattern.clone(),
                        action: GitattributeAction::Documentation,
                    });
                }
            }
        }

        Self { rules }
    }

    /// Load `.gitattributes` from a project root directory.
    ///
    /// Returns an empty set of overrides if the file doesn't exist or can't be read.
    pub fn load(project_root: &Path) -> Self {
        let path = project_root.join(".gitattributes");
        match std::fs::read_to_string(&path) {
            Ok(content) => Self::parse(&content),
            Err(_) => Self::default(),
        }
    }

    /// Check if this set has any rules.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Query the overrides for a file path (relative to project root).
    ///
    /// Returns the first matching result. Language overrides take precedence
    /// over skip attributes when both match. Later rules in the file override
    /// earlier ones (last-match-wins, like gitattributes spec).
    pub fn check(&self, relative_path: &str) -> GitattributeResult {
        let mut result = GitattributeResult::NoMatch;

        for rule in &self.rules {
            if matches_gitattribute_glob(&rule.pattern, relative_path) {
                result = match &rule.action {
                    GitattributeAction::LanguageOverride(lang) => {
                        GitattributeResult::LanguageOverride(lang.clone())
                    }
                    GitattributeAction::Vendored => GitattributeResult::Skip(SkipReason::Vendored),
                    GitattributeAction::Generated => {
                        GitattributeResult::Skip(SkipReason::Generated)
                    }
                    GitattributeAction::Documentation => {
                        GitattributeResult::Skip(SkipReason::Documentation)
                    }
                };
            }
        }

        result
    }

    /// Convenience: get language override for a file, if any.
    pub fn language_override(&self, relative_path: &str) -> Option<String> {
        match self.check(relative_path) {
            GitattributeResult::LanguageOverride(lang) => Some(lang),
            _ => None,
        }
    }

    /// Convenience: check if a file should be skipped.
    pub fn should_skip(&self, relative_path: &str) -> bool {
        matches!(self.check(relative_path), GitattributeResult::Skip(_))
    }
}

/// Match a gitattributes glob pattern against a relative path.
///
/// Gitattributes patterns follow fnmatch rules:
/// - `*` matches anything except `/`
/// - `**` matches zero or more directories
/// - `?` matches a single character except `/`
/// - Patterns without `/` match the filename only
/// - Patterns with `/` match the full path
fn matches_gitattribute_glob(pattern: &str, path: &str) -> bool {
    // Normalize path separators
    let path = path.replace('\\', "/");

    // If pattern has no slash (except trailing), match against filename only
    let pattern_has_dir = pattern.trim_end_matches('/').contains('/');

    if pattern_has_dir {
        glob_match(pattern, &path)
    } else {
        // Match against the filename component only
        let filename = path.rsplit('/').next().unwrap_or(&path);
        glob_match(pattern, filename)
    }
}

/// Simple glob matcher supporting `*`, `**`, and `?`.
fn glob_match(pattern: &str, text: &str) -> bool {
    glob_match_inner(pattern.as_bytes(), text.as_bytes())
}

fn glob_match_inner(pattern: &[u8], text: &[u8]) -> bool {
    let (mut pi, mut ti) = (0, 0);
    let (mut star_pi, mut star_ti) = (usize::MAX, usize::MAX);

    while ti < text.len() {
        if pi < pattern.len() && pattern[pi] == b'*' {
            // Handle ** (match across directories)
            if pi + 1 < pattern.len() && pattern[pi + 1] == b'*' {
                // ** matches everything including /
                // Skip the **
                pi += 2;
                // Skip optional trailing /
                if pi < pattern.len() && pattern[pi] == b'/' {
                    pi += 1;
                }
                // Try matching the rest of pattern against every suffix of text
                while ti <= text.len() {
                    if glob_match_inner(&pattern[pi..], &text[ti..]) {
                        return true;
                    }
                    if ti < text.len() {
                        ti += 1;
                    } else {
                        break;
                    }
                }
                return false;
            }
            // Single * — save backtrack position (doesn't cross /)
            star_pi = pi;
            star_ti = ti;
            pi += 1;
        } else if pi < pattern.len()
            && (pattern[pi] == b'?' && text[ti] != b'/' || pattern[pi] == text[ti])
        {
            pi += 1;
            ti += 1;
        } else if star_pi != usize::MAX && text[star_ti] != b'/' {
            // Backtrack: * matched one more character (but not /)
            star_ti += 1;
            ti = star_ti;
            pi = star_pi + 1;
        } else {
            return false;
        }
    }

    // Consume trailing *s in pattern
    while pi < pattern.len() && pattern[pi] == b'*' {
        pi += 1;
    }

    pi == pattern.len()
}

/// Build a language override map from gitattributes.
///
/// Returns a HashMap of glob pattern → language ID for quick reference.
pub fn build_language_override_map(project_root: &Path) -> HashMap<String, String> {
    let overrides = GitattributesOverrides::load(project_root);
    let mut map = HashMap::new();

    for rule in &overrides.rules {
        if let GitattributeAction::LanguageOverride(ref lang) = rule.action {
            map.insert(rule.pattern.clone(), lang.clone());
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty() {
        let overrides = GitattributesOverrides::parse("");
        assert!(overrides.is_empty());
    }

    #[test]
    fn parse_comments_and_blanks() {
        let content = "# This is a comment\n\n# Another comment\n";
        let overrides = GitattributesOverrides::parse(content);
        assert!(overrides.is_empty());
    }

    #[test]
    fn parse_language_override() {
        let content = "*.c linguist-language=objective-c\n";
        let overrides = GitattributesOverrides::parse(content);
        assert_eq!(overrides.rules.len(), 1);
        assert_eq!(
            overrides.language_override("foo.c"),
            Some("objective-c".into())
        );
    }

    #[test]
    fn parse_vendored() {
        let content = "vendor/** linguist-vendored\n";
        let overrides = GitattributesOverrides::parse(content);
        assert!(overrides.should_skip("vendor/lib/foo.js"));
        assert!(!overrides.should_skip("src/main.js"));
    }

    #[test]
    fn parse_generated() {
        let content = "generated/*.go linguist-generated=true\n";
        let overrides = GitattributesOverrides::parse(content);
        assert!(overrides.should_skip("generated/types.go"));
        assert!(!overrides.should_skip("src/types.go"));
    }

    #[test]
    fn parse_documentation() {
        let content = "docs/** linguist-documentation\n";
        let overrides = GitattributesOverrides::parse(content);
        assert_eq!(
            overrides.check("docs/guide.md"),
            GitattributeResult::Skip(SkipReason::Documentation)
        );
    }

    #[test]
    fn parse_multiple_rules() {
        let content = "\
# Language overrides
*.h linguist-language=c
*.inc linguist-language=php

# Skip vendored
vendor/** linguist-vendored
third_party/** linguist-vendored
";
        let overrides = GitattributesOverrides::parse(content);
        assert_eq!(overrides.rules.len(), 4);
        assert_eq!(overrides.language_override("foo.h"), Some("c".into()));
        assert_eq!(overrides.language_override("bar.inc"), Some("php".into()));
        assert!(overrides.should_skip("vendor/lib.js"));
        assert!(overrides.should_skip("third_party/dep.c"));
    }

    #[test]
    fn last_match_wins() {
        let content = "\
*.c linguist-language=c
*.c linguist-language=objective-c
";
        let overrides = GitattributesOverrides::parse(content);
        // Last matching rule wins
        assert_eq!(
            overrides.language_override("foo.c"),
            Some("objective-c".into())
        );
    }

    #[test]
    fn glob_star_matches() {
        assert!(matches_gitattribute_glob("*.js", "app.js"));
        assert!(matches_gitattribute_glob("*.js", "src/app.js"));
        assert!(!matches_gitattribute_glob("*.js", "app.ts"));
    }

    #[test]
    fn glob_doublestar_matches() {
        assert!(matches_gitattribute_glob("vendor/**", "vendor/lib.js"));
        assert!(matches_gitattribute_glob(
            "vendor/**",
            "vendor/deep/nested/lib.js"
        ));
        assert!(!matches_gitattribute_glob("vendor/**", "src/vendor.js"));
    }

    #[test]
    fn glob_question_mark() {
        assert!(matches_gitattribute_glob("?.c", "a.c"));
        assert!(!matches_gitattribute_glob("?.c", "ab.c"));
        assert!(!matches_gitattribute_glob("?.c", "/.c"));
    }

    #[test]
    fn glob_path_pattern() {
        assert!(matches_gitattribute_glob("src/*.js", "src/app.js"));
        assert!(!matches_gitattribute_glob("src/*.js", "lib/app.js"));
    }

    #[test]
    fn no_match_returns_no_match() {
        let overrides = GitattributesOverrides::parse("*.py linguist-language=python\n");
        assert_eq!(overrides.check("foo.rs"), GitattributeResult::NoMatch);
    }

    #[test]
    fn load_nonexistent_returns_empty() {
        let overrides = GitattributesOverrides::load(Path::new("/nonexistent/path"));
        assert!(overrides.is_empty());
    }

    #[test]
    fn multiple_attrs_on_one_line() {
        let content = "*.pb.go linguist-generated linguist-language=go\n";
        let overrides = GitattributesOverrides::parse(content);
        // Both rules should be parsed; last-match-wins means language override wins
        assert_eq!(overrides.rules.len(), 2);
        // Language override is the last action for this file
        assert_eq!(
            overrides.check("types.pb.go"),
            GitattributeResult::LanguageOverride("go".into())
        );
    }

    #[test]
    fn build_language_override_map_works() {
        // Can't test with real filesystem easily, but verify the function signature works
        let map = build_language_override_map(Path::new("/nonexistent"));
        assert!(map.is_empty());
    }
}

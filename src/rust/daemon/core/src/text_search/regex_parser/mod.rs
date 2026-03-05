//! Recursive descent regex literal extractor.
//!
//! Extracts literal substrings from regex patterns for FTS5 pre-filtering.
//! Tracks alternation groups separately from sequential mandatory literals.

mod parser;
mod query_builder;

use super::types::RegexLiterals;

pub(crate) use query_builder::build_fts5_query;

/// Extract literal substrings (>= 3 characters) from a regex pattern.
///
/// Walks the regex string character by character, collecting runs of literal
/// characters. Tracks alternation groups (`(a|b|c)`) separately from
/// sequential mandatory literals.
///
/// Returns a `RegexLiterals` with mandatory literals AND'd together and
/// alternation groups OR'd internally, AND'd with mandatories.
pub fn extract_literals_from_regex(pattern: &str) -> RegexLiterals {
    let mut result = RegexLiterals {
        mandatory: Vec::new(),
        alternations: Vec::new(),
    };
    parser::extract_literals_recursive(pattern, &mut result);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_literals_basic() {
        let lits = extract_literals_from_regex("async.*fn");
        assert_eq!(lits.mandatory, vec!["async"]);
        assert!(lits.alternations.is_empty());
    }

    #[test]
    fn test_extract_literals_multiple_mandatory() {
        let lits = extract_literals_from_regex("pub fn \\w+\\(\\)");
        assert_eq!(lits.mandatory, vec!["pub fn "]);
        assert!(lits.alternations.is_empty());
    }

    #[test]
    fn test_extract_literals_escaped_chars() {
        let lits = extract_literals_from_regex("log\\.info\\(");
        assert_eq!(lits.mandatory, vec!["log.info("]);
    }

    #[test]
    fn test_extract_literals_no_literals() {
        let lits = extract_literals_from_regex("^.$");
        assert!(lits.mandatory.is_empty());
        assert!(lits.alternations.is_empty());

        let lits = extract_literals_from_regex("[a-z]+");
        assert!(lits.mandatory.is_empty());

        let lits = extract_literals_from_regex("\\d+\\.\\d+");
        assert!(lits.mandatory.is_empty());
    }

    #[test]
    fn test_extract_literals_word_boundary() {
        let lits = extract_literals_from_regex("\\bclass\\b");
        assert_eq!(lits.mandatory, vec!["class"]);
    }

    #[test]
    fn test_extract_literals_top_level_alternation() {
        let lits = extract_literals_from_regex("async|await");
        assert!(lits.mandatory.is_empty());
        assert_eq!(lits.alternations.len(), 1);
        assert!(lits.alternations[0].contains(&"async".to_string()));
        assert!(lits.alternations[0].contains(&"await".to_string()));
    }

    #[test]
    fn test_extract_literals_parenthesized_alternation() {
        let lits = extract_literals_from_regex("impl \\w+ for \\w+");
        assert_eq!(lits.mandatory, vec!["impl ", " for "]);
        assert!(lits.alternations.is_empty());
    }

    #[test]
    fn test_extract_literals_group_alternation() {
        let lits = extract_literals_from_regex("use (std|tokio|serde)::\\w+");
        assert_eq!(lits.mandatory, vec!["use "]);
        assert_eq!(lits.alternations.len(), 1);
        assert_eq!(
            lits.alternations[0],
            vec!["use std::", "use tokio::", "use serde::"]
        );
    }

    #[test]
    fn test_extract_literals_pub_decls() {
        let lits = extract_literals_from_regex("pub (fn|struct|enum|trait|type) \\w+");
        assert_eq!(lits.mandatory, vec!["pub "]);
        assert_eq!(lits.alternations.len(), 1);
        assert!(lits.alternations[0].contains(&"pub struct ".to_string()));
        assert!(lits.alternations[0].contains(&"pub enum ".to_string()));
        assert!(lits.alternations[0].contains(&"pub trait ".to_string()));
        assert!(lits.alternations[0].contains(&"pub type ".to_string()));
        assert!(lits.alternations[0].contains(&"pub fn ".to_string()));
    }

    #[test]
    fn test_extract_literals_mixed() {
        let lits = extract_literals_from_regex("fn\\s+main\\(");
        assert_eq!(lits.mandatory, vec!["main("]);
    }

    #[test]
    fn test_extract_literals_escaped_backslash() {
        let lits = extract_literals_from_regex("C:\\\\Windows\\\\system32");
        assert_eq!(lits.mandatory, vec!["C:\\Windows\\system32"]);
    }

    #[test]
    fn test_build_fts5_query_mandatory_and() {
        let lits = RegexLiterals {
            mandatory: vec!["impl ".to_string(), " for ".to_string()],
            alternations: vec![],
        };
        let query = build_fts5_query(&lits);
        assert_eq!(query, Some("\"impl \" AND \" for \"".to_string()));
    }

    #[test]
    fn test_build_fts5_query_with_alternation() {
        let lits = RegexLiterals {
            mandatory: vec!["use ".to_string()],
            alternations: vec![vec![
                "std".to_string(),
                "tokio".to_string(),
                "serde".to_string(),
            ]],
        };
        let query = build_fts5_query(&lits);
        assert_eq!(
            query,
            Some("\"use \" AND (\"std\" OR \"tokio\" OR \"serde\")".to_string())
        );
    }

    #[test]
    fn test_build_fts5_query_alternation_only() {
        let lits = RegexLiterals {
            mandatory: vec![],
            alternations: vec![vec!["async".to_string(), "await".to_string()]],
        };
        let query = build_fts5_query(&lits);
        assert_eq!(query, Some("(\"async\" OR \"await\")".to_string()));
    }

    #[test]
    fn test_build_fts5_query_empty() {
        let lits = RegexLiterals {
            mandatory: vec![],
            alternations: vec![],
        };
        assert_eq!(build_fts5_query(&lits), None);
    }

    #[test]
    fn test_build_fts5_query_short_filtered() {
        let lits = RegexLiterals {
            mandatory: vec!["fn".to_string()],
            alternations: vec![],
        };
        assert_eq!(build_fts5_query(&lits), None);
    }

    #[test]
    fn test_build_fts5_query_single() {
        let lits = RegexLiterals {
            mandatory: vec!["println".to_string()],
            alternations: vec![],
        };
        let query = build_fts5_query(&lits);
        assert_eq!(query, Some("\"println\"".to_string()));
    }

    #[test]
    fn test_build_fts5_query_end_to_end_trait_impl() {
        let lits = extract_literals_from_regex("impl \\w+ for \\w+");
        let query = build_fts5_query(&lits);
        assert_eq!(query, Some("\"impl \" AND \" for \"".to_string()));
    }

    #[test]
    fn test_build_fts5_query_end_to_end_std_imports() {
        let lits = extract_literals_from_regex("use (std|tokio|serde)::\\w+");
        let query = build_fts5_query(&lits);
        assert_eq!(
            query,
            Some("(\"use std::\" OR \"use tokio::\" OR \"use serde::\")".to_string())
        );
    }

    #[test]
    fn test_build_fts5_query_end_to_end_pub_decls() {
        let lits = extract_literals_from_regex("pub (fn|struct|enum|trait|type) \\w+");
        let query = build_fts5_query(&lits);
        assert_eq!(
            query,
            Some(
                "(\"pub fn \" OR \"pub struct \" OR \"pub enum \" OR \"pub trait \" OR \"pub type \")".to_string()
            )
        );
    }

    #[test]
    fn test_build_fts5_query_end_to_end_method_chains() {
        let lits = extract_literals_from_regex("\\.(await|unwrap|expect)\\b");
        assert!(lits.mandatory.is_empty());
        assert_eq!(lits.alternations.len(), 1);
        assert!(lits.alternations[0].contains(&".await".to_string()));
        assert!(lits.alternations[0].contains(&".unwrap".to_string()));
        assert!(lits.alternations[0].contains(&".expect".to_string()));
        let query = build_fts5_query(&lits);
        assert_eq!(
            query,
            Some("(\".await\" OR \".unwrap\" OR \".expect\")".to_string())
        );
    }
}

//! Recursive descent regex literal extractor.
//!
//! Extracts literal substrings from regex patterns for FTS5 pre-filtering.
//! Tracks alternation groups separately from sequential mandatory literals.

use super::escaping::escape_fts5_pattern;
use super::types::RegexLiterals;

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
    extract_literals_recursive(pattern, &mut result);
    result
}

/// Core extraction logic, separated for recursion on group contents.
fn extract_literals_recursive(pattern: &str, result: &mut RegexLiterals) {
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
                            flush_to_mandatory(&mut current, &mut result.mandatory);
                            chars.next();
                        }
                        // Escaped literals — add the literal character
                        _ => {
                            chars.next();
                            current.push(next);
                        }
                    }
                }
            }
            // Character class — skip everything until closing `]`
            '[' => {
                flush_to_mandatory(&mut current, &mut result.mandatory);
                while let Some(inner) = chars.next() {
                    if inner == '\\' {
                        chars.next();
                    } else if inner == ']' {
                        break;
                    }
                }
            }
            // Group start — extract group content, check for alternation
            '(' => {
                let prefix = std::mem::take(&mut current);
                if prefix.len() >= 3 {
                    result.mandatory.push(prefix.clone());
                }
                let group_content = extract_group_content(&mut chars);
                let suffix = collect_literal_suffix(&mut chars);
                process_group_with_affixes(&prefix, &suffix, &group_content, result);
                if suffix.len() >= 3 {
                    result.mandatory.push(suffix);
                }
            }
            // Alternation at top level — treat remaining pattern as alternate branch.
            '|' => {
                flush_to_mandatory(&mut current, &mut result.mandatory);
                let rest: String = chars.collect();
                let mut left_lits = std::mem::take(&mut result.mandatory);
                let mut right_result = RegexLiterals {
                    mandatory: Vec::new(),
                    alternations: Vec::new(),
                };
                extract_literals_recursive(&rest, &mut right_result);
                let mut right_lits = right_result.mandatory;
                result.alternations.extend(right_result.alternations);
                if !left_lits.is_empty() || !right_lits.is_empty() {
                    let mut group = Vec::new();
                    if !left_lits.is_empty() {
                        group.append(&mut left_lits);
                    }
                    if !right_lits.is_empty() {
                        // For top-level alternation, each side becomes a branch
                        // We need to restructure: put all left as one alt, all right as another
                    }
                    group.append(&mut right_lits);
                    if !group.is_empty() {
                        result.alternations.push(group);
                    }
                }
                return; // rest already consumed
            }
            // Other metacharacters that end a literal run
            '.' | '*' | '+' | '?' | ']' | ')' | '{' | '}' | '^' | '$' => {
                flush_to_mandatory(&mut current, &mut result.mandatory);
            }
            // Literal character
            _ => {
                current.push(ch);
            }
        }
    }

    flush_to_mandatory(&mut current, &mut result.mandatory);
}

/// Extract the content of a parenthesized group, handling nested parens.
fn extract_group_content(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut content = String::new();
    let mut depth = 1;
    while let Some(ch) = chars.next() {
        match ch {
            '(' => {
                depth += 1;
                content.push(ch);
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
                content.push(ch);
            }
            '\\' => {
                content.push(ch);
                if let Some(next) = chars.next() {
                    content.push(next);
                }
            }
            _ => content.push(ch),
        }
    }
    content
}

/// Collect literal characters immediately following a group close `)`.
fn collect_literal_suffix(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    let mut suffix = String::new();
    while let Some(&ch) = chars.peek() {
        match ch {
            '\\' => {
                let mut lookahead = chars.clone();
                lookahead.next();
                if let Some(&next) = lookahead.peek() {
                    match next {
                        'd' | 'D' | 'w' | 'W' | 's' | 'S' | 'b' | 'B'
                        | 'A' | 'z' | 'Z' | 'G' => break,
                        _ => {
                            chars.next();
                            chars.next();
                            suffix.push(next);
                        }
                    }
                } else {
                    break;
                }
            }
            '.' | '*' | '+' | '?' | '[' | ']' | '(' | ')' | '{' | '}' | '|' | '^' | '$' => {
                break;
            }
            _ => {
                suffix.push(ch);
                chars.next();
            }
        }
    }
    suffix
}

/// Process a group's content with optional prefix/suffix affixes.
fn process_group_with_affixes(
    prefix: &str,
    suffix: &str,
    content: &str,
    result: &mut RegexLiterals,
) {
    let branches = split_alternation(content);
    if branches.len() <= 1 {
        extract_literals_recursive(content, result);
    } else {
        let mut alt_group: Vec<String> = Vec::new();
        for branch in &branches {
            let mut branch_result = RegexLiterals {
                mandatory: Vec::new(),
                alternations: Vec::new(),
            };
            extract_literals_recursive(branch, &mut branch_result);
            if branch_result.mandatory.is_empty() {
                let combined = format!("{}{}{}", prefix, branch, suffix);
                if combined.len() >= 3 && is_all_literal(branch) {
                    alt_group.push(combined);
                }
            } else {
                for lit in &branch_result.mandatory {
                    let combined = format!("{}{}{}", prefix, lit, suffix);
                    if combined.len() >= 3 {
                        alt_group.push(combined);
                    } else if lit.len() >= 3 {
                        alt_group.push(lit.clone());
                    }
                }
            }
            result.alternations.extend(branch_result.alternations);
        }
        if !alt_group.is_empty() {
            result.alternations.push(alt_group);
        }
    }
}

/// Split a group's content by top-level `|` (respecting nested parens).
fn split_alternation(content: &str) -> Vec<String> {
    let mut branches = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    let mut chars = content.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '(' => {
                depth += 1;
                current.push(ch);
            }
            ')' => {
                depth -= 1;
                current.push(ch);
            }
            '\\' => {
                current.push(ch);
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            '|' if depth == 0 => {
                branches.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    branches.push(current);
    branches
}

/// Check if a string contains only literal characters (no regex metacharacters).
fn is_all_literal(s: &str) -> bool {
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => {
                if let Some(next) = chars.next() {
                    match next {
                        'd' | 'D' | 'w' | 'W' | 's' | 'S' | 'b' | 'B'
                        | 'A' | 'z' | 'Z' | 'G' => return false,
                        _ => {}
                    }
                }
            }
            '.' | '*' | '+' | '?' | '[' | ']' | '(' | ')' | '{' | '}' | '|' | '^' | '$' => {
                return false;
            }
            _ => {}
        }
    }
    true
}

/// Flush the current literal buffer into the mandatory list if >= 3 chars.
fn flush_to_mandatory(current: &mut String, mandatory: &mut Vec<String>) {
    if current.len() >= 3 {
        mandatory.push(current.clone());
    }
    current.clear();
}

/// Build an FTS5 query from structured regex literals.
///
/// Mandatory literals are AND'd together. Alternation groups are OR'd
/// internally and AND'd with the mandatory terms.
///
/// Returns `None` if no usable literals were extracted.
pub(crate) fn build_fts5_query(literals: &RegexLiterals) -> Option<String> {
    let mut alt_clauses: Vec<(String, Vec<String>)> = Vec::new();
    for group in &literals.alternations {
        let raw_terms: Vec<String> = group.clone();
        let terms: Vec<String> = group
            .iter()
            .filter_map(|lit| escape_fts5_pattern(lit))
            .collect();
        if terms.len() == 1 {
            alt_clauses.push((terms.into_iter().next().unwrap(), raw_terms));
        } else if terms.len() > 1 {
            alt_clauses.push((format!("({})", terms.join(" OR ")), raw_terms));
        }
    }

    let mut clauses: Vec<String> = Vec::new();

    'mandatory: for lit in &literals.mandatory {
        for (_, raw_terms) in &alt_clauses {
            if raw_terms.len() >= 2 && raw_terms.iter().all(|t| t.starts_with(lit.as_str())) {
                continue 'mandatory;
            }
        }
        if let Some(escaped) = escape_fts5_pattern(lit) {
            clauses.push(escaped);
        }
    }

    for (clause, _) in alt_clauses {
        clauses.push(clause);
    }

    if clauses.is_empty() {
        None
    } else {
        Some(clauses.join(" AND "))
    }
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
            alternations: vec![vec!["std".to_string(), "tokio".to_string(), "serde".to_string()]],
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
            Some(
                "(\"use std::\" OR \"use tokio::\" OR \"use serde::\")".to_string()
            )
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

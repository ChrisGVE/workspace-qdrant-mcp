//! FTS5 query construction from structured regex literals.

use super::super::escaping::escape_fts5_pattern;
use super::super::types::RegexLiterals;

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

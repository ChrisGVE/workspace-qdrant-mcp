//! Identifier normalization utilities.

use super::types::LspCandidateConfig;

/// Normalize a symbol identifier into a readable phrase.
///
/// Splits camelCase and snake_case into separate words.
/// Examples:
/// - `PrimeSieve` → `prime sieve`
/// - `find_n_primes` → `find n primes`
/// - `HTMLParser` → `html parser`
/// - `getHTTPResponse` → `get http response`
pub fn normalize_identifier(ident: &str) -> String {
    let mut words = Vec::new();
    let mut current = String::new();

    // Handle snake_case first: split on underscores
    for part in ident.split('_') {
        if part.is_empty() {
            continue;
        }

        // Split camelCase within each part
        let chars: Vec<char> = part.chars().collect();
        for (i, &ch) in chars.iter().enumerate() {
            if i > 0 && ch.is_uppercase() {
                // Check for acronym: consecutive uppercase (e.g., HTTP)
                let prev_upper = chars[i - 1].is_uppercase();
                let next_lower = i + 1 < chars.len() && chars[i + 1].is_lowercase();

                if !prev_upper || next_lower {
                    // Start of new word
                    if !current.is_empty() {
                        words.push(current.to_lowercase());
                        current.clear();
                    }
                }
            }
            current.push(ch);
        }
        if !current.is_empty() {
            words.push(current.to_lowercase());
            current.clear();
        }
    }

    words.join(" ")
}

/// Strip known trivial suffixes from an identifier.
pub fn strip_suffix(ident: &str, config: &LspCandidateConfig) -> String {
    let mut result = ident.to_string();
    for suffix in &config.strip_suffixes {
        if result.ends_with(suffix.as_str()) && result.len() > suffix.len() {
            result.truncate(result.len() - suffix.len());
            break;
        }
    }
    result
}

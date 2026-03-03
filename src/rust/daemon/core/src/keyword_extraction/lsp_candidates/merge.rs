//! Merging LSP candidates with lexical candidates.

use crate::keyword_extraction::lexical_candidates::LexicalCandidate;

use super::types::LspCandidate;

/// Merge LSP candidates with lexical candidates.
///
/// LSP candidates get a priority boost. Duplicates (by normalized phrase)
/// are deduplicated, keeping the higher-scored version.
pub fn merge_candidates(
    lexical: Vec<LexicalCandidate>,
    lsp: &[LspCandidate],
    boost: f64,
) -> Vec<LexicalCandidate> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut merged = Vec::new();

    // Add LSP candidates first (higher priority)
    for c in lsp {
        let key = c.phrase.to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        merged.push(LexicalCandidate {
            phrase: c.phrase.clone(),
            raw_tf: 1,
            tf_score: boost,
            ngram_size: c.phrase.split(' ').count() as u8,
        });
    }

    // Add lexical candidates that aren't already present
    for c in lexical {
        let key = c.phrase.to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);
        merged.push(c);
    }

    merged
}

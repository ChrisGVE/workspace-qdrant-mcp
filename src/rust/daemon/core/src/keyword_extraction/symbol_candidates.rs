//! Symbol-derived candidate extraction from tree-sitter chunk metadata.
//!
//! Reuses the tree-sitter semantic chunk pipeline's symbol metadata
//! (`chunk_type`, `symbol_name`, `parent_symbol`) as a high-quality concept
//! signal for keyword/tag extraction. Public type and function names —
//! `AuthService`, `RetryPolicy`, `HttpClient` — are concept-level signals,
//! whereas TF-IDF on raw text yields term-level signals (`tokio`, `serde`,
//! `reqwest`) that still need normalization.
//!
//! See [docs/specs/21-tree-sitter-roadmap.md] item #2.

use std::collections::{HashMap, HashSet};

use super::lsp_candidates::{
    normalize_identifier, CandidateSource, LspCandidate, LspCandidateConfig,
};

/// Chunk types worth emitting as candidates.
///
/// Top-level structural symbols carry the most concept signal. Methods are
/// excluded because their names tend to be mundane (`new`, `from`, `into`,
/// `process`, `handle`) and would crowd the candidate set.
fn is_concept_bearing_chunk_type(chunk_type: &str) -> bool {
    matches!(
        chunk_type,
        "class"
            | "struct"
            | "trait"
            | "interface"
            | "enum"
            | "function"
            | "async_function"
            | "type_alias"
            | "macro"
            | "module"
    )
}

/// Strip a known suffix only when the residual is meaningful.
///
/// Mirrors `lsp_candidates::strip_suffix` but kept inline here to avoid
/// re-exporting an internal helper. Same suffix list applies.
fn strip_concept_suffix(ident: &str, config: &LspCandidateConfig) -> String {
    let mut result = ident.to_string();
    for suffix in &config.strip_suffixes {
        if result.ends_with(suffix.as_str()) && result.len() > suffix.len() {
            result.truncate(result.len() - suffix.len());
            break;
        }
    }
    result
}

/// Extract candidate concept tags from semantic chunk metadata.
///
/// Iterates the per-chunk metadata HashMaps populated by
/// [`crate::document_processor::chunking::convert_semantic_chunks_to_text_chunks`]
/// and emits one candidate per unique concept-bearing top-level symbol.
///
/// Filtering rules:
/// 1. Skip chunks without `chunk_type` or `symbol_name`.
/// 2. Skip chunks whose `chunk_type` is not concept-bearing (method, preamble,
///    impl block, constant, text fallback).
/// 3. Skip chunks with a `parent_symbol` — those are methods misclassified
///    upstream, or nested definitions that should attribute to their parent.
/// 4. Skip identifiers shorter than `config.min_identifier_len`.
/// 5. Dedupe by normalized phrase (fragments of the same symbol collapse).
///
/// Each surviving symbol yields an `LspCandidate` with
/// `source = CandidateSource::PublicSymbol` and the config's `priority_boost`.
pub fn extract_symbol_candidates(
    chunk_metadata: &[HashMap<String, String>],
    config: &LspCandidateConfig,
) -> Vec<LspCandidate> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut candidates = Vec::new();

    for metadata in chunk_metadata {
        let Some(chunk_type) = metadata.get("chunk_type") else {
            continue;
        };
        if !is_concept_bearing_chunk_type(chunk_type) {
            continue;
        }

        // Methods carry parent_symbol — drop them: noisy method names
        // crowd the candidate set without adding concept signal.
        if metadata.contains_key("parent_symbol") {
            continue;
        }

        let Some(symbol_name) = metadata.get("symbol_name") else {
            continue;
        };

        // Sentinel symbol names from the chunker for non-symbol chunks.
        if symbol_name == "_preamble" || symbol_name == "_text" || symbol_name.is_empty() {
            continue;
        }

        if symbol_name.len() < config.min_identifier_len {
            continue;
        }

        // Drop trivial suffixes (Service, Manager, Handler, ...) so
        // `AuthService` becomes `Auth`, then normalize for the phrase.
        let stripped = strip_concept_suffix(symbol_name, config);
        let phrase = normalize_identifier(&stripped);

        if phrase.is_empty() || phrase.len() < config.min_identifier_len {
            continue;
        }

        if !seen.insert(phrase.clone()) {
            continue;
        }

        candidates.push(LspCandidate {
            phrase,
            identifier: symbol_name.clone(),
            source: CandidateSource::PublicSymbol,
            priority_boost: config.priority_boost,
        });
    }

    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_extract_struct_symbol() {
        let chunks = vec![meta(&[
            ("chunk_type", "struct"),
            ("symbol_name", "AuthService"),
        ])];
        let candidates = extract_symbol_candidates(&chunks, &LspCandidateConfig::default());

        assert_eq!(candidates.len(), 1);
        // `Service` is NOT in the conservative default strip list (which is
        // shared with `extract_import_candidates`), so the full name survives
        // and only camelCase normalization applies.
        assert_eq!(candidates[0].phrase, "auth service");
        assert_eq!(candidates[0].identifier, "AuthService");
        assert_eq!(candidates[0].source, CandidateSource::PublicSymbol);
    }

    #[test]
    fn test_strip_suffix_applies_for_configured_suffixes() {
        // `Manager` is in the default strip list — strip + normalize.
        let chunks = vec![
            meta(&[("chunk_type", "struct"), ("symbol_name", "AuthManager")]),
            meta(&[("chunk_type", "struct"), ("symbol_name", "RequestHandler")]),
            meta(&[("chunk_type", "struct"), ("symbol_name", "PrimeSieveImpl")]),
        ];
        let candidates = extract_symbol_candidates(&chunks, &LspCandidateConfig::default());
        let phrases: Vec<&str> = candidates.iter().map(|c| c.phrase.as_str()).collect();
        assert!(phrases.contains(&"auth"), "got: {:?}", phrases);
        assert!(phrases.contains(&"request"), "got: {:?}", phrases);
        assert!(phrases.contains(&"prime sieve"), "got: {:?}", phrases);
    }

    #[test]
    fn test_extract_camel_case_symbol_normalized() {
        let chunks = vec![meta(&[
            ("chunk_type", "struct"),
            ("symbol_name", "HttpClient"),
        ])];
        let candidates = extract_symbol_candidates(&chunks, &LspCandidateConfig::default());

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].phrase, "http client");
    }

    #[test]
    fn test_method_chunks_skipped() {
        // Method chunks carry parent_symbol and must not contribute.
        let chunks = vec![meta(&[
            ("chunk_type", "method"),
            ("symbol_name", "login"),
            ("parent_symbol", "AuthService"),
        ])];
        let candidates = extract_symbol_candidates(&chunks, &LspCandidateConfig::default());
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_function_with_parent_skipped() {
        // Nested function classified as function but with parent_symbol set
        // (e.g., closure-named, inner def in Python) — attribute to parent.
        let chunks = vec![meta(&[
            ("chunk_type", "function"),
            ("symbol_name", "helper"),
            ("parent_symbol", "outer_function"),
        ])];
        let candidates = extract_symbol_candidates(&chunks, &LspCandidateConfig::default());
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_preamble_and_text_sentinels_skipped() {
        let chunks = vec![
            meta(&[("chunk_type", "preamble"), ("symbol_name", "_preamble")]),
            meta(&[("chunk_type", "text"), ("symbol_name", "_text")]),
        ];
        let candidates = extract_symbol_candidates(&chunks, &LspCandidateConfig::default());
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_short_identifier_filtered() {
        let chunks = vec![meta(&[("chunk_type", "function"), ("symbol_name", "io")])];
        let config = LspCandidateConfig::default(); // min_identifier_len = 3
        let candidates = extract_symbol_candidates(&chunks, &config);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_dedupe_across_fragments() {
        // A 300-line function split into 3 fragments produces 3 chunks all
        // carrying the same symbol_name. Should emit one candidate.
        let chunks = vec![
            meta(&[
                ("chunk_type", "function"),
                ("symbol_name", "process_request"),
                ("is_fragment", "true"),
                ("fragment_index", "0"),
            ]),
            meta(&[
                ("chunk_type", "function"),
                ("symbol_name", "process_request"),
                ("is_fragment", "true"),
                ("fragment_index", "1"),
            ]),
            meta(&[
                ("chunk_type", "function"),
                ("symbol_name", "process_request"),
                ("is_fragment", "true"),
                ("fragment_index", "2"),
            ]),
        ];
        let candidates = extract_symbol_candidates(&chunks, &LspCandidateConfig::default());
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].phrase, "process request");
    }

    #[test]
    fn test_mixed_top_level_symbols() {
        let chunks = vec![
            meta(&[("chunk_type", "struct"), ("symbol_name", "RetryPolicy")]),
            meta(&[("chunk_type", "trait"), ("symbol_name", "AsyncRunner")]),
            meta(&[
                ("chunk_type", "function"),
                ("symbol_name", "validate_token"),
            ]),
            meta(&[
                ("chunk_type", "method"),
                ("symbol_name", "exec"),
                ("parent_symbol", "AsyncRunner"),
            ]),
            meta(&[("chunk_type", "constant"), ("symbol_name", "MAX_RETRIES")]),
        ];
        let candidates = extract_symbol_candidates(&chunks, &LspCandidateConfig::default());

        let phrases: Vec<&str> = candidates.iter().map(|c| c.phrase.as_str()).collect();
        // Struct → "RetryPolicy" → no suffix matches (Policy isn't in default strip list) → "retry policy"
        assert!(phrases.contains(&"retry policy"), "got: {:?}", phrases);
        // Trait → "AsyncRunner" → no suffix → "async runner"
        assert!(phrases.contains(&"async runner"), "got: {:?}", phrases);
        // Function → "validate_token" → "validate token"
        assert!(phrases.contains(&"validate token"), "got: {:?}", phrases);
        // Method excluded
        assert!(
            !phrases.iter().any(|p| p.contains("exec")),
            "got: {:?}",
            phrases
        );
        // Constant excluded (not concept-bearing)
        assert!(
            !phrases.iter().any(|p| p.contains("max")),
            "got: {:?}",
            phrases
        );
    }

    #[test]
    fn test_chunk_without_chunk_type_skipped() {
        let chunks = vec![meta(&[("symbol_name", "SomeStruct")])];
        let candidates = extract_symbol_candidates(&chunks, &LspCandidateConfig::default());
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_chunk_without_symbol_name_skipped() {
        let chunks = vec![meta(&[("chunk_type", "struct")])];
        let candidates = extract_symbol_candidates(&chunks, &LspCandidateConfig::default());
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_priority_boost_propagated() {
        let mut config = LspCandidateConfig::default();
        config.priority_boost = 2.7;
        let chunks = vec![meta(&[("chunk_type", "struct"), ("symbol_name", "Widget")])];
        let candidates = extract_symbol_candidates(&chunks, &config);
        assert_eq!(candidates.len(), 1);
        assert!((candidates[0].priority_boost - 2.7).abs() < 1e-9);
    }

    #[test]
    fn test_empty_input() {
        let candidates = extract_symbol_candidates(&[], &LspCandidateConfig::default());
        assert!(candidates.is_empty());
    }
}

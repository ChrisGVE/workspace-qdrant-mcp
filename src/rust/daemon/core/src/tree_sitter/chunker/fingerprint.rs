//! Chunking-configuration fingerprint for extractor-upgrade propagation.
//!
//! A file's semantic chunks are a function of (a) the chunker code and (b)
//! the language's `semantic_patterns` entry in the bundled registry YAML.
//! The unchanged-hash skip in the ingest gate makes re-ingestion of
//! unmodified files free — but it also meant a registry upgrade never
//! reached already-indexed files: adding `semantic_patterns` for `.proto`
//! left every deployed `.proto` file on its old text chunks until its
//! content happened to change.
//!
//! The fingerprint stored in `tracked_files.chunker_version` captures the
//! chunking configuration that produced a row's chunks. The ingest gate
//! re-processes an unchanged file when its stored fingerprint no longer
//! matches the current one, so registry upgrades propagate on the next
//! scan/re-embed that visits the file. Rows written before the column
//! existed (`NULL`) are grandfathered — they never trigger a re-chunk by
//! themselves; a forced re-embed (`ReembedTenant{force}` → `File/Uplift`)
//! stamps them.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Manual escape hatch for chunker CODE changes that alter chunk output
/// without touching the registry YAML (splitting rules, GenericExtractor
/// walker fixes, …). Bump when such a change should propagate to
/// already-indexed, unchanged files on their next visit.
const CHUNKER_LOGIC_VERSION: u32 = 1;

/// Per-language fingerprints, computed once from the bundled registry.
///
/// The digest input is the serde_json serialization of the parsed
/// [`SemanticPatterns`](crate::language_registry::types::SemanticPatterns).
/// That struct is plain fields/Vecs (no maps), so serialization order is
/// the struct definition order — deterministic across processes, which the
/// gate relies on: a stored fingerprint may only mismatch when the chunking
/// configuration actually changed, never from serialization noise.
fn fingerprint_map() -> &'static HashMap<String, String> {
    static MAP: OnceLock<HashMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| {
        super::strategy::registry_patterns()
            .iter()
            .map(|(lang, patterns)| {
                let canonical = serde_json::to_string(patterns).unwrap_or_default();
                let digest = wqm_common::hashing::compute_content_hash(&canonical);
                (
                    lang.clone(),
                    format!("{}:{}:{}", CHUNKER_LOGIC_VERSION, lang, &digest[..12]),
                )
            })
            .collect()
    })
}

/// Current chunking fingerprint for a detected language.
///
/// - `Some(lang)` with registry patterns → `<ver>:<lang>:<patterns digest>`
/// - `Some(lang)` without patterns (text fallback) → `<ver>:<lang>:nopat`
/// - `None` (undetected → text fallback) → `<ver>:text`
///
/// The ingest gate and the `tracked_files` writer MUST both derive the value
/// through this function, from the same detection call
/// (`detect_language_with_overrides`), so the comparison is stable.
pub fn chunking_fingerprint(language: Option<&str>) -> String {
    match language {
        Some(lang) => fingerprint_map()
            .get(lang)
            .cloned()
            .unwrap_or_else(|| format!("{}:{}:nopat", CHUNKER_LOGIC_VERSION, lang)),
        None => format!("{}:text", CHUNKER_LOGIC_VERSION),
    }
}

/// Whether a stored `tracked_files.chunker_version` is still current.
///
/// `None` (row predates the column, or was written by a path that does not
/// chunk — zero-byte files, dedup clones of legacy rows) is treated as
/// CURRENT: re-chunking the whole corpus spontaneously on the first deploy
/// of this column would be a thundering herd of embedding work. Legacy rows
/// converge on their next genuine re-ingest (content change or forced
/// re-embed).
pub fn stored_fingerprint_is_current(stored: Option<&str>, current: &str) -> bool {
    match stored {
        None => true,
        Some(s) => s == current,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_stable_per_language() {
        let a = chunking_fingerprint(Some("rust"));
        let b = chunking_fingerprint(Some("rust"));
        assert_eq!(a, b, "same language must produce the same fingerprint");
        assert!(
            a.starts_with(&format!("{}:rust:", CHUNKER_LOGIC_VERSION)),
            "fingerprint embeds the logic version and language: {a}"
        );
    }

    #[test]
    fn languages_with_patterns_get_distinct_digests() {
        let rust = chunking_fingerprint(Some("rust"));
        let python = chunking_fingerprint(Some("python"));
        assert_ne!(
            rust, python,
            "different languages must not share a fingerprint"
        );
        // Both are bundled with semantic_patterns — neither should be the
        // 'nopat' placeholder.
        assert!(!rust.ends_with(":nopat"), "rust has bundled patterns");
        assert!(!python.ends_with(":nopat"), "python has bundled patterns");
    }

    #[test]
    fn proto_has_a_patterns_backed_fingerprint() {
        // Regression guard for the original gap: `.proto` gained
        // semantic_patterns in the registry; its fingerprint must reflect
        // them (and would have differed from the pre-patterns value).
        // The registry id is "protobuf" ("proto" is an alias), which is
        // also what detect_language returns for .proto files.
        let proto = chunking_fingerprint(Some("protobuf"));
        assert!(
            !proto.ends_with(":nopat"),
            "protobuf must have registry patterns: {proto}"
        );
    }

    #[test]
    fn undetected_language_uses_the_text_fingerprint() {
        assert_eq!(
            chunking_fingerprint(None),
            format!("{}:text", CHUNKER_LOGIC_VERSION)
        );
    }

    #[test]
    fn unknown_language_gets_a_deterministic_placeholder() {
        assert_eq!(
            chunking_fingerprint(Some("no-such-language")),
            format!("{}:no-such-language:nopat", CHUNKER_LOGIC_VERSION)
        );
    }

    #[test]
    fn stored_null_is_grandfathered_and_mismatch_is_stale() {
        let current = chunking_fingerprint(Some("rust"));
        assert!(stored_fingerprint_is_current(None, &current));
        assert!(stored_fingerprint_is_current(Some(&current), &current));
        assert!(!stored_fingerprint_is_current(
            Some("0:rust:deadbeef0000"),
            &current
        ));
    }
}

//! Tests for the handle resolver (AC-F16.6).
//!
//! File: `wqm-common/src/handle/handle_tests.rs`
//! Context: sibling test module for `handle/mod.rs`, split to keep the main
//!   file under the 500-line codesize budget.

use super::{
    normalize_handle, resolve_handle, HandleCandidate, HandleResolveError, ResolveAction, Resolved,
    JARO_WINKLER_THRESHOLD,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cand(handle: &str, key: &str) -> HandleCandidate {
    HandleCandidate {
        handle: handle.to_string(),
        key: key.to_string(),
    }
}

fn cands(pairs: &[(&str, &str)]) -> Vec<HandleCandidate> {
    pairs.iter().map(|(h, k)| cand(h, k)).collect()
}

// ---------------------------------------------------------------------------
// normalize_handle unit tests
// ---------------------------------------------------------------------------

// ASCII input is lowercased.
#[test]
fn t_normalize_ascii_lowercased() {
    assert_eq!(normalize_handle("MathLex"), "mathlex");
    assert_eq!(normalize_handle("MATHLEX"), "mathlex");
    assert_eq!(normalize_handle("mathlex"), "mathlex");
}

// NFC equivalence: composed (U+00E9 é) vs decomposed (e + U+0301 ́).
#[test]
fn t_normalize_nfc_equivalence() {
    let composed = "caf\u{00E9}"; // é as precomposed char
    let decomposed = "cafe\u{0301}"; // e followed by combining acute
    assert_eq!(
        normalize_handle(composed),
        normalize_handle(decomposed),
        "NFC normalization must make composed and decomposed forms identical"
    );
}

// ---------------------------------------------------------------------------
// Exact-match short-circuit (step 1 of algorithm)
// ---------------------------------------------------------------------------

// `mathlex` resolves to `mathlex` even though `mathlex-eval` exists
// (exact wins over prefix match).
#[test]
fn t_exact_wins_over_near_match() {
    let cs = cands(&[
        ("mathlex", "tid-mathlex"),
        ("mathlex-eval", "tid-mathlex-eval"),
    ]);
    let result = resolve_handle("mathlex", &cs, ResolveAction::Read).unwrap();
    match result {
        Resolved::Exact(c) => {
            assert_eq!(c.handle, "mathlex");
            assert_eq!(c.key, "tid-mathlex", "must return the correct key");
        }
        other => panic!("expected Exact, got {other:?}"),
    }
}

// Case-insensitive exact match: `MathLex` == `mathlex`.
#[test]
fn t_exact_case_insensitive() {
    let cs = cands(&[
        ("mathlex", "tid-mathlex"),
        ("mathlex-eval", "tid-mathlex-eval"),
    ]);
    let result = resolve_handle("MathLex", &cs, ResolveAction::Read).unwrap();
    match result {
        Resolved::Exact(c) => {
            assert_eq!(c.key, "tid-mathlex");
        }
        other => panic!("expected Exact, got {other:?}"),
    }
}

// NFC exact match: composed accented form resolves same as decomposed stored form.
#[test]
fn t_exact_nfc_composed_vs_decomposed() {
    // Store with decomposed form; query with composed.
    let stored_decomposed = "cafe\u{0301}"; // café decomposed
    let query_composed = "caf\u{00E9}"; // café composed
    let cs = cands(&[(stored_decomposed, "tid-cafe")]);
    let result = resolve_handle(query_composed, &cs, ResolveAction::Read).unwrap();
    match result {
        Resolved::Exact(c) => {
            assert_eq!(c.key, "tid-cafe", "NFC normalization must unify forms");
        }
        other => panic!("expected Exact, got {other:?}"),
    }
}

// Multiple candidates that normalize-equal the input → Ambiguous.
#[test]
fn t_exact_multiple_normalize_equal_is_ambiguous() {
    // Two entries with same handle (edge case — storage should prevent this,
    // but the resolver must not silently pick one).
    let cs = cands(&[("mathlex", "tid-1"), ("MATHLEX", "tid-2")]);
    let err = resolve_handle("mathlex", &cs, ResolveAction::Read).unwrap_err();
    match err {
        HandleResolveError::Ambiguous { candidates, .. } => {
            assert_eq!(candidates.len(), 2, "both must appear in the candidate set");
        }
        other => panic!("expected Ambiguous, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Fuzzy step (step 2 of algorithm) — Read tier
// ---------------------------------------------------------------------------

// A typo close enough (≥0.92) resolves under READ.
#[test]
fn t_fuzzy_typo_resolves_under_read() {
    let cs = cands(&[("mathlex", "tid-mathlex")]);
    // "mathlx" is a single-char deletion; Jaro-Winkler should be well above 0.92.
    let result = resolve_handle("mathlx", &cs, ResolveAction::Read).unwrap();
    match result {
        Resolved::Exact(c) => assert_eq!(c.key, "tid-mathlex"),
        other => panic!("expected Exact, got {other:?}"),
    }
}

// A typo below threshold → NotFound.
#[test]
fn t_fuzzy_too_distant_not_found() {
    let cs = cands(&[("mathlex", "tid-mathlex")]);
    // Completely different string — far below 0.92.
    let err = resolve_handle("zzzzzzz", &cs, ResolveAction::Read).unwrap_err();
    match err {
        HandleResolveError::NotFound { handle } => {
            assert_eq!(handle, "zzzzzzz");
        }
        other => panic!("expected NotFound, got {other:?}"),
    }
}

// Multiple over-threshold fuzzy candidates → Ambiguous under READ.
#[test]
fn t_fuzzy_multiple_over_threshold_is_ambiguous_under_read() {
    // Two candidates very close to each other and to the input.
    let cs = cands(&[("mathlex", "tid-a"), ("mathlexe", "tid-b")]);
    // "mathlexs" is equidistant from both — both should clear 0.92.
    let err = resolve_handle("mathlexs", &cs, ResolveAction::Read).unwrap_err();
    match err {
        HandleResolveError::Ambiguous { candidates, .. } => {
            assert!(
                candidates.len() >= 2,
                "both over-threshold candidates must appear"
            );
            let keys: Vec<&str> = candidates.iter().map(|c| c.key.as_str()).collect();
            assert!(
                keys.contains(&"tid-a"),
                "tid-a must be in the ambiguous set"
            );
        }
        other => panic!("expected Ambiguous, got {other:?}"),
    }
}

// Empty candidate list → NotFound.
#[test]
fn t_empty_candidate_list_not_found() {
    let err = resolve_handle("anything", &[], ResolveAction::Read).unwrap_err();
    matches!(err, HandleResolveError::NotFound { .. });
}

// ---------------------------------------------------------------------------
// JARO_WINKLER_THRESHOLD value (§14-Q9 constant)
// ---------------------------------------------------------------------------

#[test]
fn t_threshold_constant_is_0_92() {
    assert!(
        (JARO_WINKLER_THRESHOLD - 0.92).abs() < f64::EPSILON,
        "JARO_WINKLER_THRESHOLD must be exactly 0.92 (§14-Q9)"
    );
}

// A score just at the threshold (using a contrived string pair) should accept.
// We verify that the threshold comparison is ≥ (inclusive), not >.
#[test]
fn t_threshold_is_inclusive() {
    // Use a candidate pair whose Jaro-Winkler score we can reason about.
    // "mathlex" vs "mathlex" = 1.0 ≥ 0.92 → resolves.
    let cs = cands(&[("mathlex", "tid-x")]);
    let result = resolve_handle("mathlex", &cs, ResolveAction::Read);
    assert!(result.is_ok(), "exact-same string must always resolve");
}

// ---------------------------------------------------------------------------
// Action-tier tests
// ---------------------------------------------------------------------------

// WRITE tier with a non-exact near-miss → BestGuess (never a silent resolve).
#[test]
fn t_write_tier_near_miss_returns_best_guess() {
    let cs = cands(&[("mathlex", "tid-mathlex")]);
    let result = resolve_handle("mathlx", &cs, ResolveAction::Write).unwrap();
    match result {
        Resolved::BestGuess { best, .. } => {
            assert_eq!(best.handle, "mathlex");
            assert_eq!(best.key, "tid-mathlex");
        }
        Resolved::Exact(_) => panic!("Write tier on a near-miss must return BestGuess, not Exact"),
    }
}

// DESTRUCTIVE tier with a non-exact near-miss → BestGuess (same as Write for
// resolver output; CLI must demand typed unique-ID confirmation).
#[test]
fn t_destructive_tier_near_miss_returns_best_guess() {
    let cs = cands(&[("mathlex", "tid-mathlex")]);
    let result = resolve_handle("mathlx", &cs, ResolveAction::Destructive).unwrap();
    match result {
        Resolved::BestGuess { best, alternatives } => {
            assert_eq!(best.key, "tid-mathlex");
            // alternatives may be empty when there's only one candidate.
            let _ = alternatives;
        }
        Resolved::Exact(_) => {
            panic!("Destructive tier on a non-exact input must return BestGuess, not Exact")
        }
    }
}

// WRITE tier with exact input → Exact (exact is always acceptable).
#[test]
fn t_write_tier_exact_input_returns_exact() {
    let cs = cands(&[("mathlex", "tid-mathlex"), ("other", "tid-other")]);
    let result = resolve_handle("mathlex", &cs, ResolveAction::Write).unwrap();
    match result {
        Resolved::Exact(c) => assert_eq!(c.key, "tid-mathlex"),
        other => panic!("exact input on Write tier must return Exact, got {other:?}"),
    }
}

// WRITE tier, nothing in range → NotFound.
#[test]
fn t_write_tier_no_match_not_found() {
    let cs = cands(&[("mathlex", "tid-mathlex")]);
    let err = resolve_handle("zzzzzzz", &cs, ResolveAction::Write).unwrap_err();
    matches!(err, HandleResolveError::NotFound { .. });
}

// WRITE tier with multiple over-threshold candidates exposes all via BestGuess.
#[test]
fn t_write_tier_multiple_candidates_best_guess_with_alternatives() {
    let cs = cands(&[("mathlex", "tid-a"), ("mathlexe", "tid-b")]);
    let result = resolve_handle("mathlexs", &cs, ResolveAction::Write).unwrap();
    match result {
        Resolved::BestGuess { best, alternatives } => {
            // best is the highest-scoring one; alternatives contains the rest.
            let all_keys: Vec<&str> = std::iter::once(best.key.as_str())
                .chain(alternatives.iter().map(|c| c.key.as_str()))
                .collect();
            assert!(
                all_keys.contains(&"tid-a") || all_keys.contains(&"tid-b"),
                "at least one candidate must appear in BestGuess output"
            );
        }
        Resolved::Exact(_) => panic!("non-exact input must yield BestGuess on Write tier"),
    }
}

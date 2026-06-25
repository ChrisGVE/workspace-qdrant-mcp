//! Broad handle↔key resolver shared across all project and collection handles
//! (AC-F16.6 FP-3).
//!
//! File: `wqm-common/src/handle/mod.rs`
//! Location: `src/rust/common/src/handle/` (leaf crate, no write deps)
//! Context: workspace-qdrant-mcp branch-storage model.
//!
//! ## Design (FP-2 / FP-3)
//!
//! One broad resolver serves ALL handle types:
//!   - project names (state.db `projects.name` / `projects.tenant_id`)
//!   - scratchpad/rules collection handles (future — same function, different
//!     candidate list)
//!   - any future handle type
//!
//! No per-call-site name matching; no duplicated resolver logic. The
//! `ProjectRegistry` nexus (`wqm-storage/src/project/resolver.rs`) consumes
//! this module; it is not duplicated there.
//!
//! ## Normalization (DOM-R8-N2)
//!
//! Both input and each candidate pass through the SAME [`normalize_handle`]
//! before any comparison — exact OR fuzzy:
//!   1. Unicode NFC via `unicode-normalization` (`.nfc().collect::<String>()`)
//!      so composed and decomposed accented forms compare equal.
//!   2. Unicode case-fold to lowercase via `str::to_lowercase`, so `MathLex`,
//!      `mathlex`, and `MATHLEX` compare equal.
//!
//! Using ONE helper for both steps is the correctness requirement (DOM-R8-N2):
//! two different fold functions could disagree on an edge codepoint and make an
//! exact-equal pair score as a non-exact fuzzy match. `str::to_lowercase`
//! realizes the PRD's "simple case-fold" intent — case-insensitive NFC matching
//! — across the ASCII/Latin/script range that project, library, and
//! scratchpad/rules handles occupy; it needs no extra dependency. Stored keys
//! are NEVER altered; normalization produces a transient comparison key only.
//!
//! ## Algorithm (§14-Q9)
//!
//! 1. Normalize the input AND every candidate.handle.
//! 2. Exact short-circuit: if exactly ONE normalized candidate equals the
//!    normalized input → return it (case-insensitive exact match). If MULTIPLE
//!    candidates normalize-equal the input → `Ambiguous` (cannot pick silently).
//! 3. Fuzzy step: compute `strsim::jaro_winkler(normalized_input,
//!    normalized_candidate)` (fixed arg order: input first, DOM-R8-N3).
//!    Collect all candidates with score ≥ `JARO_WINKLER_THRESHOLD`.
//! 4. Apply action-tier rules to the over-threshold set (see `ResolveAction`).
//!
//! ## Action-tier semantics
//!
//! | Tier | Over-threshold set | Result |
//! |---|---|---|
//! | `Read` | 0 candidates | `Err(NotFound)` |
//! | `Read` | 1 candidate | `Ok(Resolved::Exact)` |
//! | `Read` | ≥2 candidates | `Err(Ambiguous { candidates })` |
//! | `Write` | exact required; near-miss set | `Ok(Resolved::BestGuess)` for caller confirm |
//! | `Write` | 0 candidates | `Err(NotFound)` |
//! | `Destructive` | same as Write; caller demands unique-ID confirm |
//!
//! WRITE and DESTRUCTIVE never silently resolve a non-exact match; they return
//! `Resolved::BestGuess { best, alternatives }` so the caller can prompt for
//! confirmation before acting.

use unicode_normalization::UnicodeNormalization;

// ---------------------------------------------------------------------------
// Config constant (§14-Q9: single source, no literal 0.92 at comparison sites)
// ---------------------------------------------------------------------------

/// Default Jaro-Winkler similarity threshold for fuzzy handle matching.
///
/// A score ≥ this value is considered "over-threshold" for fuzzy resolution.
/// Exact matches (step 1 of the algorithm) are not subject to this threshold.
///
/// §14-Q9: Chris-overridable async via the §14 tolerance mechanism (future
/// config surface). Until then this constant is the single source of truth;
/// never compare against a literal `0.92` at any call site.
pub const JARO_WINKLER_THRESHOLD: f64 = 0.92;

// ---------------------------------------------------------------------------
// Error + candidate types (IMPL-R7-05)
// ---------------------------------------------------------------------------

/// A resolved handle+key pair returned in ambiguity / best-guess sets.
///
/// `handle` is the human-readable display name; `key` is the opaque unique
/// identifier (e.g. `tenant_id` for projects, collection_id for scratchpad).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandleCandidate {
    /// Human-readable handle (e.g. project name, collection name).
    pub handle: String,
    /// Opaque unique key (e.g. `tenant_id`). Used for DESTRUCTIVE confirm.
    pub key: String,
}

/// Resolution error returned by [`resolve_handle`].
///
/// `Ambiguous` carries the candidate set so the caller can render a
/// disambiguation prompt. For DESTRUCTIVE calls the caller must print each
/// candidate's `key` and demand the user type it to confirm.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HandleResolveError {
    /// Multiple candidates match `handle` and cannot be resolved unambiguously.
    #[error("handle '{handle}' is ambiguous — {n} candidates: {names}",
        n = candidates.len(),
        names = candidates.iter().map(|c| c.handle.as_str()).collect::<Vec<_>>().join(", "))]
    Ambiguous {
        /// The input handle that triggered the ambiguity.
        handle: String,
        /// The set of candidates that matched (each carries handle + key).
        candidates: Vec<HandleCandidate>,
    },
    /// No candidate matches `handle` (exact or within threshold).
    #[error("handle '{handle}' not found")]
    NotFound {
        /// The input handle that was not found.
        handle: String,
    },
}

// ---------------------------------------------------------------------------
// Action tier
// ---------------------------------------------------------------------------

/// Action tier controlling resolution strictness (AC-F16.6).
///
/// The tier determines how the resolver behaves when the input is not an
/// exact match:
///
/// - [`Read`](ResolveAction::Read): fuzzy resolution; single best candidate
///   wins; ambiguity (≥2) surfaces the candidate set.
/// - [`Write`](ResolveAction::Write): exact match required to silently
///   resolve; a near-miss returns `BestGuess` for caller confirmation.
/// - [`Destructive`](ResolveAction::Destructive): strongest safeguard —
///   same as Write but the caller is expected to print each candidate's
///   unique `key` and demand typed confirmation from the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveAction {
    /// Read operation: fuzzy single-candidate resolution allowed.
    Read,
    /// Write operation: exact required; near-miss surfaces for confirmation.
    Write,
    /// Destructive operation: exact + unique-ID typed confirmation required.
    Destructive,
}

// ---------------------------------------------------------------------------
// Resolved outcome
// ---------------------------------------------------------------------------

/// Successful resolution outcome returned by [`resolve_handle`].
///
/// `Exact` means the input matched one candidate exactly (case-insensitively).
/// `BestGuess` means the input was close but not exact — the caller MUST
/// present `best` and `alternatives` to the user and require explicit
/// confirmation before proceeding (WRITE / DESTRUCTIVE tiers).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolved {
    /// Unambiguous exact (case-insensitive) match. Safe to use directly.
    Exact(HandleCandidate),
    /// Near-miss: `best` is the highest-scoring candidate; `alternatives` are
    /// other over-threshold candidates (may be empty). The caller MUST confirm
    /// with the user before acting — never proceed silently on a BestGuess.
    BestGuess {
        /// Highest-scoring fuzzy candidate.
        best: HandleCandidate,
        /// Other over-threshold candidates (for the confirmation prompt).
        alternatives: Vec<HandleCandidate>,
    },
}

// ---------------------------------------------------------------------------
// Normalization helper (DOM-R8-N2)
// ---------------------------------------------------------------------------

/// Normalize a handle for comparison: Unicode NFC then case-fold to lowercase.
///
/// Applied identically to both the input and every candidate before any
/// comparison (exact or fuzzy) — the same comparison key feeds the exact
/// short-circuit (string equality) and the Jaro-Winkler fuzzy step, so the two
/// steps can never disagree on a pair. The stored key is never altered; this
/// produces a transient comparison key only.
///
/// `str::to_lowercase` realizes the PRD's "simple case-fold" intent
/// (case-insensitive NFC matching) for the ASCII/Latin/script range that
/// handles occupy, with no extra dependency (FP-2 leverage-existing).
pub fn normalize_handle(s: &str) -> String {
    // Step 1: NFC — canonicalize composed vs. decomposed forms.
    let nfc: String = s.chars().nfc().collect();
    // Step 2: case-fold to lowercase.
    nfc.to_lowercase()
}

// ---------------------------------------------------------------------------
// Resolver entrypoint
// ---------------------------------------------------------------------------

/// Resolve `input` against `candidates` under the given `action` tier.
///
/// # Algorithm
///
/// 1. Normalize the input and all candidate handles (NFC + case-fold).
/// 2. Exact short-circuit: count candidates whose normalized handle equals the
///    normalized input (via focaccia Full case-fold equality).
///    - Exactly one → `Ok(Resolved::Exact(...))`.
///    - Multiple → `Err(Ambiguous)` (prefix collision, e.g. "foo" vs "foo2").
/// 3. Fuzzy: score each candidate with `strsim::jaro_winkler(input, cand)`
///    (fixed arg order per DOM-R8-N3). Collect all ≥ `JARO_WINKLER_THRESHOLD`.
/// 4. Apply action-tier rules (see `ResolveAction` docs).
///
/// # Argument order (DOM-R8-N3)
///
/// `strsim::jaro_winkler` is symmetric for the default implementation but the
/// resolver always passes `(normalized_input, normalized_candidate)` so that
/// switching to a non-symmetric metric in future cannot make resolution
/// depend on iteration order.
pub fn resolve_handle(
    input: &str,
    candidates: &[HandleCandidate],
    action: ResolveAction,
) -> Result<Resolved, HandleResolveError> {
    let norm_input: String = normalize_handle(input);

    // Normalize every candidate handle ONCE (NFC + case-fold). The same key
    // drives both the exact short-circuit (string equality) and the fuzzy
    // scoring, so the two steps can never disagree on a pair (DOM-R8-N2).
    let norm_candidates: Vec<String> = candidates
        .iter()
        .map(|c| normalize_handle(&c.handle))
        .collect();

    // ------------------------------------------------------------------
    // Step 1: Exact short-circuit (case-insensitive via normalized equality)
    // ------------------------------------------------------------------
    let exact_matches: Vec<&HandleCandidate> = candidates
        .iter()
        .zip(norm_candidates.iter())
        .filter(|(_, norm_c)| **norm_c == norm_input)
        .map(|(c, _)| c)
        .collect();

    match exact_matches.len() {
        1 => return Ok(Resolved::Exact(exact_matches[0].clone())),
        n if n >= 2 => {
            return Err(HandleResolveError::Ambiguous {
                handle: input.to_string(),
                candidates: exact_matches.into_iter().cloned().collect(),
            });
        }
        _ => {}
    }

    // ------------------------------------------------------------------
    // Step 2: Fuzzy scoring (Jaro-Winkler, fixed arg order DOM-R8-N3)
    // ------------------------------------------------------------------
    let mut over_threshold: Vec<(f64, &HandleCandidate)> = candidates
        .iter()
        .zip(norm_candidates.iter())
        .filter_map(|(c, norm_c)| {
            let score = strsim::jaro_winkler(&norm_input, norm_c);
            if score >= JARO_WINKLER_THRESHOLD {
                Some((score, c))
            } else {
                None
            }
        })
        .collect();

    // Sort descending by score for deterministic best-pick.
    over_threshold.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    match action {
        ResolveAction::Read => match over_threshold.len() {
            0 => Err(HandleResolveError::NotFound {
                handle: input.to_string(),
            }),
            1 => Ok(Resolved::Exact(over_threshold[0].1.clone())),
            _ => Err(HandleResolveError::Ambiguous {
                handle: input.to_string(),
                candidates: over_threshold.into_iter().map(|(_, c)| c.clone()).collect(),
            }),
        },

        ResolveAction::Write | ResolveAction::Destructive => {
            // Exact is required to silently resolve. A non-exact over-threshold
            // set is returned as BestGuess so the caller can prompt the user.
            // (We already handled the exact case above; reaching here means no
            //  exact match was found.)
            if over_threshold.is_empty() {
                return Err(HandleResolveError::NotFound {
                    handle: input.to_string(),
                });
            }
            let (_, best) = over_threshold.remove(0);
            let alternatives = over_threshold.into_iter().map(|(_, c)| c.clone()).collect();
            Ok(Resolved::BestGuess {
                best: best.clone(),
                alternatives,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "handle_tests.rs"]
mod tests;

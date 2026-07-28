//! N56 -- the planner. `query + caller scope -> an explicit, inspectable plan`.
//!
//! # The planner picks HOW, never WHERE
//!
//! `CONTRACTS.md`:2343 states the invariant: a plan "may narrow WITHIN the caller's
//! scope by profile evidence only, never widen it". Here the caller's scope is the
//! collection they named in `FROM`, and every decision below narrows: which legs
//! may run, and whether the object is addressable at this collection's
//! granularity. Nothing in this module can add a source.
//!
//! # Narrowing happens AT planning, and that is why the refusals are typed
//!
//! :2355 -- "a plan referencing an absent leg cannot leave the planner (DP-6.3
//! narrowing happens AT planning)". So a leg that is gated off by profile, or that
//! has no available concrete, is refused *here* with a named capability key rather
//! than emitted and failed later. The executor never has to ask whether a plan is
//! runnable.
//!
//! # Two gates, in this order, and the order is the point
//!
//! 1. **Profile gate** -- `grep_eligible` (`CONTRACTS.md`:565-566). Scratchpad is
//!    false, so a regex plan is refused before availability is consulted at all.
//!    The axis is exercised by being false, which is stronger than being untouched:
//!    it sits on the executed path.
//! 2. **Availability gate** -- N41's `available()` probe, passed in as the leg set
//!    the composition root proved. What survives both gates is the plan.

use wqm_common::names::Collection;
use wqm_common::plan::{Leg, LegMethod, LegParams, Mode, ObjectKind, Plan, Source, SourceKind};
use wqm_common::profile::{CollectionProfiles, UnitLevel};

use crate::grammar::ParsedQuery;
use crate::QueryError;

/// The default result bound (§2.2: `limit` defaults to 10).
pub const DEFAULT_LIMIT: u32 = 10;

/// Build the plan for a parsed query, or refuse it by name.
///
/// `available` is the leg set the composition root proved by probing N41's
/// concretes -- passed in rather than discovered here, so the planner stays a pure
/// function of (query, profile, availability) and is deterministic in exactly the
/// way :2354 requires.
pub fn plan(
    parsed: &ParsedQuery,
    scope: &[Collection],
    profiles: &dyn CollectionProfiles,
    available: &[LegMethod],
) -> Result<Plan, QueryError> {
    // :2343's invariant, enforced first and structurally: a plan may narrow within
    // the caller's scope, never widen it. `scope` is what the composition root
    // serves, so a `FROM` naming a collection outside it is refused here rather
    // than executed against a store that holds nothing for it -- which would
    // return an empty set indistinguishable from a true absence.
    if !scope.contains(&parsed.source) {
        return Err(QueryError::Unsupported {
            capability_key: "sources",
            message: format!(
                "this build searches {}; `{}` is outside its scope",
                scope_names(scope),
                parsed.source.name()
            ),
            suggestion: None,
        });
    }

    let profile = profiles.profile(parsed.source);
    if !profile.is_searchable {
        return Err(QueryError::Unsupported {
            capability_key: "sources",
            message: format!(
                "`{}` is not a searchable collection on this deployment",
                parsed.source.name()
            ),
            suggestion: None,
        });
    }

    check_object(parsed.object, profile.granularity)?;
    let mode = parsed.mode.unwrap_or(Mode::Semantic);
    let candidates = legs_for_mode(mode, profile.grep_eligible)?;
    let legs = narrow_to_available(mode, &candidates, available)?;

    Ok(Plan {
        mode,
        object: parsed.object,
        sources: vec![source_of(parsed)],
        // The sealed planner contract defaults `strict` to false. This build
        // executes no taxonomy broadening, so false and true would run the same
        // legs -- the value is reported because it is part of what ran, not
        // because it changed anything here.
        strict: false,
        legs,
        // No predicate reaches `filters`: the `q MATCH` clause IS the retrieval
        // leg, not a post-filter over it. A build that listed it here would be
        // claiming a filter step it does not run.
        filters: Vec::new(),
        fuse: None,
        expand: None,
        rerank: None,
        limit: parsed.limit.unwrap_or(DEFAULT_LIMIT),
    })
}

/// Which legs a mode would like to run, before availability narrows them.
fn legs_for_mode(mode: Mode, grep_eligible: bool) -> Result<Vec<LegMethod>, QueryError> {
    match mode {
        Mode::Semantic => Ok(vec![LegMethod::Dense, LegMethod::Sparse]),
        Mode::Text => Ok(vec![LegMethod::Trigram]),
        Mode::Regex if grep_eligible => Ok(vec![LegMethod::Regex]),
        // GT002 §B: no grep on libraries, scratchpad or rules. The refusal names
        // the axis so an agent learns the rule rather than the symptom.
        Mode::Regex => Err(QueryError::Unsupported {
            capability_key: "grep_eligible",
            message: "regular-expression search is not eligible on this collection; \
                      its profile declares `grep_eligible: false`"
                .into(),
            suggestion: None,
        }),
    }
}

/// Keep only the legs that have a proven concrete, and refuse what is left over.
///
/// Two refusals, and they are different failures:
///
/// - **Nothing survives** -- the mode has no concrete in this build. Serving it
///   with a different leg would be answering a question the caller did not ask.
/// - **More than one survives** -- fusion would be required, and N4 is not in this
///   build. `CONTRACTS.md`:1012-1014 makes this structural rather than a missing
///   feature: `fuse` takes `legs: Vec<RankedList>` and needs N17 dense scores, so
///   a multi-leg plan is not a bigger version of this one.
fn narrow_to_available(
    mode: Mode,
    candidates: &[LegMethod],
    available: &[LegMethod],
) -> Result<Vec<Leg>, QueryError> {
    let survivors: Vec<LegMethod> = candidates
        .iter()
        .copied()
        .filter(|m| available.contains(m))
        .collect();

    if survivors.is_empty() {
        return Err(QueryError::Unsupported {
            capability_key: "modes",
            message: format!(
                "this build has no retrieval leg for {} mode; `status` reports the \
                 modes it executes",
                mode_name(mode)
            ),
            suggestion: None,
        });
    }
    if survivors.len() > 1 {
        return Err(QueryError::Unsupported {
            capability_key: "fusion",
            message: "this build executes one retrieval leg and has no fusion step, \
                      so a plan needing more than one leg cannot be run"
                .into(),
            suggestion: None,
        });
    }

    Ok(survivors.into_iter().map(leg).collect())
}

/// A leg with the parameters its method actually uses, and `null` for the rest.
fn leg(method: LegMethod) -> Leg {
    let mut params = LegParams::none();
    if method == LegMethod::Trigram {
        // The index name is the concrete that answers, which is what an agent
        // reading the echoed plan needs in order to interpret the ordering.
        params.index = Some("fts5".into());
    }
    Leg { method, params }
}

/// Whether this collection's granularity makes the requested object addressable.
///
/// N40 enumerates the unit ladder from N35's `granularity` axis and states that
/// rules and scratchpad are document-level only (`CONTRACTS.md`:1089-1090). Asking
/// a document-level collection for a `chunk` is therefore not an empty result --
/// there is no such unit to return, and saying so is the difference between "none
/// matched" and "that is not a thing here".
fn check_object(object: ObjectKind, granularity: UnitLevel) -> Result<(), QueryError> {
    let addressable: &[ObjectKind] = match granularity {
        UnitLevel::Document => &[ObjectKind::Document, ObjectKind::Note, ObjectKind::Rule],
        UnitLevel::Section => &[ObjectKind::Document, ObjectKind::Chunk],
        UnitLevel::Chunk => &[ObjectKind::Document, ObjectKind::Chunk, ObjectKind::Line],
    };
    if addressable.contains(&object) {
        return Ok(());
    }
    Err(QueryError::Unsupported {
        capability_key: "objects",
        message: format!(
            "this collection is addressable at `{}` level, which does not carry the \
             requested object",
            granularity.as_str()
        ),
        suggestion: None,
    })
}

/// The plan's one source, in the algebra's own vocabulary.
fn source_of(parsed: &ParsedQuery) -> Source {
    Source {
        kind: source_kind(parsed),
        // `rules` and scratchpad are unnamed sources: there is exactly one of
        // each, so a name would be a second spelling of the kind.
        name: None,
    }
}

/// The served scope, for the sentence that refuses everything outside it.
fn scope_names(scope: &[Collection]) -> String {
    if scope.is_empty() {
        return "no collection".into();
    }
    scope
        .iter()
        .map(|c| format!("`{}`", c.name()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn source_kind(parsed: &ParsedQuery) -> SourceKind {
    match parsed.source {
        Collection::Projects => SourceKind::Project,
        Collection::Libraries => SourceKind::Library,
        Collection::Rules => SourceKind::Rules,
        Collection::Scratchpad => SourceKind::Scratchpad,
    }
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::Semantic => "semantic",
        Mode::Text => "text",
        Mode::Regex => "regex",
    }
}

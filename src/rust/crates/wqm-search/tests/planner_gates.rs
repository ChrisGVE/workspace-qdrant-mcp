//! The planner's gates, each exercised on the path that runs -- including the one
//! that is exercised by being FALSE.
//!
//! `SCAFFOLD.md` §7.2's claim is that grep "is not reached" because N35's
//! `grep_eligible` is false for the scratchpad, and that this is *stronger* than
//! not touching the axis. That claim is only true if the gate is load-bearing --
//! if flipping the axis changes the outcome. Both directions are asserted here, so
//! the false reading is evidence rather than an absence.

use wqm_common::names::Collection;
use wqm_common::plan::LegMethod;
use wqm_common::profile::{CollectionProfile, CollectionProfiles, UnitLevel};
use wqm_search::{parse, plan, QueryError};

/// A profile table under the test's control, so an axis can be flipped and the
/// consequence observed.
struct Profiles {
    grep_eligible: bool,
    granularity: UnitLevel,
    is_searchable: bool,
}

impl Profiles {
    /// The scratchpad as the sealed contract declares it.
    fn as_sealed() -> Self {
        Profiles {
            grep_eligible: false,
            granularity: UnitLevel::Document,
            is_searchable: true,
        }
    }
}

impl CollectionProfiles for Profiles {
    fn profile(&self, _collection: Collection) -> CollectionProfile {
        CollectionProfile {
            is_searchable: self.is_searchable,
            granularity: self.granularity,
            grep_eligible: self.grep_eligible,
        }
    }
}

const SCOPE: [Collection; 1] = [Collection::Scratchpad];
const FTS5_ONLY: [LegMethod; 1] = [LegMethod::Trigram];

fn query(text: &str) -> String {
    format!(
        "SELECT TEXT note FROM {} WHERE q MATCH 'seam'{text}",
        Collection::Scratchpad.name()
    )
}

fn regex_query() -> String {
    format!(
        "SELECT REGEX note FROM {} WHERE q MATCH 'se+am'",
        Collection::Scratchpad.name()
    )
}

fn unsupported_key(error: QueryError) -> &'static str {
    match error {
        QueryError::Unsupported { capability_key, .. } => capability_key,
        other => panic!("expected a declared capability limit, got {other:?}"),
    }
}

#[test]
fn the_text_plan_is_one_leg_with_no_fusion() {
    let parsed = parse(&query("")).expect("parses");
    let plan = plan(&parsed, &SCOPE, &Profiles::as_sealed(), &FTS5_ONLY).expect("plans");

    assert_eq!(plan.legs.len(), 1);
    assert_eq!(plan.legs[0].method, LegMethod::Trigram);
    assert!(
        plan.fuse.is_none(),
        "N4 takes a VECTOR of legs and needs N17 dense scores; one leg never \
         reaches fusion, so announcing a fuse step would describe something that \
         did not happen"
    );
    assert!(
        plan.filters.is_empty(),
        "the match clause IS the leg, not a filter"
    );
    assert_eq!(plan.limit, 10, "the sealed default");
}

/// The `grep_eligible` axis, false, on the executed path.
#[test]
fn a_regex_plan_is_refused_because_the_profile_axis_is_false() {
    let parsed = parse(&regex_query()).expect("parses");
    let error = plan(&parsed, &SCOPE, &Profiles::as_sealed(), &FTS5_ONLY)
        .expect_err("grep is not eligible on this collection");
    assert_eq!(unsupported_key(error), "grep_eligible");
}

/// The same axis, flipped. If this returned the same refusal, the test above would
/// be measuring the absence of a regex concrete rather than the profile gate --
/// and §7.2's "exercised by being false" would be a story.
#[test]
fn flipping_grep_eligible_moves_the_refusal_to_availability() {
    let profiles = Profiles {
        grep_eligible: true,
        ..Profiles::as_sealed()
    };
    let parsed = parse(&regex_query()).expect("parses");
    let error =
        plan(&parsed, &SCOPE, &profiles, &FTS5_ONLY).expect_err("no regex concrete is built");
    assert_eq!(
        unsupported_key(error),
        "modes",
        "past the profile gate the refusal is availability's, which is a different \
         reason with a different cure"
    );
}

/// Semantic mode is refused by name rather than served by the text leg. Answering
/// a different question than the one asked is the failure class the surface
/// redesign exists to end.
#[test]
fn semantic_mode_is_refused_rather_than_quietly_served_as_text() {
    let parsed = parse(&format!(
        "SELECT note FROM {} WHERE q MATCH 'seam'",
        Collection::Scratchpad.name()
    ))
    .expect("parses -- the default mode is semantic");
    let error = plan(&parsed, &SCOPE, &Profiles::as_sealed(), &FTS5_ONLY)
        .expect_err("this build embeds nothing");
    assert_eq!(unsupported_key(error), "modes");
}

/// Two available legs would need fusion, and this build has none. The planner
/// refuses rather than emitting a plan the executor cannot run -- DP-6.3's
/// narrowing happens AT planning (`CONTRACTS.md`:2355).
#[test]
fn a_plan_that_would_need_fusion_is_refused_at_planning_time() {
    let parsed = parse(&format!(
        "SELECT note FROM {} WHERE q MATCH 'seam'",
        Collection::Scratchpad.name()
    ))
    .expect("parses");
    let both = [LegMethod::Dense, LegMethod::Sparse];
    let error = plan(&parsed, &SCOPE, &Profiles::as_sealed(), &both)
        .expect_err("two legs need a fusion step");
    assert_eq!(unsupported_key(error), "fusion");
}

/// A plan may narrow within the caller's scope, never widen it
/// (`CONTRACTS.md`:2343).
#[test]
fn a_source_outside_the_served_scope_is_refused() {
    let parsed = parse(&format!(
        "SELECT TEXT document FROM {} WHERE q MATCH 'seam'",
        Collection::Libraries.name()
    ))
    .expect("parses");
    let error = plan(&parsed, &SCOPE, &Profiles::as_sealed(), &FTS5_ONLY)
        .expect_err("libraries are outside this build's scope");
    assert_eq!(unsupported_key(error), "sources");
}

/// The `granularity` axis, also on the executed path: a document-level collection
/// carries no chunk, and "there is no such unit here" is a different answer from
/// "nothing matched".
#[test]
fn an_object_finer_than_the_granularity_is_refused() {
    let parsed = parse(&format!(
        "SELECT TEXT chunk FROM {} WHERE q MATCH 'seam'",
        Collection::Scratchpad.name()
    ))
    .expect("parses");
    let error = plan(&parsed, &SCOPE, &Profiles::as_sealed(), &FTS5_ONLY)
        .expect_err("scratchpad is document-level only");
    assert_eq!(unsupported_key(error), "objects");
}

#[test]
fn a_collection_the_deployment_cannot_search_is_refused() {
    let profiles = Profiles {
        is_searchable: false,
        ..Profiles::as_sealed()
    };
    let parsed = parse(&query("")).expect("parses");
    let error = plan(&parsed, &SCOPE, &profiles, &FTS5_ONLY).expect_err("not searchable here");
    assert_eq!(unsupported_key(error), "sources");
}

//! The plan's vocabularies, asserted where they touch another owner's.
//!
//! Two renderings of one name is how a vocabulary drifts, and this file is where
//! the two that overlap are pinned to each other rather than to the eye.

use serde_json::{json, Value};
use wqm_common::names::Collection;
use wqm_common::plan::{Leg, LegMethod, LegParams, Mode, ObjectKind, Plan, Source, SourceKind};

fn wire(value: impl serde::Serialize) -> Value {
    serde_json::to_value(value).expect("the plan types serialize")
}

/// `SourceKind` and N8's collection names are different registries that agree on
/// two words. The agreement is asserted, because nothing else would notice if a
/// rename touched one and not the other.
#[test]
fn the_source_kinds_that_name_collections_match_n8() {
    assert_eq!(wire(SourceKind::Rules), json!(Collection::Rules.name()));
    assert_eq!(
        wire(SourceKind::Scratchpad),
        json!(Collection::Scratchpad.name())
    );
}

/// The two that do NOT match are singular by design (`project`, not `projects`),
/// so the test above must not be read as "every kind equals a collection name".
#[test]
fn the_singular_source_kinds_are_deliberately_not_collection_names() {
    assert_ne!(
        wire(SourceKind::Project),
        json!(Collection::Projects.name())
    );
    assert_ne!(
        wire(SourceKind::Library),
        json!(Collection::Libraries.name())
    );
}

/// §3.4: ten keys, every one present, on every response. `fuse`/`expand`/`rerank`
/// are `null` VALUES, never absent keys -- the distinction L-1 rests on.
#[test]
fn the_echoed_plan_has_all_ten_keys_with_nulls_present() {
    let plan = Plan {
        mode: Mode::Text,
        object: ObjectKind::Note,
        sources: vec![Source {
            kind: SourceKind::Scratchpad,
            name: None,
        }],
        strict: false,
        legs: vec![Leg {
            method: LegMethod::Trigram,
            params: LegParams::none(),
        }],
        filters: Vec::new(),
        fuse: None,
        expand: None,
        rerank: None,
        limit: 10,
    };

    let wire = wire(&plan);
    let object = wire.as_object().expect("the plan is a JSON object");
    let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "expand", "filters", "fuse", "legs", "limit", "mode", "object", "rerank", "sources",
            "strict",
        ]
    );

    for key in ["fuse", "expand", "rerank"] {
        assert_eq!(wire[key], Value::Null, "`{key}` is present and null");
    }
    assert_eq!(wire["legs"][0]["method"], json!("trigram"));
    // All four param slots present, `null` where the method does not use them.
    assert_eq!(wire["legs"][0]["params"]["model"], Value::Null);
    assert_eq!(wire["legs"][0]["params"]["hops"], Value::Null);
}

/// The leg vocabulary is closed at five (§2.2a). Spelled out here so that adding
/// a sixth is a decision someone takes deliberately, against the sealed set.
#[test]
fn the_leg_methods_are_the_sealed_five() {
    let spelled: Vec<Value> = [
        LegMethod::Dense,
        LegMethod::Sparse,
        LegMethod::Trigram,
        LegMethod::Regex,
        LegMethod::Graph,
    ]
    .into_iter()
    .map(wire)
    .collect();
    assert_eq!(
        spelled,
        vec![
            json!("dense"),
            json!("sparse"),
            json!("trigram"),
            json!("regex"),
            json!("graph")
        ]
    );
}

//! N8 collection-name tests (PRD F-01). One positive assertion per canonical name
//! plus the reserved name and the enumeration contract.

use wqm_common::names::{collection_name, Collection, RESERVED_IMAGES_COLLECTION};

#[test]
fn projects_name_is_canonical() {
    assert_eq!(Collection::Projects.name(), "projects");
}

#[test]
fn libraries_name_is_canonical() {
    assert_eq!(Collection::Libraries.name(), "libraries");
}

#[test]
fn rules_name_is_canonical() {
    assert_eq!(Collection::Rules.name(), "rules");
}

#[test]
fn scratchpad_name_is_canonical() {
    assert_eq!(Collection::Scratchpad.name(), "scratchpad");
}

#[test]
fn free_function_matches_method_for_every_collection() {
    for c in Collection::ALL {
        assert_eq!(collection_name(c), c.name());
    }
}

#[test]
fn all_enumerates_the_four_canonical_collections_in_order() {
    assert_eq!(
        Collection::ALL,
        [
            Collection::Projects,
            Collection::Libraries,
            Collection::Rules,
            Collection::Scratchpad,
        ]
    );
}

#[test]
fn images_is_reserved_but_not_an_active_collection() {
    assert_eq!(RESERVED_IMAGES_COLLECTION, "images");
    // The reserved name is claimed, but is not one of the active collections.
    assert!(Collection::ALL
        .iter()
        .all(|c| c.name() != RESERVED_IMAGES_COLLECTION));
}

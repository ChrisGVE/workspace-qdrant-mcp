//! `P04-GT002-WO047` — `C-ty-per-store-versions`.
//!
//! Vocabulary only, so what there is to assert is *shape*: that ordering is
//! ordinal on the logical version (I2), that four independent slots really are
//! four independent slots, and that two absences the fragment makes explicitly —
//! no `Default`, no hidden fifth slot — are still absent. The absences are the
//! half worth writing down: a later reader who assumes they were oversights adds
//! a `Default` impl in one line, and the v0.1 `GraphStores::default()`
//! silent-disable is back.

use std::mem::size_of;
use wqm_store_write::versions::{PerStoreVersions, SchemaVersion};

/// I2: comparison is ordinal on the logical version, with no time base anywhere
/// in it. Asserted against the inner integers rather than against a second
/// hand-written expectation, so the property under test is "the ordering IS the
/// integer ordering", not "these two literals sort the way I typed them".
#[test]
fn ordering_is_ordinal_on_the_logical_version() {
    assert!(SchemaVersion(5) < SchemaVersion(49));
    assert!(SchemaVersion(49) > SchemaVersion(5));
    assert_eq!(SchemaVersion(8), SchemaVersion(8));
    assert_ne!(SchemaVersion(8), SchemaVersion(9));

    // Ord agrees with the inner integer across a spread that covers zero, the
    // documented baselines, and the top of the width.
    let probes = [0_u32, 1, 5, 8, 49, 1_000, u32::MAX];
    for &left in &probes {
        for &right in &probes {
            assert_eq!(
                SchemaVersion(left).cmp(&SchemaVersion(right)),
                left.cmp(&right),
                "SchemaVersion ordering must be the u32 ordering ({left} vs {right})"
            );
        }
    }
}

/// Divergence is information: a partially applied cross-engine migration leaves
/// the main store ahead and the other three behind, and the report has to be able
/// to say so. Each slot is read back distinctly — the state that must never
/// collapse to one scalar.
#[test]
fn a_partial_migration_reads_back_as_four_distinct_slots() {
    let mid_migration = PerStoreVersions {
        store: SchemaVersion(50),
        graph: SchemaVersion(5),
        search: SchemaVersion(8),
        blob: SchemaVersion(2),
    };

    assert_eq!(mid_migration.store, SchemaVersion(50));
    assert_eq!(mid_migration.graph, SchemaVersion(5));
    assert_eq!(mid_migration.search, SchemaVersion(8));
    assert_eq!(mid_migration.blob, SchemaVersion(2));

    assert!(
        mid_migration.store > mid_migration.graph
            && mid_migration.store > mid_migration.search
            && mid_migration.store > mid_migration.blob,
        "main-ahead / others-behind must be visible in the report itself"
    );

    // And the converged state is a different value — divergence is not merely
    // representable, it is distinguishable from having finished.
    let converged = PerStoreVersions {
        store: SchemaVersion(50),
        graph: SchemaVersion(6),
        search: SchemaVersion(9),
        blob: SchemaVersion(3),
    };
    assert_ne!(mid_migration, converged);
}

/// Exhaustiveness in the form a caller experiences it: all four fields are `pub`
/// and independently writable, so no slot can be supplied by accident and none is
/// hidden behind an accessor that could later start defaulting it.
#[test]
fn every_slot_is_public_and_independently_addressable() {
    let mut report = PerStoreVersions {
        store: SchemaVersion(1),
        graph: SchemaVersion(1),
        search: SchemaVersion(1),
        blob: SchemaVersion(1),
    };

    report.store = SchemaVersion(2);
    report.graph = SchemaVersion(3);
    report.search = SchemaVersion(4);
    report.blob = SchemaVersion(5);

    assert_eq!(report.store.0, 2);
    assert_eq!(report.graph.0, 3);
    assert_eq!(report.search.0, 4);
    assert_eq!(report.blob.0, 5);
}

/// The asserted absence: exactly four slots and nothing else in the value.
///
/// An `Option<SchemaVersion>` field (the "absent" encoding the fragment forbids),
/// a fifth store added without extending the closed set deliberately, or an enum
/// discriminant would all change this size. The layout is not the invariant — the
/// invariant is that the type carries four versions and no fifth thing — but the
/// size is the cheapest observable that any of those edits would break.
#[test]
fn the_report_carries_four_versions_and_nothing_else() {
    assert_eq!(
        size_of::<PerStoreVersions>(),
        4 * size_of::<SchemaVersion>(),
        "a fifth slot, an Option, or a discriminant would show up here"
    );
    assert_eq!(size_of::<SchemaVersion>(), size_of::<u32>());

    // The no-silent-default absence itself is asserted where the compiler can be
    // made to prove it: the `compile_fail` doctest on `PerStoreVersions` in
    // `src/versions.rs`, which runs under `cargo test`. A runtime test cannot
    // observe the non-existence of an impl.
}

//! `P04-GT001-WO012` — the committed scratchpad fixture, proven rather than asserted.
//!
//! Four things need evidence, and each has a reason beyond "the code runs":
//!
//! 1. **FTS5 exists in this build.** rusqlite has no `fts5` feature flag, so the
//!    claim rests on how bundled SQLite was compiled. A test is the only place
//!    that fact becomes checkable.
//! 2. **The fixture seeds both stores.** The SoT row is written explicitly; the
//!    derived index is written by the schema's trigger. If the trigger were
//!    missing, the SoT would still look right and the index would be empty.
//! 3. **The chokepoint reaches the fixture.** A fixture exempt from the write
//!    refusal would be the one hole in the guard.
//! 4. **The index cannot desynchronise from its content table.** This is the v0.1
//!    defect measured the day this was written — `code_lines_fts` is
//!    external-content over `code_lines` with no triggers, so deleted rows left
//!    their terms behind.

use wqm_common::names::{Collection, WriteTarget};
use wqm_store::{DerivedIndex, Fts5Index};
use wqm_test_harness::scratchpad_fixture::{in_memory, seed, SeedNote};

/// The opaque branch token. N3 owns the real `BRANCH_NONE_ID` (fixed-width, and
/// explicitly not an empty string); the store round-trips whatever it is given,
/// so the test supplies a stand-in rather than minting N3's constant.
const BRANCH: &str = "branch-none-placeholder";

fn notes() -> Vec<SeedNote<'static>> {
    vec![
        SeedNote {
            keep_id: "keep-1",
            branch_id: BRANCH,
            content: "the walking skeleton crosses every seam it claims",
        },
        SeedNote {
            keep_id: "keep-2",
            branch_id: BRANCH,
            content: "a guard that arrives first is free",
        },
    ]
}

#[test]
fn fts5_is_compiled_into_this_build() {
    let index = Fts5Index::open(in_memory().expect("schema installs"));
    assert!(
        index.available(),
        "FTS5 is absent from this SQLite build; the `bundled` feature is what \
         supplies it, and the read leg has no other concrete in this slice"
    );
}

#[test]
fn seeding_populates_the_sot_and_the_derived_index() {
    let mut conn = in_memory().expect("schema installs");
    let target = WriteTarget::of(Collection::Scratchpad);
    let written = seed(&mut conn, &target, &notes()).expect("seed succeeds");
    assert_eq!(written, 2);

    // The SoT half, asserted directly rather than through the index.
    let sot: i64 = conn
        .query_row("SELECT count(*) FROM note", [], |r| r.get(0))
        .expect("count");
    assert_eq!(sot, 2, "the SoT rows the fixture wrote");

    // The derived half, which no line of the fixture wrote -- the trigger did.
    let index = Fts5Index::open(conn);
    let hits = index
        .query(Collection::Scratchpad, "skeleton", 10)
        .expect("query runs");
    assert_eq!(hits.len(), 1, "one note matches `skeleton`");
    assert_eq!(hits[0].keep_id, "keep-1");
    assert_eq!(hits[0].branch_id, BRANCH);
}

#[test]
fn the_deployed_name_is_what_reaches_the_store() {
    let mut conn = in_memory().expect("schema installs");
    seed(
        &mut conn,
        &WriteTarget::of(Collection::Scratchpad),
        &notes(),
    )
    .expect("seed succeeds");

    let stored: String = conn
        .query_row("SELECT DISTINCT collection FROM note", [], |r| r.get(0))
        .expect("one collection value");
    assert_eq!(
        stored,
        Collection::Scratchpad.deployed_name(),
        "the SoT must carry the DEPLOYED name; storing the logical one would put \
         v0.2's rows under production's identity"
    );
    assert_ne!(
        stored,
        Collection::Scratchpad.name(),
        "the logical and deployed names must differ while running in parallel"
    );
}

/// The chokepoint, exercised from the direction that matters: a bare production
/// collection name cannot become a `WriteTarget`, so it cannot reach `seed` at
/// all. This is a compile-time shape assertion expressed at runtime — there is no
/// way to write the failing call, which is the point of the private field.
#[test]
fn a_production_collection_name_cannot_become_a_write_target() {
    assert!(
        WriteTarget::try_from_name("scratchpad").is_err(),
        "the unsuffixed production name must be refused"
    );
    assert!(
        WriteTarget::try_from_name(Collection::Scratchpad.deployed_name()).is_ok(),
        "the deployed name must be accepted"
    );
}

/// The v0.1 defect, made impossible rather than merely avoided.
///
/// v0.1's `search.db` declares `code_lines_fts` as external-content over
/// `code_lines` with **no triggers**, so deleting a content row left its terms in
/// the index — the index and its content table could disagree forever, and
/// nothing detected it. Here the delete trigger issues FTS5's `'delete'` command
/// inside the same transaction, so a writer that has never heard of FTS5 still
/// cannot leave a stale term behind.
#[test]
fn deleting_a_note_removes_it_from_the_derived_index() {
    let mut conn = in_memory().expect("schema installs");
    seed(
        &mut conn,
        &WriteTarget::of(Collection::Scratchpad),
        &notes(),
    )
    .expect("seed succeeds");

    // Deleting touches the SoT only. Nothing here mentions the index.
    conn.execute("DELETE FROM note WHERE keep_id = ?1", ["keep-1"])
        .expect("delete succeeds");

    let index = Fts5Index::open(conn);
    let hits = index
        .query(Collection::Scratchpad, "skeleton", 10)
        .expect("query runs");
    assert!(
        hits.is_empty(),
        "the term survived its content row -- exactly the v0.1 desync this \
         schema's triggers exist to prevent"
    );

    // The surviving note is untouched, so the delete was surgical rather than a
    // wholesale index drop that would also have made the assertion above pass.
    let survivors = index
        .query(Collection::Scratchpad, "guard", 10)
        .expect("query runs");
    assert_eq!(survivors.len(), 1, "the other note must still be indexed");
}

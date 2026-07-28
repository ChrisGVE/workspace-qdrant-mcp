//! The committed scratchpad fixture -- the seam the narrowed slice buys.
//!
//! # Why a fixture exists at all
//!
//! `CHARTER.md` §5A.1 chose a `note` → `query` round trip and stated what would
//! make that wrong: the write half pulling in so much of GT002/GT003 that the
//! skeleton becomes the kernel. `SCAFFOLD.md` §7.1 measured it at **24 borrowed
//! surfaces** and the condition tripped, so the charter's own fallback applied --
//! the slice narrows to a read over a fixture-seeded store, and the write half
//! moves to `P04-GT002`.
//!
//! This module is that fixture. It populates the SoT and lets the schema's
//! triggers populate the derived index, **without crossing the ingest pipeline**:
//! no queue item, no idempotency key, no embedding, no Qdrant. Those are the
//! surfaces the measurement counted, and their absence here is the narrowing made
//! literal rather than described.
//!
//! # It is subject to the refusal, not exempt from it
//!
//! The fixture writes, so the axis-D chokepoint applies to it. Seeding takes a
//! [`WriteTarget`] — the type whose private field and constructors make an
//! out-of-deployment collection name inexpressible — and stores
//! `WriteTarget::as_str()` in the `collection` column.
//!
//! This matters more than it looks. A fixture that took a `&str` collection would
//! be the one writer in the workspace exempt from the guard, and test data is
//! exactly where a bare `"scratchpad"` would slip in unnoticed. An exemption
//! without an enforced converse is a hole (`P04-GT001-WO006`'s law); the cheapest
//! way not to have the hole is not to take the exemption.

use rusqlite::Connection;
use wqm_common::names::WriteTarget;
use wqm_store::schema::SCHEMA_SQL;

/// One note to seed.
#[derive(Debug, Clone)]
pub struct SeedNote<'a> {
    /// Opaque keep identity. N3 owns the real vocabulary; the fixture does not
    /// mint it (see `wqm_store::Hit`).
    pub keep_id: &'a str,
    /// Opaque branch membership. Scratchpad collapses to N3's `BRANCH_NONE_ID`
    /// sentinel, whose value and fixed width are N3's to declare -- so callers
    /// pass whatever opaque token they are testing with rather than this crate
    /// inventing the constant.
    pub branch_id: &'a str,
    /// The text the derived index will hold.
    pub content: &'a str,
}

/// Apply the schema to a fresh connection. Idempotent, and safe to call on a
/// database that already has it.
pub fn install_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(SCHEMA_SQL)
}

/// Seed notes into a proven in-deployment collection.
///
/// Returns the number of rows written. The FTS5 index is **not** written here:
/// the schema's `note_ai` trigger does it, inside this transaction. That is the
/// design point rather than a convenience -- see [`crate::scratchpad_fixture`]
/// and `wqm_store::schema`.
pub fn seed(
    conn: &mut Connection,
    target: &WriteTarget,
    notes: &[SeedNote<'_>],
) -> rusqlite::Result<usize> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO note (collection, keep_id, branch_id, content) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for n in notes {
            stmt.execute(rusqlite::params![
                target.as_str(),
                n.keep_id,
                n.branch_id,
                n.content
            ])?;
        }
    }
    tx.commit()?;
    Ok(notes.len())
}

/// An in-memory database with the schema installed -- the ordinary entry point
/// for a read test.
///
/// In-memory rather than a temp file because the read leg has nothing to prove
/// about durability, and a database that cannot outlive the process is one fewer
/// thing to leak. The hermetic-endpoint discipline (CR-007 defect (a)) is
/// unaffected: this touches no network and no ambient path.
pub fn in_memory() -> rusqlite::Result<Connection> {
    let conn = Connection::open_in_memory()?;
    install_schema(&conn)?;
    Ok(conn)
}

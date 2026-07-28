//! The FTS5 concrete -- one of N41's four adapters, and the only one this slice
//! needs.

use rusqlite::Connection;
use wqm_common::names::Collection;

use crate::{DerivedIndex, Hit, StoreError};

/// N41's FTS5 adapter over the scratchpad SoT.
pub struct Fts5Index {
    conn: Connection,
}

impl Fts5Index {
    /// Open an index over an existing database.
    pub fn open(conn: Connection) -> Self {
        Fts5Index { conn }
    }

    /// Open an index over the database at `path`.
    ///
    /// This exists so a *consumer* -- the S2 bin, which composes the read pipeline
    /// in-process -- does not have to link SQLite itself. How a derived index is
    /// opened is the index's own business, and a client that linked `rusqlite`
    /// directly would be one refactor away from issuing its own SQL against the
    /// SoT, which is the boundary this crate's module note exists to hold.
    pub fn open_path(path: &std::path::Path) -> Result<Self, StoreError> {
        Ok(Fts5Index::open(Connection::open(path)?))
    }

    /// The connection, for callers that own the database's lifecycle (schema
    /// creation, fixtures). The read leg itself never mutates.
    pub fn connection(&self) -> &Connection {
        &self.conn
    }
}

impl DerivedIndex for Fts5Index {
    fn name(&self) -> &'static str {
        "fts5"
    }

    /// Probe, do not assume.
    ///
    /// rusqlite exposes no `fts5` feature flag -- FTS5 is compiled in by the
    /// `bundled` SQLite build, which is a property of how this binary was built
    /// and of nothing the type system can see. So availability is asked of
    /// SQLite itself: create a temporary FTS5 table and observe whether the
    /// module exists. A Cargo feature list is documentation; this is evidence.
    fn available(&self) -> bool {
        self.conn
            .execute_batch("CREATE VIRTUAL TABLE temp.__fts5_probe USING fts5(x);")
            .and_then(|()| self.conn.execute_batch("DROP TABLE temp.__fts5_probe;"))
            .is_ok()
    }

    fn query(
        &self,
        collection: Collection,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Hit>, StoreError> {
        if !self.available() {
            return Err(StoreError::Unavailable { index: "fts5" });
        }

        // `note_fts` is external-content, so the join back to `note` is how a hit
        // reaches its identity columns -- the index itself stores no copy of them.
        // Ordering is FTS5's own `rank` (ascending: better matches sort first).
        let mut stmt = self.conn.prepare(
            "SELECT n.keep_id, n.branch_id, n.content \
               FROM note_fts f \
               JOIN note n ON n.rowid = f.rowid \
              WHERE note_fts MATCH ?1 AND n.collection = ?2 \
              ORDER BY f.rank \
              LIMIT ?3",
        )?;

        let rows = stmt.query_map(
            rusqlite::params![query, collection.deployed_name(), limit as i64],
            |row| {
                Ok(Hit {
                    keep_id: row.get(0)?,
                    branch_id: row.get(1)?,
                    content: row.get(2)?,
                })
            },
        )?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StoreError::from)
    }
}

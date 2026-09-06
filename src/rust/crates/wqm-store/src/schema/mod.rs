//! The scratchpad SoT table and its FTS5 derived index.
//!
//! # Why the schema lives here, and what that owes
//!
//! ADR-003 homes SQLite schema, creation and migration in the **daemon**. Nothing
//! daemon-side exists in this build, so the shape is declared here — where the
//! reader that depends on it lives — rather than being spelled twice. When
//! `wqm-store-write` and N19's migration surface arrive they take ownership of
//! creation; this constant becomes the read-side's assertion about what it reads.
//! Recorded as debt in `SCAFFOLD.md` §7 rather than left implicit.
//!
//! # The triggers are the point
//!
//! `note_fts` is an **external-content** FTS5 table: it stores no copy of the
//! text, indexing `note` by rowid. That arrangement has a failure mode worth
//! naming, because it was observed live on the v0.1 server the same day this was
//! written: v0.1's `search.db` declares `code_lines_fts` as external-content over
//! `code_lines` **with no triggers at all**, so a row deleted from the content
//! table left its terms in the index. The index and its content table could
//! disagree indefinitely, and nothing detected it.
//!
//! FTS5 does not synchronise itself — the application must issue the matching
//! `'delete'` command *before* the content row goes, because the delete is
//! computed from the old text. Leaving that to caller discipline is exactly the
//! DP-8 anti-pattern: one forgotten statement and the index rots silently.
//!
//! So the synchronisation is **structural**: three triggers own it, inside the
//! same transaction as the write that fires them. A writer that never heard of
//! FTS5 cannot desynchronise the index, because it is not the writer's job.
//!
//! # Why the tokenizer is `trigram` (`P04-GT001-WO013`)
//!
//! `WO012` declared the table with FTS5's default word tokenizer. `WO013` then had
//! to name this leg in the plan the response carries back, and the sealed leg
//! vocabulary (MCP-SURFACE.md §2.2a) closes `leg.method` at
//! `dense|sparse|trigram|regex|graph` -- with `trigram` being the literal-text leg
//! `TEXT` mode compiles to (§1.6: "literal substring/phrase, FTS5 trigram").
//!
//! A response announcing `{"method":"trigram"}` over a word-token index would be a
//! false statement in the one field an agent reads to learn what ran, and the
//! failure would be invisible exactly where it matters: `MATCH 'kelet'` finds
//! nothing under a word tokenizer and finds `skeleton` under a trigram one, so the
//! declaration and the behaviour disagree only on the queries a caller writes
//! *because* the declaration said substring. Changing the tokenizer is the
//! one-token fix; relabelling the leg was not available, because the vocabulary is
//! closed and sealed.
//!
//! The cost is stated rather than discovered: SQLite's trigram tokenizer indexes
//! three-character sequences, so a MATCH term shorter than three characters
//! matches nothing at all. That is a real narrowing of the query surface and it
//! belongs to the read leg's declared limits, not to a footnote.

/// The schema families this crate declares, one module per family.
///
/// Each module holds DDL and nothing else: a `&str` constant per table, plus
/// the doc comment that says which fill fragment the shape was transcribed
/// from. `P04-GT002` is vocabulary only, so no module here opens a
/// connection, migrates, or writes a row.
pub mod cas;
pub mod classification;

/// The scratchpad SoT table, its FTS5 derived index, and the triggers that keep
/// the two in agreement. Idempotent — every object is `IF NOT EXISTS`.
///
/// `collection` holds the **deployed** name (`scratchpad-v2` while running in
/// parallel), never the logical one. It is a column rather than an implied
/// property of the file so the axis-D refusal reaches the SoT too, not only
/// Qdrant: a row can only be written through a proven
/// [`WriteTarget`](wqm_common::names::WriteTarget).
pub const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS note (
    rowid       INTEGER PRIMARY KEY,
    collection  TEXT NOT NULL,
    keep_id     TEXT NOT NULL,
    branch_id   TEXT NOT NULL,
    content     TEXT NOT NULL,
    UNIQUE (collection, keep_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS note_fts USING fts5(
    content,
    content='note',
    content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS note_ai AFTER INSERT ON note BEGIN
    INSERT INTO note_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS note_ad AFTER DELETE ON note BEGIN
    INSERT INTO note_fts(note_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS note_au AFTER UPDATE ON note BEGIN
    INSERT INTO note_fts(note_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO note_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"#;

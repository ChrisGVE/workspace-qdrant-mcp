//! N36's library families in storedb — document metadata, keywords, the reference
//! graph, and source precedence.
//!
//! Transcribed from the sealed fill fragments under
//! `P04-build/T1-kernel/fill/anchor/` (`P02-GT004`) — `A-tbl-lib-doc-metadata`,
//! `A-tbl-lib-keywords`, `A-tbl-lib-references`, `A-tbl-ref-source-authority`.
//! Vocabulary only: no module here opens a connection, migrates, or writes a row.
//!
//! # These DDLs name a table this crate does not declare
//!
//! `document_id` and `src_doc` are the library item's minted `item_id` in
//! `items_libraries` (GT002 §E), and the fragments declare the foreign keys to it
//! literally. `A-tbl-items` is **out of `P04-GT002`'s scope**, so that table
//! arrives with the work order that owns it. The references are transcribed as
//! written rather than dropped: silently removing an FK the fragment declares
//! would make this declaration disagree with its own source, and the disagreement
//! would be invisible — SQLite resolves a foreign key at DML time, not at
//! `CREATE TABLE`, so the DDL applies either way.

/// Per-document library metadata, keyed on the item's own surrogate.
///
/// `document_id` is both the PK and the FK into `items_libraries`, so the row is
/// scoped by that reference — there is no tenant or collection column, and none is
/// needed. DATA-01: deleting the library item takes its metadata with it.
///
/// `norm_key` is N37's ONE canonical normaliser (FP-2) and is indexed, because
/// matching a reference to a document is a lookup on the normalised form rather
/// than on any of the raw fields.
///
/// `authors` holds a JSON array, and `title_method` records *how* a title was
/// determined — provenance the store keeps and never interprets. Column types and
/// index tuning beyond this are P02/PRD detail; the columns and index intent are
/// architecture (ARCH §5.3 family 1).
pub const LIB_DOC_METADATA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS lib_doc_metadata (
    document_id          INTEGER PRIMARY KEY
                           REFERENCES items_libraries(item_id) ON DELETE CASCADE,
    kind                 TEXT NOT NULL,
    title                TEXT,
    authors              TEXT,
    year                 INTEGER,
    isbn                 TEXT,
    doi                  TEXT,
    publication          TEXT,
    pub_ref              TEXT,
    page                 TEXT,
    curated_class_raw    TEXT,
    title_method         TEXT,
    content_start_offset INTEGER,
    norm_key             BLOB
);
CREATE INDEX IF NOT EXISTS ix_lib_doc_metadata_norm_key ON lib_doc_metadata(norm_key);
"#;

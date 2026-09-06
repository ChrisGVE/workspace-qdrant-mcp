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

/// Per-document keywords with ONE score column.
///
/// The single `score` is deliberate. v0.1 carried three — `semantic_score`,
/// `lexical_score`, `stability_count` — and they were artifacts of the extractor
/// that produced them; they retire with it under B2, and the new N54 extractor
/// defines what a score means (§11.2 J-7). Carrying the three forward would have
/// preserved a shape whose meaning had already been retired.
///
/// The `keyword` index exists because induction reads the whole corpus as a
/// bounded stream, not by point lookup — an index chosen for the access pattern
/// that exists rather than the one a keyword column suggests.
pub const LIB_KEYWORDS_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS lib_keywords (
    document_id  INTEGER NOT NULL,
    keyword      TEXT    NOT NULL,
    score        REAL    NOT NULL,
    PRIMARY KEY (document_id, keyword),
    FOREIGN KEY (document_id) REFERENCES items_libraries(item_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS ix_lib_keywords_keyword ON lib_keywords(keyword);
"#;

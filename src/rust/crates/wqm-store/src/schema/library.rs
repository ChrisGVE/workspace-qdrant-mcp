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

/// The reference graph — one row per citation, resolved or not.
///
/// Raw reference fields are per-kind and sparse-tolerant: a book row fills
/// `title`/`author`/`section`/`year`/`isbn`, an article row
/// `publication`/`ref`/`page`, a website row `url`. They share one table because a
/// reference's kind is data, not a schema decision.
///
/// The two document references differ in their delete rule, and the difference is
/// the design. `src_doc` — the CITING document — cascades: a reference has no
/// meaning without the document that made it. `resolved_doc` — the resolution
/// TARGET — is `ON DELETE SET NULL`: deleting the cited document does not delete
/// the citation, it *unresolves* it, returning the row to the orphan wait-set it
/// came from.
///
/// That wait-set is the partial index: a reference is pending **iff**
/// `resolved_doc IS NULL`, so `ix_lib_references_orphan` indexes exactly the rows
/// awaiting resolution and shrinks as they resolve. A full index on `norm_key`
/// would answer the same query and carry every resolved row forever.
pub const LIB_REFERENCES_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS lib_references (
    ref_id                INTEGER PRIMARY KEY,
    src_doc               INTEGER NOT NULL
                            REFERENCES items_libraries(item_id) ON DELETE CASCADE,
    kind                  TEXT NOT NULL,
    title                 TEXT,
    author                TEXT,
    section               TEXT,
    year                  INTEGER,
    isbn                  TEXT,
    publication           TEXT,
    ref                   TEXT,
    page                  TEXT,
    url                   TEXT,
    norm_key              BLOB,
    resolved_doc          INTEGER
                            REFERENCES items_libraries(item_id) ON DELETE SET NULL,
    resolution_method     TEXT,
    resolution_confidence REAL,
    ambiguous             INTEGER
);
CREATE INDEX IF NOT EXISTS ix_lib_references_orphan
    ON lib_references(norm_key) WHERE resolved_doc IS NULL;
CREATE INDEX IF NOT EXISTS ix_lib_references_resolved ON lib_references(resolved_doc);
CREATE INDEX IF NOT EXISTS ix_lib_references_src ON lib_references(src_doc);
"#;

/// Source precedence — two columns, no index, and both of those are decisions.
///
/// The row carries no tenant, collection or document key because precedence is a
/// property of the *source* (`arxiv`, `doi`, `isbn`, `url`), cutting across the
/// whole reference graph. A scope column would make precedence answerable
/// differently in two places, which is the one thing a precedence table must not
/// permit.
///
/// No secondary index, deliberately: the table is tiny and is read by full scan
/// whenever candidate sources must be ordered. The ordering convention that `rank`
/// expresses — ascending or descending — is P02 detail and is not decided here.
pub const REF_SOURCE_AUTHORITY_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS ref_source_authority (
    source  TEXT PRIMARY KEY,
    rank    INTEGER NOT NULL
);
"#;

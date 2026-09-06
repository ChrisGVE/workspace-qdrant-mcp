//! N52's taxonomy and classification families in storedb.
//!
//! Transcribed from the sealed fill fragments under
//! `P04-build/T1-kernel/fill/anchor/` (`P02-GT004`) — `A-tbl-taxonomy-category`,
//! `A-tbl-category-keyword`, `A-tbl-doc-classification`,
//! `A-tbl-curation-decision`. Vocabulary only: no module here opens a connection,
//! migrates, or writes a row.
//!
//! # The family is global, and reads that way on purpose
//!
//! None of these tables carries a tenant or collection column. The taxonomy is
//! one cross-tenant backbone on the `libraries` axis (N35), so a scope column
//! would not narrow a query — it would fork the backbone.
//!
//! # Retention is schema, not discipline
//!
//! Rounds coexist so induction can append-then-flip, and the *current + previous*
//! retention bound is held by `ON DELETE CASCADE` off `taxonomy_category` rather
//! than by a GC routine that remembers to visit three tables. Round-GC deletes
//! the superseded round's categories; its profiles and assignments go with them,
//! mechanically. These are composition deletes of a round's own component rows —
//! not the FP-1 cross-entity cascade, which orders consumers before their target.

/// The per-round category table — the taxonomy backbone.
///
/// Two identifiers, deliberately: `category_id` is a per-round surrogate rowid,
/// and `category_key` is the **round-stable lineage key** minted by N52. Downstream
/// rows reference the lineage key, so a cross-round query needs no id mapping.
///
/// The two `UNIQUE` constraints do different jobs. `UNIQUE(name, round)` scopes
/// name uniqueness to a round, which is what lets two rounds coexist during
/// append-then-flip. `UNIQUE(category_key, round)` is the candidate key
/// `DOC_CLASSIFICATION_SQL` names in its composite foreign key — without it that
/// FK cannot be declared at all.
///
/// `parent_id` references a `category_id` **in the same round only**. Induction
/// emits a cluster dendrogram, a forest acyclic by construction, so no cyclic or
/// cross-round parent can arise.
pub const TAXONOMY_CATEGORY_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS taxonomy_category (
    category_id  INTEGER PRIMARY KEY,
    category_key BLOB    NOT NULL,
    name         TEXT    NOT NULL,
    description  TEXT,
    level        INTEGER NOT NULL,
    parent_id    INTEGER,
    is_frozen    INTEGER NOT NULL,
    is_utility   INTEGER NOT NULL,
    round        INTEGER NOT NULL,
    UNIQUE (name, round),
    UNIQUE (category_key, round)
);
CREATE INDEX IF NOT EXISTS ix_taxonomy_category_parent ON taxonomy_category(parent_id);
CREATE INDEX IF NOT EXISTS ix_taxonomy_category_key ON taxonomy_category(category_key);
"#;

/// The per-category keyword profile — one weight per (category, keyword).
///
/// The composite PK forbids a duplicate keyword inside one profile, which is what
/// N53 emits anyway: a per-cluster keyword→weight map cannot carry a duplicate
/// key. Stating it in the schema makes the store agree with the producer instead
/// of trusting it.
///
/// `weight` is an **opaque stored score**. N52 stores it; what a weight means
/// belongs to N53/N54, and the store never interprets or branches on it.
///
/// The `ON DELETE CASCADE` is a composition delete: a profile belongs to its
/// category row, so the retention bound holds for profiles by the same mechanism
/// that holds it for categories.
pub const CATEGORY_KEYWORD_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS category_keyword (
    category_id INTEGER NOT NULL
        REFERENCES taxonomy_category(category_id) ON DELETE CASCADE,
    keyword     TEXT    NOT NULL,
    weight      REAL    NOT NULL,
    PRIMARY KEY (category_id, keyword)
);
CREATE INDEX IF NOT EXISTS ix_category_keyword_keyword ON category_keyword(keyword);
"#;

/// Document→category assignments, keyed on the lineage key rather than the
/// per-round surrogate.
///
/// The composite FK targets `taxonomy_category`'s `UNIQUE(category_key, round)`,
/// which makes an assignment to an absent category in its round **structurally
/// impossible** (DATA-04) — the store half of a two-sided defense whose other half
/// is N53 never proposing one.
///
/// The partial `UNIQUE(document_id, round) WHERE is_primary` holds *one primary
/// per document per round*. N53 already guarantees it by construction (highest
/// Jaccard, ties broken by the proposal's total order), so the store's rejection
/// is defense in depth — which is the point: the guarantee survives a future
/// producer that does not carry it.
///
/// `method` and `confidence` are opaque stored values; the store records them and
/// never branches on `method`.
pub const DOC_CLASSIFICATION_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS doc_classification (
    document_id  INTEGER NOT NULL,
    category_key BLOB    NOT NULL,
    round        INTEGER NOT NULL,
    confidence   REAL    NOT NULL,
    method       TEXT    NOT NULL,
    is_primary   INTEGER NOT NULL,
    PRIMARY KEY (document_id, category_key, round),
    FOREIGN KEY (category_key, round)
        REFERENCES taxonomy_category(category_key, round) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_doc_primary
    ON doc_classification(document_id, round) WHERE is_primary;
CREATE INDEX IF NOT EXISTS ix_doc_classification_doc ON doc_classification(document_id);
"#;

//! The recorded-file family in storedb (N59 custody).
//!
//! Transcribed from the sealed fill fragment
//! `P04-build/T1-kernel/fill/anchor/A-tbl-file-entry.md` (`P02-GT004`).
//! Vocabulary only: nothing here opens a connection, migrates, or writes a row.

/// One row per physical file instance, keyed on the instance — never on content.
///
/// # The key is the whole declaration
///
/// `PRIMARY KEY (keep_id, relpath)` is pinned by rev12 DATA MF-1, and the
/// alternative it rules out is the reason it is pinned. Keying on `file_hash`
/// would collapse N same-content files onto ONE name and relpath, and deport would
/// then silently reproduce one of them and lose the rest — the exact violation
/// AGP-12's "structural" claim is premised on being impossible. Many rows sharing
/// one `file_hash` is not a flaw to be normalised away: that sharing **is** the
/// dedup case the CAS exists to create.
///
/// # Recorded truth, not filesystem truth
///
/// `name`, `relpath` and `file_hash` never live only in the physical filesystem.
/// They are recorded here so deport is never missing its inputs, whatever custody
/// mode the keep declares. The recorded `file_hash` is deport's verification
/// *referent*: comparing reproduced bytes against the source bytes would be
/// tautological, so the comparison is against what was recorded at ingest.
///
/// `custody` stores the keep's declared mode at record time as an integer
/// discriminant. The `CustodyMode` type it corresponds to is `wqm-common`'s
/// (`P04-GT002-WO042`); this crate stores the value and does not interpret it.
pub const FILE_ENTRY_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS file_entry (
    keep_id   BLOB NOT NULL,
    relpath   TEXT NOT NULL,
    name      TEXT NOT NULL,
    file_hash BLOB NOT NULL,
    custody   INTEGER NOT NULL,
    PRIMARY KEY (keep_id, relpath)
);
CREATE INDEX IF NOT EXISTS ix_file_entry_hash ON file_entry(file_hash);
"#;

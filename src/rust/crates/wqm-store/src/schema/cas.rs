//! The CAS substrate's storedb tables (N59).
//!
//! Transcribed from the sealed fill fragment
//! `P04-build/T1-kernel/fill/anchor/A-tbl-cas-entry.md` (`P02-GT004`), which is
//! the declaration of record; `anchor.shape` is a one-line abstract and is not.
//!
//! # Why a DDL constant and not a migration
//!
//! ADR-003 homes SQLite schema creation and migration in the daemon, and
//! `P04-GT002` is vocabulary only — zero behavior. So this is the *shape*, stated
//! once, in the crate `ARCHITECTURE.md` §9.1 homes it in. Nothing here opens a
//! connection or applies anything; the write leg that will own creation lands in
//! `wqm-store-write` with the slice that needs it.
//!
//! # The CAS has no boundary, and that is the whole point
//!
//! There is deliberately **no keep or collection column**. The same bytes under
//! several keeps are ONE entry, and that sharing *is* the global dedup the CAS
//! exists to create. A scope column here would silently partition the dedup and
//! the loss would be invisible: every query still answers, just against a smaller
//! world than the one the design promised.

/// The immutable content-addressed entry table — one row per unique content hash.
///
/// `file_hash` is a raw-byte 32-byte SHA256 BLOB, carrying the INTEGER-surrogate
/// discipline every identity key has under DP-ID4 (GT002 lock §A). `held`
/// discriminates hash-only (`0`) from blob-held (`1`), and the `CHECK` makes the
/// discriminant **exact** rather than conventional: a blob-held row without a
/// locator, or a hash-only row carrying one, is rejected by the store instead of
/// being caught — or not caught — by a writer that remembered to look.
///
/// `created_at`, `size` and any further columns are P03 detail per the fragment;
/// what is pinned here is the key, the discriminant, and the invariant between
/// them.
pub const CAS_ENTRY_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS cas_entry (
    file_hash    BLOB PRIMARY KEY,
    held         INTEGER NOT NULL,
    blob_locator BLOB,
    CHECK ((held = 1) = (blob_locator IS NOT NULL))
);
"#;

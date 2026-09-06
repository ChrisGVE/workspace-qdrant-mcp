//! N19's per-store schema-version report (`C-ty-per-store-versions`).
//!
//! Transcribed from the sealed fill fragment
//! `project-notes/wqm-0.2/P04-build/T1-kernel/fill/anchor/A-per-store-versions.md`.
//! Vocabulary only: nothing here reads a `schema_version` record, opens a store,
//! or compares a version against the binary.
//!
//! # Purpose
//!
//! The product persists across four heterogeneous engines -- the SQLite main
//! store, the graph store, the search/FTS store and blob-Qdrant -- and they
//! cannot participate in one atomic commit. So "what version is this package at?"
//! has no single-scalar answer, and answering it with one anyway is precisely
//! what produced the SEC-N3 false-"Ready" hole: a partially applied cross-engine
//! migration reported a healthy version and the process served.
//!
//! [`PerStoreVersions`] is the shape that makes the four answers separately
//! visible and separately type-checked at every consumer. `A-if-N19.versions()`
//! produces it; `guard_boot` consumes it and compares every slot (plus the
//! registry fingerprint, ARCH §6.3) against the binary, refusing to serve on any
//! mismatch. N46 reads it on the boot / auto-restart path, N48 through the serve
//! gate (ARCH §3.3, §4.4).
//!
//! # Sources
//!
//! - contracts N19 -- per-store `schema_version` records; `versions() ->
//!   PerStoreVersions`; the guard checks ALL before serve; build-path baselines
//!   main v49 / graph v5 / search v8 / blob empty.
//! - ARCH §3.3 + §4.4 (N48/N46 consumption), §6.3 (`schema_version` per store
//!   with the all-store boot guard), §9.1 (`wqm-store-write` homing).
//! - NEXUSES N19 (four engines cannot single-atomic-commit; partial-failure
//!   divergence must be detectable, never reported Ready) and §4.10 invariant 2
//!   (I2).

/// One store's schema version: an ordered *logical* version.
///
/// I2 (NEXUSES §4.10 invariant 2): this is a sequencing source and never a wall
/// clock. Ordering is ordinal on the integer -- no timestamp participates in a
/// comparison, and none may be smuggled in later by widening this type into a
/// date. The only question it answers is "is this store ahead of, level with, or
/// behind that one".
///
/// # The width is delegated latitude
///
/// The fragment's `## Open` records the integer width as P03 latitude: the
/// contract fixes the four slots and their store identities, not the arithmetic
/// type underneath. `u32` is the implementer's choice, taken because the
/// baselines are two-digit consolidation counters (main v49, graph v5, search
/// v8) and a per-store migration count that overflows 4.29 billion is not a
/// scenario this system has. It is unsigned because a version behind zero has no
/// meaning; a widening later is a mechanical change, since nothing outside this
/// type is written against the width.
///
/// Nothing is stored *about* the version here -- no origin, no applied-at, no
/// store identity. Which store a version belongs to is carried by the field it
/// occupies in [`PerStoreVersions`], not by a tag inside the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SchemaVersion(pub u32);

/// All four store schema versions (F6) -- the in-memory read projection of the
/// persisted per-store `schema_version` records (table names from N8).
///
/// The durable truth is NOT this struct: it is the records N19 keeps, with the
/// registry fingerprint persisted alongside (ARCH §6.3). This is the report
/// `versions()` returns and `guard_boot` evaluates, as-of the enclosing read.
///
/// # Baselines
///
/// Documented, deliberately not encoded. The build path's consolidation
/// baselines are main **v49**, graph **v5**, search **v8**, and blob **empty**.
/// They are prose here because "empty" is not a number: minting a constant for
/// it -- `0`, or a sentinel -- would decide something the contract does not, and
/// would put a fifth meaning inside a type whose whole job is to carry four
/// versions and nothing else. When N19's reader lands it will have to say what
/// an empty blob store reads as; this row does not pre-empt that.
///
/// # Invariants
///
/// - **Closed and exhaustive.** Exactly four slots, covering EVERY store. A
///   fifth store enters the engine only by extending this type, so the boot
///   guard's coverage grows with the storage topology by construction rather
///   than by somebody remembering to add a check.
/// - **No optional slots.** A store whose version record cannot be read has no
///   "absent" encoding here. The read fails typed upstream; construction never
///   defaults a missing store. This is why the type derives no [`Default`], why
///   no field is an `Option`, and why there is no constructor that could fill a
///   slot the caller did not supply -- the v0.1 `GraphStores::default()`
///   silent-disable, which turned an unreadable store into a healthy-looking
///   zero, is the named anti-pattern.
/// - **Read-only projection.** Producing the value mutates nothing.
/// - **Divergence is information.** Unequal per-store progress is a legal,
///   meaningful state -- a resumable partial migration. After a crash mid-
///   migration the report shows the main store bumped while graph/search/blob
///   lag: exactly the detectable divergence `migrate_to_latest`'s resume keys
///   on, and the state in which Ready is never reported. The type must expose
///   it, never collapse it to one scalar.
///
/// # The absent `Default`, demonstrated
///
/// The no-optional-slots invariant is a compile-time fact, not a comment:
///
/// ```compile_fail,E0599
/// use wqm_store_write::versions::PerStoreVersions;
/// // No `Default` impl exists, and none may be added: a defaulted value would
/// // report four healthy stores that nobody read.
/// let _ = PerStoreVersions::default();
/// ```
///
/// Every value is built by naming all four stores:
///
/// ```
/// use wqm_store_write::versions::{PerStoreVersions, SchemaVersion};
///
/// // Illustrative values, not the baselines -- in particular the blob slot is
/// // NOT shown at its documented "empty", because what empty reads as is N19's
/// // to decide when its reader lands.
/// let report = PerStoreVersions {
///     store: SchemaVersion(49),
///     graph: SchemaVersion(5),
///     search: SchemaVersion(8),
///     blob: SchemaVersion(3),
/// };
/// assert!(report.store > report.search);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerStoreVersions {
    /// SQLite storedb -- the main store (consolidation baseline v49).
    pub store: SchemaVersion,
    /// The graph store (baseline v5).
    pub graph: SchemaVersion,
    /// The search / FTS store (baseline v8).
    pub search: SchemaVersion,
    /// blob-Qdrant (baseline empty -- see the type's Baselines section).
    pub blob: SchemaVersion,
}

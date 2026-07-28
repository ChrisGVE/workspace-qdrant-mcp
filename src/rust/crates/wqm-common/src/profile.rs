//! N35's collection profile, at the READ face and at the axes this slice reads.
//!
//! # The face, and why it is a trait
//!
//! N35 has two faces (`CONTRACTS.md`:481-489): an injected READ trait a kernel
//! crate consults, and a glue-side registry that owns the table. The split is not
//! stylistic -- :585-586 makes it structural, because **no kernel crate may link
//! `wqm-conventions`** and `ci/guard_link_closure.py` enforces that direction. So
//! the planner receives a [`CollectionProfiles`] and never names the table; the
//! table is supplied by the composition root, which in this build is the S2 bin.
//!
//! # Three axes, and the budget they come out of
//!
//! `SCAFFOLD.md` §7.2 measured the narrowed read leg at four borrowed surfaces and
//! priced this one at "`CollectionProfile` -- two axes only (`is_searchable`,
//! `granularity`)". The same section's closing paragraph then relies on a third,
//! `grep_eligible`, "exercised by being false", and `P04-GT001-WO013`'s acceptance
//! requires it by name. Two axes and three axes cannot both be the measurement, so
//! §7.2's table is corrected to three rather than this module quietly taking one
//! more than its budget. The other six axes (`embedding_model`, `sparse_strategy`,
//! `dedup_scope`, `has_tags`, `requires_write_capability`, `taxonomy`) are absent,
//! not stubbed.
//!
//! # Keyed on `Collection`, and the debt that carries
//!
//! The sealed read face is keyed on `OpaqueCollectionId` (:483-486) so that a
//! kernel crate reads a profile without naming an up-crate type. Here the key is
//! [`Collection`], which lives in this same floor crate -- so the link direction
//! the opaque id protects is not violated. What is deferred is the *re-key* itself,
//! and it is deferred rather than approximated: minting `OpaqueCollectionId` is
//! item 4 of the 24 surfaces §7.1 counted, and it belongs to N35's own slice.
//! Recorded in `SCAFFOLD.md` §7.

use crate::names::Collection;

/// N40's addressable-unit ladder (`CONTRACTS.md`:1089-1090), which the
/// `granularity` axis enumerates from.
///
/// The ladder is `{document, section, chunk}` and the contract states outright
/// that rules and scratchpad are document-level only -- so a scratchpad note is a
/// `document`-level unit even though the surface calls the *object* a `note`.
/// Those are two vocabularies (unit level, result object), not one word used
/// twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitLevel {
    /// The whole document is the addressable unit.
    Document,
    /// A section within a document.
    Section,
    /// A retrieval chunk.
    Chunk,
}

impl UnitLevel {
    /// The wire spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            UnitLevel::Document => "document",
            UnitLevel::Section => "section",
            UnitLevel::Chunk => "chunk",
        }
    }
}

/// What a collection does, at the axes this build reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectionProfile {
    /// Whether search may reach this collection at all.
    pub is_searchable: bool,
    /// The finest unit this collection is addressable at.
    pub granularity: UnitLevel,
    /// Whether the grep concrete may dispatch here (`CONTRACTS.md`:565-566).
    ///
    /// GT002 §B's rule is "no grep on library/scratchpad/rules", so on this
    /// build's one collection the axis is **false**, and the planner refuses a
    /// regex plan before ever asking whether a grep concrete exists. The axis is
    /// therefore exercised by being false, which is a stronger statement than not
    /// touching it: the gate is on the executed path, not merely declared.
    pub grep_eligible: bool,
}

/// The injected READ face (`CONTRACTS.md`:483-484).
///
/// A kernel crate takes `&dyn CollectionProfiles` and asks; it never constructs
/// the table, and it cannot, because the table's home is the glue side it may not
/// link.
pub trait CollectionProfiles {
    /// The profile of one collection. Total by construction: the collection set is
    /// closed (ADR-001), so every key has a row and there is no "unknown
    /// collection" case for a caller to mishandle.
    fn profile(&self, collection: Collection) -> CollectionProfile;
}

//! `RulesWriteCap` — the unforgeable authority that gates writes to the rules
//! collection.
//!
//! Transcribed from the sealed fill fragment
//! `P04-build/T4-glue/fill/anchor/A-rules-write-cap.md` (`P04-GT002-WO074`,
//! concrete `C-ty-rules-write-cap`). Vocabulary only: this file declares a type
//! and nothing else.
//!
//! # Why the type exists at all
//!
//! F15 routes rules CRUD through the *same* public N5 drain that ingestion uses,
//! so an origin field carried in the payload would be forgeable by anyone able to
//! call `N5.enqueue` — a watched-folder document, a scratchpad write, an MCP
//! `store` with `collection: rules`. The capability makes "only user-authored
//! writes reach the rules collection" structural instead of a payload convention;
//! without it, ingested content could smuggle a rules row through the generic
//! enqueue, which is a persistent prompt-injection vector.
//!
//! # What is ABSENT here, rather than stubbed
//!
//! - **The mint.** [`RulesWriteCap`] values are produced only on N33's daemon-side
//!   mint half (`C-mod-N33`, `wqm-serve`), on the dedicated rules RPC — a later
//!   slice. This file deliberately offers no constructor, no `Default`, and no
//!   `impl` block at all.
//! - **The Layer-2 MAC marker.** The persisted drain-visible marker is N5's
//!   artifact, not this type's: N5 stamps it and N5 owns its key. When it lands
//!   (N5's slice) its one legitimate setter carries the guard anchor that
//!   `ci/guard_mac_single_setter.py` counts -- that anchor is spelled nowhere in
//!   this crate, deliberately: the guard is a substring count with no prose
//!   exemption, so quoting it here would register this file as the setter.
//! - **The drain check.** Verifying a persisted row's marker, and rejecting a
//!   `collection=rules` row that carries none, is N5's drain in `wqm-store-write`.
//!
//! Sources: contracts N33 *Provides*; ARCH §6.2 N5 (the MAC over
//! `(row-identity || collection)`, the single-setter guard, the key lifecycle) and
//! §6.2 N33 (unforgeability = MAC + CI guard, not visibility — R1/A5) and §9.1
//! (the cap TYPE row homed in `wqm-common`); PRD F-08 (home + AC3
//! constructibility), F-13 (the single-setter CI guard), F-38 (the end-to-end
//! chain).

/// The capability a caller must hold to enqueue a write to the rules collection.
///
/// # The two layers, and why one type is not the whole story
///
/// **Layer 1 — this value.** A sealed, non-`pub`-constructible token that gates
/// the *enqueue* call: `N50.rules_write(item, cap)` takes one and hands it to
/// `N5.enqueue_rules`. It is minted only on N33's daemon-side mint half
/// (`C-mod-N33`, `wqm-serve`) and returned as a runtime VALUE, so it appears in
/// no dependency view (R1/H1) — the same unforgeability pattern as the N11
/// `AuthToken`.
///
/// **Layer 2 — the persisted marker, which is NOT this type.** The drain reads
/// from the *persisted* queue, so after a crash or restart the in-memory value is
/// gone. `N5.enqueue_rules` therefore stamps a server-set, drain-visible marker on
/// the queue row: a MAC keyed by a daemon-held secret over
/// `(row-identity || collection)` (ARCH §6.2 N5). The drain re-checks that marker
/// and rejects a `collection=rules` row that lacks a valid one. The drain trusts
/// only the persisted marker — never ephemeral memory — which is
/// order-by-recoverability applied to authority.
///
/// So **unforgeability rests on the MAC plus the F-13 CI single-setter guard, not
/// on Rust visibility** (R1/A5): the private field below cannot cover the trust
/// boundary the persisted row crosses. Treat the seal as the cheap first layer,
/// not the guarantee.
///
/// "Rules require a capability" is itself the per-collection N35 axis
/// `requires_write_capability`, not a hard-coded drain special case.
///
/// # Homed in `wqm-common`
///
/// The mint half (`wqm-serve`) and the drain (`wqm-store-write`) both name this
/// type. The floor crate is the only place they can share ONE declaration without
/// a crate cycle (PRD F-08, ARCH §9.1). How the out-of-crate mint in `wqm-serve`
/// is granted access is N33's slice to decide and is deliberately unanswered here;
/// today the type has no constructor anywhere.
///
/// # Naming it is legal; building one is not
///
/// Any crate may name the type, take it by reference, and pass it along:
///
/// ```
/// use wqm_common::rules_write_cap::RulesWriteCap;
///
/// fn rules_write(_cap: &RulesWriteCap) {}
/// ```
///
/// Building one from outside the crate is refused, because the seal field is
/// private (E0451). This is PRD F-08 AC3's compile/visibility check, and the
/// snippet above is its control: the path and spelling compile, so the failure
/// below is the private field and not a typo.
///
/// The block records its expected code, and this is the one where that matters
/// most: rustc runs its privacy pass AFTER type-check, so any earlier error --
/// a renamed field, a changed shape -- aborts before E0451 is ever reached, and
/// a bare `compile_fail` would stay green while asserting nothing. Note that
/// rustdoc does NOT check the code (measured, rustc 1.98.0, stable and nightly:
/// a deliberately wrong code still passes), so it is a review anchor rather than
/// a gate. The snippet was compiled standalone against this crate and raises
/// E0451 alone; re-run that check if it is ever edited.
///
/// ```compile_fail,E0451
/// use wqm_common::rules_write_cap::RulesWriteCap;
///
/// let _forged = RulesWriteCap { _sealed: () };
/// ```
///
/// There is no `Default` either — a `Default` impl is a public constructor with a
/// different name, and would hand every caller the authority the type exists to
/// withhold:
///
/// ```compile_fail,E0599
/// use wqm_common::rules_write_cap::RulesWriteCap;
///
/// let _forged = RulesWriteCap::default();
/// ```
///
/// And the type is not `Clone`: a holder of one capability must not be able to
/// manufacture a second and hand it on, so the grant stays a move rather than
/// becoming a supply. Asserted through a bound, since no value can be obtained to
/// call `.clone()` on:
///
/// ```compile_fail,E0277
/// use wqm_common::rules_write_cap::RulesWriteCap;
///
/// fn assert_clone<T: Clone>() {}
/// assert_clone::<RulesWriteCap>();
/// ```
///
/// # And no inherent constructor
///
/// This is the absence most likely to erode, because it is the path of least
/// resistance: when N33's mint slice (`C-mod-N33`, `wqm-serve`) is written, a
/// `pub fn new()` here is the shortest way to make the mint compile, and it
/// would hand the authority to every crate in the workspace at the same time.
/// Nothing mechanical would notice -- F-13's `ci/guard_mac_single_setter.py`
/// counts the setter of N5's Layer-2 marker, not the minters of this type -- so
/// the tripwire has to be here:
///
/// ```compile_fail,E0599
/// use wqm_common::rules_write_cap::RulesWriteCap;
///
/// let _forged = RulesWriteCap::new();
/// ```
///
/// When the mint does arrive it must not arrive as a public `new()`; a doctest
/// that fails at that point is this file objecting, not a stale test.
#[derive(Debug)]
pub struct RulesWriteCap {
    /// The seal. Private, so no out-of-crate literal can build the type; a unit
    /// field so the value stays zero-sized and carries nothing a holder could
    /// read, edit, or forge a copy of. Underscore-prefixed because it is never
    /// read by design — the field IS the declaration, not a slot.
    _sealed: (),
}

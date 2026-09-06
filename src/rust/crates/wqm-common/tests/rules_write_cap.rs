//! `P04-GT002-WO074` -- `C-ty-rules-write-cap`.
//!
//! At this work order the type has no constructor anywhere, so there is no value
//! to exercise: every assertion below is therefore made at the TYPE level. The
//! negatives that need a compiler refusal (out-of-crate construction, `Default`,
//! `Clone`) are `compile_fail` doctests on the type itself, since a test that
//! must fail to compile cannot live in a file that has to compile.

use core::mem::size_of;

use wqm_common::rules_write_cap::RulesWriteCap;

/// The capability is minted on the daemon's N33 half and consumed by N50/N5 --
/// plausibly on a different task from the one that minted it. A type that were
/// neither `Send` nor `Sync` would force that hand-off into one task and would be
/// discovered only when the mint path is written, several slices from now.
#[test]
fn the_capability_crosses_threads() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<RulesWriteCap>();
}

/// Proxy for "the seal carries no slot": a zero-sized type has no field a holder
/// could read, mutate, or copy out, and cannot have acquired a public field
/// without this assertion failing. It is a PROXY, not the property itself -- a
/// private non-zero-sized field would also be unreadable from outside -- but it
/// is the part of "the empty private body IS the declaration" that a test can
/// decide, and it fails loudly if a later edit smuggles state into the cap
/// instead of into N5's persisted marker where the design puts it.
#[test]
fn the_capability_carries_no_payload() {
    assert_eq!(size_of::<RulesWriteCap>(), 0);
}

/// `Debug` is the one derive the type carries, and it is safe precisely because
/// of the assertion above: a zero-sized seal has nothing to leak into a log. The
/// test pins that the derive stays present, since a cap that cannot be formatted
/// forces `{:?}` off every struct that comes to contain one.
#[test]
fn the_capability_is_debug_and_has_nothing_to_leak() {
    fn assert_debug<T: core::fmt::Debug>() {}

    assert_debug::<RulesWriteCap>();
}

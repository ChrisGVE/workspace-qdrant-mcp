//! Display forms for N8-guarded names — the one producer, this side of `CR-038`.
//!
//! # Why this module exists
//!
//! N8 (`wqm_common::names`) owns *what a thing is called on the wire*. It does not yet own
//! *what we show for it*, so every surface that wanted a human label wrote one beside the
//! registry name — and the tab bar ended up saying `Library` where the collection is
//! spelled `libraries`.
//!
//! That divergence has no guard behind it. `src/rust/ci/guard_name_registry.py` matches the
//! guarded literal inside Rust string quotes, so `"Library"` can never trip it: the guard
//! catches a *re-spelling of the identifier*, which is a different failure from a *drifting
//! display label*. Nothing downstream is waiting to catch this one (`CR-038` refuting the
//! earlier claim in `handover.md` §5).
//!
//! `CR-038` decides the invariant: **a display string for a guarded name is derived from
//! the registry, never written beside it.** N8 gains a display form alongside `name()`, and
//! surfaces render through it. Until that lands, [`display`] is this crate's single
//! producer — one function to delete, not a literal per call site to hunt down.
//!
//! # The label reads *Libraries*, not *Library*
//!
//! The UI half of `CR-038` was left to this module, and this is the answer. Deriving means
//! title-casing what the registry already says; singularising `libraries` to *Library*
//! would need a per-name exception whose only benefit is how it reads, and a table of
//! exceptions is the hand-maintained divergence the derivation exists to remove. Screen and
//! wire say the same word.

use wqm_common::names::Collection;

/// The human label for a canonical collection.
///
/// Title-case of the registry name. Every canonical name is one lowercase word, so this is
/// the whole rule — a name that needed more would be a signal that N8's display form is
/// overdue rather than a reason to grow a table here.
pub fn display(collection: Collection) -> String {
    let name = collection.name();
    let mut chars = name.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_canonical_name_has_a_label() {
        // Enumerated from N8 rather than listed here: a fifth collection must arrive with
        // a label, and this test is what notices if the derivation ever stops covering the
        // closed set.
        for collection in Collection::ALL {
            let label = display(collection);
            assert!(!label.is_empty(), "{collection:?} produced no label");
            assert_eq!(
                label.to_lowercase(),
                collection.name(),
                "{collection:?}'s label is not its registry name re-cased"
            );
        }
    }

    #[test]
    fn the_library_label_is_the_plural() {
        // The decision itself, pinned: `CR-038`'s UI half. A future session that prefers
        // the singular has to change this test, which is where the reasoning lives.
        assert_eq!(display(Collection::Libraries), "Libraries");
    }
}

//! The one switch that says *a modal owns the input*, and what it does to a colour.
//!
//! VISUAL-LANGUAGE §6, in Chris's own words twice over: 20260906, *"remove all highlighting to
//! the underlying page when in modal mode"*, and 20260907, looking at the Dashboard under one,
//! *"we still have colors on the screen while all should be muted (including the indicators)"*.
//! The second is what this module exists for — the first was implemented per widget, and a
//! per-widget rule can only mute the widgets someone remembered.
//!
//! # Why one global rather than a flag threaded down the tree
//!
//! The flag was threaded down the tree, and it leaked. `TabBar`, `ZoneHeading`, `CellPane`,
//! `AppBar`, `ConstantTop` and both views each carried an `under_modal(bool)`; between them
//! they muted the jump digits, the selected tab's fill, the key letters and the §4 alarm hues,
//! and **every other hue on the page went on painting** — the status block's RAG discs, the
//! queue's three counts, the roll-up dot at the foot, the queue triples inside the cells. Each
//! of those widgets was individually correct about the rule it had been told; none of them had
//! been told, because the rule was a parameter and a parameter has to reach you.
//!
//! So the switch moved to where the *colours* are. [`crate::tokens`] is the only place a hue is
//! decided, so a hue accessor that consults this switch is a rule that cannot be missed by a
//! widget nobody thought about — including one written next year. The view enters the scope
//! once around its whole page draw; nothing below it knows a modal exists.
//!
//! It is a process global for exactly the reason [`crate::tokens::Palette`] and
//! [`crate::encoding::Encoding`] are: a token is read at the leaf of a render tree, and a leaf
//! that had to be handed a context would be a leaf every intermediate widget has to forward
//! for — which is the parameter that just failed.
//!
//! # What it does NOT touch
//!
//! The *quiet* neutral rungs — everything at or below [`crate::tokens::muted`]: `faint`, the two
//! rule weights, the data cursor's grey tint and the modal's own layer fills. Emphasis is not
//! highlight (§1's two axes): weight, the `▌` bar and the fills are structure, and structure
//! survives.
//!
//! What goes is **hue** (every one of them, and every fill that carried one) **and the bright
//! text rungs**. Chris, 2026-09-07, looking at a Dashboard under a modal: *"we still have colors
//! on the screen while all should be muted"* — emphasis is a highlight too, so a page beneath a
//! modal carries no text brighter than `muted`. The three rungs that would — `normal`, `strong`
//! and the cursor mark — collapse onto it, alongside [`crate::tokens::header`], which is
//! `normal` by construction.
//!
//! And **the modal itself**. The scope names the page *beneath* one, so a view that draws both
//! closes the scope before it draws the box on top — a modal muted by the rule that mutes what
//! is behind it would be a dialogue you cannot read. No view in this crate draws both yet: the
//! two that carry the flag depict the page and leave the box out, deliberately.

use std::sync::atomic::{AtomicUsize, Ordering};

use ratatui::style::Color;

#[cfg(test)]
mod tests;

/// How many scopes are currently open. A depth rather than a flag so that a nested scope —
/// a modal drawn over a page that is already under one — restores the outer state on drop
/// instead of clearing it.
static DEPTH: AtomicUsize = AtomicUsize::new(0);

/// Whether a modal owns the input right now.
///
/// Every hue accessor in [`crate::tokens`] asks this. Nothing else should have to: a widget
/// that branches on it is re-implementing the rule this module exists to hold in one place.
pub fn under_modal() -> bool {
    DEPTH.load(Ordering::Relaxed) > 0
}

/// A page drawn beneath a modal, for as long as the value lives.
///
/// Held by the **view** — [`crate::views::dashboard::Dashboard`] and
/// [`crate::views::shell::ShellView`] — around the whole page draw, because "the page beneath
/// the modal" is a screen-level fact and a screen is what owns it.
///
/// ```text
/// let _modal = self.modal.then(tokens::ModalScope::enter);
/// ```
///
/// [`Option`] drops exactly like the scope itself, so the idiom above needs no `if`.
#[must_use = "the scope ends the moment it is dropped, so an unbound one mutes nothing"]
pub struct ModalScope(());

impl ModalScope {
    /// Open a scope. Nesting is counted, so an inner scope's [`Drop`] restores the outer one
    /// rather than switching the page back on underneath it.
    pub fn enter() -> Self {
        DEPTH.fetch_add(1, Ordering::Relaxed);
        ModalScope(())
    }
}

impl Drop for ModalScope {
    fn drop(&mut self) {
        DEPTH.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A colour, or [`crate::tokens::muted`] while a modal owns the input.
///
/// The whole rule, in one function, so that "which colours drop under a modal" has exactly one
/// answer. [`crate::tokens::hue`] is its main caller — every reserved hue in the vocabulary is
/// emitted through that — and [`crate::categorical::Categorical`] is another, since a data hue is
/// not a role and so does not pass through it. The bright text rungs
/// ([`crate::tokens::normal`], [`crate::tokens::strong`], [`crate::tokens::cursor_mark`]) pass
/// through it too: a page beneath a modal carries no emphasis, and emphasis is not only hue.
pub(crate) fn or_muted(colour: Color) -> Color {
    if under_modal() {
        super::muted()
    } else {
        colour
    }
}

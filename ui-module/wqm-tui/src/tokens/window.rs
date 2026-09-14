//! **Inside a window the emphasis ladder is shorter**, and this is the switch that says so.
//!
//! Chris, item 4: *"the third column … the text is unreadable. There are multiple examples of
//! that besides the third column, the bottom help is similarly hard to read, in A+ the
//! non-editable fields are also hard to read. That is what needs to be improved."*
//!
//! Three surfaces, and they are three rungs: the third column is [`crate::tokens::faint`] (50),
//! the help rows' labels and a read-only value are both [`crate::tokens::muted`] (62).
//!
//! # Holding the tint's lightness fixed the cause; it did not fix these
//!
//! [`crate::tokens::TintBlend::HoldLuminance`] gives the BODY baseline its ground back — 13 of
//! 15 bundled themes clear the WCAG floor at rung 85 where 2 did before. Measured on the same
//! surface, rung 62 clears it on 4 of 15 and **rung 50 on none at all**. Those two were under
//! the floor before any tint existed, so the tint was never their cause and no blend is their
//! cure.
//!
//! # The answer is fewer rungs, not brighter ones
//!
//! An emphasis ladder deliberately makes some text quieter; a contrast floor says all text must
//! be readable. They are not reconcilable by sliding one rung up — raising `faint` to where it
//! clears 4.5:1 lands it on top of `muted`, and raising `muted` in turn lands it on `normal`.
//! The ladder has eleven rungs because a SCREEN is large and has room for that many degrees of
//! emphasis.
//!
//! A window is not a screen. It is ~24 rows of dense, deliberately-chosen content where nothing
//! is scenery, so it needs **two** levels and not four: what you are reading, and what is beside
//! it for reference. [`RAISED_QUIET`] is that second level, and inside a window every rung below
//! it collapses onto it.
//!
//! # It is a SCOPE, for the reason [`crate::tokens::modal`] is one
//!
//! The rule has to reach a widget nobody thought about — the record view's third column, the
//! container's help rows, and whatever a later view puts inside a window — and a parameter only
//! reaches the widgets someone remembered to thread it through. That failure is written down in
//! `tokens::modal`'s own module docs, where a per-widget `under_modal(bool)` flag left half the
//! page painting. So the window holds a scope around its whole draw and the rungs consult it.
//!
//! The page beneath is drawn OUTSIDE that scope and keeps the full ladder — it has the room for
//! it, and quietening it is [`crate::tokens::modal`]'s job, not this one's.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

#[cfg(test)]
use ratatui::style::Color;

#[cfg(test)]
mod tests;

/// The one quiet rung a window has.
///
/// Rung 75 is [`crate::tokens::table_row`]'s — a table's data rows, chosen by Chris on 20260912
/// as *"a light grey … still allowing good readability"*. That it is also where the contrast
/// floor lands is not a coincidence worth hiding: both questions are *"how quiet can text get
/// and still be read for a page at a time"*, asked once by eye and once by measurement.
pub const RAISED_QUIET: u8 = 75;

/// How the rungs below the baseline behave inside a window.
/// # `Ladder` is what the window used to do, kept only to measure against
///
/// The scope is the answer to item 4 and it is in force: inside a window `faint` and `muted`
/// collapse onto [`RAISED_QUIET`]. `Ladder` is off every enumeration and no pantry variant
/// offers it — it survives because the tests below state the improvement as a COUNT (0 of 15
/// legible against 11 of 15), and a count needs both sides to be a claim rather than a number.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum WindowText {
    /// **In force**: `faint` and `muted` collapse onto [`RAISED_QUIET`] inside a window.
    #[default]
    Raised,
    /// The full eleven-rung ladder. **Retired** — the measurement baseline above.
    Ladder,
}

static MODE: AtomicU8 = AtomicU8::new(WindowText::Raised as u8);
static DEPTH: AtomicUsize = AtomicUsize::new(0);

impl WindowText {
    pub fn current() -> Self {
        match MODE.load(Ordering::Relaxed) {
            1 => Self::Ladder,
            _ => Self::Raised,
        }
    }

    pub fn set(mode: Self) {
        MODE.store(mode as u8, Ordering::Relaxed);
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Ladder => "the full ladder",
            Self::Raised => "one quiet rung",
        }
    }
}

/// Whether a window is being drawn right now.
pub fn in_window() -> bool {
    DEPTH.load(Ordering::Relaxed) > 0
}

/// A window's own draw, for as long as the value lives — held around the container AND the view
/// inside it, because the rule is about everything in the box.
#[must_use = "the scope ends the moment it is dropped, so an unbound one raises nothing"]
pub struct WindowScope(());

impl WindowScope {
    pub fn enter() -> Self {
        DEPTH.fetch_add(1, Ordering::Relaxed);
        WindowScope(())
    }
}

impl Drop for WindowScope {
    fn drop(&mut self) {
        DEPTH.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A quiet rung, raised to [`RAISED_QUIET`] while a window is drawing under
/// [`WindowText::Raised`].
///
/// Called by [`crate::tokens::faint`] and [`crate::tokens::muted`], so a widget asks for the
/// emphasis it means and never for the rung — which is the same contract every other token in
/// this module has.
pub(crate) fn quiet(percent: u8) -> u8 {
    match (WindowText::current(), in_window()) {
        (WindowText::Raised, true) => percent.max(derived_quiet()),
        _ => percent,
    }
}

/// The highest a quiet rung may go and still read as quieter than the body text.
///
/// Five rungs under [`crate::tokens::NORMAL_RUNG`]. Without a ceiling the search would climb
/// until it met the baseline on a thin theme and the window would have ONE level, which is the
/// opposite of what the third column is for — a reference column that looks exactly like the
/// value beside it has stopped being a reference.
const QUIET_CEILING: u8 = 80;

/// The lowest rung whose text can be read on THIS theme's window fill, floored at
/// [`RAISED_QUIET`] and capped at [`QUIET_CEILING`].
///
/// Derived rather than named, for the reason every other round-2 rung is: the ladder is a curve
/// through each theme's own four neutrals, so one number buys different amounts on different
/// themes. Measured at the fixed rung 75 the window's quiet text was legible on 10 of 15; the
/// three that failed while their own baseline passed — One Dark Pro, Catppuccin Latte,
/// Everforest — are exactly the themes a fixed rung cannot serve.
fn derived_quiet() -> u8 {
    let ground = super::modal_fill(super::layer1_bg());
    super::contrast::rung_for_contrast(ground, super::contrast::BODY_FLOOR)
        .unwrap_or(RAISED_QUIET)
        .clamp(RAISED_QUIET, QUIET_CEILING)
}

/// The colour a quiet rung resolves to here — for a caller that already has the colour and wants
/// to know whether it moved. Used only by the tests.
#[cfg(test)]
pub(crate) fn quiet_colour(percent: u8) -> Color {
    super::neutral_at(quiet(percent))
}

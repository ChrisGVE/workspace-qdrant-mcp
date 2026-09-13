//! The selectable tint for a modal's own border and layer fill, and the ONE blend every
//! surface inside a window goes through.
//!
//! The page beneath a modal is muted by `tokens::modal`; this module colours only the
//! window drawn above it. The pantry compares the four candidates in otherwise identical
//! windows, and [`ModalTint`] stays an enum because Chris has not picked — the default below
//! is the director's gate call (accent at 0.28), not his ruling.
//!
//! # Why the strength is a variable, and why the blend is ONE function
//!
//! [`WASH_MIX`] is 0.14 because the daemon-unreachable wash covers the WHOLE SCREEN and had
//! to stay subtle enough *"as not destroy visibility across the window"*. A modal tint covers
//! one window over an already-quiet page and is not spending that budget; read back from a
//! pixel render at 0.14 it is very nearly not there, which is the *"washed out and sad"* the
//! designer was answering. So the modal's strength is [`tint_strength`], its own number.
//!
//! Making it a variable opened a seam, and that seam is why [`modal_fill`] is now the only
//! blend in the crate. While a framework window blended at a *variable* strength and
//! [`crate::widgets::modal::Modal`] blended at the fixed 0.14, a `Layer2` confirm opened over
//! a `Layer1` window read LESS tinted than its own parent at any strength above 0.14 —
//! inverting the depth cue, because the nearer layer must read as further from the page, not
//! closer to it. Two blend functions is what made that inversion expressible. There is one
//! now, and every surface inside a window reaches it: the window's own fill, the
//! editable-field rungs, the reference band, the drop-down, and the confirm guard. They move
//! together when the strength moves, so the window stays one colour family at any strength.
//!
//! [`WASH_MIX`]: crate::tokens::WASH_MIX

use std::sync::atomic::{AtomicU32, AtomicU8, Ordering};

use ratatui::style::Color;

use super::{active_theme, family, mix, muted, Rgb};
use crate::encoding::Family;

/// The hue used to distinguish the active modal from the quietened page.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModalTint {
    Neutral,
    Accent,
    Selected,
    InFlight,
}

/// The default is **`Accent`** — Catppuccin Mocha's *blue*.
///
/// The argument is IDENTITY, not distance, and [`crate::tokens::field`]'s tests measure it
/// across all fifteen bundled themes: `Selected` does not sit *near* the data cursor's hue, it
/// **is** that hue (`selected_of`, 15 of 15); `InFlight` means *work is happening now* and
/// collapses onto `info` — the selector's own field — on 13 of 15; `info` is the selector's
/// absolutely and `success`/`warning`/`error` are the health three. `accent` is the one field
/// carrying an AFFORDANCE rather than a system state (the jump digit, a sortable column's
/// letter, a key in the help window), so a window wearing it cannot be misread as an alarm, a
/// selection or a cursor — and a window that owns the keyboard is exactly an affordance.
static ACTIVE: AtomicU8 = AtomicU8::new(ModalTint::Accent as u8);

/// Twice [`crate::tokens::WASH_MIX`] — the director's gate call, bracketed in the pantry by
/// `Tint: neutral / accent 0.14 / accent 0.28 / accent 0.40` so Chris judges a range rather
/// than a number someone argued their way to. At 0.40 the faint rungs start losing the ground
/// under them, which is what makes 0.40 the far end rather than the next step up.
pub const DEFAULT_TINT_STRENGTH: f32 = 0.28;

/// How far a tinted window's surfaces are pulled toward [`modal_border`].
///
/// Process-global like [`crate::tokens::Palette`] and [`ModalTint`], and for the same reason:
/// it is read at the leaf of a render tree, where a parameter would have to be forwarded by
/// every intermediate widget that does not care about it.
static STRENGTH: AtomicU32 = AtomicU32::new(DEFAULT_TINT_STRENGTH.to_bits());

impl ModalTint {
    pub const ALL: [Self; 4] = [Self::Neutral, Self::Accent, Self::Selected, Self::InFlight];

    pub fn current() -> Self {
        match ACTIVE.load(Ordering::Relaxed) {
            1 => Self::Accent,
            2 => Self::Selected,
            3 => Self::InFlight,
            _ => Self::Neutral,
        }
    }

    pub fn set(tint: Self) {
        ACTIVE.store(tint as u8, Ordering::Relaxed);
    }

    /// What to call this tint when writing to Chris: the Catppuccin Mocha name of the hue it
    /// resolves to, because Mocha is what the harness paints with and what he reads. A report
    /// that said `ModalTint::Accent` would be naming our enum rather than his colour.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::Accent => "blue",
            Self::Selected => "lavender",
            Self::InFlight => "sapphire",
        }
    }
}

/// How far toward the tint every surface inside a window is pulled.
pub fn tint_strength() -> f32 {
    f32::from_bits(STRENGTH.load(Ordering::Relaxed))
}

/// Set it. Clamped, because a strength outside `0..=1` is a blend that leaves the segment
/// between the two colours, and past either end it can only overshoot into a hue neither of
/// them has.
pub fn set_tint_strength(strength: f32) {
    STRENGTH.store(strength.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}

/// The modal's border hue. It is not passed through the under-modal muting rule.
pub fn modal_border() -> Color {
    let tint = ModalTint::current();
    if tint == ModalTint::Neutral {
        return muted();
    }
    if family() == Family::None {
        return Color::Reset;
    }

    // These are the same role sources as accent(), selected(), and in_flight(). Their
    // public accessors mute the page, so modal chrome reads the sources before that rule.
    match (tint, active_theme()) {
        (ModalTint::Accent, Some(theme)) => theme.accent,
        (ModalTint::Selected, Some(theme)) => crate::categorical::selected_of(&theme),
        (ModalTint::InFlight, Some(theme)) => crate::categorical::in_flight_of(&theme),
        (ModalTint::Accent | ModalTint::Selected, None) => Color::Magenta,
        (ModalTint::InFlight, None) => Color::Blue,
        (ModalTint::Neutral, _) => unreachable!("neutral returned before hue selection"),
    }
}

/// Pull a surface inside a window toward the modal's tint at [`tint_strength`].
///
/// **The one blend, and every surface inside a window takes it** — see the module docs for the
/// depth-cue inversion that having two of these produced.
///
/// Slot and indexed colours cannot be blended without knowing the terminal's actual RGB
/// values, so those encodings keep the layer fill and still colour the border.
pub fn modal_fill(layer: Color) -> Color {
    if ModalTint::current() == ModalTint::Neutral {
        return layer;
    }
    let (Some(base), Some(tint)) = (Rgb::from_color(layer), Rgb::from_color(modal_border())) else {
        return layer;
    };
    let strength = tint_strength();
    Color::Rgb(
        mix(base.r, tint.r, strength),
        mix(base.g, tint.g, strength),
        mix(base.b, tint.b, strength),
    )
}

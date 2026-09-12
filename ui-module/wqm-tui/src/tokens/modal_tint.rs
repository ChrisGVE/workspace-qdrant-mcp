//! The selectable tint for a modal's own border and layer fill.
//!
//! The page beneath a modal is muted by `tokens::modal`; this module colours only the
//! window drawn above it. The neutral default preserves the existing appearance while
//! the pantry compares three hues in otherwise identical windows.

use std::sync::atomic::{AtomicU8, Ordering};

use ratatui::style::Color;

use super::{active_theme, family, mix, muted, Rgb, WASH_MIX};
use crate::encoding::Family;

/// The hue used to distinguish the active modal from the quietened page.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModalTint {
    Neutral,
    Accent,
    Selected,
    InFlight,
}

static ACTIVE: AtomicU8 = AtomicU8::new(ModalTint::Neutral as u8);

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

/// Pull a modal layer's RGB fill 14% toward its selected border hue.
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
    Color::Rgb(
        mix(base.r, tint.r, WASH_MIX),
        mix(base.g, tint.g, WASH_MIX),
        mix(base.b, tint.b, WASH_MIX),
    )
}

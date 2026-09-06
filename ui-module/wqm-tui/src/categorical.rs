//! The **extended palette tier** — hues for data, not for roles (Chris, 20260906).
//!
//! [`crate::tokens`] answers *"what does this colour mean?"* — success, warning, error,
//! selector, and the neutral ladder under them. Ten fields, every bundled theme names all ten,
//! and that vocabulary is closed (§10). This module answers a different question that the ten
//! cannot: *"how do I tell these fifteen things apart?"* — graph nodes, tags, cell kinds, where
//! the only meaning a colour carries is **not the same as its neighbour**.
//!
//! Keeping them apart is the point. A categorical hue that reads as `error` is worse than no
//! colour at all, because the reader learns something false rather than nothing.
//!
//! # Two tiers, and the second is not always there
//!
//! **Base tier** — the ten roles, unchanged, on all fifteen themes. **Extended tier** — this,
//! and it is only worth the name when the active theme is a Catppuccin flavour, because that
//! is the palette that ships fourteen mutually-tuned accents. Every other theme falls back to
//! the one base field that carries no meaning and is not reserved — `secondary` — which is
//! nought or one entry. **Consumers must read [`Categorical::len`] before relying on distinct
//! hues** — there is no promise here that fifteen categories get fifteen colours, and on
//! eleven of the fifteen themes there is exactly one.
//!
//! # The flavour is detected from DATA, never from a name
//!
//! [`crate::tokens`] stores a [`ThemePalette`] and nothing else: by the time a colour is
//! wanted, the `ThemeName` that produced it is gone, and a future theme source may have no
//! name at all. So a flavour is recognised by `theme.bg == flavour.base`, exactly, which is a
//! property of the colours actually in force. `ratatui-themes` bundles only Mocha and Latte;
//! Frappé and Macchiato are recognised all the same, because the rule is about the data.
//!
//! # Two rules make the set, and both are measurements
//!
//! 1. **Exclusion.** Drop any accent within [`RESERVED_FLOOR`] ΔE (CIE76) of a
//!    **meaning-carrying** role: `success`, `warning`, `error` (§4's health states) and `info`
//!    (§3's selector), **and `accent`** — the hue the jump digits carry on every screen, so a
//!    data colour equal to it collides everywhere rather than on one view (Chris, 20260906,
//!    overturning his own earlier "affordance, not data"). The floor is not a second constant
//!    — it is the same one [`crate::styles::strong`] defends `strong` with, and for the same
//!    reason.
//! 2. **Ordering — farthest first.** Entry 0 is the survivor farthest from the reserved set;
//!    each next is the survivor whose *nearest* neighbour among {reserved ∪ already chosen} is
//!    largest. So a consumer taking the first three gets the three most separable, and taking
//!    all of them degrades gracefully rather than falling off a cliff. Ties break on
//!    Catppuccin's own order, which makes the result deterministic.
//!
//! # No consumer yet
//!
//! Nothing in this crate renders a categorical hue. The tier exists so the graph and tag views
//! have something correct to reach for when they are built, and so Chris can judge the
//! separability by eye now rather than after a view has been designed around it.

use catppuccin::Flavor;
use ratatui::style::Color;
use ratatui_themes::ThemePalette;

use crate::encoding::{Encoding, Family};
use crate::styles::strong::RESERVED_FLOOR;
use crate::tokens::{self, Palette, delta_e};

#[cfg(test)]
mod tests;

/// The roles a categorical hue must never be mistaken for.
///
/// Five, not ten. `bg`, `fg`, `muted` and `selection` are the neutral ladder — a data hue is
/// not in danger of reading as one — and `secondary` carries no meaning to steal.
///
/// # `accent` is here, and it was not at first (Chris, 20260906)
///
/// The first cut left `accent` out on the argument that a hotkey is an *affordance* rather
/// than a datum, so reusing its hue says nothing false. Chris overturned his own rule on the
/// better argument: the jump digits carry `accent` on **every screen**, so a data hue equal to
/// it does not collide on one view, it collides everywhere. A tag that is the same blue as the
/// key you press is a collision the reader meets constantly and can never learn to ignore.
///
/// It is not free. Reserving it costs every flavour its `blue`, and costs **Mocha** its
/// `lavender` too — Mocha's lavender sits ΔE 11.0 from its blue, just inside the floor, where
/// the other three flavours clear it. Mocha therefore drops from ten hues to eight while the
/// others drop to eight and seven from nine and eight.
pub const RESERVED_ROLES: [&str; 5] = ["success", "warning", "error", "info", "accent"];

/// [`RESERVED_ROLES`] resolved against a theme, in the same order.
fn reserved_of(theme: &ThemePalette) -> [Color; 5] {
    [
        theme.success,
        theme.warning,
        theme.error,
        theme.info,
        theme.accent,
    ]
}

/// The nearest of `others` to `colour`, in ΔE. [`f32::MAX`] when `others` is empty, which is
/// what makes the first pick of the farthest-first walk fall out of the same expression as
/// every later one.
fn min_delta(colour: Color, others: &[Color]) -> f32 {
    others
        .iter()
        .map(|other| delta_e(colour, *other))
        .fold(f32::MAX, f32::min)
}

/// The Catppuccin flavour this theme *is*, recognised by its background.
///
/// Exact equality on purpose. A near-match would mean some other theme with a similar base
/// silently acquired Catppuccin's accents, which is a worse failure than having no extended
/// tier: the hues would be tuned for a palette the rest of the screen is not drawn from.
fn flavour_of(theme: &ThemePalette) -> Option<&'static Flavor> {
    catppuccin::PALETTE
        .all_flavors()
        .into_iter()
        .find(|flavour| Color::from(flavour.colors.base) == theme.bg)
}

/// An ordered set of hues whose only meaning is that they differ from one another.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Categorical {
    entries: Vec<(&'static str, Color)>,
}

impl Categorical {
    /// The tier for the theme and encoding currently in force.
    ///
    /// Empty under [`Family::None`]: with colour refused, every entry would resolve to the same
    /// `Color::Reset`, and handing a consumer fifteen identical hues is worse than handing it
    /// none — one it can detect, the other it cannot. Empty too when no bundled theme is in
    /// force, since there is then no palette to read roles or accents from.
    pub fn current() -> Self {
        if Encoding::current().family() == Family::None || Palette::current().family() == Family::None
        {
            return Self::default();
        }
        match tokens::active_theme() {
            Some(theme) => Self::for_theme(&theme),
            None => Self::default(),
        }
    }

    /// The tier for a **stated** theme — how a pantry frame renders a flavour that is not the
    /// one in force, and how the tests measure all four without touching a global.
    pub fn for_theme(theme: &ThemePalette) -> Self {
        let reserved = reserved_of(theme);
        let entries = match flavour_of(theme) {
            Some(flavour) => farthest_first(accents(flavour), &reserved),
            None => fallback(theme, &reserved),
        };
        Self { entries }
    }

    /// The flavour's name when there is one — what a frame's header line says.
    pub fn flavour(theme: &ThemePalette) -> Option<&'static str> {
        flavour_of(theme).map(Flavor::identifier)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The hue for category `i`, **cycling**. A consumer with more categories than hues gets
    /// repeats rather than a panic — repeats are a legible failure, and `len()` is there for a
    /// consumer that wants to avoid them.
    pub fn color(&self, i: usize) -> Option<Color> {
        (!self.is_empty()).then(|| self.entries[i % self.entries.len()].1)
    }

    /// The name of category `i`'s hue, cycling with [`Categorical::color`].
    pub fn name(&self, i: usize) -> Option<&'static str> {
        (!self.is_empty()).then(|| self.entries[i % self.entries.len()].0)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, Color)> + '_ {
        self.entries.iter().copied()
    }
}

/// A flavour as a [`ThemePalette`], mirroring **`ratatui-themes`' own Catppuccin mapping**
/// (blue→accent, pink→secondary, base→bg, text→fg, overlay0→muted, surface0→selection,
/// red→error, yellow→warning, green→success, teal→info).
///
/// It exists for the two flavours `ratatui-themes` does **not** bundle. Frappé and Macchiato
/// can never be the theme in force, so the only way to show what the tier looks like on them
/// is to describe them the way that crate would if it carried them. Reading its mapping and
/// reproducing it is what keeps those frames a preview rather than an invention — a mapping
/// chosen here would show a palette nobody could ever get.
pub fn mirrored_palette(flavour: &Flavor) -> ThemePalette {
    ThemePalette {
        accent: flavour.colors.blue.into(),
        secondary: flavour.colors.pink.into(),
        bg: flavour.colors.base.into(),
        fg: flavour.colors.text.into(),
        muted: flavour.colors.overlay0.into(),
        selection: flavour.colors.surface0.into(),
        error: flavour.colors.red.into(),
        warning: flavour.colors.yellow.into(),
        success: flavour.colors.green.into(),
        info: flavour.colors.teal.into(),
    }
}

/// The nearest reserved role to `colour` under `theme`, in ΔE — what a frame prints beside
/// each entry so the exclusion rule can be read rather than trusted.
pub fn distance_to_roles(theme: &ThemePalette, colour: Color) -> f32 {
    min_delta(colour, &reserved_of(theme))
}

/// The nearest already-chosen entry, in ΔE, or [`None`] for entry 0 — the other half of what
/// makes the ordering visible in a frame.
pub fn distance_to_earlier(entries: &[(&'static str, Color)], i: usize) -> Option<f32> {
    let earlier: Vec<Color> = entries[..i].iter().map(|(_, c)| *c).collect();
    (i > 0).then(|| min_delta(entries[i].1, &earlier))
}

/// A flavour's fourteen accents, in Catppuccin's own order — which is what breaks ties below.
fn accents(flavour: &'static Flavor) -> Vec<(&'static str, Color)> {
    flavour
        .colors
        .iter()
        .filter(|colour| colour.accent)
        .map(|colour| (colour.identifier(), Color::from(*colour)))
        .collect()
}

/// Rule 1 then rule 2 from the module docs: exclude, then order farthest-first.
fn farthest_first(
    mut pool: Vec<(&'static str, Color)>,
    reserved: &[Color],
) -> Vec<(&'static str, Color)> {
    pool.retain(|(_, colour)| min_delta(*colour, reserved) >= RESERVED_FLOOR);

    let mut chosen: Vec<(&'static str, Color)> = Vec::with_capacity(pool.len());
    while !pool.is_empty() {
        let mut best = 0;
        let mut best_distance = f32::MIN;
        for (i, (_, colour)) in pool.iter().enumerate() {
            let taken: Vec<Color> = chosen.iter().map(|(_, c)| *c).collect();
            let distance = min_delta(*colour, reserved).min(min_delta(*colour, &taken));
            // Strictly greater, walking the pool in Catppuccin's order: a tie keeps the
            // earlier colour, which is what makes the whole result reproducible.
            if distance > best_distance + f32::EPSILON {
                best = i;
                best_distance = distance;
            }
        }
        chosen.push(pool.remove(best));
    }
    chosen
}

/// What a theme that is not a Catppuccin flavour can honestly offer: **`secondary`, and that
/// is all** — nought or one entry.
///
/// It was `accent` and `secondary` until 20260906. Reserving `accent` did not merely remove it
/// from the list; it made it *unofferable*, because a reserved role sits ΔE 0 from itself and
/// fails its own floor by construction. Writing the loop over both and letting `accent` filter
/// itself out would work, and would be a line nobody could read — so the candidate list says
/// what it is.
///
/// One entry is not a categorical palette and this module does not pretend otherwise: on
/// Everforest the tier is a single hue. See the module docs on reading [`Categorical::len`]
/// before assuming distinct colours exist. The filter is the same [`RESERVED_FLOOR`] the
/// Catppuccin path uses, so `secondary` drops out on a theme that puts it near a role.
fn fallback(theme: &ThemePalette, reserved: &[Color]) -> Vec<(&'static str, Color)> {
    let secondary = theme.secondary;
    if min_delta(secondary, reserved) >= RESERVED_FLOOR {
        vec![("secondary", secondary)]
    } else {
        Vec::new()
    }
}

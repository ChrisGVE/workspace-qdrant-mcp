//! **Can this be read?** — the instrument ΔE cannot be, and the rung search that answers it.
//!
//! [`crate::tokens::delta_e`] is the measure every colour *distance* in this crate is recorded
//! in, and round 2 opened with a question it cannot answer. Chris, 2026-09-14, item 4: *"the
//! text is unreadable. There are multiple examples of that besides the third column, the bottom
//! help is similarly hard to read, in A+ the non-editable fields are also hard to read. That is
//! what needs to be improved."*
//!
//! # Why a second instrument rather than a tighter ΔE floor
//!
//! ΔE is a **difference**, and legibility is a **ratio of light**. Two consequences, and both
//! of them bite here:
//!
//! 1. **ΔE has no direction.** [`crate::tokens::rung_of`] already carries this warning for the
//!    emphasis ladder — *"ΔE said the accent-tinted `strong` candidates were well clear of body
//!    text; they sit at rungs 80 and 77 against `normal` at 85, i.e. darker than the text they
//!    have to out-shout"*. The same blindness applies to text on a fill: ΔE 8 buys a legible
//!    row in one direction and an illegible one in the other.
//! 2. **ΔE is dominated by hue and chroma, which contribute nothing to reading small glyphs.**
//!    Two colours can sit ΔE 12 apart on the a\*/b\* axes alone, at identical lightness, and a
//!    line of 8×13 text between them is unreadable. The tint pulls every surface toward one
//!    blue, which moves exactly those two axes — so a ΔE floor measured against a tinted fill
//!    is measuring the half of the difference that does not help.
//!
//! So this module adds **WCAG 2.2 contrast ratio**, which is defined on relative luminance and
//! nothing else. It does not replace ΔE: a *surface against a surface* (window vs. field fill)
//! is a difference question and stays ΔE's; *text against the surface under it* is a legibility
//! question and is this module's.
//!
//! # The floors, and where they come from
//!
//! WCAG 2.2 success criterion 1.4.3 (Contrast, Minimum), level AA. [`BODY_FLOOR`] 4.5:1 for
//! normal-size text — which is every glyph in a terminal, since a cell is a cell. [`UI_FLOOR`]
//! 3:1 is 1.4.11's floor for non-text elements and is the most a *decorative* rule or a
//! scrollbar track has to clear.
//!
//! A terminal is not a web page and nothing here is a compliance claim. What the floor buys is
//! a number that can come out **negative** — the property `coding.md#measurement` asks of any
//! health signal — where "does this look readable to me on Mocha" cannot.
//!
//! # The search, and why the rung is derived rather than named
//!
//! [`rung_for_contrast`] walks the ladder from the bottom and returns the first rung that
//! clears a target against a stated background. That is the shape round 2 needs in two places:
//! the text rungs of item 4, and the field fills of item 3, where a **fixed** rung produces a
//! different lift on every theme because the ladder is a curve through each theme's own four
//! neutrals and not an even ramp.

use ratatui::style::Color;

use super::{delta_e, neutral, Rgb};

#[cfg(test)]
mod tests;

/// WCAG 2.2 SC 1.4.3 level AA, normal text. Every glyph in a terminal is normal text.
pub const BODY_FLOOR: f32 = 4.5;

/// WCAG 2.2 SC 1.4.11, non-text contrast — a rule, a scrollbar track, a field boundary.
pub const UI_FLOOR: f32 = 3.0;

/// The just-noticeable difference the field ladder is already floored at
/// ([`crate::tokens::field`]). Here so the two instruments can be printed side by side.
pub const JND: f32 = 2.3;

/// One channel, linearised out of sRGB — WCAG 2.2's own transfer function.
///
/// The gamma step is the whole difference between this and [`crate::tokens::luminance`], which
/// is a weighted sum of the raw bytes. That one is fine for *ordering* rungs, which is all it
/// is used for; it is wrong for a ratio, because the ratio is defined on light and a byte is
/// not light.
fn linearise(channel: u8) -> f32 {
    let c = channel as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// WCAG relative luminance, `0.0` (black) to `1.0` (white).
pub fn relative_luminance(colour: Color) -> f32 {
    match Rgb::from_color(colour) {
        Some(rgb) => {
            0.2126 * linearise(rgb.r) + 0.7152 * linearise(rgb.g) + 0.0722 * linearise(rgb.b)
        }
        // A colour this crate cannot resolve to RGB — a slot under a non-RGB encoding — has no
        // luminance to report. Zero rather than a guess: the caller is measuring a ladder that
        // does not exist in that family, and a plausible number would hide that.
        None => 0.0,
    }
}

/// WCAG contrast ratio between two colours, `1.0` (identical) to `21.0` (black on white).
///
/// Symmetric by construction — the lighter of the two goes on top — so a caller never has to
/// know which argument is the text and which the ground.
pub fn contrast_ratio(a: Color, b: Color) -> f32 {
    let (la, lb) = (relative_luminance(a), relative_luminance(b));
    let (lighter, darker) = if la >= lb { (la, lb) } else { (lb, la) };
    (lighter + 0.05) / (darker + 0.05)
}

/// The rungs a search may land on.
///
/// Bounded below by the layer fills — a text rung under [`crate::tokens::layer1_bg`] is text
/// darker than the window it is on — and above by 100, which is [`crate::tokens::strong`] and
/// the top of the ladder. Walked in ones, because the ladder is a curve and a coarse step can
/// jump the answer.
/// One above [`crate::tokens::layer1_bg`]'s rung 15 — a text rung at or under the window's own
/// fill is text darker than the surface it sits on.
const LOWEST_RUNG: u8 = 16;
/// [`crate::tokens::strong`], the top of the ladder.
const HIGHEST_RUNG: u8 = 100;

/// The rungs a search may land on, walked in ones because the ladder is a curve and a coarse
/// step can jump the answer.
///
/// A function rather than a `const`: `Iterator::find` takes `&mut self`, so a `const` range
/// would be copied at each use and clippy's `const_item_mutation` rightly objects.
fn search() -> std::ops::RangeInclusive<u8> {
    LOWEST_RUNG..=HIGHEST_RUNG
}

/// The lowest rung whose text clears `target` contrast against `background`.
///
/// [`None`] when no rung on the ladder does — which is a real answer and not a failure: on a
/// light theme the ladder runs *down* from the background, so a search for a bright rung finds
/// nothing and the caller has to say so rather than clamp to 100 and claim it passed.
pub fn rung_for_contrast(background: Color, target: f32) -> Option<u8> {
    search().find(|rung| contrast_ratio(neutral(*rung), background) >= target)
}

/// The lowest rung whose **surface** sits at least `target` ΔE off `background`.
///
/// The field-fill half of the same idea. A fill is a surface under a surface, so it is a
/// difference question and stays ΔE's — see the module docs for the split.
pub fn rung_for_delta_e(background: Color, target: f32) -> Option<u8> {
    search().find(|rung| delta_e(neutral(*rung), background) >= target)
}

/// The legible end of the ladder to write on `background` — the ladder's top or its bottom,
/// whichever wins the contrast.
///
/// # Why this is not just "white"
///
/// Chris, item (a): *"the previous breadcrumbs must have their background at full saturation of
/// the modal color and be written in white"*. On the theme the harness paints with, `accent` is
/// Catppuccin Mocha's **blue** — a *light* blue — and white on it measures **2.0:1**, far under
/// [`BODY_FLOOR`] and below even [`UI_FLOOR`]. Black on that same blue measures 10.6:1.
///
/// A bundled theme is free to put its accent anywhere, and across the fifteen it lands on both
/// sides of the middle, so no single text colour is legible on all of them. Naming the *role* —
/// "the end of the ladder that can be read here" — is what makes the instruction hold on every
/// theme rather than on the ones whose accent happens to be dark.
///
/// Rung 100 and rung 0 rather than literal white and black: they are the ladder's own ends, so
/// this spends no colour the palette does not already have.
pub fn text_on(background: Color) -> Color {
    let (light, dark) = (neutral(100), neutral(0));
    if contrast_ratio(light, background) >= contrast_ratio(dark, background) {
        light
    } else {
        dark
    }
}

/// The same hue, at a lightness where [`text_on`] can actually be read — for a surface that has
/// to carry text and is specified by its HUE rather than by its rung.
///
/// # The case it exists for
///
/// [`text_on`] picks the better end of the ladder, and on most themes one of them is comfortably
/// legible on the accent. On some it is not: an accent that sits in the MIDDLE of the lightness
/// range is bad for black and bad for white at once. Dracula's is one — its best case is 4.2:1,
/// under [`BODY_FLOOR`] — and a breadcrumb bar painted in it would be a bar nobody can read on
/// that theme, however saturated.
///
/// So the hue and the chroma are kept exactly as the theme named them and only `L*` moves, which
/// is the same trade [`crate::tokens::TintBlend::HoldLuminance`] makes in the other direction:
/// lightness is the ladder's axis, not the palette's. Chris's instruction names *saturation*
/// (*"full saturation of the modal color"*) and that is the axis left untouched.
///
/// Darker first, because a saturated hue holds its identity better as it darkens than as it
/// washes out, and because the window this sits on is dark on eleven of the fifteen bundled
/// themes. If no darker `L*` works, lighter is tried before giving up and returning the colour
/// unchanged — an honest failure rather than a black bar with no hue left in it.
pub fn legible_ground(colour: Color, floor: f32) -> Color {
    // A slot or indexed colour has no `L*` to move and [`super::lab`] panics on one, so an
    // encoding below RGB gets the colour back untouched. Found by rendering the `NO_COLOR`
    // frame, which crashed here rather than degrading: under that encoding every rung resolves
    // to `Color::Reset`, the ratio comes out 1.0, and the search ran on a colour with no
    // channels.
    if Rgb::from_color(colour).is_none() {
        return colour;
    }
    if contrast_ratio(text_on(colour), colour) >= floor {
        return colour;
    }
    let (lightness, a, b) = super::lab(colour);
    let start = lightness.round() as i32;
    let darker = (0..=start).rev();
    let lighter = (start + 1)..=100;
    darker
        .chain(lighter)
        .map(|l| super::lab_to_color((l as f32, a, b)))
        .find(|candidate| contrast_ratio(text_on(*candidate), *candidate) >= floor)
        .unwrap_or(colour)
}

/// Both numbers for one text rung on one ground, so a table prints them together.
///
/// A row of this is the whole argument of item 4: the ΔE column is what the design was floored
/// on and the ratio column is what the reader actually experiences.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Legibility {
    pub rung: u8,
    pub delta_e: f32,
    pub ratio: f32,
}

impl Legibility {
    /// Measure `rung` as text on `background`.
    pub fn measure(rung: u8, background: Color) -> Self {
        let colour = neutral(rung);
        Self {
            rung,
            delta_e: delta_e(colour, background),
            ratio: contrast_ratio(colour, background),
        }
    }

    /// Whether this rung is legible as body text where it is.
    pub fn passes(self) -> bool {
        self.ratio >= BODY_FLOOR
    }
}

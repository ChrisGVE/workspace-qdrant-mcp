//! The **breadcrumb trail**, and round 2's powerline form of it.
//!
//! Chris, 2026-09-14, item (a), verbatim: *"breadcrumbs: they are not visible enough, the
//! previous breadcrumbs must have their background at full saturation of the modal color and be
//! written in white, a white chevron between them, then the current breadcrumb must have a
//! lighter background (mid-way between the normal background and the full saturation, the text
//! must be white bold, and the background must finish with a chevron. There is also a chevron
//! between the last breadcrumb and the current one. Use nerd fonts to achieve the result."*
//!
//! # The shape, read off his sentence
//!
//! ```text
//!  Queue  open-books  reading_guide.py
//! └─ full saturation ─┘└─ midway ──┘
//! ```
//!
//! Previous crumbs share ONE run of the full-saturation accent with chevrons drawn *inside* it;
//! the current crumb sits on a lighter run; and each transition is a chevron in the colour of
//! the run behind it, on the colour of the run in front — which is what makes a powerline
//! segment read as a solid arrow rather than as a glyph between two blocks.
//!
//! # ⚠ "Written in white" does not survive a light accent, and it is measured
//!
//! Catppuccin Mocha's `accent` is a light blue. White on it is **2.0:1** — under every floor
//! there is — and black on it is 10.6:1. Across the fifteen bundled themes the accent lands on
//! both sides of the middle, so no one text colour is legible on all of them.
//!
//! So the instruction is kept as a **role**: the crumb text is the end of the ladder that can
//! be read on the background it is on ([`crate::tokens::contrast::text_on`]) — which IS white
//! wherever the accent is dark, and is black where white would have been unreadable. The frame
//! pair renders both so the substitution is visible rather than asserted.
//!
//! # Nerd fonts are a FONT, not an encoding
//!
//! [`CrumbStyle::Powerline`] spends `U+E0B0`, which lives in the private-use area: a terminal
//! without a patched font draws a replacement box, and nothing in the program can detect that.
//! It degrades by *colour* under a poorer encoding like everything else, but the glyph risk is
//! real and is not ours to measure — so [`CrumbStyle::Plain`] stays, and it is the fallback a
//! configuration would select rather than something the renderer guesses at.

use ratatui::{
    style::{Modifier, Style},
    text::{Line, Span},
};

use crate::tokens;

#[cfg(test)]
mod tests;

/// The powerline separator — a filled right-pointing triangle that fills its whole cell, which
/// is what lets two runs of background meet with no seam.
pub const POWERLINE: &str = "\u{e0b0}";

/// How the trail is drawn.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CrumbStyle {
    /// Round 1: plain text, ancestors muted, the leaf at the body rung, a faint `›` between.
    /// Costs no font.
    #[default]
    Plain,
    /// **Round 2, item (a)**: powerline segments in the modal's own hue. Needs a patched font.
    Powerline,
}

/// The background a *previous* crumb sits on — the modal's hue at full saturation.
///
/// Not through [`tokens::modal_fill`], and that is the point of the item: every other surface in
/// the window is the hue *blended into* a neutral, which is why the trail was *"not visible
/// enough"*. This one is the hue itself.
fn ancestor_bg() -> ratatui::style::Color {
    // …at a lightness its own text can be read on. On most themes this is the accent untouched;
    // on a theme whose accent sits mid-range it is the same hue, darker. See
    // `tokens::contrast::legible_ground`, and the module docs for why the axis it moves is the
    // one Chris's wording does not name.
    tokens::contrast::legible_ground(tokens::modal_border(), tokens::contrast::BODY_FLOOR)
}

/// The background the *current* crumb sits on — *"mid-way between the normal background and the
/// full saturation"*.
///
/// Read literally: halfway from the window's own fill to [`ancestor_bg`]. It is reached through
/// the crate's one blend at a stated strength rather than by a second mixing function, so the
/// trail moves with the window when the tint moves.
fn current_bg() -> ratatui::style::Color {
    let window = tokens::modal_fill(tokens::layer1_bg());
    tokens::contrast::legible_ground(
        tokens::blend(window, ancestor_bg(), 0.5),
        tokens::contrast::BODY_FLOOR,
    )
}

/// One crumb's spans: its text on `bg`, padded by a space each side so the run is not flush
/// against the glyphs it separates.
fn segment(text: &str, bg: ratatui::style::Color, bold: bool) -> Span<'static> {
    let mut style = Style::default().fg(tokens::contrast::text_on(bg)).bg(bg);
    if bold {
        style = style.add_modifier(Modifier::BOLD);
    }
    Span::styled(format!(" {text} "), style)
}

/// The transition between two runs: the separator in the OUTGOING background, drawn on the
/// incoming one.
///
/// That inversion is the whole trick of a powerline. The glyph is a solid triangle filling its
/// cell, so painting it in the colour behind it — over the colour in front — makes the two runs
/// meet as one shape instead of as two blocks with a mark between them.
fn transition(from: ratatui::style::Color, to: ratatui::style::Color) -> Span<'static> {
    Span::styled(POWERLINE, Style::default().fg(from).bg(to))
}

/// The separator *between two ancestors*, which sits inside one run.
///
/// Chris asks for *"a white chevron between them"* — so here the glyph is drawn in the text
/// colour on the ancestors' own background, rather than as a transition between two different
/// ones. Same glyph, different job, and the difference is visible: this one reads as punctuation
/// inside a bar, the other as the bar changing colour.
fn inner_separator() -> Span<'static> {
    let bg = ancestor_bg();
    Span::styled(
        POWERLINE,
        Style::default().fg(tokens::contrast::text_on(bg)).bg(bg),
    )
}

/// The trail, drawn in `style`.
///
/// `crumbs` is root-first and the LAST entry is the current one — the place the reader is
/// standing. A trail of one is all current and has no ancestors, which is the depth-1 frame.
pub fn line(crumbs: &[String], style: CrumbStyle) -> Line<'static> {
    match style {
        CrumbStyle::Plain => plain(crumbs),
        CrumbStyle::Powerline => powerline(crumbs),
    }
}

/// Round 1's trail, unchanged — ancestors muted, the leaf at the body rung.
fn plain(crumbs: &[String]) -> Line<'static> {
    let mut spans = Vec::new();
    let last = crumbs.len().saturating_sub(1);
    for (at, crumb) in crumbs.iter().enumerate() {
        if at > 0 {
            spans.push(Span::styled(super::CHEVRON, tokens::faint_style()));
        }
        let style = if at == last {
            tokens::normal_style()
        } else {
            tokens::muted_style()
        };
        spans.push(Span::styled(crumb.clone(), style));
    }
    Line::from(spans)
}

/// Item (a)'s trail.
fn powerline(crumbs: &[String]) -> Line<'static> {
    let Some((current, ancestors)) = crumbs.split_last() else {
        return Line::from(Vec::<Span<'static>>::new());
    };
    let (ancestor, live, window) = (
        ancestor_bg(),
        current_bg(),
        tokens::modal_fill(tokens::layer1_bg()),
    );

    let mut spans: Vec<Span<'static>> = Vec::new();
    for (at, crumb) in ancestors.iter().enumerate() {
        if at > 0 {
            spans.push(inner_separator());
        }
        spans.push(segment(crumb, ancestor, false));
    }
    // *"There is also a chevron between the last breadcrumb and the current one."* With no
    // ancestors there is no run behind the current crumb, so the transition comes off the
    // window itself and the trail still starts with a clean edge.
    let behind = if ancestors.is_empty() {
        window
    } else {
        ancestor
    };
    spans.push(transition(behind, live));
    spans.push(segment(current, live, true));
    // *"the background must finish with a chevron"* — the current run closing onto the window.
    spans.push(transition(live, window));
    Line::from(spans)
}

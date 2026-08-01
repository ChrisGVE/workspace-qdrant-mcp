//! The candidate rules for `strong`. **Rules only — the frame that shows them is
//! [`crate::styles::palette`].**
//!
//! Chris, 20260801, looking at Everforest: *"I don't quite understand where is the issue with
//! strong, it is yellowish and quite far from the accent or the success. So did you handle it
//! differently? asking because what I'd like to see is what is the consistent approach."*
//!
//! # Nothing was handled differently, and Everforest is the wrong theme to see the problem on
//!
//! What ships today is [`Candidate::Extrapolate`] — the ladder continued past the foreground
//! and scaled back into gamut — and that is what he is looking at. Everforest's `fg` is a warm
//! cream (`#d3c6aa`), so `strong` comes out `#f0e0be`, comfortably separated from body and
//! nowhere near the greens. **On Everforest the current rule works.**
//!
//! The ΔE 6.6 collision reported earlier is not about what ships. It is the risk of
//! [`Candidate::Accent`] at `k = 0.55`, a rule that has *not* been built: on Everforest that
//! lands `strong` at `#a7c38f` against a `success` of `#a7c080`, which is the one theme where
//! tinting toward accent runs into a hue §3 reserves.
//!
//! # The real defect is on the themes he has not been looking at
//!
//! `Extrapolate` is a consistent *rule* with an inconsistent *outcome*. `strong` is defined as
//! a rung past the foreground, and a theme whose foreground is already near white has nowhere
//! to go: MonokaiPro separates from body by **ΔE 1.0** — below the 2.3 just-noticeable
//! threshold, so the loudest rung in the design is invisible there. Dracula is 2.4, TokyoNight
//! 3.1, Mocha 3.6. Everforest is 9.9, which is why it looks fine.
//!
//! **That is what "consistent" has to mean here**: not one formula applied everywhere, but a
//! rule whose *worst case over the fifteen* is still legible. This frame is how that is judged
//! — read down a column and find the rule with no bad row.
//!
//! # Two numbers decide it, and they pull against each other
//!
//! - **`body`** — separation from the text `strong` has to stand out from. Below 2.3 the rung
//!   does nothing.
//! - **`res`** — distance to the nearest hue §3 reserves (the three health states and the
//!   selector). §3 reserves the selector *absolutely*, so a `strong` that lands on it is not a
//!   weak frame, it is a broken rule.
//!
//! A candidate that wins one and loses the other has not won.
//!
//! # This module used to draw a table, and that was the wrong instrument
//!
//! It rendered all four rules against all fifteen themes at once, so a rule was judged by its
//! worst row. Chris: *"that's not the way to proceed, this must be considered holistically
//! theme by theme."* He is right that a palette is judged whole: `strong` is not a colour on
//! its own, it is a colour among the ten the theme names and the ten rungs beneath it, and a
//! grid of swatches from fifteen different palettes shows none of those relationships. The
//! numbers below are still worth having — they are what a *frame* cannot say — but they belong
//! beside the colour, in the theme it lives in.

use ratatui::style::Color;

use crate::tokens::delta_e;

/// The distance from the reserved hues below which a candidate is refused.
///
/// Not a taste threshold: §3 reserves the selector absolutely and the health hues carry state,
/// so a `strong` that reads as either is a false signal rather than an ugly one. Twelve is
/// comfortably past "plainly a different colour" (5) with room for a weaker terminal to round
/// both values toward each other.
pub(crate) const RESERVED_FLOOR: f32 = 12.0;

/// How far [`Candidate::Adaptive`] will push toward the accent before giving up.
const ADAPTIVE_MAX: f32 = 0.55;

/// The four rules, in the order they were arrived at.
#[derive(Clone, Copy)]
pub(crate) enum Candidate {
    /// **What ships.** The ladder continued past `fg` along the `bg → fg` axis, scaled back so
    /// no channel clamps. Consistent formula; fails where `fg` is already near white.
    Extrapolate,
    /// `fg` mixed toward `theme.accent`. Differentiates by **hue** instead of luminance, which
    /// is in r02's grain — *"weight does the work; the colour only stops it receding"*.
    Accent(f32),
    /// `Accent`, backed off until the result clears [`RESERVED_FLOOR`] from every hue §3
    /// reserves. **The same shape as `tokens::family()`** — the lesser of what the rule wants
    /// and what the constraint permits — which is this crate's established answer to "a
    /// preference cannot exceed a limit".
    Adaptive,
}

impl Candidate {
    /// Named for **what the rule does**, not for its status.
    ///
    /// The first cut labelled these `ships` / `accent .35` / `accent .55` / `adaptive` and Chris
    /// read the table as four different tokens rather than four ways of computing one: *"I assume
    /// that on the table what you call ships is Strong? What I don't understand is the role of
    /// .35, .55 and adaptive."* Every column IS `strong`; a bare number names nothing.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Candidate::Extrapolate => "now: past fg",
            Candidate::Accent(k) if k < 0.45 => "35% accent",
            Candidate::Accent(_) => "55% accent",
            Candidate::Adaptive => "adaptive",
        }
    }

    pub(crate) fn resolve(self, theme: &ratatui_themes::ThemePalette) -> Color {
        match self {
            Candidate::Extrapolate => extrapolated(theme),
            Candidate::Accent(k) => mix(theme.fg, theme.accent, k),
            Candidate::Adaptive => {
                // Walk the mix back from the maximum until the reserved hues are clear. Coarse
                // on purpose: a finer step buys a fraction of a ΔE and hides the fact that the
                // answer is "as much accent as this theme can afford".
                let mut k = ADAPTIVE_MAX;
                while k > 0.0 {
                    let candidate = mix(theme.fg, theme.accent, k);
                    if nearest_reserved(candidate, theme) >= RESERVED_FLOOR {
                        return candidate;
                    }
                    k -= 0.05;
                }
                // No amount of accent is safe on this theme, so fall back to the rule that
                // never touches a hue at all.
                extrapolated(theme)
            }
        }
    }
}

pub(crate) const CANDIDATES: [Candidate; 4] = [
    Candidate::Extrapolate,
    Candidate::Accent(0.35),
    Candidate::Accent(0.55),
    Candidate::Adaptive,
];

/// The shipping rule, reproduced here so the frame compares rules rather than call paths.
///
/// It has to be recomputed rather than read from `tokens::strong()` because that answers for
/// the theme currently in force, and this frame asks the question of all fifteen at once.
fn extrapolated(theme: &ratatui_themes::ThemePalette) -> Color {
    let (bg, fg) = (channels(theme.bg), channels(theme.fg));
    let t = 100.0 / 85.0;
    let raw = [
        bg[0] + (fg[0] - bg[0]) * t,
        bg[1] + (fg[1] - bg[1]) * t,
        bg[2] + (fg[2] - bg[2]) * t,
    ];
    let overreach = (0..3).fold(1.0f32, |worst, i| {
        let headroom = if raw[i] > 255.0 {
            255.0 - bg[i]
        } else {
            return worst;
        };
        if headroom.abs() < f32::EPSILON {
            worst
        } else {
            worst.max((raw[i] - bg[i]) / headroom)
        }
    });
    let scale = |i: usize| {
        (bg[i] + (raw[i] - bg[i]) / overreach)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    Color::Rgb(scale(0), scale(1), scale(2))
}

/// The nearest of the four hues §3 reserves — the three health states and the selector.
pub(crate) fn nearest_reserved(colour: Color, theme: &ratatui_themes::ThemePalette) -> f32 {
    [theme.success, theme.warning, theme.error, theme.info]
        .iter()
        .map(|hue| delta_e(colour, *hue))
        .fold(f32::MAX, f32::min)
}

fn channels(colour: Color) -> [f32; 3] {
    match colour {
        Color::Rgb(r, g, b) => [r as f32, g as f32, b as f32],
        _ => [0.0; 3],
    }
}

fn mix(from: Color, to: Color, t: f32) -> Color {
    let (a, b) = (channels(from), channels(to));
    let blend = |i: usize| (a[i] + (b[i] - a[i]) * t).round().clamp(0.0, 255.0) as u8;
    Color::Rgb(blend(0), blend(1), blend(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipping rule is the one that fails, and it fails where `fg` is already near white.
    ///
    /// This is the finding the whole frame exists to make visible, and it is the one Everforest
    /// hides: there `Extrapolate` separates by ΔE 9.9 and looks entirely fine. Asserting it on
    /// MonokaiPro is the point — a guard written against the theme in the pantry would have
    /// passed while the defect sat in four others.
    #[test]
    fn the_shipping_rule_goes_invisible_where_the_foreground_is_already_near_white() {
        let theme = ratatui_themes::ThemeName::MonokaiPro.palette();
        let separation = delta_e(Candidate::Extrapolate.resolve(&theme), theme.fg);
        assert!(
            separation < 2.3,
            "MonokaiPro's `strong` now separates by {separation:.1} — if the ladder changed, \
             this frame's premise needs re-measuring, not this assertion relaxing"
        );
    }

    /// The adaptive rule clears the reserved floor on **every** theme.
    ///
    /// That is the property that would make it the consistent answer, and it is exactly what
    /// the fixed-`k` candidates cannot promise: `accent .55` lands ΔE 6.6 from Everforest's
    /// `success`, both being green. Backing off until the constraint is met is the same shape
    /// as `tokens::family()` — the lesser of what is wanted and what is permitted.
    #[test]
    fn the_adaptive_rule_never_lands_on_a_hue_that_is_reserved() {
        for name in ratatui_themes::ThemeName::all() {
            let theme = name.palette();
            let colour = Candidate::Adaptive.resolve(&theme);
            let reserved = nearest_reserved(colour, &theme);
            assert!(
                reserved >= RESERVED_FLOOR || delta_e(colour, extrapolated(&theme)) < 0.5,
                "{name:?}: adaptive landed {reserved:.1} from a reserved hue without falling \
                 back to the untinted rule"
            );
        }
    }

    /// A fixed accent mix does collide, so the adaptive rule is answering a real problem.
    ///
    /// Without this, `the_adaptive_rule_never_lands_on_a_hue_that_is_reserved` would pass just
    /// as well if nothing ever collided — a guard that cannot distinguish "the fix works" from
    /// "there was nothing to fix". Everforest is the case: its accent and its `success` are
    /// both green.
    #[test]
    fn a_fixed_accent_mix_collides_on_everforest_which_is_why_adaptive_exists() {
        let theme = ratatui_themes::ThemeName::Everforest.palette();
        let fixed = Candidate::Accent(0.55).resolve(&theme);
        assert!(
            nearest_reserved(fixed, &theme) < RESERVED_FLOOR,
            "Everforest no longer collides at k=0.55, so the adaptive rule has nothing to fix"
        );
    }
}

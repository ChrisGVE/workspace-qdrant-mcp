//! Four candidate rules for `strong`, every theme, side by side.
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
//! # The columns, and what would rule each out
//!
//! Two numbers decide it, and they pull against each other:
//!
//! - **`vs body`** — separation from the text `strong` has to stand out from. Below 2.3 the
//!   rung does nothing.
//! - **`vs res`** — distance to the nearest hue §3 reserves (the three health states and the
//!   selector). §3 reserves the selector *absolutely*, so a `strong` that lands on it is not a
//!   weak frame, it is a broken rule.
//!
//! A candidate that wins one column and loses the other has not won.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens::{self, delta_e};
use crate::widgets::config_table::fit;

/// The distance from the reserved hues below which a candidate is refused.
///
/// Not a taste threshold: §3 reserves the selector absolutely and the health hues carry state,
/// so a `strong` that reads as either is a false signal rather than an ugly one. Twelve is
/// comfortably past "plainly a different colour" (5) with room for a weaker terminal to round
/// both values toward each other.
const RESERVED_FLOOR: f32 = 12.0;

/// How far [`Candidate::Adaptive`] will push toward the accent before giving up.
const ADAPTIVE_MAX: f32 = 0.55;

/// The four rules, in the order they were arrived at.
#[derive(Clone, Copy)]
enum Candidate {
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
    fn label(self) -> &'static str {
        match self {
            Candidate::Extrapolate => "now: past fg",
            Candidate::Accent(k) if k < 0.45 => "35% accent",
            Candidate::Accent(_) => "55% accent",
            Candidate::Adaptive => "adaptive",
        }
    }

    fn resolve(self, theme: &ratatui_themes::ThemePalette) -> Color {
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

const CANDIDATES: [Candidate; 4] = [
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
fn nearest_reserved(colour: Color, theme: &ratatui_themes::ThemePalette) -> f32 {
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

/// Every candidate against every theme.
pub struct StrongCandidates;

impl Widget for StrongCandidates {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut lines = vec![
            Line::from(Span::styled(
                "`strong` — four ways of choosing ONE colour, measured against all fifteen themes.",
                tokens::strong_style(),
            )),
            Line::from(Span::styled(
                "EVERY COLUMN IS `strong`. They differ only in where its colour comes from:",
                tokens::muted_style(),
            )),
            Line::from(Span::styled(
                "  now: past fg   the grey ladder continued beyond the foreground — what the \
                 code does today",
                tokens::faint_style(),
            )),
            Line::from(Span::styled(
                "  N% accent     the foreground mixed N of the way toward theme.accent \
                 (0% = fg exactly, 100% = accent)",
                tokens::faint_style(),
            )),
            Line::from(Span::styled(
                "  adaptive      as much accent as clears the reserved hues — 55% on fourteen \
                 themes, 35% on Everforest",
                tokens::faint_style(),
            )),
            Line::default(),
            Line::from(Span::styled(
                "The two numbers under each swatch: vs body = separation from the text `strong` \
                 must beat (2.3 = just",
                tokens::faint_style(),
            )),
            Line::from(Span::styled(
                "noticeable). vs res = distance to the nearest hue §3 reserves (health or \
                 selector). A rule that wins one",
                tokens::faint_style(),
            )),
            Line::from(Span::styled(
                "and loses the other has not won — so read DOWN a column and find the one with \
                 no bad row.",
                tokens::faint_style(),
            )),
            Line::default(),
        ];

        let mut header = vec![
            Span::styled(fit("THEME", 17), tokens::muted_style()),
            Span::styled(fit("body", 8), tokens::muted_style()),
        ];
        for candidate in CANDIDATES {
            header.push(Span::styled(
                fit(candidate.label(), 18),
                tokens::muted_style(),
            ));
        }
        lines.push(Line::from(header));

        let mut worst = [f32::MAX; 4];
        let mut floors = [f32::MAX; 4];
        for name in ratatui_themes::ThemeName::all() {
            let theme = name.palette();
            let mut spans = vec![
                Span::styled(fit(&format!("{name:?}"), 17), tokens::normal_style()),
                Span::styled("  ▏", Style::default().fg(tokens::rule_internal())),
                Span::styled("   ", Style::default().bg(theme.fg)),
                Span::styled("▕  ", Style::default().fg(tokens::rule_internal())),
            ];

            for (i, candidate) in CANDIDATES.iter().enumerate() {
                let colour = candidate.resolve(&theme);
                let separation = delta_e(colour, theme.fg);
                let reserved = nearest_reserved(colour, &theme);
                worst[i] = worst[i].min(separation);
                floors[i] = floors[i].min(reserved);

                spans.push(Span::styled(
                    "▏",
                    Style::default().fg(tokens::rule_internal()),
                ));
                spans.push(Span::styled("   ", Style::default().bg(colour)));
                spans.push(Span::styled(
                    "▕",
                    Style::default().fg(tokens::rule_internal()),
                ));
                // The two numbers, each coloured by whether it passes its own floor. A wall of
                // figures is unreadable; the eye should land on the failures.
                spans.push(Span::styled(
                    format!("{separation:>5.1}"),
                    if separation < 2.3 {
                        Style::default().fg(tokens::offline())
                    } else {
                        tokens::faint_style()
                    },
                ));
                spans.push(Span::styled(
                    format!("{reserved:>6.1}  "),
                    if reserved < RESERVED_FLOOR {
                        Style::default().fg(tokens::offline())
                    } else {
                        tokens::faint_style()
                    },
                ));
            }
            lines.push(Line::from(spans));
        }

        lines.push(Line::default());
        let mut summary = vec![Span::styled(
            fit("WORST OF THE 15", 17 + 8),
            Style::default()
                .fg(tokens::header())
                .add_modifier(Modifier::BOLD),
        )];
        for i in 0..CANDIDATES.len() {
            let bad = worst[i] < 2.3 || floors[i] < RESERVED_FLOOR;
            summary.push(Span::styled(
                fit(&format!("{:>8.1}{:>6.1}  ", worst[i], floors[i]), 18),
                if bad {
                    Style::default().fg(tokens::offline())
                } else {
                    Style::default().fg(tokens::healthy())
                },
            ));
        }
        lines.push(Line::from(summary));

        Paragraph::new(lines).render(area, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[PropInfo {
        name: "candidates",
        ty: "[Candidate; 4]",
        description: "ships / accent .35 / accent .55 / adaptive — none of them adopted",
    }];

    struct Candidates;

    impl Ingredient for Candidates {
        fn tab(&self) -> &str {
            "Styles"
        }
        // An instrument: this is how the decision gets taken, not part of the language. It
        // leaves the tab the moment `strong` is settled.
        fn section(&self) -> Option<&str> {
            Some("Instruments")
        }
        fn group(&self) -> &str {
            "Strong"
        }
        fn name(&self) -> &str {
            "Four rules, fifteen themes"
        }
        fn source(&self) -> &str {
            "wqm_tui::styles::strong"
        }
        fn description(&self) -> &str {
            "Candidate rules for the top rung, with their worst case over every bundled theme"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            StrongCandidates.render(area, buf);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![Box::new(Candidates)]
    }
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

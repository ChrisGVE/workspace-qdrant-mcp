//! The bundled-theme gallery — every theme `ratatui-themes` carries, ten fields each.
//!
//! **A theme is chosen by looking**, which is the whole reason this exists: fifteen names in a
//! list settle nothing, and the field a name suggests is only *typically* the hue it carries.
//!
//! # What used to be here, and why it is gone
//!
//! This module was a *comparison* sheet: base16's sixteen positional slots (`base00`–`base0F`)
//! against `ratatui-themes`' ten semantic fields, one theme at a time, with five base16
//! schemes transcribed by hand to compare against. §10 of VISUAL-LANGUAGE closed that
//! question — the semantic source was chosen, base16 as a format was closed with it — and a
//! comparison whose question is answered is a transcription that can only drift.
//!
//! What the comparison established is kept where it is read: the mapping and its measurements
//! are in VISUAL-LANGUAGE §10, and the invariant `Palette::Bundled` rests on is kept below as
//! a test over all fifteen themes rather than as prose about four of them.
//!
//! One finding from that comparison is worth keeping in words, because nothing measures it
//! any more: `ratatui-themes`' `muted` for Catppuccin is `#6c7086` — `overlay0`, which is not
//! one of base16's sixteen slots. The crate was built from the upstream 26-colour palette
//! rather than from the base16 subset, so it can name neutrals base16 has no slot for. That is
//! why four named anchors were enough to interpolate eleven rungs from.
//!
//! Gated behind `themes-preview` — an instrument, not part of the surface.

// Everything below belongs to the gallery, and the gallery is feature-gated — so these are
// too, rather than being carried by a build that cannot reach them.
#[cfg(feature = "themes-preview")]
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

#[cfg(feature = "themes-preview")]
use crate::tokens;
#[cfg(feature = "themes-preview")]
use crate::widgets::config_table::fit;

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    #[cfg(feature = "themes-preview")]
    use super::*;
    use tui_pantry::Ingredient;
    #[cfg(feature = "themes-preview")]
    use tui_pantry::PropInfo;

    #[cfg(feature = "themes-preview")]
    const PROPS: &[PropInfo] = &[PropInfo {
        name: "themes",
        ty: "Vec<ThemeName>",
        description: "Every theme `ratatui-themes` carries — a CLOSED enum, so this list is the whole set",
    }];

    #[cfg(feature = "themes-preview")]
    struct Gallery;

    #[cfg(feature = "themes-preview")]
    impl Ingredient for Gallery {
        fn tab(&self) -> &str {
            "Styles"
        }
        fn section(&self) -> Option<&str> {
            Some("Instruments")
        }
        fn group(&self) -> &str {
            "Theme Sources"
        }
        fn name(&self) -> &str {
            "Gallery — all 15"
        }
        fn source(&self) -> &str {
            "wqm_tui::styles::theme_sheet"
        }
        fn description(&self) -> &str {
            "Every theme ratatui-themes carries, ten fields each — choose by looking, not by name"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            ThemeGallery.render(area, buf);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        #[cfg(feature = "themes-preview")]
        {
            vec![Box::new(Gallery)]
        }
        #[cfg(not(feature = "themes-preview"))]
        {
            Vec::new()
        }
    }
}

#[cfg(all(test, feature = "themes-preview"))]
mod themes_preview {
    use ratatui::style::Color;

    /// A colour as `#rrggbb`, so two of them can be compared as strings.
    fn hex(colour: Color) -> String {
        match colour {
            Color::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
            _ => "—".to_string(),
        }
    }

    /// The invariant `Palette::Bundled` will be built on, checked against all fifteen themes.
    ///
    /// §15 chose the semantic source knowing it names **four** neutrals against r02's eleven
    /// rungs, so the ladder has to be interpolated from theme data the way `Derived`
    /// interpolates it from `OSC 11`/`OSC 10`. That only works if the four are *ordered* —
    /// `bg` and `fg` as the endpoints, `selection` and `muted` as interior anchors. Nothing in
    /// `ThemePalette`'s type says so: all ten fields are plain `Color`, and a theme whose
    /// `muted` sat outside `[bg, fg]` would silently fold two rungs together.
    ///
    /// It also checks that no theme collapses two roles onto one value, which is what
    /// `Palette::Theme` does with the sixteen ANSI slots (§8.5, six rungs lost) and is the
    /// failure this source was chosen to avoid.
    ///
    /// Both halves hold on every theme today, on **both polarities** — three of the fifteen
    /// are light themes, where the whole ordering inverts and the betweenness does not.
    #[test]
    fn every_theme_gives_ten_distinct_roles_and_four_ordered_neutrals() {
        fn luma(colour: Color) -> f32 {
            match colour {
                Color::Rgb(r, g, b) => 0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32,
                other => panic!("a bundled theme must be RGB, got {other:?}"),
            }
        }

        for name in ratatui_themes::ThemeName::all() {
            let p = name.palette();
            let roles = [
                p.accent,
                p.secondary,
                p.bg,
                p.fg,
                p.muted,
                p.selection,
                p.error,
                p.warning,
                p.success,
                p.info,
            ];
            let mut distinct: Vec<String> = roles.iter().map(|c| hex(*c)).collect();
            distinct.sort();
            distinct.dedup();
            assert_eq!(
                distinct.len(),
                roles.len(),
                "{name:?} spends one colour on two roles: {distinct:?}"
            );

            // The interior anchors sit strictly between the endpoints — in that order, and
            // whichever way round the polarity puts them.
            let (bg, fg) = (luma(p.bg), luma(p.fg));
            let (low, high) = (bg.min(fg), bg.max(fg));
            for (field, value) in [("selection", luma(p.selection)), ("muted", luma(p.muted))] {
                assert!(
                    value > low && value < high,
                    "{name:?}: {field} at luma {value:.0} is outside [{low:.0}, {high:.0}], \
                     so the ladder cannot anchor on it"
                );
            }
            assert!(
                (luma(p.selection) - bg).abs() < (luma(p.muted) - bg).abs(),
                "{name:?}: selection must be the anchor NEARER the background — it is the \
                 cursor tint, and muted is a text rung"
            );
        }
    }
}

/// Every theme `ratatui-themes` carries, one row each — the gallery `palette.sh` is to the
/// palette work: the whole set on one screen, so a choice is made by looking rather than by
/// reading names.
///
/// Gated on the optional dependency because it enumerates the crate's own closed set.
#[cfg(feature = "themes-preview")]
pub struct ThemeGallery;

#[cfg(feature = "themes-preview")]
impl ThemeGallery {
    /// The crate's set, in its own order. `ThemeName` is a **closed enum**: this list cannot
    /// be extended from outside, which is the finding that decides whether we own the
    /// registry — see the module docs on `ThemePicker`.
    pub fn themes() -> Vec<ratatui_themes::ThemeName> {
        ratatui_themes::ThemeName::all().to_vec()
    }
}

#[cfg(feature = "themes-preview")]
impl Widget for ThemeGallery {
    fn render(self, area: Rect, buf: &mut Buffer) {
        const FIELDS: [&str; 10] = [
            "accent", "second", "bg", "fg", "muted", "select", "error", "warn", "ok", "info",
        ];

        let mut lines = vec![
            Line::from(Span::styled(
                "ratatui-themes — every theme it carries, and every field of each",
                tokens::normal_style().add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                "ThemeName is a CLOSED enum: this set cannot be extended from outside.",
                tokens::faint_style(),
            )),
            Line::default(),
            Line::from(vec![
                Span::styled(fit("THEME", 20), Style::default().fg(tokens::header())),
                Span::styled(fit("VAR", 7), Style::default().fg(tokens::header())),
                Span::styled(
                    FIELDS.iter().map(|f| format!("{f:<8}")).collect::<String>(),
                    Style::default().fg(tokens::header()),
                ),
            ]),
        ];

        for name in Self::themes() {
            let theme = ratatui_themes::Theme::new(name);
            let p = theme.palette();
            let mut spans = vec![
                Span::styled(fit(&format!("{name:?}"), 20), tokens::normal_style()),
                Span::styled(
                    fit(if theme.is_dark() { "dark" } else { "light" }, 7),
                    tokens::muted_style(),
                ),
            ];
            for colour in [
                p.accent,
                p.secondary,
                p.bg,
                p.fg,
                p.muted,
                p.selection,
                p.error,
                p.warning,
                p.success,
                p.info,
            ] {
                // Four cells of colour and four of gap: wide enough to judge a hue, narrow
                // enough that all ten fit beside the name.
                spans.push(Span::styled("    ", Style::default().bg(colour)));
                spans.push(Span::raw("    "));
            }
            lines.push(Line::from(spans));
        }

        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            "Every field is non-Option, so every theme has all ten — but the hue is only \
             \"typically\" the one the name suggests.",
            tokens::faint_style(),
        )));

        Paragraph::new(lines).render(area, buf);
    }
}

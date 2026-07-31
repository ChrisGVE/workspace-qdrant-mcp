//! The bundled-theme comparison sheet — both representations of one theme, side by side,
//! every swatch labelled with the name its own source gives it.
//!
//! This exists to answer one question with frames rather than prose: **are the colours
//! represented the same way across every theme?** The two columns answer it differently.
//!
//! - **base16** names slots *positionally* (`base00`–`base0F`). Every conformant scheme
//!   defines all sixteen, so presence is guaranteed; the hue is a **guideline** the spec
//!   declines to enforce, and the scheme author's own name for a slot is optional —
//!   Catppuccin names all sixteen, Solarized names none.
//! - **`ratatui-themes`** names fields *semantically* (`error`, `warning`, `success`,
//!   `info`, `accent`, `secondary`, plus `bg`/`fg`/`muted`/`selection`). All ten are
//!   non-`Option`, so presence is guaranteed by the type — but the hue is only "typically"
//!   what the name suggests, per its own docs.
//!
//! Neither guarantees red is red. Both guarantee something plays red's part, which is what
//! our own language actually asks for: §4 wants *offline*, not *red*.
//!
//! Gated behind `themes-preview` — an instrument, not an adoption.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::themes::{Scheme, Variant};
use crate::tokens;
use crate::widgets::config_table::fit;

/// A colour as `#rrggbb`, or `—` for anything that is not RGB. Nothing in either source
/// should produce a non-RGB colour; if this ever renders a dash, that is the finding.
fn hex(colour: Color) -> String {
    match colour {
        Color::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
        _ => "—".to_string(),
    }
}

/// Two cells of the colour itself. The swatch is the only part of this sheet that is
/// evidence; every other column is a label about it.
fn swatch(colour: Color) -> Span<'static> {
    Span::styled("  ", Style::default().bg(colour))
}

/// Which of OUR roles a base16 slot feeds, under full paint. Empty where the slot has no
/// job in this design — `base0D`/`base0E`/`base0F` are syntax hues and we render no syntax.
const OUR_ROLE: [&str; 16] = [
    "layer 0",
    "layer 1*",
    "layer 1* / sel fill",
    "faint",
    "muted",
    "normal",
    "strong",
    "strong+",
    "offline",
    "—",
    "degraded",
    "healthy",
    "selector",
    "—",
    "—",
    "—",
];

pub struct ThemeSheet {
    scheme: Scheme,
    /// The semantic half, when `ratatui-themes` carries the same theme. `None` says the
    /// theme exists in one source and not the other, which is itself worth seeing.
    semantic: Option<&'static [(&'static str, Color, &'static str)]>,
}

impl ThemeSheet {
    pub fn new(scheme: Scheme) -> Self {
        Self {
            scheme,
            semantic: None,
        }
    }

    pub fn with_semantic(mut self, rows: &'static [(&'static str, Color, &'static str)]) -> Self {
        self.semantic = Some(rows);
        self
    }

    fn header(&self) -> Vec<Line<'static>> {
        let variant = match self.scheme.variant {
            Variant::Dark => "dark",
            Variant::Light => "light",
        };
        let headroom = self.scheme.headroom_rungs();
        let layer1 = self.scheme.layer1_slot().slot.to_string();

        vec![
            Line::from(vec![
                Span::styled(
                    self.scheme.name.to_string(),
                    tokens::normal_style().add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("  {variant}  "), tokens::muted_style()),
                Span::styled(self.scheme.author.to_string(), tokens::faint_style()),
            ]),
            Line::from(vec![
                Span::styled("layer 1 from ", tokens::muted_style()),
                Span::styled(layer1, tokens::normal_style()),
                Span::styled("   headroom above fg: ", tokens::muted_style()),
                Span::styled(format!("{headroom} rung(s)"), tokens::normal_style()),
                Span::styled(
                    if self.scheme.lighter_background_is_actually_lighter() {
                        ""
                    } else {
                        "   ⚠ base01 is DARKER than base00"
                    },
                    tokens::strong_style(),
                ),
            ]),
            Line::default(),
            Line::from(vec![Span::styled(
                format!(
                    "{:<8}{:<4}{:<10}{:<17}{:<21}{}",
                    "SLOT", "", "HEX", "SCHEME'S", "OURS", "BASE16 SPEC ROLE"
                ),
                Style::default().fg(tokens::header()),
            )]),
        ]
    }

    fn base16_rows(&self) -> Vec<Line<'static>> {
        self.scheme
            .slots
            .iter()
            .enumerate()
            .map(|(i, slot)| {
                Line::from(vec![
                    Span::styled(format!("{:<8}", slot.slot), tokens::normal_style()),
                    swatch(slot.colour),
                    Span::raw("  "),
                    Span::styled(format!("{:<10}", hex(slot.colour)), tokens::faint_style()),
                    Span::styled(
                        fit(slot.scheme_name.unwrap_or("—"), 17),
                        if slot.scheme_name.is_some() {
                            tokens::normal_style()
                        } else {
                            tokens::faint_style()
                        },
                    ),
                    Span::styled(fit(OUR_ROLE[i], 21), tokens::normal_style()),
                    Span::styled(slot.spec_role.to_string(), tokens::muted_style()),
                ])
            })
            .collect()
    }

    fn semantic_rows(&self) -> Vec<Line<'static>> {
        let Some(rows) = self.semantic else {
            return vec![Line::from(Span::styled(
                "not carried by ratatui-themes",
                tokens::faint_style(),
            ))];
        };
        rows.iter()
            .map(|(field, colour, doc)| {
                Line::from(vec![
                    Span::styled(format!("{field:<11}"), tokens::normal_style()),
                    swatch(*colour),
                    Span::raw("  "),
                    Span::styled(format!("{:<10}", hex(*colour)), tokens::faint_style()),
                    Span::styled((*doc).to_string(), tokens::muted_style()),
                ])
            })
            .collect()
    }
}

impl Widget for ThemeSheet {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut lines = self.header();
        lines.extend(self.base16_rows());
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            format!(
                "{:<11}{:<4}{:<10}{}",
                "FIELD", "", "HEX", "ratatui-themes ThemePalette — semantic, all non-Option"
            ),
            Style::default().fg(tokens::header()),
        )));
        lines.extend(self.semantic_rows());
        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            "* layer 1 is chosen by measured depth, never by slot number — base01 is \
             darker than base00 in half of a four-scheme sample",
            tokens::faint_style(),
        )));

        Paragraph::new(lines).render(area, buf);
    }
}

/// The four themes carried by BOTH sources, with `ratatui-themes`' own field values.
///
/// Written out as data rather than read from the crate at render time so the sheet builds
/// without the optional dependency; `themes_preview::verify` checks these against the live
/// crate when the feature IS enabled, so the table cannot drift silently.
pub mod semantic {
    use ratatui::style::Color;

    pub type Rows = &'static [(&'static str, Color, &'static str)];

    macro_rules! rows {
        ($($field:literal $hex:literal $doc:literal),* $(,)?) => {
            &[$(($field, Color::Rgb(
                (($hex as u32) >> 16 & 0xff) as u8,
                (($hex as u32) >> 8 & 0xff) as u8,
                ($hex as u32 & 0xff) as u8), $doc)),*]
        };
    }

    pub const CATPPUCCIN_MOCHA: Rows = rows![
        "accent"     0x89b4fa "Primary accent — highlights, active elements",
        "secondary"  0xf5c2e7 "Secondary accent — less prominent elements",
        "bg"         0x1e1e2e "Main background",
        "fg"         0xcdd6f4 "Primary foreground / text",
        "muted"      0x6c7086 "Dimmed text, comments, placeholders",
        "selection"  0x313244 "Selection / highlight background",
        "error"      0xf38ba8 "Errors, deletions (typically red)",
        "warning"    0xf9e2af "Warnings, pending (typically yellow/orange)",
        "success"    0xa6e3a1 "Success, additions (typically green)",
        "info"       0x94e2d5 "Information, links (typically blue/cyan)",
    ];

    pub const DRACULA: Rows = rows![
        "accent"     0xbd93f9 "Primary accent — highlights, active elements",
        "secondary"  0xff79c6 "Secondary accent — less prominent elements",
        "bg"         0x282a36 "Main background",
        "fg"         0xf8f8f2 "Primary foreground / text",
        "muted"      0x6272a4 "Dimmed text, comments, placeholders",
        "selection"  0x44475a "Selection / highlight background",
        "error"      0xff5555 "Errors, deletions (typically red)",
        "warning"    0xffb86c "Warnings, pending (typically yellow/orange)",
        "success"    0x50fa7b "Success, additions (typically green)",
        "info"       0x8be9fd "Information, links (typically blue/cyan)",
    ];
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    // The module path is used only by the preview constructors below, so it is imported
    // here rather than at file scope — where a bare build, which compiles neither this
    // module nor its callers, would see it unused.
    use crate::themes;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "scheme",
            ty: "themes::Scheme",
            description: "A base16 scheme from the MIT tinted-theming YAML",
        },
        PropInfo {
            name: "semantic",
            ty: "Option<Rows>",
            description: "ratatui-themes' ThemePalette for the same theme, where it has one",
        },
    ];

    /// The whole set on one screen — the gallery entry, distinct from the per-theme sheets.
    #[cfg(feature = "themes-preview")]
    struct GalleryEntry;

    #[cfg(feature = "themes-preview")]
    impl Ingredient for GalleryEntry {
        fn group(&self) -> &str {
            "Theme Sources"
        }
        fn name(&self) -> &str {
            "Gallery — all 15"
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::theme_sheet::ThemeGallery"
        }
        fn description(&self) -> &str {
            "Every theme ratatui-themes carries, ten fields each — choose by looking, not by name"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            super::ThemeGallery.render(area, buf);
        }
    }

    struct Variant_(&'static str, &'static str, fn() -> ThemeSheet);

    impl Ingredient for Variant_ {
        fn group(&self) -> &str {
            "Theme Sources"
        }
        fn name(&self) -> &str {
            self.0
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::theme_sheet"
        }
        fn description(&self) -> &str {
            self.1
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            (self.2)().render(area, buf);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            #[cfg(feature = "themes-preview")]
            Box::new(GalleryEntry),
            Box::new(Variant_(
                "Catppuccin Mocha",
                "Both representations of the theme Chris runs. The only scheme here that names all sixteen slots",
                || {
                    ThemeSheet::new(themes::catppuccin_mocha())
                        .with_semantic(semantic::CATPPUCCIN_MOCHA)
                },
            )),
            Box::new(Variant_(
                "Catppuccin Latte",
                "Mocha's light counterpart. base01 is `mantle` on BOTH — directionally right here, wrong on Mocha",
                || ThemeSheet::new(themes::catppuccin_latte()),
            )),
            Box::new(Variant_(
                "Dracula",
                "The counter-example: base06 EQUALS base05, and the slot named Blue holds a violet",
                || ThemeSheet::new(themes::dracula()).with_semantic(semantic::DRACULA),
            )),
            Box::new(Variant_(
                "Gruvbox dark, medium",
                "The well-behaved case — a monotonic ramp, two rungs of headroom, base01 genuinely lighter",
                || ThemeSheet::new(themes::gruvbox_dark_medium()),
            )),
            Box::new(Variant_(
                "Solarized Light",
                "The polarity check: the ramp runs light-to-dark and names none of its slots",
                || ThemeSheet::new(themes::solarized_light()),
            )),
        ]
    }
}

/// Checks the hand-written [`semantic`] table against the live crate. Only compiled when
/// the optional dependency is present, so the table stays usable without it while never
/// being allowed to drift from it.
#[cfg(all(test, feature = "themes-preview"))]
mod themes_preview {
    use super::*;

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

    #[test]
    fn the_hand_written_semantic_table_matches_the_crate() {
        // This check earned its place on its first run: `secondary` had been guessed as
        // Catppuccin's mauve and the crate actually uses its pink. A table transcribed by
        // hand is a claim about another crate's data, and claims get verified.
        for (theme, expected) in [
            (
                ratatui_themes::ThemeName::CatppuccinMocha,
                semantic::CATPPUCCIN_MOCHA,
            ),
            (ratatui_themes::ThemeName::Dracula, semantic::DRACULA),
        ] {
            let p = theme.palette();
            let live = [
                ("accent", p.accent),
                ("secondary", p.secondary),
                ("bg", p.bg),
                ("fg", p.fg),
                ("muted", p.muted),
                ("selection", p.selection),
                ("error", p.error),
                ("warning", p.warning),
                ("success", p.success),
                ("info", p.info),
            ];
            for ((field, colour, _), (name, actual)) in expected.iter().zip(live) {
                assert_eq!(*field, name);
                assert_eq!(
                    *colour, actual,
                    "{theme:?} {name} drifted from ratatui-themes"
                );
            }
        }
    }

    #[test]
    fn the_semantic_source_reaches_past_base16_into_the_theme_s_own_palette() {
        // A quality signal worth pinning: ratatui-themes' `muted` for Catppuccin is
        // #6c7086 — `overlay0` — which is NOT one of the sixteen base16 slots. So it was
        // built from the upstream 26-colour Catppuccin palette rather than from the base16
        // subset, and it can therefore express neutrals base16 has no slot for.
        let muted = ratatui_themes::ThemeName::CatppuccinMocha.palette().muted;
        let base16 = crate::themes::catppuccin_mocha();
        assert!(
            !base16.slots.iter().any(|s| s.colour == muted),
            "if this ever fails, ratatui-themes switched to the base16 subset"
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::themes;

    #[test]
    fn every_slot_is_labelled_with_the_name_its_own_source_gives_it() {
        // The sheet's whole purpose: a swatch with no name attached settles nothing.
        let area = Rect::new(0, 0, 104, 40);
        let mut buf = Buffer::empty(area);
        ThemeSheet::new(themes::catppuccin_mocha())
            .with_semantic(semantic::CATPPUCCIN_MOCHA)
            .render(area, &mut buf);

        let text: String = (0..area.height)
            .flat_map(|y| (0..area.width).map(move |x| (x, y)))
            .map(|(x, y)| buf.cell((x, y)).expect("in area").symbol().to_string())
            .collect();

        // base16's positional name, the scheme's own name, and our role for the same slot.
        assert!(text.contains("base08"));
        assert!(text.contains("red"));
        assert!(text.contains("offline"));
        // …and the semantic source's field name for what is arguably the same colour.
        assert!(text.contains("error"));
    }
}

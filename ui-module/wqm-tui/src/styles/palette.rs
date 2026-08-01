//! The theme's ten fields in one frame, and the two of them the ladder is actually built on.
//!
//! Chris, 20260801: *"in the Colors section I would like to have a section Palette which would
//! show: the 10 roles and the 4 anchors, having them together will be helpful for visual
//! comparisons and to reduce (hopefully) the number of iterations when we start designing the
//! elements."*
//!
//! # The ten and the four overlap
//!
//! [`ThemePalette`] has exactly ten fields, and `bg` / `selection` / `muted` / `fg` are four of
//! them — *"the ten roles and the four anchors"* is this crate's own phrasing for that
//! (`theme_sheet::every_theme_gives_ten_distinct_roles_and_four_ordered_neutrals`). So the
//! frame is ten swatches, then those four again in ladder order: one frame, two readings of the
//! same ten colours, rather than fourteen swatches of which four are duplicates.
//!
//! # Why the second reading has two columns and not one
//!
//! Chris asked the obvious question — *"why do we have a column THE THEME'S and another for
//! OURS? why wouldn't we use the same?"* — and the answer turned out to be measurable rather
//! than a matter of taste.
//!
//! §15 says the ladder is interpolated *"from the theme's own bg and fg"*, and that is exactly
//! what [`crate::tokens`] does: `ladder_endpoints` takes **two** of the four neutrals. The
//! other two, `selection` and `muted`, are checked to be *ordered* inside that span — so a
//! ladder **could** anchor on them — and nothing does.
//!
//! It looks like an oversight, and using the theme's own values instead would look like exactly
//! what §15 asked for. **It is not, because the two `muted`s are not the same thing.** r02's
//! `muted` rung is *"the default posture of most of the screen"* — readable body text at 62% of
//! the way to the foreground. A theme's `muted` field is the subtle-UI tint an editor paints
//! comments and borders with: for Catppuccin it is `overlay0`, which sits at **38%** — below
//! our `faint` rung. Substituting it would put most of a screen's text *below* the rung reserved
//! for de-emphasised metadata, inverting the emphasis ladder.
//!
//! Measured over all fifteen bundled themes
//! ([`tests::the_themes_own_neutrals_do_not_sit_where_r02_puts_its_rungs`]): the theme's
//! `selection` lands in the layer band our `cursor_bg` occupies on **5 of 15**, and its `muted`
//! lands in the text band on **2 of 15**. The names coincide; the positions do not.
//!
//! So both columns stay, and the frame prints **where on our ladder the theme's value falls**,
//! which is the number that makes the divergence legible instead of merely visible.
//!
//! # The swatches are literal theme colours
//!
//! Every swatch here is a `Color::Rgb` taken from the theme rather than a token, because the
//! subject *is* the theme's data — the same property `theme_sheet`'s gallery has, and the one
//! case where the PNG capture path is byte-exact about hue. The prose around them goes through
//! `tokens`, so the frame degrades with everything else where an encoding refuses colour.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens;
use crate::widgets::config_table::fit;

/// Width of the field-name column, so both blocks start their swatches on one column.
const NAME: usize = 12;
/// Width of one colour's swatch: fill, sample, resolved value. The second block draws two of
/// them per row and heads each with a label of this width, so the two must agree.
const SWATCH: usize = 17;
/// Width of the "where does this fall on our ladder" column.
const RUNG: usize = 9;

/// The ten fields in the order that answers *"what may I reach for"*: the six hues first,
/// four of which our design has already claimed, then the four neutrals.
///
/// **Not the struct's field order**, and deliberately: `ThemePalette` interleaves them
/// (`accent, secondary, bg, fg, muted, selection, error, warning, success, info`), which is
/// fine for a gallery row and wrong for a lookup. The claims are the other half — a field with
/// no claim is headroom, and §3 forbids two roles landing on one hue, so which are spoken for
/// is the first thing to know.
type Field = (
    &'static str,
    fn(&ratatui_themes::ThemePalette) -> Color,
    &'static str,
);

const FIELDS: [Field; 10] = [
    ("accent", |p| p.accent, "unclaimed — headroom"),
    ("secondary", |p| p.secondary, "unclaimed — headroom"),
    ("error", |p| p.error, "offline ○"),
    ("warning", |p| p.warning, "degraded ▲, and stale timings"),
    ("success", |p| p.success, "healthy ●"),
    (
        "info",
        |p| p.info,
        "selector — reserved absolutely (§3), and chosen over accent for distance",
    ),
    (
        "bg",
        |p| p.bg,
        "the screen's full paint (§15), and the ladder's low end",
    ),
    ("selection", |p| p.selection, "the theme's own cursor tint"),
    ("muted", |p| p.muted, "the theme's own de-emphasis"),
    (
        "fg",
        |p| p.fg,
        "the ladder's high end — `normal` interpolates to exactly this",
    ),
];

/// The four neutrals in ladder order, each beside the rung our ladder puts in that place.
///
/// The token is named rather than its percentage, because a percentage written here would be
/// the transcription this module exists to remove — `palette_reference` enumerates the ladder
/// by percentage and is the place for that.
type Anchor = (
    &'static str,
    fn(&ratatui_themes::ThemePalette) -> Color,
    &'static str,
    fn() -> Color,
    &'static str,
);

const ANCHORS: [Anchor; 4] = [
    (
        "bg",
        |p| p.bg,
        "tokens::screen_bg",
        || tokens::screen_bg().unwrap_or(Color::Reset),
        "an endpoint; equal by construction",
    ),
    (
        "selection",
        |p| p.selection,
        "tokens::cursor_bg",
        tokens::cursor_bg,
        "the same word, a different place",
    ),
    (
        "muted",
        |p| p.muted,
        "tokens::muted",
        tokens::muted,
        "the same word, a different place",
    ),
    (
        "fg",
        |p| p.fg,
        "tokens::normal",
        tokens::normal,
        "an endpoint; equal by construction",
    ),
];

/// The whole palette, both readings.
pub struct PaletteFrame;

impl Widget for PaletteFrame {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let Some(theme) = tokens::active_theme() else {
            Paragraph::new(unavailable()).render(area, buf);
            return;
        };

        let mut lines = vec![
            Line::from(Span::styled(
                "The ten fields the theme names, and the two the ladder is built on.",
                tokens::strong_style(),
            )),
            Line::from(Span::styled(
                "Live from tokens::active_theme() — this frame paints what a screen paints.",
                tokens::faint_style(),
            )),
            Line::default(),
            heading("THE TEN ROLES"),
        ];

        for (name, field, claim) in FIELDS {
            let mut spans = vec![Span::styled(
                format!("  {}", fit(name, NAME)),
                tokens::normal_style(),
            )];
            spans.extend(swatch(field(&theme)));
            spans.push(Span::styled(claim, tokens::faint_style()));
            lines.push(Line::from(spans));
        }

        lines.push(Line::default());
        lines.push(heading("THE FOUR NEUTRALS, IN LADDER ORDER"));
        lines.push(Line::from(vec![
            Span::styled(format!("  {}", fit("", NAME)), tokens::faint_style()),
            Span::styled(fit("THE THEME'S", SWATCH), tokens::muted_style()),
            Span::styled(fit("ON OURS", RUNG), tokens::muted_style()),
            Span::styled(fit("OURS", SWATCH), tokens::muted_style()),
        ]));

        for (name, field, token, rung, note) in ANCHORS {
            let mut spans = vec![Span::styled(
                format!("  {}", fit(name, NAME)),
                tokens::normal_style(),
            )];
            let theirs = field(&theme);
            spans.extend(swatch(theirs));
            // Where the THEME's value falls on our ladder, in r02's own percentage units. This
            // is the number that answers "why two columns": the two are not a disagreement
            // about a colour, they are the same word at different places on the scale.
            spans.push(Span::styled(
                fit(&format!("{:.0}%", ladder_percent(theirs, &theme)), RUNG),
                tokens::faint_style(),
            ));
            spans.extend(swatch(rung()));
            spans.push(Span::styled(
                format!("{token} — {note}"),
                tokens::faint_style(),
            ));
            lines.push(Line::from(spans));
        }

        // The number in ON OURS is the whole reason there are two columns, so it is explained
        // where it is read rather than only in the module docs.
        lines.push(Line::from(Span::styled(
            "ON OURS = where the theme's value falls on r02's scale. cursor_bg is rung 19 and \
             muted is rung 62;",
            tokens::faint_style(),
        )));
        lines.push(Line::from(Span::styled(
            "over all fifteen themes the theme's selection lands in that band 5 times and its \
             muted 2 times.",
            tokens::faint_style(),
        )));

        Paragraph::new(lines).render(area, buf);
    }
}

/// Relative luminance, the measure r02 names its rungs in.
fn luma(colour: Color) -> f32 {
    match colour {
        Color::Rgb(r, g, b) => 0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32,
        // Only reachable through the rung column under a weak encoding; the theme's own fields
        // are always RGB (`theme_sheet` panics otherwise).
        _ => f32::NAN,
    }
}

/// Where a colour falls on **our** ladder, in r02's percentage units.
///
/// `NORMAL_RUNG` is 85 by construction — the percentage at which the ladder reaches the
/// foreground — so `bg` is 0 and `fg` is 85, and anything between reads directly against the
/// rung numbers `palette_reference` enumerates. Defined here rather than in `tokens` because it
/// is the *inverse* of the ladder and only a comparison needs it; a widget reaching for a rung
/// asks for the rung.
fn ladder_percent(colour: Color, theme: &ratatui_themes::ThemePalette) -> f32 {
    let (bg, fg) = (luma(theme.bg), luma(theme.fg));
    85.0 * (luma(colour) - bg) / (fg - bg)
}

/// A block heading — the frame has two readings and they must not run together.
fn heading(text: &'static str) -> Line<'static> {
    Line::from(Span::styled(
        text,
        Style::default()
            .fg(tokens::header())
            .add_modifier(Modifier::BOLD),
    ))
}

/// One colour three ways: as a fill, as text, and as the value it resolved to.
///
/// A hue has to work both ways round — as a block behind something and as glyphs on the
/// screen's own background — and a swatch that only shows the fill answers half the question.
/// The first version drew `Aa` *inside* the fill in a fixed dark foreground, which for `bg`
/// meant near-black on near-black: the sample was there and could show nothing.
///
/// Exactly [`SWATCH`] columns wide, so the two-colour rows line up under their headings.
fn swatch(colour: Color) -> Vec<Span<'static>> {
    vec![
        Span::styled("    ", Style::default().bg(colour)),
        Span::styled(" Aa ", Style::default().fg(colour)),
        Span::styled(format!("{}  ", fit(&hex(colour), 7)), tokens::muted_style()),
    ]
}

/// `#rrggbb`, or a dash where the colour is not an RGB triple.
///
/// Every bundled theme is RGB — `theme_sheet`'s test panics if one is not — so the dash is
/// reachable only through the *rung* column, where a weak encoding replaces our ladder with
/// indexed or reset values. That is a real state and the frame should say so rather than
/// pretend a hex.
fn hex(colour: Color) -> String {
    match colour {
        Color::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
        Color::Indexed(i) => format!("idx {i:<3}"),
        _ => format!("{colour:?}"),
    }
}

/// What to say when there is no theme in force.
///
/// Not an empty frame and not a panic: under `Palette::Theme`/`Indexed`/`Derived` the answer to
/// "what are the theme's ten fields" is genuinely *there are none*, and naming the palette that
/// is in force is what makes that legible instead of looking like a broken entry.
fn unavailable() -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled(
            "No bundled theme is in force, so there are no ten fields to show.",
            tokens::strong_style(),
        )),
        Line::default(),
        Line::from(Span::styled(
            format!(
                "The palette source is {}. Only Palette::Bundled renders from a theme; the \
                 others derive the ladder from ANSI slots or from the terminal's own \
                 background and foreground.",
                tokens::Palette::current().label()
            ),
            tokens::muted_style(),
        )),
    ]
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[PropInfo {
        name: "theme",
        ty: "ThemePalette",
        description: "The bundled theme in force — ten Color fields, none of them optional",
    }];

    struct Palette;

    impl Ingredient for Palette {
        // Styles, and NO section: this is vocabulary rather than an instrument, so it belongs
        // beside the `[colors.*]` groups the stylesheet contributes. `Instruments` is left to
        // `Palette Reference` and `Theme Sources` (§16).
        fn tab(&self) -> &str {
            "Styles"
        }
        // The group is `Colors` on purpose — Chris asked for this "in the Colors section", and
        // `[colors.*]` is exactly what the stylesheet names that group. The entry joins them.
        fn group(&self) -> &str {
            "Colors"
        }
        fn name(&self) -> &str {
            "Palette"
        }
        fn source(&self) -> &str {
            "wqm_tui::styles::palette"
        }
        fn description(&self) -> &str {
            "The theme's ten fields and the four neutrals in ladder order — live, so it cannot drift"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            PaletteFrame.render(area, buf);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![Box::new(Palette)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::Palette;

    /// Every field of the theme reaches the frame.
    ///
    /// The list is hand-ordered (six hues, then four neutrals) rather than taken from the
    /// struct, so a field added upstream — or one dropped from `FIELDS` in a reshuffle — would
    /// leave the frame quietly incomplete. Ten is the number `ThemePalette` has and the number
    /// Chris asked for, so it is worth asserting rather than counting by eye.
    #[test]
    fn all_ten_fields_are_present_and_none_is_shown_twice() {
        let theme = ratatui_themes::ThemeName::CatppuccinMocha.palette();
        let mut shown: Vec<String> = FIELDS.iter().map(|(name, _, _)| name.to_string()).collect();
        shown.sort();
        shown.dedup();
        assert_eq!(shown.len(), 10, "ten fields, each once: {shown:?}");

        let mut values: Vec<String> = FIELDS
            .iter()
            .map(|(_, field, _)| hex(field(&theme)))
            .collect();
        values.sort();
        values.dedup();
        assert_eq!(
            values.len(),
            10,
            "two of the ten resolve to one colour, so a field is wired to the wrong accessor"
        );
    }

    /// A swatch is exactly [`SWATCH`] columns, so the second block's two colours line up under
    /// their headings.
    ///
    /// Counted in **characters, not bytes** — this crate has already reported a three-column
    /// jitter that did not exist by measuring `▌` as three columns. The failure this guards is
    /// a heading that says `OURS` over a column that is not ours: nothing errors, the frame
    /// simply labels the wrong thing, and a hex read off the wrong column is exactly the kind
    /// of wrong number this frame exists to prevent.
    #[test]
    fn a_swatch_is_exactly_one_column_wide() {
        let _serial = crate::global_state_lock();
        let previous = Palette::current();
        Palette::set(Palette::Bundled);
        tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());

        // Both extremes of `hex`: an RGB theme colour, and the widest non-RGB value a rung can
        // resolve to under a weaker encoding.
        for colour in [
            Color::Rgb(0x1e, 0x1e, 0x2e),
            Color::Indexed(244),
            Color::Reset,
        ] {
            let width: usize = swatch(colour)
                .iter()
                .map(|span| span.content.chars().count())
                .sum();
            assert_eq!(
                width, SWATCH,
                "{colour:?} renders {width} columns, not {SWATCH}"
            );
        }

        Palette::set(previous);
    }

    /// The theme's own interior neutrals do NOT sit where r02 puts the rungs of the same name.
    ///
    /// This is the evidence for keeping two columns, and it is the answer to *"why wouldn't we
    /// use the same?"* — a question that reads as obviously right until the positions are
    /// measured. r02's `muted` is *"the default posture of most of the screen"*, readable body
    /// text at 62%. A theme's `muted` is the tint an editor paints comments and borders with,
    /// and on most bundled themes it sits **below our `faint` rung** — adopting it would put
    /// most of a screen's text under the rung reserved for de-emphasised metadata.
    ///
    /// The bands are the rungs each candidate would have to fall between to be substitutable:
    /// `cursor_bg` (19) sits between `layer1_bg` (15) and `layer2_bg` (23); `muted` (62) sits
    /// between `rule_frame` (54) and `cursor_mark` (70).
    ///
    /// **Asserted as a majority rather than as fifteen exact numbers**, because the finding is
    /// "these are not interchangeable" and not "Dracula's selection is 12.1%". If a future
    /// version of `ratatui-themes` moved its neutrals onto r02's rungs this test would fail,
    /// and the decision it defends should genuinely be revisited then.
    #[test]
    fn the_themes_own_neutrals_do_not_sit_where_r02_puts_its_rungs() {
        let (mut selection_fits, mut muted_fits, mut total) = (0, 0, 0);
        for name in ratatui_themes::ThemeName::all() {
            let theme = name.palette();
            let selection = ladder_percent(theme.selection, &theme);
            let muted = ladder_percent(theme.muted, &theme);
            selection_fits += (selection > 15.0 && selection < 23.0) as usize;
            muted_fits += (muted > 54.0 && muted < 70.0) as usize;
            total += 1;
        }

        assert!(
            selection_fits * 2 < total,
            "the theme's `selection` now lands in the layer band on {selection_fits}/{total} \
             themes — it may be substitutable for `cursor_bg` after all"
        );
        assert!(
            muted_fits * 2 < total,
            "the theme's `muted` now lands in the text band on {muted_fits}/{total} themes — \
             it may be substitutable for our `muted` rung after all"
        );
    }

    /// The four anchors are the four neutrals, in ladder order.
    ///
    /// Order is the whole point of the second block — it is what makes it a *ladder* rather
    /// than four more swatches — and it is the one property a reader cannot check from the
    /// frame without already knowing the answer.
    #[test]
    fn the_anchors_are_the_four_neutrals_in_ladder_order() {
        let names: Vec<&str> = ANCHORS.iter().map(|(name, _, _, _, _)| *name).collect();
        assert_eq!(names, ["bg", "selection", "muted", "fg"]);
    }

    /// The two endpoints agree with the ladder; the two interior anchors need not.
    ///
    /// This is the finding the second block exists to make visible, so it is asserted rather
    /// than described: §15's *"interpolated from the theme's own bg and fg"* means `screen_bg`
    /// and `normal` ARE the theme's endpoints, while `cursor_bg` and `muted` are percentages
    /// that land where they land. If a later change made all four agree, the block would be
    /// four identical pairs and would no longer be worth drawing — this test is what would
    /// say so.
    #[test]
    fn the_endpoints_are_the_themes_own_and_the_middles_are_ours() {
        let _serial = crate::global_state_lock();
        let theme = ratatui_themes::ThemeName::CatppuccinMocha.palette();
        let previous = Palette::current();
        Palette::set(Palette::Bundled);
        tokens::set_theme(theme);

        assert_eq!(
            tokens::screen_bg(),
            Some(theme.bg),
            "the low end is the theme's background — §15's full paint"
        );
        assert_eq!(
            tokens::normal(),
            theme.fg,
            "the high end is the theme's foreground — NORMAL_RUNG is defined so it lands there"
        );
        assert_ne!(
            tokens::cursor_bg(),
            theme.selection,
            "if these agreed, the second block would be showing nothing"
        );
        assert_ne!(tokens::muted(), theme.muted, "likewise");

        Palette::set(previous);
    }

    /// A frame with nothing to show says which source is in force.
    ///
    /// The failure this guards is silence: an empty preview pane reads as a broken entry, and
    /// the reader's next move is to go looking at the code rather than at the palette switch
    /// that caused it.
    #[test]
    fn without_a_bundled_theme_the_frame_names_the_source_instead_of_going_blank() {
        let _serial = crate::global_state_lock();
        let previous = Palette::current();
        Palette::set(Palette::Derived);

        let lines = unavailable();
        let text: String = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect();
        assert!(text.contains("Derived"), "it must name the source: {text}");

        Palette::set(previous);
    }
}

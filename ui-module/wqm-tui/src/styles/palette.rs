//! Every colour the design can reach for, on one scale — the theme's ten and our eleven rungs.
//!
//! Chris, 20260801: *"in the Colors section I would like to have a section Palette which would
//! show: the 10 roles and the 4 anchors, having them together will be helpful for visual
//! comparisons and to reduce (hopefully) the number of iterations when we start designing the
//! elements."*
//!
//! # One list, two vocabularies
//!
//! Chris, 20260801, on the first cut's separate anchors block: *"since the four neutrals are also
//! included in the ten roles we don't need a special section for them, what we need is just a
//! distinction between custom roles and standard roles."* So there is no second block. There are
//! two **kinds** — `STD`, a field the theme names, and `OURS`, a rung we derive from it — marked
//! in a column, and the neutrals are ordered by luminance in one run *"starting by the background
//! and finishing by the color we use for Strong"*.
//!
//! The ordering is what makes the two vocabularies legible against each other. `theme.selection`
//! lands at 9% on Mocha and `theme.muted` at 38%, so they interleave with our rungs rather than
//! coinciding with the two that share their names — `tokens::cursor_bg` at 19% and
//! `tokens::muted` at 62%. Sorted into one list that is impossible to miss; drawn as two blocks
//! it was invisible.
//!
//! **Names are qualified for exactly that reason.** `muted` alone names two different colours at
//! two different places on the scale, and the bare word is what let them be confused. Reach for
//! `tokens::muted`; `theme.muted` is not ours to use.
//!
//! # Ours are derived from the theme, never neutral greys
//!
//! Chris: *"I assume that those custom roles are derived from the theme equivalent colors … if
//! not it's a must (again themes are rarely neutral)."* They are: `ladder_endpoints` interpolates
//! every rung between the theme's own `bg` and `fg`, so each carries its tint —
//! `tests::every_custom_rung_carries_the_themes_tint_and_none_is_grey` fails if the ladder ever
//! regresses to black-to-white, which renders perfectly well and looks *almost* right.
//!
//! **They are not anchored on `selection` and `muted`, and that is deliberate**, not an
//! oversight: those two sit at 9% and 38% rather than at r02's 19% and 62%, and a theme's `muted`
//! is the tint an editor paints comments with, not body text. Substituting it would put most of a
//! screen's text below our `faint` rung.
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

use crate::styles::palette_reference::{resolve, RUNGS};
use crate::styles::strong::{nearest_reserved, CANDIDATES, RESERVED_FLOOR};
use crate::tokens::{self, delta_e};
use crate::widgets::config_table::fit;

/// Width of the STANDARD / CUSTOM marker.
const KIND: usize = 8;
/// Width of the name column. Names are **qualified** — `theme.muted` and `tokens::muted` are
/// different colours at different places on the scale, and the leaf word alone is what let them
/// be confused in the first place.
const NAME: usize = 22;
/// Width of the swatch: brackets, fill, sample. See [`swatch`] for why it is bracketed.
/// Counted from what [`swatch`] BUILDS, not from the parts added up by eye — the first value
/// here was 14 against a real 13, and the guard is what said so.
const SWATCH: usize = 13;
/// Width of the resolved value.
const VALUE: usize = 9;
/// Width of the ladder position.
const AT: usize = 6;

/// Which vocabulary a colour belongs to.
///
/// This is the distinction Chris asked for (*"what we need is just a distinction between custom
/// roles and standard roles"*), and it replaces the separate anchors block: the four neutrals
/// were already among the ten, so drawing them twice said less than marking them once.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// A field the **theme** names. Ten of them, and the whole of what a theme gives us.
    Standard,
    /// A rung **we** name, derived from the theme. r02 specifies eleven of these by luminance
    /// percentage and the theme has no field for any of them.
    Custom,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Standard => "STD",
            Kind::Custom => "OURS",
        }
    }

    /// `STD` recedes and `OURS` does not: the custom rungs are the ones a widget reaches for, so
    /// they are the ones that should catch the eye in a list where both appear.
    fn style(self) -> Style {
        match self {
            Kind::Standard => tokens::faint_style(),
            Kind::Custom => tokens::normal_style(),
        }
    }
}

/// The theme's six **hues** — the fields that are not part of the neutral ladder.
///
/// Kept as their own group because ordering hues by luminance says nothing: what matters about
/// `error` is that it is the alarm, not that it is brighter than `accent`. The claims are the
/// other half — a field with no claim is headroom, and §3 forbids two roles landing on one hue.
type Hue = (
    &'static str,
    fn(&ratatui_themes::ThemePalette) -> Color,
    &'static str,
);

const HUES: [Hue; 6] = [
    ("theme.accent", |p| p.accent, "unclaimed — headroom"),
    ("theme.secondary", |p| p.secondary, "unclaimed — headroom"),
    ("theme.error", |p| p.error, "offline ○"),
    (
        "theme.warning",
        |p| p.warning,
        "degraded ▲, and stale timings",
    ),
    ("theme.success", |p| p.success, "healthy ●"),
    (
        "theme.info",
        |p| p.info,
        "selector — reserved absolutely (§3), chosen over accent",
    ),
];

/// The theme's four **neutrals**, with what each is for and whether we use it.
///
/// Two of them anchor our ladder and two do not, and saying which is the point: `theme.selection`
/// and `theme.muted` are a theme's answers to questions r02 also answers, at different places on
/// the scale. They are listed so the divergence is visible, not so it is adopted.
type Neutral = (
    &'static str,
    fn(&ratatui_themes::ThemePalette) -> Color,
    &'static str,
);

const NEUTRALS: [Neutral; 4] = [
    (
        "theme.bg",
        |p| p.bg,
        "the screen's full paint (§15) — and the ladder's low end",
    ),
    (
        "theme.selection",
        |p| p.selection,
        "its cursor tint — NOT ours; cf tokens::cursor_bg",
    ),
    (
        "theme.muted",
        |p| p.muted,
        "its comment/border tint — NOT ours; cf tokens::muted",
    ),
    (
        "theme.fg",
        |p| p.fg,
        "the ladder's high end — the SAME colour as tokens::normal below",
    ),
];

/// One row of the ladder, ready to be ordered.
struct Rung {
    kind: Kind,
    name: String,
    colour: Color,
    at: f32,
    note: String,
    /// Set only on the `strong` candidates. Two numbers a frame genuinely cannot show: how far
    /// this colour is from the body text it must beat, and how close it comes to a hue §3
    /// reserves. The eye settles the first badly and cannot settle the second at all.
    verdict: Option<Verdict>,
}

/// The two measurements that decide a `strong` candidate.
struct Verdict {
    body: f32,
    reserved: f32,
}

/// The whole palette on one scale: the hues, then every neutral in luminance order.
///
/// # It can be pinned to a theme, and that is how `strong` gets judged
///
/// Chris, 20260801: *"this must be considered holistically theme by theme … duplicate your
/// existing catppuccin theme, add your variants of strong, and for each duplicate have it show
/// the palette in its own theme."* A palette is judged whole — `strong` is not a colour on its
/// own, it is a colour among the ten the theme names and the ten rungs beneath it — so the
/// pantry carries **one entry per bundled theme**, each rendering *in* its own theme with every
/// candidate for `strong` drawn beside the rungs it has to live with.
///
/// Pinning follows `palette_reference::NeutralRungs`: force the globals for the duration of the
/// render, restore them after. That is the only way one frame can show a theme the harness is
/// not set to, and it is why fifteen entries cost fifteen lines rather than fifteen harnesses.
pub struct PaletteFrame {
    pinned: Option<ratatui_themes::ThemeName>,
}

impl PaletteFrame {
    /// The theme currently in force — what the Styles tab shows when nothing is pinned.
    pub fn current() -> Self {
        Self { pinned: None }
    }

    /// One named theme, whatever the harness is set to.
    pub fn pinned(theme: ratatui_themes::ThemeName) -> Self {
        Self {
            pinned: Some(theme),
        }
    }
}

impl Widget for PaletteFrame {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let restore = self.pinned.map(|name| {
            let previous = (tokens::Palette::current(), tokens::theme());
            tokens::Palette::set(tokens::Palette::Bundled);
            tokens::set_theme(name.palette());
            previous
        });

        self.paint(area, buf);

        if let Some((palette, theme)) = restore {
            tokens::Palette::set(palette);
            if let Some(theme) = theme {
                tokens::set_theme(theme);
            }
        }
    }
}

impl PaletteFrame {
    fn paint(&self, area: Rect, buf: &mut Buffer) {
        let Some(theme) = tokens::active_theme() else {
            Paragraph::new(unavailable()).render(area, buf);
            return;
        };

        let mut lines = vec![
            Line::from(Span::styled(
                "Every colour the design can reach for — the theme's, and the ones we derive from it.",
                tokens::strong_style(),
            )),
            Line::from(Span::styled(
                "Live from tokens::active_theme(): this frame paints what a screen paints.",
                tokens::faint_style(),
            )),
            Line::default(),
            heading("HUES — the theme's, and each one as a SURFACE"),
            hue_header(),
        ];

        for (name, field, claim) in HUES {
            lines.push(hue_row(name, field(&theme), theme.bg, claim));
        }

        lines.push(Line::default());
        lines.push(heading(
            "THE NEUTRAL LADDER — by luminance, background first, `strong` last",
        ));
        lines.push(header());
        for rung in ladder(&theme) {
            lines.push(row(&rung));
        }

        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            "ON BG @14% is the hue mixed into the background at WASH_MIX — a tinted SURFACE \
             rather than another grey.",
            tokens::faint_style(),
        )));
        lines.push(Line::from(Span::styled(
            "AT is the position on r02's scale, where the theme's background is 0 and its \
             foreground is 85. Every",
            tokens::faint_style(),
        )));
        lines.push(Line::from(Span::styled(
            "OURS rung is interpolated between those two, so it carries the theme's tint and is \
             never a neutral grey.",
            tokens::faint_style(),
        )));

        Paragraph::new(lines).render(area, buf);
    }
}

/// Every neutral, the theme's and ours, on one scale in luminance order.
///
/// The interleaving is the finding. `theme.selection` lands at 9% on Mocha and `theme.muted` at
/// 38%, so they do not sit anywhere near the rungs that share their names — and a single ordered
/// list is the only presentation that makes that impossible to miss. Ties break **STD first**,
/// so `theme.fg` reads as the source and `tokens::normal` as the name we give it.
fn ladder(theme: &ratatui_themes::ThemePalette) -> Vec<Rung> {
    let mut rungs: Vec<Rung> = NEUTRALS
        .iter()
        .map(|(name, field, note)| {
            let colour = field(theme);
            Rung {
                kind: Kind::Standard,
                name: name.to_string(),
                colour,
                at: ladder_percent(colour, theme),
                note: note.to_string(),
                verdict: None,
            }
        })
        .chain(
            RUNGS
                .iter()
                // `strong` is drawn once per candidate rule at the end, not once here.
                .filter(|(_, name, _)| *name != "strong")
                .map(|(percent, name, spec)| Rung {
                    kind: Kind::Custom,
                    // Qualified, because `muted` alone names two different colours.
                    name: format!("tokens::{name}"),
                    colour: resolve(*percent),
                    at: *percent as f32,
                    note: spec.to_string(),
                    verdict: None,
                }),
        )
        .collect();

    rungs.sort_by(|a, b| {
        a.at.partial_cmp(&b.at)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| (a.kind == Kind::Custom).cmp(&(b.kind == Kind::Custom)))
    });

    // Appended rather than sorted in: a tinted candidate can be *darker* than the foreground,
    // which would scatter the four across the ladder and hide that they are one decision. They
    // are the top of the ladder whatever their luminance, so they go at the top.
    rungs.extend(CANDIDATES.iter().map(|candidate| {
        let colour = candidate.resolve(theme);
        Rung {
            kind: Kind::Custom,
            name: format!("strong: {}", candidate.label()),
            colour,
            at: ladder_percent(colour, theme),
            note: String::new(),
            verdict: Some(Verdict {
                body: delta_e(colour, theme.fg),
                reserved: nearest_reserved(colour, theme),
            }),
        }
    }));
    rungs
}

/// The hues' header — they have no ladder position, and they have a second swatch instead.
fn hue_header() -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {}", fit("", KIND)), tokens::faint_style()),
        Span::styled(fit("NAME", NAME), tokens::muted_style()),
        Span::styled(fit("", SWATCH), tokens::muted_style()),
        Span::styled(fit("VALUE", VALUE), tokens::muted_style()),
        Span::styled(fit("ON BG @14%", SWATCH), tokens::muted_style()),
        Span::styled("WHAT IT IS FOR", tokens::muted_style()),
    ])
}

/// One hue, and the same hue **mixed into the background** at the wash strength.
///
/// Chris, 20260801: *"I would keep the greys (just in case) and add bg hues in the hues
/// category."* The second swatch is what a **surface** in that hue looks like — a modal fill, a
/// zone tint — rather than what the hue looks like as a glyph. `WASH_MIX` is the strength the
/// condition wash already uses (*"we need one for the 14% red anyway"*), so this shows the
/// existing instance and the five hypothetical ones side by side, at one strength, on the
/// theme's own background.
///
/// This does **not** adopt anything. It is the frame the "do we need so many greys" question is
/// to be judged from: a tinted surface beside the grey ladder above it.
fn hue_row(name: &str, hue: Color, bg: Color, claim: &'static str) -> Line<'static> {
    let mut spans = vec![
        Span::styled(
            format!("  {}", fit(Kind::Standard.label(), KIND)),
            Kind::Standard.style(),
        ),
        Span::styled(fit(name, NAME), Kind::Standard.style()),
    ];
    spans.extend(swatch(hue));
    spans.push(Span::styled(fit(&hex(hue), VALUE), tokens::muted_style()));
    // The surface's own value is derivable and not the point — he needs to SEE it, not read it.
    spans.extend(swatch(surface(bg, hue)));
    spans.push(Span::styled(claim.to_string(), tokens::faint_style()));
    Line::from(spans)
}

/// A background pulled toward a hue by [`tokens::WASH_MIX`] — the morph, not another grey.
fn surface(bg: Color, hue: Color) -> Color {
    let channels = |colour: Color| match colour {
        Color::Rgb(r, g, b) => [r as f32, g as f32, b as f32],
        _ => [0.0; 3],
    };
    let (base, toward) = (channels(bg), channels(hue));
    let blend = |i: usize| {
        (base[i] + (toward[i] - base[i]) * tokens::WASH_MIX).round().clamp(0.0, 255.0) as u8
    };
    Color::Rgb(blend(0), blend(1), blend(2))
}

/// The column header, repeated over each group so a long frame stays readable when scrolled.
fn header() -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {}", fit("", KIND)), tokens::faint_style()),
        Span::styled(fit("NAME", NAME), tokens::muted_style()),
        Span::styled(fit("", SWATCH), tokens::muted_style()),
        Span::styled(fit("VALUE", VALUE), tokens::muted_style()),
        Span::styled(fit("AT", AT), tokens::muted_style()),
        Span::styled("WHAT IT IS FOR", tokens::muted_style()),
    ])
}

/// One row: kind, qualified name, swatch, value, ladder position, purpose.
fn row(rung: &Rung) -> Line<'static> {
    let mut spans = vec![
        Span::styled(format!("  {}", fit(rung.kind.label(), KIND)), rung.kind.style()),
        Span::styled(fit(&rung.name, NAME), rung.kind.style()),
    ];
    spans.extend(swatch(rung.colour));
    spans.push(Span::styled(
        fit(&hex(rung.colour), VALUE),
        tokens::muted_style(),
    ));
    spans.push(Span::styled(
        fit(
            &if rung.at.is_nan() {
                "—".to_string()
            } else {
                format!("{:.0}%", rung.at)
            },
            AT,
        ),
        tokens::faint_style(),
    ));
    match &rung.verdict {
        None => spans.push(Span::styled(rung.note.clone(), tokens::faint_style())),
        Some(verdict) => {
            // Coloured against their own floors, so the eye lands on the failures rather than
            // reading eight numbers. A candidate below either floor is not a weaker option, it
            // is a rule that does not work on this theme.
            spans.push(Span::styled(
                format!("body {:>5.1}  ", verdict.body),
                if verdict.body < 2.3 {
                    Style::default().fg(tokens::offline())
                } else {
                    tokens::faint_style()
                },
            ));
            spans.push(Span::styled(
                format!("reserved {:>5.1}", verdict.reserved),
                if verdict.reserved < RESERVED_FLOOR {
                    Style::default().fg(tokens::offline())
                } else {
                    tokens::faint_style()
                },
            ));
        }
    }
    Line::from(spans)
}

/// A block heading — the frame has two groups and they must not run together.
fn heading(text: &'static str) -> Line<'static> {
    Line::from(Span::styled(
        text,
        Style::default()
            .fg(tokens::header())
            .add_modifier(Modifier::BOLD),
    ))
}

/// One colour as a **bracketed** fill plus a text sample.
///
/// The brackets are not decoration. Chris: *"the first one of OURS is hardly visible against the
/// bg so this one is useless"* — and he was right about more than that one row: `theme.bg` **is**
/// the screen's background, so its swatch paints background on background and shows nothing at
/// all. Every dark rung near it has the same problem to a lesser degree. Delimiting the fill with
/// `rule_internal` gives the swatch an edge that does not depend on its own contrast, so the
/// bottom of the ladder is a visible extent rather than a gap.
fn swatch(colour: Color) -> Vec<Span<'static>> {
    let edge = Style::default().fg(tokens::rule_internal());
    vec![
        Span::styled("▏", edge),
        Span::styled("      ", Style::default().bg(colour)),
        Span::styled("▕", edge),
        Span::styled(" Aa ", Style::default().fg(colour)),
        Span::raw(" "),
    ]
}

/// `#rrggbb`, or the indexed/reset value where a weak encoding replaced our ladder.
fn hex(colour: Color) -> String {
    match colour {
        Color::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
        Color::Indexed(i) => format!("idx {i}"),
        _ => format!("{colour:?}"),
    }
}

/// Relative luminance, the measure r02 names its rungs in.
fn luma(colour: Color) -> f32 {
    match colour {
        Color::Rgb(r, g, b) => 0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32,
        _ => f32::NAN,
    }
}

/// Where a colour falls on **our** ladder, in r02's percentage units.
///
/// `NORMAL_RUNG` is 85 by construction — the percentage at which the ladder reaches the
/// foreground — so `bg` is 0 and `fg` is 85, and anything between reads directly against the rung
/// numbers. Defined here rather than in `tokens` because it is the *inverse* of the ladder and
/// only a comparison needs it; a widget reaching for a rung asks for the rung.
fn ladder_percent(colour: Color, theme: &ratatui_themes::ThemePalette) -> f32 {
    let (bg, fg) = (luma(theme.bg), luma(theme.fg));
    85.0 * (luma(colour) - bg) / (fg - bg)
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

    /// One entry per bundled theme, each rendering **in** that theme.
    ///
    /// Chris, 20260801: *"duplicate your existing catppuccin theme, add your variants of
    /// strong, and for each duplicate have it show the palette in its own theme."* So the
    /// variant list is the theme list, and every frame is a whole palette rather than a row of
    /// one — which is what makes `strong` judgeable at all: it has to be read against the ten
    /// the theme names and the ten rungs beneath it, not against the same colour on fourteen
    /// other palettes.
    struct Palette(Option<ratatui_themes::ThemeName>);

    impl Ingredient for Palette {
        // Styles, and NO section: this is vocabulary rather than an instrument, so it belongs
        // with the vocabulary. `Instruments` is left to `Palette Reference` and `Theme Sources`
        // (§16).
        fn tab(&self) -> &str {
            "Styles"
        }
        // The group is `Colors` because Chris asked for it "in the Colors section", and since
        // 20260801 it is the ONLY group there — the four TOML groups that used to share the
        // name were transcriptions of what this frame renders live.
        fn group(&self) -> &str {
            "Colors"
        }
        fn name(&self) -> &str {
            match self.0 {
                // The harness's own theme, whatever `widget_preview` set it to.
                None => "Palette — as set",
                Some(theme) => theme.display_name(),
            }
        }
        fn source(&self) -> &str {
            "wqm_tui::styles::palette"
        }
        fn description(&self) -> &str {
            "The whole palette in one theme: ten named fields, ten rungs, and every candidate for `strong`"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            match self.0 {
                None => PaletteFrame::current().render(area, buf),
                Some(theme) => PaletteFrame::pinned(theme).render(area, buf),
            }
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        std::iter::once(Box::new(Palette(None)) as Box<dyn Ingredient>)
            .chain(
                ratatui_themes::ThemeName::all()
                    .iter()
                    .map(|name| Box::new(Palette(Some(*name))) as Box<dyn Ingredient>),
            )
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::Palette;

    fn mocha() -> ratatui_themes::ThemePalette {
        ratatui_themes::ThemeName::CatppuccinMocha.palette()
    }

    /// Every field of the theme reaches the frame, exactly once, across the two groups.
    ///
    /// The lists are hand-ordered — six hues, four neutrals — rather than taken from the struct,
    /// so a field added upstream, or one dropped in a reshuffle, would leave the frame quietly
    /// incomplete. Ten is the number `ThemePalette` has; it is worth asserting rather than
    /// counting by eye.
    #[test]
    fn all_ten_theme_fields_are_present_and_none_is_shown_twice() {
        let theme = mocha();
        let mut names: Vec<&str> = HUES
            .iter()
            .map(|(n, _, _)| *n)
            .chain(NEUTRALS.iter().map(|(n, _, _)| *n))
            .collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 10, "ten fields, each once: {names:?}");

        let mut values: Vec<String> = HUES
            .iter()
            .map(|(_, f, _)| hex(f(&theme)))
            .chain(NEUTRALS.iter().map(|(_, f, _)| hex(f(&theme))))
            .collect();
        values.sort();
        values.dedup();
        assert_eq!(
            values.len(),
            10,
            "two fields resolve to one colour, so one is wired to the wrong accessor"
        );
    }

    /// A hue's surface is a nudge off the background, not a wash of colour.
    ///
    /// The whole proposition is that a tinted surface can replace a grey one *without shouting*
    /// — Chris's *"some subtle tint for the background of our modal windows"*. So the guard is
    /// two-sided: the surface must be visibly off the background (or it is just the background
    /// and buys nothing) and must stay far nearer the background than the hue (or it is a block
    /// of colour and the modal is shouting). Checked on every theme, because a hue that is
    /// already close to the background — Everforest's `muted`-adjacent greens — is where the
    /// first half fails.
    #[test]
    fn a_tinted_surface_reads_as_the_background_and_not_as_the_hue() {
        for name in ratatui_themes::ThemeName::all() {
            let theme = name.palette();
            for (label, hue) in [
                ("accent", theme.accent),
                ("error", theme.error),
                ("info", theme.info),
            ] {
                let tinted = surface(theme.bg, hue);
                let to_bg = distance(tinted, theme.bg);
                let to_hue = distance(tinted, hue);
                assert!(
                    to_bg > 1.0,
                    "{name:?}/{label}: the surface is {to_bg:.1} from the background — invisible"
                );
                assert!(
                    to_bg * 3.0 < to_hue,
                    "{name:?}/{label}: the surface is {to_bg:.1} from bg and {to_hue:.1} from \
                     the hue — that is a coloured panel, not a tint"
                );
            }
        }
    }

    /// Plain RGB distance — enough to say "nudge" from "shout"; the ΔE machinery lives in
    /// `tokens::tests` and is not worth duplicating for a ratio test.
    fn distance(a: Color, b: Color) -> f32 {
        let ch = |c: Color| match c {
            Color::Rgb(r, g, b) => [r as f32, g as f32, b as f32],
            _ => [0.0; 3],
        };
        let (a, b) = (ch(a), ch(b));
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// The ladder is ordered by luminance from the background up, and ends with the candidates.
    ///
    /// Chris asked for the greys *"in order of luminance starting by the background and
    /// finishing by the color we use for Strong"*, and that is the property that makes the frame
    /// worth reading: the interleaving of STD and OURS is only legible if the list is genuinely
    /// sorted. A frame that merely *looked* sorted — the theme's four, then ours — would show
    /// the same rows and say nothing.
    ///
    /// The four `strong` candidates are **appended rather than sorted in**: a tinted candidate
    /// can be darker than the foreground, and sorting would scatter them through the ladder and
    /// hide that they are one decision taken four ways.
    #[test]
    fn the_ladder_runs_from_the_background_upward_and_ends_with_the_candidates() {
        let _serial = crate::global_state_lock();
        let previous = Palette::current();
        Palette::set(Palette::Bundled);
        tokens::set_theme(mocha());

        let rungs = ladder(&mocha());
        let (sorted, candidates) = rungs.split_at(rungs.len() - CANDIDATES.len());

        assert_eq!(sorted.first().map(|r| r.name.as_str()), Some("theme.bg"));
        for pair in sorted.windows(2) {
            assert!(
                pair[0].at <= pair[1].at,
                "{} at {:.1}% precedes {} at {:.1}%",
                pair[0].name,
                pair[0].at,
                pair[1].name,
                pair[1].at
            );
        }

        assert!(
            sorted.iter().all(|r| !r.name.starts_with("strong")),
            "a `strong` candidate leaked into the sorted run"
        );
        for (rung, candidate) in candidates.iter().zip(CANDIDATES) {
            assert_eq!(rung.name, format!("strong: {}", candidate.label()));
            assert!(
                rung.verdict.is_some(),
                "{} is drawn without the two numbers that decide it",
                rung.name
            );
        }

        Palette::set(previous);
    }

    /// The theme's two unused neutrals land BETWEEN our rungs rather than on them.
    ///
    /// This is the whole reason both vocabularies appear in one list. If `theme.selection` sat at
    /// 19% and `theme.muted` at 62% they would be our rungs, the STD/OURS marking would be
    /// pedantry, and the sensible thing would be to adopt them. They do not:
    /// `the_themes_own_neutrals_do_not_sit_where_r02_puts_its_rungs` measures that across all
    /// fifteen themes, and this one pins what it means for the *frame* — the interleaving is
    /// real, so the ordering is telling the reader something.
    #[test]
    fn the_themes_unused_neutrals_interleave_with_our_rungs() {
        let _serial = crate::global_state_lock();
        let previous = Palette::current();
        Palette::set(Palette::Bundled);
        tokens::set_theme(mocha());

        let rungs = ladder(&mocha());
        let position = |name: &str| {
            rungs
                .iter()
                .position(|r| r.name == name)
                .unwrap_or_else(|| panic!("{name} is missing from the ladder"))
        };

        // Neither is at an end, so each has one of our rungs on both sides of it.
        for name in ["theme.selection", "theme.muted"] {
            let at = position(name);
            assert!(
                at > 0 && at < rungs.len() - 1,
                "{name} is at the edge of the ladder, so it interleaves with nothing"
            );
            assert!(
                rungs[at - 1].kind == Kind::Custom || rungs[at + 1].kind == Kind::Custom,
                "{name} has no OURS rung adjacent to it"
            );
        }

        Palette::set(previous);
    }

    /// The theme's own interior neutrals do NOT sit where r02 puts the rungs of the same name.
    ///
    /// This is the evidence for keeping both vocabularies, and it is the answer to *"why wouldn't
    /// we use the same?"* — a question that reads as obviously right until the positions are
    /// measured. r02's `muted` is *"the default posture of most of the screen"*, readable body
    /// text at 62%. A theme's `muted` is the tint an editor paints comments and borders with, and
    /// on most bundled themes it sits **below our `faint` rung** — adopting it would put most of
    /// a screen's text under the rung reserved for de-emphasised metadata.
    ///
    /// The bands are the rungs each candidate would have to fall between to be substitutable:
    /// `cursor_bg` (19) sits between `layer1_bg` (15) and `layer2_bg` (23); `muted` (62) sits
    /// between `rule_frame` (54) and `cursor_mark` (70).
    ///
    /// **Asserted as a majority rather than as fifteen exact numbers**, because the finding is
    /// "these are not interchangeable" and not "Dracula's selection is 12.1%". If a future
    /// version of `ratatui-themes` moved its neutrals onto r02's rungs this test would fail, and
    /// the decision it defends should genuinely be revisited then.
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

    /// Every OURS rung is derived from the theme, so none of them is a neutral grey.
    ///
    /// Chris: *"I assume that those custom roles are derived from the theme equivalent colors …
    /// if not it's a must (again themes are rarely neutral)."* They are — `ladder_endpoints`
    /// interpolates between the theme's own `bg` and `fg`, so every rung carries its tint. The
    /// failure this guards is a regression to a black-to-white ladder, which renders perfectly
    /// well and looks *almost* right against a tinted theme.
    #[test]
    fn every_custom_rung_carries_the_themes_tint_and_none_is_grey() {
        let _serial = crate::global_state_lock();
        let previous = Palette::current();
        Palette::set(Palette::Bundled);
        tokens::set_theme(mocha());

        // Mocha's bg and fg are both blue-tinted, so every interpolation between them must be.
        // A grey has r == g == b; the endpoints do not, so nothing between them may either.
        for rung in ladder(&mocha()).iter().filter(|r| r.kind == Kind::Custom) {
            let Color::Rgb(r, g, b) = rung.colour else {
                panic!("{} is not RGB under Bundled: {:?}", rung.name, rung.colour)
            };
            assert!(
                !(r == g && g == b),
                "{} resolved to the grey {:?} — the ladder is no longer derived from the theme",
                rung.name,
                rung.colour
            );
        }

        Palette::set(previous);
    }

    /// A swatch is exactly [`SWATCH`] columns, so the columns after it line up.
    ///
    /// Counted in **characters, not bytes** — `▏` and `▕` are three bytes and one column each,
    /// and this crate has already reported a three-column jitter that did not exist by measuring
    /// `▌` as three. The failure this guards is a header that labels the wrong column, and a
    /// value read off the wrong column is exactly the kind of wrong number this frame exists to
    /// prevent.
    #[test]
    fn a_swatch_is_exactly_one_column_wide() {
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
    }

    /// The swatch has an edge that does not depend on its own contrast.
    ///
    /// `theme.bg` IS the screen's background, so an unbracketed swatch of it paints background on
    /// background — Chris saw exactly that (*"hardly visible against the bg, so this one is
    /// useless"*). The brackets are what make the bottom of the ladder a visible extent, and they
    /// have to be drawn in something other than the swatch colour or they solve nothing.
    #[test]
    fn the_darkest_swatch_still_has_an_edge() {
        let _serial = crate::global_state_lock();
        let previous = Palette::current();
        Palette::set(Palette::Bundled);
        tokens::set_theme(mocha());

        let spans = swatch(mocha().bg);
        let edges: Vec<&Span> = spans
            .iter()
            .filter(|s| s.content == "▏" || s.content == "▕")
            .collect();
        assert_eq!(edges.len(), 2, "a swatch is delimited on both sides");
        for edge in edges {
            assert_ne!(
                edge.style.fg,
                Some(mocha().bg),
                "the edge is drawn in the swatch's own colour, so it is invisible too"
            );
        }

        Palette::set(previous);
    }

    /// A frame with nothing to show says which source is in force.
    ///
    /// The failure this guards is silence: an empty preview pane reads as a broken entry, and the
    /// reader's next move is to go looking at the code rather than at the palette switch that
    /// caused it.
    #[test]
    fn without_a_bundled_theme_the_frame_names_the_source_instead_of_going_blank() {
        let _serial = crate::global_state_lock();
        let previous = Palette::current();
        Palette::set(Palette::Derived);

        let text: String = unavailable()
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect();
        assert!(text.contains("Derived"), "it must name the source: {text}");

        Palette::set(previous);
    }
}

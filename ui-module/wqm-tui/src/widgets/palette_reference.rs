//! The palette itself, enumerated — what colours exist, and what each one resolves to.
//!
//! The other sheets show the vocabulary *in use*. This one is the lookup table behind it:
//! every ANSI slot the design draws a hue from, and every neutral rung, printed next to the
//! value it actually resolved to. That last part is the point. A rung whose swatch looks
//! wrong is ambiguous — the fault could be in the rung, in the palette mode, or in the
//! terminal's theme. Printing `Rgb(133, 138, 162)` or `Indexed(244)` beside the swatch
//! collapses that ambiguity to one question: is *this value* the intended one?
//!
//! # Why the hues are a fixed list of sixteen
//!
//! A terminal theme redefines slots 0–15 and nothing else, so those sixteen are the entire
//! palette a theme can speak through. The design draws every hue from them deliberately
//! ([`crate::tokens`]), which is why this table names which role owns which slot: an unowned
//! slot is a colour the design has left on the table, and two roles on one slot is the
//! collision §3 forbids.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::encoding::Encoding;
use crate::tokens::{self, Palette};

/// The sixteen slots a terminal theme can redefine, with the role that claims each.
///
/// Kept in slot order rather than grouped by role, because the question this answers is
/// "what is slot 4 doing" as often as "where does the selector come from". `None` marks a
/// slot the design does not use — deliberate headroom, not an oversight.
const SLOTS: [(u8, &str, Option<&str>); 16] = [
    (
        0,
        "Black",
        Some("SELECTOR_FG — text inside the selector block"),
    ),
    (1, "Red", Some("OFFLINE — the ○ glyph")),
    (2, "Green", Some("HEALTHY — the ● glyph")),
    (
        3,
        "Yellow",
        Some("DEGRADED — the ▲ glyph, and stale timings"),
    ),
    (4, "Blue", None),
    (5, "Magenta", None),
    (
        6,
        "Cyan",
        Some("SELECTOR — reserved, appears in nothing else"),
    ),
    (7, "Gray", Some("Palette::Theme muted / header")),
    (8, "DarkGray", Some("Palette::Theme faint / internal rule")),
    (9, "LightRed", None),
    (10, "LightGreen", None),
    (11, "LightYellow", None),
    (12, "LightBlue", None),
    (13, "LightMagenta", None),
    (14, "LightCyan", None),
    (
        15,
        "White",
        Some("strong — but see tokens::strong(), it can land BELOW normal"),
    ),
];

/// Every rung [`tokens::neutral`] is asked for, with the role that asks and the section of
/// VISUAL-LANGUAGE.md that specifies it.
const RUNGS: [(u8, &str, &str); 11] = [
    (15, "layer1_bg", "§6 modal over a full screen"),
    (19, "cursor_bg", "§3 data-cursor row fill"),
    (23, "layer2_bg", "§6 modal over a modal — must read lighter"),
    (30, "rule_internal", "§2 divides within a screen"),
    (35, "edit_bg", "§3 editing cell — lighter than the cursor"),
    (50, "faint", "§2 legible but de-emphasised"),
    (54, "rule_frame", "§2 top and bottom, underlining the frame"),
    (62, "muted", "§2 the default posture of most of the screen"),
    (70, "cursor_mark", "§3 the ▸ on the cursor row"),
    (85, "normal", "§2 baseline — the terminal's own foreground"),
    (100, "strong", "§2 the one datum that must be seen"),
];

/// The sixteen theme slots, each as a swatch beside the role that owns it.
pub struct AnsiSlots;

impl Widget for AnsiSlots {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let lines: Vec<Line> = SLOTS
            .iter()
            .map(|(index, name, role)| {
                let colour = Color::Indexed(*index);
                Line::from(vec![
                    Span::styled(format!("{index:>3} "), tokens::faint_style()),
                    // Both directions matter: a slot used as a fill has to work behind the
                    // selector's foreground, and as a foreground it has to survive on the
                    // terminal's own background.
                    Span::styled("      ", Style::default().bg(colour)),
                    Span::styled(" Aa ", Style::default().fg(colour)),
                    Span::styled(format!("{name:<14}"), tokens::muted_style()),
                    match role {
                        Some(text) => Span::styled(*text, tokens::normal_style()),
                        None => Span::styled("(unused — headroom)", tokens::faint_style()),
                    },
                ])
            })
            .collect();
        Paragraph::new(lines).render(area, buf);
    }
}

/// Every neutral rung under one palette, with the value it resolved to.
pub struct NeutralRungs {
    palette: Palette,
    encoding: Encoding,
}

impl NeutralRungs {
    /// The ladder as the current encoding renders it — what the reader is actually looking
    /// at in this terminal.
    pub fn new(palette: Palette) -> Self {
        Self::under(palette, Encoding::current())
    }

    /// The ladder under a *forced* encoding, which is how a degradation is judged from a
    /// frame instead of from an argument. [`tokens::family`] takes the lesser of the two, so
    /// a weak encoding is visible here as the source's ladder being replaced wholesale
    /// rather than approximated.
    pub fn under(palette: Palette, encoding: Encoding) -> Self {
        Self { palette, encoding }
    }
}

impl Widget for NeutralRungs {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let previous = (Palette::current(), Encoding::current());
        Palette::set(self.palette);
        Encoding::set(self.encoding);

        let mut lines = vec![Line::from(vec![Span::styled(
            format!(
                "PALETTE: {}   ENCODING: {}   emitted as: {:?}",
                self.palette.label(),
                self.encoding.label(),
                tokens::family()
            ),
            Style::default()
                .fg(tokens::rule_frame())
                .add_modifier(Modifier::BOLD),
        )])];

        // Collected so a rung that resolves to the same value as the one above it can be
        // called out: that is exactly the collapse Palette::Theme suffers, and naming it
        // here means it does not have to be spotted by eye.
        let mut previous_value: Option<Color> = None;
        for (percent, role, spec) in RUNGS {
            let colour = resolve(percent);
            // Under an encoding that emits no colour every rung is `Reset`, so flagging each
            // as a collision would report the design's own fallback as eleven defects. The
            // header already says the family is `None`; that is the finding.
            let collides =
                previous_value == Some(colour) && tokens::family() != crate::encoding::Family::None;
            previous_value = Some(colour);

            lines.push(Line::from(vec![
                Span::styled(format!("{percent:>4}% "), tokens::faint_style()),
                Span::styled("      ", Style::default().bg(colour)),
                Span::styled(" Aa ", Style::default().fg(colour)),
                Span::styled(format!("{role:<14}"), tokens::muted_style()),
                Span::styled(format!("{:<20}", format!("{colour:?}")), {
                    if collides {
                        Style::default().fg(tokens::degraded())
                    } else {
                        tokens::faint_style()
                    }
                }),
                Span::styled(
                    if collides {
                        "COLLIDES with the rung above"
                    } else {
                        spec
                    },
                    if collides {
                        Style::default().fg(tokens::degraded())
                    } else {
                        tokens::faint_style()
                    },
                ),
            ]));
        }

        Paragraph::new(lines).render(area, buf);
        let (palette, encoding) = previous;
        Palette::set(palette);
        Encoding::set(encoding);
    }
}

/// The rung's colour, taking the two ends of the ladder from the tokens that own them
/// rather than re-deriving them here.
fn resolve(percent: u8) -> Color {
    match percent {
        85 => tokens::normal(),
        100 => tokens::strong(),
        _ => tokens::neutral_at(percent),
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const NO_PROPS: &[PropInfo] = &[];

    struct Slots;

    impl Ingredient for Slots {
        fn tab(&self) -> &str {
            "Styles"
        }
        fn section(&self) -> Option<&str> {
            Some("Instruments")
        }
        fn group(&self) -> &str {
            "Palette Reference"
        }
        fn name(&self) -> &str {
            "ANSI 16"
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::palette_reference"
        }
        fn description(&self) -> &str {
            "The sixteen slots a theme can redefine, each as fill and as foreground, with the role that owns it"
        }
        fn props(&self) -> &[PropInfo] {
            NO_PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            AnsiSlots.render(area, buf);
        }
    }

    struct Rungs(Palette, &'static str);

    impl Ingredient for Rungs {
        fn tab(&self) -> &str {
            "Styles"
        }
        fn section(&self) -> Option<&str> {
            Some("Instruments")
        }
        fn group(&self) -> &str {
            "Palette Reference"
        }
        fn name(&self) -> &str {
            self.1
        }
        fn source(&self) -> &str {
            "wqm_tui::tokens::neutral"
        }
        fn description(&self) -> &str {
            "Every neutral rung with the value it resolved to; a rung matching the one above is flagged"
        }
        fn props(&self) -> &[PropInfo] {
            NO_PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            NeutralRungs::new(self.0).render(area, buf);
        }
    }

    /// The authored ladder under a forced encoding — the degradation story, rendered.
    ///
    /// All three use [`Palette::Derived`] as the source, because that is the mode the
    /// storyboard is authored in: what these entries show is what a *user on a weaker
    /// terminal* sees of the frames Chris is judging, which is the question the encoding axis
    /// exists to answer.
    struct Degraded(Encoding, &'static str);

    impl Ingredient for Degraded {
        fn tab(&self) -> &str {
            "Styles"
        }
        fn section(&self) -> Option<&str> {
            Some("Instruments")
        }
        fn group(&self) -> &str {
            "Palette Reference"
        }
        fn name(&self) -> &str {
            self.1
        }
        fn source(&self) -> &str {
            "wqm_tui::encoding"
        }
        fn description(&self) -> &str {
            "The authored ladder as a weaker terminal receives it — the encoding is forced, not probed"
        }
        fn props(&self) -> &[PropInfo] {
            NO_PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            NeutralRungs::under(Palette::Derived, self.0).render(area, buf);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Slots),
            Box::new(Rungs(Palette::Theme, "Rungs: Theme")),
            Box::new(Rungs(Palette::Indexed, "Rungs: Indexed")),
            Box::new(Rungs(Palette::Derived, "Rungs: Derived")),
            Box::new(Rungs(Palette::Bundled, "Rungs: Bundled")),
            Box::new(Degraded(Encoding::Ansi256, "Encoding: ANSI 256")),
            Box::new(Degraded(Encoding::Ansi16, "Encoding: ANSI 16")),
            Box::new(Degraded(Encoding::NoColor, "Encoding: No Color")),
        ]
    }
}

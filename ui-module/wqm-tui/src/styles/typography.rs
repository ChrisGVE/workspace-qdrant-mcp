//! The emphasis ladder at three weights — the same four rungs, bold then normal then italic.
//!
//! Chris, 20260801, after judging the Styles tab: *"the text hierarchy is much better, though
//! you are presenting them bolded only. It would be good to have in the same style: Strong
//! Normal / Strong Italic, i.e. keeping the bold first then a section in the same order with
//! normal weight / and one section with italics."*
//!
//! # Why this could not stay in `pantry.toml`
//!
//! A `[typography]` entry takes `color` and `description` and **nothing else**. Weight is the
//! other half of the emphasis axis and the format has no field for it, which is the standing
//! finding this request ran straight into: the tab could show four colours and say "strong is
//! bold" in prose, and that was the whole of it.
//!
//! Rendering the ladder here also removes the transcription. The TOML copy was four hex values
//! measured from `tokens` under one theme and pinned to it — correct on the day it was written
//! and silently wrong the moment a different theme was chosen. These rungs come from
//! [`crate::tokens`] directly, so the frame paints what a screen paints, whichever theme is in
//! force.
//!
//! # The trap, said out loud in the frame
//!
//! `strong` is *bold plus a colour* — [`tokens::strong_style`] adds the modifier — so a
//! normal-weight row for `strong` is a rung **no screen ever paints**. That is not a defect in
//! the frame, it is the point of it: the three sections isolate the weight axis from the colour
//! axis, and the only way to see what weight is doing is to hold colour still. The frame says
//! so, because a reader who takes it as "here are eight more rungs you may use" has read it
//! exactly backwards.
//!
//! # Italic is a terminal capability, and there is no second knob
//!
//! Chris asked whether we could distinguish *slanted* from *italic* — a fair question, since a
//! terminal font family carries a real Italic face and nvim renders it properly in the same
//! environment. **We cannot, and neither can nvim: there is one code.** SGR 3 is
//! ECMA-48 "italicized" and no `oblique` counterpart exists; `ratatui`'s whole modifier set is
//! bold / dim / italic / underlined / blink / reverse / hidden / crossed-out. Whether SGR 3
//! lands on the font's Italic face or on a synthesised slant is the **terminal's font
//! configuration**, not something the application can ask for or even observe.
//!
//! So this section emits byte-for-byte what nvim emits, and in a terminal configured with an
//! italic face it gets the same real italic. A terminal that ignores SGR 3 renders the third
//! section identically to the second — which is itself the answer to "can the design lean on
//! italic", and a thing to *look* at rather than assume either way.
//!
//! One caveat that is ours rather than the terminal's: [`crate::capture`]'s PNG path renders
//! through `soft_ratatui`, a software rasteriser with one face. **Judge italic from the ANSI
//! dump in a real terminal, never from a capture.**

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens;
use crate::widgets::config_table::fit;

/// Width of the rung-name column, so the eye reads *down* a column across the three sections
/// rather than hunting — which is the comparison being asked for.
const NAME: usize = 15;

/// One rung: its name, the token that resolves it, and why it exists.
type Rung = (&'static str, fn() -> Color, &'static str);

/// The four rungs, in the order r02 states them: loudest first.
///
/// The descriptions came from the `[typography]` table this replaces, with one edit —
/// `strong`'s used to open with *"bold, and …"*. Weight is what the sections now show, so a
/// description that also asserts it would be the transcription coming back in words, and would
/// read as a contradiction in the two sections where the row is not bold.
const RUNGS: [Rung; 4] = [
    (
        "Strong",
        tokens::strong,
        "extrapolated PAST the foreground — the one datum that must be seen",
    ),
    (
        "Normal",
        tokens::normal,
        "the theme's own fg: body text of the focused zone, the baseline",
    ),
    (
        "Muted",
        tokens::muted,
        "inactive tabs, unfocused bodies, timestamps, key hints",
    ),
    (
        "Faint",
        tokens::faint,
        "de-emphasised metadata that must still be legible: DEFAULT values, secondary counts",
    ),
];

/// The three sections, in Chris's order: what a screen paints first, then the two the weight
/// axis is isolated in.
///
/// The suffix is his naming — `Strong Bold` / `Strong Normal` / `Strong Italic` reads as one
/// rung across three weights, which is the comparison. The alternative (a bare `Strong` under
/// three headings) puts the weight only in the heading, and the heading scrolls away.
const WEIGHTS: [(&str, &str, Option<Modifier>, &str); 3] = [
    (
        "BOLD",
        "Bold",
        Some(Modifier::BOLD),
        "what a screen paints for `strong` — the other three are never bold on a real screen",
    ),
    (
        "NORMAL WEIGHT",
        "Normal",
        None,
        "the colour axis alone; `Strong Normal` is a rung no screen paints",
    ),
    (
        "ITALIC",
        "Italic",
        Some(Modifier::ITALIC),
        "SGR 3 — the same code nvim emits. Which face it lands on is the terminal's font config; there is no separate oblique code to ask for",
    ),
];

/// The ladder, three times over.
pub struct EmphasisLadder;

impl Widget for EmphasisLadder {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut lines = vec![
            Line::from(Span::styled(
                "The emphasis ladder — the same four rungs at three weights.",
                tokens::strong_style(),
            )),
            Line::from(Span::styled(
                "Live from tokens::{strong,normal,muted,faint}: the colour is what a screen paints.",
                tokens::faint_style(),
            )),
        ];

        for (heading, suffix, modifier, note) in WEIGHTS {
            lines.push(Line::default());
            lines.push(Line::from(vec![
                Span::styled(
                    heading,
                    Style::default()
                        .fg(tokens::header())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!("  {note}"), tokens::faint_style()),
            ]));

            for (rung, colour, description) in RUNGS {
                // One style for the whole row: the label and its description are the same rung
                // at the same weight, and splitting them would make the description a sample of
                // something the row is not.
                let mut style = Style::default().fg(colour());
                if let Some(modifier) = modifier {
                    style = style.add_modifier(modifier);
                }
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  {}", fit(&format!("{rung} {suffix}"), NAME)),
                        style,
                    ),
                    Span::styled(description, style),
                ]));
            }
        }

        Paragraph::new(lines).render(area, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[PropInfo {
        name: "weights",
        ty: "[Modifier; 3]",
        description: "Bold, none, italic — the axis `pantry.toml` had no field for",
    }];

    struct Ladder;

    impl Ingredient for Ladder {
        // Styles with NO section: vocabulary, not an instrument (§16, §17.3).
        fn tab(&self) -> &str {
            "Styles"
        }
        // `Typography` / `Text Hierarchy` are the names the stylesheet gave this entry while it
        // was a TOML table. Keeping them means every command that dumped the old one still
        // dumps this one — the entry changed source, not identity.
        fn group(&self) -> &str {
            "Typography"
        }
        fn name(&self) -> &str {
            "Text Hierarchy"
        }
        fn source(&self) -> &str {
            "wqm_tui::styles::typography"
        }
        fn description(&self) -> &str {
            "The four emphasis rungs at bold, normal and italic — live, so it cannot drift"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            EmphasisLadder.render(area, buf);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![Box::new(Ladder)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::Palette;
    use ratatui::layout::Rect;

    /// Render into a buffer wide enough that nothing is cut, under the palette that ships.
    fn frame(width: u16, height: u16) -> Buffer {
        let mut buf = Buffer::empty(Rect::new(0, 0, width, height));
        EmphasisLadder.render(Rect::new(0, 0, width, height), &mut buf);
        buf
    }

    /// The row at `y`, as text.
    fn row(buf: &Buffer, y: u16) -> String {
        (0..buf.area.width)
            .map(|x| buf[(x, y)].symbol())
            .collect::<String>()
            .trim_end()
            .to_string()
    }

    /// Every rung appears at every weight, and the weights are in Chris's order.
    ///
    /// The order is the request — *"keeping the bold first then … normal weight / and one
    /// section with italics"* — and the rung order inside each section is what lets the eye
    /// read down a column. A reshuffle breaks the comparison without breaking anything that
    /// renders, so nothing else would notice.
    #[test]
    fn twelve_rows_in_the_order_asked_for() {
        let _serial = crate::global_state_lock();
        let previous = Palette::current();
        Palette::set(Palette::Bundled);
        tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());

        let buf = frame(120, 24);
        let labels: Vec<String> = (0..24)
            .map(|y| row(&buf, y))
            .filter(|line| line.starts_with("  ") && !line.trim().is_empty())
            .map(|line| line.trim().split("  ").next().unwrap_or("").to_string())
            .collect();

        assert_eq!(
            labels,
            [
                "Strong Bold",
                "Normal Bold",
                "Muted Bold",
                "Faint Bold",
                "Strong Normal",
                "Normal Normal",
                "Muted Normal",
                "Faint Normal",
                "Strong Italic",
                "Normal Italic",
                "Muted Italic",
                "Faint Italic",
            ],
            "twelve rows, four rungs by three weights, in the order §17.1 states"
        );

        Palette::set(previous);
    }

    /// The weight axis actually varies, and the colour axis actually holds still.
    ///
    /// This is the property the frame exists to show, so asserting the labels alone would be
    /// asserting the caption rather than the thing: a section that forgot its modifier would
    /// still be twelve correctly-named rows. Read out of the rendered buffer, because the style
    /// a cell carries is the only evidence that the modifier survived the render.
    #[test]
    fn one_rung_changes_weight_across_the_sections_and_never_changes_colour() {
        let _serial = crate::global_state_lock();
        let previous = Palette::current();
        Palette::set(Palette::Bundled);
        tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());

        let buf = frame(120, 24);
        // `Strong` is the first row of each section; find the three by their labels rather than
        // by counting rows, so inserting a caption line does not silently move the assertion
        // onto a different rung.
        let rows: Vec<u16> = (0..24)
            .filter(|y| row(&buf, *y).trim_start().starts_with("Strong "))
            .collect();
        assert_eq!(rows.len(), 3, "one `Strong` row per section");

        let cell = |y: u16| buf[(2, y)].style();
        let (bold, normal, italic) = (cell(rows[0]), cell(rows[1]), cell(rows[2]));

        assert!(
            bold.add_modifier.contains(Modifier::BOLD),
            "section 1 is bold"
        );
        assert!(
            !normal.add_modifier.contains(Modifier::BOLD)
                && !normal.add_modifier.contains(Modifier::ITALIC),
            "section 2 carries no weight modifier at all: {:?}",
            normal.add_modifier
        );
        assert!(
            italic.add_modifier.contains(Modifier::ITALIC),
            "section 3 is italic"
        );

        let strong = tokens::strong();
        for (label, style) in [("bold", bold), ("normal", normal), ("italic", italic)] {
            assert_eq!(
                style.fg,
                Some(strong),
                "the {label} section moved the colour as well as the weight, which is the one \
                 thing this frame must not do"
            );
        }

        Palette::set(previous);
    }
}

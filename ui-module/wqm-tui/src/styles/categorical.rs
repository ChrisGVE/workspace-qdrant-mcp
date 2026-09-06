//! The extended tier, rendered — [`crate::categorical`] as something to look at.
//!
//! [`palette`](super::palette) does this for the ten roles; this does it for the hues that
//! carry no role at all. The two questions are different enough to be different frames: the
//! roles are judged one at a time against what each one *means*, and a categorical set can
//! only be judged **as a set**, by whether any two of them are confusable.
//!
//! # The numbers are on the frame because the rule is a measurement
//!
//! Each row carries its ΔE to the nearest reserved role and to the nearest entry already
//! chosen. Those two columns are the exclusion rule and the ordering rule made visible: a
//! reader can check that every row clears the floor, and that the second column descends. A
//! frame of swatches alone would be a picture of a claim rather than the claim.
//!
//! [`Strip`](Frame::Strip) is the other half, and it is the one Chris will actually judge
//! from: the swatches side by side with no numbers at all, because "are these ten
//! distinguishable at a glance" is not a question a table answers.
//!
//! # Two of the four flavours can never be the theme in force
//!
//! `ratatui-themes` bundles Mocha and Latte only. Frappé and Macchiato are rendered from
//! [`crate::categorical::mirrored_palette`], which reproduces that crate's own Catppuccin
//! mapping — a preview of what the tier would be if those flavours were bundled, and marked as
//! such on the frame rather than passed off as a theme anyone can select.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};
use ratatui_themes::ThemePalette;

use crate::categorical::{self, Categorical};
use crate::tokens;

#[cfg(feature = "tui-pantry")]
pub mod ingredient;

/// Cells a swatch occupies. Two, so a hue is a block rather than a character — the eye judges
/// an area, and a single cell of colour reads as text that happens to be coloured.
const SWATCH: &str = "  ";

/// What a frame shows: the measured table, or the bare strip.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Frame {
    /// One row per entry, with both ΔE columns.
    Table,
    /// Every swatch side by side and nothing else.
    Strip,
}

/// The categorical tier of one stated theme, painted on that theme's own background.
pub struct CategoricalFrame {
    theme: ThemePalette,
    /// Set when the theme was synthesised rather than selected — see the module docs.
    mirrored: bool,
    frame: Frame,
}

impl CategoricalFrame {
    pub fn new(theme: ThemePalette, mirrored: bool, frame: Frame) -> Self {
        Self {
            theme,
            mirrored,
            frame,
        }
    }

    /// The line that says which palette is being read, and how many hues came out of it.
    fn header(&self, entries: &[(&'static str, Color)]) -> Line<'static> {
        let mut spans = vec![match categorical::Categorical::flavour(&self.theme) {
            Some(flavour) => Span::styled(
                format!("catppuccin {flavour} — {} hues", entries.len()),
                tokens::strong_style(),
            ),
            None => Span::styled(
                format!("no categorical tier (fallback: {})", entries.len()),
                tokens::strong_style(),
            ),
        }];
        if self.mirrored {
            spans.push(Span::styled(
                "   not bundled — mapping mirrored",
                tokens::faint_style(),
            ));
        }
        Line::from(spans)
    }

    fn table(&self, entries: &[(&'static str, Color)]) -> Vec<Line<'static>> {
        let width = entries.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
        entries
            .iter()
            .enumerate()
            .map(|(i, (name, colour))| {
                let hex = match colour {
                    Color::Rgb(r, g, b) => format!("#{r:02X}{g:02X}{b:02X}"),
                    other => format!("{other:?}"),
                };
                let role = categorical::distance_to_roles(&self.theme, *colour);
                let previous = categorical::distance_to_earlier(entries, i);
                let earlier = match previous {
                    Some(distance) => format!("{distance:>5.1}"),
                    // Entry 0 has nothing before it, and a zero here would read as "identical
                    // to the previous one", which is the opposite of what it means.
                    None => "    —".to_string(),
                };
                // The quantity the ordering rule actually maximises, and the ONLY column that
                // descends. Neither `role` nor `prev` does on its own — `prev` jumps around,
                // which makes the frame look like it contradicts the rule until you take the
                // minimum yourself. Printing it is what lets the ordering be checked by eye.
                let separation = previous.map_or(role, |distance| role.min(distance));
                Line::from(vec![
                    Span::styled(format!("{i:>2} "), tokens::faint_style()),
                    Span::styled(SWATCH, Style::default().bg(*colour)),
                    Span::styled(format!("  {name:<width$}  "), tokens::normal_style()),
                    Span::styled(hex, tokens::faint_style()),
                    Span::styled("   role ", tokens::muted_style()),
                    Span::styled(format!("{role:>5.1}"), tokens::normal_style()),
                    Span::styled("   prev ", tokens::muted_style()),
                    Span::styled(earlier, tokens::normal_style()),
                    Span::styled("   sep ", tokens::muted_style()),
                    Span::styled(format!("{separation:>5.1}"), tokens::strong_style()),
                ])
            })
            .collect()
    }

    fn strip(&self, entries: &[(&'static str, Color)]) -> Vec<Line<'static>> {
        // Three rows of the same swatches: one band is easier to judge than a row of squares,
        // and separability is a property of the band.
        let band: Vec<Span<'static>> = entries
            .iter()
            .map(|(_, colour)| Span::styled("    ", Style::default().bg(*colour)))
            .collect();
        vec![Line::from(band.clone()), Line::from(band.clone()), Line::from(band)]
    }
}

impl Widget for CategoricalFrame {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }
        // The frame paints the theme it is describing, exactly as `PaletteFrame` does: a
        // swatch judged against another theme's background is a judgement about a screen
        // nobody will ever see.
        buf.set_style(area, Style::default().bg(self.theme.bg).fg(self.theme.fg));

        let entries: Vec<(&'static str, Color)> = Categorical::for_theme(&self.theme).iter().collect();
        let mut lines = vec![self.header(&entries), Line::default()];
        lines.extend(match self.frame {
            Frame::Table => self.table(&entries),
            Frame::Strip => self.strip(&entries),
        });
        Paragraph::new(lines).render(
            Rect {
                x: area.x + 2,
                width: area.width.saturating_sub(2),
                ..area
            },
            buf,
        );
    }
}

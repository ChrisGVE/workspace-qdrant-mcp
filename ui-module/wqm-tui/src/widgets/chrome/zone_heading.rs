//! A zone's heading, and the screen-level fact that decides how it is drawn.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens;

/// The bar that marks the focused zone's heading (§3). No colour: the mark is structural, and
/// hue on this element would compete with the selector.
pub const FOCUS_BAR: &str = "▌";

/// Which zone of the screen has the user's attention — a screen-level fact, deliberately.
///
/// §3 gives three treatments (focused, unfocused, and *"no zone focused"* where nothing is
/// dimmed at all), and the third is a property of the screen rather than of any zone. Making
/// it a per-zone flag would make two focused zones representable, and would make a screen
/// where one zone is dimmed and none is focused representable too — both are frames the rule
/// forbids. Passing the same [`Attention`] to every heading removes them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Attention {
    /// The default view: every zone normal, none dimmed, none accented.
    None,
    /// The nth zone is live; every other one recedes.
    Zone(usize),
}

/// The `▌` accent a zone carries when it is the live one, and nothing otherwise.
///
/// Public because the Service hub's lower zone is headed by a
/// [`crate::widgets::chrome::PaneSelector`] rather than by a [`ZoneHeading`] — §4.1 puts the
/// Config↔Logs toggle where the heading would be — and a second copy of "which zone is
/// accented" is how two zones end up accented at once.
///
/// **Open micro-question for Chris.** §3 writes the accent as a prefix (`▌ Heading`), which
/// shifts the heading text two columns to the right the moment a zone takes focus. A gutter —
/// two columns always reserved, the bar drawn into them — would hold the text still. The
/// prefix is what r02 says, so it is what is rendered; the shift is visible in one frame in
/// the pantry's `Zone Heading — Focus Shift` variant.
pub fn accent(index: usize, attention: Attention) -> Option<Span<'static>> {
    match attention {
        Attention::Zone(live) if live == index => Some(Span::styled(
            format!("{FOCUS_BAR} "),
            tokens::normal_style().add_modifier(Modifier::BOLD),
        )),
        _ => None,
    }
}

/// How a zone says it is the live one.
///
/// Two answers, because the Dashboard's cells and the Service hub's zones are different shapes
/// of thing. A hub zone is a band of the screen and the `▌` sits in the margin beside its
/// heading; a Dashboard cell is one of six tiles, and Chris ruled (2026-09-07) that the focused
/// one takes **the tab line's own selector treatment** — the heading in an inverse block, in the
/// selector hue, one space each side, exactly as the active tab is drawn.
///
/// The bar is NOT drawn with the block. Two marks on one heading is noise, and the block is
/// already the strongest signal on the screen. The key letter goes with it: the whole title
/// sits inside the block, so there is no letter left outside to accent — and a cell you are
/// already on does not need to be told which key gets you there.
///
/// [`FocusMark::Bar`] is the default so that every existing caller — `panes::status_band`,
/// `panes::storage`, and the Service hub's zones — renders exactly what it rendered before this
/// enum existed. `panes::tests` holds the digests that prove it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FocusMark {
    /// `▌ Heading` — the bar in the margin, the heading bold beside it.
    #[default]
    Bar,
    /// ` Heading ` inverted in the selector hue — the tab line's treatment, on a cell.
    Block,
}

/// A zone's heading, in the treatment [`Attention`] implies for it.
///
/// # The body is not dimmed, and that is a stated gap
///
/// §3 asks for the *body* of an unfocused zone to recede as well as its heading. The widgets
/// this screen composes have no muted mode — [`crate::widgets::store_health::StoreHealth`]
/// and the rest choose their own rungs — so only the heading carries the state today. The
/// default view (`Attention::None`) is unaffected, since nothing dims there; a frame with a
/// focused zone understates the contrast until the widgets grow the mode.
pub struct ZoneHeading {
    title: String,
    index: usize,
    attention: Attention,
    /// The one key that focuses this zone, when the screen offers one. `None` on a heading
    /// nothing jumps to — the Service hub's zones are reached by tab, not by letter.
    hotkey: Option<char>,
    mark: FocusMark,
}

impl ZoneHeading {
    pub fn new(title: impl Into<String>, index: usize, attention: Attention) -> Self {
        Self {
            title: title.into(),
            index,
            attention,
            hotkey: None,
            mark: FocusMark::default(),
        }
    }

    /// Which mark this heading wears when it is the live zone. See [`FocusMark`]; the default
    /// is the `▌` bar, so a caller that says nothing keeps the treatment it had.
    pub fn focus_mark(mut self, mark: FocusMark) -> Self {
        self.mark = mark;
        self
    }

    /// Whether the screen says THIS zone is the live one.
    fn is_live(&self) -> bool {
        matches!(self.attention, Attention::Zone(live) if live == self.index)
    }

    /// The key that focuses this zone. Its FIRST case-insensitive occurrence in the title is
    /// drawn in [`tokens::accent`] — the same treatment the tab bar gives its jump digits
    /// (Chris, 20260907).
    ///
    /// Derived, not listed: the letter is found in the title rather than carried beside it as
    /// an index, so a heading that gains a count (`Projects` → `Projects (29)`) or is
    /// relabelled from the collection registry cannot leave the accent pointing at the wrong
    /// character. A key that is not in the title accents nothing, which is the honest frame:
    /// there is no letter to press.
    pub fn hotkey(mut self, key: char) -> Self {
        self.hotkey = Some(key);
        self
    }

    /// The heading's own style — what every part of the title that is not the key wears.
    fn title_style(&self) -> Style {
        match self.attention {
            // §3's third row: the default view leaves every heading at the baseline.
            Attention::None => tokens::normal_style(),
            Attention::Zone(live) if live == self.index => {
                tokens::normal_style().add_modifier(Modifier::BOLD)
            }
            Attention::Zone(_) => tokens::muted_style(),
        }
    }

    /// The title, split around the one letter that focuses this zone.
    ///
    /// The letter takes the accent hue **and bold** (Chris, 2026-09-07: *"so that they are
    /// more visible on the screen"*), and keeps everything else the heading wears. Weight is
    /// added rather than substituted, so a focused zone's key is still distinguishable from an
    /// unfocused one's by the rung underneath it — replacing the whole style would have made
    /// the key identical on every zone, which is the one thing the heading's own rung says.
    ///
    /// Bold is not a highlight, so it survives a modal (§6): under one [`tokens::accent`] is
    /// already the muted rung and the weight stays, exactly as the selected tab keeps its bold.
    /// This heading asks no question about modals — [`crate::tokens::modal`] answers it.
    fn title_spans(&self) -> Vec<Span<'static>> {
        let style = self.title_style();
        let at = self.hotkey.and_then(|key| {
            self.title
                .char_indices()
                .find(|(_, c)| c.eq_ignore_ascii_case(&key))
        });
        let Some((at, letter)) = at else {
            return vec![Span::styled(self.title.clone(), style)];
        };
        let key_style = style.fg(tokens::accent()).add_modifier(Modifier::BOLD);
        let before = &self.title[..at];
        let after = &self.title[at + letter.len_utf8()..];
        let mut spans = Vec::with_capacity(3);
        if !before.is_empty() {
            spans.push(Span::styled(before.to_string(), style));
        }
        spans.push(Span::styled(letter.to_string(), key_style));
        if !after.is_empty() {
            spans.push(Span::styled(after.to_string(), style));
        }
        spans
    }

    /// The focused heading as the tab line draws its active tab: the title in an inverse block,
    /// one space each side inside it.
    ///
    /// Under a modal [`tokens::inverted`] is bold muted text with no fill, keeping the two
    /// spaces so nothing on the row moves — which is exactly what [`crate::widgets::tab_bar`]
    /// gets from the same call, and for the same reason (VL §6: the page beneath a modal drops
    /// every colour, and weight is not a colour).
    fn block_spans(&self) -> Vec<Span<'static>> {
        let text = format!(" {} ", self.title);
        vec![Span::styled(text, tokens::inverted(tokens::selector()))]
    }

    fn spans(&self) -> Vec<Span<'static>> {
        if self.mark == FocusMark::Block && self.is_live() {
            return self.block_spans();
        }
        accent(self.index, self.attention)
            .into_iter()
            .chain(self.title_spans())
            .collect()
    }
}

impl Widget for ZoneHeading {
    fn render(self, area: Rect, buf: &mut Buffer) {
        Paragraph::new(Line::from(self.spans())).render(area, buf);
    }
}

#[cfg(test)]
mod tests;

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use ratatui::layout::{Constraint, Layout};
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "title",
            ty: "String",
            description: "The zone's name — a structural label, never a datum",
        },
        PropInfo {
            name: "index",
            ty: "usize",
            description:
                "Which zone this is, so it can compare itself against the screen's attention",
        },
        PropInfo {
            name: "attention",
            ty: "Attention",
            description:
                "SCREEN-level: None, or Zone(n). Two focused zones is not a value this can take",
        },
    ];

    /// Two headings, always — a treatment is relative, and one heading has nothing to be
    /// relative to. Same reason the tests render two.
    struct Pair {
        name: &'static str,
        description: &'static str,
        attention: Attention,
    }

    impl Ingredient for Pair {
        fn group(&self) -> &str {
            "Zone Heading"
        }
        fn name(&self) -> &str {
            self.name
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::chrome::zone_heading"
        }
        fn description(&self) -> &str {
            self.description
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let rows = Layout::vertical([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .split(area);
            ZoneHeading::new("Status", 0, self.attention).render(rows[0], buf);
            ZoneHeading::new("Config", 1, self.attention).render(rows[2], buf);
        }
    }

    /// The same heading with and without focus, stacked — so the two-column shift is a thing
    /// the eye sees rather than a sentence in a handover.
    struct FocusShift;

    impl Ingredient for FocusShift {
        fn group(&self) -> &str {
            "Zone Heading"
        }
        fn name(&self) -> &str {
            "Focus Shift"
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::chrome::zone_heading"
        }
        fn description(&self) -> &str {
            "OPEN (Chris): the accent is a PREFIX, so the same heading sits two columns further right once focused. A gutter would hold it still"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let rows = Layout::vertical([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .split(area);
            ZoneHeading::new("Config", 1, Attention::None).render(rows[0], buf);
            ZoneHeading::new("Config", 1, Attention::Zone(1)).render(rows[1], buf);
        }
    }

    /// A heading whose key letter is lit, in the three states it can be in. The key is
    /// *derived* from the title, so the frame shows a heading that carries a count — the
    /// shape the Dashboard's cells actually have.
    struct Keyed;

    impl Ingredient for Keyed {
        fn group(&self) -> &str {
            "Zone Heading"
        }
        fn name(&self) -> &str {
            "Keyed"
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::chrome::zone_heading"
        }
        fn description(&self) -> &str {
            "The letter that focuses the zone carries `accent` — unfocused, focused, and muted under a modal"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let rows = Layout::vertical([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .split(area);
            ZoneHeading::new("Projects (29)", 0, Attention::None)
                .hotkey('p')
                .render(rows[0], buf);
            ZoneHeading::new("Projects (29)", 0, Attention::Zone(0))
                .hotkey('p')
                .render(rows[1], buf);
            {
                let _modal = tokens::ModalScope::enter();
                ZoneHeading::new("Projects (29)", 0, Attention::Zone(0))
                    .hotkey('p')
                    .render(rows[2], buf);
            }
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Pair {
                name: "No Focus",
                description: "§3's third row: nothing accented, nothing dimmed — the default view",
                attention: Attention::None,
            }),
            Box::new(Pair {
                name: "Focused",
                description: "The lower zone is live: it takes the bar and the weight, the other recedes to muted",
                attention: Attention::Zone(1),
            }),
            Box::new(Keyed),
            Box::new(FocusShift),
        ]
    }
}

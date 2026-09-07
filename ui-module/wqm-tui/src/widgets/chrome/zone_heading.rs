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
    modal: bool,
}

impl ZoneHeading {
    pub fn new(title: impl Into<String>, index: usize, attention: Attention) -> Self {
        Self {
            title: title.into(),
            index,
            attention,
            hotkey: None,
            modal: false,
        }
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

    /// Whether a modal owns the input. Under one the key letter goes muted, as the tab bar's
    /// digits do (VL §6) — nothing else about the heading changes, because the `▌` bar and
    /// the weight are structure rather than highlight (§3).
    pub fn under_modal(mut self, modal: bool) -> Self {
        self.modal = modal;
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
    /// The letter keeps the heading's weight and takes only its hue, so a focused zone's key
    /// is bold-and-accent and an unfocused one's is normal-and-accent. Replacing the whole
    /// style would have made the key the same on every zone, which is the one thing the
    /// heading's own rung is there to say.
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
        let key_style = style.fg(if self.modal {
            tokens::muted()
        } else {
            tokens::accent()
        });
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

    fn spans(&self) -> Vec<Span<'static>> {
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
mod tests {
    use super::*;
    use crate::widgets::chrome::test_support::{render, row, style_at, Restore};

    #[test]
    fn a_screen_with_no_focused_zone_dims_nothing_and_accents_nothing() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // §3's third row. Two headings are rendered so the assertion is about the SCREEN:
        // with one heading, "nothing is dimmed" is vacuous — there is nothing to dim it
        // relative to.
        for index in 0..2 {
            let buf = render(ZoneHeading::new("Status", index, Attention::None));
            let line = row(&buf, 0);
            assert!(!line.contains(FOCUS_BAR), "no accent: {line:?}");
            assert_eq!(
                style_at(&buf, 0).fg,
                Some(tokens::normal()),
                "no zone dimmed on a screen with no focus"
            );
            assert!(!style_at(&buf, 0).add_modifier.contains(Modifier::BOLD));
        }
    }

    #[test]
    fn focusing_one_zone_accents_it_and_recedes_the_other() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let attention = Attention::Zone(1);

        let live = render(ZoneHeading::new("Config", 1, attention));
        let line = row(&live, 0);
        assert!(line.starts_with("▌ Config"), "{line:?}");
        assert!(
            style_at(&live, 0).add_modifier.contains(Modifier::BOLD),
            "the focused heading is bold"
        );

        let receded = render(ZoneHeading::new("Status", 0, attention));
        assert!(!row(&receded, 0).contains(FOCUS_BAR));
        assert_eq!(style_at(&receded, 0).fg, Some(tokens::muted()));
    }

    #[test]
    fn taking_focus_shifts_the_heading_text_two_columns_right() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // The open question, stated as a measurement rather than as prose. If the prefix ever
        // becomes a gutter this test is what fails, and it fails saying exactly what changed.
        let idle = row(&render(ZoneHeading::new("Config", 1, Attention::None)), 0);
        let live = row(
            &render(ZoneHeading::new("Config", 1, Attention::Zone(1))),
            0,
        );

        // Counted in CHARACTERS, not bytes: `▌` is three bytes wide and one column wide, and
        // a byte offset would report a three-column shift the screen does not have.
        let column_of_heading = |line: &str| {
            line.chars()
                .position(|c| c == 'C')
                .expect("the heading is drawn")
        };
        let idle_x = column_of_heading(&idle);
        let live_x = column_of_heading(&live);
        assert_eq!(idle_x, 0);
        assert_eq!(
            live_x - idle_x,
            2,
            "the accent is a PREFIX (r02), so focus moves the text: {idle:?} / {live:?}"
        );
    }

    /// The accent is ONE letter — the key — and every other cell of the heading keeps the
    /// heading's own rung. A title whose key is not its initial is used deliberately, so the
    /// "before" half of the split is not empty and can be checked.
    #[test]
    fn the_key_letter_is_accented_and_nothing_beside_it_is() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(ZoneHeading::new("Last Errors", 0, Attention::Zone(0)).hotkey('e'));
        let line = row(&buf, 0);
        // `▌ Last Errors` — counted in CHARACTERS, since `▌` is three bytes and one column.
        let x = line
            .chars()
            .position(|c| c == 'E')
            .expect("the key letter is drawn") as u16;
        assert_eq!(
            style_at(&buf, x).fg,
            Some(tokens::accent()),
            "the key letter carries the accent: {line:?}"
        );
        // The heading's WEIGHT is untouched — a focused zone is bold, key letter included.
        assert!(style_at(&buf, x).add_modifier.contains(Modifier::BOLD));
        assert_eq!(
            style_at(&buf, x).add_modifier,
            style_at(&buf, x + 1).add_modifier,
            "the letter wears the heading's modifiers, only its hue differs"
        );
        assert!(
            !style_at(&buf, x).add_modifier.contains(Modifier::UNDERLINED),
            "no underline — the tab bar's digits carry none either"
        );

        for neighbour in [x - 1, x + 1] {
            assert_eq!(
                style_at(&buf, neighbour).fg,
                Some(tokens::normal()),
                "column {neighbour} is heading text, not the key: {line:?}"
            );
        }
    }

    /// VL §6: under a modal the page drops every highlight. The key letter goes muted, and the
    /// `▌` and the weight — structure, not highlight — do not move.
    #[test]
    fn a_modal_mutes_the_key_letter_and_leaves_the_bar_and_the_weight() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let live = render(ZoneHeading::new("Rules (0)", 1, Attention::Zone(1)).hotkey('r'));
        let under = render(
            ZoneHeading::new("Rules (0)", 1, Attention::Zone(1))
                .hotkey('r')
                .under_modal(true),
        );

        let line = row(&live, 0);
        let x = line
            .chars()
            .position(|c| c == 'R')
            .expect("the key letter is drawn") as u16;
        assert_eq!(style_at(&live, x).fg, Some(tokens::accent()));
        assert_eq!(style_at(&under, x).fg, Some(tokens::muted()));

        assert_eq!(row(&under, 0), line, "a modal moves nothing");
        assert!(row(&under, 0).starts_with(&format!("{FOCUS_BAR} ")));
        assert!(
            style_at(&under, x).add_modifier.contains(Modifier::BOLD),
            "the focused zone keeps its weight under a modal"
        );
    }

    /// A key that is not in the title accents nothing at all. The honest frame: there is no
    /// letter to press, so no letter is lit — and nothing panics looking for one.
    #[test]
    fn a_key_absent_from_the_title_accents_no_cell() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(ZoneHeading::new("Rules", 0, Attention::None).hotkey('z'));
        assert_eq!(row(&buf, 0).trim_end(), "Rules");
        for x in 0..crate::widgets::chrome::test_support::AREA.width {
            assert_ne!(
                style_at(&buf, x).fg,
                Some(tokens::accent()),
                "column {x} was accented for a key the title does not contain"
            );
        }
    }
}

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
            ZoneHeading::new("Projects (29)", 0, Attention::Zone(0))
                .hotkey('p')
                .under_modal(true)
                .render(rows[2], buf);
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

//! A zone's heading, and the screen-level fact that decides how it is drawn.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Modifier,
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
}

impl ZoneHeading {
    pub fn new(title: impl Into<String>, index: usize, attention: Attention) -> Self {
        Self {
            title: title.into(),
            index,
            attention,
        }
    }

    fn spans(&self) -> Vec<Span<'static>> {
        let style = match self.attention {
            // §3's third row: the default view leaves every heading at the baseline.
            Attention::None => tokens::normal_style(),
            Attention::Zone(live) if live == self.index => {
                tokens::normal_style().add_modifier(Modifier::BOLD)
            }
            Attention::Zone(_) => tokens::muted_style(),
        };
        accent(self.index, self.attention)
            .into_iter()
            .chain(std::iter::once(Span::styled(self.title.clone(), style)))
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
            Box::new(FocusShift),
        ]
    }
}

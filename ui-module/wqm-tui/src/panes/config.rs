//! The Service hub's lower zone: the Config ↔ Logs selector and the table under it.
//!
//! §16 names this a pane, and §4.1 is why it looks different from the band above it: the lower
//! band's heading **is** the sub-screen selector, so the zone accent is drawn beside the
//! selector rather than by a [`ZoneHeading`] of its own. Two zones, two ways of being headed,
//! one rule about which of them is accented — and that rule lives in
//! [`crate::widgets::chrome::accent`] precisely so this pane and a `ZoneHeading` cannot
//! disagree about which zone is live.
//!
//! # The table is not inset, and that is deliberate
//!
//! Everything else in a zone starts at [`crate::widgets::chrome::MARGIN`]. The config table
//! reaches the same column through **its own** arithmetic, which is what lets its KEY column
//! line up under the store roles in the band above without either widget knowing the other
//! exists. Insetting it here would indent it twice.
//!
//! # The Logs pane is not built
//!
//! §4.1's lower band toggles Config ↔ Logs. Nothing in this crate renders logs, so the selector
//! always shows Config selected. A `Logs` variant would be a frame of an empty band — a
//! depiction of a screen that does not exist yet, which is the one thing a storyboard must not
//! produce.

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    text::Line,
    widgets::{Paragraph, Widget},
};

use crate::widgets::chrome::{accent, inset, Attention, PaneSelector};
use crate::widgets::config_table::ConfigTable;

/// Rows above the table: the selector, then a row of negative space. §6 divides with space
/// rather than boxes, so the gap is structure and not padding.
const HEADER_ROWS: u16 = 2;

/// The lower zone, selector and table together.
pub struct ConfigPane {
    config: ConfigTable,
    zone: usize,
    attention: Attention,
}

impl ConfigPane {
    /// `zone` is this pane's index in the screen's zone order; `attention` is the screen's
    /// answer to which zone is live (§16: the view owns focus, the pane owns what focus looks
    /// like inside itself).
    pub fn new(config: ConfigTable, zone: usize, attention: Attention) -> Self {
        Self {
            config,
            zone,
            attention,
        }
    }
}

impl Widget for ConfigPane {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let [selector, _gap, table] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(HEADER_ROWS - 1),
            Constraint::Min(0),
        ])
        .areas(area);

        let head = inset(selector);
        let accent = accent(self.zone, self.attention);
        let accent_width = accent
            .as_ref()
            .map(|span| span.content.chars().count() as u16)
            .unwrap_or(0);
        if let Some(accent) = accent {
            Paragraph::new(Line::from(accent)).render(head, buf);
        }
        PaneSelector::new(vec!["Config".into(), "Logs".into()], 0).render(
            Rect {
                x: head.x + accent_width,
                width: head.width.saturating_sub(accent_width),
                ..head
            },
            buf,
        );

        self.config.render(table, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use crate::views::service::frames;
    use crate::widgets::config_table::{Edit, Focus};
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "config",
            ty: "ConfigTable",
            description: "The table. Its focus is what decides the screen's mode indicator",
        },
        PropInfo {
            name: "attention",
            ty: "Attention",
            description: "Which zone the SCREEN says is live — the accent sits beside the selector",
        },
    ];

    struct Variant(&'static str, &'static str, fn() -> ConfigPane);

    impl Ingredient for Variant {
        fn tab(&self) -> &str {
            "Panes"
        }
        fn group(&self) -> &str {
            "Config Pane"
        }
        fn name(&self) -> &str {
            self.0
        }
        fn source(&self) -> &str {
            "wqm_tui::panes::config"
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

    /// The zone index the Service hub gives this pane — its own number, not a local one, so a
    /// pane frame cannot accent a zone the screen would not.
    const ZONE: usize = 1;

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Variant(
                "Unfocused",
                "The zone at rest: no accent, the table showing values against their defaults",
                || {
                    ConfigPane::new(
                        ConfigTable::new(frames::config_rows()),
                        ZONE,
                        Attention::None,
                    )
                },
            )),
            Box::new(Variant(
                "Focused",
                "The accent beside the selector — §4.1 puts the toggle where the heading would be",
                || {
                    ConfigPane::new(
                        ConfigTable::new(frames::config_rows()),
                        ZONE,
                        Attention::Zone(ZONE),
                    )
                },
            )),
            Box::new(Variant(
                "Editing",
                "An edit open: the third highlight role, under an accented selector",
                || {
                    ConfigPane::new(
                        ConfigTable::new(frames::config_rows())
                            .focus(Focus::Editing(frames::DEBOUNCE, Edit::insert("2000"))),
                        ZONE,
                        Attention::Zone(ZONE),
                    )
                },
            )),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::views::service::frames;
    use crate::widgets::chrome::test_support::Restore;

    fn render(attention: Attention) -> Buffer {
        let area = Rect::new(0, 0, 100, 14);
        let mut buf = Buffer::empty(area);
        ConfigPane::new(ConfigTable::new(frames::config_rows()), 1, attention)
            .render(area, &mut buf);
        buf
    }

    fn row(buf: &Buffer, y: u16) -> String {
        (0..buf.area.width)
            .map(|x| buf[(x, y)].symbol())
            .collect::<String>()
            .trim_end()
            .to_string()
    }

    /// The accent shifts the selector rather than overprinting it.
    ///
    /// §3 writes the accent as a **prefix**, so a focused zone's selector starts two columns to
    /// the right. That shift is a live open question for Chris — a gutter would hold it still —
    /// and the thing this guards is the other outcome entirely: an accent drawn *over* the
    /// selector's first characters, which loses a letter and looks like a rendering glitch.
    #[test]
    fn the_accent_moves_the_selector_right_instead_of_covering_it() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let quiet = row(&render(Attention::None), 0);
        let live = row(&render(Attention::Zone(1)), 0);

        assert!(
            quiet.contains("Config"),
            "the selector is always drawn: {quiet:?}"
        );
        assert!(
            live.contains("Config"),
            "the accent overprinted the selector: {live:?}"
        );
        assert!(
            live.find("Config").unwrap() > quiet.find("Config").unwrap(),
            "§3's prefix moves the text right; nothing moved: {quiet:?} -> {live:?}"
        );
    }

    /// The table keeps its own margin — it is NOT inset by the pane.
    ///
    /// Insetting it here would indent it twice, and the visible cost is precise: the KEY column
    /// stops lining up under the store roles in the band above, which is the one alignment the
    /// whole screen is judged on. Two columns is a difference no reviewer would name and every
    /// reviewer would feel.
    ///
    /// **The target is [`crate::widgets::chrome::MARGIN`], not the selector's own column.** The
    /// first version compared the two rows and failed at 2 against 3 — the selector's block
    /// opens with a pad cell, so its text starts one column in. The subject was correct and the
    /// instrument was measuring something else, which is this crate's recurring lesson: a
    /// failing test is not evidence about the subject until you check what it counts.
    #[test]
    fn the_table_is_not_inset_a_second_time() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(Attention::None);
        // The first table row is the `qdrant` group header, two rows below the selector.
        let table = row(&buf, HEADER_ROWS);
        let indent = table.chars().count() - table.trim_start().chars().count();

        assert_eq!(
            indent,
            crate::widgets::chrome::MARGIN as usize,
            "the table starts at column {indent}; the screen margin is \
             {}, so a second inset here would put it at {}",
            crate::widgets::chrome::MARGIN,
            crate::widgets::chrome::MARGIN * 2
        );
    }
}

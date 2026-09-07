//! Line 1 of every screen — the product's name, and the ten tabs you can jump to.
//!
//! This is the first row of the constant top (Chris, 20260906): the same two things in the
//! same two places on tabs 1–9 and on the Service hub, so the row a user's eye lands on first
//! never moves. It owns no rendering of its own — it is the title span and
//! [`crate::widgets::tab_bar::TabBar`] on one row — which is what §16 means by a view
//! composing built widgets: the app bar is chrome, and chrome is a widget too.
//!
//! # The title is a constant, not a screen name
//!
//! `WQM TUI`, always, on every tab (Chris, 20260906) — *honest, it is the entry command*. A
//! screen's own name belongs to [`crate::widgets::chrome::title_bar`], one row further down
//! and only on screens that have one. So there is no width ladder here and no abbreviation
//! table: the string is seven columns wide and stays seven columns wide.
//!
//! It is drawn **bold at the normal rung** — the theme's own foreground, never a literal
//! white, because a light theme exists and would render white text on white paint. That is the
//! same treatment [`crate::widgets::chrome::title_bar`] gives a screen name and for the same
//! reason: `strong` is reserved for the one datum that must be seen, and a product name is
//! never a datum.
//!
//! # What gives when the row does not fit
//!
//! The gap first, then the right edge (Chris, 20260906). The title is the fixed half — it
//! cannot shrink and abbreviating it would be inventing a second name — so the three columns
//! of quiet between it and the tabs collapse to one, and after that the tab row simply runs
//! off the right. Losing the last tabs is the honest failure: their numbers still jump, and a
//! truncated row says *there is more here* in a way a re-flowed one does not.
//!
//! At the storyboard's 125 columns everything fits with room to spare, and
//! [`tests::the_full_title_and_all_ten_tabs_fit_the_storyboards_own_width`] pins that.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Modifier,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens;
use crate::widgets::tab_bar::{Tab, TabBar};

/// What line 1 always says. A constant, not a screen name — see the module docs.
pub const TITLE: &str = "WQM TUI";

/// Columns of quiet between the title and the first tab when the row has room for them.
///
/// Three rather than one, because the title and the tab row are two different *kinds* of
/// thing and a single space would read them as one phrase — `WQM TUI 1 Dashboard`.
pub const TITLE_GAP: u16 = 3;

/// What the gap collapses to when the row is short. Never zero: at zero the title's last
/// letter and the first jump digit touch, and `TUI1` is a word.
pub const MIN_GAP: u16 = 1;

/// The Service hub's index in [`TabBar::storyboard_tabs`] — tab ten, and therefore index nine.
///
/// Named here as well as in [`crate::views::service`] because the app bar is what a frame
/// selects a tab on, and a frame that hard-coded `9` would be a frame nobody could check
/// against the row it is describing.
pub const SERVICE_TAB: usize = 9;

/// The title and the tab row on one line — the constant top of every screen.
pub struct AppBar {
    tabs: Vec<Tab>,
    active: usize,
}

impl AppBar {
    /// The storyboard's own ten tabs, with `active` selected.
    pub fn new(active: usize) -> Self {
        Self::with_tabs(TabBar::storyboard_tabs(), active)
    }

    /// A stated tab row — how a frame shows an alarm on a tab that is not the selected one.
    pub fn with_tabs(tabs: Vec<Tab>, active: usize) -> Self {
        Self { tabs, active }
    }

    fn bar(self) -> TabBar {
        TabBar::new(self.tabs, self.active)
    }

    /// Columns this row wants: the title, the full gap, and every tab drawn in full.
    ///
    /// Public because it is the only honest way to ask *"does this fit?"* — a caller that
    /// answered it by rendering into a scratch buffer would be measuring a copy of the layout
    /// rather than the layout.
    pub fn measured_width(&self) -> u16 {
        TITLE.chars().count() as u16 + TITLE_GAP + TabBar::width_of(&self.tabs, self.active)
    }
}

impl Widget for AppBar {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }
        // One row, whatever it is handed: everything below line 1 belongs to another element,
        // and a Paragraph given three rows would happily wrap a long tab row into them.
        let area = Rect { height: 1, ..area };

        let title_width = TITLE.chars().count() as u16;
        let wanted = self.measured_width();
        // The gap is the only thing that gives before the right edge does.
        let gap = if wanted <= area.width {
            TITLE_GAP
        } else {
            MIN_GAP
        };

        Paragraph::new(Line::from(Span::styled(
            TITLE,
            tokens::normal_style().add_modifier(Modifier::BOLD),
        )))
        .render(
            Rect {
                width: title_width.min(area.width),
                ..area
            },
            buf,
        );

        let start = title_width + gap;
        if start >= area.width {
            return;
        }
        self.bar().render(
            Rect {
                x: area.x + start,
                width: area.width - start,
                ..area
            },
            buf,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widgets::chrome::test_support::Restore;

    const STORYBOARD_WIDTH: u16 = 125;

    fn render(bar: AppBar, width: u16) -> Buffer {
        let area = Rect::new(0, 0, width, 1);
        let mut buf = Buffer::empty(area);
        bar.render(area, &mut buf);
        buf
    }

    fn row(buf: &Buffer) -> String {
        (0..buf.area.width)
            .map(|x| buf.cell((x, 0)).expect("cell in area").symbol())
            .collect()
    }

    #[test]
    fn the_full_title_and_all_ten_tabs_fit_the_storyboards_own_width() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let line = row(&render(AppBar::new(0), STORYBOARD_WIDTH));
        assert!(line.starts_with(TITLE), "the title leads the row: {line:?}");
        for tab in TabBar::storyboard_tabs() {
            assert!(
                line.contains(&tab.label),
                "{:?} did not fit at {STORYBOARD_WIDTH} columns: {line:?}",
                tab.label
            );
        }
        assert!(
            line.contains(&format!("{}{}{}", TITLE, " ".repeat(TITLE_GAP as usize), "1")),
            "with room to spare the full gap is kept: {line:?}"
        );
    }

    #[test]
    fn a_row_too_narrow_loses_the_gap_before_it_loses_a_tab() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let full = AppBar::new(0).measured_width();
        // One column short of comfortable: the gap must go, and nothing else yet.
        let squeezed = row(&render(AppBar::new(0), full - 1));
        assert!(
            squeezed.starts_with(&format!("{}{}1", TITLE, " ".repeat(MIN_GAP as usize))),
            "the gap collapses to its minimum first: {squeezed:?}"
        );
        assert!(
            squeezed.contains("Dashboard"),
            "no tab is dropped to save a gap: {squeezed:?}"
        );
    }

    #[test]
    fn a_narrow_row_truncates_at_the_right_edge_rather_than_reflowing() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(AppBar::new(0), 80);
        assert_eq!(buf.area.height, 1, "the app bar is one row, whatever it holds");
        let line = row(&buf);
        assert!(line.starts_with(TITLE), "the title survives: {line:?}");
        assert!(line.contains("Dashboard"), "the leading tabs survive: {line:?}");
        assert!(
            !line.contains("Service"),
            "the row runs off the right rather than re-flowing: {line:?}"
        );
    }

    /// The Service tab is number ten, so this is the one frame in which the split matters.
    #[test]
    fn the_service_tab_is_reached_with_the_zero() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let line = row(&render(AppBar::new(SERVICE_TAB), STORYBOARD_WIDTH));
        assert!(line.contains("10  Service "), "{line:?}");
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use crate::tokens::Health;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "tabs",
            ty: "Vec<Tab>",
            description: "The ten storyboard tabs; each carries its jump number and any alarm",
        },
        PropInfo {
            name: "active",
            ty: "usize",
            description: "Index of the selected tab — the one inverted block on the row",
        },
        PropInfo {
            name: "modal",
            ty: "bool",
            description: "A modal owns the input: every highlight on this row drops away",
        },
    ];

    /// The width a variant is drawn at. `None` takes whatever the preview cell offers, which
    /// is how the row is judged against the terminal the pantry is actually running in.
    /// Name, description, builder, stated width, and whether the frame is drawn beneath a
    /// modal. The last is a property of the *frame* rather than of the bar — the bar has no
    /// opinion any more ([`crate::tokens::modal`]) — so it is the preview that opens the scope,
    /// exactly as a real view does around its page.
    struct Variant(
        &'static str,
        &'static str,
        fn() -> AppBar,
        Option<u16>,
        bool,
    );

    impl Ingredient for Variant {
        fn group(&self) -> &str {
            "App Bar"
        }
        fn name(&self) -> &str {
            self.0
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::chrome::app_bar"
        }
        fn description(&self) -> &str {
            self.1
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let width = self.3.unwrap_or(area.width).min(area.width);
            let _modal = self.4.then(tokens::ModalScope::enter);
            (self.2)().render(
                Rect {
                    height: 1,
                    width,
                    ..area
                },
                buf,
            );
        }
    }

    /// The Queue tab, alarming. Not the Service tab, deliberately: Service is the one tab whose
    /// alarm the Service view already draws, and an accent digit beside a *warning*-hued label
    /// is the pair that has never been looked at.
    fn queue_degraded() -> Vec<Tab> {
        let mut tabs = TabBar::storyboard_tabs();
        tabs[1].alarm = Some(Health::Degraded);
        tabs
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Variant(
                "Default",
                "Dashboard selected — the row every screen opens on, and the accent digits against it",
                || AppBar::new(0),
                None,
                false,
            )),
            Box::new(Variant(
                "Service active",
                "Tab ten: only the 0 is accented, because 0 is the key that jumps here",
                || AppBar::new(SERVICE_TAB),
                None,
                false,
            )),
            Box::new(Variant(
                "Under modal",
                "Every highlight gone — no accent, no inverse block, the selected tab bold only",
                || AppBar::new(0),
                None,
                true,
            )),
            Box::new(Variant(
                "Alarm on Queue",
                "An accent digit immediately left of a warning-hued label: do the two hues fight?",
                || AppBar::with_tabs(queue_degraded(), 0),
                None,
                false,
            )),
            Box::new(Variant(
                "Narrow 80",
                "Eighty columns: the gap has already gone and the row runs off the right edge",
                || AppBar::new(0),
                Some(80),
                false,
            )),
        ]
    }
}

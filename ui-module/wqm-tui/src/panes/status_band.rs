//! The Service hub's upper zone: the store federation beside the liveness master.
//!
//! §16 names this a pane — *"the Service hub's status band"* — and until now it was six lines
//! inside `views::service::render`, which meant the one thing a design instrument must not do
//! to an element: it could not be looked at on its own.
//!
//! # Why two columns and not one list
//!
//! They answer different questions. The left column is *"what is behind this deployment"*; the
//! right is *"is the thing that knows answering"*. §7 makes the second the **master** of the
//! first — an unreachable daemon does not make the stores unhealthy, it makes every reading of
//! them unknown — and two columns is what says that a list of four rows would not.
//!
//! # The daemon is named once
//!
//! §7 lists the daemon among the federation's axes *and* gives it a panel. Rendering it in both
//! places puts the same glyph on the screen twice and invites the two to disagree, so this pane
//! takes the backing stores and the daemon report **separately**: the panel says how the daemon
//! is, the store list says how everything it is responsible for is.

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    widgets::Widget,
};
use wqm_client::DaemonReport;

use crate::widgets::chrome::{inset, Attention, ZoneHeading};
use crate::widgets::store_health::StoreRow;
use crate::widgets::{daemon_status::DaemonPanel, store_health::StoreHealth};

/// Rows the band itself reserves, under its heading.
///
/// Four is what [`DaemonPanel`] emits at its longest (daemon, version, detail, index). The band
/// never shrinks below it, so the rule under the band does not move when a daemon starts
/// reporting a detail line — a screen whose furniture shifts on a status change is a screen
/// that reads as broken rather than as informative.
const BAND_ROWS: u16 = 4;

/// Width of the store column. Wide enough for the longest role plus its backing name; the
/// daemon panel takes what is left, which is what keeps it right of the stores on any screen
/// this design targets.
const STORE_COLUMN: u16 = 30;

/// The status band, headed and composed.
pub struct StatusBand {
    stores: Vec<StoreRow>,
    daemon: DaemonReport,
    zone: usize,
    attention: Attention,
}

impl StatusBand {
    /// Rows the whole pane occupies, heading included.
    ///
    /// A view asks the pane how tall it is rather than counting its parts, because the view is
    /// exactly the place that cannot see them: `views::service` used to spell `Length(1)` for
    /// the heading and `Length(BAND_ROWS)` for the band, and adding a row to the band would
    /// have been a change in two files with nothing to catch the second one being missed.
    pub const ROWS: u16 = 1 + BAND_ROWS;

    /// `zone` is this pane's index in the screen's zone order and `attention` is the
    /// screen-level answer to which zone is live — §16: *the view owns which pane has focus*.
    pub fn new(
        stores: Vec<StoreRow>,
        daemon: DaemonReport,
        zone: usize,
        attention: Attention,
    ) -> Self {
        Self {
            stores,
            daemon,
            zone,
            attention,
        }
    }
}

impl Widget for StatusBand {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let [heading, band] =
            Layout::vertical([Constraint::Length(1), Constraint::Length(BAND_ROWS)]).areas(area);

        ZoneHeading::new("Status", self.zone, self.attention).render(inset(heading), buf);

        let [stores, daemon] =
            Layout::horizontal([Constraint::Length(STORE_COLUMN), Constraint::Min(0)])
                .areas(inset(band));
        StoreHealth::new(self.stores).render(stores, buf);
        DaemonPanel::new(self.daemon).render(daemon, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use crate::tokens::Health;
    use crate::views::service::frames;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "stores",
            ty: "Vec<StoreRow>",
            description: "The backing stores. The daemon is NOT among them — it has the panel",
        },
        PropInfo {
            name: "daemon",
            ty: "DaemonReport",
            description: "N49's answer — the liveness master, and the master of the readings",
        },
        PropInfo {
            name: "attention",
            ty: "Attention",
            description:
                "Which zone the SCREEN says is live; the pane decides what that looks like",
        },
    ];

    struct Variant(&'static str, &'static str, fn() -> StatusBand);

    impl Ingredient for Variant {
        fn tab(&self) -> &str {
            "Panes"
        }
        fn group(&self) -> &str {
            "Status Band"
        }
        fn name(&self) -> &str {
            self.0
        }
        fn source(&self) -> &str {
            "wqm_tui::panes::status_band"
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

    /// The zone index the Service hub gives this pane. A pane frame that dimmed the wrong zone
    /// would still render, so the frames use the screen's own number rather than a local one.
    const ZONE: usize = 0;

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Variant(
                "Nominal",
                "Everything healthy and the daemon serving — the band at rest",
                || {
                    StatusBand::new(
                        frames::stores(Health::Healthy),
                        frames::serving(),
                        ZONE,
                        Attention::None,
                    )
                },
            )),
            Box::new(Variant(
                "Degraded store",
                "One store past its SLA: the master is fine, so the reading is trusted and shown",
                || {
                    StatusBand::new(
                        frames::stores(Health::Degraded),
                        frames::serving(),
                        ZONE,
                        Attention::None,
                    )
                },
            )),
            Box::new(Variant(
                "Daemon unreachable",
                "§7's master rule: the daemon is not answering, so every store reading below it is unknown",
                || {
                    StatusBand::new(
                        frames::stores(Health::Healthy),
                        frames::unreachable(),
                        ZONE,
                        Attention::None,
                    )
                },
            )),
            Box::new(Variant(
                "Focused",
                "The zone accent on the heading — what the band looks like when it is the live zone",
                || {
                    StatusBand::new(
                        frames::stores(Health::Healthy),
                        frames::serving(),
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
    use crate::tokens::Health;
    use crate::views::service::frames;
    use crate::widgets::chrome::test_support::Restore;

    fn render(attention: Attention) -> Buffer {
        let area = Rect::new(0, 0, 100, StatusBand::ROWS);
        let mut buf = Buffer::empty(area);
        StatusBand::new(
            frames::stores(Health::Healthy),
            frames::serving(),
            0,
            attention,
        )
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

    /// Both columns are drawn, and the daemon's starts where the stores' ends.
    ///
    /// The failure this catches is the one a screenshot hides: a band whose two halves overlap
    /// renders perfectly well, with the daemon panel simply painted over the last store's
    /// backing name. Read the column out of the buffer rather than recomputing it — and count
    /// in CHARACTERS, which is the mistake this crate has already made once.
    #[test]
    fn the_daemon_panel_begins_where_the_store_column_ends() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(Attention::None);
        let first = row(&buf, 1);
        let column = crate::widgets::chrome::MARGIN + STORE_COLUMN;

        assert!(
            first.chars().count() > column as usize,
            "nothing is drawn at or past the daemon's first column: {first:?}"
        );
        // The store column's own content must stop before the boundary, or the two halves are
        // sharing cells and the layout is decorative.
        let stores: String = first.chars().take(column as usize).collect();
        assert!(
            stores.trim_end().chars().count() < column as usize,
            "the store column fills its width exactly, so an overlap would be invisible: {stores:?}"
        );
    }

    /// The heading takes the zone accent when the screen says this zone is live, and not
    /// otherwise.
    ///
    /// Rendered twice on purpose: with one frame, "the accent is absent" has nothing to be
    /// absent relative to, and a pane that never drew an accent at all would pass.
    #[test]
    fn the_heading_accents_only_when_this_zone_is_the_live_one() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let quiet = row(&render(Attention::None), 0);
        let live = row(&render(Attention::Zone(0)), 0);
        let other = row(&render(Attention::Zone(1)), 0);

        assert!(
            quiet.contains("Status"),
            "the heading is always drawn: {quiet:?}"
        );
        assert_ne!(
            quiet, live,
            "the live zone must look different from the default view"
        );
        assert_eq!(
            quiet, other,
            "a zone that is not the live one is not accented — only dimmed, which is a style"
        );
    }
}

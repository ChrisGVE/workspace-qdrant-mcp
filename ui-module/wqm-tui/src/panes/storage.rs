//! The Dashboard's **Storage** cell — STORYBOARD §4.1's fourth zone, and the first of the six.
//!
//! §4.1 names the Dashboard's cells and closes the question that was open here for a week:
//! *"Dashboard — 6 zones: Projects · Queue · Libraries · **Storage** · Embedder · Errors"*, which
//! maps onto §16's 2 × 3 grid, one pane per cell. Storage is built first because it is the only
//! one whose drill-down already exists — the Service hub — so it is the one cell whose agreement
//! with its drill-down can be *checked* rather than promised.
//!
//! # A cell is a projection, never a summary it invented
//!
//! STORYBOARD §4.2 for this row: *"workspace at a glance; **every cell agrees with its
//! drill-down** (SYS-3)"*, and §4.5's TF-03 says it of a number in particular — *"count from the
//! SAME source as the view it drills into"*.
//!
//! This pane takes **the same two arguments [`crate::panes::StatusBand`] takes**, in the same
//! order: the backing stores, and N49's report. That is the strongest available form of the
//! rule — not a test that two screens agree, but a constructor that cannot be handed different
//! data than the screen it drills into without the caller doing it on purpose.
//!
//! # The daemon has no cell, so the master rule has to be honoured *here*
//!
//! On the Service hub the liveness master is drawn beside the federation, in a panel of its own,
//! under a screen-wide wash. The Dashboard has none of that: there is no daemon cell among the
//! six, and this cell is one of six things competing for a glance.
//!
//! So a Storage cell that painted three green dots while the daemon was unreachable would be
//! **the exact lie §6.19 and [`crate::health::SystemHealth::rollup`] already forbid one level
//! up**: when the master is not answering, the component readings are *unknown*, not good. The
//! rollup's own words are *"an unreachable daemon rolls up as itself, not as a count"*, and this
//! cell says the same thing the same way — the master's statement, in N49's own wording, instead
//! of readings nobody currently has.
//!
//! **Stated assumption, one line to correct** (`§18`, the standing top-down instruction): the
//! unreachable body also names the roles it is *not* reporting on, faint and glyphless. The
//! alternative is a cell that is one line tall in the state where the user most wants to know
//! what is affected. Naming a role claims nothing about it — the glyph is what claims — so this
//! stays inside §4's vocabulary rather than inventing a fourth health state, which would be a
//! change to the visual language and therefore Chris's.
//!
//! # Overflow: what a cell drops must never be the thing that is not green
//!
//! `CR-036`(a) settled that the component set is **data**, so a cell sized for three roles will
//! one day be handed six. A cell that simply stopped drawing at its last row would hide whichever
//! rows fell off the bottom — and §4's must-see rule exists precisely so a degraded store cannot
//! be the thing nobody sees.
//!
//! Reordering by health would fix it and break something else: [`StatusBand`]'s own reason for a
//! fixed height is that *"a screen whose furniture shifts on a status change is a screen that
//! reads as broken"*, and rows that reshuffle when a store degrades are that fault with legs. So
//! the order is the report's, and the last line **spends itself saying what is not shown** —
//! the count, and the worst state among the hidden rows. Nothing non-green becomes invisible;
//! nothing moves.

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};
use wqm_client::DaemonReport;

use crate::health::SystemHealth;
use crate::tokens::{self, Health};
use crate::widgets::chrome::{inset, Attention, ZoneHeading};
use crate::widgets::store_health::{StoreHealth, StoreRow};

/// Rows below which the cell draws nothing at all.
///
/// A heading with no body is a label, not a cell, and a grid that squeezed one to that point has
/// a layout problem the cell cannot fix by drawing half of itself. Same judgement
/// [`crate::views::service::ServiceView`] makes with its `body.height < 8` floor.
pub const MIN_ROWS: u16 = 3;

/// The two candidate cell geometries, computed from the storyboard's own 125 × 34 (VL §6) minus
/// a view's fixed chrome — tab row, its rule, the title, two rows of negative space, the bottom
/// rule and the status line, which is **7 rows** and leaves a 125 × 27 region (§16: *a view is
/// fixed chrome plus one varying region*).
///
/// **These are not a decision.** §16 records the grid as *"2 × 3"* and that phrase has two
/// readings; §18 then quotes *"roughly 16 rows by 60 columns"*, which matches neither — 16 rows
/// of a 27-row region cannot be had twice. Rather than inherit the arithmetic, both readings are
/// rendered side by side in the `Grid Geometry` frame. The law this crate has already paid for
/// three times applies: **compare a measurement to a stated value, not to another measurement.**
///
/// # And the frame's answer is that this cell cannot settle it
///
/// Rendered, the two are indistinguishable: three roles and their bindings are ~26 columns and
/// four rows, so they sit comfortably inside 41 × 13 *and* 62 × 9 with room to spare either way.
/// Width would only bite on a role name longer than any the federation has; height only on a
/// federation roughly three times its size.
///
/// So the geometry is a question for a **denser** cell — Queue or Errors, which carry lists that
/// grow with the workspace rather than with the deployment. Recorded here rather than left as an
/// impression, because §18 expected this cell to settle the grid and it does not.
pub mod geometry {
    /// The region a view's varying middle gets at the storyboard's 125 × 34.
    pub const REGION: (u16, u16) = (125, 27);

    /// Three columns of two rows — cells that are nearly square.
    pub const THREE_WIDE: (u16, u16) = (REGION.0 / 3, REGION.1 / 2);

    /// Two columns of three rows — wide, shallow cells.
    pub const TWO_WIDE: (u16, u16) = (REGION.0 / 2, REGION.1 / 3);
}

/// The Storage cell: the store federation at a glance, under the liveness master's authority.
pub struct StorageCell {
    stores: Vec<StoreRow>,
    daemon: DaemonReport,
    zone: usize,
    attention: Attention,
}

impl StorageCell {
    /// Deliberately the same arguments, in the same order, as [`StatusBand::new`] — see the
    /// module docs. `zone` is this cell's index in the Dashboard's zone order and `attention` is
    /// the screen-level answer to which zone is live (§16).
    ///
    /// [`StatusBand::new`]: crate::panes::StatusBand::new
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

    /// The one health value this cell answers from — the same type, built the same way, as the
    /// Service hub's. A cell and its drill-down disagreeing is therefore not a thing that can be
    /// arranged by feeding them the same data.
    fn system(&self) -> SystemHealth {
        SystemHealth::from_report(
            &self.daemon,
            self.stores
                .iter()
                .map(|row| crate::health::Component::new(row.role.clone(), row.health))
                .collect(),
        )
    }
}

/// The body drawn when the master is not answering: its own statement, then the roles whose
/// readings are consequently unknown.
///
/// The first line's words come from N49 (`UnreachableReason`) by way of
/// [`SystemHealth::rollup`], never from a sentence written here — the same rule
/// `views::service::frames::degraded_deck` had to learn about toasts.
fn unreachable_body(system: &SystemHealth, roles: &[String]) -> Vec<Line<'static>> {
    let rollup = system.rollup();
    let mut lines = vec![Line::from(vec![
        Span::styled(
            format!("{} ", rollup.health.glyph()),
            Style::default().fg(rollup.health.color()),
        ),
        Span::styled(rollup.label, tokens::normal_style()),
    ])];

    if !roles.is_empty() {
        lines.push(Line::from(Span::styled(
            roles.join(" · "),
            tokens::faint_style(),
        )));
    }
    lines
}

/// The line that spends itself saying what did not fit.
///
/// It carries the worst hidden state's glyph, so a degraded store that fell off the bottom is
/// still announced — the hiding is honest about *what* it hid, which is the whole reason the row
/// is worth its place.
fn overflow_line(hidden: &[StoreRow]) -> Line<'static> {
    let worst = hidden
        .iter()
        .map(|row| row.health)
        .max()
        .unwrap_or(Health::Healthy);

    let mut spans = vec![Span::styled(
        format!("+{} more", hidden.len()),
        tokens::faint_style(),
    )];
    if worst != Health::Healthy {
        spans.push(Span::styled(
            format!(" {}", worst.glyph()),
            Style::default().fg(worst.color()),
        ));
    }
    Line::from(spans)
}

impl Widget for StorageCell {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.height < MIN_ROWS {
            return;
        }

        let [heading, body] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(area);
        ZoneHeading::new("Storage", self.zone, self.attention).render(inset(heading), buf);

        let body = inset(body);
        let system = self.system();

        if system.daemon == Health::Offline {
            let roles: Vec<String> = self.stores.iter().map(|row| row.role.clone()).collect();
            Paragraph::new(unreachable_body(&system, &roles)).render(body, buf);
            return;
        }

        let room = body.height as usize;
        if self.stores.len() <= room {
            StoreHealth::new(self.stores).render(body, buf);
            return;
        }

        // One row is spent on the overflow line, so what is *shown* is one shorter than what
        // fits. `room` is at least 1 here, because MIN_ROWS reserves a heading and two body rows.
        let shown = room.saturating_sub(1);
        let mut stores = self.stores;
        let hidden = stores.split_off(shown);

        let [rows, tail] =
            Layout::vertical([Constraint::Length(shown as u16), Constraint::Length(1)]).areas(body);
        StoreHealth::new(stores).render(rows, buf);
        Paragraph::new(overflow_line(&hidden)).render(tail, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use crate::views::service::frames;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "stores",
            ty: "Vec<StoreRow>",
            description: "The backing stores — the SAME argument the Service hub's band takes",
        },
        PropInfo {
            name: "daemon",
            ty: "DaemonReport",
            description:
                "The liveness master. Offline here means the readings are unknown, not bad",
        },
        PropInfo {
            name: "attention",
            ty: "Attention",
            description:
                "Which cell the DASHBOARD says is live; the cell decides what that looks like",
        },
    ];

    /// Storage is §4.1's fourth zone, counting from Projects. A cell frame that accented the
    /// wrong index would still render, so the number is the Dashboard's rather than a local one.
    const ZONE: usize = 3;

    struct Variant(&'static str, &'static str, fn() -> StorageCell);

    impl Ingredient for Variant {
        fn tab(&self) -> &str {
            "Panes"
        }
        fn group(&self) -> &str {
            "Storage"
        }
        fn name(&self) -> &str {
            self.0
        }
        fn source(&self) -> &str {
            "wqm_tui::panes::storage"
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

    /// The two readings of *"2 × 3"*, drawn at their real sizes side by side, because that is a
    /// question the eye answers and prose has already answered twice differently (see
    /// [`super::geometry`]).
    struct GridGeometry;

    impl Ingredient for GridGeometry {
        fn tab(&self) -> &str {
            "Panes"
        }
        fn group(&self) -> &str {
            "Storage"
        }
        fn name(&self) -> &str {
            "Grid Geometry"
        }
        fn source(&self) -> &str {
            "wqm_tui::panes::storage::geometry"
        }
        fn description(&self) -> &str {
            "MEASURED: 3 columns × 2 rows (41×13) beside 2 columns × 3 rows (62×9). The two are indistinguishable here, and that IS the reading — a three-store federation is comfortable in both, so the grid must be settled by a denser cell (Queue, Errors), not by this one"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let cell = |size: (u16, u16), x: u16| Rect {
                x: area.x + x,
                y: area.y,
                width: size.0.min(area.width.saturating_sub(x)),
                height: size.1.min(area.height),
            };
            StorageCell::new(
                frames::stores(Health::Degraded),
                frames::serving(),
                ZONE,
                Attention::None,
            )
            .render(cell(geometry::THREE_WIDE, 0), buf);
            StorageCell::new(
                frames::stores(Health::Degraded),
                frames::serving(),
                ZONE,
                Attention::None,
            )
            .render(cell(geometry::TWO_WIDE, geometry::THREE_WIDE.0 + 2), buf);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Variant(
                "Nominal",
                "The federation at rest — three roles, three glyphs, and nothing else asking for the eye",
                || {
                    StorageCell::new(
                        frames::stores(Health::Healthy),
                        frames::serving(),
                        ZONE,
                        Attention::None,
                    )
                },
            )),
            Box::new(Variant(
                "Degraded store",
                "One store past its SLA: the master is answering, so the reading is trusted and shown",
                || {
                    StorageCell::new(
                        frames::stores(Health::Degraded),
                        frames::serving(),
                        ZONE,
                        Attention::None,
                    )
                },
            )),
            Box::new(Variant(
                "Daemon unreachable",
                "No daemon cell exists, so the master rule is honoured HERE: the readings are unknown, not green",
                || {
                    StorageCell::new(
                        frames::stores(Health::Healthy),
                        frames::unreachable(),
                        ZONE,
                        Attention::None,
                    )
                },
            )),
            Box::new(Variant(
                "Overflow",
                "CR-036(a)'s arriving set in a cell built for three: the last row says how many are hidden AND the worst state among them",
                || {
                    StorageCell::new(
                        frames::arriving_stores(),
                        frames::serving(),
                        ZONE,
                        Attention::None,
                    )
                },
            )),
            Box::new(Variant(
                "Focused",
                "The zone accent on the cell heading — what one cell of six looks like when it is the live one",
                || {
                    StorageCell::new(
                        frames::stores(Health::Healthy),
                        frames::serving(),
                        ZONE,
                        Attention::Zone(ZONE),
                    )
                },
            )),
            Box::new(GridGeometry),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::views::service::frames;
    use crate::widgets::chrome::test_support::Restore;

    const ZONE: usize = 3;

    fn render_at(cell: StorageCell, size: (u16, u16)) -> Buffer {
        let area = Rect::new(0, 0, size.0, size.1);
        let mut buf = Buffer::empty(area);
        cell.render(area, &mut buf);
        buf
    }

    fn row(buf: &Buffer, y: u16) -> String {
        (0..buf.area.width)
            .map(|x| buf[(x, y)].symbol())
            .collect::<String>()
            .trim_end()
            .to_string()
    }

    fn lines(buf: &Buffer) -> Vec<String> {
        (0..buf.area.height).map(|y| row(buf, y)).collect()
    }

    /// Whether any cell in the frame is painted a stated hue.
    ///
    /// The RAG is one disc in three colours since 20260906, so a claim about *state* can no
    /// longer be made against the glyph: `Healthy.glyph()` and `Offline.glyph()` are the same
    /// string. The hue is the channel that carries it, so the hue is what the assertions read.
    fn paints(buf: &Buffer, colour: ratatui::style::Color) -> bool {
        (0..buf.area.height).any(|y| {
            (0..buf.area.width)
                .any(|x| buf.cell((x, y)).expect("cell in area").style().fg == Some(colour))
        })
    }

    /// The cell and the band it drills into cannot be given different federations, because they
    /// take the same argument — and the cell's own rollup is the screen's.
    ///
    /// The failure this catches is the one SYS-3 names: a cell whose figure came from anywhere
    /// but the view it opens. Asserted as an equality between the cell's derived health and the
    /// Service hub's, both built from `frames::stores`.
    #[test]
    fn the_cell_answers_from_the_same_health_the_service_hub_does() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        for health in [Health::Healthy, Health::Degraded, Health::Offline] {
            let cell = StorageCell::new(
                frames::stores(health),
                frames::serving(),
                ZONE,
                Attention::None,
            );
            let hub = frames::hub(health);
            assert_eq!(
                cell.system().rollup(),
                hub.rollup(),
                "the cell and its drill-down disagree about {health:?}"
            );
        }
    }

    /// An unreachable daemon must not leave three green dots on the Dashboard.
    ///
    /// Two frames, because "no green dot" is vacuous against a cell that never drew one: the
    /// reachable frame proves the glyph is there to be lost.
    #[test]
    fn an_unreachable_master_withdraws_the_readings_rather_than_greening_them() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let serving_buf = render_at(
            StorageCell::new(
                frames::stores(Health::Healthy),
                frames::serving(),
                ZONE,
                Attention::None,
            ),
            geometry::THREE_WIDE,
        );
        let down_buf = render_at(
            StorageCell::new(
                frames::stores(Health::Healthy),
                frames::unreachable(),
                ZONE,
                Attention::None,
            ),
            geometry::THREE_WIDE,
        );
        let down = lines(&down_buf).join("\n");

        assert!(
            paints(&serving_buf, tokens::healthy()),
            "the healthy hue is painted when the master is answering: {:?}",
            lines(&serving_buf)
        );
        assert!(
            !paints(&down_buf, tokens::healthy()),
            "a reading nobody has must not be painted as good: {down:?}"
        );
        assert!(
            paints(&down_buf, tokens::offline()),
            "the master's own state is what the cell says instead: {down:?}"
        );
        // The roles stay named — the cell says what it is not reporting on, rather than
        // becoming a one-line cell in the state the user most needs it.
        for role in ["vector", "graph", "relational"] {
            assert!(down.contains(role), "{role} is still named: {down:?}");
        }
    }

    /// A cell too small for its rows says how many it hid **and** the worst state among them.
    ///
    /// The mutation this is written against: dropping the glyph from the overflow line. That
    /// version still reports a count, still looks tidy, and silently loses the one fact §4's
    /// must-see rule exists to protect.
    #[test]
    fn what_does_not_fit_is_counted_and_its_worst_state_is_named() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // Six components into a body with room for three: two rows shown, one spent on the tail.
        let buf = render_at(
            StorageCell::new(
                frames::arriving_stores(),
                frames::serving(),
                ZONE,
                Attention::None,
            ),
            (geometry::THREE_WIDE.0, 4),
        );
        let rendered = lines(&buf);
        let tail = rendered.last().expect("the cell drew rows").clone();

        let total = frames::arriving_stores().len();
        let shown = rendered.len() - 2; // heading, and the tail itself
        assert_eq!(
            tail.trim(),
            format!("+{} more {}", total - shown, Health::Degraded.glyph()),
            "the tail names the count and the worst hidden state: {rendered:?}"
        );
        // The disc alone no longer says WHICH state, so the hue has to be read as well —
        // without this, an overflow line that painted the worst state green would still pass.
        assert!(
            paints(&buf, tokens::degraded()),
            "the tail's mark carries the worst hidden state's hue: {rendered:?}"
        );
        // And the hidden rows really are hidden — otherwise the tail is decorative.
        assert!(
            !rendered.join("\n").contains("language_registry"),
            "a row past the cell's room must not also be drawn: {rendered:?}"
        );
    }

    /// A federation that fits draws no tail at all.
    ///
    /// Without this, an overflow line that appeared unconditionally would pass the test above
    /// and put `+0 more` on every healthy Dashboard.
    #[test]
    fn a_federation_that_fits_says_nothing_about_overflow() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render_at(
            StorageCell::new(
                frames::stores(Health::Healthy),
                frames::serving(),
                ZONE,
                Attention::None,
            ),
            geometry::THREE_WIDE,
        );
        assert!(
            !lines(&buf).join("\n").contains("more"),
            "nothing overflowed, so nothing is said about overflow"
        );
    }

    /// The heading takes the zone accent only when the Dashboard says this cell is the live one.
    ///
    /// Three frames for the same reason `status_band` renders three: with one, "not accented"
    /// has nothing to be unaccented relative to.
    #[test]
    fn the_heading_accents_only_when_this_cell_is_the_live_one() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let head = |attention| {
            row(
                &render_at(
                    StorageCell::new(
                        frames::stores(Health::Healthy),
                        frames::serving(),
                        ZONE,
                        attention,
                    ),
                    geometry::THREE_WIDE,
                ),
                0,
            )
        };

        let quiet = head(Attention::None);
        let live = head(Attention::Zone(ZONE));
        let other = head(Attention::Zone(ZONE + 1));

        assert!(quiet.contains("Storage"), "{quiet:?}");
        assert_ne!(quiet, live, "the live cell must look different");
        assert_eq!(
            quiet, other,
            "another cell being live does not accent this one"
        );
    }

    /// A cell squeezed below its floor draws nothing rather than half of itself.
    #[test]
    fn a_cell_below_its_floor_draws_nothing() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render_at(
            StorageCell::new(
                frames::stores(Health::Healthy),
                frames::serving(),
                ZONE,
                Attention::None,
            ),
            (geometry::THREE_WIDE.0, MIN_ROWS - 1),
        );
        assert!(
            lines(&buf).iter().all(|line| line.is_empty()),
            "a heading with no body is a label, not a cell: {:?}",
            lines(&buf)
        );
    }
}

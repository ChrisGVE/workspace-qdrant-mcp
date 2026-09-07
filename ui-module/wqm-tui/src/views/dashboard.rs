//! The Dashboard — tab 1, and the workspace at a glance.
//!
//! Chris, 20260907: *"split into a 2x3 table, 2 columns, 3 rows. As a starting point, we will
//! have the same layout for the dashboard as what we have today with wqm tui"*. So the six
//! cells are **v0.1's six**, in v0.1's order and with v0.1's columns:
//!
//! | | |
//! |---|---|
//! | Projects | Libraries |
//! | Scratchpad | Rules |
//! | Active Projects | Last Errors |
//!
//! The capture this was read off is kept verbatim beside this crate at
//! `design-notes/V01-DASHBOARD-CAPTURE.txt`, so the frame can be compared with the thing it
//! reproduces rather than with a memory of it.
//!
//! # This is NOT STORYBOARD §4.1's six, and that is a choice with a date on it
//!
//! §4.1 names a different set — *Projects · Queue · Libraries · Storage · Embedder · Errors* —
//! and that set is not abandoned; it is not what Chris asked for as the **starting point**
//! (20260907). Storage and Embedder have no cell here, and the Queue lives in the constant
//! status block above the grid rather than in a cell of its own. When §4.1's set is revisited,
//! three of these six survive unchanged and the difference is exactly the three named here.
//!
//! # What a row of the grid is, and what it is not
//!
//! **Equal thirds**, not sized to content. v0.1 sizes each band to its data — its Projects band
//! is nine rows and its Scratchpad band three — which means the Rules cell is in a different
//! place on every workspace. A grid whose cells move is a grid you cannot learn, so the rows
//! are equal and the cells scroll inside themselves (§18). The remainder, when the height does
//! not divide by three, goes to the **first** row: it holds the two most populated projections
//! on every workspace anyone has looked at.
//!
//! At 125 × 34 that is: 6 rows of constant top, 1 for the status line, 27 for the grid — three
//! rows of 9, 8 and 8 with the two internal rules between them, each cell 59 columns wide. A
//! cell therefore shows its heading, its column header, and six or five data rows.
//!
//! # No vertical rule between the columns
//!
//! v0.1 has none, and VL §6 leaves the side divider an explicit open micro-choice (*"a single
//! vertical rule, not a box — open micro-choice for Chris"*). So the two columns are separated
//! by whitespace and nothing else, which is also §6's own preference for dividing with space.
//! If Chris wants the rule, it is one constraint in [`grid`].

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    widgets::Widget,
};

use crate::health::Rollup;
use crate::panes::cell::CellPane;
use crate::panes::status_block::{self, StatusBlock};
use crate::tokens::Health;
use crate::widgets::chrome::{inset, Attention, StatusLine};
use crate::views::top::ConstantTop;

pub mod frames;
#[cfg(feature = "tui-pantry")]
pub mod ingredient;
#[cfg(test)]
mod tests;

/// The Dashboard's index in [`crate::widgets::tab_bar::TabBar::storyboard_tabs`] — tab 1.
pub const DASHBOARD_TAB: usize = 0;

/// Cells across, and rows down. Chris's words, and the whole shape of the screen.
pub const COLUMNS: usize = 2;
pub const GRID_ROWS: usize = 3;
pub const CELLS: usize = COLUMNS * GRID_ROWS;

/// Columns of quiet between the two cell columns. No rule — see the module docs.
const COLUMN_GAP: u16 = 3;

/// Rows the grid needs before the status block is asked to collapse: three cells of a heading,
/// a column header and one data row, plus the two rules between them.
const MIN_GRID_ROWS: u16 = GRID_ROWS as u16 * 3 + (GRID_ROWS as u16 - 1);

/// The keys that focus each cell, in the grid's own order — the hint v0.1 spells
/// `p/l/s/r/a/e`. Letters, not numbers, because the digits are already the tab jumps.
pub const FOCUS_KEYS: [char; CELLS] = ['p', 'l', 's', 'r', 'a', 'e'];

/// The Dashboard, composed.
pub struct Dashboard {
    cells: Vec<CellPane>,
    status: StatusBlock,
    /// The roll-up the bottom line shows. Derived by [`Dashboard::new`] from the same health
    /// the status block was built from, so the dot and the block cannot disagree.
    rollup: Rollup,
    attention: Attention,
    modal: bool,
}

impl Dashboard {
    /// `cells` must be [`CELLS`] long, row-major. The health is taken rather than re-derived so
    /// that the bottom dot and the top block answer from one value.
    pub fn new(cells: Vec<CellPane>, status: StatusBlock, overall: Health) -> Self {
        Self {
            cells,
            status,
            rollup: Rollup {
                health: overall,
                // The block's INTERIM rule has no `Offline`, so two words cover it. Spelled
                // here rather than reaching `health::verb`, which answers §7's question — a
                // different rule, and one that would disagree with the block above.
                label: match overall {
                    Health::Healthy => "healthy".to_string(),
                    _ => "degraded".to_string(),
                },
            },
            attention: Attention::None,
            modal: false,
        }
    }

    pub fn attention(mut self, attention: Attention) -> Self {
        self.attention = attention;
        self
    }

    /// Whether a modal owns the input. Told to the constant top **and** to every cell, because
    /// VL §6's rule is about the whole page beneath the modal: the tab bar's jump digits and
    /// the cells' key letters are the same affordance, and one going muted while the other
    /// stayed lit would say the page was half live.
    pub fn under_modal(mut self, modal: bool) -> Self {
        self.modal = modal;
        self
    }

    /// The keys the foot of the screen offers.
    ///
    /// `F Global` is in v0.1's hint line and is **not** here: [`StatusLine`] carries a roll-up,
    /// an edit-mode indicator and key hints, and nothing this crate has built makes a global
    /// filter mean anything yet. A hint for an action that does nothing is worse than a missing
    /// one — it is the storyboard depicting a screen that does not exist.
    fn hints(&self) -> Vec<(String, &'static str)> {
        vec![
            (
                FOCUS_KEYS.iter().map(|k| k.to_string()).collect::<Vec<_>>().join("/"),
                "Focus cell",
            ),
            ("Enter".to_string(), "Detail"),
            ("?".to_string(), "Help"),
            ("q".to_string(), "Quit"),
        ]
    }
}

/// The six cell rectangles, row-major, and the two rules that divide the rows.
///
/// Returned together because the rules are part of the grid rather than of any cell: a view
/// that drew them separately would be free to put them somewhere the cells are not.
pub fn grid(area: Rect) -> (Vec<Rect>, Vec<Rect>) {
    let spare = area.height.saturating_sub(GRID_ROWS as u16 - 1) % GRID_ROWS as u16;
    let each = area.height.saturating_sub(GRID_ROWS as u16 - 1) / GRID_ROWS as u16;
    let mut constraints: Vec<Constraint> = Vec::new();
    for r in 0..GRID_ROWS {
        if r > 0 {
            constraints.push(Constraint::Length(1));
        }
        // The remainder goes to the first row — the two most populated projections.
        constraints.push(Constraint::Length(each + if r == 0 { spare } else { 0 }));
    }
    let bands = Layout::vertical(constraints).split(area);

    let mut cells = Vec::with_capacity(CELLS);
    let mut rules = Vec::with_capacity(GRID_ROWS - 1);
    for (i, band) in bands.iter().enumerate() {
        if i % 2 == 1 {
            rules.push(*band);
            continue;
        }
        let [left, _, right] = Layout::horizontal([
            Constraint::Fill(1),
            Constraint::Length(COLUMN_GAP),
            Constraint::Fill(1),
        ])
        .areas(*band);
        cells.push(left);
        cells.push(right);
    }
    (cells, rules)
}

impl Widget for Dashboard {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Read before `self.cells` is consumed below — `hints` borrows the whole value.
        let hints = self.hints();
        let body = ConstantTop::new(DASHBOARD_TAB)
            .status(self.status)
            .under_modal(self.modal)
            .content_floor(MIN_GRID_ROWS + 1)
            .draw(area, buf);
        if body.height < MIN_GRID_ROWS + 1 {
            return;
        }

        // The status line is carved off the bottom before the grid is laid out, so no cell is
        // ever drawn into it and then overwritten.
        let grid_area = Rect {
            height: body.height - 1,
            ..body
        };
        let (cells, rules) = grid(inset(grid_area));
        for rule in rules {
            crate::widgets::chrome::Rule::internal().render(
                Rect {
                    x: area.x,
                    width: area.width,
                    ..rule
                },
                buf,
            );
        }
        for (zone, (pane, at)) in self.cells.into_iter().zip(cells).enumerate() {
            pane.placed(zone, self.attention)
                .hotkey(FOCUS_KEYS[zone])
                .under_modal(self.modal)
                .render(at, buf);
        }

        let mut status = StatusLine::new(self.rollup);
        for (key, label) in hints {
            status = status.hint(key, label);
        }
        status.render(
            inset(crate::views::top::row(body, body.height - 1)),
            buf,
        );
    }
}

/// The roll-up the Dashboard shows, from the same inputs its status block is built from.
///
/// Exposed so a frame cannot construct a Dashboard whose bottom dot disagrees with its top
/// block — the same discipline `views::service` applies to its own rollup.
pub fn overall(daemon: Health, entries: &[Health]) -> Health {
    status_block::rollup(daemon, entries)
}

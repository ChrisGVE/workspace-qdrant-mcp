//! One cell of the Dashboard grid: a heading, a column header, and rows that scroll inside it.
//!
//! §18's reading of the Dashboard is that **a cell is a projection** — a narrow view of
//! something a whole tab is dedicated to — and that *"every cell agrees with its drill-down"*.
//! A projection is the same shape whatever it projects, which is why there is one pane here
//! and not six: six near-identical files would be six chances for the Projects cell and the
//! Rules cell to disagree about what a column header looks like.
//!
//! The six data shapes live in [`crate::views::dashboard::frames`], each a function that maps
//! its own fixture onto a [`CellTable`]. What differs between cells is *what the columns are*,
//! and that is data.
//!
//! # The grid keeps its shape; the cell scrolls
//!
//! §18 again: cells scroll **inside** themselves. A cell that grew to fit its rows would move
//! every cell below it, and the Dashboard's whole promise is that you learn where things are.
//! So [`CellTable`] takes an `offset`, draws what fits, and spends its last line saying how
//! much it did not draw. That tail is the answer to "what does a full cell look like" — a
//! question that can only be answered by a frame, which is why the pantry has one.
//!
//! # The count in the heading is the TOTAL, not what is visible
//!
//! v0.1 writes `Projects (29)` and it means twenty-nine projects, not twenty-nine rows on
//! screen. That is kept: the heading is the size of the projection, and the overflow tail is
//! what did not fit. A heading counting only visible rows would change when the terminal was
//! resized, which makes it a fact about the window rather than about the workspace.

use ratatui::{buffer::Buffer, layout::Rect, widgets::Widget};

use crate::widgets::chrome::{Attention, FocusMark, ZoneHeading};

#[cfg(feature = "tui-pantry")]
pub mod ingredient;
pub mod sort;
pub mod table;
#[cfg(test)]
mod tests;

pub use sort::{Direction, Sort};
pub use table::{Align, Cell, CellTable, Column, EMPTY};

/// A cell of the grid: [`ZoneHeading`] over a [`CellTable`].
pub struct CellPane {
    title: String,
    /// The size of the projection, shown in parentheses as v0.1 does. `None` for a cell that
    /// counts nothing meaningful — `Last Errors` is a list, not a population.
    count: Option<usize>,
    table: CellTable,
    zone: usize,
    attention: Attention,
    /// The key that focuses this cell, accented in the heading. The **view** owns which key
    /// that is — a cell that named its own would be a second copy of the screen's key table.
    hotkey: Option<char>,
}

impl CellPane {
    /// Sort this cell's rows by one of its columns.
    ///
    /// Applied to the table straight away rather than held here and passed down at render
    /// time, so [`CellPane::table`] answers with the order the cell will draw — a guard reading
    /// the data and a reader reading the screen then look at one order rather than two.
    pub fn sorted(mut self, sort: Sort) -> Self {
        self.table = self.table.sorted(sort);
        self
    }
}

impl CellPane {
    pub fn new(title: impl Into<String>, count: Option<usize>, table: CellTable) -> Self {
        Self {
            title: title.into(),
            count,
            table,
            zone: 0,
            attention: Attention::None,
            hotkey: None,
        }
    }

    /// `zone` is this cell's index in the screen's zone order; `attention` is the screen-level
    /// answer to which zone is live — §16: *the view owns which pane has focus*.
    pub fn placed(mut self, zone: usize, attention: Attention) -> Self {
        self.zone = zone;
        self.attention = attention;
        self
    }

    /// The key that focuses this cell. Passed in rather than derived here for the same reason
    /// `placed` takes the zone: the grid's key order is the view's fact, and the view reads it
    /// from one table ([`crate::views::dashboard::FOCUS_KEYS`]).
    pub fn hotkey(mut self, key: char) -> Self {
        self.hotkey = Some(key);
        self
    }

    /// The table this cell draws, so a guard can inspect the data rather than the pixels.
    pub fn table(&self) -> &CellTable {
        &self.table
    }

    fn heading(&self) -> String {
        match self.count {
            Some(count) => format!("{} ({count})", self.title),
            None => self.title.clone(),
        }
    }
}

impl CellPane {
    /// Whether the screen says THIS cell is the live one.
    ///
    /// Asked once and spent twice — on the heading's block and on the row that takes the data
    /// cursor. Two readings of `attention` would be two chances for a cell to wear the block
    /// and put the cursor somewhere else.
    fn is_live(&self) -> bool {
        matches!(self.attention, Attention::Zone(zone) if zone == self.zone)
    }
}

impl Widget for CellPane {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }
        // [`FocusMark::Block`] is set HERE rather than passed in by the view, because a
        // `CellPane` *is* a Dashboard grid cell — there is no other kind. Letting the view
        // choose would make "a grid cell wearing the `▌` bar" a frame someone could build, and
        // Chris ruled the block for these cells specifically (2026-09-07).
        // Bold on every cell, focused or not (Chris, 2026-09-07): a tile's label sits directly
        // above the column header and the data, and without weight the three read as one block
        // of text on the default view — where no cell is focused and so nothing else carries it.
        let mut heading = ZoneHeading::new(self.heading(), self.zone, self.attention)
            .focus_mark(FocusMark::Block)
            .bold();
        if let Some(key) = self.hotkey {
            heading = heading.hotkey(key);
        }
        let cursor = self.is_live().then_some(0);
        // A cell offers its sort keys only while it is the live one AND holds more than one
        // row: a lit letter is a promise that the key does something, and there is nothing to
        // reorder in a list of one. Same shape as the foot's own three cases, and read off the
        // same two facts, so the screen cannot offer a key the foot does not.
        let sortable = self.is_live() && self.table.len() > 1;
        heading.render(crate::views::top::row(area, 0), buf);
        if area.height > 1 {
            self.table.sortable(sortable).cursor(cursor).render(
                Rect {
                    y: area.y + 1,
                    height: area.height - 1,
                    ..area
                },
                buf,
            );
        }
    }
}

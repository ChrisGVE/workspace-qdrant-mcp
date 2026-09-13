//! The **table view** inside a window — the Queue's own table, drilled into and pinned.
//!
//! The first of the two view kinds. Chris, 20:30: *"one is a table which fundamentally works
//! like the tables we have or will create, but with their own behavior … the main difference
//! is that pressing Enter on a row will drill down one step with the focus on that row"*, and
//! the `(*)` clarification: *"the view itself start from the column title to the bottom of the
//! visible table content"*.
//!
//! That region is exactly what [`crate::panes::list::ListPane`] already draws, so this is not
//! a table. It is a **pre-filter and a column policy wrapped around one**: everything about
//! drawing a table stays in `ListPane`, and what a drilled-in table adds is the floor it was
//! opened at, the column it was pinned by, and the verbs its window allows.
//!
//! # The floor is a floor, structurally
//!
//! *"the user can do further filtering but cannot display more than the record in the queue
//! corresponding to that library"*. The pin is applied BEFORE the reader's own term, always,
//! so a narrowing that would reach outside it cannot be expressed rather than merely being
//! refused — [`TableView::visible`] has no path that skips [`Pin`]. A filter that could widen
//! and was checked afterwards would be one forgotten check away from showing another tenant's
//! rows under this tenant's breadcrumb.
//!
//! # A drilled-in view drops the column it was pinned by
//!
//! Generalised from the designer's frame pair, and it is Tufte: in the undropped arm every row
//! of the table says `open-books` in its `Tenant` column — twenty columns of the window
//! carrying one repeated value, and it is the widest fixed column there. Dropping it hands the
//! width to `Object`, the column a reader is actually trying to finish reading, and nothing is
//! lost because the breadcrumb overhead already says which library this is.
//!
//! It is a policy rather than a fact about the Queue: any view pinned by a column drops that
//! column. [`TableView::show_pinned_column`] keeps the other arm reachable for the pantry.
//!
//! # The behaviours are the window's, not the table's
//!
//! *"filtering on op won't make sense for a practical table"* — so the verbs a window offers
//! are declared by the composition, in its [`crate::widgets::modal_frame::Decoration`], and
//! nothing here inherits the Queue's own key set.

use ratatui::layout::Rect;

use crate::panes::cell::{Cell, Column};
use crate::panes::list::ListPane;
use crate::widgets::modal_frame::{Scroll, SearchRow};

#[cfg(test)]
mod tests;

/// The column a drilled-in view was opened from, and the value it was opened at.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Pin {
    /// Index into the view's COLUMNS — the list's own numbering, where column 0 is the row
    /// number and carries no cell. [`Pin::cell`] converts; nothing else should do the `- 1`.
    pub column: usize,
    /// What that column must read for a row to be inside the floor.
    pub value: String,
}

impl Pin {
    pub fn new(column: usize, value: impl Into<String>) -> Self {
        Self {
            column,
            value: value.into(),
        }
    }

    /// The index of the CELL this column names.
    ///
    /// A row's cells sit one place left of the columns they fill, because the row-number column
    /// is drawn from each row's position and no row carries a cell for it. Written once, here,
    /// rather than as a `- 1` at each call site where it would read as an off-by-one.
    pub fn cell(&self) -> usize {
        self.column.saturating_sub(1)
    }

    /// Whether a row is inside the floor.
    fn holds(&self, cells: &[Cell]) -> bool {
        cells
            .get(self.cell())
            .is_some_and(|cell| cell.plain() == self.value)
    }
}

/// A table drilled into from another view: a pre-filter floor, a column policy, and a cursor.
pub struct TableView {
    columns: Vec<Column>,
    rows: Vec<Vec<Cell>>,
    pin: Option<Pin>,
    /// The reader's own narrowing, on top of the floor. `None` while nothing is typed.
    narrowed: Option<String>,
    /// Whether the pinned column is still drawn. See the module docs — off is the gate's call.
    show_pinned: bool,
    cursor: usize,
    offset: usize,
}

impl TableView {
    pub fn new(columns: Vec<Column>, rows: Vec<Vec<Cell>>) -> Self {
        Self {
            columns,
            rows,
            pin: None,
            narrowed: None,
            show_pinned: false,
            cursor: 0,
            offset: 0,
        }
    }

    /// Open this view pinned to one column's value — the floor.
    pub fn pinned(mut self, pin: Pin) -> Self {
        self.pin = Some(pin);
        self
    }

    /// The reader's own filter term, applied INSIDE the floor.
    pub fn narrowed(mut self, term: impl Into<String>) -> Self {
        self.narrowed = Some(term.into());
        self
    }

    /// Keep the pinned column on screen — the other arm of the pantry pair.
    pub fn show_pinned_column(mut self, show: bool) -> Self {
        self.show_pinned = show;
        self
    }

    pub fn cursor(mut self, at: usize) -> Self {
        self.cursor = at;
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    pub fn pin(&self) -> Option<&Pin> {
        self.pin.as_ref()
    }

    pub fn at(&self) -> usize {
        self.cursor
    }

    /// The rows this view may draw, by their index in the buffer it was given.
    ///
    /// The floor first and the reader's term second, in one expression with no branch that
    /// skips the floor — see the module docs for why that is the whole mechanism.
    pub fn visible(&self) -> Vec<usize> {
        self.rows
            .iter()
            .enumerate()
            .filter(|(_, cells)| self.pin.as_ref().is_none_or(|pin| pin.holds(cells)))
            .filter(|(_, cells)| match &self.narrowed {
                None => true,
                Some(term) if term.is_empty() => true,
                Some(term) => {
                    let needle = term.to_lowercase();
                    cells
                        .iter()
                        .any(|cell| cell.plain().to_lowercase().contains(&needle))
                }
            })
            .map(|(at, _)| at)
            .collect()
    }

    /// How many rows are on screen — what the container's scrollbar is a fraction of.
    pub fn len(&self) -> usize {
        self.visible().len()
    }

    pub fn is_empty(&self) -> bool {
        self.visible().is_empty()
    }

    /// The scroll state to hand the container.
    pub fn scroll(&self) -> Scroll {
        Scroll {
            offset: self.offset,
            total: self.len(),
        }
    }

    /// The fifth row this view mounts: the filter that made the floor visible.
    ///
    /// **The floor is drawn, not merely enforced** (the gate's arm A). The pre-filter is a
    /// limit the reader cannot cross, and the fifth row is what turns that from a surprise
    /// into a stated condition — it costs one data row, which is error prevention bought
    /// cheaply. A view with no pin mounts nothing and keeps the row for its content.
    pub fn search_row(&self) -> Option<SearchRow> {
        let pin = self.pin.as_ref()?;
        let shown = match &self.narrowed {
            Some(term) if !term.is_empty() => format!("{} {term}", pin.value),
            _ => pin.value.clone(),
        };
        Some(SearchRow::filter(shown, false))
    }

    /// Which columns survive. See the module docs: a drilled-in view drops the column it was
    /// pinned by, unless the pantry is showing the other arm.
    fn drop_column(&self) -> Option<usize> {
        match (&self.pin, self.show_pinned) {
            (Some(pin), false) => Some(pin.column),
            _ => None,
        }
    }

    /// The list this view draws — `ListPane`, unchanged, over the surviving rows and columns.
    ///
    /// Consuming, because a `ListPane` takes its rows by value and this is the one place the
    /// view's own rows are spent.
    pub fn pane(self) -> ListPane {
        let visible = self.visible();
        let drop = self.drop_column();
        // `into_iter`, not `iter`: `Column` is not `Clone` and this view's columns are spent
        // here anyway — `pane` consumes, because a `ListPane` takes its own by value.
        let columns: Vec<Column> = self
            .columns
            .into_iter()
            .enumerate()
            .filter(|(at, _)| Some(*at) != drop)
            .map(|(_, column)| column)
            .collect();
        let drop_cell = drop.map(|column| column.saturating_sub(1));
        let rows: Vec<Vec<Cell>> = visible
            .iter()
            .map(|at| {
                let mut cells = self.rows[*at].clone();
                if let Some(cell) = drop_cell
                    && cell < cells.len()
                {
                    cells.remove(cell);
                }
                cells
            })
            .collect();
        ListPane::new(columns, rows)
            .cursor(self.cursor)
            .offset(self.offset)
    }

    /// The row the cursor is on, as an index into the buffer this view was given — what a
    /// drill-down carries into the next view (*"pressing Enter on a row will drill down one
    /// step with the focus on that row"*).
    pub fn focused_row(&self) -> Option<usize> {
        self.visible().get(self.cursor).copied()
    }

    /// Where the value cell of the focused row sits on screen, for a widget that has to hang
    /// off it. `area` is the viewport the pane was drawn into.
    pub fn cursor_row_rect(&self, area: Rect) -> Rect {
        let row = self.cursor.saturating_sub(self.offset) as u16;
        Rect {
            y: area.y + 1 + row,
            height: 1,
            ..area
        }
    }
}

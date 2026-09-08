//! The Queue's columns, and the one function that turns a state into the list it draws.
//!
//! Every frame and every guard goes through [`pane`]. A guard that built its own `ListPane` from
//! the fixture would be checking one reading of the rules against another, and both would agree
//! for exactly as long as they were wrong in the same way.

use super::fixture::{QueueRow, ROWS};
use super::state::{project, QueueState};
use crate::panes::cell::{Cell, Column};
use crate::panes::list::{ListPane, LIST_PAGE};

/// Which column is which, for the guards and for the frames that sort by one.
///
/// Named rather than written as a number at each call site: the day a column is inserted, a bare
/// index sorts by a different column and nothing says so.
pub const NO: usize = 0;
pub const T: usize = 1;
pub const TENANT: usize = 2;
pub const OBJECT: usize = 3;
pub const TYPE: usize = 4;
pub const OP: usize = 5;
pub const STATUS: usize = 6;
pub const SIZE: usize = 7;
pub const AGE: usize = 8;

/// The Queue's columns, in v0.1's own order, with the sort keys this screen binds.
///
/// **NOT contract-bound (UIQ pending).** Nine columns shaped like v0.1's captured screen, which
/// is what Chris asked the tab to start from — not because any wire message names these fields.
/// Labelled rather than quietly treated as settled, so the day the contract does name them the
/// difference is a diff and not a discovery.
///
/// **`No` is untitled and offers no key, and both are the same decision.** v0.1 leads with an
/// eight-character hash; Chris replaced it with `No` (2026-09-07), the number a person can
/// actually say out loud — and the number is POSITIONAL: computed at render from each row's
/// place in the list as displayed, so a sort or a filter renumbers the visible column 1, 2, 3
/// downward. It names the position, not the row, so no number is stored on the row to survive a
/// reorder. A sort key here would invite reordering the list by its own naming scheme, and a
/// title would spend three columns saying so. `o`, freed by that decision, is the operation
/// selector's now.
///
/// **`T` offers no key.** The type selector is gone and the free-text filter covers narrowing
/// by type — a reader who wants one type types the word — and `Type` next door sorts by the
/// same fact when they want it gathered rather than removed.
///
/// Nothing else about the column set is this crate's invention.
///
/// The widths are sized to the captured data, not to the titles: `Tenant` is twenty because
/// `workspace-qdrant-mcp` is twenty, and `Status` is eleven because `in progress` is eleven.
/// Every sortable column holds its own `↓` inside its own width, so no header borrows the gap
/// beside it — which is what `views::queue::tests` checks rather than assumes.
/// The index of the CELL a column index names.
///
/// The row-number column is the list pane's own — it is drawn from each row's position and no row
/// carries a cell for it — so a row's cells sit one place left of the columns they fill. Written
/// once here rather than as a `- 1` at each call site, where it would read as an off-by-one.
pub const fn cell_at(column: usize) -> usize {
    column - 1
}

pub fn columns() -> Vec<Column> {
    vec![
        Column::number("", 3),
        Column::text("T", 1),
        Column::text("Tenant", 20).sort('e'),
        // The one column that shortens from the LEFT: the end of a path is the file, and the
        // file is what a reader is looking for. v0.1 does the same.
        Column::flex("Object").sort('b').elide_left(),
        Column::text("Type", 6).sort('t'),
        Column::text("Op", 6).sort('p'),
        Column::text("Status", 11).sort('u'),
        Column::number("Size", 8).sort('z'),
        Column::text("Age", 8).sort('a'),
    ]
}

/// One row's eight data values — the number column is drawn by the list itself, from the row's
/// position, so no cell is produced for it here.
///
/// `Size` and `Age` are [`Cell::Measured`] rather than text: both print in units and order by
/// magnitude, and a `Size` column that sorted its own strings would file `4.0 MB` between
/// `381.2 KB` and `460 B`. `Status` is [`Cell::Tinted`] because its hue is the fact — the same
/// three hues the status block gives its own counts.
fn cells(row: &QueueRow) -> Vec<Cell> {
    vec![
        Cell::Text(row.kind.letter().to_string()),
        Cell::Text(row.tenant.to_string()),
        Cell::Text(row.object.to_string()),
        Cell::Text(row.item.to_string()),
        Cell::Text(row.op.label().to_string()),
        Cell::Tinted {
            text: row.status.label().to_string(),
            hue: row.status.hue(),
        },
        Cell::Measured {
            shown: row.size.to_string(),
            order: row.bytes,
        },
        Cell::Measured {
            shown: row.age.to_string(),
            order: row.seconds,
        },
    ]
}

/// The list this state draws, over the captured buffer.
pub fn pane(state: &QueueState) -> ListPane {
    pane_over(&ROWS, state)
}

/// The same, over any buffer — so a guard can state a small one and read the rules off it.
///
/// `more` is *"the projection filled the page"*: the fixture is one page of a store that really
/// does hold eleven thousand items, so a full page means there is more behind it and a short one
/// means the filter found everything there was. The pane then applies its own half of the rule
/// ([`crate::panes::list::ListPane::shows_load_more`]) — two facts, neither sufficient alone.
pub fn pane_over(buffer: &[QueueRow], state: &QueueState) -> ListPane {
    let rows = project(buffer, state);
    let full = rows.len() == LIST_PAGE;
    let mut pane = ListPane::new(columns(), rows.iter().map(|row| cells(row)).collect())
        .more(full)
        .cursor(state.cursor)
        .relative(state.relative);
    if let Some(sort) = state.sort {
        pane = pane.sorted(sort);
    }
    pane
}

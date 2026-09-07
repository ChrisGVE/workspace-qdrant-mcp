//! The Dashboard's fixtures, and how each becomes a cell.
//!
//! Every struct here is marked NOT contract-bound. They are shaped like v0.1's captured screen
//! because that is what Chris asked the Dashboard to start from — not because any wire message
//! carries these fields. `wqm-common` names collections and the response envelope; it names
//! none of this. Binding a frame to an invented contract is how a fiction becomes load-bearing,
//! so the fiction is labelled instead, at every one of them.
//!
//! # Six mapping functions, one pane
//!
//! What differs between the Projects cell and the Rules cell is *which columns exist*, and
//! that is data. So there is one [`CellPane`] and six functions that build a [`CellTable`],
//! rather than six panes — six near-identical files would be six chances for two cells to
//! disagree about what a column header looks like, and nothing would fail when they did.
//!
//! # The numbers are the capture's own
//!
//! `.config` really does have 2'790 files and 2'635 queued in the capture. Inventing rounder
//! ones would have cost the frame the thing it is for: 29 projects into a six-row cell is what
//! produces the overflow tail, and `2'635/0/0` is what proves the queue triple's three hues
//! survive being crammed into a nine-column field.

use crate::panes::cell::{Cell, CellPane, CellTable, Column};

/// A project as the Dashboard's first cell shows it. **NOT contract-bound (UIQ pending).**
pub struct ProjectRow {
    pub name: &'static str,
    pub branches: u64,
    pub points: u64,
    pub files: u64,
    pub queue: (u64, u64, u64),
}

/// A library. **NOT contract-bound (UIQ pending).**
pub struct LibraryRow {
    pub name: &'static str,
    pub points: u64,
    pub files: u64,
    pub queue: (u64, u64, u64),
    pub sync: &'static str,
}

/// A scratchpad scope. **NOT contract-bound (UIQ pending).**
pub struct ScratchpadRow {
    pub scope: &'static str,
    pub notes: u64,
    pub queue: (u64, u64, u64),
}

/// A rules scope. **NOT contract-bound (UIQ pending).**
pub struct RuleRow {
    pub scope: &'static str,
    pub rules: u64,
    pub queue: (u64, u64, u64),
}

/// A project currently being watched. **NOT contract-bound (UIQ pending).**
pub struct ActiveProjectRow {
    pub name: &'static str,
    pub branch: &'static str,
    pub files: u64,
    pub queue: (u64, u64, u64),
}

/// One entry of the error tail. **NOT contract-bound (UIQ pending).**
pub struct ErrorRow {
    pub collection: &'static str,
    pub error: &'static str,
}

fn queue(triple: (u64, u64, u64)) -> Cell {
    Cell::Queue {
        pending: triple.0,
        in_flight: triple.1,
        failed: triple.2,
    }
}

/// Column sets are v0.1's, name for name — see the capture beside this crate.
pub fn projects(rows: &[ProjectRow], total: usize) -> CellPane {
    let table = CellTable::new(
        vec![
            Column::flex("Name"),
            Column::number("Bch", 3),
            Column::number("Pts", 3),
            Column::number("Files", 5),
            Column::number("Queue", 9),
        ],
        rows.iter()
            .map(|r| {
                vec![
                    Cell::Text(r.name.to_string()),
                    Cell::Num(r.branches),
                    Cell::Num(r.points),
                    Cell::Num(r.files),
                    queue(r.queue),
                ]
            })
            .collect(),
    );
    CellPane::new("Projects", Some(total), table)
}

pub fn libraries(rows: &[LibraryRow], total: usize) -> CellPane {
    let table = CellTable::new(
        vec![
            Column::flex("Name"),
            Column::number("Pts", 3),
            Column::number("Files", 5),
            Column::number("Queue", 7),
            Column::text("Sync", 4),
        ],
        rows.iter()
            .map(|r| {
                vec![
                    Cell::Text(r.name.to_string()),
                    Cell::Num(r.points),
                    Cell::Num(r.files),
                    queue(r.queue),
                    Cell::Text(r.sync.to_string()),
                ]
            })
            .collect(),
    );
    CellPane::new("Libraries", Some(total), table)
}

pub fn scratchpad(rows: &[ScratchpadRow], total: usize) -> CellPane {
    let table = CellTable::new(
        vec![
            Column::flex("Scope"),
            Column::number("Notes", 5),
            Column::number("Queue", 7),
        ],
        rows.iter()
            .map(|r| {
                vec![
                    Cell::Text(r.scope.to_string()),
                    Cell::Num(r.notes),
                    queue(r.queue),
                ]
            })
            .collect(),
    );
    CellPane::new("Scratchpad", Some(total), table)
}

pub fn rules(rows: &[RuleRow], total: usize) -> CellPane {
    let table = CellTable::new(
        vec![
            Column::flex("Scope"),
            Column::number("Rules", 5),
            Column::number("Queue", 7),
        ],
        rows.iter()
            .map(|r| {
                vec![
                    Cell::Text(r.scope.to_string()),
                    Cell::Num(r.rules),
                    queue(r.queue),
                ]
            })
            .collect(),
    );
    CellPane::new("Rules", Some(total), table)
}

pub fn active_projects(rows: &[ActiveProjectRow], total: usize) -> CellPane {
    let table = CellTable::new(
        vec![
            Column::flex("Name"),
            Column::text("Branch", 19),
            Column::number("Files", 5),
            Column::number("Queue", 8),
        ],
        rows.iter()
            .map(|r| {
                vec![
                    Cell::Text(r.name.to_string()),
                    Cell::Text(r.branch.to_string()),
                    Cell::Num(r.files),
                    queue(r.queue),
                ]
            })
            .collect(),
    );
    CellPane::new("Active Projects", Some(total), table)
}

/// **No count.** `Last Errors` is a tail, not a population: a number in its heading would
/// invite the reading "there are exactly three errors", which is a claim about the world that
/// a list of the most recent ones cannot make.
pub fn last_errors(rows: &[ErrorRow]) -> CellPane {
    let table = CellTable::new(
        vec![Column::text("Collection", 13), Column::flex("Error")],
        rows.iter()
            .map(|r| {
                vec![
                    Cell::Text(r.collection.to_string()),
                    Cell::Text(r.error.to_string()),
                ]
            })
            .collect(),
    );
    CellPane::new("Last Errors", None, table)
}

// --- the capture's own workspace, and two frames derived from it ---------------------

/// The seven projects v0.1 had room to draw, out of the twenty-nine it counted.
pub const PROJECTS: [ProjectRow; 7] = [
    ProjectRow { name: ".config", branches: 1, points: 0, files: 2_790, queue: (2_635, 0, 0) },
    ProjectRow { name: "ArraySwift", branches: 0, points: 0, files: 0, queue: (101, 0, 0) },
    ProjectRow { name: "claude", branches: 1, points: 0, files: 954, queue: (877, 0, 0) },
    ProjectRow { name: "de-slop", branches: 0, points: 0, files: 0, queue: (286, 0, 0) },
    ProjectRow { name: "ExtendedSwiftMath", branches: 1, points: 0, files: 118, queue: (102, 0, 0) },
    ProjectRow { name: "inkyfingers", branches: 0, points: 0, files: 0, queue: (285, 0, 0) },
    ProjectRow { name: "localdata-mcp", branches: 2, points: 0, files: 136, queue: (153, 0, 0) },
];

/// The workspace counted twenty-nine. The cell holds far fewer, which is the point.
pub const PROJECT_TOTAL: usize = 29;

pub const LIBRARIES: [LibraryRow; 1] = [LibraryRow {
    name: "programming",
    points: 0,
    files: 21,
    queue: (7, 0, 0),
    sync: "inc",
}];

pub const ACTIVE: [ActiveProjectRow; 2] = [
    ActiveProjectRow { name: "open-books", branch: "fix/s277-224-matter", files: 449, queue: (247, 4, 0) },
    ActiveProjectRow { name: "workspace-qdrant-mcp", branch: "dev", files: 93, queue: (50, 0, 0) },
];

/// Long enough to prove the Error column elides rather than clipping silently.
pub const ERRORS: [ErrorRow; 3] = [
    ErrorRow { collection: "[P] PlotSwift", error: "destination failure on success path (qdrant unreachable at QDRANT_URL)" },
    ErrorRow { collection: "[P] PlotSwift", error: "destination failure on success path (qdrant unreachable at QDRANT_URL)" },
    ErrorRow { collection: "[P] PlotSwift", error: "destination failure on success path (qdrant unreachable at QDRANT_URL)" },
];

/// The six cells of the captured workspace, row-major.
pub fn populated() -> Vec<CellPane> {
    vec![
        projects(&PROJECTS, PROJECT_TOTAL),
        libraries(&LIBRARIES, 1),
        scratchpad(&[], 0),
        rules(&[], 0),
        active_projects(&ACTIVE, 2),
        last_errors(&ERRORS),
    ]
}

/// Six empty projections — the screen a fresh install shows, and the one place `No data`
/// appears six times over.
pub fn empty() -> Vec<CellPane> {
    vec![
        projects(&[], 0),
        libraries(&[], 0),
        scratchpad(&[], 0),
        rules(&[], 0),
        active_projects(&[], 0),
        last_errors(&[]),
    ]
}

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

use crate::panes::cell::{Cell, CellPane, CellTable, Column, Direction, Sort};

/// A project as the Dashboard's first cell shows it. **NOT contract-bound (UIQ pending).**
pub struct ProjectRow {
    pub name: &'static str,
    pub branches: u64,
    pub files: u64,
    pub queue: (u64, u64, u64),
}

/// A library. **NOT contract-bound (UIQ pending).**
pub struct LibraryRow {
    pub name: &'static str,
    pub files: u64,
    pub queue: (u64, u64, u64),
    pub sync: &'static str,
}

/// One note in the scratchpad. **NOT contract-bound (UIQ pending).**
///
/// The item, then the scope it belongs to — not a scope with a count beside it (Chris,
/// 2026-09-07: *"their scope should be inverted with their respective Notes and Rules, instead
/// of showing the number of Notes and Rules"*). A cell listing `global 34` answers a question
/// nobody asked of a dashboard; a cell listing the notes answers *what is in there*.
pub struct ScratchpadRow {
    pub note: &'static str,
    pub scope: &'static str,
    pub queue: (u64, u64, u64),
}

/// One behavioural rule. **NOT contract-bound (UIQ pending).**
pub struct RuleRow {
    pub rule: &'static str,
    pub scope: &'static str,
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

/// **`Pts` is gone from this cell and from [`libraries`]** (Chris, 2026-09-07). v0.1 draws a
/// points column and it is all zeros on every workspace anyone has captured — three columns
/// spent saying nothing, on the two cells whose names are the longest and the first to elide.
/// The width goes to the flex `Name` column, which is what a reader is actually trying to
/// finish reading.
///
/// Every other column set is v0.1's, name for name — see the capture beside this crate.
pub fn projects(rows: &[ProjectRow], total: usize) -> CellPane {
    let table = CellTable::new(
        vec![
            Column::flex("Name").sort('n'),
            Column::number("Bch", 3).sort('b'),
            Column::number("Files", 5).sort('f'),
            Column::number("Queue", 9).sort('u'),
        ],
        rows.iter()
            .map(|r| {
                vec![
                    Cell::Text(r.name.to_string()),
                    Cell::Num(r.branches),
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
            Column::flex("Name").sort('n'),
            Column::number("Files", 5).sort('f'),
            Column::number("Queue", 7).sort('u'),
            // Five columns for a four-letter title: this is the LAST column of its cell, so
            // there is no gap to its right for the sort mark to borrow, and a column that
            // cannot show its own mark must not offer a key. The one column comes out of the
            // flex `Name` beside it, which has forty to spare.
            Column::text("Sync", 5).sort('y'),
        ],
        rows.iter()
            .map(|r| {
                vec![
                    Cell::Text(r.name.to_string()),
                    Cell::Num(r.files),
                    queue(r.queue),
                    Cell::Text(r.sync.to_string()),
                ]
            })
            .collect(),
    );
    CellPane::new("Libraries", Some(total), table)
}

/// How wide a scope name is drawn.
///
/// A scope is either the word `global` or a project's name, so the field is sized to the
/// project names this workspace actually has: thirteen columns holds `global` and six of the
/// captured seven, and it is the same width [`last_errors`] gives a collection. The longest
/// (`ExtendedSwiftMath`) elides, which is what a text column is allowed to do — the flex column
/// beside it is the one carrying the item, and it gets everything else.
const SCOPE_WIDTH: u16 = 13;

/// How wide a `Queue` column is on the two item cells.
///
/// The same seven columns [`libraries`] gives its own, so the two low-count cells on the grid
/// are drawn at one width rather than at two nearly-equal ones. Seven holds `102/0/0` — the
/// widest triple anything on the captured workspace carries outside the Projects cell.
const ITEM_QUEUE_WIDTH: u16 = 7;

/// **The `Queue` column is back on this cell and on [`rules`]** (Chris, 2026-09-07).
///
/// It was dropped on the reading that ingest is queued per collection rather than per item, so
/// a per-item triple would be either the collection's number repeated down the column or an
/// invention. Chris asked for the column back, so the column is back — and the fixture answers
/// the objection honestly rather than by inventing counts: every rule below carries `0/0/0`,
/// because **nothing is known to be queued for any of them**. Zeros are muted by the shared
/// `queue()` helper, so the column recedes to exactly the weight a column of no news deserves.
///
/// If a real per-item queue depth turns out not to exist behind the contract, the honest
/// outcome is a column of zeros — which is what this frame shows — and not a column of numbers
/// borrowed from somewhere else.
pub fn scratchpad(rows: &[ScratchpadRow], total: usize) -> CellPane {
    let table = CellTable::new(
        vec![
            Column::flex("Note").sort('n'),
            Column::text("Scope", SCOPE_WIDTH).sort('c'),
            Column::number("Queue", ITEM_QUEUE_WIDTH).sort('u'),
        ],
        rows.iter()
            .map(|r| {
                vec![
                    Cell::Text(r.note.to_string()),
                    Cell::Text(r.scope.to_string()),
                    queue(r.queue),
                ]
            })
            .collect(),
    );
    CellPane::new("Scratchpad", Some(total), table)
}

/// See [`scratchpad`] for what the `Queue` column on these two cells shows.
///
/// The flex column is titled **`Rule name`** rather than `Rule` (Chris, 2026-09-07). `Rule` on
/// its own reads as the row's type — the same word the cell's heading already says — where
/// what the column holds is the rule's NAME.
pub fn rules(rows: &[RuleRow], total: usize) -> CellPane {
    let table = CellTable::new(
        vec![
            Column::flex("Rule name").sort('n'),
            Column::text("Scope", SCOPE_WIDTH).sort('c'),
            Column::number("Queue", ITEM_QUEUE_WIDTH).sort('u'),
        ],
        rows.iter()
            .map(|r| {
                vec![
                    Cell::Text(r.rule.to_string()),
                    Cell::Text(r.scope.to_string()),
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
            Column::flex("Name").sort('n'),
            Column::text("Branch", 19).sort('b'),
            Column::number("Files", 5).sort('f'),
            Column::number("Queue", 8).sort('u'),
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
        vec![
            Column::text("Collection", 13).sort('c'),
            Column::flex("Error").sort('o'),
        ],
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
    ProjectRow { name: ".config", branches: 1, files: 2_790, queue: (2_635, 0, 0) },
    ProjectRow { name: "ArraySwift", branches: 0, files: 0, queue: (101, 0, 0) },
    ProjectRow { name: "claude", branches: 1, files: 954, queue: (877, 0, 0) },
    ProjectRow { name: "de-slop", branches: 0, files: 0, queue: (286, 0, 0) },
    ProjectRow { name: "ExtendedSwiftMath", branches: 1, files: 118, queue: (102, 0, 0) },
    ProjectRow { name: "inkyfingers", branches: 0, files: 0, queue: (285, 0, 0) },
    ProjectRow { name: "localdata-mcp", branches: 2, files: 136, queue: (153, 0, 0) },
];

/// The workspace counted twenty-nine. The cell holds far fewer, which is the point.
pub const PROJECT_TOTAL: usize = 29;

pub const LIBRARIES: [LibraryRow; 1] = [LibraryRow {
    name: "programming",
    files: 21,
    queue: (7, 0, 0),
    sync: "inc",
}];

pub const ACTIVE: [ActiveProjectRow; 2] = [
    ActiveProjectRow { name: "open-books", branch: "fix/s277-224-matter", files: 449, queue: (247, 4, 0) },
    ActiveProjectRow { name: "workspace-qdrant-mcp", branch: "dev", files: 93, queue: (50, 0, 0) },
];

/// The eight rules the cell has room to draw, out of the [`RULE_TOTAL`] the store holds.
///
/// **Real rules, read from this machine's own rules store**, first eight by name — the same
/// discipline the projects fixture follows. A dashboard fixture of invented rule names would
/// look plausible and teach nothing: `release-gatekeeper` is eighteen columns wide and that is
/// the fact the flex column has to survive.
///
/// **Every queue triple is `0/0/0`, and that is a measurement.** Nothing is known to be queued
/// for any rule in this store, so the column says nothing is queued. Inventing a count to make
/// the new column look busy would put a fiction on the one screen Chris judges the design from
/// — the same reason [`scratchpad`]'s fixture is empty rather than plausible.
pub const RULES: [RuleRow; 8] = [
    RuleRow { rule: "auto-file-defects", scope: "global", queue: (0, 0, 0) },
    RuleRow { rule: "collab-spirit", scope: "global", queue: (0, 0, 0) },
    RuleRow { rule: "docker-test-rm", scope: "global", queue: (0, 0, 0) },
    RuleRow { rule: "human-voice", scope: "global", queue: (0, 0, 0) },
    RuleRow { rule: "instr-supersede", scope: "global", queue: (0, 0, 0) },
    RuleRow { rule: "match-register", scope: "global", queue: (0, 0, 0) },
    RuleRow { rule: "mesh-field-log", scope: "global", queue: (0, 0, 0) },
    RuleRow { rule: "release-gatekeeper", scope: "global", queue: (0, 0, 0) },
];

/// The store holds eleven; the cell has room for five. That gap is the whole point of the
/// overflow tail, and it is why this frame carries the real number rather than the row count.
pub const RULE_TOTAL: usize = 11;

/// **Scratchpad has no fixture, and that is a measurement rather than an omission.** The
/// captured workspace had no notes and there is no real source of them on this machine to read,
/// so the cell shows `No data`. Inventing a note would put a fiction on the one screen Chris
/// judges the design from.
/// Long enough to prove the Error column elides rather than clipping silently.
pub const ERRORS: [ErrorRow; 3] = [
    ErrorRow { collection: "[P] PlotSwift", error: "destination failure on success path (qdrant unreachable at QDRANT_URL)" },
    ErrorRow { collection: "[P] PlotSwift", error: "destination failure on success path (qdrant unreachable at QDRANT_URL)" },
    ErrorRow { collection: "[P] PlotSwift", error: "destination failure on success path (qdrant unreachable at QDRANT_URL)" },
];

/// Which column of the Projects cell is `Files`, for the frame that sorts by it.
///
/// Named rather than written as `2` at the call site: the day a column is added ahead of it,
/// a bare index sorts by a different column and nothing says so.
pub const PROJECTS_FILES: usize = 2;
/// Likewise `Name` and `Queue`, for the guards that exercise the other two comparators.
pub const PROJECTS_NAME: usize = 0;
pub const PROJECTS_QUEUE: usize = 3;

/// The six cells of the captured workspace, row-major.
pub fn populated() -> Vec<CellPane> {
    vec![
        projects(&PROJECTS, PROJECT_TOTAL),
        libraries(&LIBRARIES, 1),
        scratchpad(&[], 0),
        rules(&RULES, RULE_TOTAL),
        active_projects(&ACTIVE, 2),
        last_errors(&ERRORS),
    ]
}

/// The captured workspace with its Projects cell sorted by `Files`, descending — the frame the
/// sort ruling is judged from.
///
/// `Files` on purpose: it is the narrowest column that can be sorted, so it is the one that
/// answers whether the `↓` mark can be shown at all without the table giving up a column.
pub fn sorted_by_files() -> Vec<CellPane> {
    let mut cells = populated();
    cells[0] = projects(&PROJECTS, PROJECT_TOTAL).sorted(Sort {
        column: PROJECTS_FILES,
        direction: Direction::Desc,
    });
    cells
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

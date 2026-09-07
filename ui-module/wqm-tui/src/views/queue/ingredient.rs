//! The pantry variants for [`super`].
//!
//! The status block at the top is the Dashboard's own
//! ([`crate::views::dashboard::ingredient::block`]) — the same captured workspace, so the two
//! tabs are two views of one machine rather than two machines.

use super::*;
use crate::panes::cell::{Direction, Sort};
use crate::views::dashboard::ingredient::{block, captured_queue, CAPTURED_ENTRIES};
use tui_pantry::{Ingredient, PropInfo};

const PROPS: &[PropInfo] = &[
    PropInfo {
        name: "state",
        ty: "QueueState",
        description: "Dialog, selectors, sort and cursor — everything done to the buffer",
    },
    PropInfo {
        name: "status",
        ty: "StatusBlock",
        description: "The constant top's block — the same one every other tab carries",
    },
    PropInfo {
        name: "modal",
        ty: "Option<Modal>",
        description: "The help window; the page beneath it goes quiet on its own",
    },
    PropInfo {
        name: "fixture::ROWS",
        ty: "[QueueRow; 200]",
        description:
            "NOT contract-bound (UIQ pending) — v0.1's captured screen, not a wire message",
    },
];

/// A search that has been accepted, with its counts read off the projection rather than typed.
fn searching(term: &str) -> QueueState {
    QueueState {
        dialog: Dialog::SearchInput(term.into()),
        ..QueueState::default()
    }
    .accept_search(&fixture::ROWS)
}

/// Likewise a filter.
fn filtering(term: &str) -> QueueState {
    QueueState {
        dialog: Dialog::FilterInput(term.into()),
        ..QueueState::default()
    }
    .accept_filter(&fixture::ROWS)
}

/// A Queue tab on the captured workspace, showing `state`.
pub fn queue(state: QueueState) -> Queue {
    Queue::new(state, block(CAPTURED_ENTRIES, captured_queue()))
}

pub struct Variant(
    pub &'static str,
    pub &'static str,
    pub fn() -> Queue,
    pub Option<(u16, u16)>,
);

impl Ingredient for Variant {
    fn tab(&self) -> &str {
        "Views"
    }
    fn group(&self) -> &str {
        "Queue"
    }
    fn name(&self) -> &str {
        self.0
    }
    fn source(&self) -> &str {
        "wqm_tui::views::queue"
    }
    fn description(&self) -> &str {
        self.1
    }
    fn props(&self) -> &[PropInfo] {
        PROPS
    }
    fn render(&self, area: Rect, buf: &mut Buffer) {
        let (width, height) = self.3.unwrap_or((area.width, area.height));
        (self.2)().render(
            Rect {
                width: width.min(area.width),
                height: height.min(area.height),
                ..area
            },
            buf,
        );
    }
}

/// The list itself: what the screen looks like with nothing said to it.
///
/// Split from [`dialog_frames`] because they answer different questions, not because the list
/// was long: these are about the TABLE — its rows, its order, its end, its two extreme widths —
/// and every one of them can be judged without knowing what a dialog is.
fn list_frames() -> Vec<Box<dyn Ingredient>> {
    vec![
        Box::new(Variant(
            "Populated",
            "The captured buffer: two hundred rows, the cursor on row 1, nothing narrowed",
            || queue(QueueState::default()),
            None,
        )),
        Box::new(Variant(
            "Empty",
            "A selector that matches nothing: `No data`, and a foot of two hints because every other key acts on a row",
            || {
                queue(QueueState {
                    kind: Some(Kind::Library),
                    ..QueueState::default()
                })
            },
            None,
        )),
        Box::new(Variant(
            "Sorted by Size ↓",
            "`z` lit, `↓` after the name, and 4.0 MB at the top — the column that proves a size sorts by bytes and not by its own text",
            || {
                queue(QueueState {
                    sort: Some(Sort {
                        column: frames::SIZE,
                        direction: Direction::Desc,
                    }),
                    ..QueueState::default()
                })
            },
            None,
        )),
        Box::new(Variant(
            "Buffer end",
            "The cursor on the load-more line: the list scrolled to the end, and the only line that is an offer rather than a datum",
            || {
                queue(QueueState {
                    cursor: crate::panes::list::LIST_PAGE,
                    ..QueueState::default()
                })
            },
            None,
        )),
        Box::new(Variant(
            "Under modal",
            "The page beneath a modal, with no modal on it: does everything go quiet, or does the Status column stay alight?",
            || queue(QueueState::default()).under_modal(true),
            None,
        )),
        Box::new(Variant(
            "Small 80x24",
            "Eighty by twenty-four: the flex Object column is the first thing to go, and the foot falls back to two hints",
            || queue(QueueState::default()),
            Some((80, 24)),
        )),
    ]
}

/// Every frame this view offers: the table's, then the slot's.
pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
    let mut frames = list_frames();
    frames.extend(dialog_frames());
    frames
}

/// The dialog slot's five states, the two selectors, and the window that lists every key.
fn dialog_frames() -> Vec<Box<dyn Ingredient>> {
    vec![
        Box::new(Variant(
            "Search input",
            "`/` pressed: the prompt, then the crate's own edit-in-place field running to the end of the row",
            || {
                queue(QueueState {
                    dialog: Dialog::SearchInput("reading_gui".into()),
                    ..QueueState::default()
                })
            },
            None,
        )),
        Box::new(Variant(
            "Search input, reloaded",
            "`/` pressed again while a search is on: the same field, carrying the term rather than empty",
            || queue(searching("reading_guide").open_search()),
            None,
        )),
        Box::new(Variant(
            "Search on",
            "Enter: the term, which hit of how many, and how to leave — and the cursor on the first hit, ninety rows into the list",
            || queue(searching("reading_guide")),
            None,
        )),
        Box::new(Variant(
            "Filter input",
            "`f` pressed: the same field with the other verb — the two dialogs differ in one word, which is the point",
            || {
                queue(QueueState {
                    dialog: Dialog::FilterInput("open-book".into()),
                    ..QueueState::default()
                })
            },
            None,
        )),
        Box::new(Variant(
            "Filter on",
            "The list reloaded as what matched: a row count rather than a hit, and no n/N in the foot",
            || queue(filtering("open-books")),
            None,
        )),
        Box::new(Variant(
            "Type P, status failed",
            "Two selectors at the right of the slot, cumulative: three rows left, and the foot loses Navigate",
            || {
                queue(QueueState {
                    kind: Some(Kind::Project),
                    status: Some(Status::Failed),
                    ..QueueState::default()
                })
            },
            None,
        )),
        Box::new(Variant(
            "Search on, type P",
            "A dialog and a selector on one row: does the slot read as two things, or as one long sentence?",
            || {
                queue(QueueState {
                    kind: Some(Kind::Project),
                    ..searching("reading_guide")
                })
            },
            None,
        )),
        Box::new(Variant(
            "Help",
            "Every key of this view, the four paging chords included — the only place they are ever shown",
            || queue(QueueState::default()).modal(Queue::help()),
            None,
        )),
    ]
}

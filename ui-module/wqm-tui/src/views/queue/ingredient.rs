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
];

/// A Queue tab on the captured workspace, showing `state`.
pub fn queue(state: QueueState) -> Queue {
    let (status, overall) = block(CAPTURED_ENTRIES, captured_queue());
    Queue::new(state, status, overall)
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

pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
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

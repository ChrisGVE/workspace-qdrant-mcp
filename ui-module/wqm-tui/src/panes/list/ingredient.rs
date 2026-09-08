//! The pantry variants for [`super`].
//!
//! Four frames, and the two that matter are the ones a Dashboard cell cannot produce: a cursor
//! somewhere other than the top, and the line that offers the next page. Both are answers to
//! *what does a list do when the data outruns the screen*, which is the question the whole
//! buffer-page design exists for.

use super::*;
use crate::panes::cell::{Cell, Column, Direction, Sort};
use ratatui::{buffer::Buffer, layout::Rect, widgets::Widget};
use tui_pantry::{Ingredient, PropInfo};

const PROPS: &[PropInfo] = &[
    PropInfo {
        name: "columns",
        ty: "Vec<Column>",
        description: "The cell's own vocabulary — bold here, and the sort key is always offered",
    },
    PropInfo {
        name: "cursor",
        ty: "usize",
        description: "Any line of the buffer, including the load-more line; the list scrolls to it",
    },
    PropInfo {
        name: "more",
        ty: "bool",
        description:
            "Whether the store may hold more behind this page — the one fact the pane cannot derive",
    },
];

fn columns() -> Vec<Column> {
    vec![
        Column::number("No", 4),
        Column::text("Tenant", 20).sort('e'),
        Column::flex("Object").sort('b').elide_left(),
        Column::text("Status", 11).sort('u'),
    ]
}

fn rows(n: usize) -> Vec<Vec<Cell>> {
    (0..n)
        .map(|i| {
            vec![
                Cell::Text("workspace-qdrant-mcp".into()),
                Cell::Text(format!("docs/archives/prd-workspace/audit-report-r{i}.md")),
                Cell::Tinted {
                    text: "pending".into(),
                    hue: crate::tokens::degraded,
                },
            ]
        })
        .collect()
}

struct Variant(&'static str, &'static str, fn() -> ListPane);

impl Ingredient for Variant {
    fn tab(&self) -> &str {
        "Panes"
    }
    fn group(&self) -> &str {
        "List"
    }
    fn name(&self) -> &str {
        self.0
    }
    fn source(&self) -> &str {
        "wqm_tui::panes::list"
    }
    fn description(&self) -> &str {
        self.1
    }
    fn props(&self) -> &[PropInfo] {
        PROPS
    }
    fn render(&self, area: Rect, buf: &mut Buffer) {
        // The Queue tab's own body geometry at 125 × 34: the full width less the two-column
        // margin at each edge, and the rows left once the constant top and the foot are gone.
        (self.2)().render(
            Rect {
                width: 121.min(area.width),
                height: 24.min(area.height),
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
            "A full page with the cursor on row 1 — the state a list opens in",
            || ListPane::new(columns(), rows(LIST_PAGE)).more(true),
        )),
        Box::new(Variant(
            "Buffer end",
            "The cursor on the load-more line: the only line that is an offer rather than a datum",
            || {
                ListPane::new(columns(), rows(LIST_PAGE))
                    .more(true)
                    .cursor(LIST_PAGE)
            },
        )),
        Box::new(Variant(
            "Sorted by Object ↓",
            "The mark on the sorted column, the key still lit — does the header read as sorted or as selected?",
            || {
                ListPane::new(columns(), rows(LIST_PAGE))
                    .more(true)
                    .sorted(Sort {
                        column: 2,
                        direction: Direction::Desc,
                    })
            },
        )),
        Box::new(Variant(
            "Empty",
            "No rows at all: `No data`, faint, and no cursor because there is no line to put one on",
            || ListPane::new(columns(), vec![]),
        )),
    ]
}

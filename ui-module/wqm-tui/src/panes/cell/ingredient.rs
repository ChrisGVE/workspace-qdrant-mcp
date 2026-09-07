//! The pantry variants for [`super`].
//!
//! Four frames, each answering something the Dashboard as a whole cannot: a cell judged inside
//! a six-cell grid is judged against five other things competing for the eye.

use super::*;
use tui_pantry::{Ingredient, PropInfo};

const PROPS: &[PropInfo] = &[
    PropInfo {
        name: "columns",
        ty: "Vec<Column>",
        description: "Title, alignment and width. Numbers right, words left — the only rule",
    },
    PropInfo {
        name: "rows",
        ty: "Vec<Vec<Cell>>",
        description: "Text elides with `…`; figures never do, they are sized for their widest",
    },
    PropInfo {
        name: "offset",
        ty: "usize",
        description: "Scroll INSIDE the cell — the grid never changes shape (§18)",
    },
];

fn columns() -> Vec<Column> {
    vec![
        Column::flex("Name"),
        Column::number("Bch", 3),
        Column::number("Pts", 3),
        Column::number("Files", 5),
        Column::number("Queue", 9),
    ]
}

fn rows(n: usize) -> Vec<Vec<Cell>> {
    (0..n)
        .map(|i| {
            vec![
                Cell::Text(format!("project-with-a-long-name-{i}")),
                Cell::Num(i as u64 % 3),
                Cell::Num(0),
                Cell::Num(2_790 - i as u64 * 37),
                Cell::Queue {
                    pending: 2_635 - i as u64 * 21,
                    in_flight: (i % 2) as u64,
                    failed: (i % 3) as u64,
                },
            ]
        })
        .collect()
}

struct Variant(&'static str, &'static str, fn() -> CellPane);

impl Ingredient for Variant {
    fn tab(&self) -> &str {
        "Panes"
    }
    fn group(&self) -> &str {
        "Cell"
    }
    fn name(&self) -> &str {
        self.0
    }
    fn source(&self) -> &str {
        "wqm_tui::panes::cell"
    }
    fn description(&self) -> &str {
        self.1
    }
    fn props(&self) -> &[PropInfo] {
        PROPS
    }
    fn render(&self, area: Rect, buf: &mut Buffer) {
        // The Dashboard's own cell geometry at 125x34, so a cell is judged at the size it will
        // actually be rather than at whatever the preview pane happens to offer.
        (self.2)().render(
            Rect {
                width: 59.min(area.width),
                height: 9.min(area.height),
                ..area
            },
            buf,
        );
    }
}

pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
    vec![
        Box::new(Variant(
            "Overflowing",
            "Twenty-nine rows into six: the tail spends the last line saying how much it did not draw",
            || CellPane::new("Projects", Some(29), CellTable::new(columns(), rows(29))),
        )),
        Box::new(Variant(
            "Empty",
            "No rows at all. `No data` is faint — an empty projection should read as empty, not as broken",
            || CellPane::new("Scratchpad", Some(0), CellTable::new(columns(), vec![])),
        )),
        Box::new(Variant(
            "Focused",
            "The selector block on the heading and the tint on the first row — no marker, no shift",
            || {
                CellPane::new("Projects", Some(29), CellTable::new(columns(), rows(29)))
                    .placed(0, Attention::Zone(0))
            },
        )),
        Box::new(Variant(
            "Numeric alignment",
            "Figures of every magnitude down one column: do the last digits line up, and does 0 recede?",
            || {
                CellPane::new(
                    "Projects",
                    Some(4),
                    CellTable::new(
                        columns(),
                        vec![
                            vec![Cell::Text("a".into()), Cell::Num(1), Cell::Num(0), Cell::Num(9), Cell::Queue { pending: 0, in_flight: 0, failed: 0 }],
                            vec![Cell::Text("bb".into()), Cell::Num(0), Cell::Num(0), Cell::Num(118), Cell::Queue { pending: 102, in_flight: 0, failed: 0 }],
                            vec![Cell::Text("ccc".into()), Cell::Num(2), Cell::Num(0), Cell::Num(2_790), Cell::Queue { pending: 2_635, in_flight: 4, failed: 3 }],
                            vec![Cell::Text("dddd".into()), Cell::Num(1), Cell::Num(0), Cell::Num(11_236), Cell::Queue { pending: 11_236, in_flight: 0, failed: 0 }],
                        ],
                    ),
                )
            },
        )),
    ]
}

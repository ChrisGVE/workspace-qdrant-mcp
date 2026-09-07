//! What a cell is pinned to.

use super::table::{fit, GUTTER};
use super::*;
use crate::tokens;
use crate::widgets::chrome::test_support::Restore;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::Widget;

const WIDE: u16 = 59;
const TALL: u16 = 9;

fn columns() -> Vec<Column> {
    vec![
        Column::flex("Name"),
        Column::number("Files", 5),
        Column::number("Queue", 9),
    ]
}

fn rows(n: usize) -> Vec<Vec<Cell>> {
    (0..n)
        .map(|i| {
            vec![
                Cell::Text(format!("row-{i}")),
                Cell::Num(2_790),
                Cell::Queue {
                    pending: 2_635,
                    in_flight: 0,
                    failed: 0,
                },
            ]
        })
        .collect()
}

fn render(pane: CellPane, width: u16, height: u16) -> Buffer {
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    pane.render(area, &mut buf);
    buf
}

fn line(buf: &Buffer, y: u16) -> String {
    (0..buf.area.width)
        .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
        .collect::<String>()
        .trim_end()
        .to_string()
}

/// One row of the table's BODY — the marker column stripped off the front.
///
/// Every row is indented by [`GUTTER`], cursor or not, so a test asserting what a row says has
/// to look past the gutter. Stripped by width rather than by trimming: the cursor row's marker
/// is `▸ ` and trimming would silently drop it too, which is the one thing a cursor test needs
/// to see.
fn body(buf: &Buffer, y: u16) -> String {
    line(buf, y).chars().skip(GUTTER as usize).collect()
}

/// §18: a cell scrolls inside itself and the grid keeps its shape, so a cell that cannot show
/// everything must say how much it hid. The two halves are asserted together — a tail reading
/// `… 22 more` above a body that drew one row more than it counted is a lie nothing catches.
#[test]
fn an_overflowing_cell_spends_its_last_line_saying_how_much_it_hid() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let table = CellTable::new(columns(), rows(29));
    let (shown, hidden) = table.budget(TALL - 1);
    assert_eq!(shown + hidden, 29, "every row is either drawn or counted");
    assert!(hidden > 0, "29 rows into this cell must overflow");

    let buf = render(CellPane::new("Projects", Some(29), table), WIDE, TALL);
    assert_eq!(body(&buf, TALL - 1), format!("… {hidden} more"));
    // The last DRAWN row is the one the budget says it is, and no further.
    assert!(body(&buf, TALL - 2).starts_with(&format!("row-{}", shown - 1)));
}

/// A projection with nothing in it must read as empty rather than as broken — v0.1's own word.
#[test]
fn an_empty_projection_says_no_data() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(
        CellPane::new("Scratchpad", Some(0), CellTable::new(columns(), vec![])),
        WIDE,
        TALL,
    );
    assert!(line(&buf, 0).contains("Scratchpad (0)"));
    assert!(line(&buf, 1).contains("Name"), "the header survives an empty table");
    assert_eq!(body(&buf, 2), EMPTY);
}

/// Text elides; figures never do. An elided name is still recognisable and the `…` says so; an
/// elided number is a different number, silently.
#[test]
fn text_elides_with_an_ellipsis_and_never_grows_the_column() {
    assert_eq!(fit("short", 10), "short");
    assert_eq!(fit("exactlyten", 10), "exactlyten");
    assert_eq!(fit("elevenchars", 10), "elevencha…");
    assert_eq!(
        fit("elevenchars", 10).chars().count(),
        10,
        "the ellipsis takes one of the column's own cells, never an extra"
    );

    // And the renderer actually calls it. Testing `fit` alone passes just as happily against
    // a table that never elides anything — measured: removing the call left this green.
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();
    let table = CellTable::new(
        vec![Column::text("Name", 6), Column::number("Files", 5)],
        vec![vec![Cell::Text("elevenchars".into()), Cell::Num(1)]],
    );
    let buf = render(CellPane::new("Projects", Some(1), table), WIDE, TALL);
    assert!(
        body(&buf, 2).starts_with("eleve…"),
        "the table must elide its text, not merely be able to: {:?}",
        body(&buf, 2)
    );
}

/// A figure too wide for its column is replaced, never clipped — a clipped number is a wrong
/// number with nothing to mark it.
#[test]
fn a_figure_that_does_not_fit_is_replaced_rather_than_cut() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let narrow = vec![Column::flex("Name"), Column::number("Files", 5)];
    let table = CellTable::new(
        narrow,
        vec![
            vec![Cell::Text("fits".into()), Cell::Num(2_790)],
            vec![Cell::Text("does not".into()), Cell::Num(11_236)],
        ],
    );
    let buf = render(CellPane::new("Projects", Some(2), table), WIDE, TALL);
    assert!(line(&buf, 2).ends_with("2'790"), "{:?}", line(&buf, 2));
    assert!(
        line(&buf, 3).ends_with('…'),
        "a six-character figure in a five-cell column must not render as five of its \
         characters: {:?}",
        line(&buf, 3)
    );
    assert!(!line(&buf, 3).contains("11'23"), "{:?}", line(&buf, 3));
}

/// The geometry `columns()` produces at [`WIDE`], stated rather than measured.
///
/// `Fill(1)` + `Length(5)` + `Length(9)` with one column of spacing between each: the flex
/// column takes what is left, so Files ends at 48 and Queue at 58. Every assertion below reads
/// a cell by this arithmetic instead of searching for digits — `find("4")` lands inside `247`,
/// which is a different span with a different hue, and the test would measure the wrong cell
/// while looking perfectly correct.
const FILES_RIGHT: u16 = WIDE - 9 - 1 - 1;
const QUEUE_RIGHT: u16 = WIDE - 1;

/// Numbers right, words left — the only alignment rule, and the one that lets a column of
/// figures be compared down its last digit.
#[test]
fn figures_are_flush_right_and_words_flush_left() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let table = CellTable::new(
        columns(),
        vec![
            vec![Cell::Text("a".into()), Cell::Num(9), Cell::Queue { pending: 1, in_flight: 0, failed: 0 }],
            vec![Cell::Text("bb".into()), Cell::Num(2_790), Cell::Queue { pending: 2_635, in_flight: 0, failed: 0 }],
        ],
    );
    let buf = render(CellPane::new("Projects", Some(2), table), WIDE, TALL);

    assert!(body(&buf, 2).starts_with("a "), "a word starts at its column");
    assert!(body(&buf, 3).starts_with("bb "));

    // Both figures end on the SAME cell whatever their width — the whole point of right
    // alignment, and the thing a left-aligned column would silently lose.
    let at = |y: u16, x: u16| buf.cell((x, y)).expect("cell in area").symbol().to_string();
    assert_eq!(at(2, FILES_RIGHT), "9");
    assert_eq!(at(3, FILES_RIGHT), "0", "2'790 ends on the same cell 9 does");
    assert_eq!(at(2, QUEUE_RIGHT), "0", "the queue triple is right-aligned too");
}

/// The queue triple is three facts, so it is three hues — and a zero recedes in all three.
#[test]
fn a_queue_triple_carries_three_hues_and_mutes_its_zeros() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let table = CellTable::new(
        columns(),
        vec![vec![
            Cell::Text("a".into()),
            Cell::Num(0),
            Cell::Queue { pending: 247, in_flight: 4, failed: 0 },
        ]],
    );
    let buf = render(CellPane::new("Projects", Some(1), table), WIDE, TALL);
    let fg = |x: u16| buf.cell((x, 2)).expect("cell in area").style().fg;

    // `247/4/0` is seven cells ending at QUEUE_RIGHT, so each figure's position is arithmetic.
    let start = QUEUE_RIGHT - 6;
    assert_eq!(fg(start), Some(tokens::degraded()), "waiting work is the warning hue");
    assert_eq!(fg(start + 4), Some(tokens::in_flight()), "work moving is the in-flight role");
    assert_eq!(fg(QUEUE_RIGHT), Some(tokens::muted()), "a failure count of zero is not news");
    assert_eq!(
        fg(FILES_RIGHT),
        Some(tokens::normal()),
        "the zero rule is the queue triple's alone — a plain figure column keeps its zeros, \
         or v0.1's all-zero `Pts` column would vanish rather than recede"
    );
    // The separators are structure, not data.
    assert_eq!(fg(start + 3), Some(tokens::muted()));
}

/// The heading counts the PROJECTION, not what fitted — see the module docs.
#[test]
fn the_heading_counts_the_projection_and_not_the_visible_rows() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    // SEVEN rows in hand, TWENTY-NINE in the workspace — v0.1's own situation, and the only
    // fixture in which "counts the projection" and "counts the rows" differ. With 29 rows the
    // two readings coincide and the guard asserts nothing.
    let buf = render(
        CellPane::new("Projects", Some(29), CellTable::new(columns(), rows(7))),
        WIDE,
        TALL,
    );
    assert!(line(&buf, 0).contains("Projects (29)"), "{:?}", line(&buf, 0));
    assert!(
        !line(&buf, 0).contains("(7)"),
        "the heading must not count what it happens to hold: {:?}",
        line(&buf, 0)
    );

    // The same projection in a shorter cell still says 29: the heading is a fact about the
    // workspace, and resizing a terminal does not change how many projects exist.
    let short = render(
        CellPane::new("Projects", Some(29), CellTable::new(columns(), rows(29))),
        WIDE,
        5,
    );
    assert!(line(&short, 0).contains("Projects (29)"));
}

/// A cell with no count renders no parentheses — `Last Errors` is a tail, not a population.
#[test]
fn a_cell_without_a_count_shows_no_parentheses() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(
        CellPane::new("Last Errors", None, CellTable::new(columns(), rows(2))),
        WIDE,
        TALL,
    );
    assert_eq!(line(&buf, 0), "Last Errors");
}

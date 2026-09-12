//! What a cell is pinned to.

use super::value::fit;
use super::*;
use crate::tokens;
use crate::widgets::chrome::test_support::Restore;
use crate::widgets::chrome::Attention;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
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

/// One row of the table's body.
///
/// The marker gutter is gone (Chris, 2026-09-07), so a row starts at the cell's own first
/// column and this is [`line`] under another name. Kept as its own function rather than
/// inlined at each call: what these assertions are about is *what the table said*, and the day
/// a lead-in column comes back it comes back here rather than in nine places.
fn body(buf: &Buffer, y: u16) -> String {
    line(buf, y)
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

/// Figure edges follow the same fitted geometry the renderer uses.
fn right_edges(table: &CellTable) -> (u16, u16) {
    let fitted = super::fit::fit(table.columns(), table.rows(), WIDE, 12, 0);
    let rects = super::fit::laid_out(Rect::new(0, 0, WIDE, 1), &fitted);
    (rects[1].right() - 1, rects[2].right() - 1)
}

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
    let (files_right, queue_right) = right_edges(&table);
    let buf = render(CellPane::new("Projects", Some(2), table), WIDE, TALL);

    assert!(body(&buf, 2).starts_with("a "), "a word starts at its column");
    assert!(body(&buf, 3).starts_with("bb "));

    // Both figures end on the SAME cell whatever their width — the whole point of right
    // alignment, and the thing a left-aligned column would silently lose.
    let at = |y: u16, x: u16| buf.cell((x, y)).expect("cell in area").symbol().to_string();
    assert_eq!(at(2, files_right), "9");
    assert_eq!(at(3, files_right), "0", "2'790 ends on the same cell 9 does");
    assert_eq!(at(2, queue_right), "0", "the queue triple is right-aligned too");
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
    let (files_right, queue_right) = right_edges(&table);
    let buf = render(CellPane::new("Projects", Some(1), table), WIDE, TALL);
    let fg = |x: u16| buf.cell((x, 2)).expect("cell in area").style().fg;

    // `247/4/0` is seven cells ending at the Queue rect's right edge.
    let start = queue_right - 6;
    assert_eq!(fg(start), Some(tokens::degraded()), "waiting work is the warning hue");
    assert_eq!(fg(start + 4), Some(tokens::in_flight()), "work moving is the in-flight role");
    assert_eq!(fg(queue_right), Some(tokens::muted()), "a failure count of zero is not news");
    assert_eq!(
        fg(files_right),
        Some(tokens::table_row()),
        "the zero rule is the queue triple's alone — a plain figure column keeps its zeros at \
         the row rung, or v0.1's all-zero `Pts` column would vanish rather than recede"
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

/// The two values the Queue tab added, each pinned to the thing that would break it.
///
/// A [`Cell::Measured`] prints one thing and orders by another, so the failure to catch is a
/// comparator that read the printed form: `4.0 MB` sorts under `9` as text and above
/// `903.5 KB` as a size. A [`Cell::Tinted`] carries a hue that is a FACT rather than a rank, so
/// the failure to catch is a comparator that ordered by it — the three queue states would then
/// sort in whatever order the palette happens to list them.
#[test]
fn a_measured_value_orders_by_its_magnitude_and_a_tinted_one_by_its_word() {
    fn measured(shown: &str, order: u64) -> Cell {
        Cell::Measured {
            shown: shown.into(),
            order,
        }
    }

    let mb = measured("4.0 MB", 4_194_304);
    let kb = measured("903.5 KB", 925_184);
    assert_eq!(
        super::sort::compare(&mb, &kb),
        std::cmp::Ordering::Greater,
        "read as text, `4.0 MB` files under `9` and this is Less"
    );

    // A blank size — v0.1 leaves it empty on a delete — orders as nothing at all, not as a
    // string that sorts before every digit.
    assert_eq!(
        super::sort::compare(&measured("", 0), &kb),
        std::cmp::Ordering::Less
    );

    let pending = Cell::Tinted {
        text: "pending".into(),
        hue: crate::tokens::degraded,
    };
    let failed = Cell::Tinted {
        text: "failed".into(),
        hue: crate::tokens::offline,
    };
    assert_eq!(
        super::sort::compare(&pending, &failed),
        std::cmp::Ordering::Greater,
        "`failed` before `pending`, alphabetically — the hue is a state, not a rank"
    );

    // And both are figures for the purpose of eliding: a shortened `4.0 M` is a unitless
    // number, so a column too narrow for one shows `…` rather than a plausible wrong value.
    assert!(mb.is_figure());
    assert_eq!(
        mb.spans(5, crate::panes::cell::Elide::Right, false)[0].content,
        "…"
    );
    assert!(!pending.is_figure(), "a state word elides like any other word");
}

/// The fixed columns drop in priority order — lowest first, the flex column never — until the
/// flex column reaches its floor. A synthetic table, so the order is read from the data rather
/// than from any particular cell's fixture.
#[test]
fn a_narrow_table_drops_fixed_columns_in_priority_order_and_never_the_flex() {
    let table = CellTable::new(
        vec![
            Column::flex("Name"),
            Column::text("Branch", 10).priority(2),
            Column::number("Files", 5).priority(1),
            Column::number("Queue", 8).priority(0),
        ],
        vec![],
    );

    // 47 columns leaves the flex 21, above its floor of 12: nothing drops.
    let fitted = |width| super::fit::fit(table.columns(), table.rows(), width, 12, 0).active;
    assert_eq!(fitted(47), vec![0, 1, 2, 3]);

    // 30 leaves it 4, so the queue triple (priority 0) goes first.
    assert_eq!(fitted(30), vec![0, 1, 2], "the lowest-priority column drops first");

    // At 25 the Branch text truncates to its title, preserving Files and Name's floor.
    assert_eq!(fitted(25), vec![0, 1, 2], "text truncates before a figure drops");

    // At 20 the figure drops, while Branch still fits at its title-width floor.
    assert_eq!(fitted(20), vec![0, 1], "then the figure drops");
    assert_eq!(fitted(17), vec![0], "the flex column is never dropped");
}

/// On a priority tie the RIGHTMOST column drops first — the one furthest from the flex identity.
#[test]
fn a_priority_tie_drops_the_rightmost_column_first() {
    let table = CellTable::new(
        vec![
            Column::flex("Name"),
            Column::number("A", 5).priority(1),
            Column::number("B", 5).priority(1),
        ],
        vec![],
    );

    // 20 columns: fixed 5 + 5 and one gap is 11, leaving the flex 9 — below the floor, so one
    // of the two tied columns must go, and it is `B`, the rightmost.
    assert_eq!(
        super::fit::fit(table.columns(), table.rows(), 20, 12, 0).active,
        vec![0, 1],
        "the rightmost of the tie drops first"
    );
}

#[test]
fn a_sorted_header_uses_a_space_only_when_the_whole_gap_is_affordable() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();
    for (width, expected) in [(9, "Files↓"), (10, "Files ↓")] {
        let table = CellTable::new(
            vec![Column::text("Files", 5), Column::number("N", 3)],
            vec![vec![Cell::Text("entry".into()), Cell::Num(1)]],
        )
        .sorted(Sort { column: 0, direction: Direction::Desc });
        let area = Rect::new(0, 0, width, 2);
        let mut buf = Buffer::empty(area);
        table.render(area, &mut buf);
        let header = line(&buf, 0);
        assert!(header.contains(expected), "{width}: {header:?}");
        assert!(header.contains('N'), "the mark clipped its neighbour: {header:?}");
    }
}

mod header;

//! What a list is pinned to: its italic header, its free cursor, and the line that offers a page.

use super::*;
use crate::panes::cell::{Align, Cell, Column, Direction, Sort};
use crate::tokens;
use crate::widgets::chrome::test_support::Restore;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::widgets::Widget;

const WIDE: u16 = 125;

fn columns() -> Vec<Column> {
    vec![
        // The first column is the row-number column, which the pane draws itself from each
        // row's position — untitled, unsorted, and carrying no cell.
        Column::number("", 4),
        Column::flex("Object").sort('b').elide_left(),
        Column::text("Status", 11).sort('u'),
    ]
}

/// `n` rows. No number cell: the pane numbers the rows itself, from their positions.
fn rows(n: usize) -> Vec<Vec<Cell>> {
    (0..n)
        .map(|i| {
            vec![
                Cell::Text(format!("a/very/long/path/that/will/not/fit/file-{i}.txt")),
                Cell::Text("pending".into()),
            ]
        })
        .collect()
}

fn render(pane: ListPane, width: u16, height: u16) -> Buffer {
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    pane.render(area, &mut buf);
    buf
}

fn text(buf: &Buffer, y: u16) -> String {
    (0..buf.area.width)
        .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
        .collect::<String>()
        .trim_end()
        .to_string()
}

/// The header is italic at every rung, and the sort letter is the only thing on it wearing the
/// accent hue — offered only while there is something to reorder.
///
/// Both halves in one guard because they are one sentence: the header says *this is structure*
/// with slant, and *this letter sorts it* with the accent hue and weight. A frame that lost
/// either would still look like a header.
#[test]
fn the_header_is_italic_and_lights_its_sort_key_only_when_there_is_more_than_one_row() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let many = render(ListPane::new(columns(), rows(5)), WIDE, 8);
    let lit: Vec<String> = (0..WIDE)
        .filter(|x| {
            many.cell((*x, 0)).expect("cell in area").style().fg == Some(tokens::accent())
        })
        .map(|x| {
            many.cell((x, 0))
                .expect("cell in area")
                .symbol()
                .to_string()
        })
        .collect();
    assert_eq!(
        lit,
        vec!["b", "u"],
        "the lit letters are the two sortable columns' keys, left to right — the number column
         offers none, because a positional number is nothing that can be ordered"
    );

    // Every painted cell of the header row is italic, and bold is the sort key's alone — weight
    // is what marks the key out from the slanted label around it.
    for x in 0..WIDE {
        let cell = many.cell((x, 0)).expect("cell in area");
        if cell.symbol().trim().is_empty() {
            continue;
        }
        assert!(
            cell.style().add_modifier.contains(Modifier::ITALIC),
            "column {x} ({:?}) is not italic",
            cell.symbol()
        );
        let is_key = cell.style().fg == Some(tokens::accent());
        assert_eq!(
            cell.style().add_modifier.contains(Modifier::BOLD),
            is_key,
            "column {x} ({:?}): bold is the sort key's alone",
            cell.symbol()
        );
    }

    // One row: nothing to reorder, so nothing is offered.
    let one = render(ListPane::new(columns(), rows(1)), WIDE, 8);
    let lit_on_one = (0..WIDE)
        .filter(|x| one.cell((*x, 0)).expect("cell in area").style().fg == Some(tokens::accent()))
        .count();
    assert_eq!(lit_on_one, 0, "a list of one offers no sort key");
}

/// A header span is italic and NOT bold; its sort key is the accent hue AND bold, on top of the
/// same italic.
///
/// The header marks itself as structure with slant, never with weight (Chris, 2026-09-07) — a
/// bold header reads as a heading rather than as a label under it. The one exception is the key
/// letter, which adds weight on top of the slant so a single letter of an italic word reads as
/// the key to press rather than as a gap in the label.
#[test]
fn a_header_span_is_italic_not_bold_while_its_sort_key_is_accent_and_bold() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(
        ListPane::new(vec![Column::flex("Object").sort('b')], rows(2)),
        WIDE,
        8,
    );

    // `Object`: the `b` is the sort key, so the leading `O` is header text and the `b` is the key.
    let rest = buf.cell((0, 0)).expect("cell in area").style();
    assert_eq!(
        rest.fg,
        Some(tokens::header()),
        "the header text sits at the text rung"
    );
    assert!(rest.add_modifier.contains(Modifier::ITALIC), "the header is italic");
    assert!(!rest.add_modifier.contains(Modifier::BOLD), "the header is not bold");

    let key = buf.cell((1, 0)).expect("cell in area").style();
    assert_eq!(key.fg, Some(tokens::accent()), "the sort key is the accent hue");
    assert!(key.add_modifier.contains(Modifier::BOLD), "the sort key is bold");
    assert!(
        key.add_modifier.contains(Modifier::ITALIC),
        "the sort key keeps the header's italic"
    );
}

/// The cursor sits on ANY row, and the list scrolls the minimum distance to keep it visible.
///
/// This is the whole of what a list has that a cell does not — a cell's cursor is always its
/// first drawn row. Read off the tint rather than off the field, because the field is what the
/// caller said and the tint is what the reader sees.
#[test]
fn the_cursor_sits_on_any_row_and_the_list_scrolls_the_minimum_to_keep_it_visible() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    // Four body rows for twenty rows of data.
    const HEIGHT: u16 = 5;
    let tinted = |buf: &Buffer| -> Vec<u16> {
        (0..HEIGHT)
            .filter(|y| {
                buf.cell((0, *y)).expect("cell in area").style().bg == Some(tokens::cursor_bg())
            })
            .collect()
    };

    // Third row, no scroll needed: the cursor is where it was put and the list has not moved.
    let near = render(ListPane::new(columns(), rows(20)).cursor(2), WIDE, HEIGHT);
    assert_eq!(tinted(&near), vec![3], "row 2 is the third body line");
    assert!(text(&near, 1).starts_with("   1"), "{:?}", text(&near, 1));

    // Row 10, which is past the window: the list scrolls just far enough to show it, so the
    // cursor lands on the LAST body line rather than in the middle.
    let far = render(ListPane::new(columns(), rows(20)).cursor(10), WIDE, HEIGHT);
    assert_eq!(
        tinted(&far),
        vec![HEIGHT - 1],
        "the cursor is on the last body line"
    );
    assert!(
        text(&far, HEIGHT - 1).starts_with("  11"),
        "the eleventh row carries No 11: {:?}",
        text(&far, HEIGHT - 1)
    );
    assert!(
        text(&far, 1).starts_with("   8"),
        "the window moved by exactly the seven rows it had to: {:?}",
        text(&far, 1)
    );
}

/// A list has no `… N more` tail: rows past the height are reached by scrolling, not counted.
///
/// The cell's tail exists because a cell cannot scroll to where the grid is not. A list can, so
/// a tail would be spending the screen's last line on a number the scrollbar-shaped act of
/// pressing `j` already answers.
#[test]
fn a_list_spends_no_line_on_saying_what_it_did_not_draw() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(ListPane::new(columns(), rows(200)), WIDE, 6);
    for y in 0..6 {
        assert!(
            !text(&buf, y).contains("more"),
            "line {y} carries an overflow tail: {:?}",
            text(&buf, y)
        );
    }
    // Every body line is a row of data — five of them, and the fifth is row five.
    assert!(text(&buf, 5).starts_with("   5"), "{:?}", text(&buf, 5));
}

/// The load-more line appears on a full page the caller says may have more, and nowhere else.
///
/// Four cases because the rule is a conjunction and a conjunction fails three ways.
#[test]
fn the_load_more_line_needs_a_full_page_and_a_caller_who_says_there_is_more() {
    let full = ListPane::new(columns(), rows(LIST_PAGE)).more(true);
    assert!(
        full.shows_load_more(),
        "a full page with more behind it offers the next"
    );

    let no_more = ListPane::new(columns(), rows(LIST_PAGE));
    assert!(
        !no_more.shows_load_more(),
        "a full page is not evidence of more: a store of exactly {LIST_PAGE} fills one exactly"
    );

    let short = ListPane::new(columns(), rows(LIST_PAGE - 1)).more(true);
    assert!(
        !short.shows_load_more(),
        "a partial page is the end of the store, whatever the caller believes"
    );

    let empty = ListPane::new(columns(), vec![]).more(true);
    assert!(
        !empty.shows_load_more(),
        "nothing loaded, nothing to load more of"
    );
}

/// The line reads its two numbers off the buffer and off [`LIST_PAGE`], and takes the cursor.
#[test]
fn the_load_more_line_reads_its_numbers_and_takes_the_cursor_like_a_row() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let pane = ListPane::new(columns(), rows(LIST_PAGE)).more(true);
    assert_eq!(
        pane.load_more_text(),
        "200 rows, press Enter to load 200 more rows",
        "both numbers are read, not written"
    );

    // Two pages in, and the two numbers part company — which is the only shape that tells a
    // read sentence from a written one, since at one page they happen to be the same figure.
    let second = ListPane::new(columns(), rows(LIST_PAGE * 2)).more(true);
    assert_eq!(
        second.load_more_text(),
        "400 rows, press Enter to load 200 more rows",
        "the first number is the buffer's, the second is the page size's"
    );

    // The cursor is on the load-more line, which is the line after the last row.
    const HEIGHT: u16 = 5;
    let buf = render(
        ListPane::new(columns(), rows(LIST_PAGE))
            .more(true)
            .cursor(LIST_PAGE),
        WIDE,
        HEIGHT,
    );
    let last = text(&buf, HEIGHT - 1);
    assert_eq!(
        last, "200 rows, press Enter to load 200 more rows",
        "{last:?}"
    );
    assert_eq!(
        buf.cell((0, HEIGHT - 1)).expect("cell in area").style().bg,
        Some(tokens::cursor_bg()),
        "the offer takes the cursor tint like any other line"
    );
    // And the row above it is the last real row, so the offer is the LAST line, not a footer.
    assert!(
        text(&buf, HEIGHT - 2).starts_with(" 200"),
        "{:?}",
        text(&buf, HEIGHT - 2)
    );
}

/// The numbers are POSITIONS, so a sort renumbers them 1, 2, 3 down the screen while the rows
/// move underneath.
///
/// This is the reversal of what this guard used to assert (Chris, 2026-09-08). The number used to
/// be assigned at load and to travel with its row; it is now the row's place in what is
/// displayed, computed at render. It is read off the drawn frame rather than off a cell, because
/// no row carries it any more.
#[test]
fn a_sort_renumbers_the_rows_by_their_new_positions() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let pane = ListPane::new(columns(), rows(5)).sorted(Sort {
        column: 1,
        direction: Direction::Desc,
    });
    let buf = render(pane, WIDE, 7);

    let drawn: Vec<String> = (1..=5)
        .map(|row| text(&buf, row).split_whitespace().next().unwrap_or("").to_string())
        .collect();
    assert_eq!(
        drawn,
        vec!["1", "2", "3", "4", "5"],
        "the column counts the screen, not the load order"
    );
    // Object descending: file-4 leads. The ROWS moved; the numbers did not follow them.
    assert!(
        text(&buf, 1).contains("file-4.txt"),
        "the sort did not take: {:?}",
        text(&buf, 1)
    );
}

/// The `Object` column drops its HEAD, so the filename survives; every other column drops its
/// tail.
#[test]
fn a_path_column_elides_from_the_left_and_the_rest_from_the_right() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    // Narrow enough that the flex column cannot hold the whole path.
    let buf = render(ListPane::new(columns(), rows(1)), 40, 3);
    let row = text(&buf, 1);
    assert!(row.contains("…"), "the path did not elide at all: {row:?}");
    assert!(
        row.contains("file-0.txt"),
        "the filename is what a reader is looking for: {row:?}"
    );
    assert!(
        !row.contains("a/very"),
        "an elision from the right would have kept the head: {row:?}"
    );
}

/// An empty list says v0.1's own words rather than nothing at all.
#[test]
fn an_empty_list_reads_as_empty_rather_than_as_broken() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(ListPane::new(columns(), vec![]), WIDE, 6);
    assert_eq!(text(&buf, 1), crate::panes::cell::EMPTY);
    assert_ne!(
        buf.cell((0, 1)).expect("cell in area").style().bg,
        Some(tokens::cursor_bg()),
        "there is no line for a cursor to sit on"
    );
}

/// A column's alignment survives into the list, so figures can be compared down their last
/// digit. Read off the drawn row rather than off the `Column`, which would be the declaration
/// agreeing with itself.
#[test]
fn a_figure_column_is_flush_right_in_a_list_as_it_is_in_a_cell() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    assert_eq!(columns()[0].align, Align::Right, "No is a figure column");
    let buf = render(ListPane::new(columns(), rows(12)), WIDE, 14);
    // `9` and `12` are three characters apart in width and must end in the same column.
    let ninth = text(&buf, 9);
    let twelfth = text(&buf, 12);
    assert_eq!(&ninth[..4], "   9", "{ninth:?}");
    assert_eq!(&twelfth[..4], "  12", "{twelfth:?}");
}

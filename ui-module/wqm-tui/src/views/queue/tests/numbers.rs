//! The row-number column — a POSITION, and the relative mode that reads distances from it.
//!
//! Chris, 2026-09-08: *"the raw number is always fix with respect to the top of the list …
//! could be a bit subdued (for example using the same grey as the one used to display 'search
//! term/regex')"*, and `r` toggles the count-from-the-cursor mode. Read off the drawn page
//! rather than off [`ListPane::number_at`], because the failure worth catching is a column that
//! computes the right number and paints it somewhere else, or in the wrong grey.

use super::*;
use crate::widgets::chrome::MARGIN;

/// The number column's width, as [`frames::columns`] declares it.
const NO_WIDTH: u16 = 3;

/// The numbers the page actually draws, top row of the body downward, one string per body row.
///
/// Empty strings are kept rather than skipped: a blank number cell is a defect this guard should
/// be able to name, not a row it silently walks past.
fn drawn_numbers(buf: &Buffer, rows: usize) -> Vec<String> {
    let first = header_row() + 1;
    (0..rows as u16)
        .map(|i| {
            (MARGIN..MARGIN + NO_WIDTH)
                .map(|x| buf.cell((x, first + i)).expect("cell in area").symbol())
                .collect::<String>()
                .trim()
                .to_string()
        })
        .collect()
}

/// With the mode off, every row reads its own place from the top — 1, 2, 3 downward.
#[test]
fn the_number_is_the_rows_place_from_the_top() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(
        view(QueueState {
            cursor: 5,
            ..QueueState::default()
        }),
        WIDE,
        TALL,
    );

    assert_eq!(
        drawn_numbers(&buf, 8),
        vec!["1", "2", "3", "4", "5", "6", "7", "8"],
        "the cursor sits on row 6 and changes nothing about what any row is numbered"
    );
}

/// With the mode on, the cursor row alone keeps its absolute position and its neighbours both
/// read 1 — the distance, above and below, which is what makes `5j` countable off the screen.
#[test]
fn the_relative_mode_counts_from_the_cursor_and_the_cursor_keeps_its_position() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(
        view(QueueState {
            cursor: 5,
            relative: true,
            ..QueueState::default()
        }),
        WIDE,
        TALL,
    );

    let numbers = drawn_numbers(&buf, 8);
    assert_eq!(
        numbers,
        vec!["5", "4", "3", "2", "1", "6", "1", "2"],
        "distances above and below, and 6 — the cursor's own place — on the cursor row"
    );
}

/// The column is drawn in the muted rung, on every row INCLUDING the cursor's.
///
/// The cursor row is the one worth naming: it wears the cursor tint, and a number that took the
/// row's emphasis with it would be the first thing to brighten.
#[test]
fn the_number_column_is_muted_on_every_row_but_the_cursors_block() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(
        view(QueueState {
            cursor: 5,
            ..QueueState::default()
        }),
        WIDE,
        TALL,
    );

    let muted = crate::tokens::muted();
    let first = header_row() + 1;
    for i in 0..8u16 {
        // The CURSOR's row is the exception, and it became one on 20260912 (ruling 7): the
        // cursor is a block now, and a block inverts everything standing on it — the number
        // column with the rest. The old name of this guard said "the cursor's included" and
        // meant the opposite of what the surface now does.
        let want = if i == 5 {
            crate::tokens::selector_fg()
        } else {
            muted
        };
        for x in MARGIN..MARGIN + NO_WIDTH {
            let cell = buf.cell((x, first + i)).expect("cell in area");
            if cell.symbol().trim().is_empty() {
                continue;
            }
            assert_eq!(
                cell.fg,
                want,
                "row {i} column {x} draws {:?} in {:?}",
                cell.symbol(),
                cell.fg
            );
        }
    }
}

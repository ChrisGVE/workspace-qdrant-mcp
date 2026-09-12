//! The `moved` transition — how a [`crate::motion::Motion`] becomes a new cursor.

use super::*;
use crate::motion::Motion;

/// The projection this state draws, as the buffer index of each row — read through
/// [`state::project_indices`], the one producer of the order, so the guard moves over the same
/// rows the frames draw rather than over a copy of them.
///
/// The row NUMBER is positional (Chris, 2026-09-08), so it is the position in this list plus
/// one and there is nothing else to hold; what a motion needs to be handed is the identity of
/// each drawn row, which is what this is.
fn rows(state: &QueueState) -> Vec<usize> {
    state::project_indices(&fixture::ROWS, state)
}

#[test]
fn moved_clamps_the_cursor_at_both_ends() {
    let numbers = rows(&QueueState::default());
    let last = numbers.len() - 1;

    assert_eq!(
        QueueState::default()
            .moved(Motion::Down, 1000, 10, &numbers)
            .cursor,
        last,
        "Down clamps to the last row"
    );
    assert_eq!(
        QueueState {
            cursor: last,
            ..QueueState::default()
        }
        .moved(Motion::Up, 1000, 10, &numbers)
        .cursor,
        0,
        "Up clamps to the first row"
    );
}

#[test]
fn paging_moves_by_the_number_of_visible_rows() {
    let numbers = rows(&QueueState::default());

    assert_eq!(
        QueueState::default()
            .moved(Motion::PageDown, 2, 10, &numbers)
            .cursor,
        20,
        "two pages of ten down from the top"
    );
    assert_eq!(
        QueueState {
            cursor: 20,
            ..QueueState::default()
        }
        .moved(Motion::PageUp, 1, 10, &numbers)
        .cursor,
        10,
        "one page of ten back up"
    );
}

#[test]
fn top_and_bottom_ignore_the_count() {
    let state = QueueState {
        cursor: 7,
        ..QueueState::default()
    };
    let numbers = rows(&state);

    assert_eq!(state.moved(Motion::Top, 9, 10, &numbers).cursor, 0);
    assert_eq!(
        state.moved(Motion::Bottom, 9, 10, &numbers).cursor,
        numbers.len() - 1
    );
}

#[test]
fn row_places_the_cursor_on_the_row_at_that_position() {
    let numbers = rows(&QueueState::default());

    // The number is the row's PLACE (Chris, 2026-09-08), so `<n>g` counts from the top of what is
    // displayed: row 1 is the first line, row 200 the last.
    assert_eq!(
        QueueState::default()
            .moved(Motion::Row(1), 1, 10, &numbers)
            .cursor,
        0,
        "row 1 is the top of the projection"
    );
    assert_eq!(
        QueueState::default()
            .moved(Motion::Row(200), 1, 10, &numbers)
            .cursor,
        199,
        "row 200 is the last line"
    );
}

#[test]
fn row_past_the_end_of_the_projection_leaves_the_cursor_where_it_was() {
    // A selector narrows to the three failed rows, which are therefore rows 1, 2 and 3. A `<n>g`
    // naming a row past the end must not move the cursor — there is no such line, and moving to
    // the nearest one would look like the row was found.
    let numbers = rows(&QueueState {
        status: Some(Status::Failed),
        ..QueueState::default()
    });
    assert_eq!(numbers.len(), 3, "the selector leaves three rows");

    let state = QueueState {
        status: Some(Status::Failed),
        cursor: 1,
        ..QueueState::default()
    };
    assert_eq!(
        state.moved(Motion::Row(188), 1, 10, &numbers).cursor,
        1,
        "a row the projection does not reach leaves the cursor alone"
    );
}

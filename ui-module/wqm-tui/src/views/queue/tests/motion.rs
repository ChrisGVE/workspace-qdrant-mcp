//! The `moved` transition — how a [`crate::motion::Motion`] becomes a new cursor.

use super::*;
use crate::motion::Motion;

/// The `No` values a projection holds, read through [`state::project`] — the one producer of the
/// projection — so the guard reads the same rows the frames draw, not a copy of them.
fn nos(state: &QueueState) -> Vec<u16> {
    state::project(&fixture::ROWS, state)
        .iter()
        .map(|row| row.no)
        .collect()
}

#[test]
fn moved_clamps_the_cursor_at_both_ends() {
    let numbers = nos(&QueueState::default());
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
    let numbers = nos(&QueueState::default());

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
    let numbers = nos(&state);

    assert_eq!(state.moved(Motion::Top, 9, 10, &numbers).cursor, 0);
    assert_eq!(
        state.moved(Motion::Bottom, 9, 10, &numbers).cursor,
        numbers.len() - 1
    );
}

#[test]
fn row_places_the_cursor_on_the_row_named_by_its_no() {
    let numbers = nos(&QueueState::default());

    // The unsorted projection leads with the rows in progress — No 188 first, No 200 last.
    assert_eq!(
        QueueState::default()
            .moved(Motion::Row(188), 1, 10, &numbers)
            .cursor,
        0,
        "No 188 leads the projection"
    );
    assert_eq!(
        QueueState::default()
            .moved(Motion::Row(200), 1, 10, &numbers)
            .cursor,
        199,
        "No 200 is the last row"
    );
}

#[test]
fn row_on_a_number_the_projection_does_not_hold_leaves_the_cursor_where_it_was() {
    // A selector narrows to the three failed rows: No 198, 199, 200. A `<n>g` naming a row the
    // projection dropped (188) must not move the cursor — there is no row to move to, and moving
    // to the nearest neighbour would look like the row was found.
    let numbers = nos(&QueueState {
        status: Some(Status::Failed),
        ..QueueState::default()
    });
    assert_eq!(numbers, vec![198, 199, 200], "the filtered projection");

    let state = QueueState {
        status: Some(Status::Failed),
        cursor: 1,
        ..QueueState::default()
    };
    assert_eq!(
        state.moved(Motion::Row(188), 1, 10, &numbers).cursor,
        1,
        "No 188 is not in the projection, so the cursor stays"
    );
}

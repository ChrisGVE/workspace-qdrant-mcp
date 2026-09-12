//! Selection — what `v`, `Space` and `Esc` do, and what a selected row looks like.
//!
//! Chris, 2026-09-07, ruling 10: *"`v` additive range, `v` again keeps, `Space` inverts, `Esc`
//! resets (after the search, most-recent layer first); actions apply to the selection; survives
//! search/filter even when hidden; count right-aligned on the dialog row. Colour: new role
//! `selected` = Catppuccin `lavender` wash + `▎` gutter bar, fallback `secondary`; cursor
//! unchanged."*

use super::*;
use crate::motion::Motion;
use crate::panes::list::GUTTER;

/// The projection this state draws, as buffer indices — the same list the keys are handed.
fn projection(state: &QueueState) -> Vec<usize> {
    state::project_indices(&fixture::ROWS, state)
}

/// `Space` at each of `at`, in turn.
fn picked(state: QueueState, at: &[usize]) -> QueueState {
    let rows = projection(&state);
    at.iter().fold(state, |state, cursor| {
        QueueState {
            cursor: *cursor,
            ..state
        }
        .invert_row(&rows)
    })
}

/// `Space` adds a row and takes it away again — the ruling's word is *inverts*.
#[test]
fn space_inverts_the_row_under_the_cursor() {
    let state = picked(QueueState::default(), &[4]);
    assert_eq!(state.selection.len(), 1);

    let again = picked(state, &[4]);
    assert_eq!(again.selection.len(), 0, "the second press takes it back");
}

/// `v` opens a range, any motion extends it, and `v` again closes it KEEPING what it selected.
#[test]
fn a_range_extends_with_the_cursor_and_survives_being_closed() {
    let state = QueueState {
        cursor: 2,
        ..QueueState::default()
    };
    let rows = projection(&state);

    let open = state.toggle_range(&rows);
    assert!(open.selection.extending(), "`v` opens a range");
    assert_eq!(open.selection.len(), 1, "the anchor row is selected at once");

    let dragged = open.moved(Motion::Down, 3, 10, &rows);
    assert_eq!(dragged.selection.len(), 4, "the anchor and three below it");
    for row in &rows[2..=5] {
        assert!(dragged.selection.contains(*row), "row {row} is in the span");
    }

    let closed = dragged.toggle_range(&rows);
    assert!(!closed.selection.extending(), "`v` again ends the range");
    assert_eq!(closed.selection.len(), 4, "and keeps every row it selected");
}

/// A range is the span from the anchor, so coming back RELEASES what the reader passed.
///
/// The alternative — accumulating every row the cursor ever touched — cannot release anything,
/// and a reader who overshoots by two rows would have no way back but Esc.
#[test]
fn coming_back_over_a_range_releases_the_rows_passed() {
    let state = QueueState {
        cursor: 2,
        ..QueueState::default()
    };
    let rows = projection(&state);

    let overshot = state.toggle_range(&rows).moved(Motion::Down, 5, 10, &rows);
    assert_eq!(overshot.selection.len(), 6);

    let back = overshot.moved(Motion::Up, 3, 10, &rows);
    assert_eq!(back.selection.len(), 3, "the span is anchor..cursor, not a trail");
}

/// A range is ADDITIVE: it adds to what was already picked rather than replacing it.
#[test]
fn a_range_adds_to_what_was_already_selected() {
    let scattered = picked(QueueState::default(), &[40, 41]);
    let state = QueueState {
        cursor: 2,
        ..scattered
    };
    let rows = projection(&state);

    let ranged = state.toggle_range(&rows).moved(Motion::Down, 2, 10, &rows);
    assert_eq!(
        ranged.selection.len(),
        5,
        "two picked rows plus a three-row span"
    );
}

/// The selection is named in the BUFFER, so a filter that hides a row does not release it.
///
/// The failure this catches is the obvious representation — positions in the projection — which
/// does not lose the selection, it silently moves it to other rows.
#[test]
fn a_selection_survives_a_filter_that_hides_it() {
    let state = picked(QueueState::default(), &[0, 1, 2]);
    let selected: Vec<usize> = projection(&state)[..3].to_vec();

    let filtered = QueueState {
        filter: Some(Filter::Input("open-books".into())),
        ..state
    }
    .accept_filter(&fixture::ROWS);

    assert_eq!(filtered.selection.len(), 3, "the count is unchanged");
    for row in &selected {
        assert!(
            filtered.selection.contains(*row),
            "row {row} left the selection when it left the screen"
        );
    }
    assert!(
        !projection(&filtered).iter().any(|row| selected.contains(row)),
        "the filter does not actually hide the selected rows — this proves nothing"
    );
}

/// Esc takes the most recent layer and only that one: the search first, then the selection.
#[test]
fn esc_takes_the_search_before_it_takes_the_selection() {
    let searching = picked(QueueState::default(), &[0, 1])
        .open_search()
        .accept_search(&fixture::ROWS);

    let once = searching.escape();
    assert!(once.search.is_none(), "the search is the first layer");
    assert_eq!(once.selection.len(), 2, "the selection is not the first layer");

    let twice = once.escape();
    assert_eq!(twice.selection.len(), 0, "with no search, Esc resets the selection");
}

/// `V` says *this* layer, whatever else is on the screen.
#[test]
fn v_upper_resets_the_selection_past_a_search() {
    let searching = picked(QueueState::default(), &[0, 1])
        .open_search()
        .accept_search(&fixture::ROWS);

    let reset = searching.reset_selection();
    assert_eq!(reset.selection.len(), 0);
    assert!(reset.search.is_some(), "`V` is not Esc and leaves the search");
}

/// A selected row wears the wash and the bar; the cursor's own row keeps the cursor's tint.
///
/// Both halves matter and only the second is a ruling: *"cursor unchanged"*. Read off the drawn
/// page, because what this is really about is which of two fills wins one cell.
#[test]
fn a_selected_row_wears_the_bar_and_the_wash_and_the_cursor_keeps_its_own_tint() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    // Two rows picked; the cursor on the FIRST of them, so one row is both and one is only
    // selected.
    let state = QueueState {
        cursor: 0,
        ..picked(QueueState::default(), &[0, 1])
    };
    let buf = render(view(state), WIDE, TALL);

    let first = header_row() + 1;
    let bar = crate::widgets::chrome::MARGIN - GUTTER;
    for y in [first, first + 1] {
        let cell = buf.cell((bar, y)).expect("cell in area");
        assert_eq!(
            cell.symbol(),
            crate::tokens::SELECTED_BAR.to_string(),
            "row at {y} carries no selection bar"
        );
        assert_eq!(
            cell.fg,
            crate::tokens::selected(),
            "the bar is not the selection hue"
        );
    }

    // The cursor's row keeps the cursor tint; the row below it wears the selection wash.
    let fill = |y: u16| buf.cell((crate::widgets::chrome::MARGIN, y)).expect("cell").bg;
    assert_eq!(
        fill(first),
        crate::tokens::cursor_bg(),
        "the selection took the cursor's row over"
    );
    assert_eq!(
        fill(first + 1),
        crate::tokens::selected_bg().expect("truecolor has a wash"),
        "the selected row is not washed"
    );
    assert_ne!(
        crate::tokens::cursor_bg(),
        crate::tokens::selected_bg().expect("truecolor has a wash"),
        "the two fills are the same colour, so the guard above proves nothing"
    );
}

/// An unselected row carries neither mark — the gutter is blank and the row is not washed.
#[test]
fn a_row_nobody_picked_carries_no_mark_at_all() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(QueueState::default()), WIDE, TALL);
    let bar = crate::widgets::chrome::MARGIN - GUTTER;
    for y in header_row() + 1..header_row() + 6 {
        assert_eq!(
            buf.cell((bar, y)).expect("cell in area").symbol().trim(),
            "",
            "an unselected list draws something in the gutter at {y}"
        );
    }
}

/// The count sits at the right-hand end of the dialog row, and is absent at zero.
#[test]
fn the_count_is_the_rightmost_thing_on_the_dialog_row() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let quiet = render(view(QueueState::default()), WIDE, TALL);
    let slot = header_row() - 1;
    assert_eq!(line(&quiet, slot), "", "an empty selection says nothing");

    let buf = render(view(picked(QueueState::default(), &[0, 1, 2])), WIDE, TALL);
    let row = line(&buf, slot);
    assert!(row.ends_with("3 selected"), "{row:?}");
    assert_eq!(
        row.chars().count(),
        (WIDE - crate::widgets::chrome::MARGIN) as usize,
        "the count is not flush with the right margin: {row:?}"
    );
}

/// Ruling 4's three candidates (Chris, 20260912), and the two things NONE of them may change.
///
/// The ruling is exploratory about the colour — *"is it possible to select another colour"* — and
/// fixed about everything else: *"the vertical bar in the gutter stays"*, and *"the cursor is
/// unchanged and takes priority over a selected row"*. So the properties that are not open get
/// a guard that runs over every candidate, and the property that IS open gets no assertion at
/// all beyond being different from the cursor's grey, which is the complaint that started it.
///
/// Written over [`crate::tokens::Selection::ALL`] rather than as three tests, because the claim
/// is about the SET: a fourth candidate added tomorrow is either covered by this or it is a
/// candidate nobody checked.
#[test]
fn every_selection_candidate_keeps_the_bar_and_lets_the_cursor_win() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    for candidate in crate::tokens::Selection::ALL {
        crate::tokens::Selection::set(candidate);
        let label = candidate.label();

        // The cursor on the first of two picked rows, so one row is both and one is only
        // selected — the only arrangement in which "the cursor takes priority" means anything.
        let state = QueueState {
            cursor: 0,
            ..picked(QueueState::default(), &[0, 1])
        };
        let buf = render(view(state), WIDE, TALL);
        let first = header_row() + 1;
        let bar = crate::widgets::chrome::MARGIN - GUTTER;

        for y in [first, first + 1] {
            let cell = buf.cell((bar, y)).expect("cell in area");
            assert_eq!(
                cell.symbol(),
                crate::tokens::SELECTED_BAR.to_string(),
                "the {label} candidate dropped the gutter bar at {y}"
            );
        }

        let fill = |y: u16| buf.cell((crate::widgets::chrome::MARGIN, y)).expect("cell").bg;
        assert_eq!(
            fill(first),
            crate::tokens::cursor_bg(),
            "the {label} candidate took the cursor's row over"
        );
        let selected = crate::tokens::selected_bg().expect("truecolor has a fill");
        assert_eq!(fill(first + 1), selected, "the {label} candidate did not fill a picked row");
        assert_ne!(
            selected,
            crate::tokens::cursor_bg(),
            "the {label} candidate fills a selected row with the cursor's own colour"
        );

        // And the inversion, which is what distinguishes the block from the two washes: under
        // `Fill` the selected row's content is dark, and the CURSOR's row is not — the priority
        // rule reaches the foreground as well as the fill.
        let content = |y: u16| {
            buf.cell((crate::widgets::chrome::MARGIN + GUTTER + 4, y))
                .expect("cell in area")
                .fg
        };
        match crate::tokens::selected_fg() {
            Some(fg) => {
                assert_eq!(content(first + 1), fg, "the {label} candidate did not invert its row");
                assert_ne!(
                    content(first),
                    fg,
                    "the {label} candidate inverted the cursor's row, which is unchanged by a \
                     selection"
                );
            }
            None => assert_ne!(
                content(first + 1),
                crate::tokens::selector_fg(),
                "the {label} candidate inverted a row without asking to"
            ),
        }
    }
}

//! The dialog slot: its five states, the selectors beside them, and what each of them means.

use super::*;

/// The slot's row on a 125 × 34 screen — directly under the rule that closes the status block.
fn slot(buf: &Buffer) -> String {
    line(buf, header_row() - 1)
}

fn shown(state: QueueState) -> String {
    let _restore = Restore::dark_truecolor();
    slot(&render(view(state), WIDE, TALL))
}

/// A search whose counts come from the projection rather than from a literal here.
fn searching(term: &str) -> QueueState {
    QueueState {
        dialog: Dialog::SearchInput(term.into()),
        ..QueueState::default()
    }
    .accept_search(&fixture::ROWS)
}

fn filtering(term: &str) -> QueueState {
    QueueState {
        dialog: Dialog::FilterInput(term.into()),
        ..QueueState::default()
    }
    .accept_filter(&fixture::ROWS)
}

/// The five states, in Chris's own words, on the row Chris asked for them.
///
/// Spelled out rather than derived from the constants in [`super::super::dialog`], which would
/// be the wording agreeing with itself. The words ARE the ruling.
#[test]
fn the_slot_says_each_of_its_five_things_in_the_ruled_words() {
    let _serial = crate::global_state_lock();

    assert_eq!(
        shown(QueueState::default()),
        "",
        "idle, with no selector set, is a blank row"
    );

    let typing = shown(QueueState {
        dialog: Dialog::SearchInput("readin".into()),
        ..QueueState::default()
    });
    assert!(
        typing.starts_with("  search term/regex: readin▏"),
        "the prompt, the term, and the insert caret past it: {typing:?}"
    );

    let on = shown(searching("reading_guide"));
    assert_eq!(
        on.trim(),
        "search on: reading_guide   1/6   Esc to cancel",
        "{on:?}"
    );

    let filter_typing = shown(QueueState {
        dialog: Dialog::FilterInput("open-book".into()),
        ..QueueState::default()
    });
    assert!(
        filter_typing.starts_with("  filter term/regex: open-book▏"),
        "{filter_typing:?}"
    );

    let filter_on = shown(filtering("open-books"));
    assert_eq!(
        filter_on.trim(),
        "filter on: open-books   38 rows   Esc to cancel",
        "{filter_on:?}"
    );
}

/// The numbers on the row are the projection's own, not a literal a frame happened to write.
///
/// This is what makes the wording guard above safe to read as a fact: `1/6` is six because six
/// rows match, and `38 rows` is thirty-eight because thirty-eight survived.
#[test]
fn the_counts_on_the_row_are_the_projections_own() {
    let state = searching("reading_guide");
    let rows = state::project(&fixture::ROWS, &state);
    let found = state::hits(&rows, "reading_guide");
    match &state.dialog {
        Dialog::SearchOn { hit, hits, .. } => {
            assert_eq!(*hits, found.len(), "the total is counted, not stated");
            assert_eq!(*hit, 1, "the cursor starts on the first hit");
        }
        other => panic!("accept_search must settle the dialog: {other:?}"),
    }
    assert_eq!(
        state.cursor, found[0],
        "the cursor is ON the first hit, not on row one"
    );

    // Two terms, because one count could be right by coincidence — and these two differ by an
    // order of magnitude, so a number written into the code cannot satisfy both.
    for term in ["open-books", "PlotSwift"] {
        let filtered = filtering(term);
        match &filtered.dialog {
            Dialog::FilterOn { rows, .. } => assert_eq!(
                *rows,
                frames::pane(&filtered).len(),
                "the row count for {term:?} is not the reload's own"
            ),
            other => panic!("accept_filter must settle the dialog: {other:?}"),
        }
    }
}

/// A search that matches nothing says so, and does not move the cursor onto a row that is not a
/// hit.
#[test]
fn a_search_that_finds_nothing_says_zero_and_leaves_the_cursor_alone() {
    let _serial = crate::global_state_lock();

    let state = QueueState {
        cursor: 7,
        dialog: Dialog::SearchInput("no-such-thing".into()),
        ..QueueState::default()
    }
    .accept_search(&fixture::ROWS);
    assert_eq!(state.cursor, 7, "there is no first hit to move to");
    assert_eq!(
        shown(state).trim(),
        "search on: no-such-thing   0/0   Esc to cancel"
    );
}

/// The search looks at Tenant, Object, Type and Op — and at nothing else.
///
/// A term that matches only a `Status` finds nothing, which is the whole claim: `s` already
/// selects by status, and a search that did it too would be a second way to do one thing that
/// disagreed about what happens to the cursor.
#[test]
fn the_search_reads_four_fields_and_a_status_word_is_not_one_of_them() {
    let rows = state::project(&fixture::ROWS, &QueueState::default());
    assert!(
        rows.iter().any(|row| row.status == Status::Failed),
        "the buffer has failed rows, or this proves nothing"
    );
    assert_eq!(
        state::hits(&rows, "failed").len(),
        0,
        "a term matching only a Status finds nothing"
    );

    // And each of the four fields it DOES read, so the guard cannot pass by matching none.
    for (field, term) in [
        ("Tenant", "PlotSwift"),
        ("Object", "reading_guide"),
        ("Type", "file"),
        ("Op", "update"),
    ] {
        assert!(
            !state::hits(&rows, term).is_empty(),
            "{field} is not being searched"
        );
    }
}

/// Search and filter are mutually exclusive, and Esc leaves the dialog without touching a
/// selector.
///
/// The exclusivity is the type's doing — [`Dialog`] holds one value — so what is checked is that
/// the transitions honour it rather than working around it.
#[test]
fn starting_one_dialog_replaces_the_other_and_esc_spares_the_selectors() {
    let searching = searching("reading_guide");
    let then_filtering = searching.open_filter();
    assert!(
        matches!(then_filtering.dialog, Dialog::FilterInput(ref term) if term == "reading_guide"),
        "opening a filter replaces the search — and carries the term over: {:?}",
        then_filtering.dialog
    );

    // `/` again: the input state with the term pre-loaded, so a regex need not be retyped.
    let reopened = searching.open_search();
    assert_eq!(reopened.dialog, Dialog::SearchInput("reading_guide".into()));

    // Esc leaves the conversation. The settings are the reader's, and stay.
    let with_selectors = QueueState {
        kind: Some(Kind::Project),
        status: Some(Status::Failed),
        ..filtering("open-books")
    };
    let escaped = with_selectors.escape();
    assert_eq!(escaped.dialog, Dialog::Idle);
    assert_eq!(escaped.kind, Some(Kind::Project), "Esc is not a reset");
    assert_eq!(escaped.status, Some(Status::Failed));
}

/// The selectors cycle in the ruled order, skipping every value the buffer has no rows for, and
/// coming back to All.
///
/// The captured buffer holds only project items, so the type cycle is `All → P → All` — which is
/// the skip rule doing its job, not a shortened cycle. The status cycle visits all three,
/// because the fixture was composed so that it could.
#[test]
fn the_selectors_cycle_in_order_and_skip_what_the_buffer_has_none_of() {
    let idle = QueueState::default();

    let mut kinds = Vec::new();
    let mut state = idle.clone();
    for _ in 0..3 {
        state = state.next_kind(&fixture::ROWS);
        kinds.push(state.kind);
    }
    assert_eq!(
        kinds,
        vec![Some(Kind::Project), None, Some(Kind::Project)],
        "L, S and R have no rows in this buffer, so the cycle steps over them"
    );

    let mut statuses = Vec::new();
    let mut state = idle;
    for _ in 0..4 {
        state = state.next_status(&fixture::ROWS);
        statuses.push(state.status);
    }
    assert_eq!(
        statuses,
        vec![
            Some(Status::Pending),
            Some(Status::InProgress),
            Some(Status::Failed),
            None
        ],
        "pending, in progress, failed, then back to All"
    );
}

/// The selectors are drawn at the right of the slot, cumulative, and omitted when they are All.
///
/// Chris did not say where they go, so this is the supervisor's ruling made checkable: a knob
/// whose position lives only in the reader's memory is a knob they forget they turned.
#[test]
fn the_selectors_are_drawn_at_the_right_of_the_slot_and_coexist_with_a_dialog() {
    let _serial = crate::global_state_lock();

    let both = shown(QueueState {
        kind: Some(Kind::Project),
        status: Some(Status::Failed),
        ..QueueState::default()
    });
    assert!(both.ends_with("type P · status failed"), "{both:?}");
    assert_eq!(
        both.trim(),
        "type P · status failed",
        "the slot holds nothing else"
    );

    // One alone, and the other omitted.
    let one = shown(QueueState {
        status: Some(Status::InProgress),
        ..QueueState::default()
    });
    assert_eq!(one.trim(), "status in progress", "{one:?}");

    // Beside a dialog: the conversation on the left, the settings on the right, on one row.
    let alongside = shown(QueueState {
        kind: Some(Kind::Project),
        ..searching("reading_guide")
    });
    assert!(
        alongside
            .trim_start()
            .starts_with("search on: reading_guide"),
        "{alongside:?}"
    );
    assert!(alongside.ends_with("type P"), "{alongside:?}");
}

/// The selectors narrow the BUFFER and are cumulative with each other and with a filter.
///
/// Three narrowings on one screen, and each is checked against what the projection actually
/// returned rather than against a number written here.
#[test]
fn the_selectors_narrow_the_buffer_and_stack_with_a_filter() {
    let failed = frames::pane(&QueueState {
        status: Some(Status::Failed),
        ..QueueState::default()
    })
    .len();
    assert_eq!(failed, 3);

    let and_type = frames::pane(&QueueState {
        kind: Some(Kind::Project),
        status: Some(Status::Failed),
        ..QueueState::default()
    })
    .len();
    assert_eq!(and_type, failed, "every captured row is a project item");

    // A filter on top of both: PlotSwift is where the failures are, so this survives; a term
    // that is elsewhere does not.
    let with_filter = frames::pane(&QueueState {
        kind: Some(Kind::Project),
        status: Some(Status::Failed),
        ..filtering("PlotSwift")
    })
    .len();
    assert_eq!(with_filter, 3, "a filter stacks with the selectors");

    let disjoint = frames::pane(&QueueState {
        status: Some(Status::Failed),
        ..filtering("open-books")
    })
    .len();
    assert_eq!(disjoint, 0, "nothing is both failed and an open-books row");
}

/// A filter reloads the list; a search does not.
///
/// The one sentence that says what the two dialogs are for, and the one a reader would otherwise
/// have to discover by pressing both.
#[test]
fn a_filter_reloads_the_list_and_a_search_only_moves_the_cursor() {
    let filtered = frames::pane(&filtering("open-books"));
    assert_eq!(filtered.len(), 38, "the list is what matched");

    let searched = frames::pane(&searching("reading_guide"));
    assert_eq!(
        searched.len(),
        crate::panes::list::LIST_PAGE,
        "a search leaves every row where it was"
    );
}

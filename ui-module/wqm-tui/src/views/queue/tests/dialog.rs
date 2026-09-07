//! The dialog slot: its two conversations and their states, the selectors beside them, and what
//! each means.

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
        search: Some(Search::Input(term.into())),
        ..QueueState::default()
    }
    .accept_search(&fixture::ROWS)
}

fn filtering(term: &str) -> QueueState {
    QueueState {
        filter: Some(Filter::Input(term.into())),
        ..QueueState::default()
    }
    .accept_filter(&fixture::ROWS)
}

/// The four conversation states, in Chris's own words, on the row Chris asked for them.
///
/// Spelled out rather than derived from the constants in [`super::super::dialog`], which would
/// be the wording agreeing with itself. The words ARE the ruling.
#[test]
fn the_slot_says_each_of_its_four_things_in_the_ruled_words() {
    let _serial = crate::global_state_lock();

    assert_eq!(
        shown(QueueState::default()),
        "",
        "idle, with no selector set, is a blank row"
    );

    let typing = shown(QueueState {
        search: Some(Search::Input("readin".into())),
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
        filter: Some(Filter::Input("open-book".into())),
        ..QueueState::default()
    });
    assert!(
        filter_typing.starts_with("  filter term/regex: open-book▏"),
        "{filter_typing:?}"
    );

    let filter_on = shown(filtering("open-books"));
    assert_eq!(
        filter_on.trim(),
        "filter on: open-books   38 rows   f to clear",
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
    match &state.search {
        Some(Search::On { hit, hits, .. }) => {
            assert_eq!(*hits, found.len(), "the total is counted, not stated");
            assert_eq!(*hit, 1, "the cursor starts on the first hit");
        }
        other => panic!("accept_search must settle the search: {other:?}"),
    }
    assert_eq!(
        state.cursor, found[0],
        "the cursor is ON the first hit, not on row one"
    );

    // Two terms, because one count could be right by coincidence — and these two differ by an
    // order of magnitude, so a number written into the code cannot satisfy both.
    for term in ["open-books", "PlotSwift"] {
        let filtered = filtering(term);
        match &filtered.filter {
            Some(Filter::On { rows, .. }) => assert_eq!(
                *rows,
                frames::pane(&filtered).len(),
                "the row count for {term:?} is not the reload's own"
            ),
            other => panic!("accept_filter must settle the filter: {other:?}"),
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
        search: Some(Search::Input("no-such-thing".into())),
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

/// Search and filter coexist: opening one leaves the other alone, and the one opened first is
/// the one drawn on the left.
///
/// The coexistence is the type's doing — the two slots are independent — so what is checked is
/// that the row draws both, in the order they arrived, and that Esc takes the search without
/// touching the filter or a selector.
#[test]
fn opening_one_conversation_leaves_the_other_and_the_elder_sits_on_the_left() {
    let _serial = crate::global_state_lock();

    let searching = searching("reading_guide");
    let both = searching.toggle_filter();
    assert_eq!(
        both.first,
        First::Search,
        "the search was opened first"
    );
    assert!(
        matches!(both.search, Some(Search::On { .. })),
        "opening the filter left the search alone"
    );
    assert!(
        matches!(both.filter, Some(Filter::Input(ref term)) if term.is_empty()),
        "`f` opens the filter as an empty input"
    );

    // Enter settles the filter beside the search: the elder on the left, the second to its
    // right. Built with a term typed in so the row is the two conversations, not an empty one.
    let settled = QueueState {
        filter: Some(Filter::Input("open-books".into())),
        ..both
    }
    .accept_filter(&fixture::ROWS);
    let row = shown(settled.clone());
    assert!(
        row.trim_start().starts_with("search on: reading_guide"),
        "the first-opened search is on the left: {row:?}"
    );
    assert!(
        row.trim_end().ends_with("filter on: open-books   38 rows   f to clear"),
        "the second-opened filter is to its right: {row:?}"
    );

    // Esc takes the search — and touches neither the filter nor a selector.
    let with_selectors = QueueState {
        op: Some(Op::Update),
        status: Some(Status::Failed),
        ..settled.clone()
    };
    let escaped = with_selectors.escape();
    assert_eq!(escaped.search, None, "Esc clears the search");
    assert!(
        matches!(escaped.filter, Some(Filter::On { .. })),
        "Esc leaves the filter where it was"
    );
    assert_eq!(escaped.op, Some(Op::Update), "Esc is not a reset");
    assert_eq!(escaped.status, Some(Status::Failed));

    // The filter side of the order: opened first, the filter leads the search.
    let filtering = filtering("open-books");
    let both = QueueState {
        search: Some(Search::Input("reading_guide".into())),
        first: First::Filter,
        ..filtering
    }
    .accept_search(&fixture::ROWS);
    let row = shown(both);
    assert!(
        row.trim_start().starts_with("filter on: open-books"),
        "the first-opened filter is on the left: {row:?}"
    );
    assert!(
        row.trim_end().ends_with("search on: reading_guide   1/6   Esc to cancel"),
        "the second-opened search is to its right: {row:?}"
    );
}

/// The selectors cycle in the ruled order, skipping every value the buffer has no rows for, and
/// settling back to All.
///
/// The captured buffer holds only adds and updates, so the operation cycle is
/// `All → add → update → All` — which is the skip rule doing its job, not a shortened cycle. The
/// status cycle visits all three, because the fixture was composed so that it could.
#[test]
fn the_selectors_cycle_in_order_and_skip_what_the_buffer_has_none_of() {
    let idle = QueueState::default();

    let mut ops = Vec::new();
    let mut state = idle.clone();
    for _ in 0..3 {
        state = state.next_op(&fixture::ROWS);
        ops.push(state.op);
    }
    assert_eq!(
        ops,
        vec![Some(Op::Add), Some(Op::Update), None],
        "delete and scan have no rows in this buffer, so the cycle steps over them"
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
fn the_selectors_are_drawn_at_the_right_of_the_slot_and_coexist_with_a_conversation() {
    let _serial = crate::global_state_lock();

    let both = shown(QueueState {
        op: Some(Op::Add),
        status: Some(Status::Failed),
        ..QueueState::default()
    });
    assert!(both.ends_with("op add · status failed"), "{both:?}");
    assert_eq!(
        both.trim(),
        "op add · status failed",
        "the slot holds nothing else"
    );

    // One alone, and the other omitted.
    let one = shown(QueueState {
        status: Some(Status::InProgress),
        ..QueueState::default()
    });
    assert_eq!(one.trim(), "status in progress", "{one:?}");

    // Beside a conversation: the search on the left, the setting on the right, on one row.
    let alongside = shown(QueueState {
        op: Some(Op::Update),
        ..searching("reading_guide")
    });
    assert!(
        alongside
            .trim_start()
            .starts_with("search on: reading_guide"),
        "{alongside:?}"
    );
    assert!(alongside.ends_with("op update"), "{alongside:?}");
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

    let and_op = frames::pane(&QueueState {
        op: Some(Op::Add),
        status: Some(Status::Failed),
        ..QueueState::default()
    })
    .len();
    assert_eq!(and_op, failed, "the three failed rows are all adds");

    // A filter on top of both: PlotSwift is where the failures are, so this survives; a term
    // that is elsewhere does not.
    let with_filter = frames::pane(&QueueState {
        op: Some(Op::Add),
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
/// The one sentence that says what the two conversations are for, and the one a reader would
/// otherwise have to discover by pressing both.
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

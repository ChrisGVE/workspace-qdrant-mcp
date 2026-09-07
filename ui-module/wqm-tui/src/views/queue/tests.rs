//! What the Queue tab is pinned to. The shared scaffolding and the claims about the state
//! machine itself; the claims about what is drawn are in the submodules.

use super::*;
use crate::panes::cell::{Cell, Direction, Sort};
use crate::widgets::chrome::test_support::{coloured_cells, neutral_rungs, Restore};
use crate::widgets::chrome::MARGIN;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::widgets::Widget;

mod dialog;
mod help;
mod layout;
mod sort;

pub(super) const WIDE: u16 = 125;
pub(super) const TALL: u16 = 34;

/// The Queue on the captured workspace, showing `state` — the ONE builder every guard uses, so
/// no guard can be looking at a screen the pantry does not draw.
pub(super) fn view(state: QueueState) -> Queue {
    let entries = [
        Health::Healthy,
        Health::Degraded,
        Health::Healthy,
        Health::Healthy,
    ];
    let overall = overall(entries[0], &entries[1..]);
    Queue::new(
        state,
        StatusBlock::new(
            overall,
            "v0.2.0",
            crate::widgets::chrome::Freshness::new(
                std::time::Duration::from_secs(4),
                std::time::Duration::from_secs(60),
            ),
            entries,
            crate::panes::status_block::Queue {
                pending: 11_236,
                in_progress: 4,
                failed: 3,
                health: Health::Degraded,
            },
        ),
    )
}

pub(super) fn render(view: Queue, width: u16, height: u16) -> Buffer {
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    view.render(area, &mut buf);
    buf
}

pub(super) fn line(buf: &Buffer, y: u16) -> String {
    (0..buf.area.width)
        .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
        .collect::<String>()
        .trim_end()
        .to_string()
}

/// The row the list's column header is drawn on: the constant top, then the dialog slot.
pub(super) fn header_row() -> u16 {
    crate::views::top::CONSTANT_ROWS + crate::panes::status_block::ROWS_FULL + DIALOG_ROWS
}

/// Not one cell of the page keeps a colour while a modal owns the input.
///
/// The Dashboard's own sweep, applied to this screen, and it matters more here: the `Status`
/// column paints a hue on **every row**, so a page that failed to go quiet would fail loudly and
/// a per-widget guard would not have covered it.
#[test]
fn no_cell_of_the_queue_carries_a_colour_under_a_modal() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let neutrals = neutral_rungs();
    let live = render(view(QueueState::default()), WIDE, TALL);
    assert!(
        !coloured_cells(&live, &neutrals).is_empty(),
        "the Queue paints no colour even when it is live — this guard checks nothing"
    );

    let under = render(view(QueueState::default()).under_modal(true), WIDE, TALL);
    let survivors = coloured_cells(&under, &neutrals);
    assert!(
        survivors.is_empty(),
        "{} cells kept a colour under a modal, first ten: {:?}",
        survivors.len(),
        &survivors[..survivors.len().min(10)]
    );
}

/// The `Status` column wears the status block's own three hues, and each row wears its own.
///
/// Read off the rendered page rather than off the fixture, because the failure this catches is a
/// column drawn as plain text — which the data would still describe perfectly.
#[test]
fn the_status_column_carries_the_status_blocks_own_three_hues() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    for (status, hue) in [
        (Status::Pending, crate::tokens::degraded()),
        (Status::InProgress, crate::tokens::in_flight()),
        (Status::Failed, crate::tokens::offline()),
    ] {
        let buf = render(
            view(QueueState {
                status: Some(status),
                ..QueueState::default()
            }),
            WIDE,
            TALL,
        );
        let row = header_row() + 1;
        let painted: Vec<String> = (0..WIDE)
            .filter(|x| buf.cell((*x, row)).expect("cell in area").style().fg == Some(hue))
            .map(|x| buf.cell((x, row)).expect("cell in area").symbol().to_string())
            .collect();
        assert_eq!(
            painted.concat(),
            status.label(),
            "{status:?} is the only thing on its row wearing its own hue"
        );
    }
}

/// The reference number is invariant under a selector and under a filter, which is the whole of
/// what "assigned at load" buys.
///
/// A `No` derived from the projection would come back `1, 2, 3` from every narrowing, and the
/// number a reader wrote down would name a different row every time they looked.
#[test]
fn the_reference_number_survives_the_selectors_and_a_filter() {
    fn numbers(state: QueueState) -> Vec<u64> {
        frames::pane(&state)
            .rows()
            .iter()
            .map(|row| match &row[frames::NO] {
                Cell::Num(no) => *no,
                _ => panic!("the No column is a figure"),
            })
            .collect()
    }

    let all = numbers(QueueState::default());
    assert_eq!(
        all[0], 188,
        "with no sort chosen the rows in progress lead — row 188 is the first of them"
    );
    assert_eq!(all.len(), crate::panes::list::LIST_PAGE);

    // The three failed rows are the last three of the captured page, so their numbers are the
    // last three — not 1, 2, 3.
    let failed = numbers(QueueState {
        status: Some(Status::Failed),
        ..QueueState::default()
    });
    assert_eq!(
        failed,
        vec![198, 199, 200],
        "a selector keeps each row's own number"
    );

    // And a filter, which reloads the list rather than narrowing it in place.
    let filtered = numbers(QueueState {
        filter: Some(Filter::On {
            term: "PlotSwift".into(),
            rows: 3,
        }),
        ..QueueState::default()
    });
    assert_eq!(filtered, vec![198, 199, 200], "a filter keeps them too");
}

/// The two conversations are independent, and each leaves by its own door.
///
/// Chris, 2026-09-07: *"I was thinking they were mutually exclusive but they are not"* — the
/// transitions are the whole of that ruling made checkable: opening either conversation leaves
/// the other exactly where it was, Esc takes the search alone, and `f` — which opened the
/// filter — is the key that closes it.
#[test]
fn the_two_conversations_are_independent_and_each_leaves_by_its_own_door() {
    let searching = QueueState {
        search: Some(Search::Input("reading_guide".into())),
        ..QueueState::default()
    }
    .accept_search(&fixture::ROWS);

    // `f` opens the filter beside the search, and the search survives it.
    let opened = searching.toggle_filter();
    assert!(
        matches!(opened.search, Some(Search::On { hits: 6, .. })),
        "opening the filter leaves the search alone: {:?}",
        opened.search
    );
    assert!(
        matches!(opened.filter, Some(Filter::Input(ref term)) if term.is_empty()),
        "`f` with no filter opens the input: {:?}",
        opened.filter
    );
    assert_eq!(opened.first, First::Search, "the search was opened first");

    // Enter on the filter: the search still survives, and its hits were counted over the rows
    // the filter left. PlotSwift holds three rows, two of which match `Tests` — a count over
    // the whole buffer would say otherwise, so this is the difference made checkable.
    let filtered = QueueState {
        filter: Some(Filter::Input("PlotSwift".into())),
        first: First::Filter,
        ..searching.clone()
    }
    .accept_filter(&fixture::ROWS);
    assert_eq!(filtered.filter, Some(Filter::On { term: "PlotSwift".into(), rows: 3 }));
    let searched_within = QueueState {
        search: Some(Search::Input("Tests".into())),
        ..filtered.clone()
    }
    .accept_search(&fixture::ROWS);
    assert_eq!(
        searched_within.search,
        Some(Search::On { term: "Tests".into(), hit: 1, hits: 2 }),
        "the search counts its hits over the rows the filter left"
    );
    assert_eq!(searched_within.cursor, 1, "the cursor is on the first of the two hits");

    // Esc takes the search — typing or settled — and nothing else.
    let escaped = searched_within.escape();
    assert_eq!(escaped.search, None, "Esc clears the search");
    assert_eq!(
        escaped.filter,
        Some(Filter::On { term: "PlotSwift".into(), rows: 3 }),
        "Esc leaves the filter where it was"
    );
    let escaped_typing = QueueState {
        search: Some(Search::Input("readin".into())),
        ..escaped.clone()
    }
    .escape();
    assert_eq!(escaped_typing.search, None, "Esc clears a search still typing");

    // `f` on an accepted filter clears it — the key that opened it is the key that closes it —
    // and `f` on a filter still typing is a letter in the term, not a command.
    let cleared = escaped.toggle_filter();
    assert_eq!(cleared.filter, None, "`f` clears an accepted filter");
    assert_eq!(cleared.search, None, "and touches nothing else");
    let typing = QueueState {
        filter: Some(Filter::Input("sv".into())),
        ..cleared.clone()
    };
    assert_eq!(
        typing.toggle_filter().filter,
        Some(Filter::Input("sv".into())),
        "`f` on a filter still typing is a letter in the term, not a command"
    );

    // An empty term matches everything — the whole page.
    let empty = cleared.toggle_filter().accept_filter(&fixture::ROWS);
    assert_eq!(
        empty.filter,
        Some(Filter::On { term: String::new(), rows: 200 }),
        "an empty term matches everything — the whole page"
    );

    // `/` re-opens the search input pre-loaded with the search's own term, and never with the
    // filter's.
    let reopened = searching.open_search();
    assert_eq!(
        reopened.search,
        Some(Search::Input("reading_guide".into())),
        "`/` pre-loads the search's own term"
    );
    assert_eq!(reopened.filter, None, "the filter is untouched throughout");
}

/// With no sort chosen, the rows in progress lead and the rest keep the buffer's order; a
/// chosen sort replaces the default outright.
///
/// Read off [`state::project`] — the one producer of the order — with the sort's half checked
/// through [`frames::pane`], which is where a chosen sort is applied.
#[test]
fn with_no_sort_chosen_the_rows_in_progress_lead_in_buffer_order() {
    let rows = state::project(&fixture::ROWS, &QueueState::default());
    let statuses: Vec<Status> = rows.iter().map(|row| row.status).collect();
    assert_eq!(
        statuses
            .iter()
            .filter(|status| **status == Status::InProgress)
            .count(),
        10,
        "the captured page holds ten rows in progress"
    );
    assert!(
        statuses[..10]
            .iter()
            .all(|status| *status == Status::InProgress),
        "they lead"
    );
    assert_eq!(rows[0].no, 188, "in buffer order among themselves");
    assert_eq!(
        rows[10].no, 1,
        "then the remaining rows in buffer order, from the top of the page"
    );
    assert_eq!(rows[199].no, 200, "down to the last of them");

    // A chosen sort replaces the default rather than composing with it: ascending by age puts
    // row 1 — newest, and not in progress — back at the top.
    let sorted = frames::pane(&QueueState {
        sort: Some(Sort {
            column: frames::AGE,
            direction: Direction::Asc,
        }),
        ..QueueState::default()
    });
    assert!(
        matches!(&sorted.rows()[0][frames::NO], Cell::Num(1)),
        "a chosen sort outranks the in-progress-first default"
    );
}

/// The captured page is exactly one buffer page, so the list ends in the offer of the next.
///
/// Both halves are checked, because the offer means *a full page AND more behind it*: the
/// fixture is two hundred rows of a store that holds eleven thousand.
#[test]
fn the_captured_page_ends_in_the_offer_of_the_next() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let pane = frames::pane(&QueueState::default());
    assert_eq!(pane.len(), crate::panes::list::LIST_PAGE);
    assert!(pane.shows_load_more());

    let buf = render(
        view(QueueState {
            cursor: crate::panes::list::LIST_PAGE,
            ..QueueState::default()
        }),
        WIDE,
        TALL,
    );
    let last = line(&buf, TALL - crate::views::top::FOOT_ROWS - 1);
    assert_eq!(
        last.trim(),
        "200 rows, press Enter to load 200 more rows",
        "{last:?}"
    );

    // A narrowed list is NOT a full page, so it makes no such offer.
    let narrowed = frames::pane(&QueueState {
        status: Some(Status::Failed),
        ..QueueState::default()
    });
    assert_eq!(narrowed.len(), 3);
    assert!(
        !narrowed.shows_load_more(),
        "three rows is the whole of what matched — there is no next page to offer"
    );
}

/// Every tenant in the fixture is a project name this repository **already** publishes.
///
/// The crate is public and the fixture is real captured data, so the vetted set is exactly the
/// names already in [`crate::views::dashboard::frames`] — nothing new about this machine reaches
/// the repository through the Queue tab. A guard rather than a note, because the next capture
/// will be taken by somebody reading the note only if they think to look for it.
#[test]
fn every_tenant_is_a_project_name_the_repository_already_publishes() {
    use crate::views::dashboard::frames as dash;

    let vetted: Vec<&str> = dash::PROJECTS
        .iter()
        .map(|project| project.name)
        .chain(dash::ACTIVE.iter().map(|project| project.name))
        // `Last Errors` names its collection as `[P] PlotSwift`; the tenant is the name after
        // the type tag.
        .chain(
            dash::ERRORS
                .iter()
                .map(|error| error.collection.rsplit(' ').next().unwrap_or("")),
        )
        .collect();

    for row in fixture::ROWS.iter() {
        assert!(
            vetted.contains(&row.tenant),
            "row {} names `{}`, which this repository does not already publish",
            row.no,
            row.tenant
        );
    }
}

/// The reference numbers are the fixture's own positions, 1-based, with none missing.
///
/// The invariance guards say a number moves with its row; this says the numbers were assigned
/// at load in the first place. A fixture edited by hand into `1, 2, 2, 4` would satisfy every
/// other guard here.
#[test]
fn the_reference_numbers_are_the_fixtures_own_positions() {
    for (at, row) in fixture::ROWS.iter().enumerate() {
        assert_eq!(
            row.no as usize,
            at + 1,
            "row {at} carries No {}, which is not where it sits",
            row.no
        );
    }
    assert_eq!(fixture::ROWS.len(), crate::panes::list::LIST_PAGE);
}

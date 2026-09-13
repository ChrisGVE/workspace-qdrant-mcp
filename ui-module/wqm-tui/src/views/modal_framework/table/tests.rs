//! The floor, the column policy, and the drill-down's focus — one test each.

use super::*;
use crate::encoding::Encoding;
use crate::panes::cell::Align;
use crate::tokens::{self, ModalTint, Palette};
use ratatui::buffer::Buffer;
use ratatui::widgets::Widget;

const VIEW: Rect = Rect {
    x: 0,
    y: 0,
    width: 92,
    height: 17,
};

/// The `Tenant` column, in the list's own numbering: column 0 is the row number.
const TENANT: usize = 1;

struct Restore(Palette, Encoding, ModalTint, f32);

impl Restore {
    fn mocha() -> Self {
        let restore = Restore(
            Palette::current(),
            Encoding::current(),
            ModalTint::current(),
            tokens::tint_strength(),
        );
        Palette::set(Palette::Bundled);
        Encoding::set(Encoding::TrueColor);
        tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());
        restore
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        Palette::set(self.0);
        Encoding::set(self.1);
        ModalTint::set(self.2);
        tokens::set_tint_strength(self.3);
    }
}

fn columns() -> Vec<Column> {
    vec![
        Column::number("", 3),
        Column::text("Tenant", 20),
        Column::flex("Object"),
        Column::text("Status", 11),
    ]
}

/// Three tenants, so a floor that leaked would have somewhere visible to leak from.
fn rows() -> Vec<Vec<Cell>> {
    let row = |tenant: &str, object: &str, status: &str| {
        vec![
            Cell::Text(tenant.into()),
            Cell::Text(object.into()),
            Cell::Text(status.into()),
        ]
    };
    vec![
        row("open-books", "stage_b/reading_guide.py", "pending"),
        row("open-books", "stage_b/index.py", "in progress"),
        row("mnemosyne", "src/recall.rs", "pending"),
        row("open-books", "stage_a/fetch.py", "failed"),
        row("mnemosyne", "src/store.rs", "failed"),
        row("thales", "crates/core/lib.rs", "pending"),
    ]
}

fn view() -> TableView {
    TableView::new(columns(), rows()).pinned(Pin::new(TENANT, "open-books"))
}

fn tenants(view: &TableView) -> Vec<String> {
    view.visible()
        .iter()
        .map(|at| rows()[*at][0].plain())
        .collect()
}

fn screen(view: TableView) -> String {
    let mut buf = Buffer::empty(VIEW);
    view.pane().render(VIEW, &mut buf);
    let mut out = String::new();
    for y in VIEW.y..VIEW.bottom() {
        for x in VIEW.x..VIEW.right() {
            if let Some(cell) = buf.cell((x, y)) {
                out.push_str(cell.symbol());
            }
        }
        out.push('\n');
    }
    out
}

/// The floor holds: only rows of the library the reader drilled in from, and nothing the
/// reader types changes that.
#[test]
fn the_pre_filter_floor_holds_under_any_narrowing() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    assert_eq!(tenants(&view()).len(), 3, "three rows are inside the floor");
    assert!(tenants(&view()).iter().all(|t| t == "open-books"));

    // Narrowing INSIDE the floor works and reaches fewer rows.
    let narrowed = view().narrowed("stage_b");
    assert_eq!(narrowed.len(), 2);
    assert!(tenants(&narrowed).iter().all(|t| t == "open-books"));

    // Narrowing at a value OUTSIDE the floor reaches nothing — never the other tenant's rows.
    // That is the property: the term is applied inside the floor, so it can only ever subtract.
    for term in ["mnemosyne", "thales", "src/recall.rs", "crates"] {
        let widened = view().narrowed(term);
        assert!(
            widened.is_empty(),
            "{term:?} reached outside the floor: {:?}",
            tenants(&widened)
        );
    }
}

/// An unpinned view has no floor to hold, and filtering is ordinary filtering — so the test
/// above is measuring the pin rather than the filter.
#[test]
fn an_unpinned_view_is_narrowed_by_the_same_term_to_other_tenants() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let free = TableView::new(columns(), rows()).narrowed("mnemosyne");
    assert_eq!(free.len(), 2, "the same term finds rows without a floor");
}

/// The narrowing is a substring match, and it looks at every column — the reader types what
/// they can see.
#[test]
fn narrowing_matches_any_column_case_insensitively() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    assert_eq!(view().narrowed("FAILED").len(), 1, "by status, any case");
    assert_eq!(view().narrowed("fetch").len(), 1, "by object");
    assert_eq!(
        view().narrowed("").len(),
        3,
        "an empty term narrows nothing"
    );
}

/// **A drilled-in view drops the column it was pinned by.** Twenty columns repeating one value
/// is twenty columns carrying no information, and the breadcrumb already says which library
/// this is.
#[test]
fn a_drilled_in_view_drops_the_column_it_was_pinned_by() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let dropped = screen(view());
    assert!(
        !dropped.contains("Tenant"),
        "the pinned column's header is still drawn:\n{dropped}"
    );
    assert!(
        !dropped.contains("open-books"),
        "…and its repeated value with it:\n{dropped}"
    );
    assert!(
        dropped.contains("reading_guide.py"),
        "the rows are still here"
    );

    let shown = screen(view().show_pinned_column(true));
    assert!(shown.contains("Tenant"), "the other arm keeps it");
    assert!(shown.contains("open-books"));
}

/// Dropping the column hands its width to the flex column, which is the whole point — the
/// reader gets more of the path they are trying to finish reading.
#[test]
fn the_dropped_width_goes_to_the_column_the_reader_is_reading() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();

    // A path far too long for either arm, so BOTH elide it and the question becomes *how
    // much* survives rather than *whether* it does. With a path that happens to fit, this
    // test passes whatever the column policy does — which is what the first version of it
    // did: it asserted the path was present, which is the row's NAME, not its geometry.
    const PATH: &str =
        "book_building/common/stage_b/generated/reading_guide_variants/VARIANTS_expanded.md";
    let long_row = || {
        vec![
            Cell::Text("open-books".into()),
            Cell::Text(PATH.into()),
            Cell::Text("pending".into()),
        ]
    };
    let with_long_path = |show_pinned: bool| {
        TableView::new(columns(), vec![long_row()])
            .pinned(Pin::new(TENANT, "open-books"))
            .show_pinned_column(show_pinned)
    };

    // How much of the path survived, read off the rendered row: the longest PREFIX of it
    // still on screen.
    //
    // A prefix and not a suffix, and that was worth rendering to find out. A plain
    // `Column::flex` elides from the right; the Queue's own `Object` adds `.elide_left()`
    // because the end of a path is the file — but that is the Queue's choice, not this
    // policy's, and this fixture keeps the default. The first version of this measurement
    // searched for the wrong end and reported one character for both arms.
    let visible_path = |show_pinned: bool| {
        let drawn = screen(with_long_path(show_pinned));
        let row = drawn.lines().nth(1).expect("the one data row").to_string();
        (1..=PATH.chars().count())
            .filter(|n| {
                let head: String = PATH.chars().take(*n).collect();
                row.contains(&head)
            })
            .max()
            .unwrap_or(0)
    };

    let dropped = visible_path(false);
    let shown = visible_path(true);
    assert!(
        shown > 0 && dropped > shown,
        "dropping the pinned column must hand its width to the flex column: {dropped} \
         characters of the path visible with it dropped, {shown} with it shown"
    );
    // …and the width handed over is the pinned column's own, give or take the gap between
    // columns. Stated as a floor rather than an equality so a change to the inter-column gap
    // does not fail a test about column policy.
    assert!(
        dropped - shown >= 18,
        "only {} characters were handed over; the Tenant column is 20 wide",
        dropped - shown
    );
}

/// An unpinned view drops nothing: the policy is *the column you were pinned by*, not *the
/// first column*.
#[test]
fn a_view_with_no_pin_drops_no_column() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let free = screen(TableView::new(columns(), rows()));
    assert!(free.contains("Tenant"));
    assert!(free.contains("mnemosyne"), "and no floor, either");
}

/// The fifth row states the floor. Error prevention: a limit the reader can see is not a
/// limit they discover by pressing.
#[test]
fn the_fifth_row_states_the_floor_the_reader_cannot_cross() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let row = view().search_row().expect("a pinned view mounts the row");
    assert!(row.term.contains("open-books"), "{:?}", row.term);

    let narrowed = view()
        .narrowed("stage_b")
        .search_row()
        .expect("still pinned");
    assert!(
        narrowed.term.contains("open-books"),
        "the floor stays shown"
    );
    assert!(narrowed.term.contains("stage_b"), "with the term beside it");

    assert!(
        TableView::new(columns(), rows()).search_row().is_none(),
        "an unpinned view keeps that row for its content"
    );
}

/// Enter drills down *with the focus on that row*, so the view has to be able to say which row
/// of the ORIGINAL buffer the cursor is on — not which row of the filtered projection.
#[test]
fn the_cursor_names_a_row_of_the_buffer_and_not_of_the_projection() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    // Rows 0, 1 and 3 of the buffer are inside the floor; the projection's row 2 is buffer 3.
    assert_eq!(view().cursor(0).focused_row(), Some(0));
    assert_eq!(view().cursor(2).focused_row(), Some(3));
    assert_eq!(view().cursor(9).focused_row(), None, "past the end");

    // And under a narrowing, it still names the buffer.
    assert_eq!(
        view().narrowed("fetch").cursor(0).focused_row(),
        Some(3),
        "the narrowed projection's first row is buffer row 3"
    );
}

/// The scroll the container is handed counts the rows on screen, which is the floor's count
/// and not the buffer's.
#[test]
fn the_scroll_total_counts_what_is_on_screen() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    assert_eq!(view().scroll().total, 3);
    assert_eq!(view().narrowed("stage_b").scroll().total, 2);
    assert_eq!(view().offset(1).scroll().offset, 1);
}

/// A floor that matches nothing draws the table's own empty line — the column-title row stays,
/// because that row IS the view.
#[test]
fn a_floor_that_matches_nothing_keeps_the_column_titles() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let nothing = TableView::new(columns(), rows()).pinned(Pin::new(TENANT, "no-such-library"));
    assert!(nothing.is_empty());
    let drawn = screen(nothing);
    assert!(
        drawn.contains(crate::panes::cell::EMPTY),
        "the table says what it always says:\n{drawn}"
    );
    assert!(drawn.contains("Object"), "and its column titles remain");
}

/// A pin naming a column no row carries holds nothing rather than holding everything — a floor
/// that failed open would be worse than no floor at all.
#[test]
fn a_pin_on_a_column_that_does_not_exist_fails_closed() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let bad = TableView::new(columns(), rows()).pinned(Pin::new(99, "open-books"));
    assert!(bad.is_empty(), "a floor must never fail open");
}

/// The column numbering is the list's, and the `- 1` to reach a cell is written once.
#[test]
fn a_pin_names_a_column_and_converts_to_a_cell_in_one_place() {
    let _ = Align::Left;
    assert_eq!(Pin::new(TENANT, "x").cell(), 0);
    assert_eq!(Pin::new(3, "x").cell(), 2);
    assert_eq!(Pin::new(0, "x").cell(), 0, "the number column carries none");
}

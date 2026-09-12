//! The shape of the screen: what rows it has, what it does NOT draw, and what the foot offers.

use super::*;
use crate::widgets::chrome::rule::RULE;

/// Top to bottom: constant top, dialog slot, column header, rows, foot rule, foot.
///
/// Read off the rendered page rather than off a layout function, because the failure this
/// catches is a row drawn and then overwritten — which no arithmetic can see.
#[test]
fn the_screen_is_the_constant_top_a_dialog_row_a_list_and_a_foot() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(QueueState::default()), WIDE, TALL);

    // The rule that closes the status block, then the slot, then the header.
    let block_rule = crate::views::top::CONSTANT_ROWS + crate::panes::status_block::ROWS_FULL - 1;
    assert!(
        line(&buf, block_rule).starts_with(RULE),
        "the status block closes with a rule: {:?}",
        line(&buf, block_rule)
    );
    assert_eq!(
        line(&buf, block_rule + 1),
        "",
        "the dialog slot is blank while nothing is being said"
    );
    let header = line(&buf, header_row());
    assert!(
        // Two blanks before `Tenant`, one before `T`: ruling 5(c)'s gaps, and `T` is the one
        // column of this list that offers no sort key.
        header.starts_with("      T  Tenant"),
        "the column header follows the slot: {header:?}"
    );

    // And the foot: the rule, then the hint line, on the last two rows.
    assert!(
        line(&buf, TALL - 2).starts_with(RULE),
        "{:?}",
        line(&buf, TALL - 2)
    );
    assert!(
        line(&buf, TALL - 1).ends_with("q Quit"),
        "{:?}",
        line(&buf, TALL - 1)
    );
}

/// No frame. Chris, 2026-09-07: *"no frame around the table, valid for all views"*.
///
/// Every box-drawing glyph except the horizontal rule is forbidden — corners and verticals are
/// exactly what a box is made of, and VL §6 reserves a box for a modal or a toast.
#[test]
fn nothing_on_the_page_draws_a_box_around_the_list() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(QueueState::default()), WIDE, TALL);
    for y in 0..TALL {
        let row = line(&buf, y);
        for glyph in ['│', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴'] {
            assert!(
                !row.contains(glyph),
                "row {y} draws {glyph:?}, which is half a box: {row:?}"
            );
        }
    }
}

/// The list starts at the screen's own margin, under the dialog slot above it.
#[test]
fn the_list_starts_on_the_screens_own_margin() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(QueueState::default()), WIDE, TALL);
    let header = line(&buf, header_row());
    // The `No` column is an untitled figure field at the margin, so the first painted character
    // is the `T` column's header — three (empty) columns past the margin, plus the gap.
    assert_eq!(
        header.find('T'),
        Some((MARGIN + 3 + 1) as usize),
        "{header:?}"
    );
}

/// The foot, in the order the rules give it, and in each of the three shapes it has.
///
/// Read off `hints()` rather than off the drawn row, because the drawn row can drop hints when
/// it is narrow ([`crate::widgets::chrome::status_line::ALWAYS`]) and this is about what the
/// screen OFFERS, not about what fits.
#[test]
fn the_foot_follows_the_list_and_the_search_in_the_ruled_order() {
    let full = view(QueueState::default()).hints();
    assert_eq!(
        full,
        vec![
            ("↓↑/jk", "Navigate"),
            ("/", "Search"),
            ("f", "Filter"),
            ("o", "Op"),
            ("s", "Status"),
            ("y", "Retry"),
            ("c", "Cancel"),
            ("x", "Remove"),
            ("?", "Help"),
            ("q", "Quit"),
        ],
        "move, then find, then narrow, then act, then leave"
    );

    // `n/N` appears only while a search is on, and directly after `/ Search`.
    let searching = view(QueueState {
        search: Some(Search::On {
            term: "yml".into(),
            hit: 1,
            hits: 4,
        }),
        ..QueueState::default()
    })
    .hints();
    assert_eq!(searching[1], ("/", "Search"));
    assert_eq!(searching[2], ("n/N", "Next/Prev"));
    assert_eq!(searching.len(), full.len() + 1);

    // One row: somewhere to act, but nowhere to navigate to.
    let one = view(QueueState {
        filter: Some(Filter::On {
            term: "VISUAL-LANGUAGE".into(),
            rows: 1,
        }),
        ..QueueState::default()
    });
    assert_eq!(frames::pane(&one.state).len(), 1, "this frame must hold one row");
    assert_eq!(
        one.hints()[0],
        ("/", "Search"),
        "a list of one offers no Navigate: {:?}",
        one.hints()
    );

    // An empty list offers two hints and no more: every other key acts on a row.
    let empty = view(QueueState {
        op: Some(Op::Delete),
        ..QueueState::default()
    });
    assert_eq!(
        frames::pane(&empty.state).len(),
        0,
        "this frame must be empty"
    );
    assert_eq!(empty.hints(), vec![("?", "Help"), ("q", "Quit")]);
}

/// At eighty columns the whole hint row cannot fit, so the foot falls back to the two keys that
/// open the rest — the rule [`crate::widgets::chrome::status_line::ALWAYS`] holds for every view.
///
/// The width is the guard: 125 is where this screen is judged and 80 is where it is stressed, and
/// a fallback nobody has seen fire is a fallback nobody has checked.
#[test]
fn at_eighty_columns_the_foot_keeps_only_the_two_keys_that_open_the_rest() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let narrow = render(view(QueueState::default()), 80, 24);
    let foot = line(&narrow, 23);
    assert!(foot.ends_with("? Help   q Quit"), "{foot:?}");
    for absent in ["Navigate", "Search", "Filter", "Remove"] {
        assert!(
            !foot.contains(absent),
            "{absent:?} survived a foot too narrow for it: {foot:?}"
        );
    }

    // And at 125 it does not fire — the fallback must be a fallback, not the normal case.
    let wide = line(&render(view(QueueState::default()), WIDE, TALL), TALL - 1);
    assert!(wide.contains("x Remove"), "{wide:?}");
}

/// Ruling 5(c) (Chris, 20260912): **one blank column before every non-sortable column, never
/// before the first, and two before every sortable one.**
///
/// Measured off the column RECTS rather than off the drawn header, and the reason is the ruling
/// itself. Chris's complaint was that the spacing *"is inconsistent, one to five blanks"* — and
/// the blanks he counted are two different things the page cannot tell apart: the gap between
/// two columns, and the padding a column puts around a value narrower than itself. `Op` is six
/// columns wide because `update` is, so `add` is followed by three blanks of its own before the
/// gap even begins. A guard reading blank runs off the header would therefore be measuring the
/// sum and failing on a renderer that had obeyed the rule exactly.
///
/// The expectation is written out BY HAND from the Queue's own column list rather than asked of
/// [`gap_before`]. Asking the function would be checking it against itself; this states what the
/// ruling produces for these nine columns, which is a fact a reader can verify against Chris's
/// sentence without running anything.
///
/// The Queue is where this is measured because it is the only table holding both kinds of
/// column — `No` and `T` offer no sort key and the other seven do. On the Dashboard, whose every
/// column sorts, the same guard would pass on a renderer that gave every gap two columns
/// unconditionally.
#[test]
fn every_column_gap_is_one_blank_or_two_by_whether_the_column_sorts() {
    /// `No`, `T`, `Tenant`, `Object`, `Type`, `Op`, `Status`, `Size`, `Age` — the gap BEFORE
    /// each. The first has none; `T` is the one titled column offering no key.
    const GAPS: [u16; 9] = [0, 1, 2, 2, 2, 2, 2, 2, 2];

    let columns = frames::columns();
    assert_eq!(columns.len(), GAPS.len(), "a column was added and this list did not move");

    let table = Rect::new(0, 0, WIDE - MARGIN * 2 - crate::panes::list::GUTTER, 1);
    let rects = crate::panes::cell::table::laid_out(table, &columns.iter().collect::<Vec<_>>());

    for (at, (rect, want)) in rects.iter().zip(GAPS).enumerate() {
        if at == 0 {
            assert_eq!(rect.x, table.x, "the first column starts at the table's own left edge");
            continue;
        }
        let previous = rects[at - 1];
        let blanks = rect.x - (previous.x + previous.width);
        assert_eq!(
            blanks, want,
            "{:?} is preceded by {blanks} blank columns and the ruling gives it {want} — it {} \
             a sort key",
            columns[at].title,
            if columns[at].sort_key.is_some() { "offers" } else { "offers no" }
        );
    }
}

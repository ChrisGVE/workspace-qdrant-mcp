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
    assert!(header.contains("Tenant"), "the column header follows the slot: {header:?}");

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
    let pane = frames::pane(&QueueState::default());
    let table_width = WIDE - MARGIN * 2;
    let fitted = crate::panes::cell::fit::fit(pane.columns(), pane.rows(), table_width, 12, 1);
    assert_eq!(
        header.find('T'),
        Some((MARGIN + 3 + fitted.gap) as usize),
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

/// Every gap has the same width, and two-cell gaps appear only after all columns
/// have reached their natural widths. The Queue's Object path makes a narrow
/// frame exercise one-cell gaps while a wide frame affords two.
///
/// Measured off the column RECTS rather than off the drawn header, and the reason is the ruling
/// itself. Chris's complaint was that the spacing *"is inconsistent, one to five blanks"* — and
/// the blanks he counted are two different things the page cannot tell apart: the gap between
/// two columns, and the padding a column puts around a value narrower than itself. `Op` is six
/// columns wide because `update` is, so `add` is followed by three blanks of its own before the
/// gap even begins. A guard reading blank runs off the header would therefore be measuring the
/// sum and failing on a renderer that had obeyed the rule exactly.
///
#[test]
fn queue_gaps_and_floors_hold_across_four_terminal_widths() {
    let pane = frames::pane(&QueueState::default());
    let mut object_at_125 = 0;
    for screen_width in [80, 100, 125, 160] {
        let table = Rect::new(
            MARGIN,
            0,
            screen_width - MARGIN * 2,
            1,
        );
        let fitted = crate::panes::cell::fit::fit(pane.columns(), pane.rows(), table.width, 12, 1);
        let rects = crate::panes::cell::fit::laid_out(table, &fitted);
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();
        let frame = render(view(QueueState::default()), screen_width, TALL);
        let (header_y, header) = (0..TALL)
            .map(|y| (y, line(&frame, y)))
            .find(|(_, row)| row.contains("Object"))
            .expect("Queue header is visible");
        assert!(header.contains("Object"), "Object missing at {screen_width}: {header:?}");
        assert_eq!(rects[0].x, table.x);
        for (at, pair) in rects.windows(2).enumerate() {
            assert_eq!(pair[1].x - pair[0].right(), fitted.gap, "gap {at} at {screen_width}");
        }
        for (&at, rect) in fitted.active.iter().zip(&rects) {
            let column = &pane.columns()[at];
            let title = column.title.chars().count() as u16;
            if title > 0 {
                let start = if column.align == crate::panes::cell::Align::Right {
                    rect.right() - title
                } else {
                    rect.x
                };
                let drawn: String = (start..start + title)
                    .map(|x| frame.cell((x, header_y)).expect("header cell").symbol())
                    .collect();
                assert_eq!(drawn, column.title, "{} moved at {screen_width}", column.title);
            }
            let floor = if column.align == crate::panes::cell::Align::Right {
                column.fixed().unwrap_or(title)
            } else if let Some(natural) = column.fixed() {
                natural.min(title)
            } else {
                title.max(12)
            };
            assert!(rect.width >= floor, "{} is below its floor at {screen_width}", column.title);
            if fitted.gap == 2 {
                let natural = column.fixed().unwrap_or_else(|| {
                    pane.rows()
                        .iter()
                        .filter_map(|row| row.get(frames::cell_at(at)))
                        .map(|cell| cell.natural_width() as u16)
                        .max()
                        .unwrap_or(title)
                        .max(title)
                });
                assert!(rect.width >= natural, "{} is truncated despite wide gaps", column.title);
            }
        }
        let object = fitted.active.iter().position(|&at| at == frames::OBJECT).unwrap();
        assert_object_data(&pane, &frame, rects[object], header_y, screen_width);
        if screen_width == 80 {
            assert_eq!(fitted.gap, 1);
            assert!(rects[object].width > 12);
        }
        if screen_width == 125 {
            object_at_125 = rects[object].width;
        }
        if screen_width == 160 {
            assert_eq!(fitted.gap, 2);
            assert!(rects[object].width >= object_at_125);
        }
    }
}

fn assert_object_data(
    pane: &crate::panes::list::ListPane, frame: &Buffer, rect: Rect, header_y: u16,
    screen_width: u16,
) {
    let cell = &pane.rows()[0][frames::cell_at(frames::OBJECT)];
    let Cell::Text(object) = cell else { panic!("Object must be text") };
    let expected = pane.columns()[frames::OBJECT].elide.fit(object, rect.width);
    let drawn: String = (rect.x..rect.right())
        .map(|x| frame.cell((x, header_y + 1)).expect("Object data cell").symbol())
        .collect();
    assert_eq!(drawn.trim_end(), expected, "Object data moved at {screen_width}");
}

//! What the Dashboard's FOCUS treatment is pinned to.
//!
//! A sibling of `super` rather than more of it: the focus rulings (2026-09-07 — the selector
//! block, the data cursor, the foot that follows the live cell) doubled the file, and a test
//! file nobody can hold in one screenful is the readability defect it is meant to prevent.
//! `super`'s helpers — the fixture view, the row reader, the grid geometry — are shared rather
//! than copied; two copies of `heading_rows` is two chances to measure a different screen.

use super::*;

/// The structural guard that [`FOCUS_KEYS`] and the fixture's titles agree: every cell's
/// heading contains the letter that focuses it, that letter carries the accent, and it is the
/// only accented cell in the heading.
///
/// Written against the KEY TABLE rather than against a list of letters — a second list would
/// pass while the screen offered `p/l/s/r/a/e` and lit something else.
#[test]
fn every_cell_accents_the_key_that_focuses_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(frames::populated()), WIDE, TALL);
    let (_, cells) = heading_rows();

    for (zone, cell) in cells.iter().enumerate() {
        let key = FOCUS_KEYS[zone];
        let text = heading_text(&buf, *cell);
        let offset = text
            .chars()
            .position(|c| c.eq_ignore_ascii_case(&key))
            .unwrap_or_else(|| {
                panic!("zone {zone}'s heading {text:?} has no `{key}` for the foot to offer")
            }) as u16;
        for x in cell.x..cell.x + cell.width {
            let fg = buf.cell((x, cell.y)).expect("cell in area").style().fg;
            if x == cell.x + offset {
                assert_eq!(
                    fg,
                    Some(crate::tokens::accent()),
                    "zone {zone}: the `{key}` of {text:?} is what the foot says to press"
                );
            } else {
                assert_ne!(
                    fg,
                    Some(crate::tokens::accent()),
                    "zone {zone}: column {x} of {text:?} is accented, and it is not the key"
                );
            }
        }
    }
}

/// R6 (Chris, 2026-09-07): the focused cell's heading takes the TAB LINE's selector block, and
/// loses its key letter.
///
/// The `▌` bar is not drawn with it — two marks on one heading is noise — and the whole title
/// sits inside the block, so there is no letter left outside to accent. A cell you are already
/// on does not need telling which key gets you there.
#[test]
fn the_focused_cells_heading_is_a_selector_block_with_no_key_letter() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const RULES: usize = 3;
    let buf = render(
        view(frames::populated()).attention(Attention::Zone(RULES)),
        WIDE,
        TALL,
    );
    let (_, cells) = heading_rows();
    let cell = cells[RULES];
    let text = heading_text(&buf, cell);

    assert!(
        text.starts_with(" Rules (11) "),
        "the block pads one space each side, as the active tab does: {text:?}"
    );
    assert!(
        !text.contains(crate::widgets::chrome::zone_heading::FOCUS_BAR),
        "the block replaces the bar rather than joining it: {text:?}"
    );

    // Every cell of the title — the two padding spaces included — is filled with the selector.
    let block = " Rules (11) ".chars().count() as u16;
    for x in cell.x..cell.x + block {
        assert_eq!(
            buf.cell((x, cell.y)).expect("cell in area").style().bg,
            Some(crate::tokens::selector()),
            "column {x} of {text:?} is inside the block and must be filled"
        );
    }
    // And nothing on the heading row is accented any more.
    for x in cell.x..cell.x + cell.width {
        assert_ne!(
            buf.cell((x, cell.y)).expect("cell in area").style().fg,
            Some(crate::tokens::accent()),
            "column {x} still lights a key letter under the block: {text:?}"
        );
    }
}

/// The other five cells are untouched: each still lights the letter that focuses it, so the
/// screen still says how to get to the cell you are NOT on.
#[test]
fn the_cells_beside_the_focused_one_keep_their_key_letters() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const RULES: usize = 3;
    let buf = render(
        view(frames::populated()).attention(Attention::Zone(RULES)),
        WIDE,
        TALL,
    );
    let (_, cells) = heading_rows();

    for (zone, cell) in cells.iter().enumerate() {
        if zone == RULES {
            continue;
        }
        let text = heading_text(&buf, *cell);
        let at = cell.x
            + text
                .chars()
                .position(|c| c.eq_ignore_ascii_case(&FOCUS_KEYS[zone]))
                .expect("the key letter is drawn") as u16;
        assert_eq!(
            buf.cell((at, cell.y)).expect("cell in area").style().fg,
            Some(crate::tokens::accent()),
            "zone {zone} lost its key while zone {RULES} was focused: {text:?}"
        );
    }
}

/// The focused cell's FIRST data row takes the data cursor — the tint across the whole row and
/// the `▸` in the marker column — and the row below it does not.
///
/// Both halves matter: a tint on every row is not a cursor, and a `▸` with no tint is not the
/// treatment §3 specifies.
#[test]
fn the_focused_cells_first_row_takes_the_data_cursor_and_the_second_does_not() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const RULES: usize = 3;
    let buf = render(
        view(frames::populated()).attention(Attention::Zone(RULES)),
        WIDE,
        TALL,
    );
    let (_, cells) = heading_rows();
    let cell = cells[RULES];
    // heading, then the column header, then the first data row.
    let (first, second) = (cell.y + 2, cell.y + 3);

    assert_eq!(
        buf.cell((cell.x, first)).expect("cell in area").symbol(),
        "▸",
        "the marker sits in the row's first column: {:?}",
        heading_text(&buf, Rect { y: first, ..cell })
    );
    for x in cell.x..cell.x + cell.width {
        assert_eq!(
            buf.cell((x, first)).expect("cell in area").style().bg,
            Some(crate::tokens::cursor_bg()),
            "column {x} of the cursor row is not tinted"
        );
        assert_ne!(
            buf.cell((x, second)).expect("cell in area").style().bg,
            Some(crate::tokens::cursor_bg()),
            "column {x} of the SECOND row is tinted — that is a fill, not a cursor"
        );
    }
    assert_ne!(
        buf.cell((cell.x, second)).expect("cell in area").symbol(),
        "▸",
        "only one row carries the mark"
    );
}

/// A cell with nothing in it highlights nothing, and the foot does not grow.
#[test]
fn focusing_an_empty_cell_moves_no_cursor_and_adds_no_hint() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const SCRATCHPAD: usize = 2;
    let dashboard = view(frames::populated()).attention(Attention::Zone(SCRATCHPAD));
    assert_eq!(dashboard.hints(), vec![("?", "Help"), ("q", "Quit")]);

    let buf = render(
        view(frames::populated()).attention(Attention::Zone(SCRATCHPAD)),
        WIDE,
        TALL,
    );
    let (_, cells) = heading_rows();
    let cell = cells[SCRATCHPAD];
    for x in cell.x..cell.x + cell.width {
        assert_ne!(
            buf.cell((x, cell.y + 2)).expect("cell in area").style().bg,
            Some(crate::tokens::cursor_bg()),
            "an empty projection has no row to put a cursor on"
        );
    }
    assert!(line(&buf, TALL - 1).ends_with("? Help   q Quit"));
}

/// A cell holding exactly ONE row offers `Enter`, and no navigation — there is nowhere to go.
#[test]
fn focusing_a_single_row_cell_offers_enter_and_no_navigation() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const LIBRARIES: usize = 1;
    assert_eq!(
        frames::populated()[LIBRARIES].table().len(),
        1,
        "this guard is about a one-row cell; the fixture has to be one"
    );

    let dashboard = view(frames::populated()).attention(Attention::Zone(LIBRARIES));
    assert_eq!(
        dashboard.hints(),
        vec![("Enter", "Detail"), ("?", "Help"), ("q", "Quit")]
    );

    let foot = line(
        &render(
            view(frames::populated()).attention(Attention::Zone(LIBRARIES)),
            WIDE,
            TALL,
        ),
        TALL - 1,
    );
    assert!(foot.ends_with("Enter Detail   ? Help   q Quit"), "{foot:?}");
    assert!(!foot.contains("Navigate"), "{foot:?}");
}

/// More than one row, and the foot offers the cursor keys too.
#[test]
fn focusing_a_many_row_cell_offers_navigation_and_enter() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const RULES: usize = 3;
    assert!(frames::populated()[RULES].table().len() > 1);

    let dashboard = view(frames::populated()).attention(Attention::Zone(RULES));
    assert_eq!(
        dashboard.hints(),
        vec![
            (NAVIGATE_KEYS, "Navigate"),
            ("Enter", "Detail"),
            ("?", "Help"),
            ("q", "Quit")
        ]
    );

    let foot = line(
        &render(
            view(frames::populated()).attention(Attention::Zone(RULES)),
            WIDE,
            TALL,
        ),
        TALL - 1,
    );
    assert!(
        foot.ends_with("↓↑/jk Navigate   Enter Detail   ? Help   q Quit"),
        "{foot:?}"
    );
}

/// VL §6: the page under a modal drops every highlight. No cell key is lit anywhere in the
/// grid, and not one row moves — the same pairing `views::shell` pins for the tab bar.
#[test]
fn a_modal_mutes_every_cell_key_without_moving_a_row() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let live = render(view(frames::populated()), WIDE, TALL);
    let under = render(view(frames::populated()).under_modal(true), WIDE, TALL);
    let (first, _) = heading_rows();

    let accented = |buf: &Buffer| {
        (first..TALL - 1)
            .flat_map(|y| (0..WIDE).map(move |x| (x, y)))
            .filter(|(x, y)| {
                buf.cell((*x, *y)).expect("cell in area").style().fg
                    == Some(crate::tokens::accent())
            })
            .count()
    };
    assert_eq!(
        accented(&live),
        CELLS,
        "one accented letter per cell, or this guard is checking nothing"
    );
    assert_eq!(accented(&under), 0, "a modal leaves no lit key behind it");

    for y in 0..TALL {
        assert_eq!(line(&live, y), line(&under, y), "row {y} moved under a modal");
    }
}

/// R4 (Chris, 2026-09-07): Scratchpad and Rules list their ITEMS with the scope beside them —
/// *"their scope should be inverted with their respective Notes and Rules"*.
///
/// The header row is read off the render, because the ruling is about what a reader sees: a
/// check against the column titles in `frames` would pass on a table whose header was never
/// drawn.
#[test]
fn the_rules_cell_lists_rules_with_their_scope_and_carries_no_queue() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const RULES_ZONE: usize = 3;
    let buf = render(view(frames::populated()), WIDE, TALL);
    let (_, cells) = heading_rows();
    let cell = cells[RULES_ZONE];

    let header = heading_text(&buf, Rect { y: cell.y + 1, ..cell });
    assert!(header.trim_start().starts_with("Rule"), "{header:?}");
    assert!(header.contains("Scope"), "{header:?}");
    for gone in ["Queue", "Notes"] {
        assert!(!header.contains(gone), "{gone:?} survives on the Rules cell: {header:?}");
    }

    // The first data row is a rule NAME and the scope beside it, not a scope and a count.
    let first = heading_text(&buf, Rect { y: cell.y + 2, ..cell });
    assert!(first.contains(frames::RULES[0].rule), "{first:?}");
    assert!(first.contains("global"), "{first:?}");
}

/// The heading counts the STORE, and the tail counts what the cell could not draw. They are
/// different numbers on purpose — the same split `Projects (29)` has always had.
#[test]
fn the_rules_heading_counts_the_store_and_the_tail_counts_the_overflow() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    assert!(
        frames::RULES.len() < frames::RULE_TOTAL,
        "the fixture must not be able to draw the whole store, or this guard checks nothing"
    );

    let joined: String = (0..TALL)
        .map(|y| line(&render(view(frames::populated()), WIDE, TALL), y))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        joined.contains(&format!("Rules ({})", frames::RULE_TOTAL)),
        "the heading is the size of the projection: {joined}"
    );
}

/// Scratchpad shows `No data` because there were no notes to read, not because the cell is
/// broken. Pinned so that the day a note appears in this frame, it is a deliberate act.
#[test]
fn the_scratchpad_cell_is_empty_because_nothing_real_was_found_to_put_in_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const SCRATCHPAD_ZONE: usize = 2;
    assert!(frames::populated()[SCRATCHPAD_ZONE].table().is_empty());

    let buf = render(view(frames::populated()), WIDE, TALL);
    let (_, cells) = heading_rows();
    let cell = cells[SCRATCHPAD_ZONE];
    assert!(heading_text(&buf, cell).starts_with("Scratchpad (0)"));
    assert!(
        heading_text(&buf, Rect { y: cell.y + 2, ..cell })
            .trim_start()
            .starts_with(crate::panes::cell::EMPTY),
        "an empty projection says so rather than showing a blank cell"
    );
}

/// R5 (Chris, 2026-09-07): `Pts` is gone from Projects and Libraries, and the width it held
/// goes to the flex `Name` column.
///
/// The `Name` width is asserted against the ARITHMETIC, spelled out here, rather than against
/// another render: comparing two renders of the same code would agree with any width at all,
/// and comparing against a remembered number would need someone to remember it.
///
/// At 125 columns the screen insets by [`crate::widgets::chrome::MARGIN`] on each side (121),
/// splits into two cells with a three-column gap (59 each). The table reserves
/// [`crate::panes::cell::table::GUTTER`] for its cursor marker, leaving 57. It then spends `Bch` 3 +
/// `Files` 5 + `Queue` 9 = 17 on fixed columns and three single-column gaps between its four
/// columns, leaving 57 − 17 − 3 = 37 for `Name`. With `Pts` it was 57 − 20 − 4 = 33.
#[test]
fn dropping_pts_gives_its_columns_to_the_name() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const NAME_WIDTH: u16 = 37;
    /// What the same arithmetic gave while `Pts` was there: 57 − 20 − 4.
    const NAME_WIDTH_WITH_PTS: u16 = 33;

    let buf = render(view(frames::populated()), WIDE, TALL);
    let (_, cells) = heading_rows();
    let projects = cells[0];

    // `Bch` is right-aligned in the column after `Name`, and its column is exactly as wide as
    // its title — so the first `B` sits one gap column past the end of `Name`.
    let header = heading_text(&buf, Rect { y: projects.y + 1, ..projects });
    let bch = header
        .chars()
        .position(|c| c == 'B')
        .expect("the Bch header is drawn") as u16;
    assert_eq!(
        bch,
        crate::panes::cell::table::GUTTER + NAME_WIDTH + 1,
        "the Name column is {NAME_WIDTH} wide with one gap after it: {header:?}"
    );
    assert!(
        bch - 1 - crate::panes::cell::table::GUTTER > NAME_WIDTH_WITH_PTS,
        "the measured Name column must be wider than the {NAME_WIDTH_WITH_PTS} it had while \
         `Pts` was drawn: {header:?}"
    );

    // And no cell on the whole screen says `Pts` any more.
    let joined: String = (0..TALL).map(|y| line(&buf, y)).collect::<Vec<_>>().join("\n");
    assert!(!joined.contains("Pts"), "a Pts header survives: {joined}");
}

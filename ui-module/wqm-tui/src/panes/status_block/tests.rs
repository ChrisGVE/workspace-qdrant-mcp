//! What the status block is pinned to.
//!
//! A sibling file rather than an inline module: the pane and its guards together ran past this
//! crate's 500-line file limit, and the guards are the half that reads on its own — each one
//! names a rule from the module docs above and compares the render against the CONSTANT that
//! rule is stated with, never against a second measurement.

use super::*;
use crate::widgets::chrome::test_support::Restore;
use std::time::Duration;

const WIDE: u16 = 125;

/// A terminal wide enough that the grid would spread past its cap if nothing stopped it.
const WIDE_SCREEN: u16 = 200;

fn block() -> StatusBlock {
    StatusBlock::new(
        Health::Healthy,
        "v0.2.0",
        Freshness::new(Duration::from_secs(4), Duration::from_secs(60)),
        [Health::Healthy; ENTRY_LABELS.len()],
        Queue {
            pending: 0,
            in_progress: 0,
            failed: 0,
            health: Health::Healthy,
        },
    )
}

fn render(block: StatusBlock, width: u16, height: u16) -> Buffer {
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    block.render(area, &mut buf);
    buf
}

fn row(buf: &Buffer, y: u16) -> String {
    (0..buf.area.width)
        .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
        .collect()
}

fn style_of(buf: &Buffer, y: u16, needle: &str) -> Style {
    let line = row(buf, y);
    let byte = line
        .find(needle)
        .unwrap_or_else(|| panic!("{needle:?} not on row {y}: {line:?}"));
    let x = line[..byte].chars().count() as u16;
    buf.cell((x, y)).expect("cell in area").style()
}

/// The style of count `i` on the queue row, reached by ARITHMETIC rather than by searching
/// for its digits.
///
/// A count is right-aligned so its last character sits exactly on the field's right edge,
/// which the constants put one cell left of column `i + 1`'s glyph position. Searching for the
/// digits instead would land inside a neighbouring number — `find("2")` hits the `2` in `123`,
/// a different span with a different hue, and the test measures the wrong cell while looking
/// perfectly correct.
/// The column a needle starts on, counted in CHARACTERS — the mistake this crate has made
/// once already.
fn column_of(buf: &Buffer, y: u16, needle: &str) -> u16 {
    let line = row(buf, y);
    let byte = line
        .find(needle)
        .unwrap_or_else(|| panic!("{needle:?} not on row {y}: {line:?}"));
    line[..byte].chars().count() as u16
}

fn count_style(buf: &Buffer, width: u16, i: usize) -> Style {
    let x = MARGIN + column_width(width) * (i as u16 + 1);
    buf.cell((x, QUEUE_ROW)).expect("cell in area").style()
}

/// The INTERIM roll-up, checked against the rule as stated rather than against §7's.
///
/// §7 makes an unreachable daemon roll up as *offline* — the liveness master's own word.
/// Chris overrode that for this block (20260906): *"the daemon down is not equal to the
/// other services being down, the overall service is however obviously degraded"*. The two
/// rules coexist deliberately, so this test pins the difference rather than hiding it.
#[test]
fn a_daemon_that_is_down_makes_the_service_degraded_and_never_offline() {
    assert_eq!(rollup(Health::Offline, &[Health::Healthy; 3]), Health::Degraded);
    assert_eq!(rollup(Health::Degraded, &[Health::Healthy; 3]), Health::Degraded);
    assert_eq!(
        rollup(Health::Healthy, &[Health::Healthy, Health::Offline, Health::Healthy]),
        Health::Degraded
    );
    assert_eq!(
        rollup(Health::Healthy, &[Health::Healthy, Health::Degraded, Health::Healthy]),
        Health::Degraded
    );
    assert_eq!(rollup(Health::Healthy, &[Health::Healthy; 3]), Health::Healthy);
}

/// §4's two halves, one frame each: quiet while healthy, pushed forward while not.
#[test]
fn the_title_takes_the_state_hue_only_when_there_is_something_to_see() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let healthy = render(block(), WIDE, ROWS_FULL);
    assert_eq!(
        style_of(&healthy, 0, "Service status").fg,
        Some(tokens::normal()),
        "a healthy status line is quiet — only the glyph carries colour (§4)"
    );

    let degraded = render(block().overall(Health::Degraded), WIDE, ROWS_FULL);
    assert_eq!(
        style_of(&degraded, 0, "Service status").fg,
        Some(Health::Degraded.color()),
        "the must-see rule: bold AND the state colour"
    );
    for buf in [&healthy, &degraded] {
        assert!(
            style_of(buf, 0, "Service status")
                .add_modifier
                .contains(Modifier::BOLD),
            "the title is bold either way"
        );
    }
}

/// A count of nothing is not news, so it recedes to the same rung as its own label.
#[test]
fn a_queue_count_of_zero_is_muted_and_a_count_of_anything_is_not() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let idle = render(block(), WIDE, ROWS_FULL);
    for (i, label) in QUEUE_LABELS.iter().enumerate() {
        assert_eq!(
            count_style(&idle, WIDE, i).fg,
            Some(tokens::muted()),
            "a {label} count of zero is not news"
        );
    }

    let busy = render(
        block().queue(Queue {
            pending: 123,
            in_progress: 45,
            failed: 67,
            health: Health::Degraded,
        }),
        WIDE,
        ROWS_FULL,
    );
    assert_eq!(
        count_style(&busy, WIDE, 0).fg,
        Some(tokens::degraded()),
        "work waiting is the warning hue"
    );
    // A role of its own since 20260907, resolved by the categorical tier: a flavour's
    // `sapphire` where there is one, the selector's `info` on the eleven themes with nothing
    // spare. The guard names the ROLE rather than either field, because which field answers is
    // exactly what this pane must not know.
    assert_eq!(
        count_style(&busy, WIDE, 1).fg,
        Some(tokens::in_flight()),
        "work moving carries the in-flight role"
    );
    assert_eq!(
        count_style(&busy, WIDE, 2).fg,
        Some(tokens::offline()),
        "work that failed is the error hue"
    );
}

/// The four columns are equal, and each starts where its own column does.
#[test]
fn the_four_entries_stand_on_an_even_grid() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(block(), WIDE, ROWS_FULL);
    let line = row(&buf, ENTRIES_ROW);
    let column = (WIDE - MARGIN * 2) / ENTRY_LABELS.len() as u16;
    for (i, label) in ENTRY_LABELS.iter().enumerate() {
        let byte = line.find(label).unwrap_or_else(|| panic!("{label} missing: {line:?}"));
        let at = line[..byte].chars().count() as u16;
        // glyph + one space, at the column's own origin, inset by the screen margin.
        assert_eq!(
            at,
            MARGIN + column * i as u16 + 2,
            "{label} is off its column"
        );
    }
}

/// Row 3's labels stand on row 2's columns — the whole point of the change (Chris, 20260906,
/// after seeing the block in the pantry).
///
/// Both rows are checked against the SAME stated expression rather than against each other: a
/// test that read row 2's label positions out of the buffer and compared row 3's to them would
/// pass just as happily if both rows had drifted together.
#[test]
fn the_queue_labels_stand_on_the_columns_the_entry_labels_stand_on() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(block(), WIDE, ROWS_FULL);
    let column = (WIDE - MARGIN * 2) / ENTRY_LABELS.len() as u16;

    for (i, label) in QUEUE_LABELS.iter().enumerate() {
        // Column i + 1: the queue's own glyph and word occupy column 0.
        let want = MARGIN + column * (i as u16 + 1) + 2;
        assert_eq!(column_of(&buf, QUEUE_ROW, label), want, "{label} is off its column");
        assert_eq!(
            column_of(&buf, ENTRIES_ROW, ENTRY_LABELS[i + 1]),
            want,
            "{} moved, so the row below it is aligned to nothing",
            ENTRY_LABELS[i + 1]
        );
    }
}

/// A count is right-aligned into the slack of the column to its LEFT, its units digit landing
/// **on** its own column's glyph position — so the digits grow away from the label they belong
/// to, the label never moves, and the count's last cell sits directly under the disc above it.
///
/// Chris, 20260906: *"given the symbol, right-align the number on the left"* — the symbol's
/// slot is where the number ends, not the cell before it. This replaces the earlier reading
/// (right edge at the glyph column minus one), which left two blanks before the label and made
/// the count read as belonging to the column on its left.
#[test]
fn a_count_ends_under_its_own_columns_glyph() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(
        block().queue(Queue {
            pending: 1_240,
            in_progress: 8,
            failed: 3,
            health: Health::Degraded,
        }),
        WIDE,
        ROWS_FULL,
    );
    let column = (WIDE - MARGIN * 2) / ENTRY_LABELS.len() as u16;
    let line = row(&buf, QUEUE_ROW);
    let chars: Vec<char> = line.chars().collect();

    for (i, drawn) in ["1 240", "8", "3"].iter().enumerate() {
        let glyph_column = MARGIN + column * (i as u16 + 1);
        let end = glyph_column as usize;
        let start = end + 1 - drawn.chars().count();
        let read: String = chars[start..=end].iter().collect();
        assert_eq!(
            &read,
            drawn,
            "count {i} does not end on its column's glyph position: {line:?}"
        );
        assert_eq!(
            chars[end + 1],
            ' ',
            "exactly one blank stands between a count and the label it belongs to"
        );
    }
}

/// Chris's standing rule for an isolated number: grouped in threes, with a PLAIN space.
///
/// Plain, not thin or narrow: those are not width-1 in every terminal, and a separator that
/// changes width changes the column a right-aligned number ends on.
#[test]
fn a_count_is_grouped_in_threes_with_a_plain_space() {
    assert_eq!(grouped(0), "0");
    assert_eq!(grouped(999), "999");
    assert_eq!(grouped(1_240), "1 240");
    assert_eq!(grouped(9_999_999), "9 999 999");
    assert_eq!(
        grouped(9_999_999).chars().count() as u16,
        COUNT_WIDTH,
        "COUNT_WIDTH is the width of the widest count the field is sized for"
    );
    assert!(
        !grouped(1_240).contains('\u{202f}') && !grouped(1_240).contains('\u{2009}'),
        "the separator is a plain space, not a narrow or thin one"
    );
}

/// The reason `MIN_COLUMN` had to grow: at the narrowest aligned width, a full-width count
/// must still clear the label of the column it spills into.
#[test]
fn the_widest_count_still_clears_the_previous_columns_label() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let exact = MARGIN * 2 + MIN_COLUMN * ENTRY_LABELS.len() as u16;
    let buf = render(
        block().queue(Queue {
            pending: 9_999_999,
            in_progress: 9_999_999,
            failed: 9_999_999,
            health: Health::Degraded,
        }),
        exact,
        ROWS_FULL,
    );
    let chars: Vec<char> = row(&buf, QUEUE_ROW).chars().collect();

    for (i, _) in QUEUE_LABELS.iter().enumerate() {
        // The cell immediately left of the widest count must be blank, or the count has run
        // into the word before it.
        let start = (MARGIN + MIN_COLUMN * (i as u16 + 1) + 1 - COUNT_WIDTH) as usize;
        assert_eq!(
            chars[start - 1],
            ' ',
            "count {i} touches the label to its left at the narrowest aligned width: {:?}",
            row(&buf, QUEUE_ROW)
        );
    }
}

/// At the narrowest width the squeeze rule calls aligned, every column still holds its whole
/// contents — which is what [`MIN_COLUMN`] claims and the only way to check the claim.
///
/// The boundary assertions alone would be worthless: `columns_align` is *computed from*
/// `MIN_COLUMN`, so a wrong `MIN_COLUMN` moves the boundary and the arithmetic agrees with
/// itself. The render is the independent measurement — one column too narrow and ratatui
/// truncates `vector db` inside its own cell, silently, exactly as it would on a real screen.
#[test]
fn the_narrowest_aligned_width_still_holds_every_label_whole() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let exact = MARGIN * 2 + MIN_COLUMN * ENTRY_LABELS.len() as u16;
    assert!(columns_align(exact), "the stated minimum must itself fit");
    assert!(!columns_align(exact - 1), "one column below it must not");

    let buf = render(block(), exact, ROWS_FULL);
    let line: Vec<char> = row(&buf, ENTRIES_ROW).chars().collect();
    let column = (exact - MARGIN * 2) / ENTRY_LABELS.len() as u16;
    for (i, label) in ENTRY_LABELS.iter().enumerate() {
        let start = (MARGIN + column * i as u16) as usize;
        let cell: String = line[start..start + column as usize].iter().collect();
        assert_eq!(
            cell.trim_end(),
            format!("{} {label}", Health::Healthy.glyph()),
            "column {i} cannot hold `glyph + space + {label}` at the stated minimum width"
        );
    }
}

/// Chris, 20260906: on a very wide terminal the four columns spread until they are hard to
/// read. They stop at [`MAX_COLUMN`] and pack left; the width past them stays empty.
///
/// Both ends are asserted against the STATED constants — the cap, and the storyboard's own
/// width, which is below the cap and must therefore still divide evenly.
#[test]
fn the_columns_stop_stretching_at_the_stated_cap() {
    assert_eq!(column_width(WIDE_SCREEN), MAX_COLUMN, "a wide screen is capped");
    assert_eq!(
        column_width(WIDE),
        (WIDE - MARGIN * 2) / ENTRY_LABELS.len() as u16,
        "the storyboard's own width is under the cap and divides as it always did"
    );
}

/// Packed left, and genuinely empty to the right — otherwise the cap has only moved the
/// spreading somewhere the eye still has to travel.
#[test]
fn a_capped_grid_packs_left_and_leaves_the_rest_of_the_row_empty() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(block(), WIDE_SCREEN, ROWS_FULL);
    for (i, label) in ENTRY_LABELS.iter().enumerate() {
        assert_eq!(
            column_of(&buf, ENTRIES_ROW, label),
            MARGIN + MAX_COLUMN * i as u16 + 2,
            "{label} is not packed onto the capped grid"
        );
    }

    // Everything right of the last column's content is untouched.
    let last = ENTRY_LABELS.last().expect("four labels");
    let end = MARGIN + MAX_COLUMN * (ENTRY_LABELS.len() as u16 - 1) + 2 + last.len() as u16;
    let line: Vec<char> = row(&buf, ENTRIES_ROW).chars().collect();
    assert!(
        line[end as usize..].iter().all(|c| *c == ' '),
        "the grid spread past its cap: {:?}",
        row(&buf, ENTRIES_ROW)
    );
}

/// The cap governs the GRID, never the row. Row 1's age is a property of the screen, so it
/// stays flush with the screen's own right margin whatever the columns are doing.
#[test]
fn the_freshness_stays_flush_right_of_the_full_width_when_the_grid_is_capped() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(block(), WIDE_SCREEN, ROWS_FULL);
    let line = row(&buf, STATUS_ROW);
    let last = column_of(&buf, STATUS_ROW, "updated 4s ago") + "updated 4s ago".len() as u16 - 1;
    assert_eq!(
        last,
        WIDE_SCREEN - MARGIN - 1,
        "the age follows the screen, not the grid: {line:?}"
    );
}

/// Both triggers, each on its own, and the shape that comes out of each.
#[test]
fn either_a_narrow_screen_or_a_short_one_collapses_the_block() {
    let tall = Rect::new(0, 0, 125, 34);
    assert_eq!(Collapse::decide(tall, MIN_CONTENT_ROWS), Collapse::Full);

    let narrow = Rect::new(0, 0, MARGIN * 2 + MIN_COLUMN * 4 - 1, 34);
    assert_eq!(
        Collapse::decide(narrow, MIN_CONTENT_ROWS),
        Collapse::Collapsed,
        "columns that cannot align are the width trigger"
    );

    let short = Rect::new(0, 0, 125, ROWS_FULL + MIN_CONTENT_ROWS - 1);
    assert_eq!(
        Collapse::decide(short, MIN_CONTENT_ROWS),
        Collapse::Collapsed,
        "a content region squeezed under its floor is the height trigger"
    );

    assert_eq!(StatusBlock::rows_for(tall, MIN_CONTENT_ROWS), ROWS_FULL);
    assert_eq!(StatusBlock::rows_for(short, MIN_CONTENT_ROWS), ROWS_COLLAPSED);
}

/// Collapsed is row 1 and the rule, and nothing in between.
#[test]
fn a_collapsed_block_keeps_the_rollup_and_drops_the_detail() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(block(), WIDE, ROWS_COLLAPSED);
    assert!(row(&buf, 0).contains("Service status"), "the roll-up survives");
    let closing = row(&buf, ROWS_COLLAPSED - 1);
    assert!(
        closing.starts_with(crate::widgets::chrome::rule::RULE),
        "the block still closes with its rule: {closing:?}"
    );
    for label in ENTRY_LABELS {
        assert!(
            !row(&buf, 0).contains(label),
            "{label} must not survive a collapse"
        );
    }
}

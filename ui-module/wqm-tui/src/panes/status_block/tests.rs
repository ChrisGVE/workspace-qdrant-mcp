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

/// The style of the first cell AFTER a label — how a count is reached without spelling the
/// count. `find("2")` would land inside `123`, which is a different span with a different
/// hue, and the test would be measuring the wrong cell while looking correct.
fn style_after(buf: &Buffer, y: u16, label: &str) -> Style {
    let line = row(buf, y);
    let byte = line
        .find(label)
        .unwrap_or_else(|| panic!("{label:?} not on row {y}: {line:?}"));
    let x = (line[..byte].chars().count() + label.chars().count()) as u16;
    buf.cell((x, y)).expect("cell in area").style()
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
    for label in [PENDING, IN_PROGRESS, FAILED] {
        assert_eq!(
            style_after(&idle, QUEUE_ROW, label).fg,
            Some(tokens::muted()),
            "{label}0 is not news"
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
        style_after(&busy, QUEUE_ROW, PENDING).fg,
        Some(tokens::degraded()),
        "work waiting is the warning hue"
    );
    assert_eq!(
        style_after(&busy, QUEUE_ROW, IN_PROGRESS).fg,
        Some(tokens::secondary()),
        "work moving is `secondary` — NOT `info`, which §3 reserves to the selector"
    );
    assert_eq!(
        style_after(&busy, QUEUE_ROW, FAILED).fg,
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

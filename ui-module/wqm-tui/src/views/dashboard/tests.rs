//! What the Dashboard is pinned to.

use super::*;
use crate::panes::cell::{Cell, CellTable, Column};
use crate::panes::status_block::Queue;
use crate::widgets::chrome::rule::RULE;
use crate::widgets::chrome::test_support::Restore;
use crate::widgets::chrome::{Freshness, MARGIN};
use ratatui::buffer::Buffer;
use std::time::Duration;

const WIDE: u16 = 125;
const TALL: u16 = 34;

fn view(cells: Vec<CellPane>) -> Dashboard {
    let entries = [Health::Healthy, Health::Degraded, Health::Healthy, Health::Healthy];
    let overall = overall(entries[0], &entries[1..]);
    Dashboard::new(
        cells,
        StatusBlock::new(
            overall,
            "v0.2.0",
            Freshness::new(Duration::from_secs(4), Duration::from_secs(60)),
            entries,
            Queue { pending: 11_236, in_progress: 4, failed: 3, health: Health::Degraded },
        ),
        overall,
    )
}

fn render(view: Dashboard, width: u16, height: u16) -> Buffer {
    let area = Rect::new(0, 0, width, height);
    let mut buf = Buffer::empty(area);
    view.render(area, &mut buf);
    buf
}

fn line(buf: &Buffer, y: u16) -> String {
    (0..buf.area.width)
        .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
        .collect::<String>()
        .trim_end()
        .to_string()
}

/// Chris's words, in the geometry: two across, three down, row-major.
#[test]
fn the_grid_is_two_across_and_three_down_in_row_major_order() {
    let area = Rect::new(0, 0, 121, 27);
    let (cells, rules) = grid(area);

    assert_eq!(cells.len(), CELLS);
    assert_eq!(rules.len(), GRID_ROWS - 1);
    for r in 0..GRID_ROWS {
        let (left, right) = (cells[r * COLUMNS], cells[r * COLUMNS + 1]);
        assert_eq!(left.y, right.y, "row {r}'s two cells must sit on one line");
        assert!(left.x < right.x, "row-major: the left cell comes first");
        assert_eq!(left.height, right.height);
    }
    for r in 1..GRID_ROWS {
        assert!(
            cells[r * COLUMNS].y > cells[(r - 1) * COLUMNS].y,
            "row {r} sits below row {}",
            r - 1
        );
    }
}

/// Equal thirds, with the remainder to the FIRST row — the two most populated projections.
///
/// Asserted against the arithmetic the module doc states, not against a rendered measurement:
/// a test reading the heights back out of `grid` would agree with any split whatsoever.
#[test]
fn the_rows_are_equal_thirds_and_the_remainder_goes_to_the_first() {
    for height in [27, 28, 29, 30] {
        let (cells, _) = grid(Rect::new(0, 0, 121, height));
        let heights: Vec<u16> = (0..GRID_ROWS).map(|r| cells[r * COLUMNS].height).collect();
        let body = height - (GRID_ROWS as u16 - 1);
        assert_eq!(
            heights[0],
            body / 3 + body % 3,
            "at {height} rows the first band takes the remainder"
        );
        assert_eq!(heights[1], body / 3);
        assert_eq!(heights[2], body / 3);
        assert_eq!(heights.iter().sum::<u16>() + 2, height, "the grid fills its area exactly");
    }
}

/// The two columns are divided by whitespace and nothing else — v0.1 has no vertical rule and
/// VL §6 leaves it an open micro-choice, so the frame must not quietly answer it.
///
/// The rows carrying a full-width rule are found by READING them back, not by re-deriving the
/// layout: a test that recomputed the geometry would be checking its own copy of the
/// arithmetic against the renderer's, and would agree with a renderer that had drifted.
#[test]
fn nothing_is_drawn_between_the_two_columns() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(frames::populated()), WIDE, TALL);
    let full = RULE.repeat(WIDE as usize);

    // The gap is between the two cells of a row; its columns are known from the grid's own fn.
    let (cells, _) = grid(crate::widgets::chrome::inset(Rect::new(0, 0, WIDE, 27)));
    let gap = cells[0].x + cells[0].width..cells[1].x;
    assert!(!gap.is_empty(), "there must be a gap to check");

    // Only the GRID's rows. The constant top spans the full width by design — its status
    // block's four columns march straight through where the gap would be.
    let first = crate::views::top::CONSTANT_ROWS + crate::panes::status_block::ROWS_FULL;
    let mut checked = 0;
    for y in first..TALL - 1 {
        if line(&buf, y) == full {
            continue;
        }
        for x in gap.clone() {
            assert_eq!(
                buf.cell((x, y)).expect("cell in area").symbol(),
                " ",
                "something is drawn in the column gap at {x},{y}"
            );
        }
        checked += 1;
    }
    assert!(checked >= 20, "only {checked} grid rows were checked");
}

/// R3 (Chris, 2026-09-07): a row rule BREAKS over the column gap, and the frame rules do not.
///
/// *"while the top separation and the bottom separation lines are continuous, the lines
/// separating the columns should be discontinued with a blank in between marking the limits of
/// the two columns"*.
///
/// The two kinds are counted as well as inspected: exactly two rows on the screen are a rule
/// from edge to edge, and they are the constant top's, not the grid's.
#[test]
fn a_row_rule_breaks_over_the_column_gap_and_the_frame_rules_do_not() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(frames::populated()), WIDE, TALL);
    let full = RULE.repeat(WIDE as usize);

    // The full-width rules are the constant top's two — the top rule and the one that closes
    // the status block. The grid's own two are no longer among them.
    let continuous: Vec<u16> = (0..TALL).filter(|y| line(&buf, *y) == full).collect();
    assert_eq!(
        continuous.len(),
        2,
        "the top rule and the block's closing rule, and nothing else: {continuous:?}"
    );
    let block_rule = crate::views::top::CONSTANT_ROWS + crate::panes::status_block::ROWS_FULL - 1;
    assert!(
        continuous.contains(&block_rule),
        "the status block still closes with a rule across the full width: {continuous:?}"
    );

    // The grid's own rule rows, located from the layout the renderer uses rather than from a
    // remembered y — a fixture of numbers would keep passing after the grid moved.
    let (_, rules) = grid(crate::widgets::chrome::inset(Rect::new(
        0,
        crate::views::top::CONSTANT_ROWS + crate::panes::status_block::ROWS_FULL,
        WIDE,
        TALL - 1 - (crate::views::top::CONSTANT_ROWS + crate::panes::status_block::ROWS_FULL),
    )));
    let (_, cells) = heading_rows();
    let gap = cells[0].x + cells[0].width..cells[1].x;
    assert_eq!(rules.len(), GRID_ROWS - 1);
    assert!(!gap.is_empty(), "there must be a gap for the rule to break over");

    for rule in &rules {
        let drawn = line(&buf, rule.y);
        assert_ne!(drawn, full, "row {} runs straight through the gap", rule.y);
        for x in gap.clone() {
            assert_eq!(
                buf.cell((x, rule.y)).expect("cell in area").symbol(),
                " ",
                "the gap column {x} of row {} carries a rule glyph",
                rule.y
            );
        }
        // Every column a cell occupies IS ruled, margins included: the break is the gap and
        // nothing else. Checked at both ends of both segments, which is where an off-by-one in
        // the arithmetic would land.
        for x in [0, cells[0].x, cells[0].x + cells[0].width - 1, cells[1].x, WIDE - 1] {
            assert_eq!(
                buf.cell((x, rule.y)).expect("cell in area").symbol(),
                RULE,
                "column {x} of row {} is under a cell and should be ruled: {drawn:?}",
                rule.y
            );
        }
    }
}

/// The segments are derived from the cells, so the arithmetic is checked without a render.
#[test]
fn the_two_rule_segments_cover_every_column_except_the_gap() {
    let screen = Rect::new(0, 0, WIDE, 1);
    let (cells, rules) = grid(crate::widgets::chrome::inset(Rect::new(0, 0, WIDE, 27)));
    let [left, right] = rule_segments(screen, rules[0], cells[0], cells[1]);

    assert_eq!(left.x, screen.x, "the left segment starts at the page edge");
    assert_eq!(
        left.x + left.width,
        cells[0].x + cells[0].width,
        "and stops where the left cell does"
    );
    assert_eq!(right.x, cells[1].x, "the right segment starts where the right cell does");
    assert_eq!(
        right.x + right.width,
        screen.x + screen.width,
        "and runs to the page edge"
    );
    assert_eq!(
        WIDE - left.width - right.width,
        cells[1].x - (cells[0].x + cells[0].width),
        "what the two segments leave uncovered is exactly the gap"
    );
}

/// SYS-3, in the chrome: the dot at the foot and the block at the head answer from one value.
#[test]
fn the_bottom_rollup_agrees_with_the_status_block_above_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(frames::populated()), WIDE, TALL);
    let block = buf.cell((MARGIN, crate::views::top::CONSTANT_ROWS)).expect("cell");
    let foot = buf.cell((MARGIN, TALL - 1)).expect("cell");
    assert_eq!(
        block.style().fg,
        foot.style().fg,
        "a green dot under a degraded block is not a screen this view may produce"
    );
    assert!(line(&buf, TALL - 1).contains("degraded"));
}

/// R2 (Chris, 2026-09-07): with no cell focused the foot offers exactly TWO hints.
///
/// `p/l/s/r/a/e Focus cell` is gone because the headings carry the letters now, and
/// `Enter Detail` is gone because with nothing focused there is nothing to open. Counted
/// structurally as well as read off the screen: "the row does not contain `Focus cell`" would
/// also pass on a foot that had grown three new hints.
#[test]
fn the_default_foot_offers_two_hints_and_neither_is_a_key_the_headings_already_carry() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let hints = view(frames::populated()).hints();
    assert_eq!(
        hints,
        vec![("?", "Help"), ("q", "Quit")],
        "the default foot is two hints, in this order"
    );

    let foot = line(&render(view(frames::populated()), WIDE, TALL), TALL - 1);
    assert!(foot.ends_with("? Help   q Quit"), "{foot:?}");
    for absent in ["Focus cell", "Detail", "Global", "Navigate"] {
        assert!(
            !foot.contains(absent),
            "{absent:?} is on a foot that should carry two hints: {foot:?}"
        );
    }
}

/// §18: the grid keeps its shape and the cells lose rows. A cramped terminal must not reflow
/// the six into some other arrangement.
#[test]
fn a_cramped_screen_keeps_the_grid_and_takes_the_rows_from_the_cells() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(frames::populated()), 80, 24);
    let rendered: Vec<String> = (0..24).map(|y| line(&buf, y)).collect();
    let joined = rendered.join("\n");

    for heading in ["Projects (29)", "Libraries (1)", "Scratchpad (0)", "Rules (11)", "Active Projects (2)", "Last Errors"] {
        assert!(joined.contains(heading), "{heading} left the grid at 80x24");
    }
    assert!(
        joined.contains("… 3 more"),
        "the Projects cell must shed rows and say so: {joined}"
    );
}

/// Every figure in every frame fits the column it is drawn in, checked against the DATA.
///
/// This is the guard the `…` net in [`crate::panes::cell`] sits under: the ellipsis stops a
/// wrong number reaching the screen, and this stops the ellipsis being needed. Asserted
/// structurally rather than by reading the render, because on screen a figure's ellipsis and a
/// name's are the same character — the first version of this test failed on `(qdrant …`, which
/// is a text column doing exactly what it should.
#[test]
fn every_frame_figure_fits_the_column_it_is_drawn_in() {
    for pane in frames::populated() {
        let table = pane.table();
        for row in table.rows() {
            for (cell, column) in row.iter().zip(table.columns()) {
                let Some(width) = column.fixed() else { continue };
                if !cell.is_figure() {
                    continue;
                }
                assert!(
                    cell.natural_width() <= width as usize,
                    "a {} of {} columns does not fit its {width}-column field — widen the \
                     column, do not accept the ellipsis",
                    column.title,
                    cell.natural_width()
                );
            }
        }
    }
}

/// A cell whose columns are all fixed starves its flex column before it drops a column. That is
/// what breaks first on a narrow screen, and it is pinned so the day it is fixed is deliberate.
#[test]
fn a_narrow_cell_starves_its_flexible_column_rather_than_dropping_a_fixed_one() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let table = CellTable::new(
        vec![
            Column::flex("Name"),
            Column::text("Branch", 19),
            Column::number("Files", 5),
            Column::number("Queue", 8),
        ],
        vec![vec![
            Cell::Text("workspace-qdrant-mcp".into()),
            Cell::Text("dev".into()),
            Cell::Num(93),
            Cell::Queue { pending: 50, in_flight: 0, failed: 0 },
        ]],
    );
    let area = Rect::new(0, 0, 38, 4);
    let mut buf = Buffer::empty(area);
    CellPane::new("Active Projects", Some(2), table).render(area, &mut buf);

    let drawn = line(&buf, 2);
    let name = drawn.split_whitespace().next().expect("a name is drawn");
    assert!(
        name.ends_with('…') && name.chars().count() < "workspace-qdrant-mcp".chars().count(),
        "the flex column is what gives, down to an ellipsis: {drawn:?}"
    );
    assert!(
        line(&buf, 2).contains("dev"),
        "the fixed columns keep their width: {:?}",
        line(&buf, 2)
    );
}

/// The first grid row, and the six cell rectangles as the renderer lays them out.
///
/// Reconstructed from the view's own `grid` over the same area the render uses, rather than
/// from a table of coordinates: a fixture of numbers would keep passing after the layout moved
/// and would then be checking the wrong columns.
fn heading_rows() -> (u16, Vec<Rect>) {
    let first = crate::views::top::CONSTANT_ROWS + crate::panes::status_block::ROWS_FULL;
    let (cells, _) = grid(crate::widgets::chrome::inset(Rect::new(
        0,
        first,
        WIDE,
        TALL - 1 - first,
    )));
    (first, cells)
}

/// One heading row, read back out of the buffer.
fn heading_text(buf: &Buffer, cell: Rect) -> String {
    (cell.x..cell.x + cell.width)
        .map(|x| buf.cell((x, cell.y)).expect("cell in area").symbol())
        .collect()
}

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

/// A focused cell carries BOTH marks: the `▌` that says which zone is live, and the accent on
/// the letter that gets you there. They answer different questions, so neither replaces the
/// other.
#[test]
fn a_focused_cell_keeps_its_bar_and_its_key() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const RULES: usize = 3;
    let buf = render(
        view(frames::populated()).attention(Attention::Zone(RULES)),
        WIDE,
        TALL,
    );
    let (_, cells) = heading_rows();
    let text = heading_text(&buf, cells[RULES]);
    assert!(
        text.starts_with(&format!(
            "{} Rules",
            crate::widgets::chrome::zone_heading::FOCUS_BAR
        )),
        "the focused cell keeps the bar: {text:?}"
    );

    let x = cells[RULES].x
        + text
            .chars()
            .position(|c| c.eq_ignore_ascii_case(&FOCUS_KEYS[RULES]))
            .expect("the key letter is drawn") as u16;
    assert_eq!(
        buf.cell((x, cells[RULES].y)).expect("cell in area").style().fg,
        Some(crate::tokens::accent()),
        "focus does not take the key's accent away: {text:?}"
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
        heading_text(&buf, Rect { y: cell.y + 2, ..cell }).starts_with(crate::panes::cell::EMPTY),
        "an empty projection says so rather than showing a blank cell"
    );
}

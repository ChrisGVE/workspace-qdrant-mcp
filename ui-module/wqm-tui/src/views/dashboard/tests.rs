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

/// The rules divide the ROWS and run edge to edge, as every other rule on this surface does.
#[test]
fn the_row_separators_span_the_whole_width() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(frames::populated()), WIDE, TALL);
    let full = RULE.repeat(WIDE as usize);
    let separators = (0..TALL).filter(|y| line(&buf, *y) == full).count();
    // Two frame-level rules from the constant top, plus the grid's own two.
    assert_eq!(
        separators,
        2 + (GRID_ROWS - 1),
        "expected the top rule, the block's closing rule and one rule per row seam"
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

/// The foot offers the six focus keys as one hint, and does NOT offer `F Global`.
#[test]
fn the_foot_offers_the_focus_keys_and_nothing_that_does_nothing() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let foot = line(&render(view(frames::populated()), WIDE, TALL), TALL - 1);
    assert!(foot.contains("p/l/s/r/a/e Focus cell"), "{foot:?}");
    for (key, label) in [("Enter", "Detail"), ("?", "Help"), ("q", "Quit")] {
        assert!(foot.contains(&format!("{key} {label}")), "{foot:?}");
    }
    assert!(
        !foot.contains("Global"),
        "a hint for an action nothing implements is a screen that does not exist: {foot:?}"
    );
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

    for heading in ["Projects (29)", "Libraries (1)", "Scratchpad (0)", "Rules (0)", "Active Projects (2)", "Last Errors"] {
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

//! What the CELL's own furniture is pinned to: the weight of its heading, and the fact that
//! nothing is indented beneath it.
//!
//! R9 (Chris, 2026-09-07) is two rulings that arrive together and are one change to the eye —
//! the heading takes weight so it stops reading as another row of data, and the two-column
//! marker gutter goes so the table starts under the first letter of that heading:
//! *"This gives us the ability to remove the indent under the title and thus regaining two
//! columns"*. A sibling of `super` for the same reason `focus` is one: a test file nobody can
//! hold in one screenful is the readability defect it exists to prevent.

use super::*;
use ratatui::style::Modifier;

/// Every cell heading is BOLD, focused or not.
///
/// Asserted against [`Modifier::BOLD`] — the stated constant — rather than by comparing this
/// render with another one. Two renders of the same code agree with each other whatever weight
/// they carry, which is the shape of guard that blesses whatever it finds.
///
/// Both frames are swept, because the focused cell reaches its weight down a different path:
/// five headings are spans of text and the sixth is an inverted block, and a `bold()` that
/// reached only the first path would leave the block reading lighter than the cells around it.
#[test]
fn every_cell_heading_is_bold_focused_or_not() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const RULES: usize = 3;
    for attention in [Attention::None, Attention::Zone(RULES)] {
        let buf = render(view(frames::populated()).attention(attention), WIDE, TALL);
        let (_, cells) = heading_rows();

        for (zone, cell) in cells.iter().enumerate() {
            let text = heading_text(&buf, *cell);
            let mut lit = 0;
            for x in cell.x..cell.x + cell.width {
                let drawn = buf.cell((x, cell.y)).expect("cell in area");
                if drawn.symbol() == " " {
                    continue;
                }
                assert!(
                    drawn.style().add_modifier.contains(Modifier::BOLD),
                    "zone {zone} under {attention:?}: column {x} of {text:?} is not bold"
                );
                lit += 1;
            }
            assert!(lit > 0, "zone {zone}'s heading drew nothing at all: {text:?}");
        }
    }
}

/// The table starts at the cell's own first column: no indent under the heading.
///
/// Read as a pair of x positions rather than as "the header does not begin with a space",
/// because the second phrasing passes just as happily on a cell drawn one column further left
/// than its own heading. What the ruling is about is the two lining UP.
#[test]
fn the_column_header_starts_on_the_same_column_as_the_heading_above_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(frames::populated()), WIDE, TALL);
    let (_, cells) = heading_rows();

    for (zone, cell) in cells.iter().enumerate() {
        let heading = heading_text(&buf, *cell);
        let header = heading_text(&buf, Rect { y: cell.y + 1, ..*cell });
        assert_ne!(
            buf.cell((cell.x, cell.y)).expect("cell in area").symbol(),
            " ",
            "zone {zone}'s heading does not start at its own first column: {heading:?}"
        );
        assert_ne!(
            buf.cell((cell.x, cell.y + 1)).expect("cell in area").symbol(),
            " ",
            "zone {zone}'s column header is indented under its heading: {header:?}"
        );
    }
}

/// The design floor is 100 × 30 (Chris, 2026-09-07): every column of `Active Projects` is
/// drawn, and neither project name elides. 80 × 24 is a stress case, not the floor — below it
/// the cell drops fixed columns rather than starve the flex `Name`.
///
/// At 100 columns the grid insets to 96 and the two cells get 47 and 46; `Active Projects` is
/// the left cell, and it spends `Branch` 11 + `Files` 5 + `Queue` 8 = 24 on fixed columns with
/// three single-column gaps, leaving 47 − 24 − 3 = **20** for `Name` — exactly the width of
/// `workspace-qdrant-mcp`, the longer of the two names.
#[test]
fn at_100x30_the_active_projects_cell_shows_every_column_and_no_name_elides() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(frames::populated()), 100, 30);
    let heading_y = (0..30)
        .find_map(|y| line(&buf, y).find("Active Projects").map(|_| y))
        .expect("the Active Projects cell is on a 100x30 screen");

    let header = line(&buf, heading_y + 1);
    for title in ["Name", "Branch", "Files", "Queue"] {
        assert!(
            header.contains(title),
            "{title:?} left the Active Projects header at 100x30: {header:?}"
        );
    }

    // Both names draw whole — a name column of 20 holds `workspace-qdrant-mcp` exactly, so the
    // full string is present in its row and no `…` stands in for its tail.
    for name in ["open-books", "workspace-qdrant-mcp"] {
        let whole = (heading_y + 2..heading_y + 4)
            .map(|y| line(&buf, y))
            .any(|row| row.contains(name));
        assert!(whole, "{name} is not drawn whole at the 100x30 floor");
    }
}

/// 80 × 24 is a stress case, not a target: the cell drops the queue triple first — its counts
/// are the one thing a reader can find again in the status block — and the flex `Name` keeps
/// its floor rather than being starved.
#[test]
fn at_80x24_the_active_projects_cell_drops_the_queue_column_and_keeps_the_name_at_its_floor() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(frames::populated()), 80, 24);
    let (heading_y, heading_x) = (0..24)
        .find_map(|y| line(&buf, y).find("Active Projects").map(|x| (y, x as u16)))
        .expect("the Active Projects cell is on an 80x24 screen");
    let header = line(&buf, heading_y + 1);

    // The queue triple is the first column to go.
    assert!(
        !header.contains("Queue"),
        "the Queue column survived the stress size: {header:?}"
    );
    assert!(
        header.contains("Name") && header.contains("Branch"),
        "the flex identity and the text column survive: {header:?}"
    );

    // `Branch` is left-aligned one gap past the flex column, so its position measures the Name
    // column's width directly — and it must be at least the twelve-cell floor.
    let branch = header.find("Branch").expect("its Branch column header is drawn") as u16;
    let name_width = branch - heading_x - 1;
    assert!(
        name_width >= 12,
        "the Name column is {name_width} wide at 80x24, below its floor: {header:?}"
    );

    // And the dropped column vanishes from the data rows too, not only from the header: the
    // first project's queue triple `247/4/0` is nowhere on the screen.
    let first_data = line(&buf, heading_y + 2);
    assert!(
        !first_data.contains("247/4/0"),
        "the dropped queue triple is still drawn in a data row: {first_data:?}"
    );
}

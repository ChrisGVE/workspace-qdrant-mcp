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

/// The two columns the gutter used to hold go to the flex column, and at 80 × 24 that is the
/// difference between a `Name` column and no `Name` column at all.
///
/// The arithmetic, stated: 80 columns inset by [`crate::widgets::chrome::MARGIN`] on each side
/// is 76; the three-column gap leaves 73 for the two cells, and the remainder goes to the left
/// one — 37 and 36. `Active Projects` is the left cell of the third row, and it spends
/// `Branch` 19 + `Files` 5 + `Queue` 8 = 32 on fixed columns with three single-column gaps
/// between its four, leaving 37 − 32 − 3 = **2** for `Name`. With the old two-column gutter it
/// was **0**: the column existed and drew nothing.
#[test]
fn at_80x24_the_active_projects_name_column_is_two_columns_rather_than_none() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    /// 37 − 32 − 3, with the gutter gone.
    const NAME_WIDTH: u16 = 2;
    /// What the same cell had while the gutter took its two columns.
    const NAME_WIDTH_WITH_GUTTER: u16 = 0;

    let buf = render(view(frames::populated()), 80, 24);
    let (heading_y, heading_x) = (0..24)
        .find_map(|y| line(&buf, y).find("Active Projects").map(|x| (y, x as u16)))
        .expect("the Active Projects cell is on an 80x24 screen");
    let header = line(&buf, heading_y + 1);
    let branch = header.find("Branch").expect("its Branch column header is drawn") as u16;

    // `Branch` is left-aligned in the column after `Name`, and one gap column separates them.
    let measured = branch - heading_x - 1;
    assert_eq!(measured, NAME_WIDTH, "the Name column is {NAME_WIDTH} wide at 80x24: {header:?}");

    // The width alone does not say it: with the two-column gutter back, the gutter takes
    // exactly the two columns `Name` has and `Branch` lands on the very same x — the distance
    // between the heading and `Branch` is invariant across the change it is meant to detect.
    // What is NOT invariant is whether the Name column draws anything, so that is what is read:
    // the header row must begin, at the cell's own first column, with the (clipped) title.
    let drawn: String = (heading_x..heading_x + NAME_WIDTH + 1)
        .map(|x| buf.cell((x, heading_y + 1)).expect("cell in area").symbol())
        .collect();
    assert_eq!(
        drawn, "Na ",
        "a Name column of {NAME_WIDTH_WITH_GUTTER} columns draws nothing at all: {header:?}"
    );
}

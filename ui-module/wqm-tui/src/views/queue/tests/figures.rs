//! The figure columns, as they land on the page.
//!
//! Chris, 2026-09-07 (ruling 11): *"right-aligned; sizes integer + unit aligned on the space
//! (`31 KB`); ages `1m`/`3h`/`2d`, no `ago`, aligned on the unit."*
//!
//! `format`'s own guards hold the SHAPE of a figure. These hold the thing the ruling is actually
//! about, which no unit test of a producer can see: that on the drawn page, down a column of two
//! hundred rows, the space and the unit each land in one column.

use super::*;
use crate::panes::cell::fit::{fit, laid_out};

/// Where the two figure columns end, worked back from the right-hand edge of the drawn page.
///
/// Both edges come from the same fit used by the renderer, including its
/// uniform gap and any width returned to the flex Object column.
fn right_edges() -> (usize, usize) {
    let pane = frames::pane(&QueueState::default());
    let area = Rect::new(
        MARGIN,
        0,
        WIDE - MARGIN * 2,
        1,
    );
    let fitted = fit(pane.columns(), pane.rows(), area.width, 12, 1);
    let rects = laid_out(area, &fitted);
    let edge = |at| {
        let position = fitted.active.iter().position(|&column| column == at).unwrap();
        rects[position].right() as usize
    };
    (edge(frames::SIZE), edge(frames::AGE))
}

/// The DATA rows the page draws, as strings — not the rules, not the load-more line, not the
/// foot.
///
/// A data row is one that carries its position in the number column, which is the one mark
/// every data row has and nothing else on the page does.
fn body(buf: &Buffer) -> Vec<String> {
    let first = header_row() + 1;
    (first..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
                .collect::<String>()
        })
        .filter(|row| {
            let number: String = row
                .chars()
                .skip(MARGIN as usize)
                .take(3)
                .collect::<String>()
                .trim()
                .to_string();
            !number.is_empty() && number.chars().all(|c| c.is_ascii_digit())
        })
        .collect()
}

/// Every size puts its space in one column, and its unit in the two cells after it.
///
/// The failure this catches is the one right-alignment alone produces: `169 B` and `31 KB`
/// aligned on their right edge have their spaces one column apart, and a reader scanning the
/// column for magnitudes sees a ragged seam.
#[test]
fn the_size_columns_space_lands_in_one_column_on_every_row() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(QueueState::default()), WIDE, TALL);
    let (size_right, _) = right_edges();
    // The unit is padded to two cells, so the space sits three back from the column's edge.
    let space = size_right - 3;

    let mut seen = 0;
    for row in body(&buf) {
        let cells: Vec<char> = row.chars().collect();
        let field: String = cells[space - 4..size_right].iter().collect();
        if field.trim().is_empty() {
            continue; // A row whose size is not known draws nothing at all.
        }
        seen += 1;
        assert_eq!(
            cells[space], ' ',
            "the size on {field:?} does not break at column {space}"
        );
        assert!(
            cells[space + 1].is_ascii_alphabetic(),
            "the unit does not start at column {}: {field:?}",
            space + 1
        );
        assert!(
            cells[space - 1].is_ascii_digit(),
            "the number does not end at column {}: {field:?}",
            space - 1
        );
    }
    assert!(seen > 5, "only {seen} sized rows on the page — proves little");
}

/// Every age ends on the page's own right edge: one unit letter, in one column, on every row.
///
/// Unlike a size, an age needs no padding to manage this — its unit is a single character and
/// the column is right-aligned — so what this guard really pins is that nothing has grown a
/// suffix. An `ago` would move the letter and this is what would say so.
#[test]
fn the_age_columns_unit_lands_in_one_column_and_carries_no_ago() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(QueueState::default()), WIDE, TALL);
    let (_, age_right) = right_edges();

    let mut seen = 0;
    for row in body(&buf) {
        let cells: Vec<char> = row.chars().collect();
        let unit = cells[age_right - 1];
        seen += 1;
        assert!(
            unit.is_ascii_alphabetic(),
            "row {:?} does not end its age at column {}",
            row.trim_end(),
            age_right - 1
        );
        assert!(
            cells[age_right - 2].is_ascii_digit(),
            "the age's number does not run up to its unit: {:?}",
            row.trim_end()
        );
        assert!(
            !row.contains("ago"),
            "the column prints a figure, not a sentence: {:?}",
            row.trim_end()
        );
    }
    assert!(seen > 5, "only {seen} rows on the page — proves little");
}

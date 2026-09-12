//! Ruling 3 (Chris, 20260912): what a table that is NOT the live one looks like.
//!
//! *"With no table selected every table is at baseline and no sort letters are lit anywhere.
//! With one selected: that table lights its sort letters; every other table goes dull entirely
//! — its zone title and its column names lose their highlighting, exactly as under a modal —
//! while the rest of the UI keeps its vibrancy."*
//!
//! A sibling of `focus` rather than more of it. `focus` is about the cell that HAS the
//! attention — the selector block, the data cursor, the foot that follows it. This is about the
//! five that do not, which is a different claim and fails independently: a cell can take its
//! block correctly while its neighbours keep painting red status words beside it.
//!
//! The two halves of the ruling that were already guarded elsewhere are not restated here: the
//! sort letters (`sort::a_column_key_is_lit_only_on_the_focused_cell_…`, whose last assertion is
//! the *nothing focused, nothing lit* case) and the jump letter that survives receding
//! (`focus::the_cells_beside_the_focused_one_keep_their_key_letters`). What is left is the part
//! WO-H's `Cell::spans(…, recede)` made possible and nothing yet reads back: the **body**.

use super::*;

/// Zone 3 — Rules, eight rows. The cell every guard here focuses, so that the other five are
/// the ones under test.
const RULES: usize = 3;

/// Zone 0 — Projects, and the only cell whose rows carry the queue triple's three hues. It is
/// the strongest witness the sweep has: a receded table still painting `degraded`, `in flight`
/// and `offline` beside a grey one is the exact failure ruling 3 names.
const PROJECTS: usize = 0;

/// Everything a RECEDED cell is allowed to paint: the unpainted ground, and the two rungs at or
/// below [`crate::tokens::muted`].
///
/// Deliberately narrower than the modal sweep's set in `super` — that one admits the structural
/// backgrounds because a modal leaves the whole page's furniture standing, whereas one cell of
/// the grid draws no rule and takes no cursor. A background appearing inside a receded cell is a
/// finding, not a false alarm, so it is left out and the guard is allowed to say so.
fn quiet() -> [ratatui::style::Color; 3] {
    [
        ratatui::style::Color::Reset,
        crate::tokens::faint(),
        crate::tokens::muted(),
    ]
}

/// The body of one cell — every row under its zone heading, which is the column header and the
/// data rows and the overflow tail.
///
/// The heading row itself is excluded on purpose: it keeps its accented jump letter even while
/// receded (that is what makes the cell reachable), and `focus` owns that claim.
fn body(cell: Rect) -> Rect {
    Rect {
        y: cell.y + 1,
        height: cell.height.saturating_sub(1),
        ..cell
    }
}

/// Every cell of `area` painting a foreground or a background outside `allowed`.
///
/// The shape `super`'s modal sweep uses, over a sub-rectangle rather than the whole screen —
/// `coloured_cells` scans a buffer from its own origin, and the question here is about one
/// sixth of the grid rather than about the screen.
fn loud(
    buf: &Buffer,
    area: Rect,
    allowed: &[ratatui::style::Color],
) -> Vec<(u16, u16, &'static str, ratatui::style::Color)> {
    let mut found = Vec::new();
    for y in area.y..area.y + area.height {
        for x in area.x..area.x + area.width {
            let style = buf.cell((x, y)).expect("cell in area").style();
            if let Some(fg) = style.fg.filter(|fg| !allowed.contains(fg)) {
                found.push((x, y, "fg", fg));
            }
            if let Some(bg) = style.bg.filter(|bg| !allowed.contains(bg)) {
                found.push((x, y, "bg", bg));
            }
        }
    }
    found
}

/// The whole body of every cell that is not the live one drops to the muted rung — the column
/// names, the data, and the hues inside the data.
///
/// The live cell is checked in the same pass for the opposite property, because "everything is
/// grey" would satisfy an assertion about the five and is the other way this can fail.
#[test]
fn every_cell_but_the_live_one_paints_nothing_brighter_than_muted() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(
        view(frames::populated()).attention(Attention::Zone(RULES)),
        WIDE,
        TALL,
    );
    let (_, cells) = heading_rows();

    for (zone, cell) in cells.iter().enumerate() {
        let found = loud(&buf, body(*cell), &quiet());
        if zone == RULES {
            assert!(
                !found.is_empty(),
                "the live cell went grey with its neighbours — then nothing distinguishes it"
            );
            continue;
        }
        assert!(
            found.is_empty(),
            "zone {zone} has receded and still paints {} bright cells, first ten: {:?}",
            found.len(),
            &found[..found.len().min(10)]
        );
    }
}

/// The hues specifically: the queue triple that Projects draws in three colours is one grey
/// figure once the cell has receded.
///
/// Named separately from the sweep above because it is the claim Chris made in his own words —
/// *"dull entirely"* — and a sweep phrased as "no bright rungs" would still pass on a table that
/// had muted its text and kept its status colours.
#[test]
fn a_receded_cell_draws_its_queue_triple_in_one_grey() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let hues = [
        crate::tokens::degraded(),
        crate::tokens::in_flight(),
        crate::tokens::offline(),
    ];
    let (_, cells) = heading_rows();
    let projects = body(cells[PROJECTS]);

    let live = render(view(frames::populated()), WIDE, TALL);
    let painted = (projects.y..projects.y + projects.height)
        .flat_map(|y| (projects.x..projects.x + projects.width).map(move |x| (x, y)))
        .filter(|(x, y)| {
            live.cell((*x, *y))
                .expect("cell in area")
                .style()
                .fg
                .is_some_and(|fg| hues.contains(&fg))
        })
        .count();
    assert!(
        painted > 0,
        "Projects paints no queue hue even at baseline — this guard is checking nothing"
    );

    let receded = render(
        view(frames::populated()).attention(Attention::Zone(RULES)),
        WIDE,
        TALL,
    );
    for y in projects.y..projects.y + projects.height {
        for x in projects.x..projects.x + projects.width {
            let fg = receded.cell((x, y)).expect("cell in area").style().fg;
            assert!(
                !fg.is_some_and(|fg| hues.contains(&fg)),
                "a queue hue survives at {x},{y} on a cell that has receded"
            );
        }
    }
}

/// With NO cell focused, nothing recedes: every cell paints its data at
/// [`crate::tokens::table_row`], the baseline WO-H set.
///
/// The complement of the sweep above, and the half that catches a `receded` flag wired to the
/// wrong side of its condition — which would leave the default view entirely grey and every
/// focus guard still green.
#[test]
fn nothing_recedes_while_no_cell_is_focused() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(view(frames::populated()), WIDE, TALL);
    let (_, cells) = heading_rows();

    for (zone, cell) in cells.iter().enumerate() {
        if frames::populated()[zone].table().is_empty() {
            continue; // an empty projection has no data row to paint at any rung.
        }
        let area = body(*cell);
        let rows = (area.y..area.y + area.height)
            .flat_map(|y| (area.x..area.x + area.width).map(move |x| (x, y)))
            .filter(|(x, y)| {
                buf.cell((*x, *y)).expect("cell in area").style().fg
                    == Some(crate::tokens::table_row())
            })
            .count();
        assert!(
            rows > 0,
            "zone {zone} paints nothing at the table-row rung with no cell focused — it has \
             receded while nothing is live"
        );
    }
}

/// The rest of the UI keeps its vibrancy: focusing a cell changes the GRID and nothing above
/// it.
///
/// Compared cell by cell rather than by reading the lines back, because the claim is about
/// colour — a status block that had gone grey would print the identical characters.
///
/// The FOOT is excluded, and that is not a hole in the guard: the foot follows the live cell by
/// design (`focus::focusing_a_many_row_cell_offers_navigation_and_enter`), so it is the one
/// thing below the grid that is *supposed* to change. Vibrancy above the grid is what ruling 3
/// is about — the tab line, the status block and their hues.
#[test]
fn focusing_a_cell_leaves_everything_above_the_grid_untouched() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let live = render(view(frames::populated()), WIDE, TALL);
    let focused = render(
        view(frames::populated()).attention(Attention::Zone(RULES)),
        WIDE,
        TALL,
    );
    let (first, _) = heading_rows();

    for y in 0..first {
        for x in 0..WIDE {
            assert_eq!(
                live.cell((x, y)).expect("cell in area").style(),
                focused.cell((x, y)).expect("cell in area").style(),
                "focusing a cell changed {x},{y}, which is above the grid"
            );
        }
    }
}

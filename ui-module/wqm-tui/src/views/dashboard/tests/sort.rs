//! What the SORT ruling is pinned to: which letters light up, when, and what pressing one does.
//!
//! Chris, 2026-09-07: *"When a section is selected and contains multiple items we highlight
//! (using the selection color) one letter of the column name, non-ambiguous with another of the
//! 5 sections; when pressing on that letter the user can sort by the column: first press
//! ascending, second press descending."*
//!
//! Three separate claims live in that sentence and each gets its own guard, because they fail
//! independently: the letters are offered only where they mean something, each lit letter is
//! the key it claims to be, and no key collides with one the screen has already spoken for.

use super::*;
use crate::panes::cell::{Cell, Direction, Sort};
use ratatui::style::Modifier;

/// The Dashboard's cell geometry at 125 × 34, for the guards that judge one cell alone.
const CELL: Rect = Rect { x: 0, y: 0, width: 59, height: 9 };

/// Every cell of the grid whose foreground is the sort key's signature — the accent hue AND the
/// header's underline — as `(x, y, symbol)`.
///
/// Scanned over the grid only. The constant top has a selector of its own — the active tab —
/// and it is drawn as an inverted block, so its letters carry the selector as a BACKGROUND;
/// scanning the whole screen for a selector foreground would still be correct today and would
/// stop being correct the day anything above the grid lights a letter.
///
/// The UNDERLINE is what tells a sort key from a heading's focus key: both wear the accent hue
/// and both are bold, but only the sort key sits on the header row, and the header row carries a
/// rule under the whole of it (Chris, 20260912, ruling 1). It was the italic until that ruling
/// retired it.
fn lit(buf: &Buffer, first: u16) -> Vec<(u16, u16, String)> {
    let mut found = Vec::new();
    for y in first..TALL {
        for x in 0..WIDE {
            let cell = buf.cell((x, y)).expect("cell in area");
            if cell.style().fg == Some(crate::tokens::accent())
                && cell.style().add_modifier.contains(Modifier::UNDERLINED)
            {
                found.push((x, y, cell.symbol().to_string()));
            }
        }
    }
    found
}

/// A key is offered only on the cell that is live AND holds more than one row.
///
/// Both halves of the condition are exercised: focusing the Rules cell (eight rows) lights
/// exactly its three column keys and nothing anywhere else, and focusing Libraries (one row)
/// lights nothing at all — there is nothing to reorder in a list of one, and a lit letter is a
/// promise that the key does something.
#[test]
fn a_column_key_is_lit_only_on_the_focused_cell_and_only_when_it_has_rows_to_reorder() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const RULES: usize = 3;
    const LIBRARIES: usize = 1;

    let (first, cells) = heading_rows();
    let rules_cell = cells[RULES];

    let many = render(
        view(frames::populated()).attention(Attention::Zone(RULES)),
        WIDE,
        TALL,
    );
    let lit_on_rules = lit(&many, first);
    assert_eq!(
        lit_on_rules.len(),
        3,
        "the Rules cell has three sortable columns, so three letters light: {lit_on_rules:?}"
    );
    for (x, y, symbol) in &lit_on_rules {
        assert_eq!(*y, rules_cell.y + 1, "{symbol:?} is lit off the Rules column header");
        assert!(
            (rules_cell.x..rules_cell.x + rules_cell.width).contains(x),
            "{symbol:?} is lit outside the focused cell"
        );
    }

    assert!(
        frames::populated()[LIBRARIES].table().len() == 1,
        "this guard is about a one-row cell; the fixture has to be one"
    );
    let one = render(
        view(frames::populated()).attention(Attention::Zone(LIBRARIES)),
        WIDE,
        TALL,
    );
    assert_eq!(
        lit(&one, first),
        Vec::new(),
        "a cell with one row offers no sort key — there is nothing to reorder"
    );

    // And with nothing focused at all, nothing is offered anywhere.
    assert_eq!(lit(&render(view(frames::populated()), WIDE, TALL), first), Vec::new());
}

/// Each lit letter is the column's own sort key, and it is the ONLY thing lit on that header.
///
/// Read as the sequence of SYMBOLS rather than as a count: a guard that counted three lit cells
/// would pass on a header that lit three arbitrary characters, which is the failure this is for.
#[test]
fn the_lit_letters_are_the_sort_keys_of_their_columns_in_column_order() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    const RULES: usize = 3;
    let buf = render(
        view(frames::populated()).attention(Attention::Zone(RULES)),
        WIDE,
        TALL,
    );
    let (first, _) = heading_rows();

    // `Rule name` lights its `n`, `Scope` its `c`, `Queue` its `u` — the first case-insensitive
    // occurrence in each title, left to right across the row.
    let symbols: Vec<String> = lit(&buf, first).into_iter().map(|(_, _, s)| s).collect();
    assert_eq!(symbols, vec!["n", "c", "u"], "the lit letters are not the keys");

    let keys: Vec<char> = frames::populated()[RULES]
        .table()
        .columns()
        .iter()
        .filter_map(|column| column.sort_key)
        .collect();
    assert_eq!(
        symbols
            .iter()
            .map(|s| s.chars().next().expect("one letter"))
            .collect::<Vec<char>>(),
        keys,
        "what is drawn and what the columns declare are two different lists"
    );
}

/// Pressing the key sorts the rows, and the header says which column and which way.
///
/// The expected order is built here from the fixture with the comparator SPELLED OUT — files
/// descending, ties keeping the fixture's own order — rather than by sorting the rendered rows,
/// which would compare the renderer against itself and agree with any order at all.
#[test]
fn sorting_by_files_reorders_the_rows_and_marks_the_column_it_sorted_by() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    /// The fixture's names in `files` order, ties keeping the order the fixture states them
    /// in — which is what a STABLE sort gives, and the reason `descending` is not simply
    /// `ascending` reversed: three of the seven projects hold zero files, and reversing would
    /// turn that tie group over while the renderer leaves it alone.
    fn expected(descending: bool) -> Vec<&'static str> {
        let mut rows: Vec<&frames::ProjectRow> = frames::PROJECTS.iter().collect();
        rows.sort_by(|a, b| {
            if descending {
                b.files.cmp(&a.files)
            } else {
                a.files.cmp(&b.files)
            }
        });
        rows.iter().map(|row| row.name).collect()
    }

    let expected_desc = expected(true);
    assert_ne!(
        expected_desc,
        frames::PROJECTS.iter().map(|row| row.name).collect::<Vec<&str>>(),
        "the fixture is already in Files order — sorting it would prove nothing"
    );

    let drawn: Vec<String> = frames::sorted_by_files()[0]
        .table()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Cell::Text(name) => name.clone(),
            _ => panic!("the Projects cell's first column must be a name"),
        })
        .collect();
    assert_eq!(drawn, expected_desc, "the rows are not in Files-descending order");

    // And the FIRST press, which Chris said is ascending.
    let ascending: Vec<String> = frames::populated()
        .remove(0)
        .sorted(Sort { column: frames::PROJECTS_FILES, direction: Direction::Asc })
        .table()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Cell::Text(name) => name.clone(),
            _ => panic!("the Projects cell's first column must be a name"),
        })
        .collect();
    assert_eq!(ascending, expected(false), "the first press is ascending");

    // And the header says so, on the sorted column and on no other.
    let (_, cells) = heading_rows();
    let buf = render(
        view(frames::sorted_by_files()).attention(Attention::Zone(0)),
        WIDE,
        TALL,
    );
    let header = heading_text(&buf, Rect { y: cells[0].y + 1, ..cells[0] });
    assert!(header.contains("Files ↓"), "the sorted column carries a spaced mark: {header:?}");
    assert_eq!(header.matches('↓').count(), 1, "one column is sorted, not several: {header:?}");
    assert!(!header.contains('↑'), "{header:?}");

    let unsorted = heading_text(
        &render(view(frames::populated()).attention(Attention::Zone(0)), WIDE, TALL),
        Rect { y: cells[0].y + 1, ..cells[0] },
    );
    assert!(
        !unsorted.contains('↓') && !unsorted.contains('↑'),
        "an unsorted table carries no mark: {unsorted:?}"
    );
}

/// The other two comparators, each exercised on the column that shows it.
///
/// `Files` is the numeric one and has its own guard above; these are the two that a plausible
/// implementation gets wrong in a way nothing else would catch. **Text sorts case-insensitively**
/// — a list a person reads is alphabetical, not ASCII, and a byte comparison would file every
/// capitalised project above every lowercase one. **A queue triple sorts by its first number**,
/// the pending count, which is the one that says how much work is waiting; sorting by either of
/// the other two would leave this fixture in the order it came in, because they are all zero.
#[test]
fn text_sorts_case_insensitively_and_a_queue_triple_sorts_by_what_is_waiting() {
    fn names(column: usize, direction: Direction) -> Vec<String> {
        frames::populated()
            .remove(0)
            .sorted(Sort { column, direction })
            .table()
            .rows()
            .iter()
            .map(|row| match &row[0] {
                Cell::Text(name) => name.clone(),
                _ => panic!("the Projects cell's first column must be a name"),
            })
            .collect()
    }

    assert_eq!(
        names(frames::PROJECTS_NAME, Direction::Asc),
        vec![".config", "ArraySwift", "claude", "de-slop", "ExtendedSwiftMath", "inkyfingers", "localdata-mcp"],
        "a byte comparison would put every capitalised name above every lowercase one"
    );

    // Pending, descending: 2'635, 877, 286, 285, 153, 102, 101 — and every `failed` is zero, so
    // sorting on that number instead would hand back the fixture's own order untouched.
    assert_eq!(
        names(frames::PROJECTS_QUEUE, Direction::Desc),
        vec![".config", "claude", "de-slop", "inkyfingers", "localdata-mcp", "ExtendedSwiftMath", "ArraySwift"],
        "the triple sorts by what is WAITING, not by what failed"
    );
}

/// Chris's *non-ambiguous* requirement, made structural: no cell offers one letter twice, and
/// no cell offers a letter the screen has already bound.
///
/// Written against [`DASHBOARD_BOUND_KEYS`] rather than against a list of letters spelled out
/// here, so the day a key is bound the collision is a red test rather than a surprise on the
/// screen. That is the whole reason the set is a constant.
#[test]
fn every_sort_key_is_unique_in_its_cell_and_free_of_the_keys_the_screen_has_bound() {
    for (zone, pane) in frames::populated().iter().enumerate() {
        let keys: Vec<char> = pane
            .table()
            .columns()
            .iter()
            .filter_map(|column| column.sort_key)
            .collect();
        assert!(!keys.is_empty(), "cell {zone} offers no sort key at all");
        for (i, key) in keys.iter().enumerate() {
            assert!(
                !DASHBOARD_BOUND_KEYS.contains(key),
                "cell {zone} sorts on `{key}`, which the screen has already bound"
            );
            assert!(
                !keys[..i].contains(key),
                "cell {zone} offers `{key}` on two columns: {keys:?}"
            );
        }
    }
}

/// Every sortable column can actually SHOW its mark, and none of them clips a neighbour to do
/// it.
///
/// The mark borrows the column gap to its right rather than taking width from the table
/// ([`crate::panes::cell::table`]), which works only while every sortable column overflows by
/// at most one column and every LAST column holds its own mark. Both are properties of the
/// column widths, so they are checked against every column of every cell rather than trusted:
/// the first cut of this had `Sync` four columns wide and lost its mark silently.
#[test]
fn every_sortable_column_shows_its_mark_without_clipping_the_column_beside_it() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    for zone in 0..CELLS {
        let titles: Vec<&'static str> = frames::populated()[zone]
            .table()
            .columns()
            .iter()
            .map(|column| column.title)
            .collect();
        let sortable: Vec<usize> = frames::populated()[zone]
            .table()
            .columns()
            .iter()
            .enumerate()
            .filter_map(|(at, column)| column.sort_key.map(|_| at))
            .collect();

        for column in sortable {
            for direction in [Direction::Asc, Direction::Desc] {
                let mut buf = Buffer::empty(CELL);
                frames::populated()
                    .remove(zone)
                    .sorted(Sort { column, direction })
                    .render(CELL, &mut buf);
                let header: String = (0..CELL.width)
                    .map(|x| buf.cell((x, 1)).expect("cell in area").symbol())
                    .collect();

                assert!(
                    header.contains(direction.glyph()),
                    "cell {zone} column {column} ({}) sorted {direction:?} shows no mark: \
                     {header:?}",
                    titles[column]
                );
                for title in &titles {
                    assert!(
                        header.contains(title),
                        "cell {zone} sorted on {} clipped the {title:?} header: {header:?}",
                        titles[column]
                    );
                }
            }
        }
    }
}

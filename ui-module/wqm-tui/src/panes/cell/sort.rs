//! Sorting, and the room its mark needs.
//!
//! Split out of [`super::table`] when the sort ruling took that file past its size limit. The
//! division is a real one rather than a place to put the overflow: everything here exists
//! *because* a table can be sorted, and nothing in it would exist otherwise. [`compare`] is the
//! whole semantic content of sorting — three rules, one per shape of value — and [`grown`] is
//! the whole of what the `↑`/`↓` mark costs the layout; both are far easier to argue with
//! sitting on their own than buried under column arithmetic.

use ratatui::layout::Rect;

use super::value::Cell;

/// Which way a sorted column runs. Chris's own sequence (2026-09-07): *"first press
/// ascending, second press descending"*.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Asc,
    Desc,
}

impl Direction {
    /// The mark the sorted column's header carries after its name.
    pub const fn glyph(self) -> &'static str {
        match self {
            Direction::Asc => "↑",
            Direction::Desc => "↓",
        }
    }
}

/// Which column a table is sorted by, and which way.
///
/// One value rather than two fields on the table, because a direction with no column is not a
/// state a sorted table can be in — and a table that held them separately would be free to
/// represent it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Sort {
    pub column: usize,
    pub direction: Direction,
}

/// How two values of one column compare.
///
/// Three rules, one per shape of value, and each is the reading the column's own alignment
/// already implies: **figures numerically** (so 118 sorts below 2'790 rather than above it,
/// which is what a string comparison would say), **text case-insensitively** (a list a person
/// reads is alphabetical, not ASCII), and **a queue triple by its first number** — the pending
/// count, which is the one that says how much work is waiting.
///
/// The Queue tab added two more, and both are the same rule read again: **a value is ordered by
/// what it MEANS, never by how it prints.** A [`Cell::Measured`] compares on the magnitude it
/// carries, so `4.0 MB` sorts above `903.5 KB` where a string comparison would file it under
/// `9`; a [`Cell::Tinted`] compares on its word, case-insensitively, exactly as text does —
/// its hue is a fact about the state, not a rank, and ordering by colour would invent one.
///
/// Two values of different shapes compare equal. A column holding both is a defect in the
/// projection that built it, and inventing an order between a name and a number would hide it.
pub(crate) fn compare(a: &Cell, b: &Cell) -> std::cmp::Ordering {
    match (a, b) {
        (Cell::Num(a), Cell::Num(b)) => a.cmp(b),
        (Cell::Text(a), Cell::Text(b)) => a.to_lowercase().cmp(&b.to_lowercase()),
        (Cell::Queue { pending: a, .. }, Cell::Queue { pending: b, .. }) => a.cmp(b),
        (Cell::Measured { order: a, .. }, Cell::Measured { order: b, .. }) => a.cmp(b),
        (Cell::Tinted { text: a, .. }, Cell::Tinted { text: b, .. }) => {
            a.to_lowercase().cmp(&b.to_lowercase())
        }
        _ => std::cmp::Ordering::Equal,
    }
}

/// The rect a column header is drawn into: its own column, grown RIGHTWARDS into the column
/// gap when the sort mark makes the header wider than the column.
///
/// **The mark must not cost the table any width.** `Files` is a five-column field holding a
/// five-character title, so there is nowhere inside it for a `↓` — and the answers that do not
/// work are worth naming, because each looks reasonable until it is drawn. Clipping loses the
/// mark silently, which is the one thing a sort indicator may never do. Widening the column
/// permanently would take those columns from the flex column on every cell, sorted or not,
/// undoing what removing the marker gutter won back (R9) and starving `Active Projects` down to
/// a zero-width `Name` at 80 × 24. Widening it only while sorted would move every column on the
/// row the moment a key was pressed.
///
/// So the header borrows the [`COLUMN_GAP`] to its right — one column of deliberate whitespace
/// that no value is ever drawn into — and never the column to its left. Leftwards was tried and
/// is why this is written down: `Files` grown two columns left reached back through the gap into
/// `Bch` and rendered `BcFiles ↓`, a neighbour silently truncated to make room for a mark. The
/// mark is therefore appended with **no leading space**, so one gap column is always enough:
/// the widest overflow any column in this crate has is exactly one.
///
/// Clamped to the table's own body, so the last column can never grow off the end of the cell.
/// A last column must hold its own mark, which is why `Sync` is five columns wide for a
/// four-letter title — `views::dashboard::tests::sort` guards that every sortable column can
/// actually show its mark, and that none of them clips a neighbour to do it.
pub(crate) fn grown(area: Rect, body: Rect, needed: u16) -> Rect {
    if needed <= area.width {
        return area;
    }
    let end = (area.x + needed).min(body.x + body.width);
    Rect {
        width: end.saturating_sub(area.x),
        ..area
    }
}

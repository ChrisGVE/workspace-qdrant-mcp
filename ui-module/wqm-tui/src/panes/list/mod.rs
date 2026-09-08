//! The full-width list — one table filling a tab's body, and the shape every list tab takes.
//!
//! A [`crate::panes::cell::CellPane`] is a *projection*: a narrow view of something a whole tab
//! is dedicated to. This is the tab. The two share their column vocabulary
//! ([`crate::panes::cell::Column`], [`crate::panes::cell::Cell`],
//! [`crate::panes::cell::Sort`]) because a reader who learns what a column header means on the
//! Dashboard must not have to learn it again on the Queue — and they differ in exactly four
//! places, each of which is a consequence of a list being the whole screen rather than a sixth
//! of it.
//!
//! | | cell | list |
//! |---|---|---|
//! | header | header rung, not bold | **bold**, like a Dashboard section heading |
//! | sort keys offered | only while the cell is the live one | always, whenever there is more than one row |
//! | data cursor | row 1 of the live cell | any row, and it scrolls the list to stay visible |
//! | overflow | a `… N more` tail on the last line | the list scrolls; there is no tail |
//!
//! # The header is bold because the list has nothing else to be a heading
//!
//! Chris, 2026-09-07: the column headers are *"similar to the section headers of the
//! dashboard"*. A Dashboard cell carries a [`crate::widgets::chrome::ZoneHeading`] above its
//! column header, so the header can sit at the quiet [`crate::tokens::header`] rung and still
//! read as structure — the bold line above it has already said "this is a zone". A list has no
//! such line: its column header is the first thing on the screen after the status block, and at
//! the header rung alone it reads as a faint first row of data.
//!
//! # There is no `Attention` here, and that is the same fact stated again
//!
//! A cell asks the screen which zone is live, because six cells share one cursor. A list IS its
//! screen's only zone, so it is always the live one — a `sortable` flag taken from a view would
//! be a value that could only ever be `true`, and a state that cannot vary is not a state.
//!
//! # No frame
//!
//! Chris, 2026-09-07: *"no frame around the table, valid for all views"*. v0.1 draws its queue
//! inside a `┌ Queue ┐` box; VL §6 already forbids it — a box means a modal or a toast — and
//! the box was also spending two columns and two rows on saying what the tab bar says.

use ratatui::layout::Rect;

pub mod help;
pub mod table;
#[cfg(feature = "tui-pantry")]
pub mod ingredient;
#[cfg(test)]
mod tests;

pub use table::ListPane;

/// How many rows a list holds in memory at once — the **buffer page**.
///
/// Chris, 2026-09-07: *"configurable, but valid for all lists"*. Two halves, and both matter:
///
/// - **Valid for all lists.** One number, here, rather than one per tab. A Queue that paged 200
///   at a time beside a Logs tab that paged 500 would teach the reader nothing transferable,
///   and `press Enter to load 200 more rows` would be a sentence they had to re-read on every
///   screen.
/// - **Configurable.** It is a `const` today because this crate is a storyboard and has no
///   settings store to read from. When one exists the key is **`tui.list_page`**, named here so
///   the future config table and this constant are already talking about the same thing.
///
/// Two hundred is v0.1's own number — its Queue tab reports `(200 items)` — so the frames
/// reproduce a page a reader has already seen rather than a page this crate invented.
pub const LIST_PAGE: usize = 200;

/// The paging keys, spelled once for every list's help modal — [`help::navigation`] consumes
/// these two pairs rather than re-spelling the chords.
///
/// Chris, 2026-09-07: these are *"shown only in the help… valid for all lists including the
/// dashboard"*. Both halves are load-bearing. **Help-only**, because the foot line is the
/// screen's scarcest row and a key that does the obvious thing faster does not earn a place on
/// it — `↓↑/jk Navigate` already tells the reader the list moves. **Every list**, because the
/// day the Dashboard grows a help modal it must offer the same pairs in the same words, and
/// the only way to guarantee that is for there to be one copy of the words.
///
/// The spelling is v0.1's own, read off its help modal: `^d/^u ^f/^b  Half / full page
/// down/up`. What is added is the two names a reader who has never used vim already knows —
/// `PgDn`/`PgUp` — because `^F` is discoverable only to someone who already knows it.
pub const NAV_HELP: [(&str, &str); 2] = [
    ("^D/^U", "Half page down/up"),
    ("^F/^B, PgDn/PgUp", "Full page down/up"),
];

/// One row of an area, by index — the same helper [`crate::views::top::row`] is, kept local so a
/// pane does not reach into a view for arithmetic.
pub(crate) fn line(area: Rect, n: u16) -> Rect {
    Rect {
        y: area.y + n,
        height: 1,
        ..area
    }
}

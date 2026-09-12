//! The help window's two-level shape, shared by every list.
//!
//! A list's help was one flat list of `(key, what)` pairs. It is now a list of
//! [`HelpSection`]s — each a named group with its own entries — because a reader looking for one
//! thing (how do I page down) should not have to read every key the screen has to find it. The
//! sections, their order, and the two that every list shares ([`navigation`] and [`general`])
//! live here, so the day the Dashboard grows a help modal it offers the same words in the same
//! order rather than its own recollection of them.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use super::NAV_HELP;
use crate::tokens;

/// One named group of key entries in a list's help window.
pub struct HelpSection {
    /// The heading this section prints on its own line.
    pub title: &'static str,
    /// The `(key, what)` pairs, drawn in this order.
    pub entries: Vec<(&'static str, &'static str)>,
    /// A prose line printed after this section's entries — per-view data (for the Queue, the
    /// columns the search and the filter look at), so it is carried on the section rather than
    /// invented by the shared renderer.
    pub note: Option<&'static str>,
}

/// The keys that move the list — shared by every list.
///
/// The two paging pairs are read from [`NAV_HELP`], the single source every list's help consumes
/// rather than re-spells, so the day a chord changes there is one place to change it. The up and
/// down arrows are the crate's own spelling — `↓` and `↑`, as in the foot's `↓↑/jk`.
pub fn navigation() -> HelpSection {
    HelpSection {
        title: "Navigation",
        entries: vec![
            ("↓ / j", "Down one row"),
            ("↑ / k", "Up one row"),
            NAV_HELP[0],
            NAV_HELP[1],
            // `n` and `N` left this section on 20260912 (Chris, ruling 6): they move between
            // SEARCH HITS, which is a thing the search does, and a reader looking for them looks
            // where the search is.
            ("#<move>", "Repeat the move # times"),
            ("Home / gg", "Top"),
            ("End / G", "Bottom"),
            ("#g", "Go to row #"),
            ("r", "Relative row numbers, on and off"),
        ],
        note: None,
    }
}

/// How a table is sorted — shared by every list, because the rule is the surface's.
///
/// Chris, 20260912, ruling 6: *"a column's highlighted letter cycles ascending → descending → no
/// sort, and Shift+letter cycles the other way round."*
///
/// ⚠ **The cycle is not modelled in state yet.** Sorting reaches a table today by construction
/// — `CellTable::sorted(Sort { .. })`, which is what the frames and the pantry set — and no
/// keypress advances it, because this crate is a storyboard and its state machines cover only
/// what a frame has needed. So this section states the BINDING, which is what the ruling is
/// about and what the header's lit letter already promises; the state machine behind it is
/// owed. Written down here rather than left to be noticed, because a help window is the one
/// place a promise the code does not keep is invisible.
pub fn sorting() -> HelpSection {
    HelpSection {
        title: "Sorting",
        entries: vec![
            ("<letter>", "Sort by that column: ascending, descending, then off"),
            ("\u{21e7}<letter>", "The same cycle reversed: descending, ascending, then off"),
        ],
        note: Some("A column that can be sorted lights one letter of its own name."),
    }
}

/// What takes the focus — shared by every list.
///
/// Chris, 20260912, ruling 6: *"number keys change tab, and where several areas can take focus
/// one letter of each name is accented."* The second half is the Dashboard's rule stated
/// generally: a screen divided into zones accents the letter that jumps to each one, and a zone
/// that has receded still lights its letter, which is what keeps it reachable.
pub fn focus() -> HelpSection {
    HelpSection {
        title: "Focus",
        entries: vec![
            ("1 … 9, 10", "Change tab"),
            ("<letter>", "Go to an area of this screen"),
        ],
        note: Some("Where several areas take focus, one letter of each name is accented."),
    }
}

/// The keys every list's window offers — shared by every list.
pub fn general() -> HelpSection {
    HelpSection {
        title: "General",
        entries: vec![
            ("?", "Help"),
            ("q", "Quit"),
            ("Esc", "Close the current overlay window"),
        ],
        note: None,
    }
}

/// The modifier glyphs this surface spells keys with, and what each one names.
///
/// Chris, 20260912, ruling 6: a row at the foot of the help explaining the glyphs for control,
/// shift, command and option — *"the shift glyph must NOT be the same arrow as the up-arrow
/// key."* It is `⇧` U+21E7, a hollow outline; the up-arrow key is `↑` U+2191, the same solid
/// arrow the foot draws in `↓↑/jk`. Two different marks, and the legend is what tells a reader
/// which is which the first time they meet one.
///
/// Stated as a table rather than as a sentence so a key entry and this legend cannot come to
/// disagree about a glyph: an entry that spells a chord writes the character, and this names
/// the character it wrote.
pub const MODIFIERS: [(&str, &str); 4] = [
    ("\u{2303}", "control"),
    ("\u{21e7}", "shift"),
    ("\u{2318}", "command"),
    ("\u{2325}", "option"),
];

/// The legend row: every glyph in [`MODIFIERS`] with its name beside it, glyphs accented like
/// any other key and the names muted, three spaces between pairs.
pub fn legend() -> Line<'static> {
    let mut spans = Vec::new();
    for (at, (glyph, name)) in MODIFIERS.iter().enumerate() {
        if at > 0 {
            spans.push(Span::styled("   ", tokens::muted_style()));
        }
        spans.push(Span::styled(*glyph, key_style()));
        spans.push(Span::styled(format!(" {name}"), tokens::muted_style()));
    }
    Line::from(spans)
}

/// What a key is drawn in: the accent hue, the unclaimed §10 field, which is what a sortable
/// column's lit letter already wears.
///
/// Chris, 20260912, ruling 6: *"keys in the accent hue"*. The same field rather than the
/// reserved selector, and for the same reason the sort key uses it — a key you may press is a
/// hint, not a selection.
fn key_style() -> Style {
    Style::default().fg(tokens::accent())
}

/// Lay `sections` out into a help window's body lines.
///
/// Each section title is on its own line and **bold**; each entry is indented two spaces with
/// its key in the accent hue, padded to the width of the widest key ACROSS ALL SECTIONS —
/// measured here rather than written down, so a chord that grows in another module cannot
/// silently stop fitting — and a section's note, when it has one, is printed after its entries
/// in *italic*. A blank line separates sections; then `tail` — a closing sentence about the
/// whole window rather than about any one section — and then the [`legend`], which is always
/// the last row.
///
/// The tail is a PARAMETER rather than something a caller appends afterwards, so the legend
/// cannot end up in the middle of a window: ruling 6 asks for it *"on a bottom row"*, and a
/// caller free to push lines after it is a caller free to get that wrong.
///
/// The three treatments are ruling 6's (Chris, 20260912) and they are three different KINDS of
/// thing rather than decoration: a title names a group, a key is something to press, and a note
/// is prose about the group rather than an entry in it. The italic is what stops a note being
/// read as a nameless entry, which is how it read while everything was one colour.
pub fn render(sections: &[HelpSection], tail: Option<&str>) -> Vec<Line<'static>> {
    let column = sections
        .iter()
        .flat_map(|section| section.entries.iter())
        .map(|(key, _)| key.chars().count())
        .max()
        .unwrap_or(0)
        + 2;

    let mut body = Vec::new();
    for (at, section) in sections.iter().enumerate() {
        if at > 0 {
            body.push(Line::default());
        }
        body.push(Line::from(Span::styled(
            section.title.to_string(),
            tokens::normal_style().add_modifier(Modifier::BOLD),
        )));
        for (key, what) in &section.entries {
            body.push(Line::from(vec![
                Span::styled("  ", tokens::normal_style()),
                // The key and its padding are ONE span, so the accent stops where the key does
                // and the gap after it is not a coloured run of blanks — invisible on most
                // terminals and not on all of them.
                Span::styled(key.to_string(), key_style()),
                Span::styled(
                    " ".repeat(column - key.chars().count()),
                    tokens::normal_style(),
                ),
                Span::styled(what.to_string(), tokens::normal_style()),
            ]));
        }
        if let Some(note) = section.note {
            body.push(Line::from(Span::styled(
                note.to_string(),
                tokens::normal_style().add_modifier(Modifier::ITALIC),
            )));
        }
    }
    if let Some(tail) = tail {
        body.push(Line::default());
        body.push(Line::from(Span::styled(
            tail.to_string(),
            tokens::normal_style().add_modifier(Modifier::ITALIC),
        )));
    }
    body.push(Line::default());
    body.push(legend());
    body
}

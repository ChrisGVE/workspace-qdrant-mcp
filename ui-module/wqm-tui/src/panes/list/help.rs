//! The help window's two-level shape, shared by every list.
//!
//! A list's help was one flat list of `(key, what)` pairs. It is now a list of
//! [`HelpSection`]s — each a named group with its own entries — because a reader looking for one
//! thing (how do I page down) should not have to read every key the screen has to find it. The
//! sections, their order, and the two that every list shares ([`navigation`] and [`general`])
//! live here, so the day the Dashboard grows a help modal it offers the same words in the same
//! order rather than its own recollection of them.

use super::NAV_HELP;

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
            ("n", "Next hit, while a search is on"),
            ("N", "Previous hit, while a search is on"),
            ("#<move>", "Repeat the move # times"),
            ("Home / gg", "Top"),
            ("End / G", "Bottom"),
            ("#g", "Go to row #"),
            ("r", "Relative row numbers, on and off"),
        ],
        note: None,
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

/// Lay `sections` out into a help window's body lines.
///
/// Each section title is on its own line; each entry is indented two spaces, with the key padded
/// to the width of the widest key ACROSS ALL SECTIONS — measured here rather than written down,
/// so a chord that grows in another module cannot silently stop fitting — and a section's note,
/// when it has one, is printed after its entries. A blank line separates sections.
pub fn render(sections: &[HelpSection]) -> Vec<String> {
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
            body.push(String::new());
        }
        body.push(section.title.to_string());
        for (key, what) in &section.entries {
            body.push(format!("  {key:<column$}{what}"));
        }
        if let Some(note) = section.note {
            body.push(note.to_string());
        }
    }
    body
}

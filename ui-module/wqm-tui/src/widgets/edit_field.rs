//! The edit-in-place field: a value being typed, its mode, and the caret between the two halves.
//!
//! Written for [`crate::widgets::config_table`] and lifted out when the Queue tab's dialog slot
//! needed the same field. Lifted rather than copied, and the difference is the whole reason this
//! file exists: a second caret would be a second answer to *what does typing look like here*,
//! and §3 gives that answer once — a lighter fill than the cursor row, a bar between two
//! characters in insert, a reversed block on one in normal.
//!
//! # What is here, and what deliberately is not
//!
//! The **caret and the two halves of the text** are here, because they are the same everywhere.
//! The **fill, the underline and the width** are the caller's: a config table's field is one
//! cell of a grid and stops at its column; the Queue's runs to the end of the row. So this hands
//! back spans in whatever [`Style`] it is given, and says nothing about how far they reach.
//!
//! # Visual joined the two, and where the live state comes from (2026-09-13)
//!
//! Ruling B (Chris, 21:42) made the editor an operator grammar, and the adoption of `modalkit`
//! (21:57) put a real engine under every text input — [`crate::editor::Field`]. This file is
//! now the **render snapshot** that engine derives via `Field::edit()`, and still the value a
//! storyboard frame constructs by hand: `Edit::insert("2000")` needs no machine behind it. The
//! third mode, [`EditMode::Visual`], reverses the selected characters; a conventional
//! shift-selection is the same paint without a mode, so an insert-mode edit may carry a
//! [`Edit::selection`] too. Whatever the mode, [`caret_spans`] returns exactly three spans.
use ratatui::{
    style::{Modifier, Style},
    text::Span,
};
use std::ops::Range;

/// Which vim mode the edit is in. §3 gives each its own caret and all the same indicator
/// treatment: **bold, no hue** — cyan belongs to the selector and this must not read as one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EditMode {
    Insert,
    Normal,
    Visual,
}

impl EditMode {
    /// The status-line indicator. One producer, so the two surfaces cannot drift.
    pub const fn indicator(self) -> &'static str {
        match self {
            EditMode::Insert => "-- INSERT --",
            EditMode::Normal => "-- NORMAL --",
            EditMode::Visual => "-- VISUAL --",
        }
    }

    /// §3, and Chris's r06 mark #8 — *"this color is already used to indicate the
    /// selection"*. Weight carries it; hue is not available to this element.
    pub fn indicator_span(self) -> Span<'static> {
        Span::styled(
            self.indicator(),
            Style::default().add_modifier(Modifier::BOLD),
        )
    }
}

/// An edit in progress: the text as it stands, the mode, and where the caret is.
///
/// The caret is a **character** index and may equal the value's length — that is the caret
/// past the last character, which insert mode reaches on every keystroke.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Edit {
    value: String,
    mode: EditMode,
    caret: usize,
    selection: Option<Range<usize>>,
}

impl Edit {
    /// Insert mode with the caret past the last character — where typing leaves it.
    pub fn insert(value: impl Into<String>) -> Self {
        let value = value.into();
        let caret = value.chars().count();
        Self {
            value,
            mode: EditMode::Insert,
            caret,
            selection: None,
        }
    }

    /// Insert mode with the caret placed by hand, clamped to the value's length.
    pub fn insert_at(value: impl Into<String>, caret: usize) -> Self {
        let value = value.into();
        let caret = caret.min(value.chars().count());
        Self {
            value,
            mode: EditMode::Insert,
            caret,
            selection: None,
        }
    }

    /// Normal mode: the caret is a block **on** a character rather than a bar between two.
    pub fn normal(value: impl Into<String>, caret: usize) -> Self {
        let value = value.into();
        let caret = caret.min(value.chars().count());
        Self {
            value,
            mode: EditMode::Normal,
            caret,
            selection: None,
        }
    }

    /// Visual mode with a selected character range, clamped to the value's length.
    pub fn visual(value: impl Into<String>, range: Range<usize>) -> Self {
        let value = value.into();
        let len = value.chars().count();
        let start = range.start.min(len);
        let end = range.end.clamp(start, len);
        let caret = end.saturating_sub(1);
        Self {
            value,
            mode: EditMode::Visual,
            caret,
            selection: Some(start..end),
        }
    }

    pub fn value(&self) -> &str {
        &self.value
    }

    pub fn mode(&self) -> EditMode {
        self.mode
    }

    pub fn caret(&self) -> usize {
        self.caret
    }

    pub fn selection(&self) -> Option<Range<usize>> {
        self.selection.clone()
    }

    /// Set the selected character range in a live field's render snapshot.
    pub(crate) fn set_selection(&mut self, selection: Option<Range<usize>>) {
        let len = self.value.chars().count();
        self.selection = selection.map(|range| {
            let start = range.start.min(len);
            let end = range.end.clamp(start, len);
            start..end
        });
    }
}

fn selected_spans(chars: &[char], range: &Range<usize>, style: Style) -> Vec<Span<'static>> {
    let before: String = chars[..range.start].iter().collect();
    let selected: String = chars[range.clone()].iter().collect();
    let after: String = chars[range.end..].iter().collect();
    vec![
        Span::styled(before, style),
        Span::styled(selected, style.add_modifier(Modifier::REVERSED)),
        Span::styled(after, style),
    ]
}

/// The three spans an edit draws in `style`: before, caret or selection, and after.
///
/// Always three, even when a half is empty, so a caller measuring the result gets the same
/// arithmetic whatever the caret index is — the reason the config table's column stopped
/// shifting when a row went into edit.
pub fn caret_spans(edit: &Edit, style: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = edit.value().chars().collect();
    match edit.mode() {
        // A bar between two characters — vim insert.
        EditMode::Insert => {
            // Modeless shift-selection stays in Insert mode and paints the selected characters.
            if let Some(range) = edit.selection.as_ref() {
                return selected_spans(&chars, range, style);
            }
            let before: String = chars[..edit.caret()].iter().collect();
            let after: String = chars[edit.caret()..].iter().collect();
            vec![
                Span::styled(before, style),
                Span::styled("▏", style),
                Span::styled(after, style),
            ]
        }
        // A block ON a character — vim normal. Past the last character there is no character to
        // reverse, so the block falls on the space where one would go.
        EditMode::Normal => {
            let before: String = chars[..edit.caret()].iter().collect();
            let after: String = chars[edit.caret()..].iter().collect();
            let mut rest = after.chars();
            let under = rest.next().unwrap_or(' ');
            vec![
                Span::styled(before, style),
                Span::styled(under.to_string(), style.add_modifier(Modifier::REVERSED)),
                Span::styled(rest.collect::<String>(), style),
            ]
        }
        EditMode::Visual => {
            // Before, selected, after: Visual preserves the three-span invariant. A visual
            // snapshot without a range — the engine reporting visual mode before any paint
            // shows one — draws the block on the caret, which is what vim shows the instant
            // `v` is pressed.
            let caret = edit.caret().min(chars.len());
            let fallback = caret..(caret + 1).min(chars.len());
            let range = edit.selection.clone().unwrap_or(fallback);
            selected_spans(&chars, &range, style)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visual_reverses_selected_characters_in_three_spans() {
        let spans = caret_spans(&Edit::visual("2000", 1..3), Style::default());
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].content, "2");
        assert_eq!(spans[1].content, "00");
        assert_eq!(spans[2].content, "0");
        assert!(spans[1].style.add_modifier.contains(Modifier::REVERSED));
    }

    #[test]
    fn modeless_selection_reverses_characters_while_mode_stays_insert() {
        let mut edit = Edit::insert_at("ab", 1);
        edit.set_selection(Some(1..2));
        let spans = caret_spans(&edit, Style::default());
        assert_eq!(edit.mode(), EditMode::Insert);
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[1].content, "b");
        assert!(spans[1].style.add_modifier.contains(Modifier::REVERSED));
    }
}

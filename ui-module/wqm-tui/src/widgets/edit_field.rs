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
use ratatui::{
    style::{Modifier, Style},
    text::Span,
};

/// Which vim mode the edit is in. §3 gives each its own caret and both the same indicator
/// treatment: **bold, no hue** — cyan belongs to the selector and this must not read as one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EditMode {
    Insert,
    Normal,
}

impl EditMode {
    /// The status-line indicator. One producer, so the two surfaces cannot drift.
    pub const fn indicator(self) -> &'static str {
        match self {
            EditMode::Insert => "-- INSERT --",
            EditMode::Normal => "-- NORMAL --",
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
}

/// The three spans an edit draws in `style`: everything before the caret, the caret, everything
/// after it.
///
/// Always three, even when a half is empty, so a caller measuring the result gets the same
/// arithmetic whatever the caret index is — the reason the config table's column stopped
/// shifting when a row went into edit.
pub fn caret_spans(edit: &Edit, style: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = edit.value().chars().collect();
    let before: String = chars[..edit.caret()].iter().collect();
    let after: String = chars[edit.caret()..].iter().collect();

    let mut spans = vec![Span::styled(before, style)];
    match edit.mode() {
        // A bar between two characters — vim insert.
        EditMode::Insert => {
            spans.push(Span::styled("▏", style));
            spans.push(Span::styled(after, style));
        }
        // A block ON a character — vim normal. Past the last character there is no character to
        // reverse, so the block falls on the space where one would go.
        EditMode::Normal => {
            let mut rest = after.chars();
            let under = rest.next().unwrap_or(' ');
            spans.push(Span::styled(
                under.to_string(),
                style.add_modifier(Modifier::REVERSED),
            ));
            spans.push(Span::styled(rest.collect::<String>(), style));
        }
    }
    spans
}

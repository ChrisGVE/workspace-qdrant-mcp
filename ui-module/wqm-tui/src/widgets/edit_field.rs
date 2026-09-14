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
        // **A named pair, not `REVERSED`** — item 3 asks for *"a color for the selected text"*,
        // and a modifier is not one: it swaps whatever is under it, so inside an edit field the
        // selection came out wearing the active field's own fill. See
        // `tokens::field::SelectedText` for the two arms and what each costs.
        Span::styled(
            selected,
            style
                .bg(crate::tokens::field::selected_text_bg())
                .fg(crate::tokens::field::selected_text_fg()),
        ),
        Span::styled(after, style),
    ]
}

/// Which keymap the field is under, which is what decides whether a caret is drawn at all.
///
/// Chris, item 3: *"When we are EMACS style, the cursor is the terminal default cursor, when we
/// are in vim-mode, the cursor in normal and visual mode is a block cursor, noblink in normal
/// and blink in visual, in insert mode the cursor is a single blinking line."*
///
/// So the two keymaps differ in a way no colour can express: under vim the program OWNS the
/// caret and paints it into the cell; under the conventional keymap the caret is the terminal's
/// own, positioned by `Frame::set_cursor_position` and drawn by the terminal in whatever shape
/// the user configured. A painted caret there would be a second cursor beside the real one.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Caret {
    /// Vim: the program paints the caret, shaped and blinking by mode.
    #[default]
    Painted,
    /// Conventional/Emacs: the terminal's own cursor. Nothing is painted, and a still frame
    /// therefore shows the field with no caret in it — which is the honest depiction, since
    /// there is no terminal in a headless render to draw one.
    Terminal,
}

/// **Blink, in a still frame.**
///
/// A still cannot blink, and drawing a "blinking" caret as some third glyph would be inventing a
/// shape the running program never shows. So the real `SLOW_BLINK` attribute is set — the one a
/// terminal actually honours — and the frame carries it as a CELL ATTRIBUTE rather than as
/// anything visible.
///
/// That makes it judgeable by the right instrument and not by the wrong one: the attribute is
/// there to be read in a grid/cell dump, and a PNG of the frame cannot show it and should not be
/// asked to. Same division as wqm#283, where the `●` glyph is right in the dump and missing from
/// the capture — judge shape and attributes from the dump, colour from the pixels.
fn blink() -> Modifier {
    Modifier::SLOW_BLINK
}

/// The three spans an edit draws in `style`, under the painted (vim) caret.
///
/// Always three, even when a half is empty, so a caller measuring the result gets the same
/// arithmetic whatever the caret index is — the reason the config table's column stopped
/// shifting when a row went into edit.
pub fn caret_spans(edit: &Edit, style: Style) -> Vec<Span<'static>> {
    caret_spans_with(edit, style, Caret::Painted)
}

/// [`caret_spans`] under a stated keymap.
///
/// Still three spans under both, so the column arithmetic downstream is unchanged — under
/// [`Caret::Terminal`] the middle span is simply the text with no caret treatment on it.
pub fn caret_spans_with(edit: &Edit, style: Style, caret: Caret) -> Vec<Span<'static>> {
    if caret == Caret::Terminal {
        // The selection is still ours to paint — shift-selection exists under both keymaps, and
        // a terminal draws a cursor, never a range.
        if let Some(range) = edit.selection.as_ref() {
            let chars: Vec<char> = edit.value().chars().collect();
            return selected_spans(&chars, range, style);
        }
        let chars: Vec<char> = edit.value().chars().collect();
        let before: String = chars[..edit.caret()].iter().collect();
        let after: String = chars[edit.caret()..].iter().collect();
        return vec![
            Span::styled(before, style),
            Span::styled(String::new(), style),
            Span::styled(after, style),
        ];
    }
    painted_caret_spans(edit, style)
}

fn painted_caret_spans(edit: &Edit, style: Style) -> Vec<Span<'static>> {
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
                Span::styled("▏", style.add_modifier(blink())),
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
                // Normal mode is the one caret that does NOT blink (Chris, item 3).
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
            // Visual's block BLINKS where normal's does not (Chris, item 3), and that is the
            // only thing separating the two block carets — so the attribute goes on the
            // selection span, which is where the block is.
            let mut spans = selected_spans(&chars, &range, style);
            if let Some(block) = spans.get_mut(1) {
                block.style = block.style.add_modifier(blink());
            }
            spans
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Selection is a NAMED colour pair now, not `REVERSED` — item 3, *"we'll have to define a
    /// color for the selected text"*. The span split is unchanged; only what is on it moved.
    #[test]
    fn visual_paints_selected_characters_in_three_spans() {
        // Reads `tokens::field`, which is process-global: without this the expected
        // colour and the painted one can be computed under two different themes.
        let _serial = crate::global_state_lock();
        let spans = caret_spans(&Edit::visual("2000", 1..3), Style::default());
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0].content, "2");
        assert_eq!(spans[1].content, "00");
        assert_eq!(spans[2].content, "0");
        assert_eq!(
            spans[1].style.bg,
            Some(crate::tokens::field::selected_text_bg())
        );
        assert_eq!(
            spans[1].style.fg,
            Some(crate::tokens::field::selected_text_fg())
        );
        assert!(
            !spans[1].style.add_modifier.contains(Modifier::REVERSED),
            "a modifier cannot be the selection colour: it inverts whatever fill is under it, \
             which inside an edit field is the active field's own"
        );
    }

    /// Visual's block blinks and normal's does not, which is the only difference between the two
    /// block carets (Chris, item 3). The attribute is the real one a terminal honours, so a
    /// still frame carries it in the cell rather than depicting it.
    #[test]
    fn only_the_visual_block_blinks() {
        // Reads `tokens::field`, which is process-global: without this the expected
        // colour and the painted one can be computed under two different themes.
        let _serial = crate::global_state_lock();
        let visual = caret_spans(&Edit::visual("2000", 1..3), Style::default());
        let normal = caret_spans(&Edit::normal("2000", 1), Style::default());
        assert!(visual[1].style.add_modifier.contains(Modifier::SLOW_BLINK));
        assert!(!normal[1].style.add_modifier.contains(Modifier::SLOW_BLINK));
        let insert = caret_spans(&Edit::insert("2000"), Style::default());
        assert!(
            insert[1].style.add_modifier.contains(Modifier::SLOW_BLINK),
            "the insert bar blinks too"
        );
    }

    /// Under the conventional keymap the terminal owns the caret, so nothing is painted — and
    /// the three-span shape survives, because the column arithmetic downstream depends on it.
    #[test]
    fn the_conventional_keymap_paints_no_caret_at_all() {
        let spans = caret_spans_with(&Edit::insert("2000"), Style::default(), Caret::Terminal);
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[1].content, "", "no caret glyph is drawn");
        let text: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(text, "2000", "and the value is whole");
    }

    /// A shift-selection is still OURS to paint under the conventional keymap: a terminal draws
    /// a cursor, never a range.
    #[test]
    fn a_selection_is_painted_even_where_the_terminal_owns_the_caret() {
        // Reads `tokens::field`, which is process-global: without this the expected
        // colour and the painted one can be computed under two different themes.
        let _serial = crate::global_state_lock();
        let mut edit = Edit::insert_at("ab", 1);
        edit.set_selection(Some(1..2));
        let spans = caret_spans_with(&edit, Style::default(), Caret::Terminal);
        assert_eq!(spans[1].content, "b");
        assert_eq!(
            spans[1].style.bg,
            Some(crate::tokens::field::selected_text_bg())
        );
    }

    #[test]
    fn modeless_selection_is_painted_while_the_mode_stays_insert() {
        // Reads `tokens::field`, which is process-global: without this the expected
        // colour and the painted one can be computed under two different themes.
        let _serial = crate::global_state_lock();
        let mut edit = Edit::insert_at("ab", 1);
        edit.set_selection(Some(1..2));
        let spans = caret_spans(&edit, Style::default());
        assert_eq!(edit.mode(), EditMode::Insert);
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[1].content, "b");
        assert_eq!(
            spans[1].style.bg,
            Some(crate::tokens::field::selected_text_bg())
        );
    }
}

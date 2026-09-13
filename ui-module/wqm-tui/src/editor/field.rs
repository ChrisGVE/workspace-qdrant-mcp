//! The live state under one text input: a `modalkit` buffer, one of the two [`Keymap`]s, and
//! the rules that sit BETWEEN a key and the engine.
//!
//! Adopted 2026-09-13 21:57 (Chris: *"Yes A, go ahead and adopt it"*). A [`Field`] owns the
//! three things every editable field needs — the `Store`, the binding machine and the
//! `TextBoxState` — and drives them the way `acceptance.rs` drives the probe: a key goes in,
//! actions come out, each is dispatched onto the text box. What this file adds is the part
//! of the 20:30 composition ruling that the engine cannot know:
//!
//! - **`Tab` / `Shift-Tab` are never fed.** Field traversal is the container's, so they come
//!   back as [`Outcome::Unhandled`].
//! - **`Esc` is two-stage in vim and one-stage otherwise.** In vim the first `Esc` returns to
//!   normal mode (fed to the engine) and the second — pressed while already in normal —
//!   [`Outcome::Leave`]s. The conventional table has no modes, so `Esc` leaves at once.
//! - **`Enter` submits a single-line field** ([`Outcome::Submit`]); a multi-line body feeds
//!   it and gets a newline. The director's call, cheap to overturn.
//!
//! # The snapshot, and the one render-read in the crate
//!
//! [`Field::edit`] derives the [`Edit`] the widgets draw: the text, the caret from the
//! engine's cursor, and the mode from the keymap. The selection is the exception. The
//! adapter keeps its `CursorGroupId` private and the id's field is private too, so no public
//! call returns the selection extent; this file renders the text box into a scratch
//! one-line `Buffer` and reads the run of `REVERSED` cells instead. It is the only place the
//! crate reads a paint rather than state, it is fenced inside one function, and an upstream
//! accessor should replace it the day one exists.
//!
//! # Undo in the conventional table needs a checkpoint
//!
//! The Emacs table does not checkpoint typed text before an undo, so `Ctrl-Z` then `Ctrl-Y`
//! would lose the redo. The field seals a checkpoint before dispatching a conventional undo;
//! the vim table checkpoints itself on leaving insert mode and needs nothing.

use modalkit::{
    actions::{Action, Editable, EditorAction, HistoryAction, Jumpable, Scrollable},
    crossterm::event::{KeyCode, KeyEvent},
    editing::{application::EmptyInfo, context::Resolve, store::Store},
    key::TerminalKey,
};
use modalkit_ratatui::textbox::{TextBox, TextBoxState};
use ratatui::{buffer::Buffer, layout::Rect, style::Modifier, widgets::StatefulWidget};

use super::keymap::Keymap;
use crate::widgets::edit_field::{Edit, EditMode};

pub struct Field {
    store: Store<EmptyInfo>,
    keymap: Keymap,
    tbox: TextBoxState<EmptyInfo>,
    single_line: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    Handled,
    Leave,
    Submit,
    Unhandled,
}

impl Field {
    fn new(text: &str, keymap: Keymap) -> Self {
        let mut store = Store::default();
        let mut tbox = TextBoxState::new(store.load_buffer("field".to_string()));
        tbox.set_text(text);
        Self {
            store,
            keymap,
            tbox,
            single_line: true,
        }
    }

    pub fn vim(text: &str) -> Self {
        Self::new(text, Keymap::vim())
    }

    pub fn conventional(text: &str) -> Self {
        Self::new(text, Keymap::conventional())
    }

    pub fn multi_line(mut self) -> Self {
        self.single_line = false;
        self
    }

    pub fn key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Tab | KeyCode::BackTab => return Outcome::Unhandled,
            KeyCode::Esc
                if matches!(self.keymap, Keymap::Conventional(_))
                    || self.mode() == EditMode::Normal =>
            {
                return Outcome::Leave;
            }
            KeyCode::Enter if self.single_line => return Outcome::Submit,
            _ => {}
        }

        self.keymap.input_key(TerminalKey::from(key));
        while let Some((act, ctx)) = self.keymap.pop() {
            match act {
                Action::Editor(a) => {
                    // The Emacs table does not checkpoint typed text before undo.
                    // Seal the current edit so Ctrl-Y can redo it after Ctrl-Z.
                    if matches!(self.keymap, Keymap::Conventional(_))
                        && matches!(a, EditorAction::History(HistoryAction::Undo(_)))
                    {
                        let _ = self.tbox.editor_command(
                            &EditorAction::History(HistoryAction::Checkpoint),
                            &ctx,
                            &mut self.store,
                        );
                    }
                    let _ = self.tbox.editor_command(&a, &ctx, &mut self.store);
                }
                Action::Scroll(s) => {
                    let _ = self.tbox.scroll(&s, &ctx, &mut self.store);
                }
                Action::Jump(l, d, c) => {
                    let _ = self.tbox.jump(l, d, ctx.resolve(&c), &ctx);
                }
                Action::Repeat(rt) => self.keymap.repeat(rt, ctx),
                _ => {}
            }
        }
        Outcome::Handled
    }

    pub fn text(&self) -> String {
        self.tbox.get_text().trim_end_matches('\n').to_string()
    }

    pub fn mode(&self) -> EditMode {
        self.keymap.mode()
    }

    pub fn status_mode(&self) -> Option<EditMode> {
        match self.keymap {
            Keymap::Vim(_) => Some(self.mode()),
            Keymap::Conventional(_) => None,
        }
    }

    /// Read the visual selection from the adapter's paint. `TextBoxState` hides its cursor
    /// group, so its public API cannot expose selection intervals; a future accessor should
    /// replace this render read. The adapter leaves the terminal caret separate from the
    /// buffer paint, so every reversed cell in the conventional table is selected text,
    /// including a one-character Shift+Left selection. Vim Normal is never a selection.
    pub fn edit(&mut self) -> Edit {
        let text = self.text();
        let mode = self.mode();
        let caret = self.tbox.get_cursor().x;
        let width = text
            .chars()
            .count()
            .saturating_add(1)
            .min(u16::MAX as usize) as u16;
        let area = Rect::new(0, 0, width, 1);
        let mut buffer = Buffer::empty(area);
        TextBox::new()
            .oneline()
            .render(area, &mut buffer, &mut self.tbox);
        let selected: Vec<usize> = (0..width)
            .filter(|&x| {
                buffer
                    .cell((x, 0))
                    .is_some_and(|cell| cell.modifier.contains(Modifier::REVERSED))
            })
            .map(usize::from)
            .collect();
        let selection = selected
            .first()
            .zip(selected.last())
            .map(|(first, last)| *first..last + 1);
        let selection = selection
            .filter(|_| mode == EditMode::Visual || matches!(self.keymap, Keymap::Conventional(_)));
        match mode {
            EditMode::Insert => {
                let mut edit = Edit::insert_at(text, caret);
                edit.set_selection(selection);
                edit
            }
            EditMode::Normal => Edit::normal(text, caret),
            EditMode::Visual => Edit::visual(text, selection.unwrap_or(caret..caret + 1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use modalkit::crossterm::event::KeyModifiers;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn feed(field: &mut Field, keys: &str) {
        for ch in keys.chars() {
            let modifiers = if ch.is_ascii_uppercase() {
                KeyModifiers::SHIFT
            } else {
                KeyModifiers::NONE
            };
            assert_eq!(
                field.key(KeyEvent::new(KeyCode::Char(ch), modifiers)),
                Outcome::Handled
            );
        }
    }

    #[test]
    fn vim_visual_motion_selects_then_deletes_to_second_comma() {
        let input = "alpha, beta, gamma, delta";
        let mut field = Field::vim(input);
        feed(&mut field, "0v2t,");
        let second_comma = input.match_indices(',').nth(1).unwrap().0;
        assert_eq!(field.mode(), EditMode::Visual);
        assert_eq!(field.edit().selection(), Some(0..second_comma));
        feed(&mut field, "d");
        assert_eq!(field.text(), ", gamma, delta");
    }

    #[test]
    fn vim_escape_returns_to_normal_then_leaves() {
        let mut field = Field::vim("");
        feed(&mut field, "i");
        assert_eq!(field.mode(), EditMode::Insert);
        assert_eq!(field.key(key(KeyCode::Esc)), Outcome::Handled);
        assert_eq!(field.mode(), EditMode::Normal);
        assert_eq!(field.key(key(KeyCode::Esc)), Outcome::Leave);
    }

    #[test]
    fn conventional_selection_and_escape() {
        let mut field = Field::conventional("");
        feed(&mut field, "ab");
        assert_eq!(field.text(), "ab");
        assert_eq!(field.mode(), EditMode::Insert);
        assert_eq!(field.edit().caret(), 2);
        assert_eq!(
            field.key(KeyEvent::new(KeyCode::Left, KeyModifiers::SHIFT)),
            Outcome::Handled
        );
        assert_eq!(field.edit().selection(), Some(1..2));
        assert_eq!(field.key(key(KeyCode::Esc)), Outcome::Leave);
    }

    #[test]
    fn conventional_undo_and_redo() {
        let mut field = Field::conventional("");
        feed(&mut field, "ab");
        assert_eq!(
            field.key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::CONTROL)),
            Outcome::Handled
        );
        assert_eq!(field.text(), "");
        assert_eq!(
            field.key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::CONTROL)),
            Outcome::Handled
        );
        assert_eq!(field.text(), "ab");
    }

    #[test]
    fn tab_and_enter_belong_to_the_container_on_single_line_fields() {
        for mut field in [Field::vim(""), Field::conventional("")] {
            assert_eq!(field.key(key(KeyCode::Tab)), Outcome::Unhandled);
            assert_eq!(field.key(key(KeyCode::BackTab)), Outcome::Unhandled);
            assert_eq!(field.key(key(KeyCode::Enter)), Outcome::Submit);
            assert_eq!(
                field.multi_line().key(key(KeyCode::Enter)),
                Outcome::Handled
            );
        }
    }

    #[test]
    fn only_vim_has_a_status_indicator() {
        assert_eq!(Field::vim("").status_mode(), Some(EditMode::Normal));
        assert_eq!(Field::conventional("").status_mode(), None);
    }

    #[test]
    fn vim_append_snapshot_uses_insert_caret_after_typed_character() {
        let mut field = Field::vim("20");
        feed(&mut field, "A0");
        let edit = field.edit();
        assert_eq!(edit.value(), "200");
        assert_eq!(edit.caret(), 3);
        assert_eq!(edit.mode(), EditMode::Insert);
    }
}

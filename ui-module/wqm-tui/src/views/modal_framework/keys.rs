//! **The record's keyboard**, and the live [`crate::editor::Field`] under its text values.
//!
//! Round 1 built every field kind as a still: a [`Value::Radio`] knew which button was picked
//! and a [`Mode::Edit`] carried an [`Edit`] snapshot, but nothing turned a keystroke into either
//! of them. Traversal was tested at the stack boundary and no caret ever moved inside a field.
//! This module is the half that was missing — one entry point, [`RecordState::key`], and the
//! rules the 20:30 and 19:05 rulings give for each field kind.
//!
//! # Why the live editor does not replace the snapshot
//!
//! [`crate::widgets::edit_field::Edit`] stays what the renderer draws, and a still frame keeps
//! constructing one by hand. What changes is that a LIVE record now derives it: after every
//! keystroke [`RecordState::refresh`] asks the engine for a new snapshot and stores it in the
//! mode. Rendering therefore stays `&self` all the way down — `Field::edit` needs `&mut` to
//! render into its scratch buffer, and threading that through `Stack::render` would have made
//! every draw path mutable for the sake of one accessor.
//!
//! # The two keymaps differ in more than their bindings
//!
//! [`Keys`] picks which table a new editor is built with, and it also decides whether a caret is
//! PAINTED at all: under vim the program owns the caret and draws it by mode, and under the
//! conventional table the terminal owns it and a painted one would be a second cursor beside the
//! real one. That is why the choice lives here, beside the editor, rather than in the widget.
//!
//! # Every key in this file is a key some ruling names
//!
//! | field | keys | ruling |
//! |---|---|---|
//! | radio, one row | `h`/`l`, `←`/`→` | item 3 |
//! | radio, a column | `j`/`k`, `↑`/`↓` | item 3 |
//! | tick box | `Space` | item 3 |
//! | drop-down | `↓`/`j` opens; `j`/`k`, `^D`/`^U`, `^F`/`^B` inside; typing filters; `Esc` resets then closes; `↵` commits | items 3 and 21:08 |
//! | text, number, multi | the editor's own, via [`crate::editor::Field`] | 01:09, 21:42 |
//! | anywhere | `Tab`/`Shift-Tab` traverse the editable fields | 20:30 |
//!
//! Nothing here invents a binding. A key that no ruling names is [`Reaction::Ignored`], which is
//! what lets the composition above try it — `?`, `Backspace` and `q` are the window's, not the
//! record's.

use modalkit::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::record::{Mode, Value};
use super::stack::RecordState;
use crate::editor::{Field, Outcome};
use crate::widgets::edit_field::Caret;

/// Which key table a new editor is built with — and therefore who owns the caret.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Keys {
    /// Chris, 01:09: *"vim-mode applies to every text fields when selected"*. The program paints
    /// the caret, shaped and blinking by mode.
    #[default]
    Vim,
    /// The non-vim table. The terminal draws its own cursor, so nothing is painted.
    Conventional,
}

impl Keys {
    pub fn field(self, text: &str) -> Field {
        match self {
            Keys::Vim => Field::vim(text),
            Keys::Conventional => Field::conventional(text),
        }
    }

    /// Who draws the caret under this table.
    pub fn caret(self) -> Caret {
        match self {
            Keys::Vim => Caret::Painted,
            Keys::Conventional => Caret::Terminal,
        }
    }
}

/// What a keystroke did, from the point of view of whatever holds the record.
///
/// Named outcomes rather than a `bool` for the reason [`super::stack::Pop`] is: *the record did
/// not want this key* and *the record handled it and there is nothing more to do* are different
/// facts to the caller, and a silent no-op is indistinguishable from a defect.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reaction {
    /// Not a key this record claims. The composition above may still want it — `?`, `Backspace`
    /// and `q` all reach the window this way.
    Ignored,
    /// Handled; the view has changed and wants redrawing.
    Handled,
    /// Edit mode was left. The record is back in [`Mode::View`] on the same field.
    Left,
    /// `↵` on a single-line field. The caller decides what committing means.
    Submitted,
}

/// How far a page key moves inside an open drop-down, when nothing has told it the real height.
///
/// The drop-down's height is decided at RENDER time by [`super::record::DropDown::rect`], from
/// the room the window has and the list it holds; a record being driven headlessly has not been
/// rendered yet. So [`Picker::page`] carries the real number once a frame has been drawn, and
/// this is what it starts at — a plausible list height rather than a magic number, and it is
/// only ever the `^F`/`^B` stride.
pub const PICKER_PAGE: usize = 10;

/// An open drop-down: where its cursor is and what has been typed into it.
///
/// It is NOT a copy of the field's choices. The list a reader sees is derived from the field
/// every time it is drawn ([`super::record::DropDown::visible`]), so an open picker cannot come
/// to disagree with the value it was opened from — which is exactly what a second copy of the
/// choices would eventually do.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Picker {
    /// Which VISIBLE row the cursor is on — visible, because a filter changes what the rows are.
    pub cursor: usize,
    /// What has been typed. Empty is *nothing typed*, which is what makes the two `Esc`s
    /// distinguishable.
    pub filter: String,
    /// The drawn height, once something has drawn it.
    pub page: usize,
}

impl Picker {
    pub fn new() -> Self {
        Self {
            cursor: 0,
            filter: String::new(),
            page: PICKER_PAGE,
        }
    }
}

impl RecordState {
    /// The keymap every editor under this record is built with.
    pub fn with_keys(mut self, keys: Keys) -> Self {
        self.keys = keys;
        self
    }

    /// Re-derive the render snapshot from the live engine.
    ///
    /// Called after every keystroke the editor handled. Without it the caret in the frame is
    /// wherever it was when edit mode was entered, which is the exact defect round 1 shipped —
    /// a field that accepted keys and drew none of them.
    pub(crate) fn refresh(&mut self) {
        if let Mode::Edit { edit, .. } = &mut self.mode {
            *edit = self.editor.as_mut().map(|field| field.edit());
        }
    }

    /// Put a live editor under field `index`, or take it away where the field is not typed into.
    fn mount_editor(&mut self, index: usize) {
        let text = self
            .fields
            .get(index)
            .filter(|field| field.is_editable())
            .and_then(|field| field.value().as_text())
            .map(str::to_string);
        self.editor = text.map(|text| {
            let mut field = self.keys.field(&text);
            if matches!(self.fields[index].value(), Value::Multi(_)) {
                field = field.multi_line();
            }
            Box::new(field)
        });
    }

    /// Enter edit mode on the field the cursor is on, if this window lets it be changed.
    pub fn begin_edit(&mut self) -> Reaction {
        let at = self.mode.at();
        if !self.fields.get(at).is_some_and(|f| f.is_editable()) {
            return Reaction::Ignored;
        }
        self.mode = Mode::Edit { at, edit: None };
        self.mount_editor(at);
        self.refresh();
        Reaction::Handled
    }

    /// Leave edit mode, keeping the cursor where it was.
    pub fn end_edit(&mut self) -> Reaction {
        self.commit_editor();
        self.mode = Mode::View { at: self.mode.at() };
        self.editor = None;
        Reaction::Left
    }

    /// Write the live editor's text back into the field it is editing.
    ///
    /// The engine owns the text while a field is open; the record owns it the rest of the time.
    /// Doing this on every traversal and on leaving is what makes a drill-down's edits visible
    /// to the parent, which the 12:52 ruling asks for.
    fn commit_editor(&mut self) {
        let Some(field) = self.editor.as_ref() else {
            return;
        };
        let text = field.text();
        let at = self.mode.at();
        if let Some(row) = self.fields.get_mut(at) {
            match row.value_mut() {
                Value::Number(value) | Value::Text(value) | Value::Multi(value) => *value = text,
                _ => {}
            }
        }
    }

    /// The editable fields, in order — what `Tab` walks.
    fn editable(&self) -> Vec<usize> {
        self.fields
            .iter()
            .enumerate()
            .filter(|(_, field)| field.is_editable())
            .map(|(index, _)| index)
            .collect()
    }

    /// `Tab` / `Shift-Tab`: the next or previous editable field, wrapping.
    ///
    /// Wrapping rather than stopping, because a form is a ring: the 20:30 ruling puts traversal
    /// on the container precisely so the reader never has to know where the list ends.
    fn traverse(&mut self, forward: bool) -> Reaction {
        let order = self.editable();
        if order.is_empty() {
            return Reaction::Ignored;
        }
        self.commit_editor();
        let at = self.mode.at();
        let here = order.iter().position(|index| *index == at);
        let next = match here {
            Some(pos) if forward => (pos + 1) % order.len(),
            Some(pos) => (pos + order.len() - 1) % order.len(),
            // The cursor is on a read-only field: `Tab` goes to the first editable one forward
            // and the last one backward, rather than refusing.
            None if forward => 0,
            None => order.len() - 1,
        };
        let index = order[next];
        self.mode = Mode::Edit {
            at: index,
            edit: None,
        };
        self.mount_editor(index);
        self.refresh();
        Reaction::Handled
    }

    /// Move the VIEW-mode data cursor by one field.
    fn move_cursor(&mut self, down: bool) -> Reaction {
        if self.fields.is_empty() {
            return Reaction::Ignored;
        }
        let at = self.mode.at();
        let last = self.fields.len() - 1;
        let next = if down {
            at.saturating_add(1).min(last)
        } else {
            at.saturating_sub(1)
        };
        if next == at {
            return Reaction::Handled;
        }
        self.mode = Mode::View { at: next };
        Reaction::Handled
    }

    /// One keystroke.
    pub fn key(&mut self, key: KeyEvent) -> Reaction {
        if self.picker.is_some() {
            return self.picker_key(key);
        }
        match &self.mode {
            Mode::View { .. } => self.view_key(key),
            Mode::Edit { .. } => self.edit_key(key),
        }
    }

    /// VIEW mode: the cursor moves, and `e` opens the field it is on.
    ///
    /// Scrolling is the Queue's navigation (20:30), so `j`/`k` and the arrows both move — the
    /// same pair every table in this crate answers to.
    fn view_key(&mut self, key: KeyEvent) -> Reaction {
        match key.code {
            KeyCode::Char('j') | KeyCode::Down => self.move_cursor(true),
            KeyCode::Char('k') | KeyCode::Up => self.move_cursor(false),
            KeyCode::Char('e') => self.begin_edit(),
            _ => Reaction::Ignored,
        }
    }

    /// EDIT mode: traversal first, then whatever the active field's kind answers to.
    fn edit_key(&mut self, key: KeyEvent) -> Reaction {
        match key.code {
            KeyCode::Tab => return self.traverse(true),
            KeyCode::BackTab => return self.traverse(false),
            _ => {}
        }
        let at = self.mode.at();
        let Some(kind) = self.fields.get(at).map(|field| field.value().clone()) else {
            return Reaction::Ignored;
        };
        match kind {
            // The engine owns every typed field. `Esc` is two-stage under vim and one under the
            // conventional table, and that rule is `editor::Field`'s — see its module docs.
            Value::Number(_) | Value::Text(_) | Value::Multi(_) => self.editor_key(key),
            Value::Radio { .. } => self.radio_key(key, true),
            Value::RadioColumn { .. } => self.radio_key(key, false),
            Value::Bool(_) => self.bool_key(key),
            Value::Choice { .. } => self.choice_key(key),
        }
    }

    fn editor_key(&mut self, key: KeyEvent) -> Reaction {
        let Some(field) = self.editor.as_mut() else {
            // An editable text field with no engine under it is a bug, not a state — but it is
            // better to refuse the key than to swallow it silently.
            return Reaction::Ignored;
        };
        let outcome = field.key(key);
        self.refresh();
        match outcome {
            Outcome::Handled => Reaction::Handled,
            Outcome::Leave => self.end_edit(),
            Outcome::Submit => {
                self.commit_editor();
                self.refresh();
                Reaction::Submitted
            }
            Outcome::Unhandled => Reaction::Ignored,
        }
    }

    /// A radio group. `row` picks which axis moves it — item 3 gives the two forms different
    /// keys because they are laid out along different axes.
    fn radio_key(&mut self, key: KeyEvent, row: bool) -> Reaction {
        let forward = match (key.code, row) {
            (KeyCode::Char('l') | KeyCode::Right, true) => true,
            (KeyCode::Char('h') | KeyCode::Left, true) => false,
            (KeyCode::Char('j') | KeyCode::Down, false) => true,
            (KeyCode::Char('k') | KeyCode::Up, false) => false,
            (KeyCode::Esc, _) => return self.end_edit(),
            _ => return Reaction::Ignored,
        };
        let at = self.mode.at();
        let Some(value) = self.fields.get_mut(at).map(|field| field.value_mut()) else {
            return Reaction::Ignored;
        };
        let (Value::Radio { choices, at: pick } | Value::RadioColumn { choices, at: pick }) = value
        else {
            return Reaction::Ignored;
        };
        if choices.is_empty() {
            return Reaction::Ignored;
        }
        let last = choices.len() - 1;
        // Clamped, not wrapped. A radio row is a spatial arrangement the reader can see the ends
        // of, and a selection that jumped from the last button back to the first would contradict
        // what the frame shows.
        *pick = if forward {
            pick.saturating_add(1).min(last)
        } else {
            pick.saturating_sub(1)
        };
        Reaction::Handled
    }

    /// A tick box. Chris: *"Space toggles"*, and the mark is a shape so the toggle is visible
    /// without colour (r06 #8).
    fn bool_key(&mut self, key: KeyEvent) -> Reaction {
        match key.code {
            KeyCode::Char(' ') => {
                let at = self.mode.at();
                if let Some(Value::Bool(on)) = self.fields.get_mut(at).map(|f| f.value_mut()) {
                    *on = !*on;
                    return Reaction::Handled;
                }
                Reaction::Ignored
            }
            KeyCode::Esc => self.end_edit(),
            _ => Reaction::Ignored,
        }
    }

    /// A categorical field. `↓`/`j` opens the list (21:08); everything else about it is
    /// [`RecordState::picker_key`]'s.
    fn choice_key(&mut self, key: KeyEvent) -> Reaction {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.picker = Some(Picker::new());
                Reaction::Handled
            }
            KeyCode::Esc => self.end_edit(),
            _ => Reaction::Ignored,
        }
    }

    /// The rows an open drop-down is showing, under the filter in force.
    pub fn picker_rows(&self) -> Vec<String> {
        let Some(picker) = self.picker.as_ref() else {
            return Vec::new();
        };
        let at = self.mode.at();
        let Some((choices, current)) = self
            .fields
            .get(at)
            .and_then(|field| field.value().choices())
        else {
            return Vec::new();
        };
        super::record::current_first_then_sorted(choices, current)
            .into_iter()
            .filter(|choice| super::record::fuzzy_matches(choice, &picker.filter))
            .collect()
    }

    /// An open drop-down.
    ///
    /// The two `Esc`s are the subtle part and they are Chris's: *"Esc once resets the filter
    /// (current value back on top), Esc again closes unchanged"*. So the first `Esc` is an undo
    /// of the typing and the second is an undo of the opening, which means a reader who has typed
    /// nothing gets the close on the first press — there is nothing for it to undo.
    fn picker_key(&mut self, key: KeyEvent) -> Reaction {
        let rows = self.picker_rows();
        let Some(picker) = self.picker.as_mut() else {
            return Reaction::Ignored;
        };
        let last = rows.len().saturating_sub(1);
        let page = picker.page.max(1);
        match key.code {
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                picker.cursor = picker.cursor.saturating_add(page / 2).min(last);
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                picker.cursor = picker.cursor.saturating_sub(page / 2);
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                picker.cursor = picker.cursor.saturating_add(page).min(last);
            }
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                picker.cursor = picker.cursor.saturating_sub(page);
            }
            KeyCode::Down | KeyCode::Char('j') if picker.filter.is_empty() => {
                picker.cursor = picker.cursor.saturating_add(1).min(last);
            }
            KeyCode::Up | KeyCode::Char('k') if picker.filter.is_empty() => {
                picker.cursor = picker.cursor.saturating_sub(1);
            }
            KeyCode::Down => picker.cursor = picker.cursor.saturating_add(1).min(last),
            KeyCode::Up => picker.cursor = picker.cursor.saturating_sub(1),
            KeyCode::Backspace => {
                picker.filter.pop();
                picker.cursor = 0;
            }
            KeyCode::Char(c) => {
                // **Typing REPLACES and filters** (item 3). Once a term is being typed, `j` and
                // `k` are letters rather than motions — which is why the two motion arms above
                // are guarded on an empty filter and the arrows are not: an arrow is never a
                // character a reader meant to type.
                picker.filter.push(c);
                picker.cursor = 0;
            }
            KeyCode::Esc if !picker.filter.is_empty() => {
                picker.filter.clear();
                picker.cursor = 0;
            }
            KeyCode::Esc => {
                self.picker = None;
                return Reaction::Handled;
            }
            KeyCode::Enter => {
                let chosen = rows.get(picker.cursor).cloned();
                self.picker = None;
                if let Some(chosen) = chosen {
                    let at = self.mode.at();
                    if let Some(Value::Choice { choices, at: pick }) =
                        self.fields.get_mut(at).map(|f| f.value_mut())
                        && let Some(index) = choices.iter().position(|c| *c == chosen)
                    {
                        *pick = index;
                    }
                }
                return Reaction::Handled;
            }
            _ => return Reaction::Ignored,
        }
        Reaction::Handled
    }
}

#[cfg(test)]
mod tests;

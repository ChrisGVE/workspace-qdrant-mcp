//! Every one of these presses a key and reads the state back.
//!
//! That is the whole point of the file: round 1 built each field kind at construction time and
//! proved it by the shape it drew, so a radio whose `at` nothing could move drew a correct
//! frame and satisfied every test. A test that sets `at: 1` and checks for `(●) update` is
//! testing the renderer; these send `l` and ask the field where it ended up.

use modalkit::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::*;
use crate::views::modal_framework::record::{FieldRow, Value};
use crate::views::modal_framework::stack::{Layer, Stack, View};
use crate::widgets::edit_field::EditMode;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn ctrl(c: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
}

fn chars(state: &mut RecordState, text: &str) {
    for c in text.chars() {
        state.key(key(KeyCode::Char(c)));
    }
}

const CHOICES: [&str; 6] = [
    "tree-sitter/function",
    "fixed-size",
    "semantic",
    "paragraph",
    "line",
    "token",
];

fn choices() -> Vec<String> {
    CHOICES.iter().map(|c| c.to_string()).collect()
}

/// A record with one of every kind that answers to a key, in a stable order.
fn record() -> RecordState {
    RecordState::new(
        vec![
            FieldRow::new("Tenant", Value::Text("open-books".into())).read_only(),
            FieldRow::new("Label", Value::Text("reading guide".into())),
            FieldRow::new(
                "Operation",
                Value::Radio {
                    choices: vec!["add".into(), "update".into(), "delete".into()],
                    at: 1,
                },
            ),
            FieldRow::new(
                "Stage",
                Value::RadioColumn {
                    choices: vec!["scan".into(), "parse".into(), "embed".into()],
                    at: 0,
                },
            ),
            FieldRow::new("Confirm before discard", Value::Bool(true)),
            FieldRow::new(
                "Chunking",
                Value::Choice {
                    choices: choices(),
                    at: 2,
                },
            ),
            FieldRow::new(
                "Note",
                Value::Multi("held back once already, the retry is queued".into()),
            ),
        ],
        Mode::View { at: 0 },
    )
}

fn value_at(state: &RecordState, index: usize) -> Value {
    state.fields[index].value().clone()
}

/// The record wrapped in the window that owns `?`, `Backspace` and `q`.
fn stack_of(state: RecordState) -> Stack {
    Stack::new(Layer::new(
        "Queue item",
        "Queue item \u{2014} reading_guide.py",
        View::Record(state),
    ))
}

/// Where the record inside a stack has its cursor.
fn record_at(stack: &Stack) -> usize {
    match &stack.top().view {
        View::Record(record) => record.mode.at(),
        View::Table(_) => panic!("not a record"),
    }
}

// ---------------------------------------------------------------- view mode

/// `j`/`k` move the data cursor, and they stop at the ends rather than wrapping — the same
/// navigation a table gives, which is what the 20:30 ruling asks for.
#[test]
fn the_view_cursor_moves_with_jk_and_stops_at_both_ends() {
    let mut state = record();
    assert_eq!(state.mode.at(), 0);
    assert_eq!(state.key(key(KeyCode::Char('k'))), Reaction::Handled);
    assert_eq!(state.mode.at(), 0, "the top does not wrap to the bottom");
    for expected in 1..=6 {
        state.key(key(KeyCode::Char('j')));
        assert_eq!(state.mode.at(), expected);
    }
    state.key(key(KeyCode::Char('j')));
    assert_eq!(state.mode.at(), 6, "the bottom does not wrap to the top");
    assert_eq!(state.key(key(KeyCode::Down)), Reaction::Handled);
    assert_eq!(state.key(key(KeyCode::Up)), Reaction::Handled);
    assert_eq!(state.mode.at(), 5, "the arrows are the same pair");
}

/// `e` opens the field the cursor is on — and refuses on a read-only one, because editability
/// is declared per field by the composing view (Chris, 20:30).
#[test]
fn e_enters_edit_mode_except_on_a_read_only_field() {
    let mut state = record();
    assert_eq!(state.key(key(KeyCode::Char('e'))), Reaction::Ignored);
    assert!(!state.mode.editing(), "field 0 is read-only");

    state.key(key(KeyCode::Char('j')));
    assert_eq!(state.key(key(KeyCode::Char('e'))), Reaction::Handled);
    assert!(state.mode.editing());
    assert_eq!(state.mode.at(), 1);
}

// ---------------------------------------------------------------- the live editor

/// **The caret moves inside the field.** Round 1 could not do this at all: the snapshot was
/// whatever the frame was constructed with, so a field accepted keys and drew none of them.
#[test]
fn typing_into_a_text_field_moves_the_caret_and_the_text() {
    let mut state = record();
    state.key(key(KeyCode::Char('j')));
    state.key(key(KeyCode::Char('e')));

    // vim opens in NORMAL, so `A` is what starts appending.
    let Mode::Edit {
        edit: Some(edit), ..
    } = &state.mode
    else {
        panic!("no snapshot under the active field");
    };
    assert_eq!(edit.mode(), EditMode::Normal);
    let before = edit.caret();

    state.key(KeyEvent::new(KeyCode::Char('A'), KeyModifiers::SHIFT));
    chars(&mut state, "s");
    let Mode::Edit {
        edit: Some(edit), ..
    } = &state.mode
    else {
        panic!("no snapshot");
    };
    assert_eq!(edit.mode(), EditMode::Insert);
    assert_eq!(edit.value(), "reading guides");
    assert!(
        edit.caret() > before,
        "the caret did not follow the typing: {} -> {}",
        before,
        edit.caret()
    );
}

/// The three carets of item 3, read off the snapshot rather than off the frame — a still cannot
/// show a blink, so what a mode IS gets asserted here and how it PAINTS in `edit_field`.
#[test]
fn the_snapshot_carries_the_mode_each_caret_shape_is_drawn_from() {
    let mut state = record();
    state.key(key(KeyCode::Char('j')));
    state.key(key(KeyCode::Char('e')));

    let mode_now = |state: &RecordState| match &state.mode {
        Mode::Edit {
            edit: Some(edit), ..
        } => edit.mode(),
        _ => panic!("no snapshot"),
    };
    assert_eq!(mode_now(&state), EditMode::Normal);
    state.key(key(KeyCode::Char('i')));
    assert_eq!(mode_now(&state), EditMode::Insert);
    state.key(key(KeyCode::Esc));
    assert_eq!(
        mode_now(&state),
        EditMode::Normal,
        "one Esc returns to normal"
    );
    state.key(key(KeyCode::Char('v')));
    assert_eq!(mode_now(&state), EditMode::Visual);
}

/// Esc under vim is two-stage and under the conventional table is one — the 20:30 ruling, and
/// the rule lives in `editor::Field` where both tables can reach it.
#[test]
fn esc_leaves_in_one_press_conventionally_and_two_under_vim() {
    let mut vim = record();
    vim.key(key(KeyCode::Char('j')));
    vim.key(key(KeyCode::Char('e')));
    vim.key(key(KeyCode::Char('i')));
    assert_eq!(vim.key(key(KeyCode::Esc)), Reaction::Handled);
    assert!(vim.mode.editing(), "the first Esc is the engine's");
    assert_eq!(vim.key(key(KeyCode::Esc)), Reaction::Left);
    assert!(!vim.mode.editing());

    let mut plain = record().with_keys(Keys::Conventional);
    plain.key(key(KeyCode::Char('j')));
    plain.key(key(KeyCode::Char('e')));
    assert_eq!(plain.key(key(KeyCode::Esc)), Reaction::Left);
    assert!(!plain.mode.editing());
}

/// Leaving writes the engine's text back into the field. Without this the record would render
/// an edit that vanished the moment the reader tabbed away.
#[test]
fn leaving_a_field_commits_what_was_typed_into_it() {
    let mut state = record();
    state.key(key(KeyCode::Char('j')));
    state.key(key(KeyCode::Char('e')));
    state.key(KeyEvent::new(KeyCode::Char('A'), KeyModifiers::SHIFT));
    chars(&mut state, "!");
    state.key(key(KeyCode::Esc));
    state.key(key(KeyCode::Esc));
    assert_eq!(
        value_at(&state, 1),
        Value::Text("reading guide!".into()),
        "the typed text did not reach the field"
    );
}

/// Tab and Shift-Tab walk the EDITABLE fields only, wrapping, and the read-only one is not
/// among them.
#[test]
fn tab_walks_the_editable_fields_and_wraps() {
    let mut state = record();
    state.key(key(KeyCode::Char('j')));
    state.key(key(KeyCode::Char('e')));
    assert_eq!(state.mode.at(), 1);
    for expected in [2, 3, 4, 5, 6, 1] {
        assert_eq!(state.key(key(KeyCode::Tab)), Reaction::Handled);
        assert_eq!(state.mode.at(), expected, "Tab skipped or stopped");
    }
    assert_eq!(state.key(key(KeyCode::BackTab)), Reaction::Handled);
    assert_eq!(state.mode.at(), 6, "Shift-Tab wraps the other way");
}

/// A live engine exists under a text field and under no other kind — a radio has nothing to
/// type into, and an engine there would be a caret with no text.
#[test]
fn only_a_text_field_gets_an_engine_under_it() {
    let mut state = record();
    state.key(key(KeyCode::Char('j')));
    state.key(key(KeyCode::Char('e')));
    assert!(state.editor.is_some(), "a Text field");
    state.key(key(KeyCode::Tab));
    assert!(state.editor.is_none(), "a Radio field");
    state.key(key(KeyCode::Tab));
    assert!(state.editor.is_none(), "a RadioColumn field");
    state.key(key(KeyCode::Tab));
    assert!(state.editor.is_none(), "a Bool field");
    state.key(key(KeyCode::Tab));
    assert!(state.editor.is_none(), "a Choice field");
    state.key(key(KeyCode::Tab));
    assert!(state.editor.is_some(), "a Multi field");
}

// ---------------------------------------------------------------- radio, tick box

/// A radio ROW moves along its axis with `h`/`l` and the arrows, and clamps at both ends.
#[test]
fn a_radio_row_moves_with_hl_and_clamps() {
    let mut state = record();
    state.mode = Mode::Edit { at: 2, edit: None };
    assert_eq!(state.key(key(KeyCode::Char('l'))), Reaction::Handled);
    assert!(matches!(value_at(&state, 2), Value::Radio { at: 2, .. }));
    state.key(key(KeyCode::Char('l')));
    assert!(
        matches!(value_at(&state, 2), Value::Radio { at: 2, .. }),
        "the last button does not wrap to the first"
    );
    state.key(key(KeyCode::Left));
    state.key(key(KeyCode::Left));
    assert!(matches!(value_at(&state, 2), Value::Radio { at: 0, .. }));
    state.key(key(KeyCode::Char('h')));
    assert!(matches!(value_at(&state, 2), Value::Radio { at: 0, .. }));

    // The other axis is not this field's.
    assert_eq!(state.key(key(KeyCode::Char('j'))), Reaction::Ignored);
}

/// A radio COLUMN moves with `j`/`k` instead — item 3 gives the two forms different keys
/// because they are laid out along different axes.
#[test]
fn a_radio_column_moves_with_jk_and_not_with_hl() {
    let mut state = record();
    state.mode = Mode::Edit { at: 3, edit: None };
    assert_eq!(state.key(key(KeyCode::Char('j'))), Reaction::Handled);
    assert!(matches!(
        value_at(&state, 3),
        Value::RadioColumn { at: 1, .. }
    ));
    state.key(key(KeyCode::Down));
    assert!(matches!(
        value_at(&state, 3),
        Value::RadioColumn { at: 2, .. }
    ));
    state.key(key(KeyCode::Char('k')));
    assert!(matches!(
        value_at(&state, 3),
        Value::RadioColumn { at: 1, .. }
    ));
    assert_eq!(state.key(key(KeyCode::Char('l'))), Reaction::Ignored);
}

/// Space toggles a tick box, and nothing else does.
#[test]
fn space_toggles_a_tick_box() {
    let mut state = record();
    state.mode = Mode::Edit { at: 4, edit: None };
    assert_eq!(value_at(&state, 4), Value::Bool(true));
    assert_eq!(state.key(key(KeyCode::Char(' '))), Reaction::Handled);
    assert_eq!(value_at(&state, 4), Value::Bool(false));
    state.key(key(KeyCode::Char(' ')));
    assert_eq!(value_at(&state, 4), Value::Bool(true));
    assert_eq!(state.key(key(KeyCode::Char('x'))), Reaction::Ignored);
}

// ---------------------------------------------------------------- the drop-down

/// `↓`/`j` opens it, with the current value first and the cursor on it (21:08 + item 3).
#[test]
fn down_opens_the_drop_down_on_the_current_value() {
    let mut state = record();
    state.mode = Mode::Edit { at: 5, edit: None };
    assert!(state.picker.is_none());
    assert_eq!(state.key(key(KeyCode::Char('j'))), Reaction::Handled);
    let picker = state.picker.as_ref().expect("the list is open");
    assert_eq!(picker.cursor, 0);
    let rows = state.picker_rows();
    assert_eq!(rows[0], "semantic", "the current value is first");
    let mut rest = rows[1..].to_vec();
    let mut sorted = rest.clone();
    sorted.sort();
    rest.sort();
    assert_eq!(rows[1..].to_vec(), sorted, "…and the rest are sorted");
}

/// Inside it, `j`/`k` move and the page keys move further — and every one of them stops at the
/// ends rather than running off the list.
#[test]
fn the_open_list_moves_with_jk_and_the_page_keys() {
    let mut state = record();
    state.mode = Mode::Edit { at: 5, edit: None };
    state.key(key(KeyCode::Char('j')));
    let last = state.picker_rows().len() - 1;

    state.key(key(KeyCode::Char('j')));
    assert_eq!(state.picker.as_ref().unwrap().cursor, 1);
    state.key(key(KeyCode::Char('k')));
    assert_eq!(state.picker.as_ref().unwrap().cursor, 0);
    state.key(key(KeyCode::Char('k')));
    assert_eq!(
        state.picker.as_ref().unwrap().cursor,
        0,
        "clamped at the top"
    );

    state.key(ctrl('d'));
    let half = state.picker.as_ref().unwrap().cursor;
    assert!(half > 0 && half <= last, "^D moved half a page: {half}");
    state.key(ctrl('f'));
    assert_eq!(
        state.picker.as_ref().unwrap().cursor,
        last,
        "^F runs to the end of a list shorter than a page"
    );
    state.key(ctrl('u'));
    assert!(state.picker.as_ref().unwrap().cursor < last);
    state.key(ctrl('b'));
    assert_eq!(state.picker.as_ref().unwrap().cursor, 0);
}

/// Typing narrows by SUBSEQUENCE, which is what "fuzzy" means to anyone who has used a fuzzy
/// finder: `tsf` finds `tree-sitter/function`.
#[test]
fn typing_narrows_the_list_by_subsequence() {
    let mut state = record();
    state.mode = Mode::Edit { at: 5, edit: None };
    state.key(key(KeyCode::Char('j')));
    let all = state.picker_rows().len();

    chars(&mut state, "tsf");
    let narrowed = state.picker_rows();
    assert!(narrowed.len() < all, "the filter did not narrow anything");
    assert_eq!(narrowed, vec!["tree-sitter/function".to_string()]);
    assert_eq!(state.picker.as_ref().unwrap().cursor, 0, "back to the top");

    // Backspace walks the term back, and emptying it restores the whole list.
    //
    // Measured over the WHOLE term rather than one keypress, deliberately: dropping the `f`
    // leaves `ts`, which is a subsequence of `tree-sitter/function` and of nothing else in this
    // fixture, so a one-keypress assertion would be pinning the fixture rather than the filter.
    for _ in 0..3 {
        state.key(key(KeyCode::Backspace));
    }
    assert!(state.picker.as_ref().unwrap().filter.is_empty());
    assert_eq!(
        state.picker_rows().len(),
        all,
        "an empty term is no filter at all"
    );
    assert_eq!(state.picker_rows()[0], "semantic", "current value back on top");
}

/// **The two `Esc`s.** The first undoes the typing, the second undoes the opening — so a reader
/// who typed nothing gets the close on the first press, there being nothing for it to undo.
#[test]
fn the_first_esc_resets_the_filter_and_the_second_closes_unchanged() {
    let mut state = record();
    state.mode = Mode::Edit { at: 5, edit: None };
    state.key(key(KeyCode::Char('j')));
    chars(&mut state, "li");
    assert!(!state.picker.as_ref().unwrap().filter.is_empty());

    state.key(key(KeyCode::Esc));
    let picker = state
        .picker
        .as_ref()
        .expect("still open after the first Esc");
    assert!(picker.filter.is_empty(), "the filter was not reset");
    assert_eq!(
        state.picker_rows()[0],
        "semantic",
        "current value back on top"
    );

    state.key(key(KeyCode::Esc));
    assert!(state.picker.is_none(), "the second Esc closes it");
    assert!(
        matches!(value_at(&state, 5), Value::Choice { at: 2, .. }),
        "…and closing changed nothing"
    );
}

/// A single `Esc` closes a list nothing was typed into.
#[test]
fn one_esc_closes_a_list_with_no_filter() {
    let mut state = record();
    state.mode = Mode::Edit { at: 5, edit: None };
    state.key(key(KeyCode::Char('j')));
    state.key(key(KeyCode::Esc));
    assert!(state.picker.is_none());
}

/// `↵` commits the row the cursor is on and closes — and the index it writes is the one in the
/// FIELD's own order, not the row number in a sorted, filtered list.
#[test]
fn enter_commits_the_highlighted_row_then_closes() {
    let mut state = record();
    state.mode = Mode::Edit { at: 5, edit: None };
    state.key(key(KeyCode::Char('j')));
    chars(&mut state, "tsf");
    assert_eq!(state.key(key(KeyCode::Enter)), Reaction::Handled);
    assert!(state.picker.is_none(), "committing closes it");
    let Value::Choice { at, choices } = value_at(&state, 5) else {
        panic!("not a choice field");
    };
    assert_eq!(choices[at], "tree-sitter/function");
    assert_eq!(at, 0, "the index is the field's own, not the visible row");
}

/// While a list is open it owns every key, so `Tab` does not traverse out from under it.
#[test]
fn an_open_list_owns_the_keyboard() {
    let mut state = record();
    state.mode = Mode::Edit { at: 5, edit: None };
    state.key(key(KeyCode::Char('j')));
    state.key(key(KeyCode::Tab));
    assert_eq!(state.mode.at(), 5, "Tab traversed out of an open list");
    assert!(state.picker.is_some());
}

// ---------------------------------------------------------------- the help, and the help rows

/// **`?` opens the contextual help from a KEYPRESS**, which round 1 could not do — its help was
/// a frame the pantry drew and nothing else could reach.
#[test]
fn question_mark_opens_the_help_and_two_keys_close_it() {
    let mut stack = stack_of(record());
    assert!(!stack.helping());
    assert_eq!(stack.key(key(KeyCode::Char('?'))), Reaction::Handled);
    assert!(stack.helping());

    assert_eq!(stack.key(key(KeyCode::Esc)), Reaction::Handled);
    assert!(!stack.helping());
    stack.key(key(KeyCode::Char('?')));
    stack.key(key(KeyCode::Char('?')));
    assert!(!stack.helping(), "? closes it as well as opening it");
}

/// It scrolls, which is the half the ruling names: the content does not fit and is not meant to.
#[test]
fn the_help_scrolls_and_stops_at_both_ends() {
    let mut stack = stack_of(record());
    stack.key(key(KeyCode::Char('?')));
    assert_eq!(stack.help_offset(), 0);
    stack.key(key(KeyCode::Char('k')));
    assert_eq!(stack.help_offset(), 0, "clamped at the top");
    stack.key(key(KeyCode::Char('j')));
    assert_eq!(stack.help_offset(), 1);
    let total = stack.help_lines().len();
    assert!(total > 20, "a help that fits is not the one the ruling describes: {total}");
    for _ in 0..total * 2 {
        stack.key(key(KeyCode::Down));
    }
    assert_eq!(stack.help_offset(), total - 1, "clamped at the bottom");
}

/// While it is open it owns the keyboard: the record underneath does not move.
#[test]
fn the_open_help_owns_the_keyboard() {
    let mut stack = stack_of(record());
    stack.key(key(KeyCode::Char('j')));
    let before = record_at(&stack);
    stack.key(key(KeyCode::Char('?')));
    stack.key(key(KeyCode::Char('j')));
    stack.key(key(KeyCode::Char('e')));
    assert_eq!(record_at(&stack), before, "the record moved under the help");
}

/// **`?` is the window's, so a view that wants the key keeps it.** Inside an open drop-down a
/// `?` is a character to filter with, not a request for help.
#[test]
fn the_record_gets_the_key_first_and_help_only_takes_what_is_left() {
    let mut stack = stack_of(record());
    // Open the drop-down, which claims every key.
    for _ in 0..5 {
        stack.key(key(KeyCode::Char('j')));
    }
    stack.key(key(KeyCode::Char('e')));
    stack.key(key(KeyCode::Char('j')));
    stack.key(key(KeyCode::Char('?')));
    assert!(!stack.helping(), "the open list did not keep the key");
}

/// Every entry of the record's help names a key the dispatcher actually answers to.
///
/// Not a spelling check — a promise check. A help window is the one place a binding the code does
/// not implement is invisible, so the keys it advertises for each field kind are pressed here and
/// the state is read back.
#[test]
fn the_help_advertises_only_keys_that_do_something() {
    let mut state = record();
    // Radio row: `l`.
    state.mode = Mode::Edit { at: 2, edit: None };
    assert_eq!(state.key(key(KeyCode::Char('l'))), Reaction::Handled);
    // Radio column: `j`.
    state.mode = Mode::Edit { at: 3, edit: None };
    assert_eq!(state.key(key(KeyCode::Char('j'))), Reaction::Handled);
    // Tick box: Space.
    state.mode = Mode::Edit { at: 4, edit: None };
    assert_eq!(state.key(key(KeyCode::Char(' '))), Reaction::Handled);
    // Drop-down: down opens, Esc closes.
    state.mode = Mode::Edit { at: 5, edit: None };
    assert_eq!(state.key(key(KeyCode::Down)), Reaction::Handled);
    assert_eq!(state.key(key(KeyCode::Esc)), Reaction::Handled);
    // And the window's own two.
    let mut stack = stack_of(record());
    assert_eq!(stack.key(key(KeyCode::Char('?'))), Reaction::Handled);
    stack.key(key(KeyCode::Esc));
    assert_eq!(stack.key(key(KeyCode::Backspace)), Reaction::Handled);
}

/// **The bottom help rows are what the ACTIVE field answers to**, and they change as the cursor
/// moves across the kinds — which is the whole difference from round 1's fixed strings.
#[test]
fn the_help_rows_follow_the_field_the_cursor_is_on() {
    let mut state = record();
    let keys_of = |state: &RecordState| -> Vec<String> {
        state.hints().into_iter().map(|(k, _)| k).collect()
    };
    assert!(keys_of(&state).contains(&"e".to_string()), "view mode offers `e`");

    state.mode = Mode::Edit { at: 2, edit: None };
    assert!(keys_of(&state).iter().any(|k| k.starts_with("h/l")));
    state.mode = Mode::Edit { at: 3, edit: None };
    assert!(keys_of(&state).iter().any(|k| k.starts_with("j/k")));
    state.mode = Mode::Edit { at: 4, edit: None };
    assert!(keys_of(&state).contains(&"Space".to_string()));
    state.mode = Mode::Edit { at: 5, edit: None };
    assert!(keys_of(&state).iter().any(|k| k.contains("Open list") || k.contains('j')));

    state.key(key(KeyCode::Down));
    let open = state.hints();
    assert!(
        open.iter().any(|(_, label)| label == "Choose"),
        "an open list advertises its own keys: {open:?}"
    );
}

/// A window whose layer states no hints takes the view's, and one that states them keeps its own
/// — the override the composition needs for a verb that is not the view's.
#[test]
fn the_layer_can_override_the_rows_and_otherwise_the_view_supplies_them() {
    let derived = stack_of(record());
    let rows = derived.decoration().hints().to_vec();
    assert!(rows.iter().any(|(k, _)| k == "e"), "derived from the view: {rows:?}");

    let mut layer = Layer::new("Queue item", "Queue item", View::Record(record()));
    layer.hints.push(("x".into(), "Composition's own".into()));
    let overridden = Stack::new(layer);
    let rows = overridden.decoration().hints().to_vec();
    assert_eq!(rows, vec![("x".to_string(), "Composition's own".to_string())]);
}

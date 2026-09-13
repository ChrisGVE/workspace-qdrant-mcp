//! ONE engine, TWO key tables — how *"you'll need two anyways: a non-vim and a vim version"*
//! (Chris, 2026-09-13 21:42) is served without a second crate.
//!
//! The engine consumes `Action`s and knows nothing about keys; a table is what turns keys into
//! actions. [`Keymap::Vim`] is `modalkit`'s vim table as shipped — the operator grammar of
//! ruling B, with 637 assertions behind it that are not ours to maintain. [`Keymap::Conventional`]
//! is its modeless Emacs table re-keyed to what a non-vim user expects: arrows, Home/End,
//! Shift+arrows to select (already there), Backspace/Delete (already there), and `Ctrl-Z` /
//! `Ctrl-Y` for undo/redo — the two mappings added here, over Emacs's own `C-z` (suspend) and
//! `C-y` (yank). They are built as edge paths by hand because the crate's key-string parser
//! is private.
//!
//! # Why the typed machines, not `KeyManager`
//!
//! `KeyManager` boxes the machine behind the `BindingMachine` trait, which only reports the mode
//! as a display string. Holding the `ModalMachine` itself keeps `mode()` typed, so
//! [`Keymap::mode`] maps `VimMode` onto the three [`EditMode`]s the widgets draw — insert,
//! normal, visual — and the conventional table, having no modes, always reads as insert.
//! Macros, the one thing `KeyManager` adds, are not part of any ruling.

use std::str::FromStr;

use modalkit::{
    actions::{Action, HistoryAction},
    editing::{application::EmptyInfo, context::EditContext},
    env::{
        emacs::{
            EmacsMode,
            keybindings::{EmacsMachine, InputStep, default_emacs_keys},
        },
        vim::{
            VimMode,
            keybindings::{VimMachine, default_vim_keys},
        },
    },
    key::TerminalKey,
    keybindings::{BindingMachine, EdgeEvent, EdgeRepeat},
    prelude::{Count, RepeatType},
};

use crate::widgets::edit_field::EditMode;

/// One engine with either Vim's modal grammar or conventional field bindings.
pub enum Keymap {
    Vim(VimMachine<TerminalKey, EmptyInfo>),
    Conventional(EmacsMachine<TerminalKey, EmptyInfo>),
}

impl Keymap {
    pub fn vim() -> Self {
        Self::Vim(default_vim_keys::<EmptyInfo>())
    }

    pub fn conventional() -> Self {
        let mut machine = default_emacs_keys::<EmptyInfo>();
        // The conventional table is arrows, Home/End, Shift+arrows selection,
        // Backspace/Delete, Ctrl-Z/Ctrl-Y. These two mappings override Emacs's
        // C-z (suspend) and C-y (yank).
        for (key, action) in [
            ("<C-Z>", HistoryAction::Undo(Count::Contextual)),
            ("<C-Y>", HistoryAction::Redo(Count::Contextual)),
        ] {
            let path = [(
                EdgeRepeat::Once,
                EdgeEvent::Key(TerminalKey::from_str(key).expect("valid control key")),
            )];
            let step = InputStep::new().actions(vec![action.into()]);
            machine.add_mapping(EmacsMode::Insert, &path, &step);
        }
        Self::Conventional(machine)
    }

    pub fn mode(&self) -> EditMode {
        match self {
            Self::Vim(machine) => match machine.mode() {
                VimMode::Insert => EditMode::Insert,
                VimMode::Visual | VimMode::Select => EditMode::Visual,
                _ => EditMode::Normal,
            },
            Self::Conventional(_) => EditMode::Insert,
        }
    }

    pub(crate) fn input_key(&mut self, key: TerminalKey) {
        match self {
            Self::Vim(machine) => machine.input_key(key),
            Self::Conventional(machine) => machine.input_key(key),
        }
    }

    pub(crate) fn pop(&mut self) -> Option<(Action<EmptyInfo>, EditContext)> {
        match self {
            Self::Vim(machine) => machine.pop(),
            Self::Conventional(machine) => machine.pop(),
        }
    }

    pub(crate) fn repeat(&mut self, repeat: RepeatType, ctx: EditContext) {
        match self {
            Self::Vim(machine) => machine.repeat(repeat, Some(ctx)),
            Self::Conventional(machine) => machine.repeat(repeat, Some(ctx)),
        }
    }
}

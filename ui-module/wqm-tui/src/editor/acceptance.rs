//! The §E-2 probe, kept as the acceptance test the adoption inherits.
//!
//! Every case here is one line of ruling B run against the real crate, headless: the keys are
//! fed through `modalkit`'s vim table, the actions are dispatched onto a `TextBoxState`, and the
//! text is read back. A bump of the `=0.0.27` pin that breaks one of these has changed the
//! grammar Chris ruled on, whatever its changelog says.
use modalkit::{
    actions::{Action, Editable, Jumpable, Scrollable},
    editing::{application::EmptyInfo, context::Resolve, key::KeyManager, store::Store},
    env::vim::keybindings::default_vim_keys,
    keybindings::BindingMachine,
};
use modalkit::{key::TerminalKey, prelude::RepeatType};
use modalkit_ratatui::textbox::{TextBox, TextBoxState};
use ratatui::{buffer::Buffer, layout::Rect, widgets::StatefulWidget};
use modalkit::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// The three things a field needs under it: the store, the binding machine, the buffer state.
struct Probe {
    store: Store<EmptyInfo>,
    bindings: KeyManager<TerminalKey, Action<EmptyInfo>, RepeatType>,
    tbox: TextBoxState<EmptyInfo>,
}

impl Probe {
    fn new(text: &str) -> Self {
        let mut store: Store<EmptyInfo> = Store::default();
        let bindings = KeyManager::new(default_vim_keys::<EmptyInfo>());
        let mut tbox = TextBoxState::new(store.load_buffer("probe".to_string()));
        tbox.set_text(text);
        Self { store, bindings, tbox }
    }

    /// Feed keys; `\x1b` in `keys` is Escape, an ASCII uppercase letter carries SHIFT.
    fn feed(&mut self, keys: &str) {
        for ch in keys.chars() {
            let code = if ch == '\x1b' { KeyCode::Esc } else { KeyCode::Char(ch) };
            let mods = if ch.is_ascii_uppercase() { KeyModifiers::SHIFT } else { KeyModifiers::NONE };
            self.bindings.input_key(KeyEvent::new(code, mods).into());
            while let Some((act, ctx)) = self.bindings.pop() {
                match act {
                    Action::Editor(a) => {
                        let _ = self.tbox.editor_command(&a, &ctx, &mut self.store);
                    }
                    Action::Scroll(s) => {
                        let _ = self.tbox.scroll(&s, &ctx, &mut self.store);
                    }
                    Action::Jump(l, d, c) => {
                        let _ = self.tbox.jump(l, d, ctx.resolve(&c), &ctx);
                    }
                    Action::Repeat(rt) => self.bindings.repeat(rt, Some(ctx)),
                    _ => {}
                }
            }
        }
    }

    /// The buffer text without the rope's trailing newline (§E-2, cost 5).
    fn text(&self) -> String {
        self.tbox.get_text().trim_end_matches('\n').to_string()
    }
}

/// A count on a find motion under an operator.
#[test]
fn d2t_deletes_up_to_the_second_comma() {
    let mut p = Probe::new("alpha, beta, gamma, delta");
    p.feed("0d2t,");
    assert_eq!(p.text(), ", gamma, delta");
}

/// `v2t,` — a count composing inside visual mode, the case Chris named.
#[test]
fn v2t_selects_and_d_deletes_the_same_range() {
    let mut p = Probe::new("alpha, beta, gamma, delta");
    p.feed("0v2t,d");
    assert_eq!(p.text(), ", gamma, delta");
}

/// `F` backwards, `;` repeats it, `x` deletes under the caret.
#[test]
fn f_backwards_and_semicolon_repeat() {
    let mut p = Probe::new("a,b,c");
    p.feed("$F,;x");
    assert_eq!(p.text(), "ab,c");
}

/// A text object under `c`, then `.` repeats the whole change on the next word.
#[test]
fn ciw_then_dot_repeats_the_change() {
    let mut p = Probe::new("one two three");
    p.feed("ciwX\x1bw.");
    assert_eq!(p.text(), "X X three");
}

/// The adapter renders into a `Buffer` with no terminal, on one line.
#[test]
fn renders_headless_into_a_single_line_buffer() {
    let mut p = Probe::new("one two three");
    p.feed("ciwX\x1bw.");
    let area = Rect::new(0, 0, 20, 1);
    let mut buf = Buffer::empty(area);
    TextBox::new().render(area, &mut buf, &mut p.tbox);
    let row: String = (0..20).map(|x| buf.cell((x, 0)).unwrap().symbol().to_string()).collect();
    assert_eq!(row, "X X three           ");
}

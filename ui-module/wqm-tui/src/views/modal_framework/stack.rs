//! The drill-down **stack** — one window carrying many views, and what a pop costs.
//!
//! Chris, 2026-09-13 01:06: *"having modal window that would have the mechanisms to drill
//! down: from one view going to another view by selection from a table for instance, and
//! having breadcrumbs in the window (chevron style)"*, ruled at 12:52 as **option A**: ONE
//! window carrying a stack of views, rather than a window per step.
//!
//! # The stack holds live state, not render snapshots
//!
//! *"Every view on the stack KEEPS its full state — cursor position, dirty flag, edited
//! content — so a pop restores the parent exactly as left, AND a drill-down's edits can update
//! the parent's content."* A stack of pictures could not do the second half. So a [`Layer`]
//! owns its view, and a view owns its cursor, its scroll offset and its edits; popping
//! discards the top and finds the parent where it was left, because it was never anywhere
//! else.
//!
//! # Backspace belongs to the editor first
//!
//! Pop is Backspace, and **only outside edit mode** — inside one it is a keystroke the field
//! is entitled to, and a stack that took it would delete a character's worth of navigation.
//! [`Stack::pop`] answers [`Pop::HeldByEditor`] rather than silently doing nothing, so the
//! caller can tell *refused* from *nothing happened* and a test can tell them apart too.
//!
//! # A dirty pop asks
//!
//! Popping over a view with unsaved edits opens a confirm on `Fill::Layer2` — the *modal over
//! a modal* variant, which is precisely the job §6 left that layer. The guard is the existing
//! [`crate::widgets::modal::Modal`], unchanged, centred on the WINDOW rather than the screen:
//! a dialogue belongs to the thing it is asking about, and the window sits below the page
//! header, so centring it on the screen puts it above the window's own middle.
//!
//! # The breadcrumb is the stack, read out
//!
//! Which is what makes it start-dependent (task 3): the same view reached two ways shows two
//! trails, because the trail IS the path taken and nothing here consults a hierarchy.

use ratatui::{buffer::Buffer, layout::Rect, style::Style, widgets::Widget};

use modalkit::crossterm::event::{KeyCode, KeyEvent};

use super::keys::{Keys, Picker, Reaction};
use super::record::{DropDown, Mode, RecordView};
use super::table::TableView;
use crate::tokens;
use crate::panes::list::help::{self, HelpSection};
use crate::widgets::modal::{Fill, Modal};
use crate::widgets::modal_frame::{Container, Decoration, Scroll};

#[cfg(test)]
mod tests;

/// What a window can hold — the two view kinds of the composition ruling.
pub enum View {
    Record(RecordState),
    Table(TableView),
}

/// A record view's own state, kept across a push so a pop finds it as it was left.
///
/// The fields are here rather than inside a built [`RecordView`] because a widget is consumed
/// by its own render and a stacked view is drawn many times. [`RecordState::view`] builds one
/// on demand.
pub struct RecordState {
    pub fields: Vec<super::record::FieldRow>,
    pub mode: Mode,
    pub reference: super::record::Reference,
    pub scheme: super::record::Scheme,
    pub offset: usize,
    /// Which key table a field opened here is driven by, and therefore who draws the caret.
    pub keys: Keys,
    /// The live engine under the active TEXT field, or [`None`] where the active field is a
    /// radio, a tick box or a drop-down — those have no text to type into.
    ///
    /// **Boxed**, and the reason is measurable: a `modalkit` `Store` plus a `TextBoxState` is
    /// about 1.7 KB, which every `View::Record` would otherwise carry whether or not a field
    /// was open — and `View` is moved on every push and pop. One pointer when closed is the
    /// right trade for a value that exists only while somebody is typing.
    ///
    /// It is skipped by [`PartialEq`] and by [`Clone`]: a `modalkit` buffer is neither, and a
    /// record is compared and copied in tests for its VALUES. What the renderer needs from the
    /// engine is already in `mode`, refreshed after every keystroke.
    pub editor: Option<Box<crate::editor::Field>>,
    /// An open drop-down over the active field.
    pub picker: Option<Picker>,
}

impl RecordState {
    pub fn new(fields: Vec<super::record::FieldRow>, mode: Mode) -> Self {
        Self {
            fields,
            mode,
            reference: super::record::Reference::None,
            scheme: super::record::Scheme::default(),
            offset: 0,
            keys: Keys::default(),
            editor: None,
            picker: None,
        }
    }

    pub fn reference(mut self, reference: super::record::Reference) -> Self {
        self.reference = reference;
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// The widget for this state, built fresh each render.
    pub fn view(&self) -> RecordView {
        RecordView::new(self.fields.clone(), self.mode.clone())
            .reference(self.reference)
            .scheme(self.scheme)
            .offset(self.offset)
            .caret(self.keys.caret())
    }

    /// The drop-down this record has open, anchored on the value cell it came out of.
    ///
    /// [`None`] unless one is open. The list is derived from the field every time, so an open
    /// picker cannot come to disagree with the value it was opened from.
    pub fn drop_down(&self, viewport: Rect) -> Option<DropDown> {
        let picker = self.picker.as_ref()?;
        let at = self.mode.at();
        let (choices, current) = self.fields.get(at)?.value().choices()?;
        let anchor = self.value_cell(viewport, at);
        let mut list = DropDown::new(choices.to_vec(), current, anchor).item_three();
        if !picker.filter.is_empty() {
            list = list.filter(picker.filter.clone());
        }
        Some(list)
    }

    /// Where field `index`'s VALUE cell sits inside `viewport` — what a drop-down opens out of.
    fn value_cell(&self, viewport: Rect, index: usize) -> Rect {
        let view = self.view();
        let header = view.header_rows() as u16;
        let rows_above: usize = self
            .fields
            .iter()
            .take(index)
            .map(|field| field.value().rows(view.value_width_at(viewport.width)))
            .sum();
        let x = viewport.x + (super::record::GUTTER + super::record::W_LABEL) as u16;
        Rect {
            x,
            y: viewport.y + header + rows_above.saturating_sub(self.offset) as u16,
            width: view.value_width_at(viewport.width) as u16,
            height: 1,
        }
    }
}

impl View {
    /// Whether an editor currently owns the keystrokes.
    pub fn editing(&self) -> bool {
        match self {
            View::Record(record) => record.mode.editing(),
            // A table drills down and narrows; nothing in it is typed into in round 1.
            View::Table(_) => false,
        }
    }

    /// Where the view is scrolled to, and how far it could scroll — what the container's
    /// scrollbar is a fraction of.
    pub fn scroll(&self, viewport: Rect) -> Scroll {
        match self {
            View::Record(record) => Scroll {
                offset: record.offset,
                total: record.view().rows(viewport.width),
            },
            View::Table(table) => table.scroll(),
        }
    }

    /// Rows of chrome the view draws above its data — the reference header, for a record.
    fn header_rows(&self, viewport: Rect) -> u16 {
        match self {
            View::Record(record) => record.view().header_rows() as u16,
            // `ListPane` draws its own column-title row inside the area it is given, so it
            // reports none: the row is already inside the height it was handed.
            View::Table(_) => {
                let _ = viewport;
                0
            }
        }
    }

    /// One keystroke, offered to whatever the window is holding.
    ///
    /// A table answers [`Reaction::Ignored`] to everything: its own navigation is
    /// [`crate::panes::list::ListPane`]'s and in-table editing is task 2, which round 2 does not
    /// cover. Saying so here rather than silently doing nothing is what lets the window try the
    /// key next.
    pub fn key(&mut self, key: KeyEvent) -> Reaction {
        match self {
            View::Record(record) => record.key(key),
            View::Table(_) => Reaction::Ignored,
        }
    }

    /// The bottom help rows this view offers right now.
    pub fn hints(&self) -> Vec<(String, String)> {
        match self {
            View::Record(record) => record.hints(),
            View::Table(_) => vec![
                ("\u{2193}\u{2191}/jk".into(), "Move".into()),
                ("\u{21b5}".into(), "Drill down".into()),
                ("\u{232b}".into(), "Back".into()),
                ("?".into(), "Help".into()),
                ("q".into(), "Close".into()),
            ],
        }
    }

    /// What `?` shows over this view.
    ///
    /// Per view, because the 20:30 ruling makes the help window a COMPOSITION like any other —
    /// the same container around content the view supplies. A table inside a window reuses the
    /// Queue's own sections, so the keys a reader learned on the Queue tab are the keys the help
    /// names when that table is reached through a drill-down.
    pub fn help_sections(&self) -> Vec<HelpSection> {
        match self {
            View::Record(_) => record_help(),
            View::Table(_) => crate::views::queue::Queue::help_sections(),
        }
    }

    /// Draw the view into `area` — public because the slide renders a view into an off-screen
    /// buffer of its own before blitting it.
    pub fn render_into(&self, area: Rect, buf: &mut Buffer) {
        match self {
            View::Record(record) => record.view().render(area, buf),
            View::Table(table) => table.pane().render(area, buf),
        }
    }
}

/// One step of the drill-down: a view, what the breadcrumb calls it, and whether it is dirty.
pub struct Layer {
    pub view: View,
    /// This step's own crumb. The trail is every layer's, in order.
    pub crumb: String,
    /// The window's title while this layer is on top.
    pub title: String,
    /// An OVERRIDE for the bottom help rows — per-window, because *"filtering on op won't make
    /// sense for a practical table"*: some verbs belong to the composition rather than to the
    /// view.
    ///
    /// Empty is the normal case, and then the rows are [`View::hints`] — the bindings the view
    /// actually answers to, derived from the same dispatch that answers them. Round 1 spelled
    /// them out per frame, which is how a window comes to advertise a key nothing handles.
    pub hints: Vec<(String, String)>,
    /// Whether this view holds edits that have not been saved.
    pub dirty: bool,
}

impl Layer {
    pub fn new(crumb: impl Into<String>, title: impl Into<String>, view: View) -> Self {
        Self {
            view,
            crumb: crumb.into(),
            title: title.into(),
            hints: Vec::new(),
            dirty: false,
        }
    }

    pub fn hint(mut self, key: impl Into<String>, label: impl Into<String>) -> Self {
        self.hints.push((key.into(), label.into()));
        self
    }

    pub fn dirty(mut self, dirty: bool) -> Self {
        self.dirty = dirty;
        self
    }
}

/// What a Backspace did.
///
/// Named outcomes rather than a `bool`, because *refused because the editor owns the key* and
/// *refused because this is the root* are different facts that a caller has to tell apart —
/// and because a silent no-op is indistinguishable from a bug.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pop {
    /// Popped. The parent is where it was left.
    Popped,
    /// Refused: the top view is in edit mode, and Backspace is the field's.
    HeldByEditor,
    /// Refused for now: the top view is dirty, so the confirm is open instead.
    Guarded,
    /// Refused: this is the only view, and a window's last view is the window.
    AtRoot,
}

/// The window's view stack.
pub struct Stack {
    /// Never empty: a window with no view is not a window.
    layers: Vec<Layer>,
    /// Open while a dirty pop is waiting for an answer.
    guard: bool,
    /// How far the contextual help is scrolled, while it is open.
    help: Option<usize>,
}

impl Stack {
    /// A window showing one view. The root, and the thing a pop can never take away.
    pub fn new(root: Layer) -> Self {
        Self {
            layers: vec![root],
            guard: false,
            help: None,
        }
    }

    /// Drill down. *"pressing Enter on a row will drill down one step with the focus on that
    /// row"* — which row that is, is the caller's; this takes the view it produced.
    pub fn push(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    pub fn top(&self) -> &Layer {
        self.layers.last().expect("a stack is never empty")
    }

    pub fn top_mut(&mut self) -> &mut Layer {
        self.layers.last_mut().expect("a stack is never empty")
    }

    /// Whether the discard guard is open over the window.
    pub fn guarded(&self) -> bool {
        self.guard
    }

    /// Whether the contextual help is open over the window.
    pub fn helping(&self) -> bool {
        self.help.is_some()
    }

    /// The help window's scroll offset, while it is open.
    pub fn help_offset(&self) -> usize {
        self.help.unwrap_or(0)
    }

    /// The lines `?` shows over this window — the top view's sections, rendered.
    pub fn help_lines(&self) -> Vec<ratatui::text::Line<'static>> {
        help::render(&self.top().view.help_sections())
    }

    /// **One keystroke, routed.**
    ///
    /// The order is the layer order and nothing else: whatever is nearest the reader gets the
    /// key first. A guard is nearest, then the help, then the view, and only what none of them
    /// claimed reaches the window's own verbs — which is why `?` and `Backspace` work in a
    /// record and not inside an open drop-down.
    pub fn key(&mut self, key: KeyEvent) -> Reaction {
        if self.guard {
            return match key.code {
                KeyCode::Enter => {
                    self.discard();
                    Reaction::Handled
                }
                KeyCode::Esc => {
                    self.keep_editing();
                    Reaction::Handled
                }
                _ => Reaction::Ignored,
            };
        }
        if let Some(offset) = self.help {
            let total = self.help_lines().len();
            return match key.code {
                KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('q') => {
                    self.help = None;
                    Reaction::Handled
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.help = Some(offset.saturating_add(1).min(total.saturating_sub(1)));
                    Reaction::Handled
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.help = Some(offset.saturating_sub(1));
                    Reaction::Handled
                }
                _ => Reaction::Ignored,
            };
        }
        let claimed = self.top_mut().view.key(key);
        if claimed != Reaction::Ignored {
            return claimed;
        }
        match key.code {
            KeyCode::Char('?') => {
                self.help = Some(0);
                Reaction::Handled
            }
            KeyCode::Backspace => {
                self.pop();
                Reaction::Handled
            }
            _ => Reaction::Ignored,
        }
    }

    /// The trail, root first — see the module docs for why it is the stack and not a
    /// hierarchy.
    pub fn crumbs(&self) -> Vec<String> {
        self.layers.iter().map(|l| l.crumb.clone()).collect()
    }

    /// Backspace.
    pub fn pop(&mut self) -> Pop {
        if self.guard {
            return Pop::Guarded;
        }
        if self.top().view.editing() {
            return Pop::HeldByEditor;
        }
        if self.layers.len() == 1 {
            return Pop::AtRoot;
        }
        if self.top().dirty {
            self.guard = true;
            return Pop::Guarded;
        }
        self.layers.pop();
        Pop::Popped
    }

    /// The guard's *discard and pop*: throw the top view's edits away and go back.
    pub fn discard(&mut self) -> Pop {
        if !self.guard {
            return Pop::Guarded;
        }
        self.guard = false;
        if self.layers.len() == 1 {
            return Pop::AtRoot;
        }
        self.layers.pop();
        Pop::Popped
    }

    /// The guard's *keep editing*: close it and leave the stack exactly as it was.
    pub fn keep_editing(&mut self) {
        self.guard = false;
    }

    /// The decoration this window wears right now: the trail, the top layer's title, its keys,
    /// and the fifth row when the top view mounts one.
    pub fn decoration(&self) -> Decoration {
        let top = self.top();
        let mut deco = Decoration::new(top.title.clone()).crumbs(self.crumbs());
        if let View::Table(table) = &top.view
            && let Some(row) = table.search_row()
        {
            deco = deco.search(row);
        }
        let hints = if top.hints.is_empty() {
            top.view.hints()
        } else {
            top.hints.clone()
        };
        for (key, label) in hints {
            deco = deco.hint(key, label);
        }
        deco
    }

    /// The window, drawn: container, decoration, the top view, and the guard over it.
    ///
    /// The container is measured before the view is built, because the scrollbar costs a
    /// column and the view has to be asked how many rows it needs at the width it will
    /// actually get.
    pub fn render(&self, rect: Rect, buf: &mut Buffer) {
        let probe = Container::new(self.decoration()).scroll(Scroll {
            offset: 0,
            total: 0,
        });
        let measured = probe.viewport(rect);
        let top = self.top();
        let scroll = top.view.scroll(measured);
        let data_rows = measured
            .height
            .saturating_sub(top.view.header_rows(measured));

        let mut container = Container::new(self.decoration());
        if scroll.total > data_rows as usize {
            container = container.scroll(scroll);
        }
        let viewport = container.viewport(rect);
        container.render(rect, buf);
        top.view.render_into(viewport, buf);

        // An open drop-down sits over the window, on the layer the confirm uses — it is the
        // field opening, so it is drawn after the view and inside the window's own rect.
        if let View::Record(record) = &top.view
            && let Some(list) = record.drop_down(viewport)
        {
            list.render(rect, buf);
        }

        if self.help.is_some() {
            self.render_help(rect, buf);
        }
        if self.guard {
            discard_guard().render(rect, buf);
        }
    }

    /// **The contextual help is a window like any other** (task 6: *"the help window is also a
    /// composition"*).
    ///
    /// Container plus fixed scrollable content, on the same rect as the window it is over and on
    /// `Layer2` — the layer §6 reserves for a thing opened from inside a window. Its content does
    /// not fit and is not meant to: the scroll is what the ruling asks for.
    fn render_help(&self, rect: Rect, buf: &mut Buffer) {
        let offset = self.help_offset();
        let lines = self.help_lines();
        let mut crumbs = self.crumbs();
        crumbs.push("Help".to_string());
        let deco = Decoration::new(format!("{} \u{2014} keys", self.top().title))
            .crumbs(crumbs)
            .hint("\u{2193}\u{2191}/jk", "Scroll")
            .hint("?/Esc", "Close");
        let container = Container::new(deco).fill(Fill::Layer2).scroll(Scroll {
            offset,
            total: lines.len(),
        });
        let viewport = container.viewport(rect);
        container.render(rect, buf);
        let shown: Vec<_> = lines.into_iter().skip(offset).collect();
        ratatui::widgets::Paragraph::new(shown).render(viewport, buf);
    }
}

/// **What `?` shows over a record**, and every entry is a key `views::modal_framework::keys`
/// actually dispatches.
///
/// Written as sections for the reason `panes::list::help` gives: a reader looking for one thing
/// should not have to read every key the window has to find it. The order is the order a reader
/// meets them — move, open, change the kind of field they are standing on, leave.
///
/// The field-kind section is the one worth reading twice. Its entries look redundant beside the
/// bottom help rows, and they are not: the bottom rows show only the ACTIVE field's keys, because
/// there are two of them, and this is where a reader finds out that a radio drawn as a column
/// answers to different keys from one drawn as a row — which is a thing they cannot discover by
/// standing on the row form.
pub fn record_help() -> Vec<HelpSection> {
    vec![
        HelpSection {
            title: "Moving",
            entries: vec![
                ("\u{2193} / j", "Down one field"),
                ("\u{2191} / k", "Up one field"),
                ("\u{21b9}", "Next editable field, while editing"),
                ("\u{21e7}\u{21b9}", "Previous editable field, while editing"),
            ],
            note: Some("Traversal visits the fields this window lets you change, and wraps."),
        },
        HelpSection {
            title: "Editing",
            entries: vec![
                ("e", "Edit the field under the cursor"),
                ("Esc", "Leave the field; under vim, Esc returns to normal mode first"),
                ("\u{21b5}", "Commit a single-line field"),
            ],
            note: Some("A field this window does not let you change cannot be opened."),
        },
        HelpSection {
            title: "Field kinds",
            entries: vec![
                ("h / l, \u{2194}", "Radio drawn as a row: change the choice"),
                ("j / k, \u{2195}", "Radio drawn as a column: change the choice"),
                ("Space", "Tick box: toggle"),
                ("\u{2193} / j", "Drop-down: open the list"),
            ],
            note: None,
        },
        HelpSection {
            title: "Drop-down list",
            entries: vec![
                ("j / k, \u{2195}", "Move the cursor"),
                ("^D / ^U", "Half a page"),
                ("^F / ^B", "A whole page"),
                ("a-z", "Type to narrow the list; matching is fuzzy"),
                ("Esc", "Reset what you typed; Esc again closes, unchanged"),
                ("\u{21b5}", "Choose the highlighted value and close"),
            ],
            note: Some("The current value is first; everything else is sorted under it."),
        },
        HelpSection {
            title: "The window",
            entries: vec![
                ("?", "This help; ? or Esc closes it"),
                ("\u{232b}", "Back one step, outside edit mode"),
                ("q", "Close the window"),
            ],
            note: Some("Going back over unsaved edits asks first."),
        },
    ]
}

/// The unsaved-edit guard: the existing modal, on the layer §6 reserved for exactly this.
pub fn discard_guard() -> Modal {
    Modal::with_body(
        "Discard changes?",
        vec!["This view has edits that have not been saved.".into()],
    )
    .fill(Fill::Layer2)
    .action("\u{21b5}", "discard")
    .action("Esc", "keep editing")
}

/// The push transition, frozen at `t`.
///
/// Task 3: *"Push slides new content in right-to-left"*. So at `t` the outgoing view sits
/// `t · width` columns to the LEFT of where it was, and the incoming view is arriving from the
/// right. **Only the viewport moves.** The decoration is the WINDOW's, not the view's, and it
/// is already at the destination — a breadcrumb that slid with its content would be a
/// breadcrumb that is briefly wrong, which is worse than one that arrives early.
///
/// Still frames, because this crate does not animate: `t = 0`, `0.5` and `1` are three
/// renders, and the middle one is the only one that says whether the motion will read. The
/// timer belongs to the live app.
pub fn slide(viewport: Rect, t: f32, outgoing: &Buffer, incoming: &Buffer, buf: &mut Buffer) {
    let width = viewport.width;
    if width == 0 {
        return;
    }
    let shift = ((t.clamp(0.0, 1.0) * width as f32).round() as u16).min(width);
    for row in 0..viewport.height {
        for column in 0..width {
            // Past the right edge of the outgoing view, the incoming one has arrived.
            let source = if column + shift < width {
                outgoing.cell((column + shift, row))
            } else {
                incoming.cell((column + shift - width, row))
            };
            if let (Some(source), Some(target)) = (
                source.cloned(),
                buf.cell_mut((viewport.x + column, viewport.y + row)),
            ) {
                *target = source;
            }
        }
    }
}

/// Draw `view` into an off-screen buffer the size of `viewport`, pre-filled with the window's
/// own surface so a blitted cell never punches a hole in the layer beneath it.
pub fn scratch(viewport: Rect, draw: impl FnOnce(Rect, &mut Buffer)) -> Buffer {
    let local = Rect {
        x: 0,
        y: 0,
        width: viewport.width,
        height: viewport.height,
    };
    let mut buf = Buffer::empty(local);
    buf.set_style(
        local,
        Style::default().bg(tokens::modal_fill(tokens::layer1_bg())),
    );
    draw(local, &mut buf);
    buf
}

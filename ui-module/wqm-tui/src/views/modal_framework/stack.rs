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

use super::record::{Mode, RecordView};
use super::table::TableView;
use crate::tokens;
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
}

impl RecordState {
    pub fn new(fields: Vec<super::record::FieldRow>, mode: Mode) -> Self {
        Self {
            fields,
            mode,
            reference: super::record::Reference::None,
            scheme: super::record::Scheme::default(),
            offset: 0,
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
    /// The keys this window offers — per-window, because *"filtering on op won't make sense
    /// for a practical table"*: the verbs belong to the composition, not to the view.
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
}

impl Stack {
    /// A window showing one view. The root, and the thing a pop can never take away.
    pub fn new(root: Layer) -> Self {
        Self {
            layers: vec![root],
            guard: false,
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
        for (key, label) in &top.hints {
            deco = deco.hint(key.clone(), label.clone());
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

        if self.guard {
            discard_guard().render(rect, buf);
        }
    }
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

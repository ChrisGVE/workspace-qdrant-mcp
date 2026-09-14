//! The frames round 1 is judged from — whole screens, never windows on an empty buffer.
//!
//! Every frame here is a **full composition**: the Queue page beneath, drawn inside a
//! [`crate::tokens::ModalScope`] so it goes quiet exactly as it does under the help window,
//! and the framework window on top of it. That is deliberate. A field background judged on an
//! empty buffer is judged against nothing, and *"is this window washed out and sad"* only has
//! an answer over the page it is covering.
//!
//! The fixture is a record this system genuinely has — one queue item, drilled into from the
//! Queue table — rather than a form invented to show the widgets off. It exercises every field
//! kind the ruling names, which is what makes a frame capable of being wrong.

use ratatui::{buffer::Buffer, layout::Rect, widgets::Widget};

use super::record::{CursorExtent, FieldRow, Mode, Reference, Scheme, Value, W_LABEL, W_REFERENCE};
use super::stack::{discard_guard, scratch, slide, Layer, RecordState, Stack, View};
use super::table::{Pin, TableView};
use crate::panes::status_block::{Queue as QueueCounts, StatusBlock, ENTRY_LABELS};
use crate::tokens::{self, Health, ModalTint};
use crate::views::queue::state::QueueState;
use crate::views::queue::{fixture, frames as queue, Queue};
use crate::widgets::chrome::Freshness;
use crate::widgets::modal_frame::{Container, Footprint, TooSmall};

#[cfg(test)]
mod tests;

/// The wash strength this round proposes. Twice [`tokens::WASH_MIX`] — see
/// [`tokens::DEFAULT_TINT_STRENGTH`], which is this number.
pub const PROPOSED_WASH: f32 = tokens::DEFAULT_TINT_STRENGTH;

/// How stale a reading may be before the block says so. The same sixty seconds every other
/// frame in this crate uses.
const FRAME_SLA: std::time::Duration = std::time::Duration::from_secs(60);

/// The trail the reader took to get here: depth 3, which is what makes the chevrons a
/// navigation aid rather than a decoration on a one-item list.
pub const CRUMBS: [&str; 3] = ["Queue", "open-books", "reading_guide.py"];

/// The library every drilled-in frame is pinned to.
pub const LIBRARY: &str = "open-books";

/// The choices the categorical field opens.
pub fn chunking() -> (Vec<String>, usize) {
    (
        vec![
            "tree-sitter/function".into(),
            "tree-sitter/class".into(),
            "fixed/512".into(),
            "fixed/1024".into(),
            "paragraph".into(),
            "whole-file".into(),
        ],
        0,
    )
}

/// The record the frames show: one queue item.
///
/// Every field kind is here — number, single-line text, a multi-line box, a tick box, a
/// four-choice radio and a six-choice drop-down — and the read-only fields are read-only
/// because a queue item's tenant and object are facts rather than settings. That is what makes
/// the EDIT-mode frames show a SET rather than a wall of slots.
pub fn record() -> Vec<FieldRow> {
    let (choices, at) = chunking();
    vec![
        FieldRow::new("Tenant", Value::Text(LIBRARY.into())).read_only(),
        FieldRow::new(
            "Object",
            Value::Text("book_building/common/stage_b/reading_guide.py".into()),
        )
        .read_only(),
        FieldRow::new("Queued", Value::Number("7 min ago".into())).read_only(),
        FieldRow::new("Size", Value::Number("31'539 B".into())).read_only(),
        FieldRow::new(
            "Operation",
            Value::Radio {
                choices: vec![
                    "add".into(),
                    "update".into(),
                    "delete".into(),
                    "scan".into(),
                ],
                at: 1,
            },
        )
        .reference("update"),
        FieldRow::new("Chunking", Value::Choice { choices, at }).reference("fixed/512"),
        FieldRow::new("Chunk overlap", Value::Number("128".into())).reference("64"),
        FieldRow::new("Watch for changes", Value::Bool(true)).reference("yes"),
        FieldRow::new("Re-embed on branch switch", Value::Bool(false)).reference("no"),
        FieldRow::new(
            "Note",
            Value::Multi(
                "Held back once already: the grammar download timed out against the registry, \
                 and the retry is queued behind the scan."
                    .into(),
            ),
        )
        .reference("\u{2014}"),
    ]
}

/// The page under every frame: the Queue tab with its captured buffer, quietened.
///
/// The same four health entries and the same counts the Dashboard's own fixtures use, so a
/// framework frame and a Dashboard frame cannot disagree about the state of the system.
pub fn page() -> Queue {
    let entries: [Health; ENTRY_LABELS.len()] = [
        Health::Healthy,
        Health::Degraded,
        Health::Healthy,
        Health::Healthy,
    ];
    let counts = QueueCounts {
        pending: 11_236,
        in_progress: 4,
        failed: 3,
        health: Health::Degraded,
    };
    let block = StatusBlock::new(
        crate::views::queue::overall(entries[0], &entries[1..]),
        "v0.2.0",
        Freshness::new(std::time::Duration::from_secs(4), FRAME_SLA),
        entries,
        counts,
    );
    Queue::new(QueueState::default(), block).under_modal(true)
}

fn draw_page(area: Rect, buf: &mut Buffer) {
    page().render(area, buf);
}

/// The keys a record window offers, which differ by mode — the verbs are the window's.
fn record_hints(layer: Layer, editing: bool) -> Layer {
    if editing {
        layer
            .hint("\u{21b9}", "Next field")
            .hint("\u{21e7}\u{21b9}", "Previous")
            .hint("\u{21b5}", "Open list")
            .hint("Esc", "Leave edit")
            .hint("?", "Help")
    } else {
        layer
            .hint("\u{2193}\u{2191}/jk", "Move")
            .hint("e", "Edit")
            .hint("\u{232b}", "Back")
            .hint("?", "Help")
            .hint("q", "Close")
    }
}

/// Every axis of a record frame, named rather than positional.
///
/// Eight axes spelled as positional arguments is where two get swapped and the frame is
/// quietly of something else — the designer hit that, and this is the shape that answers it.
pub struct RecordFrame {
    pub mode: Mode,
    pub reference: Reference,
    pub scheme: Scheme,
    pub footprint: Footprint,
    pub cursor_extent: CursorExtent,
    /// Draw the REJECTED arm A — the fill with no underline. **Evidence only**: see
    /// [`super::record::RecordView::rejected_arm_a`] for why A is not a shipping option, and
    /// use [`with_theme`] to render it on the theme that decides the question rather than on
    /// the roomy one the harness paints with.
    ///
    /// ⚠ **The polarity reads backwards at a glance and is worth reading twice.** `true` draws
    /// the arm that was THROWN OUT; the shipping frame is `false`. Every other flag on this
    /// struct turns something on, and this one turns the SET mark off — which is how the first
    /// version of `the_two_field_background_arms_differ_by_the_underline` came to assert the
    /// exact opposite of what it meant. The test went red, but that was luck rather than
    /// design, so the trap is written down here where the field is.
    pub arm_a: bool,
    pub offset: usize,
    /// Drawn as an empty record — the state the first round of frames did not show.
    pub empty: bool,
}

impl Default for RecordFrame {
    fn default() -> Self {
        Self {
            mode: Mode::View { at: 5 },
            reference: Reference::Band("DEFAULT"),
            scheme: Scheme::default(),
            footprint: Footprint::Max,
            cursor_extent: CursorExtent::default(),
            arm_a: false,
            offset: 0,
            empty: false,
        }
    }
}

impl RecordFrame {
    pub fn view(mode: Mode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }

    /// The stack this frame draws: one record, with the trail it was reached by.
    pub fn stack(&self) -> Stack {
        let editing = self.mode.editing();
        let fields = if self.empty { Vec::new() } else { record() };
        let state = RecordState {
            fields,
            mode: self.mode.clone(),
            reference: self.reference,
            scheme: self.scheme,
            cursor_extent: self.cursor_extent,
            offset: self.offset,
        };
        // Depth 3, which is what makes the chevrons a navigation aid rather than a decoration
        // on a one-item list: the Queue, narrowed to a library, then one item of it.
        let mut stack = Stack::new(Layer::new(CRUMBS[0], "Queue", View::Table(drilled_table())));
        stack.push(Layer::new(
            CRUMBS[1],
            "open-books \u{2014} queue",
            View::Table(drilled_table()),
        ));
        stack.push(record_hints(
            Layer::new(
                CRUMBS[2],
                "Queue item \u{2014} reading_guide.py",
                View::Record(state),
            ),
            editing,
        ));
        stack
    }

    /// Draw the whole screen: the page, quiet, and the window over it.
    ///
    /// [`None`] when the page is too small to hold a window at all — item 0's last sentence. The
    /// page is still drawn and [`TooSmall`] goes over it, so the frame says what is wrong rather
    /// than coming out blank; the caller gets nothing back because there is no window to hang a
    /// drop-down or a confirm on.
    pub fn draw(&self, area: Rect, buf: &mut Buffer) -> Option<(Rect, Rect)> {
        draw_page(area, buf);
        let Some(rect) = self.footprint.window(area) else {
            TooSmall::new(area).render(area, buf);
            return None;
        };
        let stack = self.stack();
        // Arm A is not something a `Stack` can be asked for — see `rejected_arm_a` — so the
        // evidence frame is drawn here, outside the shipping path, rather than by handing the
        // stack a flag it should not have.
        let viewport = Container::new(stack.decoration()).viewport(rect);
        if self.arm_a {
            draw_arm_a(&stack, rect, buf);
        } else {
            stack.render(rect, buf);
        }
        Some((rect, viewport))
    }
}

/// The REJECTED arm A, drawn so it can be looked at: the same window with no underline.
///
/// It is worth rendering precisely because it is the arm the measurement throws out, and a
/// PNG settles it in a way the ΔE number does not — on a thin theme the whole block of
/// editable fields is very nearly the window it sits on. Render it through
/// [`with_theme`] on Solarized Dark, not on the harness's own Mocha, or the frame shows the
/// best case of the thing being rejected.
fn draw_arm_a(stack: &Stack, rect: Rect, buf: &mut Buffer) {
    let mut container = Container::new(stack.decoration());
    let viewport = container.viewport(rect);
    if stack.top().view.editing() {
        container = Container::new(stack.decoration()).title_banner(super::stack::EDIT_BANNER);
    }
    container.render(rect, buf);
    if let View::Record(record) = &stack.top().view {
        record.view().rejected_arm_a().render(viewport, buf);
    }
}

/// The Queue's own table, pinned to the library the reader drilled in from.
///
/// Built from [`crate::views::queue::frames::cells`] — the Queue's own projection — so a
/// drilled-in table is the Queue's table rather than a second one that could come to disagree
/// with it.
pub fn drilled_table() -> TableView {
    let rows: Vec<Vec<crate::panes::cell::Cell>> = fixture::ROWS.iter().map(queue::cells).collect();
    TableView::new(queue::columns(), rows)
        .pinned(Pin::new(queue::TENANT, LIBRARY))
        .cursor(3)
}

/// The table window: the Queue's table inside a window, with the trail that reached it.
///
/// Trail depth 3 — `Libraries › open-books › Queue` — which is the entry point the ruling's
/// own example describes: a library selected, its detail opened, the queue field drilled into.
/// The two ancestors are steps of the path rather than views this fixture builds, which is
/// exactly the point of the breadcrumb being start-dependent.
pub fn table_stack(show_pinned: bool) -> Stack {
    let mut stack = Stack::new(Layer::new(
        "Libraries",
        "Libraries",
        View::Table(drilled_table()),
    ));
    stack.push(Layer::new(
        LIBRARY,
        LIBRARY,
        View::Record(RecordState::new(record(), Mode::View { at: 0 })),
    ));
    stack.push(
        Layer::new(
            "Queue",
            "open-books \u{2014} queue",
            View::Table(drilled_table().show_pinned_column(show_pinned)),
        )
        .hint("\u{2193}\u{2191}/jk", "Move")
        .hint("\u{21b5}", "Drill down")
        .hint("f", "Filter")
        .hint("\u{232b}", "Back")
        .hint("?", "Help"),
    );
    stack
}

/// A table frame over the quietened page.
pub fn table_frame(area: Rect, buf: &mut Buffer, show_pinned: bool) -> Rect {
    draw_page(area, buf);
    let rect = Container::footprint(area);
    table_stack(show_pinned).render(rect, buf);
    rect
}

/// A table whose floor matches nothing — the empty state the first round of frames did not
/// show.
pub fn empty_table_frame(area: Rect, buf: &mut Buffer) -> Rect {
    draw_page(area, buf);
    let rect = Container::footprint(area);
    let rows: Vec<Vec<crate::panes::cell::Cell>> = fixture::ROWS.iter().map(queue::cells).collect();
    let stack = Stack::new(
        Layer::new(
            "Libraries",
            "mnemosyne \u{2014} queue",
            View::Table(
                TableView::new(queue::columns(), rows).pinned(Pin::new(queue::TENANT, "mnemosyne")),
            ),
        )
        .hint("f", "Filter")
        .hint("\u{232b}", "Back")
        .hint("?", "Help"),
    );
    stack.render(rect, buf);
    rect
}

/// The unsaved-edit guard over a record window.
pub fn confirm_frame(area: Rect, buf: &mut Buffer) -> Option<Rect> {
    let frame = RecordFrame {
        mode: Mode::Edit {
            at: 6,
            edit: Some(crate::widgets::edit_field::Edit::insert("256")),
        },
        ..RecordFrame::default()
    };
    let (rect, _) = frame.draw(area, buf)?;
    discard_guard().render(rect, buf);
    Some(rect)
}

/// The categorical drop-down, open over the record view.
pub fn dropdown_frame(area: Rect, buf: &mut Buffer) -> Option<Rect> {
    // `Chunking` is field 5, and in EDIT mode it is the active one.
    let frame = RecordFrame {
        mode: Mode::Edit { at: 5, edit: None },
        ..RecordFrame::default()
    };
    let (rect, viewport) = frame.draw(area, buf)?;
    let (choices, at) = chunking();
    // The anchor is the value cell the list belongs to: the gutter, the label column, and then
    // as wide as the value cell itself — so the list is the cell, opened.
    let label_end = viewport.x + super::record::GUTTER as u16 + W_LABEL as u16;
    let header = 1; // the reference column's header row
    let anchor = Rect {
        x: label_end,
        y: viewport.y + header + 5,
        width: viewport
            .right()
            .saturating_sub(label_end + W_REFERENCE as u16 + 1),
        height: 1,
    };
    super::record::DropDown::new(choices, at, anchor).render(rect, buf);
    Some(rect)
}

/// The contextual help a view owns, opened with `?`.
///
/// The composition ruling's last consequence: *"the help window is also a composition: a modal
/// container and a fixed scrollable content"* — so it is this framework's Container with the
/// Queue's own help lines inside it, rather than a window of its own kind. Which is also the
/// §01 argument arriving from another direction: the help content does not fit the window, so
/// it scrolls, which arm A could never let it do.
pub fn help_frame(area: Rect, buf: &mut Buffer) -> Rect {
    draw_page(area, buf);
    let rect = Container::footprint(area);
    let lines = crate::panes::list::help::render(&Queue::help_sections());
    let deco = crate::widgets::modal_frame::Decoration::new("Queue \u{2014} keys")
        .crumbs(vec!["Queue", "Help"])
        .hint("\u{2193}\u{2191}/jk", "Scroll")
        .hint("Esc", "Close");
    let container = Container::new(deco).scroll(crate::widgets::modal_frame::Scroll {
        offset: 0,
        total: lines.len(),
    });
    let viewport = container.viewport(rect);
    container.render(rect, buf);
    ratatui::widgets::Paragraph::new(lines).render(viewport, buf);
    rect
}

/// The push transition, frozen at `t`: a record leaving left, the Queue arriving from the
/// right, and the decoration already at its destination.
pub fn slide_frame(area: Rect, buf: &mut Buffer, t: f32) {
    draw_page(area, buf);
    let rect = Container::footprint(area);

    // The decoration is the DESTINATION's: the window is already where it is going, and only
    // its content is still arriving.
    let destination = table_stack(false);
    let container = Container::new(destination.decoration());
    let viewport = container.viewport(rect);
    container.render(rect, buf);

    let outgoing = scratch(viewport, |area, buf| {
        RecordFrame::default()
            .stack()
            .top()
            .view
            .render_into(area, buf);
    });
    let incoming = scratch(viewport, |area, buf| {
        drilled_table().pane().render(area, buf);
    });
    slide(viewport, t, &outgoing, &incoming, buf);
}

/// The theme that decides the field-background question, which is **not** the one the harness
/// paints with.
///
/// Solarized Dark has the thinnest neutral ladder of the fifteen: its editable fill sits ΔE
/// 4.2 off the window before any tint and 2.6 at the proposed strength — one just-noticeable
/// difference — where Catppuccin Mocha has 6.8 and 4.8. A reserved treatment judged on Mocha
/// alone is a treatment judged on its best case, which is what §15 forbids.
pub const ADVERSARIAL_THEME: ratatui_themes::ThemeName = ratatui_themes::ThemeName::SolarizedDark;

/// Run `draw` under a stated theme, and put back the one that was in force.
///
/// So a frame can be judged on the theme that decides it rather than on the one we happen to
/// be looking at. `Palette::Bundled` is set alongside, because `tokens::active_theme` answers
/// `None` for every other source — a theme that has been *chosen* is not a theme that is *in
/// force*, and setting one without the other renders the slot fallbacks instead.
pub fn with_theme<T>(name: ratatui_themes::ThemeName, draw: impl FnOnce() -> T) -> T {
    struct Restore(tokens::Palette, Option<ratatui_themes::ThemePalette>);
    impl Drop for Restore {
        fn drop(&mut self) {
            tokens::Palette::set(self.0);
            if let Some(theme) = self.1 {
                tokens::set_theme(theme);
            }
        }
    }
    let _restore = Restore(tokens::Palette::current(), tokens::theme());
    tokens::Palette::set(tokens::Palette::Bundled);
    tokens::set_theme(name.palette());
    draw()
}

/// Run `draw` with a tint and a strength in force, and put back whatever was there — including
/// on a panic, because a leaked process-global tint would silently repaint every later frame.
pub fn with_tint<T>(tint: ModalTint, strength: f32, draw: impl FnOnce() -> T) -> T {
    struct Restore(ModalTint, f32);
    impl Drop for Restore {
        fn drop(&mut self) {
            ModalTint::set(self.0);
            tokens::set_tint_strength(self.1);
        }
    }
    let _restore = Restore(ModalTint::current(), tokens::tint_strength());
    ModalTint::set(tint);
    tokens::set_tint_strength(strength);
    draw()
}

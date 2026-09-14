//! **Round 2's frames** — one per clause of Chris's 2026-09-14 rulings, over the same page.
//!
//! Every frame is a whole screen with the Queue page quietened beneath it, for the reason round
//! 1 gave and that has not changed: a window judged on an empty buffer is judged against
//! nothing.
//!
//! # The proposal is in force, and it is one set of globals
//!
//! [`proposed`] puts round 2's whole look on at once — accent at 0.40 with lightness held, the
//! field rungs derived for black text, the SET mark carried by the fill with the underline as an
//! encoding fallback, powerline crumbs. A frame drawn under a partial set would be a frame of a
//! design nobody is proposing, and the parts interact: the underline can only go BECAUSE the
//! blend holds lightness, and black text on the active field is only legible BECAUSE the rung is
//! derived. The bracket variants turn one thing off at a time, which is what makes them
//! evidence.
//!
//! # Composed here rather than through `Stack`
//!
//! [`super::stack::Stack`] builds its own [`Decoration`] and knows nothing about the round-2
//! chrome options, and teaching it would mean deciding them before Chris has seen them. So these
//! frames put Container, Decoration and View together directly — which is what the composition
//! ruling says a window IS, so nothing is being smuggled past it.

use ratatui::{buffer::Buffer, layout::Rect, widgets::Widget};

use super::frames::{self, CRUMBS};
use super::record::{FieldRow, Mode, RecordView, Reference, Value};
use super::table::{Pin, TableView};
use crate::tokens::{self, field, ModalTint, TintBlend};
use crate::widgets::edit_field::Edit;
use crate::widgets::modal_frame::{
    Container, CrumbStyle, Decoration, Edge, Footprint, Scroll, TooSmall,
};

#[cfg(feature = "tui-pantry")]
pub mod ingredient;
#[cfg(test)]
mod tests;

/// The strength Chris ruled: *"I think your Tint: blue 0.40 is much better"*.
pub const RULED_STRENGTH: f32 = 0.40;

/// Every process-global round 2 touches, restored on the way out — including on a panic, because
/// a leaked tint silently repaints every later frame in the same process.
struct Restore(ModalTint, f32, TintBlend, field::FieldRungs, field::SetMark);

impl Drop for Restore {
    fn drop(&mut self) {
        ModalTint::set(self.0);
        tokens::set_tint_strength(self.1);
        TintBlend::set(self.2);
        field::FieldRungs::set(self.3);
        field::SetMark::set(self.4);
    }
}

/// Round 2's look, in force for `draw`.
pub fn proposed<T>(draw: impl FnOnce() -> T) -> T {
    let _restore = Restore(
        ModalTint::current(),
        tokens::tint_strength(),
        TintBlend::current(),
        field::FieldRungs::current(),
        field::SetMark::current(),
    );
    ModalTint::set(ModalTint::Accent);
    tokens::set_tint_strength(RULED_STRENGTH);
    TintBlend::set(TintBlend::HoldLuminance);
    field::FieldRungs::set(field::FieldRungs::BlackText);
    field::SetMark::set(field::SetMark::FillWithFallback);
    draw()
}

/// Round 1's look, for the arm of a pair that shows what changed.
pub fn round_one<T>(draw: impl FnOnce() -> T) -> T {
    let _restore = Restore(
        ModalTint::current(),
        tokens::tint_strength(),
        TintBlend::current(),
        field::FieldRungs::current(),
        field::SetMark::current(),
    );
    ModalTint::set(ModalTint::Accent);
    tokens::set_tint_strength(RULED_STRENGTH);
    TintBlend::set(TintBlend::Straight);
    field::FieldRungs::set(field::FieldRungs::Fixed);
    field::SetMark::set(field::SetMark::FillAndUnderline);
    draw()
}

/// The window's decoration, with round 2's trail and the verbs for the mode it is in.
fn decoration(crumbs: &[&str], title: &str, editing: bool) -> Decoration {
    let mut deco = Decoration::new(title)
        .crumbs(crumbs.to_vec())
        .crumb_style(CrumbStyle::Powerline);
    deco = if editing {
        deco.hint("\u{21b9}", "Next field")
            .hint("\u{21e7}\u{21b9}", "Previous")
            .hint("j/k", "Change")
            .hint("Esc", "Leave edit")
    } else {
        deco.hint("\u{2193}\u{2191}/jk", "Move")
            .hint("e", "Edit")
            .hint("\u{232b}", "Back")
            .hint("?", "Help")
            .hint("q", "Close")
    };
    deco
}

/// Everything a round-2 record frame can vary, named rather than positional.
pub struct Frame {
    pub mode: Mode,
    pub reference: Reference,
    pub footprint: Footprint,
    pub edge: Edge,
    pub fields: Vec<FieldRow>,
    pub title: &'static str,
    pub crumbs: &'static [&'static str],
    /// A horizontal bar, for the drill-down frame whose table cannot shrink.
    pub hscroll: Option<Scroll>,
}

impl Default for Frame {
    fn default() -> Self {
        Self {
            mode: Mode::View { at: 5 },
            reference: Reference::Band("DEFAULT"),
            footprint: Footprint::Max,
            edge: Edge::Bordered,
            fields: frames::record(),
            title: "Queue item \u{2014} reading_guide.py",
            crumbs: &CRUMBS,
            hscroll: None,
        }
    }
}

impl Frame {
    /// Draw the page and the window over it; [`None`] when the page cannot hold one, in which
    /// case the too-small message is what got drawn instead.
    pub fn draw(&self, area: Rect, buf: &mut Buffer) -> Option<(Rect, Rect)> {
        frames::page().render(area, buf);
        let Some(rect) = self.footprint.window(area) else {
            TooSmall::new(area).render(area, buf);
            return None;
        };
        let editing = self.mode.editing();
        let view = RecordView::new(self.fields.clone(), self.mode.clone()).reference(self.reference);

        let probe = Container::new(decoration(self.crumbs, self.title, editing));
        let measured = probe.viewport(rect);
        let rows = view.rows(measured.width);
        let data_rows = view.data_height(measured.height);

        let mut container = Container::new(decoration(self.crumbs, self.title, editing))
            .edge(self.edge);
        if rows > data_rows as usize {
            container = container.scroll(Scroll {
                offset: 0,
                total: rows,
            });
        }
        if let Some(hscroll) = self.hscroll {
            container = container.hscroll(hscroll);
        }
        let viewport = container.viewport(rect);
        container.render(rect, buf);
        RecordView::new(self.fields.clone(), self.mode.clone())
            .reference(self.reference)
            .render(viewport, buf);
        Some((rect, viewport))
    }
}

/// The record a Configuration-like window holds — **short**, which is the point of item 1's
/// first half: a window that is not part of a drill-down is as tall as what it contains.
pub fn short_record() -> Vec<FieldRow> {
    vec![
        FieldRow::new("Theme", Value::Choice {
            choices: vec![
                "Catppuccin Mocha".into(),
                "Catppuccin Latte".into(),
                "Gruvbox Dark".into(),
                "Nord".into(),
                "Solarized Dark".into(),
                "Tokyo Night".into(),
            ],
            at: 0,
        }),
        FieldRow::new("Editor keys", Value::Radio {
            choices: vec!["vim".into(), "conventional".into()],
            at: 0,
        }),
        FieldRow::new("Confirm before discard", Value::Bool(true)),
        FieldRow::new("Watch debounce [ms]", Value::Number("2000".into())),
    ]
}

/// The same choice set as a COLUMN — item 3's second radio form.
pub fn column_record() -> Vec<FieldRow> {
    vec![
        FieldRow::new("Tenant", Value::Text(frames::LIBRARY.into())).read_only(),
        FieldRow::new(
            "Operation",
            Value::RadioColumn {
                choices: vec![
                    "add".into(),
                    "update".into(),
                    "delete".into(),
                    "scan".into(),
                ],
                at: 1,
            },
        ),
        FieldRow::new("Chunk overlap", Value::Number("128".into())).reference("64"),
    ]
}

/// A record whose text fields are the ones being typed into, for the caret frames.
pub fn text_record(edit: Edit) -> (Vec<FieldRow>, Mode) {
    let fields = vec![
        FieldRow::new("Tenant", Value::Text(frames::LIBRARY.into())).read_only(),
        FieldRow::new("Label", Value::Text("reading guide".into())),
        FieldRow::new(
            "Note",
            Value::Multi(
                "Held back once already: the grammar download timed out against the registry, \
                 and the retry is queued behind the scan."
                    .into(),
            ),
        ),
    ];
    (
        fields,
        Mode::Edit {
            at: 1,
            edit: Some(edit),
        },
    )
}

/// **Item 1's drill-down frame**: a window at the maximum with a table too wide to shrink, so
/// both bars appear.
///
/// The table is the Queue's own, with the pinned column KEPT — which is what makes it too wide,
/// and is the honest way to produce the state rather than inventing a wider fixture.
pub fn wide_table(area: Rect, buf: &mut Buffer) -> Option<Rect> {
    frames::page().render(area, buf);
    let rect = Footprint::Max.window(area)?;
    let deco = Decoration::new("open-books \u{2014} queue")
        .crumbs(vec!["Libraries", "open-books", "Queue"])
        .crumb_style(CrumbStyle::Powerline)
        .hint("\u{2193}\u{2191}/jk", "Move")
        .hint("h/l", "Pan")
        .hint("\u{21b5}", "Drill down")
        .hint("\u{232b}", "Back")
        .hint("?", "Help");
    let rows: Vec<Vec<crate::panes::cell::Cell>> = crate::views::queue::fixture::ROWS
        .iter()
        .map(crate::views::queue::frames::cells)
        .collect();
    let table = TableView::new(crate::views::queue::frames::columns(), rows)
        .pinned(Pin::new(crate::views::queue::frames::TENANT, frames::LIBRARY))
        .show_pinned_column(true)
        .cursor(3);
    let container = Container::new(deco)
        .scroll(table.scroll())
        // The columns the table WANTS against the columns it is given. A real pan offset, so the
        // thumb is off the left edge rather than sitting at zero and saying nothing.
        .hscroll(Scroll {
            offset: 18,
            total: 150,
        });
    let viewport = container.viewport(rect);
    container.render(rect, buf);
    table.pane().render(viewport, buf);
    Some(rect)
}

/// **Item 7**: the confirm on `Layer2` over a window that is itself quietened.
///
/// Chris: *"the modal window under the new one must get the same modal treatement as the main
/// background and becoming readable but greyish"*. That treatment already exists and is
/// [`tokens::ModalScope`] — the thing the PAGE is drawn inside. So the window is drawn inside one
/// too, and the guard outside it, which is the whole implementation: the same rule at one more
/// level rather than a second rule that means the same thing.
pub fn confirm(area: Rect, buf: &mut Buffer) -> Option<Rect> {
    let frame = Frame {
        mode: Mode::Edit {
            at: 6,
            edit: Some(Edit::insert("256")),
        },
        ..Frame::default()
    };
    let rect = {
        let _quiet = tokens::ModalScope::enter();
        frame.draw(area, buf)?.0
    };
    super::stack::discard_guard().render(rect, buf);
    Some(rect)
}

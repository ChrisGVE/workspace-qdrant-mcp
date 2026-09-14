//! The **`Modal Framework` group** — one frame per state the window actually ships in.
//!
//! These were `Modal Framework R2` while round 2 was a proposal beside the round Chris had
//! already reacted to. His 19:05 rulings closed every question they were asking, so there is no
//! longer a question and an answer to keep apart: the two lists are one, the arms he retired are
//! gone from it, and what is left is what a reader will meet. Names keep the `NN item — what it
//! shows` form, which sorts the list into the order his message was written in.
//!
//! Two kinds of entry survive that are not simply *a state*:
//!
//! - **Frames on an adversarial theme.** Solarized Dark has the thinnest ladder of the fifteen,
//!   and a design judged only on the harness's own Catppuccin Mocha is judged on one of the
//!   roomiest.
//! - **Degradation frames.** `NO_COLOR` and `ansi16` are states the window really enters, and
//!   they are the ones Chris cannot see by looking at his own terminal.
//!
//! The one open pair left is the selected-text colour, which he asked to be shown
//! (*"we'll have to define a color for the selected text"*); its arm B is labelled as the
//! alternative the measurement rejects.

use ratatui::{buffer::Buffer, layout::Rect, widgets::Widget};
use tui_pantry::{Ingredient, PropInfo};

use super::{
    column_record, confirm, driven, proposed, short_record, text_record, typed_into,
    wide_table, Frame, RULED_STRENGTH,
};
use crate::tokens::{self, field, TintBlend};
use crate::views::modal_framework::frames as r1;
use crate::views::modal_framework::record::{DropDown, Mode, Reference, Value};
use crate::widgets::edit_field::Edit;
use crate::widgets::modal_frame::{Container, CrumbStyle, Decoration, Edge, Footprint};

const PROPS: &[PropInfo] = &[
    PropInfo {
        name: "size",
        ty: "widgets::modal_frame::Footprint",
        description: "Max = the page inset by 5 on all four sides; Content = sized to what it holds",
    },
    PropInfo {
        name: "tint",
        ty: "tokens::ModalTint + tokens::TintBlend",
        description: "blue (accent) at 0.40, lightness HELD — the blend is what makes 0.40 readable",
    },
    PropInfo {
        name: "field marks",
        ty: "tokens::field::{SetMark, FieldRungs}",
        description: "SET = fill alone (underline only where colour cannot carry it); POINT = derived for black text",
    },
    PropInfo {
        name: "trail",
        ty: "widgets::modal_frame::CrumbStyle",
        description: "Powerline segments in the modal hue; text is whichever ladder end is legible",
    },
    PropInfo {
        name: "edge",
        ty: "widgets::modal_frame::Edge",
        description: "Bordered, or Spacing — the same padding with no glyphs",
    },
];

struct Variant(&'static str, &'static str, fn(Rect, &mut Buffer));

impl Ingredient for Variant {
    fn tab(&self) -> &str {
        "Views"
    }
    fn group(&self) -> &str {
        "Modal Framework"
    }
    fn name(&self) -> &str {
        self.0
    }
    fn source(&self) -> &str {
        "wqm_tui::views::modal_framework::shipping"
    }
    fn description(&self) -> &str {
        self.1
    }
    fn props(&self) -> &[PropInfo] {
        PROPS
    }
    fn render(&self, area: Rect, buf: &mut Buffer) {
        (self.2)(area, buf)
    }
}

/// The `Chunking` field, mid-edit, which is where every drop-down frame opens from.
fn choice_open(area: Rect, buf: &mut Buffer, filter: Option<&'static str>) {
    let frame = Frame {
        mode: Mode::Edit { at: 5, edit: None },
        ..Frame::default()
    };
    let Some((rect, viewport)) = frame.draw(area, buf) else {
        return;
    };
    let (choices, at) = r1::chunking();
    let label_end = viewport.x
        + crate::views::modal_framework::record::GUTTER as u16
        + crate::views::modal_framework::record::W_LABEL as u16;
    let header = 1;
    let anchor = Rect {
        x: label_end,
        y: viewport.y + header + 5,
        width: viewport.right().saturating_sub(
            label_end + crate::views::modal_framework::record::W_REFERENCE as u16 + 1,
        ),
        height: 1,
    };
    let list = DropDown::new(choices, at, anchor).item_three();
    let list = match filter {
        Some(term) => list.filter(term),
        None => list,
    };
    ratatui::widgets::Widget::render(list, rect, buf);
}

/// The selected-text arms, on a field with a live range.
fn selection(area: Rect, buf: &mut Buffer, arm: field::SelectedText) {
    let restore = field::SelectedText::current();
    field::SelectedText::set(arm);
    let (fields, mode) = text_record(Edit::visual("reading guide", 8..13));
    Frame {
        fields,
        mode,
        reference: Reference::None,
        title: "Queue item \u{2014} label",
        ..Frame::default()
    }
    .draw(area, buf);
    field::SelectedText::set(restore);
}

pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
    vec![
        // ---- item 0 / 1, the size model -------------------------------------------------
        Box::new(Variant(
            "00 size — max window (dump at 125x34, 100x30, 200x40)",
            "Item 0: five columns of page each side, five rows top and bottom, at ANY size. Dump this one at all three to see it hold",
            |area, buf| {
                proposed(|| {
                    Frame::default().draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "01 size — the minimum window (dump at 44x20)",
            "The floor, derived: border + the four top rows + the three bottom ones + ONE content row. One row less and the viewport is empty",
            |area, buf| {
                proposed(|| {
                    Frame {
                        fields: short_record(),
                        reference: Reference::None,
                        title: "Configuration",
                        crumbs: &["Configuration"],
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "02 size — screen too small (dump at 40x18)",
            "Below the floor there is no window, so the message is not drawn in one. It names the size needed AND the size present, and shortens rather than truncating",
            |area, buf| {
                proposed(|| {
                    Frame::default().draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "03 size — content-sized, not a drill-down (A)",
            "Item 1's first half: a Configuration-like window is as tall as what it holds, capped by the maximum",
            |area, buf| {
                proposed(|| {
                    Frame {
                        fields: short_record(),
                        reference: Reference::None,
                        title: "Configuration",
                        crumbs: &["Configuration"],
                        footprint: Footprint::Content { cols: 62, rows: 14 },
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "04 size — drill-down at max, both bars (B)",
            "Item 1's second half: a drill-down opens at the maximum, and a table that cannot shrink gets a horizontal bar as well as a vertical one",
            |area, buf| {
                proposed(|| {
                    wide_table(area, buf);
                });
            },
        )),
        // ---- item 2 + 4, the tint and what it costs the text ----------------------------
        Box::new(Variant(
            "06 tint — blue 0.40, lightness held",
            "PROPOSED. The same hue at the same strength, with L* left where the rung put it: 13 of 15 themes clear the floor, and the two that fail fail untinted too",
            |area, buf| {
                proposed(|| {
                    Frame::default().draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "07 tint — held, on Solarized Dark",
            "The adversarial theme: the thinnest ladder of the fifteen, and the one where round 1's SET fill fell below a just-noticeable difference",
            |area, buf| {
                r1::with_theme(r1::ADVERSARIAL_THEME, || {
                    proposed(|| {
                        Frame::default().draw(area, buf);
                    });
                });
            },
        )),
        Box::new(Variant(
            "08 tint — held, on Catppuccin Latte (light)",
            "A light theme, where the ladder runs the other way — the case a design measured only on Mocha cannot see",
            |area, buf| {
                r1::with_theme(ratatui_themes::ThemeName::CatppuccinLatte, || {
                    proposed(|| {
                        Frame::default().draw(area, buf);
                    });
                });
            },
        )),
        // ---- item 3, the fields ---------------------------------------------------------
        Box::new(Variant(
            "09 fields — view mode, black on the cursor row",
            "The selected field takes the table's own block and its text goes black bold, which ruling 7 already gave it",
            |area, buf| {
                proposed(|| {
                    Frame {
                        mode: Mode::View { at: 6 },
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "11 fields — edit mode, fill alone",
            "PROPOSED. The underline gone, the SET carried by the fill, and the ACTIVE field derived light enough to carry black text",
            |area, buf| {
                proposed(|| {
                    Frame {
                        mode: Mode::Edit {
                            at: 6,
                            edit: Some(Edit::insert("256")),
                        },
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "10 fields — tick box, the two states side by side",
            "Space toggles it, and the mark is a SHAPE — `[\u{2713}] yes` against `[ ] no` — so the state survives a terminal that refuses colour (r06 #8)",
            |area, buf| {
                proposed(|| {
                    Frame {
                        // Field 7 is ticked and field 8 is not, so one frame carries both
                        // states: a tick box drawn alone says what it looks like, and a pair
                        // says what tells them apart.
                        mode: Mode::Edit { at: 7, edit: None },
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "12 fields — radio on one row (A)",
            "The active button BOLD and the rest normal; h/l or left/right move within the field",
            |area, buf| {
                proposed(|| {
                    Frame {
                        mode: Mode::Edit { at: 4, edit: None },
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "13 fields — the same radio as a column (B)",
            "The whole column takes the highlight, one choice per row, and j/k move within it",
            |area, buf| {
                proposed(|| {
                    Frame {
                        fields: column_record(),
                        mode: Mode::Edit { at: 1, edit: None },
                        title: "Queue item \u{2014} operation",
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "14 fields — drop-down open, no frame",
            "Item 3: the frame gone, every other field's highlight gone, the current value first and the rest sorted under it",
            |area, buf| {
                proposed(|| choice_open(area, buf, None));
            },
        )),
        Box::new(Variant(
            "15 fields — drop-down, fuzzy filter typed",
            "`ts` typed: the list narrows by subsequence, and the cursor is on the best match rather than on a value that may no longer be in the list",
            |area, buf| {
                proposed(|| choice_open(area, buf, Some("ts")));
            },
        )),
        Box::new(Variant(
            "16 fields — single line, vim INSERT (bar, blinking)",
            "The caret is a bar carrying the real SLOW_BLINK attribute. Judge it from a cell dump — a PNG cannot show an attribute and should not be asked to",
            |area, buf| {
                proposed(|| {
                    let (fields, mode) = text_record(Edit::insert("reading guide"));
                    Frame {
                        fields,
                        mode,
                        reference: Reference::None,
                        title: "Queue item \u{2014} label",
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "17 fields — single line, vim NORMAL (block, steady)",
            "A block ON a character, and the ONE caret that does not blink — which is the only thing separating it from visual's",
            |area, buf| {
                proposed(|| {
                    let (fields, mode) = text_record(Edit::normal("reading guide", 8));
                    Frame {
                        fields,
                        mode,
                        reference: Reference::None,
                        title: "Queue item \u{2014} label",
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "18 fields — single line, conventional (terminal caret)",
            "EMACS style: the terminal owns the cursor, so nothing is painted. A still frame has no terminal, which is why this one shows no caret at all",
            |area, buf| {
                proposed(|| {
                    let (fields, mode) = text_record(Edit::insert("reading guide"));
                    Frame {
                        fields,
                        mode,
                        reference: Reference::None,
                        title: "Queue item \u{2014} label (conventional keys)",
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "19 fields — multi-line being edited",
            "The fill IS the box: a multi-line field's frame is the extent of its background, so no border is drawn around it",
            |area, buf| {
                proposed(|| {
                    let (fields, _) = text_record(Edit::insert("reading guide"));
                    Frame {
                        fields,
                        mode: Mode::Edit {
                            at: 2,
                            edit: Some(Edit::insert_at(
                                "Held back once already: the grammar download timed out against \
                                 the registry, and the retry is queued behind the scan.",
                                34,
                            )),
                        },
                        reference: Reference::None,
                        title: "Queue item \u{2014} note",
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "20 fields — selected text (A) neutral inversion",
            "PROPOSED. No hue spent: the selected-row tint at a second scale, with whichever text end is legible on it",
            |area, buf| {
                proposed(|| selection(area, buf, field::SelectedText::Inverted));
            },
        )),
        Box::new(Variant(
            "21 fields — selected text (B) theme secondary",
            "A real colour, as asked — and on every non-Catppuccin theme it IS the data cursor's hue, so the selection and the row cursor share one",
            |area, buf| {
                proposed(|| selection(area, buf, field::SelectedText::Secondary));
            },
        )),
        // ---- item 4, the third column and the headers -----------------------------------
        Box::new(Variant(
            "22 fields — the same edit, driven by real keystrokes",
            "Not constructed in the state a keystroke would produce — `jjjjjj e A 0` through Stack::key, so the frame is evidence the live editor is wired in and not only that the renderer works",
            |area, buf| {
                proposed(|| {
                    driven(area, buf, &typed_into());
                });
            },
        )),
        Box::new(Variant(
            "23 readability — third column, no band",
            "PROPOSED. Item 4: the column is informational and does not need its own region — the text stands on the window like everything else",
            |area, buf| {
                proposed(|| {
                    Frame::default().draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "24 readability — no headers at all",
            "Item 4: *the headers are optional*. With none, the row the header occupied goes back to data",
            |area, buf| {
                proposed(|| {
                    Frame {
                        reference: Reference::None,
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        // ---- items 5 to 8 ---------------------------------------------------------------
        Box::new(Variant(
            "25 breadcrumb — depth 1",
            "A trail of one is all current: no ancestor run, and the segment opens straight off the window",
            |area, buf| {
                proposed(|| {
                    Frame {
                        crumbs: &["Configuration"],
                        title: "Configuration",
                        fields: short_record(),
                        reference: Reference::None,
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "26 breadcrumb — depth 2",
            "One ancestor and the current crumb, with the transition between them",
            |area, buf| {
                proposed(|| {
                    Frame {
                        crumbs: &["Queue", "open-books"],
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "27 breadcrumb — depth 3",
            "PROPOSED. Two ancestors sharing one run of full-saturation accent, the separator drawn INSIDE it, then the current crumb on a lighter run",
            |area, buf| {
                proposed(|| {
                    Frame::default().draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "28 breadcrumb — depth 3, plain (nerd-font fallback)",
            "What it replaces: *not visible enough* — muted text and a faint chevron, spending no background at all",
            |area, buf| {
                proposed(|| {
                    let rect = Footprint::Max.window(area);
                    r1::page().render(area, buf);
                    let Some(rect) = rect else { return };
                    let deco = Decoration::new("Queue item \u{2014} reading_guide.py")
                        .crumbs(r1::CRUMBS.to_vec())
                        .crumb_style(CrumbStyle::Plain)
                        .hint("\u{2193}\u{2191}/jk", "Move")
                        .hint("e", "Edit")
                        .hint("q", "Close");
                    let container = Container::new(deco);
                    let viewport = container.viewport(rect);
                    container.render(rect, buf);
                    crate::views::modal_framework::record::RecordView::new(
                        r1::record(),
                        Mode::View { at: 5 },
                    )
                    .reference(Reference::Text("DEFAULT"))
                    .render(viewport, buf);
                });
            },
        )),
        Box::new(Variant(
            "29 banner — view vs edit, no `-- EDIT --`",
            "Item (b): the columnar change IS the indication, so the banner is gone. This is edit mode, and nothing on the title row says so",
            |area, buf| {
                proposed(|| {
                    Frame {
                        mode: Mode::Edit {
                            at: 6,
                            edit: Some(Edit::insert("256")),
                        },
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "30 confirm — a window quietened under its own guard",
            "Item (c): the window beneath gets the SAME treatment the page gets — the existing ModalScope, one level deeper, rather than a second rule meaning the same thing",
            |area, buf| {
                proposed(|| {
                    confirm(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "32 edge — spacing only",
            "PROPOSED. Item (e): the same rect, fill and padding with no glyphs — the content does not move by a column, so the pair differs in ink alone",
            |area, buf| {
                proposed(|| {
                    Frame {
                        edge: Edge::Spacing,
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "33 degradation — the SET mark under NO_COLOR",
            "The ladder collapses onto four slots, so the fill cannot carry the SET — and the underline comes back, only here",
            |area, buf| {
                let restore = crate::encoding::Encoding::current();
                crate::encoding::Encoding::set(crate::encoding::Encoding::NoColor);
                proposed(|| {
                    Frame {
                        mode: Mode::Edit {
                            at: 6,
                            edit: Some(Edit::insert("256")),
                        },
                        ..Frame::default()
                    }
                    .draw(area, buf);
                });
                crate::encoding::Encoding::set(restore);
            },
        )),
        Box::new(Variant(
            "34 degradation — the same frame at ansi16",
            "Sixteen slots: the trail keeps its glyphs and loses its hue separation, which is what item (a) costs on a poor terminal",
            |area, buf| {
                let restore = crate::encoding::Encoding::current();
                crate::encoding::Encoding::set(crate::encoding::Encoding::Ansi16);
                proposed(|| {
                    Frame::default().draw(area, buf);
                });
                crate::encoding::Encoding::set(restore);
            },
        )),
    ]
}

/// Keeps the unused-import warning honest about what this module actually reaches for.
#[allow(dead_code)]
fn _touch(_: TintBlend, _: Value, _: f32) {
    let _ = RULED_STRENGTH;
    let _ = tokens::tint_strength();
}

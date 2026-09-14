//! The `Modal Framework` group — round 1's frames and its A/B pairs, browsable in the pantry.
//!
//! Every variant is a whole screen, so they declare `tab() = "Views"`: a window over a page is
//! a full-page composition, not an atomic element (§16). The names are chosen so a pair reads
//! as a pair in the list — `… (A)` beside `… (B)` — because the list is how Chris finds them.
//!
//! # Every frame but the brackets is drawn under the PROPOSAL
//!
//! [`proposed`] puts the gate's tint and strength in force. A frame drawn under some other
//! default would be a frame of a design nobody is proposing, and the two choices interact: an
//! accent-washed form inside an accent-tinted window is a merge, and that merge is only
//! visible when both are on. The `Tint (…)` variants set their own, which is the entire point
//! of a bracket.

use super::frames::{self, RecordFrame, PROPOSED_WASH};
use super::record::{Mode, Reference, Scheme};
use crate::tokens::{self, ModalTint};
use crate::widgets::edit_field::Edit;
use ratatui::{buffer::Buffer, layout::Rect};
use tui_pantry::{Ingredient, PropInfo};

const PROPS: &[PropInfo] = &[
    PropInfo {
        name: "Container",
        ty: "widgets::modal_frame::Container",
        description: "The fixed window: one framework footprint, a viewport, a scrollbar",
    },
    PropInfo {
        name: "Decoration",
        ty: "widgets::modal_frame::Decoration",
        description:
            "Breadcrumb / blank / title / blank (+ optional search row); blank + 2 help rows",
    },
    PropInfo {
        name: "View",
        ty: "modal_framework::{record::RecordView, table::TableView}",
        description: "The two view kinds a window can hold — a record, or a table",
    },
    PropInfo {
        name: "Stack",
        ty: "modal_framework::Stack",
        description: "The drill-down: push on Enter, pop on Backspace, a guard over a dirty pop",
    },
    PropInfo {
        name: "tint + wash",
        ty: "tokens::ModalTint + tokens::tint_strength",
        description: "Proposed: blue (accent) at 0.28. The `Tint (…)` variants bracket both",
    },
];

/// The proposal, in force. See the module docs for why it is not opt-in per frame.
fn proposed<T>(draw: impl FnOnce() -> T) -> T {
    frames::with_tint(ModalTint::Accent, PROPOSED_WASH, draw)
}

/// The active field in every EDIT-mode frame: `Chunk overlap`, mid-edit.
fn editing() -> Mode {
    Mode::Edit {
        at: 6,
        edit: Some(Edit::insert("256")),
    }
}

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
        "wqm_tui::views::modal_framework"
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

pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
    vec![
        // ----- the frames the round is judged from -------------------------------------
        Box::new(Variant(
            "Record, view mode",
            "The selected field takes the table's own cursor block, and the block stops at the value so the reference band survives",
            |area, buf| {
                proposed(|| {
                    RecordFrame::default().draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Record, edit mode",
            "Every editable field ruled and filled, the active one a step above it, and the mode said in words on the title row",
            |area, buf| {
                proposed(|| {
                    RecordFrame::view(editing()).draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Record, scrolled",
            "The same record part-way down: the thumb is off both ends and the DEFAULT header has not scrolled away with the data",
            |area, buf| {
                proposed(|| {
                    RecordFrame {
                        offset: 5,
                        ..RecordFrame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Record, empty",
            "No fields at all: the word the table already uses, and no header over a column with nothing in it",
            |area, buf| {
                proposed(|| {
                    RecordFrame {
                        empty: true,
                        ..RecordFrame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Table in a modal",
            "The Queue's own ListPane, pre-filtered to the library it was drilled in from, with the fifth row naming that floor",
            |area, buf| {
                proposed(|| {
                    frames::table_frame(area, buf, false);
                });
            },
        )),
        Box::new(Variant(
            "Table, empty",
            "A floor that matches nothing: the column titles stay, because that row IS the view",
            |area, buf| {
                proposed(|| {
                    frames::empty_table_frame(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Breadcrumb at depth 3",
            "Libraries > open-books > Queue — the path the reader took, not a hierarchy: the same view reached another way shows another trail",
            |area, buf| {
                proposed(|| {
                    frames::table_frame(area, buf, false);
                });
            },
        )),
        Box::new(Variant(
            "Slide t=0",
            "Before the push: the record, whole. Only the viewport moves — the decoration is the window's and is already at its destination",
            |area, buf| {
                proposed(|| frames::slide_frame(area, buf, 0.0));
            },
        )),
        Box::new(Variant(
            "Slide t=0.5",
            "Half-way: the record leaving left, the queue arriving from the right. The only frame that says whether the motion will read",
            |area, buf| {
                proposed(|| frames::slide_frame(area, buf, 0.5));
            },
        )),
        Box::new(Variant(
            "Slide t=1",
            "After the push: the queue, whole, under the trail it arrived with",
            |area, buf| {
                proposed(|| frames::slide_frame(area, buf, 1.0));
            },
        )),
        Box::new(Variant(
            "Confirm over the window",
            "Backspace over a dirty view: the existing modal on Layer2, centred on the WINDOW because a dialogue belongs to what it asks about",
            |area, buf| {
                proposed(|| {
                    frames::confirm_frame(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Drop-down open",
            "Enter on the choice field: as wide as the cell it came out of, cursor on the current value, on the layer above the window",
            |area, buf| {
                proposed(|| {
                    frames::dropdown_frame(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Contextual help (?)",
            "The help window IS this framework: a container and a fixed scrollable content — and the content does not fit, which is the scroll the ruling asks for",
            |area, buf| {
                proposed(|| {
                    frames::help_frame(area, buf);
                });
            },
        )),
        // ----- the A/B pairs the gate keeps for Chris's look ---------------------------
        Box::new(Variant(
            "Tint: neutral",
            "Today's default: no blend at all. The window and the page are told apart by the border alone",
            |area, buf| {
                frames::with_tint(ModalTint::Neutral, PROPOSED_WASH, || {
                    RecordFrame::default().draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Tint: blue 0.14",
            "The screen wash's own strength, on one window. Read back from a pixel render it is very nearly not there",
            |area, buf| {
                frames::with_tint(ModalTint::Accent, tokens::WASH_MIX, || {
                    RecordFrame::default().draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Tint: blue 0.28",
            "PROPOSED. A distinctly cool slate surface against the page, still quiet — twice the screen wash, which covers the whole screen and cannot spend as much",
            |area, buf| {
                frames::with_tint(ModalTint::Accent, PROPOSED_WASH, || {
                    RecordFrame::default().draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Tint: blue 0.40",
            "The far end, so the range is bracketed rather than guessed: the faint rungs start losing the ground under them",
            |area, buf| {
                frames::with_tint(ModalTint::Accent, 0.40, || {
                    RecordFrame::default().draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Tint: lavender (rejected)",
            "The cursor's own hue, on the window that draws the cursor. Rejected on identity, not distance: it IS that hue on all fifteen themes",
            |area, buf| {
                frames::with_tint(ModalTint::Selected, PROPOSED_WASH, || {
                    RecordFrame::default().draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Fallback: neutral window, accent-washed form",
            "If blue's worst case (dE 11.0 vs lavender on Mocha) is too thin to defend, this inverts where the colour goes and keeps every role intact",
            |area, buf| {
                frames::with_tint(ModalTint::Neutral, PROPOSED_WASH, || {
                    RecordFrame {
                        mode: editing(),
                        scheme: Scheme::AccentWash,
                        ..RecordFrame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Edit fields: C (selection-derived, rejected)",
            "The set is the selection tint and the point is the cursor block — so EDIT mode comes out looking exactly like VIEW mode",
            |area, buf| {
                proposed(|| {
                    RecordFrame {
                        mode: editing(),
                        scheme: Scheme::SelectionDerived,
                        ..RecordFrame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Third column: A (text only)",
            "The reference values dimmer on the window's own surface — same colour, quieter, which is emphasis rather than another colour",
            |area, buf| {
                proposed(|| {
                    RecordFrame {
                        reference: Reference::Text("DEFAULT"),
                        ..RecordFrame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Third column: B (its own band)",
            "PROPOSED. 'Another color than the main window' read as a SURFACE: one quiet region, no hue spent, and it comes off in edit mode",
            |area, buf| {
                proposed(|| {
                    RecordFrame::default().draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Third column: none",
            "A two-column record, which therefore has no header row at all and gives that row back to data",
            |area, buf| {
                proposed(|| {
                    RecordFrame {
                        reference: Reference::None,
                        ..RecordFrame::default()
                    }
                    .draw(area, buf);
                });
            },
        )),
        Box::new(Variant(
            "Table: pinned column shown",
            "Every row repeating `open-books` in its widest fixed column — twenty columns carrying no information",
            |area, buf| {
                proposed(|| {
                    frames::table_frame(area, buf, true);
                });
            },
        )),
        Box::new(Variant(
            "Table: pinned column dropped",
            "PROPOSED, and it generalises: a drilled-in view drops the column it was pinned by, and Object gets the width",
            |area, buf| {
                proposed(|| {
                    frames::table_frame(area, buf, false);
                });
            },
        )),
    ]
}

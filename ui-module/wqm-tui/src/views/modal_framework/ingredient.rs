//! The states of a window that have **no clause of their own** in Chris's 19:05 message.
//!
//! These land in the same `Modal Framework` group as
//! [`super::shipping::ingredient`], and the split is by what a frame is FOR rather than by which
//! round drew it: that module has one frame per clause he ruled on, and this one has the states
//! a window enters that no clause is about — a record part-way down its scroll, an empty record,
//! a table reached by drilling in, the three stills of the push animation, the discard guard, the
//! contextual help, and the pinned column with and without its drop.
//!
//! Every variant is a whole screen, so they declare `tab() = "Views"`: a window over a page is
//! a full-page composition, not an atomic element (§16).
//!
//! # The A/B pairs are gone, and one is left on purpose
//!
//! Round 1 put fourteen arms here for Chris to choose between — two footprints, five tints, three
//! field schemes, three third columns, a cursor extent. He ruled on all of them, so they are not
//! options any more and a pantry entry for one would be an entry for a thing that cannot happen.
//! The pinned-column pair stays because his ruling made it *"the CALLER's decision"*: both values
//! are legitimate, and which one a window wants is a composition's choice rather than a design's.
//!
//! Every frame is drawn under [`proposed`], the ruled tint and strength. A frame drawn under some
//! other default would be a frame of a design nobody is proposing.

use super::frames::{self, RecordFrame, PROPOSED_WASH};
use crate::tokens::ModalTint;
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
            "The caller's other choice, and the usual one: a drilled-in view drops the column it was pinned by, and Object gets the width",
            |area, buf| {
                proposed(|| {
                    frames::table_frame(area, buf, false);
                });
            },
        )),
    ]
}

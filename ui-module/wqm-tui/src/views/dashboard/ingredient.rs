//! The pantry variants for [`super`].
//!
//! Five frames, and three of them exist to be uncomfortable. `Populated` is the captured
//! workspace and answers "does this look like v0.1"; `Empty workspace` answers "does a fresh
//! install read as empty or as broken"; `Small 80x24` answers the question a design instrument
//! is actually for — **what breaks first**.

use super::*;
use crate::panes::status_block::{Queue, ENTRY_LABELS};
use crate::widgets::chrome::Freshness;
use std::time::Duration;
use tui_pantry::{Ingredient, PropInfo};

const PROPS: &[PropInfo] = &[
    PropInfo {
        name: "cells",
        ty: "Vec<CellPane>",
        description: "Six projections, row-major. The view places them; each scrolls itself",
    },
    PropInfo {
        name: "status",
        ty: "StatusBlock",
        description: "The constant top's block — the same one every other tab carries",
    },
    PropInfo {
        name: "attention",
        ty: "Attention",
        description: "Which cell is live — screen-level, so two cannot be",
    },
];

/// Not a decision — §7 leaves the freshness SLA open (OQ-6); the number the other views' frames
/// are drawn against, so every screen ages at the same rate.
const FRAME_SLA: Duration = Duration::from_secs(60);

/// Public to the crate so the Queue tab's frames carry the SAME captured workspace at the top of
/// the screen. Two tabs whose status blocks disagreed would be two frames of two machines.
pub(crate) fn block(entries: [Health; ENTRY_LABELS.len()], queue: Queue) -> (StatusBlock, Health) {
    let overall = overall(entries[0], &entries[1..]);
    (
        StatusBlock::new(
            overall,
            "v0.2.0",
            Freshness::new(Duration::from_secs(4), FRAME_SLA),
            entries,
            queue,
        ),
        overall,
    )
}

/// The capture's own queue: 11'236 waiting, 4 moving, 3 lost. See [`block`] for why it is shared.
pub(crate) fn captured_queue() -> Queue {
    Queue {
        pending: 11_236,
        in_progress: 4,
        failed: 3,
        health: Health::Degraded,
    }
}

/// The four health entries the captured workspace had — one degraded, three well. Shared for the
/// same reason [`block`] is.
pub(crate) const CAPTURED_ENTRIES: [Health; ENTRY_LABELS.len()] = [
    Health::Healthy,
    Health::Degraded,
    Health::Healthy,
    Health::Healthy,
];

fn idle() -> Queue {
    Queue {
        pending: 0,
        in_progress: 0,
        failed: 0,
        health: Health::Healthy,
    }
}

fn dashboard(cells: Vec<CellPane>, queue: Queue, entries: [Health; 4]) -> Dashboard {
    let (status, overall) = block(entries, queue);
    Dashboard::new(cells, status, overall)
}

fn populated() -> Dashboard {
    dashboard(
        frames::populated(),
        captured_queue(),
        [
            Health::Healthy,
            Health::Degraded,
            Health::Healthy,
            Health::Healthy,
        ],
    )
}

struct Variant(&'static str, &'static str, fn() -> Dashboard, Option<(u16, u16)>);

impl Ingredient for Variant {
    fn tab(&self) -> &str {
        "Views"
    }
    fn group(&self) -> &str {
        "Dashboard"
    }
    fn name(&self) -> &str {
        self.0
    }
    fn source(&self) -> &str {
        "wqm_tui::views::dashboard"
    }
    fn description(&self) -> &str {
        self.1
    }
    fn props(&self) -> &[PropInfo] {
        PROPS
    }
    fn render(&self, area: Rect, buf: &mut Buffer) {
        let (width, height) = self.3.unwrap_or((area.width, area.height));
        (self.2)().render(
            Rect {
                width: width.min(area.width),
                height: height.min(area.height),
                ..area
            },
            buf,
        );
    }
}

pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
    vec![
        Box::new(Variant(
            "Populated",
            "The captured workspace: 29 projects into a six-row cell, so the overflow tail is what you see",
            populated,
            None,
        )),
        Box::new(Variant(
            "Empty workspace",
            "A fresh install — six `No data` cells. Does it read as empty, or as failed to load?",
            || {
                dashboard(
                    frames::empty(),
                    idle(),
                    [Health::Healthy; ENTRY_LABELS.len()],
                )
            },
            None,
        )),
        // The three cases the foot has, one frame each — a cell with many rows, one with a
        // single row, and one with none. They are browsable together because "the foot follows
        // the focused cell" is a rule about the SET, and one frame cannot show a set.
        Box::new(Variant(
            "Focus on Rules",
            "Eight rows: the heading takes the block, the first row takes the cursor, the column keys light up, and the foot offers Navigate and Enter",
            || populated().attention(Attention::Zone(3)),
            None,
        )),
        Box::new(Variant(
            "Focus on Projects, sorted by Files ↓",
            "The sort ruling in one frame: `f` lit in the selector hue, `↓` after the name, and the seven projects in Files order",
            || {
                dashboard(
                    frames::sorted_by_files(),
                    captured_queue(),
                    [Health::Healthy, Health::Degraded, Health::Healthy, Health::Healthy],
                )
                .attention(Attention::Zone(0))
            },
            None,
        )),
        Box::new(Variant(
            "Focus on Libraries",
            "One row: nothing to navigate and nothing to sort, so no column key is lit and the foot offers Enter alone",
            || populated().attention(Attention::Zone(1)),
            None,
        )),
        Box::new(Variant(
            "Focus on Scratchpad",
            "No rows: the heading still takes the block, nothing is highlighted below it, and the foot is unchanged",
            || populated().attention(Attention::Zone(2)),
            None,
        )),
        Box::new(Variant(
            "Under modal",
            "The page beneath a modal: digits and cell keys muted, nothing else moves",
            || populated().under_modal(true),
            None,
        )),
        Box::new(Variant(
            "Small 80x24",
            "Eighty by twenty-four: the block collapses, the tab row runs off, and the cells lose rows before they lose columns",
            populated,
            Some((80, 24)),
        )),
        Box::new(Variant(
            "Wide 200x50",
            "Room for everything: the grid's cells grow but the status block's own columns stop at their cap",
            populated,
            Some((200, 50)),
        )),
    ]
}

//! The pantry variants for [`super`].
//!
//! Eight frames, and the two collapse ones are the reason there are eight: collapse has two
//! independent triggers and a frame that only ever showed the short shape would not say which
//! of them produced it. Each is rendered at a stated size rather than at whatever the preview
//! cell offers, because a width-triggered collapse judged in a cell of unknown width is not a
//! measurement of anything.

use super::*;
use std::time::Duration;
use tui_pantry::{Ingredient, PropInfo};

const PROPS: &[PropInfo] = &[
    PropInfo {
        name: "overall",
        ty: "Health",
        description: "The roll-up. `rollup()` derives it; the pane takes it so a view can too",
    },
    PropInfo {
        name: "entries",
        ty: "[Health; 4]",
        description: "One per ENTRY_LABELS slot, positionally — the order is not a convention",
    },
    PropInfo {
        name: "queue",
        ty: "Queue",
        description: "pending / in progress / failed, plus the queue's own state",
    },
    PropInfo {
        name: "freshness",
        ty: "Freshness",
        description: "Age and SLA together; the word STALE is the comparison, never a flag",
    },
];

/// The SLA the frames are drawn against. Not a decision — §7 leaves it open (OQ-6), the same
/// number `views::service::frames` states for the same reason.
const FRAME_SLA: Duration = Duration::from_secs(60);

fn fresh() -> Freshness {
    Freshness::new(Duration::from_secs(4), FRAME_SLA)
}

fn idle() -> Queue {
    Queue {
        pending: 0,
        in_progress: 0,
        failed: 0,
        health: Health::Healthy,
    }
}

/// The block with every part nominal — what the other frames are read against.
fn nominal() -> StatusBlock {
    StatusBlock::new(
        Health::Healthy,
        "v0.2.0",
        fresh(),
        [Health::Healthy; ENTRY_LABELS.len()],
        idle(),
    )
}

/// One entry in a stated state, with the roll-up DERIVED from it rather than asserted beside
/// it — a frame showing a green roll-up over a red entry is not one this helper can build.
fn with_entry(index: usize, health: Health) -> StatusBlock {
    let mut entries = [Health::Healthy; ENTRY_LABELS.len()];
    entries[index] = health;
    StatusBlock::new(
        rollup(entries[0], &entries[1..]),
        "v0.2.0",
        fresh(),
        entries,
        idle(),
    )
}

/// A variant, and the size it is judged at. `None` takes the preview cell's own width.
struct Variant(&'static str, &'static str, fn() -> StatusBlock, Option<(u16, u16)>);

impl Ingredient for Variant {
    fn tab(&self) -> &str {
        "Panes"
    }
    fn group(&self) -> &str {
        "Status Block"
    }
    fn name(&self) -> &str {
        self.0
    }
    fn source(&self) -> &str {
        "wqm_tui::panes::status_block"
    }
    fn description(&self) -> &str {
        self.1
    }
    fn props(&self) -> &[PropInfo] {
        PROPS
    }
    fn render(&self, area: Rect, buf: &mut Buffer) {
        let (width, height) = self.3.unwrap_or((area.width, ROWS_FULL));
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
            "Healthy",
            "Nothing to report: one green glyph per part, and a title that stays out of the way",
            nominal,
            None,
        )),
        Box::new(Variant(
            "Degraded (vector db)",
            "One part past its SLA — the title takes the state hue because now there IS something to see",
            || with_entry(1, Health::Degraded),
            None,
        )),
        Box::new(Variant(
            "Daemon offline",
            "The interim rule: the master is down, so the SERVICE is degraded rather than offline",
            || with_entry(0, Health::Offline),
            None,
        )),
        Box::new(Variant(
            "Queue backlog",
            "Three counts, three hues — warning waiting, info moving, error lost",
            || {
                nominal().overall(Health::Degraded).queue(Queue {
                    pending: 1_240,
                    in_progress: 8,
                    failed: 3,
                    health: Health::Degraded,
                })
            },
            None,
        )),
        Box::new(Variant(
            "All zero queue",
            "Every count muted: an idle queue must read as quiet, not as three grey zeros shouting",
            nominal,
            None,
        )),
        Box::new(Variant(
            "Stale readings",
            "Past the SLA: the right-flushed age turns the degraded hue while the parts stay green",
            || {
                StatusBlock::new(
                    Health::Healthy,
                    "v0.2.0",
                    Freshness::new(Duration::from_secs(1_100), FRAME_SLA),
                    [Health::Healthy; ENTRY_LABELS.len()],
                    idle(),
                )
            },
            None,
        )),
        Box::new(Variant(
            "Wide 200",
            "Two hundred columns: the grid stops at MAX_COLUMN and packs left, and the age still follows the screen's own right edge",
            || {
                nominal().queue(Queue {
                    pending: 1_240,
                    in_progress: 8,
                    failed: 3,
                    health: Health::Healthy,
                })
            },
            Some((200, ROWS_FULL)),
        )),
        Box::new(Variant(
            "Collapsed (height)",
            "Two rows offered: the roll-up and its rule, nothing else — collapse is on or off",
            nominal,
            Some((125, ROWS_COLLAPSED)),
        )),
        Box::new(Variant(
            "Collapsed (width)",
            "Full height, but four columns can no longer align — the same short shape, other trigger",
            nominal,
            Some((MARGIN * 2 + MIN_COLUMN * ENTRY_LABELS.len() as u16 - 1, ROWS_FULL)),
        )),
    ]
}

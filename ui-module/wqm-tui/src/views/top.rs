//! The constant top, as one composition both screens call.
//!
//! Chris, 20260906: the first rows of every tab are the same rows. That claim only stays true
//! if there is one thing drawing them. [`crate::views::shell`] had the composition inline, and
//! the moment a second view needed it the choice was to copy four calls or to lift them — and
//! a copy is how "the same on every tab" quietly becomes "the same on the tabs someone
//! remembered to update".
//!
//! # It returns the leftover rather than taking a callback
//!
//! [`ConstantTop::draw`] paints and hands back the [`Rect`] beneath it. A view then owns its
//! own body, which is §16's split: the chrome decides where the chrome goes, the view decides
//! everything below it. A callback would have inverted that and made the top the thing that
//! lays out a screen.

use ratatui::{buffer::Buffer, layout::Rect, widgets::Widget};

use crate::panes::status_block::{self, StatusBlock};
use crate::tokens::Condition;
use crate::widgets::chrome::{inset, AppBar, Rule};
use crate::widgets::surface::Surface;

/// The two rows every screen carries above whatever comes next: the app bar and the frame rule
/// under it. Constant because the Service tab has these two and nothing else.
pub const APP_BAR_ROW: u16 = 0;
pub const TOP_RULE_ROW: u16 = 1;
pub const CONSTANT_ROWS: u16 = 2;

/// One row of a screen, by index.
pub fn row(area: Rect, n: u16) -> Rect {
    Rect {
        y: area.y + n,
        height: 1,
        ..area
    }
}

/// Rows a key-hint foot costs a view's body: the rule that closes the content, and the hint
/// line itself.
///
/// A view's own minimum-height arithmetic adds this rather than adding 1, so the day a foot
/// grows a third row there is one number to change instead of one per screen.
pub const FOOT_ROWS: u16 = 2;

/// Carve the key-hint foot off the bottom of `body`, DRAW the rule that closes the content
/// above it, and hand back what is left for the view plus the row the hint line goes on.
///
/// Chris, 2026-09-07: *"we should have a line just above the bottom line, valid for all views
/// as well."* Every screen that ends in a key-hint line ends the same way, so the rule is not
/// a thing a view remembers to draw — it is drawn by the call that tells the view how much room
/// it has. A fourth screen asking for its foot gets the rule whether or not anybody thought
/// about it, and a screen that skipped this call would have no foot row to render into.
///
/// The rule runs edge to edge, [`crate::widgets::chrome::MARGIN`] included, exactly as the
/// screen's other full-width rules do: it divides the screen rather than the content inside it.
/// Only the Dashboard's row rules break over the column gap, and they break because the GRID
/// has two columns — this one has nothing to break over.
///
/// Returns the whole of `body` and an empty foot when `body` is too short to hold both rows,
/// so a caller that forgets to check still draws no rule rather than drawing one over its own
/// last line.
pub fn foot(body: Rect, buf: &mut Buffer) -> (Rect, Rect) {
    if body.height < FOOT_ROWS {
        return (body, Rect { height: 0, ..body });
    }
    let content = Rect {
        height: body.height - FOOT_ROWS,
        ..body
    };
    let rule = row(body, body.height - FOOT_ROWS);
    foot_rule(rule, buf);
    (content, row(body, body.height - 1))
}

/// Draw the rule that closes a screen's content, directly above its key-hint line.
///
/// Public and separate from [`foot`] for the one screen that lays out all of its rows at once
/// ([`crate::views::service`]): its foot row is carved by its own [`ratatui::layout::Layout`]
/// alongside eight others, so it cannot take the rect back from `foot` — but it can and does
/// take the RULE from here, which is the half that has to be identical everywhere.
pub fn foot_rule(at: Rect, buf: &mut Buffer) {
    Rule::internal().render(at, buf);
}

/// App bar, frame rule, and the status block a tab carries — or does not.
pub struct ConstantTop {
    active: usize,
    /// Absent on the Service tab, which carries [`crate::panes::status_band`] instead.
    status: Option<StatusBlock>,
    content_floor: u16,
}

impl ConstantTop {
    /// A tab with no status block. [`ConstantTop::status`] adds one.
    pub fn new(active: usize) -> Self {
        Self {
            active,
            status: None,
            content_floor: status_block::MIN_CONTENT_ROWS,
        }
    }

    pub fn status(mut self, block: StatusBlock) -> Self {
        self.status = Some(block);
        self
    }

    /// Rows the view's own body must keep before the status block gives way. The **view's**
    /// number: a tab holding a six-cell grid can afford less top furniture than one holding a
    /// one-line summary.
    pub fn content_floor(mut self, rows: u16) -> Self {
        self.content_floor = rows;
        self
    }

    /// Paint the screen's ground and its top, and hand back what is left below.
    ///
    /// Returns an empty [`Rect`] when the area cannot hold the top at all, so a caller that
    /// forgets to check still draws nothing rather than drawing into negative space.
    pub fn draw(self, area: Rect, buf: &mut Buffer) -> Rect {
        if area.height < CONSTANT_ROWS + 1 {
            return Rect {
                height: 0,
                ..area
            };
        }
        // §15: the theme owns the background, and every full screen paints it first. Stated
        // rather than read from the process, because a frame is a still.
        Surface::with_condition(Condition::Nominal).render(area, buf);

        AppBar::new(self.active).render(inset(row(area, APP_BAR_ROW)), buf);
        Rule::frame().render(row(area, TOP_RULE_ROW), buf);

        // What the block and the body share. The block answers how much of it it takes;
        // whatever is left is the body's, and the block never takes the last of it.
        let body = Rect {
            y: area.y + CONSTANT_ROWS,
            height: area.height - CONSTANT_ROWS,
            ..area
        };

        let taken = match self.status {
            Some(block) => {
                let rows = StatusBlock::rows_for(body, self.content_floor);
                block.render(Rect { height: rows, ..body }, buf);
                rows
            }
            None => 0,
        };

        Rect {
            y: body.y + taken,
            height: body.height - taken,
            ..body
        }
    }
}

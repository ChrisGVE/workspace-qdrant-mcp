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

/// App bar, frame rule, and the status block a tab carries — or does not.
pub struct ConstantTop {
    active: usize,
    modal: bool,
    /// Absent on the Service tab, which carries [`crate::panes::status_band`] instead.
    status: Option<StatusBlock>,
    content_floor: u16,
}

impl ConstantTop {
    /// A tab with no status block. [`ConstantTop::status`] adds one.
    pub fn new(active: usize) -> Self {
        Self {
            active,
            modal: false,
            status: None,
            content_floor: status_block::MIN_CONTENT_ROWS,
        }
    }

    pub fn status(mut self, block: StatusBlock) -> Self {
        self.status = Some(block);
        self
    }

    pub fn under_modal(mut self, modal: bool) -> Self {
        self.modal = modal;
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

        AppBar::new(self.active)
            .under_modal(self.modal)
            .render(inset(row(area, APP_BAR_ROW)), buf);
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

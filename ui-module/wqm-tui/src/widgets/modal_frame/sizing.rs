//! **How big a window is** — the round-2 size model, and the floor below which it says so.
//!
//! Chris, 2026-09-14, item 0, verbatim: *"In my mind it is easier to think in terms of relative
//! number, thus for me the maximum size of a modal window leaves five columns of the background
//! on each side, and five lines of the background on top and bottom. So we define a window that
//! is always relative to the background. I am not sure how small we can accept a modal window to
//! be, but when we pass the point the window becomes too small then we display a message that
//! indicates the screen is too small to display content in the window."*
//!
//! And item 1: *"When a modal window is not part of a drill down … then the window can be sized
//! according to the content … However, when open a window that is part of a drilldown, the
//! window should by default being open at the maximum size."*
//!
//! # The A/B of round 1 is dissolved, not decided
//!
//! [`Footprint::Framework`] and [`Footprint::HelpDerived`] were the two readings of *"the help
//! window's size"*, and item 0 replaces the question rather than answering it: a window is now
//! **relative to the page**, and nothing derives it from the help window's content. Both arms
//! stay reachable so a frame can show what changed, and neither is proposed.
//!
//! # The minimum is derived, not chosen
//!
//! A window's floor is not a number anybody gets to like. It is the sum of what the window has
//! already promised to draw: the border, the decoration Chris specified row by row, and at
//! least one row of content for the decoration to be wrapped around. [`MIN_ROWS`] and
//! [`MIN_COLS`] add those up from the constants that define them, so a change to the decoration
//! moves the floor instead of leaving it stale.

use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Rect},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use super::{BOTTOM_ROWS, CHROME, TOP_ROWS};
use crate::tokens;

#[cfg(test)]
mod tests;

/// Columns of page kept on each side, and rows kept top and bottom, at the maximum size.
///
/// Chris's number, and it is the whole size model: *"five columns of the background on each
/// side, and five lines of the background on top and bottom"*.
pub const INSET: u16 = 5;

/// The border and its one column of padding, top and bottom.
const BORDER_ROWS: u16 = 2;

/// The least content a window can be wrapped around and still be a window.
///
/// One row. A window whose decoration met its help rows with nothing in between is a frame
/// around an absence — and the decoration is not negotiable, because every row of it was
/// specified: breadcrumb, blank, title, blank, then blank and two help rows.
const MIN_CONTENT_ROWS: u16 = 1;

/// The narrowest value column worth drawing, from [`crate::views::modal_framework::record`].
///
/// Below this a value is all ellipsis, which is a frame of a screen nobody can read rather than
/// a small one.
const MIN_CONTENT_COLS: u16 = 30;

/// The shortest window that can draw its decoration and one row of content.
///
/// Summed rather than named: border (2) + top decoration (4) + bottom decoration (3) + one
/// content row = 10.
pub const MIN_ROWS: u16 = BORDER_ROWS + TOP_ROWS + BOTTOM_ROWS + MIN_CONTENT_ROWS;

/// The narrowest window that can draw a record's gutter, label and a readable value.
///
/// Border and padding (4) + the narrowest content a view will accept (30) = 34.
pub const MIN_COLS: u16 = CHROME + MIN_CONTENT_COLS;

/// The smallest PAGE that can hold a window at all — the window's floor plus the inset it is
/// always drawn inside.
///
/// This is the number the too-small message is about: the reader cannot make the window bigger,
/// only the terminal.
pub const MIN_PAGE_COLS: u16 = MIN_COLS + 2 * INSET;
pub const MIN_PAGE_ROWS: u16 = MIN_ROWS + 2 * INSET;

/// Where a window's rectangle comes from.
///
/// # Round 1's A/B pair is gone, and it was dissolved rather than decided
///
/// Chris ruled items 0 and 1 on 2026-09-14 19:05, and the ruling does not pick one of the two
/// arms the round-1 checkpoint offered - it replaces the question they were asking. The maximum
/// is the page inset by [`INSET`] on all four sides; a window that is not part of a drill-down
/// sizes itself to its content under that cap; a drill-down opens at the maximum. Neither
/// `Framework` (`min(96, w-16) x min(24, h-8)`) nor `HelpDerived` (whatever rect the help
/// window wanted) can be produced by that rule, so both are removed rather than left as
/// settings nothing may select.
///
/// The header clamp went with them, and that is the part with a cost. Round 1 pushed every
/// window below the page's own header so a centred top border would not land on the rule under
/// it. The ruling is an inset from the SCREEN, stated without qualification, which at 125x34
/// puts the top border on row 5 - inside the status block. It was ruled AFTER Chris had seen
/// round 1's below-the-header placement, so the cost is the ruling's and taken knowingly.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Footprint {
    /// **Round 2, and what a drill-down opens at**: the page inset by [`INSET`] on all four
    /// sides. Item 0 and the second half of item 1.
    #[default]
    Max,
    /// **Round 2, item 1's first half**: a window that is not part of a drill-down sizes itself
    /// to what it holds, and is capped by [`Footprint::Max`].
    ///
    /// The caller states the size because the caller is the composition and knows its content;
    /// a container that measured its own view would have to be handed the view before it could
    /// be built, which is the cycle `Stack::render` already works around by measuring twice.
    Content { cols: u16, rows: u16 },
}

impl Footprint {
    /// The largest window `area` allows — the page inset by [`INSET`] on all four sides.
    pub fn max_size(area: Rect) -> (u16, u16) {
        (
            area.width.saturating_sub(2 * INSET),
            area.height.saturating_sub(2 * INSET),
        )
    }

    /// The window's rectangle inside `area`, or [`None`] when the page cannot hold one.
    ///
    /// [`None`] is the *screen too small* state and is returned rather than a clamped rect on
    /// purpose: a caller that got a too-small rect back would draw a broken window, and a
    /// caller that gets nothing has to decide what to say instead. [`TooSmall`] is what it
    /// says.
    pub fn window(self, area: Rect) -> Option<Rect> {
        if !fits(area) {
            return None;
        }
        let (max_cols, max_rows) = Self::max_size(area);
        let (width, height) = match self {
            Footprint::Max => (max_cols, max_rows),
            // Capped by the maximum and floored at the minimum, so a view that asks for
            // something absurd in either direction still produces a window that draws.
            Footprint::Content { cols, rows } => (
                cols.clamp(MIN_COLS, max_cols),
                rows.clamp(MIN_ROWS, max_rows),
            ),
        };
        Some(Rect {
            x: area.x + (area.width.saturating_sub(width)) / 2,
            y: area.y + (area.height.saturating_sub(height)) / 2,
            width,
            height,
        })
    }

}

/// Whether `area` can hold a window at all.
pub fn fits(area: Rect) -> bool {
    area.width >= MIN_PAGE_COLS && area.height >= MIN_PAGE_ROWS
}

/// **The screen-too-small message** — item 0's last sentence.
///
/// # It is not a window, and that is the point
///
/// The state it reports is *there is no room for a window*, so drawing it inside one would be a
/// contradiction the reader can see. It is two centred lines on the page, in the same
/// vocabulary as everything else: the fact at [`tokens::strong_style`] and the numbers under it
/// at [`tokens::muted`], with no box — item (e) says the window's own background is its frame,
/// and this has no background to be a frame of.
///
/// # It says what is needed, because the reader can act on it
///
/// *"the screen is too small"* alone leaves someone dragging a corner and guessing. The second
/// line names the size required and the size present, which turns the message into an
/// instruction — Nielsen #9, an error that says how to recover.
pub struct TooSmall {
    available: (u16, u16),
}

impl TooSmall {
    pub fn new(area: Rect) -> Self {
        Self {
            available: (area.width, area.height),
        }
    }

    /// The headline, at the longest form that fits `width`.
    ///
    /// **A message about the screen being too small must itself fit a small screen**, and the
    /// first version of this did not: at 30 columns it rendered as `This window needs a larger
    /// scr`, cut mid-word. That is worse than the short form, because a reader cannot tell a
    /// truncated sentence from a broken program — and this message is the one thing on screen
    /// when everything else has already failed to fit.
    pub fn headline(&self, width: u16) -> &'static str {
        const FULL: &str = "This window needs a larger screen";
        const SHORT: &str = "Screen too small";
        if width as usize >= FULL.chars().count() {
            FULL
        } else {
            SHORT
        }
    }

    /// The second line, at the longest form that fits `width`, or nothing if neither does.
    ///
    /// Separate from [`TooSmall::headline`] so a test can read the numbers without a buffer, and
    /// [`None`] rather than a truncation for the same reason: the headline alone is still true,
    /// and half a pair of numbers is not.
    pub fn detail(&self, width: u16) -> Option<String> {
        let full = format!(
            "{}x{} needed, {}x{} available",
            MIN_PAGE_COLS, MIN_PAGE_ROWS, self.available.0, self.available.1
        );
        let short = format!("{MIN_PAGE_COLS}x{MIN_PAGE_ROWS} needed");
        let fits = |text: &String| text.chars().count() <= width as usize;
        [full, short].into_iter().find(fits)
    }
}

impl Widget for TooSmall {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.height < 2 || area.width == 0 {
            return;
        }
        let top = area.y + area.height.saturating_sub(2) / 2;
        let row = |y: u16| Rect {
            y,
            height: 1,
            ..area
        };
        Paragraph::new(Line::from(Span::styled(
            self.headline(area.width),
            tokens::strong_style(),
        )))
        .alignment(Alignment::Center)
        .render(row(top), buf);
        if let Some(detail) = self.detail(area.width) {
            Paragraph::new(Line::from(Span::styled(detail, tokens::muted_style())))
                .alignment(Alignment::Center)
                .render(row(top + 1), buf);
        }
    }
}

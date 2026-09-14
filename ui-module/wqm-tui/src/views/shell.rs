//! The page's own chrome, on a real screen — what tabs 1–9 all begin and end with.
//!
//! Chris, 20260906: the first four rows of every tab are the same four rows. This view is
//! where that claim is *looked at* rather than asserted — the selector, the frame rule, the
//! status frame and the hairline that closes it, and since 2026-09-14 the hairline and
//! minimalist help line at the other end, over a content region that deliberately holds
//! nothing. Everything between them belongs to a tab and none of it is built yet, so the
//! region says so in one muted line instead of depicting a screen that does not exist.
//!
//! It draws none of that itself any more: it is [`crate::views::page::Page`] with a
//! placeholder where a view goes, which is the cheapest possible demonstration that the frame
//! and the view are separable.
//!
//! # Why the placeholder is a line and not a mock-up
//!
//! `views::service` already carries the rule this obeys: *a frame whose content was invented
//! is a frame of a screen that does not exist*. A plausible-looking Queue table here would be
//! exactly that, and it would also be the thing the eye went to — which would make this frame
//! useless for judging the four rows it exists to judge.
//!
//! # The Service tab is the one tab without a status frame
//!
//! It has [`crate::panes::status_band`], which says the same thing in more detail and from the
//! same readings. Two claims about one system, on one screen, is how the two come to disagree
//! — the same rule that keeps the daemon out of the store list (§6.29). So the `Service tab`
//! frame is selector, rule, content: shorter by four rows, and that difference is the point.
//!
//! # The content floor is the view's number, not the block's
//!
//! [`crate::panes::status_block::MIN_CONTENT_ROWS`] is chosen (Chris, 2026-09-07). It is
//! passed *in* rather than read inside the block because a tab holding a table can afford
//! less top furniture than one holding a summary, and that is a per-screen judgement.

use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Rect},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::panes::status_block::{self, StatusBlock};
use crate::tokens;
use crate::views::page::{row, Page};
use crate::widgets::chrome::status_line::ALWAYS;

/// What the content region says while no tab is built. One muted line — see the module docs
/// for why it is not a mock-up.
pub const PLACEHOLDER: &str = "— tab content —";

/// The constant top, over an empty content region.
pub struct ShellView {
    active: usize,
    modal: bool,
    /// Absent on the Service tab, which carries [`crate::panes::status_band`] instead.
    status: Option<StatusBlock>,
    /// Rows the content region must keep before the block gives way. The view's number — see
    /// the module docs.
    content_floor: u16,
}

impl ShellView {
    /// A tab with the constant status block above its content.
    pub fn new(active: usize, status: StatusBlock) -> Self {
        Self {
            active,
            modal: false,
            status: Some(status),
            content_floor: status_block::MIN_CONTENT_ROWS,
        }
    }

    /// The Service tab: app bar, rule, content, and no block.
    pub fn without_status(active: usize) -> Self {
        Self {
            active,
            modal: false,
            status: None,
            content_floor: status_block::MIN_CONTENT_ROWS,
        }
    }

    /// Whether a modal owns the input.
    ///
    /// A **screen-level** fact, and the screen is the only thing that carries it: [`Widget::render`]
    /// opens a [`tokens::ModalScope`] around the whole page, and every colour beneath goes muted
    /// on its own ([`crate::tokens::modal`]). No widget below is told, because the version that
    /// told them muted three things and left the rest of the page painting.
    pub fn under_modal(mut self, modal: bool) -> Self {
        self.modal = modal;
        self
    }

    /// State a different floor. Left as a builder rather than a `new` parameter because every
    /// frame in this crate wants the default, and a fifth positional argument on a constructor
    /// is how the wrong number gets passed silently.
    pub fn content_floor(mut self, rows: u16) -> Self {
        self.content_floor = rows;
        self
    }
}

impl Widget for ShellView {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Held for the whole draw, so every token read beneath it answers as a page under a
        // modal — the bar, the block, its discs and its counts alike.
        let _modal = self.modal.then(tokens::ModalScope::enter);
        let mut page = Page::new(self.active).content_floor(self.content_floor);
        if let Some(block) = self.status {
            page = page.status(block);
        }
        // A page with no tab built has nothing of its own to offer, so the foot is the two
        // hints every screen ends with. It is not empty: `? Help · q Quit` is the pair that
        // survives a foot too narrow for anything else ([`StatusLine::ALWAYS`]), and the one
        // that opens the window listing the rest.
        for (key, label) in ALWAYS {
            page = page.hint(key, label);
        }
        placeholder(page.draw(area, buf), buf);
    }
}

/// The empty content region: one muted line, centred, and nothing else at all.
fn placeholder(area: Rect, buf: &mut Buffer) {
    if area.is_empty() {
        return;
    }
    Paragraph::new(Line::from(Span::styled(PLACEHOLDER, tokens::muted_style())))
        .alignment(Alignment::Center)
        .render(row(area, area.height / 2), buf);
}

/// The frames this view is judged from — shared by the pantry and the tests, so a guard and a
/// picture are never of two different screens.
#[cfg(any(test, feature = "tui-pantry"))]
pub(crate) mod frames {
    use super::*;
    use crate::panes::status_block::{Queue, ENTRY_LABELS};
    use crate::tokens::Health;
    use crate::widgets::chrome::Freshness;
    use std::time::Duration;

    /// Not a decision — §7 leaves the freshness SLA open (OQ-6); the same number the Service
    /// frames are drawn against, so the two screens age at the same rate.
    const FRAME_SLA: Duration = Duration::from_secs(60);

    fn block(entries: [Health; ENTRY_LABELS.len()], queue: Queue) -> StatusBlock {
        StatusBlock::new(
            // Derived, never asserted: the roll-up cannot contradict the parts under it.
            status_block::rollup(entries[0], &entries[1..]),
            "v0.2.0",
            Freshness::new(Duration::from_secs(4), FRAME_SLA),
            entries,
            queue,
        )
    }

    fn idle() -> Queue {
        Queue {
            pending: 0,
            in_progress: 0,
            failed: 0,
            health: Health::Healthy,
        }
    }

    /// Tab 1, everything nominal — the frame the constant top is judged against.
    pub fn dashboard() -> ShellView {
        ShellView::new(0, block([Health::Healthy; ENTRY_LABELS.len()], idle()))
    }

    /// Tab 2 with the vector store degraded and work piling up behind it.
    pub fn queue_degraded() -> ShellView {
        let mut entries = [Health::Healthy; ENTRY_LABELS.len()];
        entries[1] = Health::Degraded;
        ShellView::new(
            1,
            block(
                entries,
                Queue {
                    pending: 1_240,
                    in_progress: 8,
                    failed: 3,
                    health: Health::Degraded,
                },
            ),
        )
    }

    /// Tab 10 — the one tab whose top is two rows rather than six.
    pub fn service_tab() -> ShellView {
        ShellView::without_status(crate::widgets::chrome::app_bar::SERVICE_TAB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::Health;
    use crate::views::page::{APP_BAR_ROW, CONSTANT_ROWS, TOP_RULE_ROW};
    use crate::widgets::chrome::rule::RULE;
    use crate::widgets::chrome::test_support::{coloured_cells, neutral_rungs, Restore};
    use crate::widgets::chrome::MARGIN;

    fn render(view: ShellView, width: u16, height: u16) -> Buffer {
        let area = Rect::new(0, 0, width, height);
        let mut buf = Buffer::empty(area);
        view.render(area, &mut buf);
        buf
    }

    fn row(buf: &Buffer, y: u16) -> String {
        (0..buf.area.width)
            .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
            .collect()
    }

    /// The four constant rows, in their stated order, at the storyboard's own geometry.
    #[test]
    fn the_top_of_the_screen_is_the_bar_the_rule_the_block_and_its_seam() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(frames::dashboard(), 125, 34);
        // Inset by the screen margin, like every other row of content — only the rules run
        // edge to edge (§6).
        assert!(
            row(&buf, APP_BAR_ROW).starts_with(&format!("{}WQM TUI", " ".repeat(MARGIN as usize))),
            "{:?}",
            row(&buf, APP_BAR_ROW)
        );
        assert_eq!(row(&buf, TOP_RULE_ROW), RULE.repeat(125));
        assert!(row(&buf, TOP_RULE_ROW + 1).contains("Service status"));
        assert_eq!(
            row(&buf, TOP_RULE_ROW + status_block::ROWS_FULL),
            RULE.repeat(125),
            "the block closes with its own rule, four rows below the frame one"
        );
    }

    /// The Service hub carries its band instead, so the block must be absent — and the content
    /// region must reclaim the rows it would have taken.
    #[test]
    fn the_service_tab_carries_no_status_block() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(frames::service_tab(), 125, 34);
        for y in 0..34 {
            assert!(
                !row(&buf, y).contains("Service status"),
                "row {y} still carries the block: {:?}",
                row(&buf, y)
            );
        }
        assert!(
            row(&buf, TOP_RULE_ROW + 1).contains(PLACEHOLDER)
                || (TOP_RULE_ROW + 1..34).any(|y| row(&buf, y).contains(PLACEHOLDER)),
            "the content region starts immediately under the frame rule"
        );
    }

    /// The floor is honoured: a screen too short for both gets the short block, never a
    /// content region squeezed under the number the view stated.
    #[test]
    fn a_short_screen_collapses_the_block_rather_than_the_content() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(frames::dashboard(), 80, 20);
        let body = 20 - CONSTANT_ROWS;
        assert_eq!(
            StatusBlock::rows_for(Rect::new(0, 0, 80, body), status_block::MIN_CONTENT_ROWS),
            status_block::ROWS_COLLAPSED,
            "80×20 is the shape this frame exists to show"
        );
        assert!(
            row(&buf, TOP_RULE_ROW + status_block::ROWS_COLLAPSED) == RULE.repeat(80),
            "the collapsed block closes two rows below the frame rule"
        );
        for label in status_block::ENTRY_LABELS {
            assert!(
                (0..20).all(|y| !row(&buf, y).contains(label)),
                "{label} survived a collapse"
            );
        }
    }

    /// A modal takes the highlight off the bar and changes nothing else about the layout.
    #[test]
    fn a_modal_de_highlights_the_bar_without_moving_a_row() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let live = render(frames::dashboard(), 125, 34);
        let under = render(frames::dashboard().under_modal(true), 125, 34);

        let x = row(&live, APP_BAR_ROW)
            .find('1')
            .expect("the first jump digit") as u16;
        assert_eq!(
            live.cell((x, APP_BAR_ROW)).expect("cell").style().fg,
            Some(tokens::accent())
        );
        assert_eq!(
            under.cell((x, APP_BAR_ROW)).expect("cell").style().fg,
            Some(tokens::muted())
        );
        for y in 1..34 {
            assert_eq!(row(&live, y), row(&under, y), "row {y} moved under a modal");
        }
    }

    /// The whole-frame companion to the guard above: under a modal not one cell of the Shell
    /// carries a colour, discs and queue counts included (VL §6, Chris 2026-09-07).
    ///
    /// The bar guard says the digits went muted. It cannot say the status block's RAG discs and
    /// its three queue counts did — those are the cells that were still painting when Chris
    /// looked at the frame, and no per-widget guard was ever going to see them.
    #[test]
    fn no_cell_of_the_shell_carries_a_colour_under_a_modal() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let neutrals = neutral_rungs();

        // A live frame first: a screen that painted nothing anyway would pass the sweep below
        // without the switch existing.
        let live = render(frames::dashboard(), 125, 34);
        assert!(
            !coloured_cells(&live, &neutrals).is_empty(),
            "the Shell paints no colour even when it is live — this guard checks nothing"
        );

        let under = render(frames::dashboard().under_modal(true), 125, 34);
        let survivors = coloured_cells(&under, &neutrals);
        assert!(
            survivors.is_empty(),
            "{} cells kept a colour under a modal, first ten: {:?}",
            survivors.len(),
            &survivors[..survivors.len().min(10)]
        );
    }

    /// The roll-up on the block is the one the entries produce — a green dot over a degraded
    /// part is not a frame this view can build.
    #[test]
    fn the_frames_rollup_is_derived_from_the_parts_it_is_showing() {
        assert_eq!(
            status_block::rollup(Health::Healthy, &[Health::Degraded, Health::Healthy]),
            Health::Degraded
        );
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();
        let buf = render(frames::queue_degraded(), 125, 34);
        let line = row(&buf, TOP_RULE_ROW + 1);
        assert!(
            line.starts_with(&format!("{}{}", " ".repeat(MARGIN as usize), tokens::DISC)),
            "the roll-up is a disc at the content margin: {line:?}"
        );
        // Since 20260906 the RAG is one disc in three hues, so the STATE is only readable off
        // the colour — a mark-only assertion here would pass against a green roll-up.
        assert_eq!(
            buf.cell((MARGIN, TOP_RULE_ROW + 1))
                .expect("cell in area")
                .style()
                .fg,
            Some(Health::Degraded.color()),
            "the roll-up hue follows the parts: {line:?}"
        );
    }
}

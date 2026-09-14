//! The modal FRAME — **Container** plus **Decoration**, the two thirds of every wqm window
//! that are not the view inside it.
//!
//! Chris, 2026-09-13 20:30: *"each such window is made from three components: a modal
//! container, a decoration (the top 4-lines and the bottom 3 lines), and a view. These are
//! composable"*. This module owns the first two; [`crate::views::modal_framework`] owns the
//! third and the composition. The split is the ruling's own, and it is what makes a view
//! reusable: the same table drawn on the main screen and inside a window differs only in what
//! is wrapped around it.
//!
//! ```text
//! ┌──────────────────────────────────────────────┐  ← Container: border, fill, modal tint
//! │  Libraries › open-books › Queue              │  ← Decoration row 0, breadcrumb (chevrons)
//! │                                              │  ← row 1, blank
//! │  open-books — queue          -- EDIT --      │  ← row 2, title (+ the mode banner)
//! │                                              │  ← row 3, blank
//! │  f open-books                                │  ← row 4, OPTIONAL search/filter input
//! │  … the View, scrolled, with a scrollbar     ▐│  ← the viewport
//! │                                              │  ← row h-3, blank
//! │  ↓↑/jk Move   ↵ Drill down   ⌫ Back          │  ← rows h-2 and h-1, minimalist help
//! │  ? Help   q Close                            │
//! └──────────────────────────────────────────────┘
//! ```
//!
//! # The footprint is ONE size, and it is not the view's to choose
//!
//! Chris: the window *"is the help window's size and does not resize between views"*. A window
//! that changed shape on every push would be a flow no reader can build a model of, so the
//! size is the container's and the view gets whatever is left. [`Footprint`] carries the two
//! readings of *which* size that is — see the variants, and the director's gate call, which is
//! [`Footprint::Framework`].
//!
//! # What the decoration spends, and what it refuses to
//!
//! Breadcrumb ancestors are [`crate::tokens::muted`] and the leaf is
//! [`crate::tokens::normal`]: the trail is metadata, the place you are standing is content.
//! The chevron is [`crate::tokens::faint`] — it is punctuation, and punctuation competing with
//! the names it separates is chrome charged as signal. The title is
//! [`crate::tokens::strong_style`], the one loud thing in the frame. The help rows go through
//! [`crate::tokens::key_hints`], the same producer the screen's status line uses, so a
//! window's help and a screen's help cannot come to spell a chord two ways.
//!
//! **No rule, no divider, no second border.** Two blank rows already separate the title from
//! the content and the content from the help; a line drawn through a gap that is already doing
//! the work is data-ink spent on nothing, and §6 gives the box itself the job of saying *this
//! is a window*.

use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph, Widget},
};

use crate::tokens;
use crate::widgets::modal::Fill;

#[cfg(test)]
mod tests;

/// Border plus one cell of padding on each side — the same chrome [`crate::widgets::modal`]
/// reserves, so a framework window and a plain modal put their text on the same column.
pub const CHROME: u16 = 4;

/// Top decoration without the optional search row: breadcrumb / blank / title / blank.
pub const TOP_ROWS: u16 = 4;

/// Bottom decoration: blank, then **two** help rows.
///
/// Two is the ruling's number and it is reserved whether or not both are filled (director's
/// gate, 2026-09-13: *"one may be blank"*). A help area that grew a row when a view had one
/// more key would move the content under the reader for a reason they cannot see.
pub const BOTTOM_ROWS: u16 = 3;

/// How many of [`BOTTOM_ROWS`] carry key hints.
pub const HELP_ROWS: usize = 2;

/// The rows a wqm screen spends on its own header before any content: the app bar and the rule
/// under it ([`crate::views::page::CONSTANT_ROWS`]), the three-row status block, and the rule
/// that closes it. A window starts below them — see [`Footprint::rect`].
pub const PAGE_HEADER_ROWS: u16 = crate::views::page::CONSTANT_ROWS + 4;

/// The chevron between two crumbs (Chris: *"breadcrumbs in the window (chevron style)"*).
pub const CHEVRON: &str = " \u{203a} ";

/// A search or filter input mounted on the optional fifth row.
///
/// **Declared at composition time, not summoned by a keypress** (director's reading, cheap to
/// overturn). A view that offers search declares the row and it sits blank until used, so the
/// viewport's height never changes under a reader who is in the middle of filtering — which is
/// the failure the row exists to avoid, not a version of it one keystroke later.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SearchRow {
    /// The key that opens it — `/` for search, `f` for filter — drawn as the prompt, in the
    /// affordance hue, because that is what a key you may press already wears.
    pub prompt: &'static str,
    pub term: String,
    /// `true` while the term is still being typed, which is what puts the caret on it.
    pub typing: bool,
}

impl SearchRow {
    pub fn search(term: impl Into<String>, typing: bool) -> Self {
        Self {
            prompt: "/",
            term: term.into(),
            typing,
        }
    }

    pub fn filter(term: impl Into<String>, typing: bool) -> Self {
        Self {
            prompt: "f",
            term: term.into(),
            typing,
        }
    }

    fn line(&self) -> Line<'static> {
        let mut spans = vec![
            Span::styled(
                self.prompt.to_string(),
                Style::default().fg(tokens::field::affordance()),
            ),
            Span::styled(" ", tokens::normal_style()),
        ];
        if self.typing {
            // The one caret this crate has, on the one surface being typed into.
            let edit = crate::widgets::edit_field::Edit::insert(self.term.clone());
            let style = tokens::normal_style().bg(tokens::field::active_bg());
            spans.extend(crate::widgets::edit_field::caret_spans(&edit, style));
            spans.push(Span::styled(" ", style));
        } else {
            spans.push(Span::styled(self.term.clone(), tokens::normal_style()));
        }
        Line::from(spans)
    }
}

/// The scroll position a Container draws a bar for.
///
/// Part of the per-view state the stack keeps (task 3): a pop restores the parent's offset
/// exactly as it was left, which is why this is data the composition hands in rather than
/// something the container works out for itself.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Scroll {
    /// First visible row of the view's own content.
    pub offset: usize,
    /// Rows the content occupies in total.
    pub total: usize,
}

/// The top four-or-five rows and the bottom three.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Decoration {
    crumbs: Vec<String>,
    title: String,
    search: Option<SearchRow>,
    help: Vec<(String, String)>,
}

impl Decoration {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            crumbs: Vec::new(),
            title: title.into(),
            search: None,
            help: Vec::new(),
        }
    }

    /// The trail, root first.
    ///
    /// **The order is the path the reader took, not a fixed hierarchy** (task 3): decorations
    /// and views are reusable across entry points, so the same view reached two ways shows two
    /// trails. Nothing here derives the trail from the view, and that is the point — a
    /// container that worked out its own breadcrumb would be a container asserting a hierarchy
    /// that the navigation does not have.
    pub fn crumbs<S: Into<String>>(mut self, crumbs: Vec<S>) -> Self {
        self.crumbs = crumbs.into_iter().map(Into::into).collect();
        self
    }

    pub fn search(mut self, row: SearchRow) -> Self {
        self.search = Some(row);
        self
    }

    pub fn hint(mut self, key: impl Into<String>, label: impl Into<String>) -> Self {
        self.help.push((key.into(), label.into()));
        self
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn crumb_trail(&self) -> &[String] {
        &self.crumbs
    }

    /// Four rows, or five when this view mounts a search/filter input.
    pub fn top_rows(&self) -> u16 {
        TOP_ROWS + u16::from(self.search.is_some())
    }

    fn crumb_line(&self) -> Line<'static> {
        let mut spans = Vec::new();
        let last = self.crumbs.len().saturating_sub(1);
        for (at, crumb) in self.crumbs.iter().enumerate() {
            if at > 0 {
                spans.push(Span::styled(CHEVRON, tokens::faint_style()));
            }
            let style = if at == last {
                tokens::normal_style()
            } else {
                tokens::muted_style()
            };
            spans.push(Span::styled(crumb.clone(), style));
        }
        Line::from(spans)
    }

    /// The two help rows, packed greedily so the first fills before the second starts.
    ///
    /// Always [`HELP_ROWS`] lines, even when the second is empty — see [`BOTTOM_ROWS`].
    ///
    /// **Two rows is a budget, and a hint that does not fit is dropped WHOLE.** The first cut
    /// packed the overflow onto row two without measuring it, and ratatui clipped whatever ran
    /// past the edge — so a window with nine long hints ended in `k6 a rather`, a key half
    /// named. A hint cut mid-word is worse than a hint absent: absent, the reader goes to `?`,
    /// which is where the complete list lives by construction. A window that loses hints here
    /// has over-declared them, and `debug_assert` says so in a test build rather than letting
    /// it pass silently into a frame.
    fn help_lines(&self, width: u16) -> [Line<'static>; HELP_ROWS] {
        let mut rows: [Vec<(String, String)>; HELP_ROWS] = Default::default();
        let mut at = 0usize;
        let mut dropped = 0usize;
        for pair in &self.help {
            loop {
                let mut trial = rows[at].clone();
                trial.push(pair.clone());
                if tokens::key_hints_width(&trial) <= width as usize {
                    rows[at] = trial;
                    break;
                }
                // It did not fit on this row. Try the next one; past the last, the budget is
                // spent and the hint does not appear at all.
                if at + 1 < HELP_ROWS {
                    at += 1;
                    continue;
                }
                dropped += 1;
                break;
            }
        }
        debug_assert!(
            dropped == 0,
            "{dropped} key hints do not fit the window's two help rows at width {width} — the \
             view has declared more than the decoration's budget"
        );
        [
            Line::from(tokens::key_hints(&rows[0])),
            Line::from(tokens::key_hints(&rows[1])),
        ]
    }
}

/// Where the fixed window's size comes from — the A/B of round 1's first frame pair.
///
/// Chris ruled that the window *"is the help window's size and does not resize between
/// views"*. There are two readings of that sentence, and only one is consistent with the rest
/// of the same ruling.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Footprint {
    /// **B — the director's gate call.** One framework size, derived from the SCREEN, which
    /// the help window ADOPTS along with every other window.
    ///
    /// This is what makes the rest of task 6 work. The same ruling says in the next breath
    /// that *"the help window is also a composition: a modal container and a fixed
    /// **scrollable** content"* — and content only scrolls inside a container it does not fit.
    /// Under arm A the help window can never scroll, because it is by construction exactly as
    /// tall as its own text.
    ///
    /// `min(96, width − 16) × min(24, height − 8)`: sixteen columns and eight rows of page are
    /// kept, and the caps stop the window growing without limit on a large terminal — past
    /// about a hundred columns a window stops reading as a window, which is the same reasoning
    /// behind `modal::MAX_TEXT_WIDTH`.
    #[default]
    Framework,
    /// **A — the ruling read literally:** whatever rect the help window currently wants.
    ///
    /// Kept reachable because it is a faithful reading and Chris may prefer it. Measured at
    /// 125×34 it is 99 × 32 — 79% of the columns and 94% of the rows, leaving thirteen columns
    /// of page either side and one row top and bottom, which leaves §6's depth model with no
    /// page left to recede. At the 100×30 floor the help content does not fit at all and the
    /// window is clamped, so the size is not even stable across the two sizes it must hold.
    HelpDerived,
}

impl Footprint {
    /// The window's rectangle inside `area`.
    ///
    /// Vertically: centred, but **never above the page's own header**, and that is not a taste
    /// call. A wqm screen spends [`PAGE_HEADER_ROWS`] on the app bar, a rule and the status
    /// block, and a mathematically centred window lands its top border exactly on that rule at
    /// BOTH 125×34 and the 100×30 floor — two box-drawing runs of the same weight on one row,
    /// which reads as the window being welded to the page rather than floating over it.
    /// Clearing the header also keeps the roll-up, the four stores and the queue counts
    /// readable while a window is up, so a modal never costs the reader the answer to *is
    /// anything wrong* (Nielsen #1).
    pub fn rect(self, area: Rect) -> Rect {
        let (width, height) = match self {
            Footprint::Framework => (
                area.width.saturating_sub(16).min(96),
                area.height.saturating_sub(8).min(24),
            ),
            Footprint::HelpDerived => {
                let help = crate::views::queue::Queue::help().rect(area);
                (
                    help.width.min(area.width.saturating_sub(4)),
                    help.height.min(area.height.saturating_sub(2)),
                )
            }
        };
        let centred = area.y + (area.height.saturating_sub(height)) / 2;
        let below_header = area.y + PAGE_HEADER_ROWS;
        let floor = area.bottom().saturating_sub(height);
        Rect {
            x: area.x + (area.width.saturating_sub(width)) / 2,
            y: centred.max(below_header).min(floor.max(area.y)),
            width,
            height,
        }
    }
}

/// The fixed window: the layer, the tinted box, the decoration, and a scroll viewport.
pub struct Container {
    decoration: Decoration,
    fill: Fill,
    scroll: Option<Scroll>,
    /// A banner beside the title. The EDIT-mode `-- EDIT --` uses it, bold and with no hue —
    /// r06 #8, and the reason it is here at all rather than left to the field fills: a mode
    /// visible only as colour is invisible under `NO_COLOR` and to a reader who is not looking
    /// at a field.
    title_banner: Option<String>,
}

impl Container {
    pub fn new(decoration: Decoration) -> Self {
        Self {
            decoration,
            fill: Fill::Layer1,
            scroll: None,
            title_banner: None,
        }
    }

    pub fn fill(mut self, fill: Fill) -> Self {
        self.fill = fill;
        self
    }

    pub fn scroll(mut self, scroll: Scroll) -> Self {
        self.scroll = Some(scroll);
        self
    }

    pub fn title_banner(mut self, banner: impl Into<String>) -> Self {
        self.title_banner = Some(banner.into());
        self
    }

    pub fn decoration(&self) -> &Decoration {
        &self.decoration
    }

    /// **The window's size**, as one function, because Chris may overturn the gate's reading
    /// back to [`Footprint::HelpDerived`] and a size spelled at each call site is a size that
    /// only moves where somebody remembers it.
    pub fn footprint(area: Rect) -> Rect {
        Footprint::Framework.rect(area)
    }

    /// Inside the border and its one column of padding.
    fn inner(rect: Rect) -> Rect {
        Rect {
            x: rect.x + 2,
            y: rect.y + 1,
            width: rect.width.saturating_sub(CHROME),
            height: rect.height.saturating_sub(2),
        }
    }

    /// The rows a view may draw into, inside `rect` — the scrollbar's column excluded.
    ///
    /// Returns a zero-height rect when the decoration alone fills the window, so a view handed
    /// one draws nothing rather than drawing over the help rows.
    pub fn viewport(&self, rect: Rect) -> Rect {
        let inner = Self::inner(rect);
        let top = self.decoration.top_rows();
        let used = top + BOTTOM_ROWS;
        if inner.height <= used {
            return Rect { height: 0, ..inner };
        }
        Rect {
            x: inner.x,
            y: inner.y + top,
            width: inner.width.saturating_sub(u16::from(self.scroll.is_some())),
            height: inner.height - used,
        }
    }

    /// The scrollbar, in the column the viewport gave back.
    ///
    /// Nothing is drawn when the content fits: a bar that is always full is a bar that says
    /// nothing, and it costs the column it stands in.
    fn draw_scrollbar(&self, viewport: Rect, buf: &mut Buffer) {
        let Some(scroll) = self.scroll else { return };
        if viewport.height == 0 || scroll.total <= viewport.height as usize {
            return;
        }
        let x = viewport.x + viewport.width;
        let rows = viewport.height as usize;
        // The thumb is at least one row: a bar that vanishes on a long list stops being a bar.
        let thumb = ((rows * rows) / scroll.total).max(1);
        let span = rows.saturating_sub(thumb);
        let travel = scroll.total.saturating_sub(rows).max(1);
        let at = (scroll.offset.min(travel) * span) / travel;
        for row in 0..rows {
            let inside = row >= at && row < at + thumb;
            let (glyph, colour) = if inside {
                ("\u{2590}", tokens::muted())
            } else {
                ("\u{2595}", tokens::rule_internal())
            };
            buf.set_string(
                x,
                viewport.y + row as u16,
                glyph,
                Style::default().fg(colour),
            );
        }
    }
}

impl Widget for Container {
    fn render(self, rect: Rect, buf: &mut Buffer) {
        if rect.width <= CHROME || rect.height < self.decoration.top_rows() + BOTTOM_ROWS + 2 {
            return;
        }
        // Occlusion, not restyling. `Block::style` restyles the cells it covers and leaves
        // their symbols standing, so a window's blank rows would show the page through them
        // wearing the window's background — `widgets::modal` learned this the hard way and the
        // comment there is the record.
        Clear.render(rect, buf);
        Block::bordered()
            .border_style(Style::default().fg(tokens::modal_border()))
            // Through `tokens::modal_fill`, the crate's one blend: the window and every
            // surface inside it move together when the tint strength moves, or the form drifts
            // out of the window's colour family the moment the wash is retuned.
            .style(Style::default().bg(tokens::modal_fill(match self.fill {
                Fill::Layer1 => tokens::layer1_bg(),
                Fill::Layer2 => tokens::layer2_bg(),
            })))
            .render(rect, buf);

        let inner = Self::inner(rect);
        let row = |n: u16| Rect {
            y: inner.y + n,
            height: 1,
            ..inner
        };

        Paragraph::new(self.decoration.crumb_line()).render(row(0), buf);
        Paragraph::new(Line::from(Span::styled(
            self.decoration.title.clone(),
            tokens::strong_style(),
        )))
        .render(row(2), buf);

        if let Some(banner) = &self.title_banner {
            Paragraph::new(Line::from(Span::styled(
                banner.clone(),
                tokens::normal_style().add_modifier(Modifier::BOLD),
            )))
            .alignment(Alignment::Right)
            .render(row(2), buf);
        }

        if let Some(search) = &self.decoration.search {
            Paragraph::new(search.line()).render(row(TOP_ROWS), buf);
        }

        let viewport = self.viewport(rect);
        self.draw_scrollbar(viewport, buf);

        let help = self.decoration.help_lines(inner.width);
        let foot = inner.y + inner.height;
        for (at, line) in help.into_iter().enumerate() {
            Paragraph::new(line).render(
                Rect {
                    y: foot - HELP_ROWS as u16 + at as u16,
                    height: 1,
                    ..inner
                },
                buf,
            );
        }
    }
}

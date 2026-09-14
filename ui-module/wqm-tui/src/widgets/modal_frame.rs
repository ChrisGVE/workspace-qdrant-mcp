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
//! size is the container's and the view gets whatever is left. Chris settled *which* size on
//! 2026-09-14: the page inset by [`INSET`] is the maximum, a window outside a drill-down sizes
//! to its content under that cap, and a drill-down opens at the maximum. See [`Footprint`].
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

pub mod crumbs;
pub mod sizing;

#[cfg(test)]
mod tests;

pub use crumbs::CrumbStyle;
pub use sizing::{Footprint, TooSmall, INSET, MIN_COLS, MIN_PAGE_COLS, MIN_PAGE_ROWS, MIN_ROWS};

/// Whether the window draws its border glyphs, or only reserves their room.
///
/// Chris, 2026-09-14, item (e): *"The window does not need a frame, it already has one: its
/// background, but I don't disagree to keep the border characters/lines unused to avoid feeling
/// crammed."*
///
/// So the two arms differ in **ink and nothing else**. [`Edge::Spacing`] keeps every cell of
/// [`CHROME`] exactly where [`Edge::Bordered`] puts it, and simply does not draw the box — the
/// content does not move by a column, which is what makes the pair a fair comparison rather than
/// two layouts. Tufte's data-ink: the fill already says *this is a window*, so the box is a
/// second statement of a fact the reader has already had.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Edge {
    /// Round 1: a drawn box in the modal's hue.
    #[default]
    Bordered,
    /// **Round 2, item (e)**: the same padding, no glyphs.
    Spacing,
}

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
    crumb_style: CrumbStyle,
    title: String,
    search: Option<SearchRow>,
    help: Vec<(String, String)>,
}

impl Decoration {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            crumbs: Vec::new(),
            crumb_style: CrumbStyle::default(),
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

    /// How the trail is drawn — item (a)'s powerline form, or round 1's plain one.
    pub fn crumb_style(mut self, style: CrumbStyle) -> Self {
        self.crumb_style = style;
        self
    }

    fn crumb_line(&self) -> Line<'static> {
        crumbs::line(&self.crumbs, self.crumb_style)
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

/// The fixed window: the layer, the tinted box, the decoration, and a scroll viewport.
pub struct Container {
    decoration: Decoration,
    fill: Fill,
    scroll: Option<Scroll>,
    /// **Item 1's horizontal bar.** Present only when the content cannot shrink to the window —
    /// Chris: *"in the case the content cannot shrink horizontally, then the horizontal
    /// scrolling should be an option"*.
    ///
    /// Its `total` and `offset` are in COLUMNS of the view's own content, exactly as the
    /// vertical one's are in rows, so one `Scroll` type serves both and a caller cannot get the
    /// units wrong by picking the wrong struct.
    hscroll: Option<Scroll>,
    edge: Edge,
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
            hscroll: None,
            edge: Edge::default(),
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

    /// Item 1's horizontal bar, for a view whose content cannot shrink to the window.
    pub fn hscroll(mut self, scroll: Scroll) -> Self {
        self.hscroll = Some(scroll);
        self
    }

    /// Item (e): draw the border, or only reserve its room.
    pub fn edge(mut self, edge: Edge) -> Self {
        self.edge = edge;
        self
    }

    pub fn title_banner(mut self, banner: impl Into<String>) -> Self {
        self.title_banner = Some(banner.into());
        self
    }

    pub fn decoration(&self) -> &Decoration {
        &self.decoration
    }

    /// **The window a caller gets when it states no size of its own** — the maximum, which is
    /// item 0's page inset.
    ///
    /// A convenience over [`Footprint::window`] for the callers that draw at a page size known
    /// to hold a window, so they are not each writing the same `expect`. A composition that
    /// sizes to its content states [`Footprint::Content`] instead.
    ///
    /// # Panics
    /// If `area` cannot hold a window at all. Every caller renders at 125x34 or the 100x30
    /// floor, both far above [`MIN_PAGE_COLS`] x [`MIN_PAGE_ROWS`], so the arm is unreachable —
    /// and saying so is better than inventing a fallback rect nobody would notice was wrong.
    pub fn footprint(area: Rect) -> Rect {
        Footprint::Max.window(area).unwrap_or_else(|| {
            panic!(
                "{}x{} cannot hold a window; the minimum page is {MIN_PAGE_COLS}x\
                 {MIN_PAGE_ROWS}, and a caller that has to meet a small screen must use \
                 `Footprint::window` and draw `TooSmall`",
                area.width, area.height
            )
        })
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
        // Each bar costs the line it stands on, and it costs it whether or not the thumb is
        // eventually drawn — a viewport that grew a row back when the content happened to fit
        // would reflow the moment a filter changed the row count.
        Rect {
            x: inner.x,
            y: inner.y + top,
            width: inner.width.saturating_sub(u16::from(self.scroll.is_some())),
            height: (inner.height - used).saturating_sub(u16::from(self.hscroll.is_some())),
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

    /// **Item 1's horizontal bar**, on the row the viewport gave back.
    ///
    /// # The glyphs are the vertical bar's, turned
    ///
    /// The vertical bar is `▐` `U+2590 RIGHT HALF BLOCK` for the thumb on `▕` `U+2595 RIGHT ONE
    /// EIGHTH BLOCK` for the track. The exact mirror of that pair is `▄` `U+2584 LOWER HALF
    /// BLOCK` on `▁` `U+2581 LOWER ONE EIGHTH BLOCK`: same two weights, same half-versus-eighth
    /// relationship, rotated. Picking a different family — `━`/`─`, say — would make the two
    /// bars read as two different mechanisms, which is the one thing a pair of scrollbars must
    /// not do.
    ///
    /// Both weights sit at the same two rungs as the vertical bar's, so a window with both bars
    /// has one scrollbar colour and not two.
    fn draw_hscrollbar(&self, viewport: Rect, buf: &mut Buffer) {
        let Some(scroll) = self.hscroll else { return };
        if viewport.width == 0 || scroll.total <= viewport.width as usize {
            return;
        }
        let y = viewport.y + viewport.height;
        let columns = viewport.width as usize;
        let thumb = ((columns * columns) / scroll.total).max(1);
        let span = columns.saturating_sub(thumb);
        let travel = scroll.total.saturating_sub(columns).max(1);
        let at = (scroll.offset.min(travel) * span) / travel;
        for column in 0..columns {
            let inside = column >= at && column < at + thumb;
            let (glyph, colour) = if inside {
                ("\u{2584}", tokens::muted())
            } else {
                ("\u{2581}", tokens::rule_internal())
            };
            buf.set_string(
                viewport.x + column as u16,
                y,
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
        // Through `tokens::modal_fill`, the crate's one blend: the window and every surface
        // inside it move together when the tint strength moves, or the form drifts out of the
        // window's colour family the moment the wash is retuned.
        let surface = Style::default().bg(tokens::modal_fill(match self.fill {
            Fill::Layer1 => tokens::layer1_bg(),
            Fill::Layer2 => tokens::layer2_bg(),
        }));
        match self.edge {
            Edge::Bordered => Block::bordered()
                .border_style(Style::default().fg(tokens::modal_border()))
                .style(surface)
                .render(rect, buf),
            // Item (e): the same rect, the same fill, no glyphs. `Block::default()` still paints
            // the whole rectangle, so the padding `inner` reserves is untouched and the content
            // sits on exactly the columns the bordered arm puts it on.
            Edge::Spacing => Block::default().style(surface).render(rect, buf),
        }

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
        self.draw_hscrollbar(viewport, buf);

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

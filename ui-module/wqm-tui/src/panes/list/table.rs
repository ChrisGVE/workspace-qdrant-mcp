//! The list itself: its header, its rows, its cursor, and the line that offers the next page.

use ratatui::{
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use super::{line, LIST_PAGE};
use crate::format::grouped;
use crate::panes::cell::sort::{compare, grown};
use crate::panes::cell::table::COLUMN_GAP;
use crate::panes::cell::{Cell, Column, Sort, EMPTY};
use crate::tokens;

/// A full-width sortable list. See the module docs for how it differs from a Dashboard cell.
pub struct ListPane {
    columns: Vec<Column>,
    rows: Vec<Vec<Cell>>,
    /// Which line the data cursor is on, counted over the whole buffer rather than over what is
    /// on screen — `rows.len()` is the load-more line when there is one.
    ///
    /// Not an [`Option`]: a list is its screen's only zone, so something always carries the
    /// cursor. An empty list draws none because there is no line to draw it on, which is a fact
    /// about the rows rather than a second state to represent.
    cursor: usize,
    /// Where the window onto the buffer starts. A *preference*, not a truth: [`ListPane::window`]
    /// moves it as far as it must to keep the cursor visible, so a frame can state a scroll
    /// position without also having to satisfy the invariant by hand.
    offset: usize,
    sort: Option<Sort>,
    /// Whether the caller believes there is more behind this page. The pane cannot know — it
    /// holds a page, not a store — so the one fact it cannot derive is the one it is given.
    more: bool,
}

impl ListPane {
    pub fn new(columns: Vec<Column>, rows: Vec<Vec<Cell>>) -> Self {
        Self {
            columns,
            rows,
            cursor: 0,
            offset: 0,
            sort: None,
            more: false,
        }
    }

    /// Put the data cursor on a line. Clamped on render rather than here, so a caller that
    /// states a cursor and then sorts the rows cannot be left pointing past the end.
    pub fn cursor(mut self, at: usize) -> Self {
        self.cursor = at;
        self
    }

    /// Where the window starts, before the cursor has its say. See the field.
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Say that the store may hold more than this page. See [`ListPane::shows_load_more`] for
    /// what makes the line appear, which is this AND a full page.
    pub fn more(mut self, more: bool) -> Self {
        self.more = more;
        self
    }

    /// Sort the rows by one column, and remember which one for the header's mark.
    ///
    /// The rows are reordered HERE rather than at render time, exactly as
    /// [`crate::panes::cell::CellTable::sorted`] does, so [`ListPane::rows`] answers with what
    /// the list will actually draw — a guard reading the data and a reader reading the screen
    /// are then looking at one order rather than at two that ought to agree.
    ///
    /// Stable, so rows that compare equal keep the order the projection handed them in. On this
    /// list that is not a nicety: a hundred and one of the two hundred captured rows are `21m
    /// ago`, and an unstable sort by `Age` would shuffle them into an order nothing chose.
    pub fn sorted(mut self, sort: Sort) -> Self {
        let column = sort.column;
        self.rows.sort_by(|a, b| {
            let ordering = match (a.get(column), b.get(column)) {
                (Some(a), Some(b)) => compare(a, b),
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            };
            match sort.direction {
                crate::panes::cell::Direction::Asc => ordering,
                crate::panes::cell::Direction::Desc => ordering.reverse(),
            }
        });
        self.sort = Some(sort);
        self
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    pub fn rows(&self) -> &[Vec<Cell>] {
        &self.rows
    }

    /// Whether the list ends in the line that offers the next page.
    ///
    /// **A full page AND a caller who says there may be more.** Either half alone gets it wrong
    /// in a way a reader would have to discover by pressing: a partial page is the end of the
    /// store whatever the caller believes, because a store with more in it would have filled the
    /// page; and a full page is *not* evidence of more, because a store holding exactly two
    /// hundred rows fills one exactly. So the line means "this page is full and the caller
    /// believes there is more", which is the only claim the two facts together support.
    pub fn shows_load_more(&self) -> bool {
        self.more && !self.rows.is_empty() && self.rows.len().is_multiple_of(LIST_PAGE)
    }

    /// What that line says: how many rows are held, and how many one press would add.
    ///
    /// Both numbers are read rather than written — the first off the buffer, the second off
    /// [`LIST_PAGE`] — so a page size that changes changes the sentence with it, and a buffer
    /// that is short says so instead of claiming a round two hundred.
    pub fn load_more_text(&self) -> String {
        format!(
            "{} rows, press Enter to load {} more rows",
            grouped(self.rows.len() as u64),
            grouped(LIST_PAGE as u64)
        )
    }

    /// Lines the buffer holds: its rows, plus the load-more line when there is one.
    fn lines(&self) -> usize {
        self.rows.len() + usize::from(self.shows_load_more())
    }

    /// The cursor, clamped to a line that exists.
    fn at(&self) -> usize {
        self.cursor.min(self.lines().saturating_sub(1))
    }

    /// Where the window actually starts, given `body` rows of room.
    ///
    /// The stated [`ListPane::offset`] is honoured when it can be, clamped so the window never
    /// runs off the end, and then moved the minimum distance that brings the cursor back into
    /// view. Minimum distance on purpose: a list that re-centred on every move would slide under
    /// a reader who pressed `j` once.
    fn window(&self, body: usize) -> usize {
        if body == 0 {
            return 0;
        }
        let last = self.lines().saturating_sub(body);
        let mut offset = self.offset.min(last);
        let at = self.at();
        if at < offset {
            offset = at;
        } else if at >= offset + body {
            offset = at + 1 - body;
        }
        offset
    }

    /// Whether this list offers its sort keys: whenever there is more than one row.
    ///
    /// A list has no `Attention` to consult (see the module docs) so only the second half of the
    /// cell's rule survives — and it survives for the same reason. A lit letter is a promise
    /// that the key does something, and there is nothing to reorder in a list of one.
    fn sortable(&self) -> bool {
        self.rows.len() > 1
    }

    /// One column header: the title in the body foreground with [`Modifier::ITALIC`], its sort
    /// key lit in the accent hue, and the sort mark when this is the column the list is sorted
    /// by.
    ///
    /// A header is not bold (Chris, 2026-09-07) — that reads as a heading rather than as a label
    /// under it — so it keeps the body foreground and marks itself with italics instead. The key
    /// is **bold** on top of the italic: one letter of an italic header that changed hue alone
    /// would read as a gap in the heading rather than as a key to press. Its hue is
    /// [`tokens::accent`], the unclaimed §10 field, not the reserved selector — a sort key is a
    /// hint, not a selection. The mark stays muted, un-bold and un-italic: it says which column
    /// is sorted, and it is never the thing being read.
    fn header_spans(&self, column: &Column, mark: Option<Sort>) -> Vec<Span<'static>> {
        let rest = Style::default()
            .fg(tokens::header())
            .add_modifier(Modifier::ITALIC);
        let key = self.sortable().then_some(column.sort_key).flatten();
        let mut spans = crate::widgets::chrome::keyed_spans(
            column.title,
            key,
            rest.fg(tokens::accent()).add_modifier(Modifier::BOLD),
            rest,
        );
        if let Some(sort) = mark {
            spans.push(Span::styled(sort.direction.glyph(), tokens::muted_style()));
        }
        spans
    }

    fn header(&self, body: Rect, cells: &[Rect], buf: &mut Buffer) {
        for (at, (column, area)) in self.columns.iter().zip(cells).enumerate() {
            let mark = self.sort.filter(|sort| sort.column == at);
            let spans = self.header_spans(column, mark);
            let needed: u16 = spans
                .iter()
                .map(|span| span.content.chars().count() as u16)
                .sum();
            // Only the MARK may take room beyond the column — the same rule the cell's header
            // follows, and for the same reason. See `panes::cell::sort::grown`.
            let at_rect = if mark.is_some() {
                grown(*area, body, needed)
            } else {
                *area
            };
            Paragraph::new(Line::from(spans))
                .alignment(column.align.to_ratatui())
                .render(at_rect, buf);
        }
    }
}

/// Mark `row` as the one the data cursor is on: the tint across the whole width, and nothing
/// else at all.
///
/// The Dashboard's ruling, applied here (2026-09-07): no marker glyph, no gutter. The tint goes
/// down FIRST so every span drawn over it keeps its own hue and inherits this background.
fn paint_cursor(row: Rect, buf: &mut Buffer) {
    buf.set_style(row, Style::default().bg(tokens::cursor_bg()));
}

impl Widget for ListPane {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() || area.height == 0 {
            return;
        }
        let constraints: Vec<Constraint> = self.columns.iter().map(|c| c.width).collect();
        let columns = Layout::horizontal(constraints)
            .spacing(COLUMN_GAP)
            .split(Rect { height: 1, ..area });
        self.header(area, &columns, buf);

        if self.rows.is_empty() {
            Paragraph::new(Line::from(Span::styled(EMPTY, tokens::faint_style())))
                .render(line(area, 1), buf);
            return;
        }

        let body = area.height.saturating_sub(1) as usize;
        let offset = self.window(body);
        let at = self.at();

        for (i, index) in (offset..self.lines()).take(body).enumerate() {
            let row = line(area, 1 + i as u16);
            if index == at {
                paint_cursor(row, buf);
            }
            match self.rows.get(index) {
                // Indexed rather than zipped by reference: a row may carry fewer cells than the
                // list has columns, and the column its value belongs to is its POSITION.
                Some(cells) => {
                    for (n, (cell, column)) in cells.iter().zip(&self.columns).enumerate() {
                        Paragraph::new(Line::from(cell.spans(columns[n].width, column.elide)))
                            .alignment(column.align.to_ratatui())
                            .render(
                                Rect {
                                    y: row.y,
                                    height: 1,
                                    ..columns[n]
                                },
                                buf,
                            );
                    }
                }
                // Past the last row, so this is the load-more line — the only other line the
                // buffer has. Muted: it is an offer, not a datum.
                None => {
                    Paragraph::new(Line::from(Span::styled(
                        self.load_more_text(),
                        tokens::muted_style(),
                    )))
                    .render(row, buf);
                }
            }
        }
    }
}

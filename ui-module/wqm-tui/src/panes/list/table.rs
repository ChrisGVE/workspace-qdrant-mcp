//! The list itself: its header, its rows, its cursor, and the line that offers the next page.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use super::{line, LIST_PAGE};
use crate::format::grouped;
use crate::panes::cell::sort::{compare, grown};
use crate::panes::cell::{Cell, Column, Sort, EMPTY};
use crate::tokens;

/// The selection gutter: one column, at the left of whatever area the pane is given.
///
/// One, not two — it holds [`crate::tokens::SELECTED_BAR`], which is a half-block and needs no
/// separator after it. The caller supplies it out of its own margin
/// ([`crate::widgets::chrome::with_gutter`]), so the table's columns are exactly as wide as
/// they were before a selection existed.
pub const GUTTER: u16 = 1;

/// A full-width sortable list. See the module docs for how it differs from a Dashboard cell.
pub struct ListPane {
    /// The list's columns. The FIRST is the row-number column — untitled, on the left, and
    /// filled by [`ListPane::render`] with each row's position rather than with a stored value,
    /// so a sort or a filter renumbers the visible column 1, 2, 3 downward. The cells a row
    /// carries answer to the columns AFTER it: one cell per data column, which is one fewer than
    /// the length of this list.
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
    /// Whether the number column shows each row's DISTANCE from the cursor rather than its own
    /// position. Off by default, and a still frame rather than a key handling: the `r` key the
    /// live screen binds for it is not this crate's to press.
    relative: bool,
    /// Which drawn rows are selected, by their position in [`ListPane::rows`].
    ///
    /// Positions rather than row identities: by the time a pane exists the projection is fixed,
    /// so the caller — which knows what a row IS — has already answered the hard half. See
    /// [`crate::views::queue::Selection`], which is where a selection survives a narrowing.
    selected: Vec<bool>,
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
            relative: false,
            selected: Vec::new(),
        }
    }

    /// Number each row by its distance from the cursor rather than by its position. See the
    /// field: the mode the live screen's `r` key toggles, exposed here as a builder because a
    /// still frame cannot handle the press.
    pub fn relative(mut self, on: bool) -> Self {
        self.relative = on;
        self
    }

    /// Mark rows as selected, by their position in the rows this pane was given.
    ///
    /// Call it AFTER [`ListPane::sorted`] or not at all: a sort reorders the rows, and a mark
    /// stated against the order before it would land on other rows. The frame builder does
    /// exactly that, which is why nothing here tries to defend it — a pane cannot tell a stale
    /// position from a fresh one.
    pub fn selected(mut self, selected: Vec<bool>) -> Self {
        self.selected = selected;
        self
    }

    /// Whether the row at `index` wears the selection.
    fn is_selected(&self, index: usize) -> bool {
        self.selected.get(index).copied().unwrap_or(false)
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
        // `sort.column` names a column of the list; the cell a row holds for it sits one place
        // left, because the number column (0) holds no cell. The number column is never a sort
        // target — it offers no key, and a positional number is nothing that can be ordered — so
        // `column` is always at least one here.
        let column = sort.column.saturating_sub(1);
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

    /// The number the column shows for the row at buffer index `index`, with the cursor at `at`.
    ///
    /// Positional and computed HERE rather than stored: `index + 1` is the row's place from the
    /// top of the list as displayed, so the column always reads 1, 2, 3 downward whatever order
    /// the rows are in. In the relative mode the cursor row alone keeps that absolute position —
    /// the number a reader could point at — and every other row shows its distance from the
    /// cursor, so the rows immediately above and below both read 1.
    fn number_at(&self, index: usize, at: usize) -> u64 {
        if self.relative && index != at {
            index.abs_diff(at) as u64
        } else {
            (index + 1) as u64
        }
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

    /// One column header: the title WHITE and upright, its sort key lit in the accent hue, and
    /// the sort mark when this is the column the list is sorted by.
    ///
    /// The list's half of ruling 1 (Chris, 20260912), identical to the cell table's by design —
    /// *"that's the baseline that applies to every table"*. White against the light grey of
    /// [`tokens::table_row`] under it, no italics (tried on 20260907 and rejected), and the rule
    /// under the whole row drawn by [`ListPane::header`]. The key is **bold** in
    /// [`tokens::accent`], the unclaimed §10 field rather than the reserved selector — a sort
    /// key is a hint, not a selection. The mark stays muted: it says which column is sorted, and
    /// it is never the thing being read.
    fn header_spans(&self, column: &Column, mark: Option<Sort>) -> Vec<Span<'static>> {
        let rest = Style::default().fg(tokens::header());
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
        // The rule under the header, across the whole row, gaps included — see
        // [`crate::panes::cell::table::CellTable::header`], which does the same for the same
        // reason. It stops at the table: the selection gutter is margin, not header.
        buf.set_style(
            Rect {
                height: 1,
                ..body
            },
            Style::default().add_modifier(Modifier::UNDERLINED),
        );
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

/// Mark `row` as the one the data cursor is on: **the block** across the whole width, and
/// nothing else at all.
///
/// Chris, 20260912, ruling 7 — the cursor and the selection swapped treatments, and this is the
/// half that gained one. No marker glyph and no gutter still (2026-09-07): the fill IS the mark.
///
/// The fill goes down FIRST so every span drawn over it inherits this background; the black bold
/// content that makes it a block goes on LAST, in [`invert_cursor`], because each span sets its
/// own foreground as it is drawn.
fn paint_cursor(row: Rect, buf: &mut Buffer) {
    buf.set_style(row, Style::default().bg(tokens::cursor_bg()));
}

/// The second half of the block: the cursor row's content in black and **bold**, over the fill
/// [`paint_cursor`] laid down.
///
/// Separate from `paint_cursor` and called after the row is drawn rather than with it, because
/// ratatui styles patch and every span of the row sets its own foreground on the way past. A
/// colour laid down with the fill is the one thing they all overwrite.
///
/// [`tokens::cursor_fg`] answers [`None`] where there is no block to invert against — a modal
/// has taken the fill, or the encoding refuses colour and the fill was reverse video anyway.
fn invert_cursor(row: Rect, buf: &mut Buffer) {
    if let Some(fg) = tokens::cursor_fg() {
        buf.set_style(
            row,
            Style::default().fg(fg).add_modifier(Modifier::BOLD),
        );
    }
}

/// Mark `row` as selected: the tint across its width, and the bar in the gutter column.
///
/// Chris, 2026-09-07, ruling 10, with the fill swapped out by ruling 7 (20260912) — the tint is
/// now [`tokens::selection_bg`], rung 19, which is the fill the CURSOR used to wear.
///
/// **The cursor is unchanged by a selection**, which is why this is painted BEFORE the cursor's
/// block and why the bar is drawn either way. A row that is both keeps the cursor's block and
/// still carries the bar, so the two marks answer different questions rather than competing for
/// one cell — and after the inversion the louder mark belongs to the cursor, which is the one a
/// reader is moving.
fn paint_selected(gutter: Rect, row: Rect, buf: &mut Buffer) {
    buf.set_style(row, Style::default().bg(tokens::selection_bg()));
    buf.set_string(
        gutter.x,
        gutter.y,
        tokens::SELECTED_BAR.to_string(),
        Style::default().fg(tokens::selected()),
    );
}

impl Widget for ListPane {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // A list without its number column is malformed — the first column is the number, and
        // nothing else has anywhere to stand.
        if area.is_empty() || area.height == 0 || self.columns.is_empty() {
            return;
        }
        // The first column of the given area is the SELECTION GUTTER and belongs to no column
        // of the table: the caller hands over one column of its own margin
        // ([`crate::widgets::chrome::with_gutter`]), so a selection bar costs the table nothing.
        // Chris removed a permanent two-column data-cursor gutter on 2026-09-07 to regain the
        // width; this must not quietly hand the bill back.
        let table = Rect {
            x: area.x + GUTTER,
            width: area.width.saturating_sub(GUTTER),
            ..area
        };
        let columns = crate::panes::cell::table::laid_out(
            Rect { height: 1, ..table },
            &self.columns.iter().collect::<Vec<_>>(),
        );
        self.header(table, &columns, buf);

        if self.rows.is_empty() {
            Paragraph::new(Line::from(Span::styled(EMPTY, tokens::faint_style())))
                .render(line(table, 1), buf);
            return;
        }

        let gutter = Rect {
            width: GUTTER,
            ..area
        };
        let body = area.height.saturating_sub(1) as usize;
        let offset = self.window(body);
        let at = self.at();

        for (i, index) in (offset..self.lines()).take(body).enumerate() {
            let row = line(table, 1 + i as u16);
            // The selection first, the cursor over it: the cursor is unchanged by a selection
            // (ruling 10), so where both land on one row the cursor's fill is what shows.
            if self.is_selected(index) {
                paint_selected(line(gutter, 1 + i as u16), row, buf);
            }
            if index == at {
                paint_cursor(row, buf);
            }
            match self.rows.get(index) {
                // Indexed rather than zipped by reference: a row may carry fewer cells than the
                // list has columns, and the column its value belongs to is its POSITION.
                Some(cells) => {
                    // The number column first: the row's position, computed from the displayed
                    // index rather than read off the row, and muted — it is there to be referred
                    // to, not to be read down the page. The cursor row wears it too.
                    Paragraph::new(Line::from(Span::styled(
                        grouped(self.number_at(index, at)),
                        tokens::muted_style(),
                    )))
                    .alignment(self.columns[0].align.to_ratatui())
                    .render(
                        Rect {
                            y: row.y,
                            height: 1,
                            ..columns[0]
                        },
                        buf,
                    );

                    // Then the data columns, drawn from the row's cells — which sit one place
                    // left of the column they answer to, the number column taking the first.
                    for (n, (cell, column)) in cells.iter().zip(&self.columns[1..]).enumerate() {
                        Paragraph::new(Line::from(cell.spans(columns[n + 1].width, column.elide, false)))
                            .alignment(column.align.to_ratatui())
                            .render(
                                Rect {
                                    y: row.y,
                                    height: 1,
                                    ..columns[n + 1]
                                },
                                buf,
                            );
                    }
                    // The block's content, last — see [`invert_cursor`] for why it cannot go
                    // down with the fill. A selected row is NOT inverted: after ruling 7 the
                    // selection is the quiet mark of the two, and its tint sits under the row's
                    // own colours exactly as the cursor's tint used to.
                    if index == at {
                        invert_cursor(row, buf);
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

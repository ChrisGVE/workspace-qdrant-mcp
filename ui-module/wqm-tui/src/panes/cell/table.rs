//! The table inside a cell: its columns, its values, and how much of it fits.
//!
//! Split out of `super` when the data cursor took the file past its size limit. The division is
//! the one the module docs already draw: a [`CellPane`] is a *projection* — a heading over a
//! table — and everything about the table itself lives here, so the pane's own file is about
//! composition rather than about column arithmetic.
//!
//! [`CellPane`]: super::CellPane

use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use super::sort::{compare, grown, Direction, Sort};
use super::value::{Cell, Elide};
use crate::tokens;

/// What a cell says when its projection is empty. v0.1's own words, kept: an empty cell that
/// said nothing at all would be indistinguishable from one that failed to load.
pub const EMPTY: &str = "No data";

/// Columns between one column of the table and the next.
///
/// Shared with [`crate::panes::list`] rather than restated there: it is also the gap a sort
/// mark borrows ([`super::sort::grown`]), so two tables holding two copies of it would be two
/// tables that disagreed about whether a mark fits.
pub(crate) const COLUMN_GAP: u16 = 1;

// # There is no marker gutter here, and that is a ruling with a date on it
//
// This table used to reserve two columns at the head of every row for a `▸` — always present,
// so that a cell taking focus did not shift its own names sideways. Chris removed it
// (2026-09-07): *"This gives us the ability to remove the indent under the title and thus
// regaining two columns"*. A Dashboard cell is 59 columns wide and its flex column is the one
// carrying the thing a reader is trying to finish reading, so two columns spent on a glyph that
// is blank on five cells out of six is two columns spent badly.
//
// What is left is the tint. [`paint_cursor`] fills the cursor row across the cell's whole width
// and draws no glyph at all — the row itself is the mark. [`crate::panes::collections`] keeps
// its own `▸`: it is a full-width list where the gutter costs nothing and the marker is the
// only thing distinguishing its cursor from a selection.

/// Which edge a column's content is flush with.
///
/// Numbers right, words left — the only alignment rule this table has, and it exists so a
/// column of figures can be compared down its last digit rather than read one at a time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Align {
    Left,
    Right,
}

impl Align {
    pub(crate) const fn to_ratatui(self) -> Alignment {
        match self {
            Align::Left => Alignment::Left,
            Align::Right => Alignment::Right,
        }
    }
}

/// One column: what it is called, which way it sits, how wide it wants to be, and the letter
/// that sorts by it.
pub struct Column {
    pub title: &'static str,
    pub align: Align,
    pub width: Constraint,
    /// The key that sorts the table by this column, lit in the header while the table is
    /// sortable. `None` on a column nothing sorts by.
    ///
    /// The letter is found in the TITLE ([`crate::widgets::chrome::keyed_spans`]) rather than
    /// carried as an index, so a renamed column cannot leave the lit letter pointing at the
    /// wrong character — and a key that is not in its own title lights nothing, which is the
    /// honest frame for a key nobody can see to press.
    pub sort_key: Option<char>,
    /// Which end of an over-long value this column drops. See [`Elide`]: it is a fact about
    /// what the column holds, so every value in it answers the question the same way.
    pub elide: Elide,
    /// How long this column survives a narrow cell. When the flex column would fall below
    /// [`CellTable::min_flex`], the fixed column with the LOWEST priority is dropped first
    /// (ties: the rightmost first), so a higher number outlasts a lower one. Default 0.
    ///
    /// The flex column is the row's identity and is never dropped, so its priority is never
    /// consulted — it is the thing every drop is spent to keep whole.
    pub priority: u8,
}

impl Column {
    /// A text column that takes what is left. At most one per table, or they share the slack.
    ///
    /// The flex column is the row's identity — it is never dropped when the cell narrows; the
    /// fixed columns are, lowest [`Column::priority`] first. See [`CellTable::fitted`].
    pub fn flex(title: &'static str) -> Self {
        Self {
            title,
            align: Align::Left,
            width: Constraint::Fill(1),
            sort_key: None,
            elide: Elide::Right,
            priority: 0,
        }
    }

    /// A figure column, right-aligned and exactly as wide as its widest possible value.
    pub fn number(title: &'static str, width: u16) -> Self {
        Self {
            title,
            align: Align::Right,
            width: Constraint::Length(width),
            sort_key: None,
            elide: Elide::Right,
            priority: 0,
        }
    }

    /// A fixed-width text column.
    pub fn text(title: &'static str, width: u16) -> Self {
        Self {
            title,
            align: Align::Left,
            width: Constraint::Length(width),
            sort_key: None,
            elide: Elide::Right,
            priority: 0,
        }
    }

    /// How long this column survives a narrow cell — see [`Column::priority`].
    pub fn priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Shorten this column's values from the LEFT — the file-path rule, and so far the Queue
    /// tab's `Object` column alone. See [`crate::panes::cell::value::fit_left`].
    pub fn elide_left(mut self) -> Self {
        self.elide = Elide::Left;
        self
    }

    /// The letter that sorts by this column (Chris, 2026-09-07): *"we highlight (using the
    /// selection color) one letter of the column name, non-ambiguous with another of the 5
    /// sections; when pressing on that letter the user can sort by the column: first press
    /// ascending, second press descending"*.
    ///
    /// *Non-ambiguous* is a claim about a whole screen, not about a column, so it is checked
    /// where the screen is — `views::dashboard::tests::sort` asserts every cell's keys are
    /// unique within the cell and disjoint from
    /// [`crate::views::dashboard::DASHBOARD_BOUND_KEYS`]. A column cannot know what else the
    /// screen has bound, so it does not pretend to.
    pub fn sort(mut self, key: char) -> Self {
        self.sort_key = Some(key);
        self
    }
}

impl Column {
    /// The column's width when it has one, and [`None`] when it takes what is left.
    pub fn fixed(&self) -> Option<u16> {
        match self.width {
            Constraint::Length(n) => Some(n),
            _ => None,
        }
    }
}

/// The table inside a cell.
pub struct CellTable {
    columns: Vec<Column>,
    rows: Vec<Vec<Cell>>,
    offset: usize,
    /// Which drawn row carries the data cursor, counted from the first row on screen. `None`
    /// on every cell but the focused one — §3 puts one cursor on a screen, not one per list.
    cursor: Option<usize>,
    /// Whether the sort keys are OFFERED — lit in the header, so the user can see what to
    /// press. Taken from the pane, which reads it off the screen: a cell is sortable when it
    /// is the live one AND holds more than one row, because a lit letter is a promise that the
    /// key does something and there is nothing to reorder in a list of one.
    sortable: bool,
    /// Whether this cell has RECEDED — some other cell is the live one, so this one's header
    /// drops to the muted rung. Taken from the pane, which reads the screen's [`Attention`] off
    /// the view: the header follows the cell's focus exactly as the heading above it does.
    ///
    /// [`Attention`]: crate::widgets::chrome::Attention
    receded: bool,
    /// How the table is sorted, if it is. Independent of `sortable`: the mark stays on the
    /// column while the cell is read, and the offer only stands while it is live.
    sort: Option<Sort>,
    /// The fewest cells the flex column is allowed to end up with. Below this the table drops
    /// its lowest-priority fixed column rather than starve the row's identity — see
    /// [`CellTable::fitted`]. Twelve (Chris, 2026-09-07): a name narrower than that is a name
    /// nobody can finish reading, and dropping a whole column is the cheaper loss.
    min_flex: u16,
}

impl CellTable {
    pub fn new(columns: Vec<Column>, rows: Vec<Vec<Cell>>) -> Self {
        Self {
            columns,
            rows,
            offset: 0,
            cursor: None,
            sortable: false,
            receded: false,
            sort: None,
            min_flex: 12,
        }
    }

    /// The floor the flex column keeps, in cells. See the field.
    pub fn min_flex(mut self, min_flex: u16) -> Self {
        self.min_flex = min_flex;
        self
    }

    /// Whether this table OFFERS its sort keys. See the field: the pane decides, from the
    /// screen's own facts, and a table that decided for itself would be a second copy of them.
    pub fn sortable(mut self, sortable: bool) -> Self {
        self.sortable = sortable;
        self
    }

    /// Whether this cell has receded behind the live one. See the field.
    pub fn receded(mut self, receded: bool) -> Self {
        self.receded = receded;
        self
    }

    /// Sort the rows by one column, and remember which one for the header's mark.
    ///
    /// The rows are reordered HERE rather than at render time, so [`CellTable::rows`] answers
    /// with what the cell will actually draw — a guard reading the data and a reader reading
    /// the screen are then looking at one order rather than at two that ought to agree.
    ///
    /// Stable, so rows that compare equal keep the order the projection handed them in: the
    /// eight rules all carry `0/0/0`, and sorting by `Queue` must not shuffle them into an
    /// order nothing chose.
    pub fn sorted(mut self, sort: Sort) -> Self {
        let column = sort.column;
        self.rows.sort_by(|a, b| {
            let ordering = match (a.get(column), b.get(column)) {
                (Some(a), Some(b)) => compare(a, b),
                // A row too short to reach the sorted column has no value to compare, and
                // sorts before one that has. Only reachable from a malformed fixture.
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            };
            match sort.direction {
                Direction::Asc => ordering,
                Direction::Desc => ordering.reverse(),
            }
        });
        self.sort = Some(sort);
        self
    }

    /// Put the data cursor on a row. Taken from the pane rather than decided here: which cell
    /// is live is the screen's fact, and a table that chose its own would be a second copy of it.
    pub fn cursor(mut self, cursor: Option<usize>) -> Self {
        self.cursor = cursor;
        self
    }

    /// Scroll within the cell. The grid does not move; see the module docs.
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// The columns, for a guard that asks whether the data fits them.
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// The rows, likewise.
    pub fn rows(&self) -> &[Vec<Cell>] {
        &self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Which columns survive the fit, as indices into [`CellTable::columns`].
    ///
    /// The flex column is the row's identity and is never dropped. The fixed columns are
    /// dropped — lowest [`Column::priority`] first, the rightmost first on a tie — until the
    /// flex column would get at least [`CellTable::min_flex`] cells, or until no fixed column
    /// remains. The surviving columns keep their widths: a dropped column takes its width and
    /// its gap with it, and the slack goes to the flex column, nowhere else.
    ///
    /// A table with no flex column returns every column: there is no identity to protect, so
    /// nothing gives.
    pub(super) fn fitted(&self, width: u16) -> Vec<usize> {
        let mut active: Vec<usize> = (0..self.columns.len()).collect();
        let Some(flex) = self.columns.iter().position(|c| c.fixed().is_none()) else {
            return active;
        };
        while active.len() > 1 {
            let fixed: u16 = active
                .iter()
                .filter(|&&at| at != flex)
                .filter_map(|&at| self.columns[at].fixed())
                .sum();
            let gaps = (active.len() as u16 - 1) * COLUMN_GAP;
            if width.saturating_sub(fixed).saturating_sub(gaps) >= self.min_flex {
                break;
            }
            // Lowest priority first; `Reverse` makes the rightmost of a tie the least, which is
            // what `min_by_key` picks.
            let drop = active
                .iter()
                .copied()
                .filter(|&at| at != flex)
                .min_by_key(|&at| (self.columns[at].priority, std::cmp::Reverse(at)))
                .expect("a fixed column remains to drop");
            active.retain(|&at| at != drop);
        }
        active
    }

    /// How many data rows fit, and how many are left over, given `height` rows for the header
    /// and the body together.
    ///
    /// Returned as a pair rather than computed twice, because the renderer and the tail have to
    /// agree about it exactly — a tail saying `… 22 more` above a body showing one row more
    /// than it counted is a lie the eye cannot catch.
    pub(super) fn budget(&self, height: u16) -> (usize, usize) {
        let body = height.saturating_sub(1) as usize;
        let remaining = self.rows.len().saturating_sub(self.offset);
        if remaining <= body {
            return (remaining, 0);
        }
        // One line goes to the tail, so one fewer row is drawn than would otherwise fit.
        let shown = body.saturating_sub(1);
        (shown, remaining - shown)
    }

    fn header(&self, body: Rect, active: &[usize], cells: &[Rect], buf: &mut Buffer) {
        for (&at, &area) in active.iter().zip(cells) {
            let column = &self.columns[at];
            let mark = self.sort.filter(|sort| sort.column == at);
            let spans = self.header_spans(column, mark);
            let needed: u16 = spans
                .iter()
                .map(|span| span.content.chars().count() as u16)
                .sum();
            // **Only the MARK may take room beyond the column.** A title too long for its own
            // column is clipped, exactly as it was before there were marks — a header that grew
            // to fit its own title would reach across the next column on every screen too
            // narrow for it, sorted or not. The narrowest screens no longer clip `Name` to
            // `Na`: [`CellTable::fitted`] drops a fixed column before the flex column falls
            // below [`CellTable::min_flex`].
            let at_rect = if mark.is_some() { grown(area, body, needed) } else { area };
            Paragraph::new(Line::from(spans))
                .alignment(column.align.to_ratatui())
                .render(at_rect, buf);
        }
    }

    /// One column header: its title in the body foreground with [`Modifier::ITALIC`], the sort
    /// key lit if the table is offering its keys, and the sort mark if this is the column the
    /// table is sorted by.
    ///
    /// The header wears the SAME foreground the data under it wears and distinguishes itself
    /// with italics (Chris, 2026-09-07): a column header on this screen is not bold, and a key
    /// that arrived bold would read as a heading rather than as a letter to press. The lit
    /// letter changes the HUE to [`tokens::accent`] and adds weight — **bold**, like a jump
    /// digit — so one letter of an italic header reads as the key to press rather than as a gap
    /// in the label. The hue is the unclaimed `accent` field (§10), not the reserved selector:
    /// a sort key is a hint, not a selection.
    ///
    /// In a cell that has receded behind the live one the title drops to the muted rung — the
    /// identical rule the heading above it follows — and the sort key is not offered anyway,
    /// since only the live cell offers its keys. A receded header is muted italic.
    ///
    /// The mark is muted and un-italic: it says which column is sorted, and it is never the
    /// thing being read. It is appended with no space between it and the title — see [`grown`]
    /// for why the space is what a five-column `Files` cannot afford.
    fn header_spans(&self, column: &Column, mark: Option<Sort>) -> Vec<Span<'static>> {
        let text = if self.receded {
            tokens::muted()
        } else {
            tokens::header()
        };
        let rest = Style::default().fg(text).add_modifier(Modifier::ITALIC);
        let key = self.sortable.then_some(column.sort_key).flatten();
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
}

/// Mark `row` as the one the data cursor is on: the tint across its whole width, and nothing
/// else at all.
///
/// The tint goes down FIRST, before any span of the row is drawn. Ratatui styles patch rather
/// than replace, so every span drawn over it afterwards keeps its own hue and inherits this
/// background — which is what makes one `set_style` enough for a row of many spans.
fn paint_cursor(row: Rect, buf: &mut Buffer) {
    buf.set_style(row, Style::default().bg(tokens::cursor_bg()));
}

impl Widget for CellTable {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() || area.height == 0 {
            return;
        }
        // The table starts at the cell's own first column — no marker gutter, so the column
        // header and every row line up under the first character of the heading above them.
        let body = area;
        let active = self.fitted(body.width);
        let constraints: Vec<Constraint> =
            active.iter().map(|&at| self.columns[at].width).collect();
        let columns = Layout::horizontal(constraints)
            .spacing(COLUMN_GAP)
            .split(Rect { height: 1, ..body });

        self.header(body, &active, &columns, buf);

        let text_row = |offset: u16, text: String| {
            (
                Paragraph::new(Line::from(Span::styled(text, tokens::faint_style()))),
                Rect {
                    y: body.y + offset,
                    height: 1,
                    ..body
                },
            )
        };

        if self.rows.is_empty() {
            let (paragraph, at) = text_row(1, EMPTY.to_string());
            paragraph.render(at, buf);
            return;
        }

        let (shown, hidden) = self.budget(area.height);
        for (i, row) in self.rows.iter().skip(self.offset).take(shown).enumerate() {
            let y = area.y + 1 + i as u16;

            if self.cursor == Some(i) {
                paint_cursor(Rect { y, height: 1, ..area }, buf);
            }

            // Indexed rather than zipped by reference: a row may carry fewer cells than the
            // table has columns, and the column its value belongs to is its POSITION. A dropped
            // column is skipped by its index — `active` names the survivors.
            for (n, &at) in active.iter().enumerate() {
                let Some(cell) = row.get(at) else { continue };
                let column = &self.columns[at];
                Paragraph::new(Line::from(cell.spans(columns[n].width, column.elide)))
                    .alignment(column.align.to_ratatui())
                    .render(
                        Rect {
                            y,
                            height: 1,
                            ..columns[n]
                        },
                        buf,
                    );
            }
        }

        if hidden > 0 {
            let (paragraph, at) = text_row(1 + shown as u16, format!("… {hidden} more"));
            paragraph.render(at, buf);
        }
    }
}

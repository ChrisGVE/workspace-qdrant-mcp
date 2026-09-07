//! One cell of the Dashboard grid: a heading, a column header, and rows that scroll inside it.
//!
//! §18's reading of the Dashboard is that **a cell is a projection** — a narrow view of
//! something a whole tab is dedicated to — and that *"every cell agrees with its drill-down"*.
//! A projection is the same shape whatever it projects, which is why there is one pane here
//! and not six: six near-identical files would be six chances for the Projects cell and the
//! Rules cell to disagree about what a column header looks like.
//!
//! The six data shapes live in [`crate::views::dashboard::frames`], each a function that maps
//! its own fixture onto a [`CellTable`]. What differs between cells is *what the columns are*,
//! and that is data.
//!
//! # The grid keeps its shape; the cell scrolls
//!
//! §18 again: cells scroll **inside** themselves. A cell that grew to fit its rows would move
//! every cell below it, and the Dashboard's whole promise is that you learn where things are.
//! So [`CellTable`] takes an `offset`, draws what fits, and spends its last line saying how
//! much it did not draw. That tail is the answer to "what does a full cell look like" — a
//! question that can only be answered by a frame, which is why the pantry has one.
//!
//! # The count in the heading is the TOTAL, not what is visible
//!
//! v0.1 writes `Projects (29)` and it means twenty-nine projects, not twenty-nine rows on
//! screen. That is kept: the heading is the size of the projection, and the overflow tail is
//! what did not fit. A heading counting only visible rows would change when the terminal was
//! resized, which makes it a fact about the window rather than about the workspace.

use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::format::{count_span, grouped};
use crate::tokens;
use crate::widgets::chrome::{Attention, FocusMark, ZoneHeading};

#[cfg(feature = "tui-pantry")]
pub mod ingredient;
#[cfg(test)]
mod tests;

/// `text` cut to `width`, ending in `…` when it did not fit.
///
/// Counted in CHARACTERS — the mistake this crate has already made once — and the ellipsis
/// takes one of them, so the result is never wider than the column it was measured against.
fn fit(text: &str, width: u16) -> String {
    let width = width as usize;
    if text.chars().count() <= width || width == 0 {
        return text.to_string();
    }
    let mut out: String = text.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// What a cell says when its projection is empty. v0.1's own words, kept: an empty cell that
/// said nothing at all would be indistinguishable from one that failed to load.
pub const EMPTY: &str = "No data";

/// Columns between one column of the table and the next.
const COLUMN_GAP: u16 = 1;

/// The marker column at the head of every row, cursor or not.
///
/// **Always present**, which is [`crate::panes::collections`]'s own answer to the same question
/// and the reason this is a constant rather than a conditional inset: a gutter that appeared
/// only on the focused cell would shift every name two columns to the right the moment a cell
/// took focus, and the point of the cursor is to say *which row*, not to move the table.
pub(crate) const GUTTER: u16 = 2;

/// The mark on the row the cursor is on. Same glyph and same rung as
/// [`crate::panes::collections`] — one data cursor in this crate, not two.
const CURSOR_MARK: &str = "▸ ";

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
    const fn to_ratatui(self) -> Alignment {
        match self {
            Align::Left => Alignment::Left,
            Align::Right => Alignment::Right,
        }
    }
}

/// One column: what it is called, which way it sits, and how wide it wants to be.
pub struct Column {
    pub title: &'static str,
    pub align: Align,
    pub width: Constraint,
}

impl Column {
    /// A text column that takes what is left. At most one per table, or they share the slack.
    pub fn flex(title: &'static str) -> Self {
        Self {
            title,
            align: Align::Left,
            width: Constraint::Fill(1),
        }
    }

    /// A figure column, right-aligned and exactly as wide as its widest possible value.
    pub fn number(title: &'static str, width: u16) -> Self {
        Self {
            title,
            align: Align::Right,
            width: Constraint::Length(width),
        }
    }

    /// A fixed-width text column.
    pub fn text(title: &'static str, width: u16) -> Self {
        Self {
            title,
            align: Align::Left,
            width: Constraint::Length(width),
        }
    }
}

/// One value in a row.
pub enum Cell {
    Text(String),
    Num(u64),
    /// The queue triple v0.1 writes as `2'635/0/0` — three figures that are three different
    /// facts, so they carry three different hues rather than being one string.
    Queue {
        pending: u64,
        in_flight: u64,
        failed: u64,
    },
}

impl Cell {
    /// The spans this value is drawn with, for a column `width` columns wide.
    ///
    /// A queue triple is several spans on purpose: its three numbers mean waiting, moving and
    /// lost, and a single-coloured `2'635/0/0` would throw away the only thing that
    /// distinguishes them.
    ///
    /// **Text elides to a shorter name; a figure that does not fit is replaced outright.** An
    /// elided name is still recognisable and the `…` says it was shortened. There is no such
    /// thing as a shortened number — `11'236` cut to five cells is `11'23`, a different value
    /// with nothing to mark it — so a figure too wide for its column is drawn as `…` and the
    /// column width is treated as the defect it is.
    ///
    /// The zero rule is the queue triple's alone: `count_span` mutes a zero because *no work
    /// waiting* is not news. A plain figure column keeps its zeros at the normal rung — v0.1's
    /// `Pts` column was all zeros, and muting them would have made the column disappear rather
    /// than recede. (That column has since been dropped altogether, 2026-09-07: a field that is
    /// always zero is better removed than styled.)
    fn spans(&self, width: u16) -> Vec<Span<'static>> {
        match self {
            Cell::Text(text) => vec![Span::styled(fit(text, width), tokens::normal_style())],
            // A figure that does not fit becomes `…` — NEVER a clipped one. Right-aligning
            // `11'236` into five cells renders `11'23`, which is not a truncated number, it is
            // a DIFFERENT number, displayed with no mark to say so. `…` says "there is a value
            // here and it did not fit", which is the only honest thing a too-narrow column can
            // say. The real fix is always the column width, and
            // `views::dashboard::tests` guards that every frame's figure columns are wide
            // enough — this is the net under that guard, not a substitute for it.
            Cell::Num(value) => {
                let text = grouped(*value);
                let text = if text.chars().count() > width as usize {
                    "…".to_string()
                } else {
                    text
                };
                vec![Span::styled(text, tokens::normal_style())]
            }
            Cell::Queue {
                pending,
                in_flight,
                failed,
            } => vec![
                count_span(*pending, tokens::degraded),
                Span::styled("/", tokens::muted_style()),
                count_span(*in_flight, tokens::in_flight),
                Span::styled("/", tokens::muted_style()),
                count_span(*failed, tokens::offline),
            ],
        }
    }
}

impl Cell {
    /// How wide this value wants to be, before any column has a say.
    ///
    /// Exists so a guard can ask "does every figure fit its column" of the DATA rather than of
    /// a rendered screen. Read off the render, a figure's ellipsis is indistinguishable from a
    /// name's — both are `…` — and the guard that matters is about numbers only.
    pub fn natural_width(&self) -> usize {
        match self {
            Cell::Text(text) => text.chars().count(),
            Cell::Num(value) => grouped(*value).chars().count(),
            Cell::Queue {
                pending,
                in_flight,
                failed,
            } => {
                grouped(*pending).chars().count()
                    + grouped(*in_flight).chars().count()
                    + grouped(*failed).chars().count()
                    + 2
            }
        }
    }

    /// Whether this value is a figure — the kind that must never be shortened.
    pub fn is_figure(&self) -> bool {
        !matches!(self, Cell::Text(_))
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
}

impl CellTable {
    pub fn new(columns: Vec<Column>, rows: Vec<Vec<Cell>>) -> Self {
        Self {
            columns,
            rows,
            offset: 0,
            cursor: None,
        }
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

    /// How many data rows fit, and how many are left over, given `height` rows for the header
    /// and the body together.
    ///
    /// Returned as a pair rather than computed twice, because the renderer and the tail have to
    /// agree about it exactly — a tail saying `… 22 more` above a body showing one row more
    /// than it counted is a lie the eye cannot catch.
    fn budget(&self, height: u16) -> (usize, usize) {
        let body = height.saturating_sub(1) as usize;
        let remaining = self.rows.len().saturating_sub(self.offset);
        if remaining <= body {
            return (remaining, 0);
        }
        // One line goes to the tail, so one fewer row is drawn than would otherwise fit.
        let shown = body.saturating_sub(1);
        (shown, remaining - shown)
    }

    fn header(&self, cells: &[Rect], buf: &mut Buffer) {
        for (column, area) in self.columns.iter().zip(cells) {
            Paragraph::new(Line::from(Span::styled(column.title, Style::default().fg(tokens::header()))))
                .alignment(column.align.to_ratatui())
                .render(*area, buf);
        }
    }
}

impl Widget for CellTable {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() || area.height == 0 || area.width <= GUTTER {
            return;
        }
        // Everything but the marker column is drawn in `body`; the marker column is the
        // leftmost [`GUTTER`] columns of `area`, and stays empty except on the cursor row.
        let body = Rect {
            x: area.x + GUTTER,
            width: area.width - GUTTER,
            ..area
        };
        let constraints: Vec<Constraint> = self.columns.iter().map(|c| c.width).collect();
        let columns = Layout::horizontal(constraints)
            .spacing(COLUMN_GAP)
            .split(Rect { height: 1, ..body });

        self.header(&columns, buf);

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

            // The tint goes down FIRST and across the cell's whole width, marker column
            // included. Ratatui styles patch rather than replace, so every span drawn over it
            // keeps its own hue and inherits this background.
            if self.cursor == Some(i) {
                buf.set_style(
                    Rect {
                        y,
                        height: 1,
                        ..area
                    },
                    Style::default().bg(tokens::cursor_bg()),
                );
                Paragraph::new(Line::from(Span::styled(
                    CURSOR_MARK,
                    Style::default().fg(tokens::cursor_mark()),
                )))
                .render(
                    Rect {
                        y,
                        height: 1,
                        width: GUTTER,
                        ..area
                    },
                    buf,
                );
            }

            // Indexed rather than zipped by reference: a row may carry fewer cells than the
            // table has columns, and the column its value belongs to is its POSITION.
            for (at, (cell, column)) in row.iter().zip(&self.columns).enumerate() {
                Paragraph::new(Line::from(cell.spans(columns[at].width)))
                    .alignment(column.align.to_ratatui())
                    .render(
                        Rect {
                            y,
                            height: 1,
                            ..columns[at]
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

/// A cell of the grid: [`ZoneHeading`] over a [`CellTable`].
pub struct CellPane {
    title: String,
    /// The size of the projection, shown in parentheses as v0.1 does. `None` for a cell that
    /// counts nothing meaningful — `Last Errors` is a list, not a population.
    count: Option<usize>,
    table: CellTable,
    zone: usize,
    attention: Attention,
    /// The key that focuses this cell, accented in the heading. The **view** owns which key
    /// that is — a cell that named its own would be a second copy of the screen's key table.
    hotkey: Option<char>,
    modal: bool,
}

impl CellPane {
    pub fn new(title: impl Into<String>, count: Option<usize>, table: CellTable) -> Self {
        Self {
            title: title.into(),
            count,
            table,
            zone: 0,
            attention: Attention::None,
            hotkey: None,
            modal: false,
        }
    }

    /// `zone` is this cell's index in the screen's zone order; `attention` is the screen-level
    /// answer to which zone is live — §16: *the view owns which pane has focus*.
    pub fn placed(mut self, zone: usize, attention: Attention) -> Self {
        self.zone = zone;
        self.attention = attention;
        self
    }

    /// The key that focuses this cell. Passed in rather than derived here for the same reason
    /// `placed` takes the zone: the grid's key order is the view's fact, and the view reads it
    /// from one table ([`crate::views::dashboard::FOCUS_KEYS`]).
    pub fn hotkey(mut self, key: char) -> Self {
        self.hotkey = Some(key);
        self
    }

    /// Whether a modal owns the input — screen-level, and passed straight through to the
    /// heading, which mutes its key letter under one.
    pub fn under_modal(mut self, modal: bool) -> Self {
        self.modal = modal;
        self
    }

    /// The table this cell draws, so a guard can inspect the data rather than the pixels.
    pub fn table(&self) -> &CellTable {
        &self.table
    }

    fn heading(&self) -> String {
        match self.count {
            Some(count) => format!("{} ({count})", self.title),
            None => self.title.clone(),
        }
    }
}

impl CellPane {
    /// Whether the screen says THIS cell is the live one.
    ///
    /// Asked once and spent twice — on the heading's block and on the row that takes the data
    /// cursor. Two readings of `attention` would be two chances for a cell to wear the block
    /// and put the cursor somewhere else.
    fn is_live(&self) -> bool {
        matches!(self.attention, Attention::Zone(zone) if zone == self.zone)
    }
}

impl Widget for CellPane {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }
        // [`FocusMark::Block`] is set HERE rather than passed in by the view, because a
        // `CellPane` *is* a Dashboard grid cell — there is no other kind. Letting the view
        // choose would make "a grid cell wearing the `▌` bar" a frame someone could build, and
        // Chris ruled the block for these cells specifically (2026-09-07).
        let mut heading = ZoneHeading::new(self.heading(), self.zone, self.attention)
            .focus_mark(FocusMark::Block)
            .under_modal(self.modal);
        if let Some(key) = self.hotkey {
            heading = heading.hotkey(key);
        }
        let cursor = self.is_live().then_some(0);
        heading.render(crate::views::top::row(area, 0), buf);
        if area.height > 1 {
            self.table.cursor(cursor).render(
                Rect {
                    y: area.y + 1,
                    height: area.height - 1,
                    ..area
                },
                buf,
            );
        }
    }
}

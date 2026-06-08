//! Reusable scrollable cell widget for the dashboard grid.
//!
//! Each cell displays a title (first letter colored as shortcut), column
//! headers, and scrollable data rows with auto-sized columns.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

// ---------------------------------------------------------------------------
// Column definition
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
}

#[derive(Debug, Clone)]
pub struct ColDef {
    pub header: &'static str,
    pub align: Align,
    pub flex: bool,
}

// ---------------------------------------------------------------------------
// Cell data
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum CellValue {
    Plain(String),
    Colored(Vec<(String, Color)>),
}

impl CellValue {
    pub fn display_width(&self) -> usize {
        match self {
            CellValue::Plain(s) => s.chars().count(),
            CellValue::Colored(parts) => parts.iter().map(|(s, _)| s.chars().count()).sum(),
        }
    }
}

pub type CellRow = Vec<CellValue>;

// ---------------------------------------------------------------------------
// Queue format helper
// ---------------------------------------------------------------------------

pub fn queue_cell(pending: i64, in_progress: i64, failed: i64) -> CellValue {
    if pending == 0 && in_progress == 0 && failed == 0 {
        return CellValue::Plain("0/0/0".into());
    }
    CellValue::Colored(vec![
        (pending.to_string(), Color::Yellow),
        ("/".into(), Color::DarkGray),
        (in_progress.to_string(), Color::Blue),
        ("/".into(), Color::DarkGray),
        (failed.to_string(), Color::Red),
    ])
}

// ---------------------------------------------------------------------------
// Scrollable cell state
// ---------------------------------------------------------------------------

use std::cell::Cell;

#[derive(Debug)]
pub struct ScrollableCell {
    pub selected: usize,
    /// Scroll offset — updated during draw via interior mutability.
    scroll_offset: Cell<usize>,
}

impl ScrollableCell {
    pub fn new() -> Self {
        Self {
            selected: 0,
            scroll_offset: Cell::new(0),
        }
    }

    pub fn scroll_offset(&self) -> usize {
        self.scroll_offset.get()
    }

    pub fn select_next(&mut self, row_count: usize) {
        if row_count > 0 && self.selected < row_count - 1 {
            self.selected += 1;
        }
    }

    pub fn select_prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Draw a cell with title (first letter = shortcut color), column headers,
/// and auto-sized scrollable rows.
#[allow(clippy::too_many_arguments)]
pub fn draw_cell(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    count: Option<usize>,
    shortcut: Option<char>,
    cols: &[ColDef],
    rows: &[CellRow],
    cell_state: &ScrollableCell,
    focused: bool,
) {
    if area.height < 2 || area.width < 4 {
        return;
    }

    // --- Title line: first letter colored, rest bold white ---
    let title_spans = build_title_spans(title, count, shortcut, focused, !rows.is_empty());
    frame.render_widget(
        Paragraph::new(Line::from(title_spans)),
        Rect::new(area.x, area.y, area.width, 1),
    );

    if area.height < 3 {
        return;
    }

    // --- Column headers ---
    let usable_width = area.width.saturating_sub(2) as usize;
    let col_widths = compute_col_widths(cols, rows, usable_width);
    let header_line = render_header(cols, &col_widths, focused);
    frame.render_widget(
        Paragraph::new(header_line),
        Rect::new(area.x, area.y + 1, area.width, 1),
    );

    let data_y = area.y + 2;
    let data_height = area.height.saturating_sub(2) as usize;

    if rows.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                " No data",
                Style::default().fg(Color::DarkGray),
            ))),
            Rect::new(area.x, data_y, area.width, 1),
        );
        return;
    }

    // --- Clamp scroll state so selection stays visible ---
    clamp_scroll(cell_state, rows.len(), data_height);

    // --- Render visible rows ---
    let offset = cell_state.scroll_offset();
    for (i, row) in rows.iter().enumerate().skip(offset).take(data_height) {
        let y = data_y + (i - offset) as u16;
        if y >= area.y + area.height {
            break;
        }
        let is_selected = focused && i == cell_state.selected;
        let row_line = render_row(cols, row, &col_widths, is_selected);
        frame.render_widget(
            Paragraph::new(row_line),
            Rect::new(area.x, y, area.width, 1),
        );
    }
}

/// Build title spans with shortcut letter highlighted in yellow.
/// When focused, the entire title turns yellow.
fn build_title_spans(
    title: &str,
    count: Option<usize>,
    shortcut: Option<char>,
    focused: bool,
    has_data: bool,
) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let show_shortcut = shortcut.is_some() && has_data;

    if focused {
        // Focused: entire title in yellow bold
        spans.push(Span::styled(
            format!(" {}", title),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    } else if show_shortcut {
        // Unfocused with shortcut: highlight the shortcut char in yellow
        let ch = shortcut.unwrap();
        if let Some(pos) = title.find(ch) {
            let before = &title[..pos];
            let after = &title[pos + ch.len_utf8()..];
            let normal = Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD);
            let highlight = Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD);
            spans.push(Span::styled(format!(" {}", before), normal));
            spans.push(Span::styled(ch.to_string(), highlight));
            spans.push(Span::styled(after.to_string(), normal));
        } else {
            spans.push(Span::styled(
                format!(" {}", title),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ));
        }
    } else {
        spans.push(Span::styled(
            format!(" {}", title),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));
    }

    if let Some(c) = count {
        spans.push(Span::styled(
            format!(" ({})", c),
            Style::default().fg(Color::Gray),
        ));
    }

    spans
}

/// Render column headers — yellow when focused, gray otherwise.
fn render_header(cols: &[ColDef], widths: &[usize], focused: bool) -> Line<'static> {
    let style = if focused {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::Gray)
    };
    let mut spans = vec![Span::styled(" ", style)];

    for (j, col) in cols.iter().enumerate() {
        let w = widths.get(j).copied().unwrap_or(0);
        let formatted = align_str(col.header, w, col.align);
        spans.push(Span::styled(formatted, style));
        if j < cols.len() - 1 {
            spans.push(Span::styled(" ", style));
        }
    }

    Line::from(spans)
}

/// Clamp scroll offset so the selected row is always visible.
/// Uses interior mutability (Cell) so it works with `&self` draw methods.
fn clamp_scroll(cell: &ScrollableCell, row_count: usize, visible: usize) {
    if row_count == 0 {
        cell.scroll_offset.set(0);
        return;
    }
    let selected = cell.selected.min(row_count - 1);
    let mut offset = cell.scroll_offset();
    if selected < offset {
        offset = selected;
    }
    if visible > 0 && selected >= offset + visible {
        offset = selected - visible + 1;
    }
    cell.scroll_offset.set(offset);
}

// ---------------------------------------------------------------------------
// Column width computation
// ---------------------------------------------------------------------------

fn compute_col_widths(cols: &[ColDef], rows: &[CellRow], usable_width: usize) -> Vec<usize> {
    let n = cols.len();
    let mut widths: Vec<usize> = cols.iter().map(|c| c.header.len()).collect();

    for row in rows {
        for (j, cell) in row.iter().enumerate() {
            if j < n {
                widths[j] = widths[j].max(cell.display_width());
            }
        }
    }

    let gaps = n.saturating_sub(1);
    let total_fixed: usize = widths.iter().sum::<usize>() + gaps;

    if total_fixed <= usable_width {
        let remaining = usable_width - total_fixed;
        let flex_cols: Vec<usize> = cols
            .iter()
            .enumerate()
            .filter(|(_, c)| c.flex)
            .map(|(i, _)| i)
            .collect();
        if !flex_cols.is_empty() {
            let extra_per = remaining / flex_cols.len();
            for &i in &flex_cols {
                widths[i] += extra_per;
            }
        }
    } else {
        shrink_columns(&mut widths, cols, usable_width, gaps);
    }

    widths
}

fn shrink_columns(widths: &mut [usize], cols: &[ColDef], usable: usize, gaps: usize) {
    let fixed_total: usize = cols
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.flex)
        .map(|(i, _)| widths[i])
        .sum::<usize>()
        + gaps;

    let available_for_flex = usable.saturating_sub(fixed_total);
    let flex_indices: Vec<usize> = cols
        .iter()
        .enumerate()
        .filter(|(_, c)| c.flex)
        .map(|(i, _)| i)
        .collect();

    if !flex_indices.is_empty() {
        let per_flex = available_for_flex / flex_indices.len();
        for &i in &flex_indices {
            widths[i] = per_flex.max(4);
        }
    }
}

// ---------------------------------------------------------------------------
// Row rendering
// ---------------------------------------------------------------------------

fn render_row(cols: &[ColDef], row: &CellRow, widths: &[usize], selected: bool) -> Line<'static> {
    let bg = if selected {
        crate::tui::theme::selected_row_style()
    } else {
        Style::default()
    };

    let mut spans = vec![Span::styled(" ", bg)];

    for (j, col) in cols.iter().enumerate() {
        let w = widths.get(j).copied().unwrap_or(0);
        let cell = row.get(j);

        let cell_spans = match cell {
            Some(CellValue::Plain(s)) => {
                let truncated = truncate_to(s, w);
                let formatted = align_str(&truncated, w, col.align);
                vec![Span::styled(formatted, bg)]
            }
            Some(CellValue::Colored(parts)) => {
                let total_width: usize = parts.iter().map(|(s, _)| s.chars().count()).sum();
                if col.align == Align::Right && total_width < w {
                    let padding = w - total_width;
                    let mut result = vec![Span::styled(" ".repeat(padding), bg)];
                    for (s, color) in parts {
                        result.push(Span::styled(s.clone(), bg.fg(*color)));
                    }
                    result
                } else {
                    parts
                        .iter()
                        .map(|(s, color)| Span::styled(s.clone(), bg.fg(*color)))
                        .collect()
                }
            }
            None => vec![Span::styled(" ".repeat(w), bg)],
        };

        spans.extend(cell_spans);
        if j < cols.len() - 1 {
            spans.push(Span::styled(" ", bg));
        }
    }

    Line::from(spans)
}

fn truncate_to(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        s.to_string()
    } else if max > 3 {
        let truncated: String = s.chars().take(max - 3).collect();
        format!("{}...", truncated)
    } else {
        s.chars().take(max).collect()
    }
}

fn align_str(s: &str, width: usize, align: Align) -> String {
    let len = s.chars().count();
    if len >= width {
        return truncate_to(s, width);
    }
    match align {
        Align::Left => format!("{}{}", s, " ".repeat(width - len)),
        Align::Right => format!("{}{}", " ".repeat(width - len), s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_cell_all_zero() {
        let c = queue_cell(0, 0, 0);
        assert!(matches!(c, CellValue::Plain(ref s) if s == "0/0/0"));
    }

    #[test]
    fn queue_cell_nonzero() {
        let c = queue_cell(3, 1, 2);
        assert!(matches!(c, CellValue::Colored(_)));
        assert_eq!(c.display_width(), 5);
    }

    #[test]
    fn truncate_to_short() {
        assert_eq!(truncate_to("hi", 10), "hi");
    }

    #[test]
    fn truncate_to_long() {
        assert_eq!(truncate_to("abcdefghij", 7), "abcd...");
    }

    #[test]
    fn align_left() {
        assert_eq!(align_str("ab", 5, Align::Left), "ab   ");
    }

    #[test]
    fn align_right() {
        assert_eq!(align_str("42", 5, Align::Right), "   42");
    }

    #[test]
    fn scrollable_cell_navigation() {
        let mut cell = ScrollableCell::new();
        cell.select_next(5);
        assert_eq!(cell.selected, 1);
        cell.select_prev();
        assert_eq!(cell.selected, 0);
        cell.select_prev();
        assert_eq!(cell.selected, 0);
    }

    #[test]
    fn clamp_scroll_adjusts_offset() {
        let mut cell = ScrollableCell::new();
        cell.selected = 20;
        clamp_scroll(&cell, 10, 5);
        // clamp_scroll sees selected=20, clamps to min(20,9)=9, offset = 9-5+1=5
        assert_eq!(cell.scroll_offset(), 5);
    }

    #[test]
    fn clamp_scroll_scrolls_up() {
        let cell = ScrollableCell::new();
        // selected = 0, offset = 0 by default; set offset high
        cell.scroll_offset.set(5);
        clamp_scroll(&cell, 10, 5);
        assert_eq!(cell.scroll_offset(), 0); // scrolled up to show selected=0
    }
}

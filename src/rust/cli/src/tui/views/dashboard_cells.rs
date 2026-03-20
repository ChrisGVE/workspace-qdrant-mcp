//! Reusable scrollable cell widget for the dashboard grid.
//!
//! Each cell displays a title with optional letter shortcut, column headers,
//! and scrollable data rows with auto-sized columns.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

// ---------------------------------------------------------------------------
// Column definition
// ---------------------------------------------------------------------------

/// Alignment for a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
}

/// Column descriptor for auto-sizing.
#[derive(Debug, Clone)]
pub struct ColDef {
    pub header: &'static str,
    pub align: Align,
    /// If true, this column gets any remaining space.
    pub flex: bool,
}

// ---------------------------------------------------------------------------
// Cell data
// ---------------------------------------------------------------------------

/// A single cell value — either plain text or colored spans.
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

/// A row of cell values matching the column definitions.
pub type CellRow = Vec<CellValue>;

// ---------------------------------------------------------------------------
// Queue format helper
// ---------------------------------------------------------------------------

/// Format queue counts as colored `x/y/z` cell value.
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

/// State for a scrollable cell in the dashboard grid.
#[derive(Debug)]
pub struct ScrollableCell {
    pub selected: usize,
    pub scroll_offset: usize,
}

impl ScrollableCell {
    pub fn new() -> Self {
        Self {
            selected: 0,
            scroll_offset: 0,
        }
    }

    pub fn select_next(&mut self, row_count: usize) {
        if row_count > 0 && self.selected < row_count - 1 {
            self.selected += 1;
        }
    }

    pub fn select_prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    /// Clamp selection and adjust scroll offset for the visible height.
    pub fn clamp(&mut self, row_count: usize, visible_rows: usize) {
        if row_count == 0 {
            self.selected = 0;
            self.scroll_offset = 0;
            return;
        }
        self.selected = self.selected.min(row_count - 1);
        // Ensure selected row is visible
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        }
        if visible_rows > 0 && self.selected >= self.scroll_offset + visible_rows {
            self.scroll_offset = self.selected - visible_rows + 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Draw a cell with title, column headers, and auto-sized rows.
///
/// Returns the number of visible data rows (for clamp calculations).
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
) -> usize {
    if area.height < 2 || area.width < 4 {
        return 0;
    }

    // --- Title line ---
    let title_style = Style::default()
        .fg(Color::White)
        .add_modifier(Modifier::BOLD);
    let mut title_spans = vec![Span::styled(format!(" {}", title), title_style)];

    if let Some(c) = count {
        title_spans.push(Span::styled(
            format!(" ({})", c),
            Style::default().fg(Color::DarkGray),
        ));
    }

    if let Some(ch) = shortcut {
        if !rows.is_empty() {
            let letter_style = if focused {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            title_spans.push(Span::styled(format!(" [{}]", ch), letter_style));
        }
    }

    let title_line = Line::from(title_spans);
    frame.render_widget(
        Paragraph::new(title_line),
        Rect::new(area.x, area.y, area.width, 1),
    );

    // Separator
    let sep = "─".repeat(area.width as usize);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            sep,
            Style::default().fg(Color::DarkGray),
        ))),
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
        return 0;
    }

    // --- Compute column widths ---
    let usable_width = area.width.saturating_sub(2) as usize; // 1 char padding each side
    let col_widths = compute_col_widths(cols, rows, usable_width);

    // --- Render visible rows ---
    let visible_rows = data_height;
    let offset = cell_state.scroll_offset;

    for (i, row) in rows.iter().enumerate().skip(offset).take(visible_rows) {
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

    visible_rows
}

/// Compute column widths based on content, honoring flex columns.
fn compute_col_widths(cols: &[ColDef], rows: &[CellRow], usable_width: usize) -> Vec<usize> {
    let n = cols.len();
    let mut widths: Vec<usize> = cols.iter().map(|c| c.header.len()).collect();

    // Measure max data width per column
    for row in rows {
        for (j, cell) in row.iter().enumerate() {
            if j < n {
                widths[j] = widths[j].max(cell.display_width());
            }
        }
    }

    // Add 1 char gap between columns
    let gaps = if n > 1 { n - 1 } else { 0 };
    let total_fixed: usize = widths.iter().sum::<usize>() + gaps;

    if total_fixed <= usable_width {
        // Distribute remaining space to flex columns
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
        // Shrink: give flex column(s) minimum, then truncate
        shrink_columns(&mut widths, cols, usable_width, gaps);
    }

    widths
}

fn shrink_columns(widths: &mut [usize], cols: &[ColDef], usable: usize, gaps: usize) {
    // Reduce flex columns first
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
            widths[i] = per_flex.max(4); // minimum 4 chars
        }
    }
}

/// Render a single row as a Line with proper alignment and coloring.
fn render_row(cols: &[ColDef], row: &CellRow, widths: &[usize], selected: bool) -> Line<'static> {
    let bg = if selected {
        Style::default()
            .bg(Color::DarkGray)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };

    let mut spans = vec![Span::styled(" ", bg)]; // left padding

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
                // For colored cells, render parts with their colors
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

        // Gap between columns
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
        return s.to_string();
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
        assert_eq!(c.display_width(), 5); // "3/1/2"
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
        cell.select_prev(); // should stay at 0
        assert_eq!(cell.selected, 0);
    }

    #[test]
    fn scrollable_cell_clamp() {
        let mut cell = ScrollableCell::new();
        cell.selected = 20;
        cell.clamp(10, 5);
        assert_eq!(cell.selected, 9);
        assert!(cell.scroll_offset <= 5);
    }

    #[test]
    fn cell_value_display_width() {
        let plain = CellValue::Plain("hello".into());
        assert_eq!(plain.display_width(), 5);

        let colored = CellValue::Colored(vec![
            ("3".into(), Color::Yellow),
            ("/".into(), Color::DarkGray),
            ("1".into(), Color::Blue),
        ]);
        assert_eq!(colored.display_width(), 3);
    }
}

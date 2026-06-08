//! Queue browser view with scrollable list, status filtering, and detail popup.
//!
//! Data is fetched from the local SQLite state database (read-only) and
//! refreshes on each tick event with a minimum 2-second interval.

use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Row, Table};
use ratatui::Frame;

use super::queue_data::{
    fetch_queue_detail, fetch_queue_rows, QueueDetail, QueueRow, StatusFilter,
};
use crate::tui::search::SearchState;
use crate::tui::theme;
use crate::tui::util::{truncate_end, truncate_path};

/// Minimum interval between data refreshes.
const REFRESH_INTERVAL_MS: u128 = 2000;

/// Queue browser view state.
pub struct QueueBrowser {
    /// Current list of queue items matching the active filter.
    items: Vec<QueueRow>,
    /// Index of the selected item in `items`.
    selected: usize,
    /// Active status filter.
    filter: StatusFilter,
    /// Detail popup for the selected item, if open.
    detail: Option<QueueDetail>,
    /// When data was last refreshed from SQLite.
    last_refresh: Option<Instant>,
    /// Search/filter state.
    search: SearchState,
}

impl QueueBrowser {
    /// Create a new, empty queue browser. Data loads on the first tick.
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            selected: 0,
            filter: StatusFilter::All,
            detail: None,
            last_refresh: None,
            search: SearchState::new(),
        }
    }

    pub fn search_mut(&mut self) -> &mut SearchState {
        &mut self.search
    }

    pub fn search_active(&self) -> bool {
        self.search.active
    }

    /// Refresh data from SQLite if enough time has elapsed.
    pub fn on_tick(&mut self) {
        let should_refresh = self
            .last_refresh
            .map_or(true, |t| t.elapsed().as_millis() >= REFRESH_INTERVAL_MS);

        if should_refresh {
            self.items = fetch_queue_rows(self.filter);
            if !self.items.is_empty() {
                self.selected = self.selected.min(self.items.len() - 1);
            } else {
                self.selected = 0;
            }
            self.last_refresh = Some(Instant::now());
        }
    }

    /// Returns the current status filter.
    #[cfg(test)]
    pub fn filter(&self) -> StatusFilter {
        self.filter
    }

    /// Returns true if the detail popup is currently visible.
    pub fn detail_open(&self) -> bool {
        self.detail.is_some()
    }

    /// Move selection up by one row.
    pub fn select_prev(&mut self) {
        if !self.items.is_empty() {
            self.selected = self.selected.saturating_sub(1);
        }
    }

    /// Move selection down by one row.
    pub fn select_next(&mut self) {
        if !self.items.is_empty() {
            self.selected = (self.selected + 1).min(self.items.len() - 1);
        }
    }

    /// Move selection up by a page.
    pub fn page_up(&mut self, page_size: usize) {
        self.selected = self.selected.saturating_sub(page_size);
    }

    /// Move selection down by a page.
    pub fn page_down(&mut self, page_size: usize) {
        if !self.items.is_empty() {
            self.selected = (self.selected + page_size).min(self.items.len() - 1);
        }
    }

    /// Cycle the status filter and force a re-fetch.
    pub fn cycle_filter(&mut self) {
        self.filter = self.filter.next();
        self.selected = 0;
        self.last_refresh = None; // force immediate refresh on next tick
    }

    /// Indices of items matching the current search pattern.
    fn search_matches(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, it)| {
                self.search.is_match(&format!(
                    "{} {} {} {}",
                    it.short_id, it.project, it.object, it.status
                ))
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Move the cursor to the first match at or after the current position.
    pub fn search_first(&mut self) {
        let m = self.search_matches();
        if let Some(i) = m
            .iter()
            .find(|&&i| i >= self.selected)
            .or_else(|| m.first())
        {
            self.selected = *i;
        }
    }

    /// Move the cursor to the next match (wrapping).
    pub fn search_next(&mut self) {
        let m = self.search_matches();
        if let Some(i) = crate::tui::search::next_index(&m, self.selected) {
            self.selected = i;
        }
    }

    /// Move the cursor to the previous match (wrapping).
    pub fn search_prev(&mut self) {
        let m = self.search_matches();
        if let Some(i) = crate::tui::search::prev_index(&m, self.selected) {
            self.selected = i;
        }
    }

    /// Open the detail popup for the currently selected item.
    pub fn open_detail(&mut self) {
        if let Some(item) = self.items.get(self.selected) {
            self.detail = fetch_queue_detail(&item.queue_id);
        }
    }

    /// Close the detail popup.
    pub fn close_detail(&mut self) {
        self.detail = None;
    }

    /// Render the queue browser into the given area.
    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(3)]).split(area);

        self.draw_filter_bar(frame, rows[0]);
        self.draw_table(frame, rows[1]);

        if let Some(ref detail) = self.detail {
            self.draw_detail_popup(frame, frame.area(), detail);
        }
    }

    /// Draw the filter status bar above the table.
    fn draw_filter_bar(&self, frame: &mut Frame, area: Rect) {
        let filter_label = self.filter.label();
        let count = self.items.len();

        let mut spans = vec![
            Span::styled(" Filter: ", Style::default().fg(Color::Gray)),
            Span::styled(
                filter_label,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  ({count} items)"),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled("  [f] cycle filter", Style::default().fg(Color::DarkGray)),
        ];
        spans.extend(crate::tui::search::prompt_spans(
            &self.search,
            self.search_matches().len(),
        ));

        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Draw the scrollable table of queue items.
    fn draw_table(&self, frame: &mut Frame, area: Rect) {
        let header = Row::new(vec![
            "ID", "T", "Tenant", "Object", "Type", "Op", "Status", "Age",
        ])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

        // Fixed columns: ID 10, T 1, Tenant 20, Type 8, Op 8, Status 12, Age 10.
        // Object flexes to fill the rest so the filename is the focus.
        let widths = [
            Constraint::Length(10),
            Constraint::Length(1),
            Constraint::Length(20),
            Constraint::Min(20),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(12),
            Constraint::Length(10),
        ];
        let object_w = (area.width as usize)
            .saturating_sub(10 + 1 + 20 + 8 + 8 + 12 + 10 + 7 + 2)
            .max(12);

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Queue ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        // Calculate scroll offset to keep selection visible
        let inner_height = area.height.saturating_sub(4) as usize;
        let offset = if inner_height > 0 && self.selected >= inner_height {
            self.selected - inner_height + 1
        } else {
            0
        };

        let visible_rows: Vec<Row> = self
            .items
            .iter()
            .enumerate()
            .skip(offset)
            .take(inner_height)
            .map(|(i, item)| {
                let row_style = if i == self.selected {
                    theme::selected_row_style()
                } else {
                    Style::default()
                };
                let fg = status_color(&item.status);

                let kind_fg = match item.kind {
                    'L' => theme::COLOR_INFO,
                    'P' => theme::COLOR_ACCENT,
                    _ => theme::COLOR_DIM,
                };

                // Spans set only fg; the row's background (cursor) shows through.
                Row::new(vec![
                    Span::styled(item.short_id.clone(), Style::default().fg(Color::Cyan)),
                    Span::styled(item.kind.to_string(), Style::default().fg(kind_fg)),
                    Span::raw(truncate_end(&item.project, 20)),
                    Span::raw(truncate_path(&item.object, object_w)),
                    Span::raw(item.item_type.clone()),
                    Span::raw(item.op.clone()),
                    Span::styled(item.status.clone(), Style::default().fg(fg)),
                    Span::styled(item.age.clone(), Style::default().fg(Color::DarkGray)),
                ])
                .style(row_style)
            })
            .collect();

        let table = Table::new(visible_rows, widths).header(header).block(block);

        frame.render_widget(table, area);

        if self.items.is_empty() {
            let inner = area.inner(ratatui::layout::Margin::new(2, 3));
            let msg =
                Paragraph::new("No queue items found").style(Style::default().fg(Color::DarkGray));
            frame.render_widget(msg, inner);
        }
    }

    /// Draw a centered detail popup overlay.
    fn draw_detail_popup(&self, frame: &mut Frame, area: Rect, detail: &QueueDetail) {
        let popup_width = 70u16.min(area.width.saturating_sub(4));
        let popup_height = 22u16.min(area.height.saturating_sub(4));

        let x = (area.width.saturating_sub(popup_width)) / 2;
        let y = (area.height.saturating_sub(popup_height)) / 2;
        let popup_area = Rect::new(x, y, popup_width, popup_height);

        frame.render_widget(Clear, popup_area);

        let mut lines = vec![
            detail_line("Queue ID", &detail.queue_id),
            detail_line("Idemp. Key", &truncate_str(&detail.idempotency_key, 40)),
            Line::from(""),
            detail_line("Project", &detail.project),
            detail_line("Tenant ID", &truncate_str(&detail.tenant_id, 40)),
            detail_line("Object", &detail.object),
            Line::from(""),
            detail_line("Type", &detail.item_type),
            detail_line("Operation", &detail.op),
            detail_line("Collection", &detail.collection),
            detail_line("Status", &detail.status),
            detail_line("Retries", &detail.retry_count.to_string()),
            Line::from(""),
            detail_line("Created", &format_local_time(&detail.created_at)),
            detail_line("Updated", &format_local_time(&detail.updated_at)),
        ];

        if let Some(ref err) = detail.error_message {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("  Error: {}", truncate_str(err, 50)),
                Style::default().fg(Color::Red),
            )));
        }

        let payload_preview = truncate_str(&detail.payload_json, 55);
        lines.push(Line::from(""));
        lines.push(detail_line("Payload", &payload_preview));

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Queue Item Detail ")
            .title_style(Style::default().add_modifier(Modifier::BOLD))
            .style(Style::default().bg(Color::Black));

        let popup = Paragraph::new(lines).block(block);
        frame.render_widget(popup, popup_area);
    }
}

/// Map a status string to a display color.
fn status_color(status: &str) -> Color {
    match status {
        "done" => Color::Green,
        "pending" => Color::Yellow,
        "in_progress" => Color::Blue,
        "failed" => Color::Red,
        _ => Color::Reset,
    }
}

/// Build a key-value detail line.
fn detail_line(key: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {:<14} ", key), Style::default().fg(Color::Gray)),
        Span::raw(value.to_string()),
    ])
}

/// Format a UTC timestamp for local display.
fn format_local_time(utc_str: &str) -> String {
    wqm_common::timestamp_fmt::format_local(utc_str)
}

/// Truncate a string to `max_len` characters, appending "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_browser_new_starts_empty() {
        let browser = QueueBrowser::new();
        assert!(browser.items.is_empty());
        assert_eq!(browser.selected, 0);
        assert_eq!(browser.filter, StatusFilter::All);
        assert!(browser.detail.is_none());
        assert!(browser.last_refresh.is_none());
    }

    #[test]
    fn select_next_clamps_to_bounds() {
        let mut browser = QueueBrowser::new();
        browser.items = make_test_rows(5);
        browser.selected = 4;
        browser.select_next();
        assert_eq!(browser.selected, 4);
    }

    #[test]
    fn select_prev_clamps_to_zero() {
        let mut browser = QueueBrowser::new();
        browser.items = make_test_rows(5);
        browser.selected = 0;
        browser.select_prev();
        assert_eq!(browser.selected, 0);
    }

    #[test]
    fn select_next_advances() {
        let mut browser = QueueBrowser::new();
        browser.items = make_test_rows(5);
        browser.selected = 2;
        browser.select_next();
        assert_eq!(browser.selected, 3);
    }

    #[test]
    fn select_prev_retreats() {
        let mut browser = QueueBrowser::new();
        browser.items = make_test_rows(5);
        browser.selected = 3;
        browser.select_prev();
        assert_eq!(browser.selected, 2);
    }

    #[test]
    fn page_up_clamps() {
        let mut browser = QueueBrowser::new();
        browser.items = make_test_rows(50);
        browser.selected = 5;
        browser.page_up(20);
        assert_eq!(browser.selected, 0);
    }

    #[test]
    fn page_down_clamps() {
        let mut browser = QueueBrowser::new();
        browser.items = make_test_rows(50);
        browser.selected = 45;
        browser.page_down(20);
        assert_eq!(browser.selected, 49);
    }

    #[test]
    fn cycle_filter_resets_selection() {
        let mut browser = QueueBrowser::new();
        browser.items = make_test_rows(10);
        browser.selected = 5;
        browser.cycle_filter();
        assert_eq!(browser.selected, 0);
        assert_eq!(browser.filter, StatusFilter::Pending);
        assert!(browser.last_refresh.is_none());
    }

    #[test]
    fn close_detail_clears() {
        let mut browser = QueueBrowser::new();
        browser.detail = Some(QueueDetail {
            queue_id: "test".into(),
            idempotency_key: "key".into(),
            item_type: "file".into(),
            op: "add".into(),
            collection: "projects".into(),
            status: "done".into(),
            project: "proj".into(),
            tenant_id: "t1".into(),
            object: "main.rs".into(),
            payload_json: "{}".into(),
            error_message: None,
            created_at: "2025-01-01T00:00:00Z".into(),
            updated_at: "2025-01-01T00:00:00Z".into(),
            retry_count: 0,
        });
        assert!(browser.detail_open());
        browser.close_detail();
        assert!(!browser.detail_open());
    }

    #[test]
    fn filter_accessor() {
        let browser = QueueBrowser::new();
        assert_eq!(browser.filter(), StatusFilter::All);
    }

    #[test]
    fn status_color_mapping() {
        assert_eq!(status_color("done"), Color::Green);
        assert_eq!(status_color("pending"), Color::Yellow);
        assert_eq!(status_color("in_progress"), Color::Blue);
        assert_eq!(status_color("failed"), Color::Red);
        assert_eq!(status_color("unknown"), Color::Reset);
    }

    #[test]
    fn truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn truncate_str_long() {
        let long = "a".repeat(40);
        let result = truncate_str(&long, 10);
        assert!(result.ends_with("..."));
        assert!(result.chars().count() <= 10);
    }

    #[test]
    fn select_on_empty_list() {
        let mut browser = QueueBrowser::new();
        browser.select_next();
        assert_eq!(browser.selected, 0);
        browser.select_prev();
        assert_eq!(browser.selected, 0);
    }

    fn make_test_rows(n: usize) -> Vec<QueueRow> {
        (0..n)
            .map(|i| QueueRow {
                queue_id: format!("id-{i}"),
                short_id: format!("id-{i}"),
                project: "project".into(),
                object: "file.rs".into(),
                item_type: "file".into(),
                op: "add".into(),
                status: "pending".into(),
                age: "1m ago".into(),
                kind: 'P',
            })
            .collect()
    }
}

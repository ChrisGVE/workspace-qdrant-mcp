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

use regex::Regex;

use super::queue_data::{
    fetch_queue_detail, fetch_queue_rows, QueueDetail, QueueRow, StatusFilter,
};
use super::service_data::format_bytes;
use crate::tui::filter::{self, FilterState};
use crate::tui::search::SearchState;
use crate::tui::theme;
use crate::tui::util::{truncate_end, truncate_path};

/// Build the text a filter or search matches against for a queue row: the
/// shortened id, project, object path, item type, operation, and status.
fn match_haystack(item: &QueueRow) -> String {
    format!(
        "{} {} {} {} {} {}",
        item.short_id, item.project, item.object, item.item_type, item.op, item.status
    )
}

/// Minimum interval between data refreshes.
const REFRESH_INTERVAL_MS: u128 = 2000;

/// Queue browser view state.
pub struct QueueBrowser {
    /// All rows from the last fetch (before page/global filtering).
    all_items: Vec<QueueRow>,
    /// Rows currently shown — `all_items` narrowed by the page + global filter.
    items: Vec<QueueRow>,
    /// Index of the selected item in `items`.
    selected: usize,
    /// Active SQL-side status filter (cycled with `s`).
    filter: StatusFilter,
    /// Per-page narrowing filter (`f`); composes (AND) with status and global.
    page_filter: FilterState,
    /// Compiled global filter pushed from the app; composes (AND) with the rest.
    global_re: Option<Regex>,
    /// Detail popup for the selected item, if open.
    detail: Option<QueueDetail>,
    /// When data was last refreshed from SQLite.
    last_refresh: Option<Instant>,
    /// Cursor-jump search state (`/`).
    search: SearchState,
}

impl QueueBrowser {
    /// Create a new, empty queue browser. Data loads on the first tick.
    pub fn new() -> Self {
        Self {
            all_items: Vec::new(),
            items: Vec::new(),
            selected: 0,
            filter: StatusFilter::All,
            page_filter: FilterState::new(),
            global_re: None,
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

    /// Mutable access to the page filter, for the key handler.
    pub fn page_filter_mut(&mut self) -> &mut FilterState {
        &mut self.page_filter
    }

    /// Whether the page-filter prompt is capturing input.
    pub fn page_filter_active(&self) -> bool {
        self.page_filter.active
    }

    /// Clear the page filter and re-narrow the list.
    pub fn clear_page_filter(&mut self) {
        self.page_filter.clear();
        self.recompute_visible();
    }

    /// Replace the global filter regex and re-narrow the list.
    pub fn set_global_filter(&mut self, re: Option<Regex>) {
        self.global_re = re;
        self.recompute_visible();
    }

    /// Rebuild `items` from `all_items` by applying the page and global filters
    /// (AND), then clamp the cursor into range.
    pub fn recompute_visible(&mut self) {
        let page = &self.page_filter;
        let global = &self.global_re;
        let filtered: Vec<QueueRow> = self
            .all_items
            .iter()
            .filter(|it| {
                let h = match_haystack(it);
                page.matches(&h) && filter::regex_matches(global, &h)
            })
            .cloned()
            .collect();
        self.items = filtered;
        self.selected = self.selected.min(self.items.len().saturating_sub(1));
    }

    /// Refresh data from SQLite if enough time has elapsed, then re-narrow.
    pub fn on_tick(&mut self) {
        let should_refresh = self
            .last_refresh
            .map_or(true, |t| t.elapsed().as_millis() >= REFRESH_INTERVAL_MS);

        if should_refresh {
            self.all_items = fetch_queue_rows(self.filter);
            self.recompute_visible();
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

    /// Jump to the first item.
    pub fn jump_first(&mut self) {
        self.selected = 0;
    }

    /// Jump to the last item.
    pub fn jump_last(&mut self) {
        if !self.items.is_empty() {
            self.selected = self.items.len() - 1;
        }
    }

    /// Cycle the SQL-side status filter and force a re-fetch.
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
            .filter(|(_, it)| self.search.is_match(&match_haystack(it)))
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
        let status_label = self.filter.label();
        let count = self.items.len();

        let mut spans = vec![
            Span::styled(" Status: ", Style::default().fg(Color::Gray)),
            Span::styled(
                status_label,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  ({count} items)"),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                "  [s] status  [f] filter",
                Style::default().fg(Color::DarkGray),
            ),
        ];
        spans.extend(filter::prompt_spans(&self.page_filter, "Filter"));
        spans.extend(crate::tui::search::prompt_spans(
            &self.search,
            self.search_matches().len(),
        ));

        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Draw the scrollable table of queue items.
    fn draw_table(&self, frame: &mut Frame, area: Rect) {
        let header = Row::new(vec![
            "ID", "T", "Tenant", "Object", "Type", "Op", "Status", "Size", "Age",
        ])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

        // Fixed columns: ID 10, T 1, Tenant 20, Type 8, Op 8, Status 12,
        // Size 9 (right-aligned), Age 10. Object flexes to fill the rest so
        // the filename is the focus.
        let widths = [
            Constraint::Length(10),
            Constraint::Length(1),
            Constraint::Length(20),
            Constraint::Min(20),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(12),
            Constraint::Length(9),
            Constraint::Length(10),
        ];
        // Fixed widths (excl. Object) + 8 inter-column gaps + 2 borders.
        let object_w = (area.width as usize)
            .saturating_sub(10 + 1 + 20 + 8 + 8 + 12 + 9 + 10 + 8 + 2)
            .max(12);

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Queue ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        // Chrome = top+bottom borders (2) + header row (1) + header margin (1).
        let inner_height = crate::tui::util::visible_rows(area.height, 4);
        let offset = crate::tui::util::scroll_offset(self.selected, inner_height);

        let visible_rows: Vec<Row> = self
            .items
            .iter()
            .enumerate()
            .skip(offset)
            .take(inner_height)
            .map(|(i, item)| {
                let matched =
                    self.search.has_query() && self.search.is_match(&match_haystack(item));
                let row_style = if i == self.selected {
                    theme::selected_row_style()
                } else if matched {
                    theme::search_match_style()
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
                    Span::raw(format!(
                        "{:>9}",
                        item.size.map(format_bytes).unwrap_or_default()
                    )),
                    Span::styled(
                        item.age.clone(),
                        Style::default().fg(age_color(&item.status, &item.age)),
                    ),
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

/// Color the Age column by queue health rather than dimming it: red for failed
/// items, yellow when a still-unprocessed item has been waiting hours or days
/// (a stalled queue), and a legible gray otherwise.
fn age_color(status: &str, age: &str) -> Color {
    if status == "failed" {
        return Color::Red;
    }
    let stale = age.contains("h ago") || age.contains("d ago");
    if stale && (status == "pending" || status == "in_progress") {
        Color::Yellow
    } else {
        Color::Gray
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
#[path = "queue_tests.rs"]
mod tests;

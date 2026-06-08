//! Project browser view with scrollable list, active/inactive indicators,
//! and a detail popup showing watch folders and queue breakdown.
//!
//! Data is fetched from the local SQLite state database (read-only) and
//! refreshes on each tick event with a minimum 5-second interval.

use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Row, Table};
use ratatui::Frame;

use super::projects_data::{
    build_status_text, fetch_project_detail, fetch_project_rows, format_local_time, ProjectDetail,
    ProjectRow,
};
use crate::tui::search::SearchState;
use crate::tui::theme;
use crate::tui::util::{truncate_end, truncate_path};

/// Minimum interval between data refreshes (projects change infrequently).
const REFRESH_INTERVAL_MS: u128 = 5000;

/// Project browser view state.
pub struct ProjectBrowser {
    /// Current list of registered projects.
    items: Vec<ProjectRow>,
    /// Index of the selected item in `items`.
    selected: usize,
    /// Detail popup for the selected project, if open.
    detail: Option<ProjectDetail>,
    /// When data was last refreshed from SQLite.
    last_refresh: Option<Instant>,
    /// Search/filter state.
    search: SearchState,
}

impl ProjectBrowser {
    /// Create a new, empty project browser. Data loads on the first tick.
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            selected: 0,
            detail: None,
            last_refresh: None,
            search: SearchState::new(),
        }
    }

    /// Refresh data from SQLite if enough time has elapsed.
    pub fn on_tick(&mut self) {
        let should_refresh = self
            .last_refresh
            .map_or(true, |t| t.elapsed().as_millis() >= REFRESH_INTERVAL_MS);

        if should_refresh {
            self.items = fetch_project_rows();
            // Active projects first, then by natural (case-insensitive) name.
            self.items.sort_by(|a, b| {
                b.is_active
                    .cmp(&a.is_active)
                    .then_with(|| crate::tui::util::natural_cmp(&a.name, &b.name))
            });
            if !self.items.is_empty() {
                self.selected = self.selected.min(self.items.len() - 1);
            } else {
                self.selected = 0;
            }
            self.last_refresh = Some(Instant::now());
        }
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

    /// Open the detail popup for the currently selected project.
    pub fn open_detail(&mut self) {
        if let Some(item) = self.items.get(self.selected) {
            self.detail = fetch_project_detail(&item.watch_id);
        }
    }

    /// Close the detail popup.
    pub fn close_detail(&mut self) {
        self.detail = None;
    }

    /// Get mutable access to search state.
    pub fn search_mut(&mut self) -> &mut SearchState {
        &mut self.search
    }

    /// Whether search input is currently active.
    pub fn search_active(&self) -> bool {
        self.search.active
    }

    /// Indices of items matching the current search pattern.
    fn search_matches(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, item)| {
                self.search
                    .is_match(&format!("{} {}", item.name, item.display_path))
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

    /// Render the project browser into the given area.
    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(3)]).split(area);

        self.draw_summary_bar(frame, rows[0]);
        self.draw_table(frame, rows[1]);

        if let Some(ref detail) = self.detail {
            self.draw_detail_popup(frame, frame.area(), detail);
        }
    }

    /// Draw the summary bar above the table.
    fn draw_summary_bar(&self, frame: &mut Frame, area: Rect) {
        let total = self.items.len();
        let active = self.items.iter().filter(|p| p.is_active).count();

        let mut spans = vec![
            Span::styled(" Projects: ", Style::default().fg(Color::Gray)),
            Span::styled(
                total.to_string(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  Active: ", Style::default().fg(Color::Gray)),
            Span::styled(
                active.to_string(),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  Inactive: {}", total - active),
                Style::default().fg(Color::DarkGray),
            ),
        ];

        spans.extend(crate::tui::search::prompt_spans(
            &self.search,
            self.search_matches().len(),
        ));

        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Draw the scrollable table of projects.
    fn draw_table(&self, frame: &mut Frame, area: Rect) {
        let header = Row::new(vec!["", "Name", "Path", "Docs", "Queue"])
            .style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .bottom_margin(1);

        let widths = [
            Constraint::Length(2),  // status indicator
            Constraint::Length(22), // name
            Constraint::Min(30),    // path
            Constraint::Length(6),  // doc count
            Constraint::Length(6),  // queue count
        ];

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Projects ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        // Chrome = top+bottom borders (2) + header row (1) + header margin (1).
        let inner_height = crate::tui::util::visible_rows(area.height, 4);
        let offset = crate::tui::util::scroll_offset(self.selected, inner_height);

        // Path flexes; compute its width so truncation keeps the trailing path.
        // Fixed columns: indicator 2, name 22, docs 6, queue 6; 4 gaps + borders.
        let path_w = (area.width as usize)
            .saturating_sub(2 + 22 + 6 + 6 + 4 + 2)
            .max(20);

        let visible_rows: Vec<Row> = self
            .items
            .iter()
            .enumerate()
            .skip(offset)
            .take(inner_height)
            .map(|(i, item)| self.render_row(i, item, path_w))
            .collect();

        let table = Table::new(visible_rows, widths).header(header).block(block);
        frame.render_widget(table, area);

        if self.items.is_empty() {
            let inner = area.inner(ratatui::layout::Margin::new(2, 3));
            let msg = Paragraph::new("No registered projects found")
                .style(Style::default().fg(Color::DarkGray));
            frame.render_widget(msg, inner);
        }
    }

    /// Render a single table row for a project.
    ///
    /// The cursor is the row's base style; spans set only `fg` so the highlight
    /// background shows through across the whole line.
    fn render_row(&self, index: usize, item: &ProjectRow, path_w: usize) -> Row<'static> {
        let matched = self.search.has_query()
            && self
                .search
                .is_match(&format!("{} {}", item.name, item.display_path));
        let row_style = if index == self.selected {
            theme::selected_row_style()
        } else if matched {
            theme::search_match_style()
        } else {
            Style::default()
        };
        // Active rows are bold (the only use of bold in the list); inactive rows
        // use a legible Gray rather than the previous near-invisible DarkGray.
        let indicator = if item.is_active {
            Span::styled("\u{25cf} ", Style::default().fg(Color::Green))
        } else {
            Span::styled("\u{25cb} ", Style::default().fg(Color::Gray))
        };
        let name_style = if item.is_active {
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        let path_fg = if item.is_active {
            Color::Gray
        } else {
            Color::DarkGray
        };
        let queue_fg = if item.queue_count > 0 {
            Color::Yellow
        } else {
            Color::DarkGray
        };

        Row::new(vec![
            indicator,
            Span::styled(truncate_end(&item.name, 22), name_style),
            Span::styled(
                truncate_path(&item.display_path, path_w),
                Style::default().fg(path_fg),
            ),
            Span::styled(item.doc_count.to_string(), Style::default().fg(Color::Cyan)),
            Span::styled(item.queue_count.to_string(), Style::default().fg(queue_fg)),
        ])
        .style(row_style)
    }

    /// Draw a centered detail popup overlay.
    fn draw_detail_popup(&self, frame: &mut Frame, area: Rect, detail: &ProjectDetail) {
        let popup_w = 70u16.min(area.width.saturating_sub(4));
        let popup_h = 24u16.min(area.height.saturating_sub(4));
        let x = (area.width.saturating_sub(popup_w)) / 2;
        let y = (area.height.saturating_sub(popup_h)) / 2;
        let popup_area = Rect::new(x, y, popup_w, popup_h);

        frame.render_widget(Clear, popup_area);

        let mut lines = vec![
            detail_line("Name", &detail.name),
            detail_line("Path", &detail.display_path),
            detail_line("Watch ID", &detail.watch_id),
            detail_line("Tenant ID", &truncate_str(&detail.tenant_id, 40)),
            detail_line("Collection", &detail.collection),
            Line::from(""),
            detail_line("Status", &build_status_text(detail)),
        ];

        if let Some(ref url) = detail.git_remote_url {
            lines.push(detail_line("Git Remote", &truncate_str(url, 50)));
        }

        lines.push(Line::from(""));
        lines.push(detail_line(
            "Created",
            &format_local_time(&detail.created_at),
        ));
        lines.push(detail_line(
            "Updated",
            &format_local_time(&detail.updated_at),
        ));
        if let Some(ref scan) = detail.last_scan {
            lines.push(detail_line("Last Scan", &format_local_time(scan)));
        }

        self.append_queue_breakdown(&mut lines, detail);
        self.append_sub_watches(&mut lines, detail);

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Project Detail ")
            .title_style(Style::default().add_modifier(Modifier::BOLD))
            .style(Style::default().bg(Color::Black));
        frame.render_widget(Paragraph::new(lines).block(block), popup_area);
    }

    /// Append queue breakdown section to popup lines.
    fn append_queue_breakdown(&self, lines: &mut Vec<Line<'static>>, detail: &ProjectDetail) {
        if detail.queue_by_status.is_empty() {
            return;
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Queue Breakdown:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        for (status, count) in &detail.queue_by_status {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("    {:<14} ", status),
                    Style::default().fg(status_color(status)),
                ),
                Span::raw(count.to_string()),
            ]));
        }
    }

    /// Append sub-watch folders section to popup lines.
    fn append_sub_watches(&self, lines: &mut Vec<Line<'static>>, detail: &ProjectDetail) {
        if detail.sub_watches.is_empty() {
            return;
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Watch Folders:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        for sw in &detail.sub_watches {
            lines.push(Line::from(Span::styled(
                format!("    {}", truncate_str(sw, 55)),
                Style::default().fg(Color::Gray),
            )));
        }
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
    use super::super::projects_data::ProjectRow;
    use super::*;

    #[test]
    fn project_browser_new_starts_empty() {
        let browser = ProjectBrowser::new();
        assert!(browser.items.is_empty());
        assert_eq!(browser.selected, 0);
        assert!(browser.detail.is_none());
        assert!(browser.last_refresh.is_none());
    }

    #[test]
    fn select_clamps_at_boundaries() {
        let mut b = ProjectBrowser::new();
        b.items = make_test_rows(5);
        b.selected = 4;
        b.select_next();
        assert_eq!(b.selected, 4);
        b.selected = 0;
        b.select_prev();
        assert_eq!(b.selected, 0);
    }

    #[test]
    fn select_advances_and_retreats() {
        let mut b = ProjectBrowser::new();
        b.items = make_test_rows(5);
        b.selected = 2;
        b.select_next();
        assert_eq!(b.selected, 3);
        b.select_prev();
        assert_eq!(b.selected, 2);
    }

    #[test]
    fn page_navigation_clamps() {
        let mut b = ProjectBrowser::new();
        b.items = make_test_rows(50);
        b.selected = 5;
        b.page_up(20);
        assert_eq!(b.selected, 0);
        b.selected = 45;
        b.page_down(20);
        assert_eq!(b.selected, 49);
    }

    #[test]
    fn close_detail_clears() {
        let mut b = ProjectBrowser::new();
        b.detail = Some(make_test_detail());
        assert!(b.detail_open());
        b.close_detail();
        assert!(!b.detail_open());
    }

    #[test]
    fn select_on_empty_list() {
        let mut b = ProjectBrowser::new();
        b.select_next();
        b.select_prev();
        assert_eq!(b.selected, 0);
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
    fn truncate_str_behavior() {
        assert_eq!(truncate_str("hello", 10), "hello");
        let long = "a".repeat(40);
        let result = truncate_str(&long, 10);
        assert!(result.ends_with("..."));
        assert!(result.chars().count() <= 10);
    }

    fn make_test_rows(n: usize) -> Vec<ProjectRow> {
        (0..n)
            .map(|i| ProjectRow {
                watch_id: format!("watch-{i}"),
                name: format!("project-{i}"),
                display_path: format!("~/dev/project-{i}"),
                is_active: i % 2 == 0,
                doc_count: (i * 10) as i64,
                queue_count: (i % 3) as i64,
            })
            .collect()
    }

    fn make_test_detail() -> ProjectDetail {
        ProjectDetail {
            watch_id: "w1".into(),
            tenant_id: "t1".into(),
            name: "test-proj".into(),
            display_path: "~/test-proj".into(),
            collection: "projects".into(),
            is_active: true,
            is_paused: false,
            is_archived: false,
            git_remote_url: None,
            created_at: "2025-01-01T00:00:00Z".into(),
            updated_at: "2025-01-01T12:00:00Z".into(),
            last_scan: None,
            sub_watches: Vec::new(),
            queue_by_status: std::collections::HashMap::new(),
        }
    }
}

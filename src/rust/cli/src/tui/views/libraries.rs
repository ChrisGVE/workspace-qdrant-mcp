//! Library browser view with scrollable list and detail popup.
//!
//! Data is fetched from the local SQLite state database (read-only) and
//! refreshes on each tick event with a minimum 5-second interval.

use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Table};
use ratatui::Frame;

use regex::Regex;

use super::confirm::{draw_toggle_confirm, tracked_cell, ToggleConfirm};
use super::libraries_data::{
    fetch_library_detail, fetch_library_rows, status_label, LibraryDetail, LibraryRow,
};
use crate::tui::filter::{self, FilterState};
use crate::tui::search::SearchState;
use crate::tui::theme;
use crate::tui::util::truncate_path;

/// Build the text a filter or search matches against for a library row: name,
/// tag, path, and the project-derived source marker (if any).
fn match_haystack(item: &LibraryRow) -> String {
    format!(
        "{} {} {} {}",
        item.name,
        item.tag,
        item.display_path,
        item.source.as_deref().unwrap_or("")
    )
}

/// Minimum interval between data refreshes (5 seconds).
const REFRESH_INTERVAL_MS: u128 = 5000;

/// Library browser view state.
pub struct LibraryBrowser {
    /// All libraries from the last fetch (before page/global filtering).
    all_items: Vec<LibraryRow>,
    /// Libraries currently shown — `all_items` narrowed by page + global filter.
    items: Vec<LibraryRow>,
    /// Index of the selected item in `items`.
    selected: usize,
    /// Per-page narrowing filter (`f`); composes (AND) with the global filter.
    page_filter: FilterState,
    /// Compiled global filter pushed from the app.
    global_re: Option<Regex>,
    /// Detail popup for the selected item, if open.
    detail: Option<LibraryDetail>,
    /// When data was last refreshed from SQLite.
    last_refresh: Option<Instant>,
    /// Cursor-jump search state (`/`).
    search: SearchState,
    /// Pending tracking-toggle confirmation, if the modal is open.
    confirm: Option<ToggleConfirm>,
    /// Transient status message (e.g. toggle result), shown in the header.
    message: Option<String>,
}

impl LibraryBrowser {
    /// Create a new, empty library browser. Data loads on the first tick.
    pub fn new() -> Self {
        Self {
            all_items: Vec::new(),
            items: Vec::new(),
            selected: 0,
            page_filter: FilterState::new(),
            global_re: None,
            detail: None,
            last_refresh: None,
            search: SearchState::new(),
            confirm: None,
            message: None,
        }
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
        let filtered: Vec<LibraryRow> = self
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

    /// Whether the toggle-confirmation modal is open.
    pub fn confirm_open(&self) -> bool {
        self.confirm.is_some()
    }

    /// Open a confirmation to toggle tracking for the selected library.
    ///
    /// A project-derived library follows its parent project and cannot be
    /// toggled here; the request is rejected with a message instead.
    pub fn request_toggle(&mut self) {
        if let Some(item) = self.items.get(self.selected) {
            if item.source.is_some() {
                self.message = Some(
                    "Project-derived library follows its project; toggle from Projects".to_string(),
                );
                return;
            }
            self.confirm = Some(ToggleConfirm {
                watch_id: item.watch_id.clone(),
                name: item.name.clone(),
                enable: !item.enabled,
            });
        }
    }

    /// Take the pending toggle (watch_id, target-enabled), clearing the modal.
    /// Call only when the user confirmed.
    pub fn take_confirm(&mut self) -> Option<(String, bool)> {
        self.confirm.take().map(|c| (c.watch_id, c.enable))
    }

    /// Cancel and close the toggle-confirmation modal.
    pub fn cancel_confirm(&mut self) {
        self.confirm = None;
    }

    /// Set a transient status message shown in the header bar.
    pub fn set_message(&mut self, msg: String) {
        self.message = Some(msg);
    }

    /// Force a data refresh on the next tick (after a state-changing action).
    pub fn force_refresh(&mut self) {
        self.last_refresh = None;
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
            self.all_items = fetch_library_rows();
            // Active libraries first, then by natural (case-insensitive) name.
            self.all_items.sort_by(|a, b| {
                b.is_active
                    .cmp(&a.is_active)
                    .then_with(|| crate::tui::util::natural_cmp(&a.name, &b.name))
            });
            self.recompute_visible();
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
            self.detail = fetch_library_detail(&item.watch_id);
        }
    }

    /// Close the detail popup.
    pub fn close_detail(&mut self) {
        self.detail = None;
    }

    /// Render the library browser into the given area.
    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(3)]).split(area);

        self.draw_header_bar(frame, rows[0]);
        self.draw_table(frame, rows[1]);

        if let Some(ref detail) = self.detail {
            self.draw_detail_popup(frame, frame.area(), detail);
        }
        if let Some(ref confirm) = self.confirm {
            draw_toggle_confirm(frame, frame.area(), confirm);
        }
    }

    /// Draw the header bar above the table.
    fn draw_header_bar(&self, frame: &mut Frame, area: Rect) {
        let count = self.items.len();
        let mut spans = vec![
            Span::styled(" Libraries: ", Style::default().fg(Color::Gray)),
            Span::styled(
                count.to_string(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                if count == 1 { " library" } else { " libraries" },
                Style::default().fg(Color::DarkGray),
            ),
        ];
        spans.extend(filter::prompt_spans(&self.page_filter, "Filter"));
        spans.extend(crate::tui::search::prompt_spans(
            &self.search,
            self.search_matches().len(),
        ));
        if let Some(ref msg) = self.message {
            spans.push(Span::styled(
                format!("  {msg}"),
                Style::default().fg(Color::Yellow),
            ));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Draw the scrollable table of library items.
    fn draw_table(&self, frame: &mut Frame, area: Rect) {
        // Shared columns (Name, Path, Docs) come first in the same order as the
        // Projects view; the library-specific columns follow. The project-derived
        // marker (`P:<project>`) is folded into Name, so there is no Source column.
        let header = Row::new(vec!["Name", "Path", "Docs", "Tracked?", "Mode"])
            .style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .bottom_margin(1);

        let widths = [
            Constraint::Length(24), // name (+ folded P:<project> marker)
            Constraint::Min(30),    // path (flex)
            Constraint::Length(8),  // docs
            Constraint::Length(9),  // tracked? (centered Yes/No)
            Constraint::Length(13), // mode
        ];
        // Path flexes; keep the trailing path (filename) visible on truncation.
        // Fixed: name 24, docs 8, tracked 9, mode 13; 4 gaps + borders.
        let path_w = (area.width as usize)
            .saturating_sub(24 + 8 + 9 + 13 + 4 + 2)
            .max(20);

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Libraries ")
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
                // Spans set only fg; the row's background (cursor) shows through.
                // Bold marks an active library (mirrors the projects view).
                let name_style = if item.is_active {
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Cyan)
                };
                // Fold the project-derived marker into the Name cell. A library
                // under a project always tracks in `sync` mode with its project.
                let (name_cell, mode) = match &item.source {
                    Some(src) => (
                        Line::from(vec![
                            Span::styled(item.name.clone(), name_style),
                            Span::raw(" "),
                            Span::styled(src.clone(), Style::default().fg(Color::Magenta)),
                        ]),
                        "sync".to_string(),
                    ),
                    None => (
                        Line::from(Span::styled(item.name.clone(), name_style)),
                        item.mode.clone(),
                    ),
                };
                let cells: Vec<Cell> = vec![
                    Cell::from(name_cell),
                    Cell::from(truncate_path(&item.display_path, path_w)),
                    Cell::from(Span::styled(
                        format!("{:>7}", crate::tui::util::fmt_count(item.doc_count as i64)),
                        Style::default().fg(Color::Cyan),
                    )),
                    Cell::from(tracked_cell(item.enabled)),
                    Cell::from(mode),
                ];
                Row::new(cells).style(row_style)
            })
            .collect();

        let table = Table::new(visible_rows, widths).header(header).block(block);
        frame.render_widget(table, area);

        if self.items.is_empty() {
            let inner = area.inner(ratatui::layout::Margin::new(2, 3));
            let msg = Paragraph::new("No libraries configured")
                .style(Style::default().fg(Color::DarkGray));
            frame.render_widget(msg, inner);
        }
    }

    /// Draw a centered detail popup overlay.
    fn draw_detail_popup(&self, frame: &mut Frame, area: Rect, detail: &LibraryDetail) {
        let popup_width = 70u16.min(area.width.saturating_sub(4));
        let popup_height = 22u16.min(area.height.saturating_sub(4));

        let x = (area.width.saturating_sub(popup_width)) / 2;
        let y = (area.height.saturating_sub(popup_height)) / 2;
        let popup_area = Rect::new(x, y, popup_width, popup_height);

        frame.render_widget(Clear, popup_area);

        let status = status_label(detail.enabled, detail.is_active);
        let paused = if detail.is_paused { "yes" } else { "no" };
        let archived = if detail.is_archived { "yes" } else { "no" };
        let symlinks = if detail.follow_symlinks { "yes" } else { "no" };
        let cleanup = if detail.cleanup_on_disable {
            "yes"
        } else {
            "no"
        };

        let name = detail
            .display_path
            .trim_end_matches('/')
            .rsplit('/')
            .find(|s| !s.is_empty())
            .unwrap_or(&detail.tag);

        let mut lines = vec![
            detail_line("Name", name),
            detail_line("Tag", &detail.tag),
            detail_line("Watch ID", &detail.watch_id),
            detail_line("Path", &detail.display_path),
            Line::from(""),
            detail_line("Status", status),
            detail_line("Mode", &detail.mode),
            detail_line("Documents", &detail.doc_count.to_string()),
            Line::from(""),
            detail_line("Paused", paused),
            detail_line("Archived", archived),
            detail_line("Symlinks", symlinks),
            detail_line("Cleanup", cleanup),
            Line::from(""),
            detail_line("Created", &format_local_time(&detail.created_at)),
            detail_line("Updated", &format_local_time(&detail.updated_at)),
        ];

        if let Some(ref scan) = detail.last_scan {
            lines.push(detail_line("Last Scan", &format_local_time(scan)));
        }
        if let Some(ref activity) = detail.last_activity_at {
            lines.push(detail_line("Last Active", &format_local_time(activity)));
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Library Detail ")
            .title_style(Style::default().add_modifier(Modifier::BOLD))
            .style(Style::default().bg(Color::Black));

        let popup = Paragraph::new(lines).block(block);
        frame.render_widget(popup, popup_area);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_browser_new_starts_empty() {
        let browser = LibraryBrowser::new();
        assert!(browser.items.is_empty());
        assert_eq!(browser.selected, 0);
        assert!(browser.detail.is_none());
        assert!(browser.last_refresh.is_none());
    }

    #[test]
    fn select_next_clamps_to_bounds() {
        let mut browser = LibraryBrowser::new();
        browser.items = make_test_rows(5);
        browser.selected = 4;
        browser.select_next();
        assert_eq!(browser.selected, 4);
    }

    #[test]
    fn select_prev_clamps_to_zero() {
        let mut browser = LibraryBrowser::new();
        browser.items = make_test_rows(5);
        browser.selected = 0;
        browser.select_prev();
        assert_eq!(browser.selected, 0);
    }

    #[test]
    fn select_next_advances() {
        let mut browser = LibraryBrowser::new();
        browser.items = make_test_rows(5);
        browser.selected = 2;
        browser.select_next();
        assert_eq!(browser.selected, 3);
    }

    #[test]
    fn select_prev_retreats() {
        let mut browser = LibraryBrowser::new();
        browser.items = make_test_rows(5);
        browser.selected = 3;
        browser.select_prev();
        assert_eq!(browser.selected, 2);
    }

    #[test]
    fn page_up_clamps() {
        let mut browser = LibraryBrowser::new();
        browser.items = make_test_rows(50);
        browser.selected = 5;
        browser.page_up(20);
        assert_eq!(browser.selected, 0);
    }

    #[test]
    fn page_down_clamps() {
        let mut browser = LibraryBrowser::new();
        browser.items = make_test_rows(50);
        browser.selected = 45;
        browser.page_down(20);
        assert_eq!(browser.selected, 49);
    }

    #[test]
    fn close_detail_clears() {
        let mut browser = LibraryBrowser::new();
        browser.detail = Some(LibraryDetail {
            watch_id: "lib-test".into(),
            tag: "test".into(),
            display_path: "/tmp/lib".into(),
            enabled: true,
            is_active: false,
            mode: "sync".into(),
            doc_count: 5,
            follow_symlinks: false,
            cleanup_on_disable: false,
            is_paused: false,
            is_archived: false,
            created_at: "2025-01-01T00:00:00Z".into(),
            updated_at: "2025-01-01T00:00:00Z".into(),
            last_scan: None,
            last_activity_at: None,
        });
        assert!(browser.detail_open());
        browser.close_detail();
        assert!(!browser.detail_open());
    }

    #[test]
    fn select_on_empty_list() {
        let mut browser = LibraryBrowser::new();
        browser.select_next();
        assert_eq!(browser.selected, 0);
        browser.select_prev();
        assert_eq!(browser.selected, 0);
    }

    #[test]
    fn request_toggle_skips_project_derived() {
        let mut b = LibraryBrowser::new();
        b.items = make_test_rows(2);
        b.items[0].source = Some("P:proj".into());
        b.selected = 0;
        b.request_toggle();
        // Project-derived library is not toggleable here: no modal, message set.
        assert!(!b.confirm_open());
        assert!(b.message.is_some());
    }

    #[test]
    fn request_toggle_opens_for_top_level_library() {
        let mut b = LibraryBrowser::new();
        b.items = make_test_rows(2);
        b.items[0].source = None;
        b.items[0].enabled = true;
        b.selected = 0;
        b.request_toggle();
        assert!(b.confirm_open());
        let (wid, enable) = b.take_confirm().unwrap();
        assert_eq!(wid, "lib-tag-0");
        assert!(!enable); // toggles to disabled
    }

    fn make_test_rows(n: usize) -> Vec<LibraryRow> {
        (0..n)
            .map(|i| LibraryRow {
                watch_id: format!("lib-tag-{i}"),
                tag: format!("tag-{i}"),
                name: format!("lib-{i}"),
                display_path: format!("/tmp/lib-{i}"),
                enabled: true,
                is_active: i % 2 == 0,
                mode: "sync".into(),
                doc_count: i as u64 * 10,
                source: None,
            })
            .collect()
    }
}

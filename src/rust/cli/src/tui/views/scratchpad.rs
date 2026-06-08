//! Scratchpad browser view — read-only list with detail popup.
//!
//! Data is fetched from the local SQLite scratchpad_mirror table
//! and refreshes on each tick with a minimum 5-second interval.

use std::collections::HashMap;
use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Row, Table};
use ratatui::Frame;
use wqm_common::constants::TENANT_GLOBAL;

use super::scratchpad_data::{fetch_scratchpad_rows, ScratchpadRow};
use crate::data::tenants;
use crate::tui::search::SearchState;
use crate::tui::theme;

/// Minimum interval between data refreshes.
const REFRESH_INTERVAL_MS: u128 = 5000;

/// Scratchpad browser view state.
pub struct ScratchpadBrowser {
    items: Vec<ScratchpadRow>,
    selected: usize,
    detail_open: bool,
    /// Scroll offset within the detail popup content.
    detail_scroll: u16,
    last_refresh: Option<Instant>,
    search: SearchState,
    /// tenant_id → project name map for tenant display.
    names: HashMap<String, String>,
}

impl ScratchpadBrowser {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            selected: 0,
            detail_open: false,
            detail_scroll: 0,
            last_refresh: None,
            search: SearchState::new(),
            names: HashMap::new(),
        }
    }

    pub fn search_mut(&mut self) -> &mut SearchState {
        &mut self.search
    }

    pub fn search_active(&self) -> bool {
        self.search.active
    }

    pub fn on_tick(&mut self) {
        let should_refresh = self
            .last_refresh
            .map_or(true, |t| t.elapsed().as_millis() >= REFRESH_INTERVAL_MS);

        if should_refresh {
            self.items = fetch_scratchpad_rows();
            self.names = tenants::name_map();
            if self.selected >= self.items.len() && !self.items.is_empty() {
                self.selected = self.items.len() - 1;
            }
            self.last_refresh = Some(Instant::now());
        }
    }

    pub fn select_next(&mut self) {
        if !self.items.is_empty() {
            self.selected = (self.selected + 1).min(self.items.len() - 1);
        }
    }

    pub fn select_prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub fn page_down(&mut self, n: usize) {
        if self.detail_open {
            self.detail_scroll = self.detail_scroll.saturating_add(n as u16);
        } else if !self.items.is_empty() {
            self.selected = (self.selected + n).min(self.items.len() - 1);
        }
    }

    pub fn page_up(&mut self, n: usize) {
        if self.detail_open {
            self.detail_scroll = self.detail_scroll.saturating_sub(n as u16);
        } else {
            self.selected = self.selected.saturating_sub(n);
        }
    }

    pub fn open_detail(&mut self) {
        if !self.items.is_empty() {
            self.detail_open = true;
            self.detail_scroll = 0;
        }
    }

    pub fn close_detail(&mut self) {
        self.detail_open = false;
        self.detail_scroll = 0;
    }

    pub fn detail_open(&self) -> bool {
        self.detail_open
    }

    pub fn scroll_detail_down(&mut self) {
        if self.detail_open {
            self.detail_scroll = self.detail_scroll.saturating_add(1);
        }
    }

    pub fn scroll_detail_up(&mut self) {
        if self.detail_open {
            self.detail_scroll = self.detail_scroll.saturating_sub(1);
        }
    }

    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        if self.items.is_empty() {
            let block = Block::default()
                .borders(Borders::ALL)
                .title(" Scratchpad ")
                .title_style(Style::default().add_modifier(Modifier::BOLD));
            let p = Paragraph::new("No scratchpad entries found.")
                .style(theme::loading_style())
                .block(block);
            frame.render_widget(p, area);
            return;
        }

        let chunks = Layout::vertical([
            Constraint::Length(3), // summary
            Constraint::Min(1),    // table
        ])
        .split(area);

        // Summary
        let summary = Paragraph::new(Line::from(vec![
            Span::styled(" Entries: ", Style::default().fg(theme::COLOR_MUTED)),
            Span::styled(
                self.items.len().to_string(),
                Style::default()
                    .fg(theme::COLOR_ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  (selected: {}/{})", self.selected + 1, self.items.len()),
                Style::default().fg(theme::COLOR_DIM),
            ),
            Span::styled("  [read-only]", Style::default().fg(theme::COLOR_DIM)),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Scratchpad ")
                .title_style(Style::default().add_modifier(Modifier::BOLD)),
        );
        frame.render_widget(summary, chunks[0]);

        self.render_scratchpad_table(frame, chunks[1]);

        // Detail popup
        if self.detail_open {
            if let Some(entry) = self.items.get(self.selected) {
                self.draw_detail_popup(frame, area, entry);
            }
        }
    }

    fn render_scratchpad_table(&self, frame: &mut Frame, area: Rect) {
        let header = Row::new(vec!["  Title", "Tenant", "Tags", "Updated"])
            .style(theme::table_header_style());

        let visible_height = area.height.saturating_sub(2) as usize;
        let start = if self.selected >= visible_height {
            self.selected - visible_height + 1
        } else {
            0
        };

        // Title takes all space left after the fixed columns (tenant 18, tags 20,
        // updated 12), their gaps, the borders, and the 2-space lead-in.
        let title_w = (area.width as usize)
            .saturating_sub(18 + 20 + 12 + 3 + 2 + 2)
            .max(10);

        let rows: Vec<Row> = self
            .items
            .iter()
            .enumerate()
            .skip(start)
            .take(visible_height)
            .map(|(i, entry)| {
                let title = truncate_str(&entry.title, title_w);
                let tenant = if entry.tenant_id == TENANT_GLOBAL {
                    TENANT_GLOBAL.to_string()
                } else {
                    truncate_str(&tenants::display_name(&self.names, &entry.tenant_id), 16)
                };
                let tags = format_tags(&entry.tags);
                let updated = format_short_date(&entry.updated_at);
                let style = if i == self.selected {
                    theme::selected_row_style()
                } else {
                    Style::default()
                };
                Row::new(vec![format!("  {}", title), tenant, tags, updated]).style(style)
            })
            .collect();

        let widths = [
            Constraint::Min(20),
            Constraint::Length(18),
            Constraint::Length(20),
            Constraint::Length(12),
        ];

        let table = Table::new(rows, widths)
            .header(header)
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(table, area);
    }

    fn draw_detail_popup(&self, frame: &mut Frame, area: Rect, entry: &ScratchpadRow) {
        let popup_width = (area.width - 4).min(90);
        let popup_height = (area.height - 4).min(40);
        let x = (area.width.saturating_sub(popup_width)) / 2;
        let y = (area.height.saturating_sub(popup_height)) / 2;
        let popup_area = Rect::new(x, y, popup_width, popup_height);

        frame.render_widget(Clear, popup_area);

        let tags = format_tags(&entry.tags);
        let tenant = if entry.tenant_id == TENANT_GLOBAL {
            TENANT_GLOBAL.to_string()
        } else {
            tenants::display_name(&self.names, &entry.tenant_id)
        };

        let mut lines = vec![
            Line::from(vec![
                Span::styled("  Title:    ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&entry.title),
            ]),
            Line::from(vec![
                Span::styled("  Tenant:   ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&tenant),
            ]),
            Line::from(vec![
                Span::styled("  Tags:     ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&tags),
            ]),
            Line::from(vec![
                Span::styled("  Created:  ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&entry.created_at),
            ]),
            Line::from(vec![
                Span::styled("  Updated:  ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&entry.updated_at),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "  Content:",
                Style::default()
                    .fg(theme::COLOR_FG)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        // Wrap content into lines that fit the popup
        let text_width = (popup_width - 6) as usize;
        for wrapped_line in wrap_text(&entry.content, text_width) {
            lines.push(Line::from(format!("  {}", wrapped_line)));
        }

        let popup = Paragraph::new(lines).scroll((self.detail_scroll, 0)).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Scratchpad Entry (j/k to scroll) ")
                .title_style(Style::default().add_modifier(Modifier::BOLD))
                .style(theme::popup_style()),
        );

        frame.render_widget(popup, popup_area);
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(1)).collect();
        format!("{}\u{2026}", truncated)
    }
}

fn format_short_date(s: &str) -> String {
    if s.len() >= 10 {
        s[..10].to_string()
    } else {
        s.to_string()
    }
}

/// Parse JSON tags array into a comma-separated display string.
fn format_tags(tags_json: &str) -> String {
    if tags_json == "[]" || tags_json.is_empty() {
        return "—".to_string();
    }
    // Try to parse as JSON array of strings
    if let Ok(tags) = serde_json::from_str::<Vec<String>>(tags_json) {
        if tags.is_empty() {
            "—".to_string()
        } else {
            tags.join(", ")
        }
    } else {
        tags_json.to_string()
    }
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for paragraph in text.lines() {
        if paragraph.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut current_line = String::new();
        for word in paragraph.split_whitespace() {
            if current_line.is_empty() {
                current_line = word.to_string();
            } else if current_line.len() + 1 + word.len() <= width {
                current_line.push(' ');
                current_line.push_str(word);
            } else {
                lines.push(current_line);
                current_line = word.to_string();
            }
        }
        if !current_line.is_empty() {
            lines.push(current_line);
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_initializes_empty() {
        let browser = ScratchpadBrowser::new();
        assert!(browser.items.is_empty());
        assert_eq!(browser.selected, 0);
        assert!(!browser.detail_open());
    }

    #[test]
    fn format_tags_empty() {
        assert_eq!(format_tags("[]"), "—");
        assert_eq!(format_tags(""), "—");
    }

    #[test]
    fn format_tags_json() {
        assert_eq!(format_tags(r#"["rust","cli","tui"]"#), "rust, cli, tui");
    }

    #[test]
    fn format_tags_single() {
        assert_eq!(format_tags(r#"["analysis"]"#), "analysis");
    }

    #[test]
    fn detail_scroll() {
        let mut browser = ScratchpadBrowser::new();
        browser.detail_open = true;
        browser.scroll_detail_down();
        assert_eq!(browser.detail_scroll, 1);
        browser.scroll_detail_up();
        assert_eq!(browser.detail_scroll, 0);
        browser.scroll_detail_up();
        assert_eq!(browser.detail_scroll, 0); // no underflow
    }
}

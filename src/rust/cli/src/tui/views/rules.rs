//! Rules browser view with scrollable list and detail popup.
//!
//! Data is fetched from the local SQLite rules_mirror table (read-only)
//! and refreshes on each tick event with a minimum 5-second interval.

use std::collections::HashMap;
use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Row, Table};
use ratatui::Frame;
use wqm_common::constants::TENANT_GLOBAL;

use super::rules_data::{fetch_rule_rows, RuleRow};
use crate::data::tenants;
use crate::tui::search::SearchState;
use crate::tui::theme;

/// Minimum interval between data refreshes.
const REFRESH_INTERVAL_MS: u128 = 5000;

/// Rules browser view state.
pub struct RuleBrowser {
    /// Current list of rules.
    items: Vec<RuleRow>,
    /// Index of the selected item.
    selected: usize,
    /// Whether the detail popup is open.
    detail_open: bool,
    /// When data was last refreshed.
    last_refresh: Option<Instant>,
    /// Search/filter state.
    search: SearchState,
    /// tenant_id → project name map for scope display.
    names: HashMap<String, String>,
}

impl RuleBrowser {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            selected: 0,
            detail_open: false,
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

    /// Refresh data from SQLite if enough time has elapsed.
    pub fn on_tick(&mut self) {
        let should_refresh = self
            .last_refresh
            .map_or(true, |t| t.elapsed().as_millis() >= REFRESH_INTERVAL_MS);

        if should_refresh {
            self.items = fetch_rule_rows();
            self.names = tenants::name_map();
            let names = self.names.clone();
            self.items.sort_by(|a, b| {
                crate::tui::util::natural_cmp(
                    &rule_tenant_key(a, &names),
                    &rule_tenant_key(b, &names),
                )
            });
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
        if !self.items.is_empty() {
            self.selected = (self.selected + n).min(self.items.len() - 1);
        }
    }

    pub fn page_up(&mut self, n: usize) {
        self.selected = self.selected.saturating_sub(n);
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

    pub fn open_detail(&mut self) {
        if !self.items.is_empty() {
            self.detail_open = true;
        }
    }

    /// Indices of rules matching the current search pattern.
    fn search_matches(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                let name = tenants::display_name(&self.names, &r.tenant_id);
                self.search
                    .is_match(&format!("{} {} {}", r.rule_text, r.scope, name))
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

    pub fn close_detail(&mut self) {
        self.detail_open = false;
    }

    pub fn detail_open(&self) -> bool {
        self.detail_open
    }

    /// Draw the rules browser.
    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        if self.items.is_empty() {
            let block = Block::default()
                .borders(Borders::ALL)
                .title(" Rules ")
                .title_style(Style::default().add_modifier(Modifier::BOLD));
            let p = Paragraph::new("No rules found. Use `wqm rules add` to create rules.")
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
        let mut summary_spans = vec![
            Span::styled(" Rules: ", Style::default().fg(theme::COLOR_MUTED)),
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
        ];
        summary_spans.extend(crate::tui::search::prompt_spans(
            &self.search,
            self.search_matches().len(),
        ));
        let summary = Paragraph::new(Line::from(summary_spans)).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Rules ")
                .title_style(Style::default().add_modifier(Modifier::BOLD)),
        );
        frame.render_widget(summary, chunks[0]);

        self.render_rules_table(frame, chunks[1]);

        // Detail popup
        if self.detail_open {
            if let Some(rule) = self.items.get(self.selected) {
                self.draw_detail_popup(frame, area, rule);
            }
        }
    }

    fn render_rules_table(&self, frame: &mut Frame, area: Rect) {
        let header =
            Row::new(vec!["  Tenant", "Rule Text", "Updated"]).style(theme::table_header_style());

        // Chrome = top+bottom borders (2) + header row (1). No header margin.
        let visible_height = crate::tui::util::visible_rows(area.height, 3);
        let start = crate::tui::util::scroll_offset(self.selected, visible_height);

        let rows: Vec<Row> = self
            .items
            .iter()
            .enumerate()
            .skip(start)
            .take(visible_height)
            .map(|(i, rule)| {
                let scope_display = if rule.scope == TENANT_GLOBAL {
                    TENANT_GLOBAL.to_string()
                } else if rule.tenant_id.is_empty() {
                    rule.scope.clone()
                } else {
                    truncate_str(&tenants::display_name(&self.names, &rule.tenant_id), 14)
                };
                let text_preview = truncate_str(&rule.rule_text, 60);
                let updated = format_short_date(&rule.updated_at);
                let style = if i == self.selected {
                    theme::selected_row_style()
                } else {
                    Style::default()
                };
                Row::new(vec![format!("  {}", scope_display), text_preview, updated]).style(style)
            })
            .collect();

        let widths = [
            Constraint::Length(16),
            Constraint::Min(30),
            Constraint::Length(12),
        ];

        let table = Table::new(rows, widths)
            .header(header)
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(table, area);
    }

    fn draw_detail_popup(&self, frame: &mut Frame, area: Rect, rule: &RuleRow) {
        let popup_width = (area.width - 4).min(80);
        let popup_height = (area.height - 4).min(30);
        let x = (area.width.saturating_sub(popup_width)) / 2;
        let y = (area.height.saturating_sub(popup_height)) / 2;
        let popup_area = Rect::new(x, y, popup_width, popup_height);

        frame.render_widget(Clear, popup_area);

        let scope_display = if rule.scope == TENANT_GLOBAL {
            TENANT_GLOBAL.to_string()
        } else if rule.tenant_id.is_empty() {
            rule.scope.clone()
        } else {
            tenants::display_name(&self.names, &rule.tenant_id)
        };

        let mut lines = vec![
            Line::from(vec![
                Span::styled("  ID:       ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&rule.rule_id),
            ]),
            Line::from(vec![
                Span::styled("  Scope:    ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&scope_display),
            ]),
            Line::from(vec![
                Span::styled("  Created:  ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&rule.created_at),
            ]),
            Line::from(vec![
                Span::styled("  Updated:  ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&rule.updated_at),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "  Rule Text:",
                Style::default()
                    .fg(theme::COLOR_FG)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        // Wrap rule text into lines that fit the popup
        let text_width = (popup_width - 6) as usize;
        for wrapped_line in wrap_text(&rule.rule_text, text_width) {
            lines.push(Line::from(format!("  {}", wrapped_line)));
        }

        let popup = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Rule Detail ")
                .title_style(Style::default().add_modifier(Modifier::BOLD))
                .style(theme::popup_style()),
        );

        frame.render_widget(popup, popup_area);
    }
}

/// Sort key for a rule: the global label for global rules, otherwise the
/// resolved project name. Used to group/sort rules by tenant.
fn rule_tenant_key(rule: &RuleRow, names: &std::collections::HashMap<String, String>) -> String {
    if rule.scope == TENANT_GLOBAL || rule.tenant_id.is_empty() {
        TENANT_GLOBAL.to_string()
    } else {
        tenants::display_name(names, &rule.tenant_id)
    }
}

/// Truncate a string to `max_len` characters, adding ellipsis if needed.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(1)).collect();
        format!("{}\u{2026}", truncated)
    }
}

/// Format an ISO datetime string to short date (YYYY-MM-DD).
fn format_short_date(s: &str) -> String {
    if s.len() >= 10 {
        s[..10].to_string()
    } else {
        s.to_string()
    }
}

/// Simple word-wrapping for display in popups.
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
        let browser = RuleBrowser::new();
        assert!(browser.items.is_empty());
        assert_eq!(browser.selected, 0);
        assert!(!browser.detail_open());
    }

    #[test]
    fn truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn truncate_str_long() {
        let result = truncate_str("hello world foo bar", 10);
        assert!(result.chars().count() <= 10);
        assert!(result.contains('\u{2026}'));
    }

    #[test]
    fn format_short_date_iso() {
        assert_eq!(format_short_date("2026-04-16T18:41:10+02:00"), "2026-04-16");
    }

    #[test]
    fn format_short_date_short_input() {
        assert_eq!(format_short_date("2026"), "2026");
    }

    #[test]
    fn wrap_text_basic() {
        let lines = wrap_text("hello world foo bar baz", 12);
        assert!(!lines.is_empty());
        for line in &lines {
            assert!(line.len() <= 12, "line too long: {}", line);
        }
    }

    #[test]
    fn wrap_text_empty() {
        let lines = wrap_text("", 80);
        assert!(lines.is_empty());
    }
}

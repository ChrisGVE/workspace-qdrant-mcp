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

use regex::Regex;

use super::confirm::{draw_target_prompt, draw_typed_confirm, TargetPrompt, TypedConfirm};
use super::scratchpad_data::{fetch_scratchpad_rows, ScratchpadRow};
use crate::data::tenants;
use crate::tui::filter::{self, FilterState};
use crate::tui::render::content::render_markdown;
use crate::tui::search::SearchState;
use crate::tui::theme;

/// Build the text a filter or search matches against for a scratchpad entry:
/// its title, content, and tags.
fn match_haystack(entry: &ScratchpadRow) -> String {
    format!("{} {} {}", entry.title, entry.content, entry.tags)
}

/// Minimum interval between data refreshes.
const REFRESH_INTERVAL_MS: u128 = 5000;

/// Scratchpad browser view state.
pub struct ScratchpadBrowser {
    /// All entries from the last fetch (before page/global filtering).
    all_items: Vec<ScratchpadRow>,
    /// Entries currently shown — `all_items` narrowed by page + global filter.
    items: Vec<ScratchpadRow>,
    selected: usize,
    detail_open: bool,
    /// Scroll offset within the detail popup content.
    detail_scroll: u16,
    /// Per-page narrowing filter (`f`); composes (AND) with the global filter.
    page_filter: FilterState,
    /// Compiled global filter pushed from the app.
    global_re: Option<Regex>,
    last_refresh: Option<Instant>,
    search: SearchState,
    /// tenant_id → project name map for tenant display.
    names: HashMap<String, String>,
    /// Pending typed-name deletion confirm with the row captured at request
    /// time (#122) — never read `items[selected]` at confirm time.
    pub(super) delete_confirm: Option<(TypedConfirm, ScratchpadRow)>,
    /// Pending reassign-target prompt with the captured row (#122).
    pub(super) reassign_prompt: Option<(TargetPrompt, ScratchpadRow)>,
    /// Transient status message shown after an action.
    message: Option<String>,
}

impl ScratchpadBrowser {
    pub fn new() -> Self {
        Self {
            all_items: Vec::new(),
            items: Vec::new(),
            selected: 0,
            detail_open: false,
            detail_scroll: 0,
            page_filter: FilterState::new(),
            global_re: None,
            last_refresh: None,
            search: SearchState::new(),
            names: HashMap::new(),
            delete_confirm: None,
            reassign_prompt: None,
            message: None,
        }
    }

    /// Slice of the currently visible items (for the actions module).
    pub(super) fn items(&self) -> &[ScratchpadRow] {
        &self.items
    }

    /// Index of the currently selected item (for the actions module).
    pub(super) fn selected_index(&self) -> usize {
        self.selected
    }

    /// tenant_id → project name map (for reassign target resolution).
    pub fn names_map(&self) -> &HashMap<String, String> {
        &self.names
    }

    /// Set a transient status message shown in the summary bar.
    pub fn set_message(&mut self, msg: String) {
        self.message = Some(msg);
    }

    /// Force a data refresh on the next tick.
    pub fn force_refresh(&mut self) {
        self.last_refresh = None;
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
        let filtered: Vec<ScratchpadRow> = self
            .all_items
            .iter()
            .filter(|e| {
                let h = match_haystack(e);
                page.matches(&h) && filter::regex_matches(global, &h)
            })
            .cloned()
            .collect();
        self.items = filtered;
        self.selected = self.selected.min(self.items.len().saturating_sub(1));
    }

    pub fn on_tick(&mut self) {
        let should_refresh = self
            .last_refresh
            .map_or(true, |t| t.elapsed().as_millis() >= REFRESH_INTERVAL_MS);

        if should_refresh {
            self.all_items = fetch_scratchpad_rows();
            self.names = tenants::name_map();
            let names = self.names.clone();
            self.all_items.sort_by(|a, b| {
                crate::tui::util::natural_cmp(
                    &scratchpad_tenant_key(a, &names),
                    &scratchpad_tenant_key(b, &names),
                )
            });
            self.recompute_visible();
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
            self.detail_scroll = 0;
        }
    }

    /// Indices of entries matching the current search pattern.
    fn search_matches(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, e)| self.search.is_match(&match_haystack(e)))
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
        let mut summary_spans = vec![
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
            Span::styled(
                "  [e edit  m move  d delete]",
                Style::default().fg(theme::COLOR_DIM),
            ),
        ];
        summary_spans.extend(filter::prompt_spans(&self.page_filter, "Filter"));
        summary_spans.extend(crate::tui::search::prompt_spans(
            &self.search,
            self.search_matches().len(),
        ));
        if let Some(ref msg) = self.message {
            summary_spans.push(Span::styled(
                format!("  {msg}"),
                Style::default().fg(theme::COLOR_WARNING),
            ));
        }
        let summary = Paragraph::new(Line::from(summary_spans)).block(
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
        // Typed-name deletion confirm modal (#122)
        if let Some((ref tc, _)) = self.delete_confirm {
            draw_typed_confirm(frame, frame.area(), tc);
        }
        // Reassign-target prompt (#122)
        if let Some((ref prompt, ref row)) = self.reassign_prompt {
            let name = super::scratchpad_actions::confirm_name(row);
            draw_target_prompt(frame, frame.area(), &name, prompt);
        }
    }

    fn render_scratchpad_table(&self, frame: &mut Frame, area: Rect) {
        let header = Row::new(vec!["  Tenant", "Title", "Tags", "Updated"])
            .style(theme::table_header_style());

        // Chrome = top+bottom borders (2) + header row (1). No header margin.
        let visible_height = crate::tui::util::visible_rows(area.height, 3);
        let start = crate::tui::util::scroll_offset(self.selected, visible_height);

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
                let matched =
                    self.search.has_query() && self.search.is_match(&match_haystack(entry));
                let style = if i == self.selected {
                    theme::selected_row_style()
                } else if matched {
                    theme::search_match_style()
                } else {
                    Style::default()
                };
                Row::new(vec![format!("  {}", tenant), title, tags, updated]).style(style)
            })
            .collect();

        let widths = [
            Constraint::Length(18),
            Constraint::Min(20),
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

        // Render content as Markdown, word-wrapped to the popup inner width.
        let text_width = popup_width.saturating_sub(6) as usize;
        lines.extend(render_markdown(&entry.content, text_width));

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

/// Sort key for an entry: the global label for global notes, otherwise the
/// resolved project name. Used to group/sort scratchpad entries by tenant.
fn scratchpad_tenant_key(
    entry: &ScratchpadRow,
    names: &std::collections::HashMap<String, String>,
) -> String {
    if entry.tenant_id == TENANT_GLOBAL {
        TENANT_GLOBAL.to_string()
    } else {
        tenants::display_name(names, &entry.tenant_id)
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

#[cfg(test)]
#[path = "scratchpad_tests.rs"]
mod tests;

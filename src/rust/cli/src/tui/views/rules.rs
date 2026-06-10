//! Rules browser view with scrollable list and detail popup.
//!
//! Data is fetched from the local SQLite rules_mirror table (read-only)
//! and refreshes on each tick event with a minimum 5-second interval.

use std::collections::HashMap;
use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;
use wqm_common::constants::TENANT_GLOBAL;

use regex::Regex;

use super::confirm::{draw_typed_confirm, TypedConfirm};
use super::rules_data::{fetch_rule_rows, RuleRow};
use crate::data::tenants;
use crate::tui::filter::{self, FilterState};
use crate::tui::search::SearchState;
use crate::tui::theme;

/// Build the text a filter or search matches against for a rule: its text,
/// scope, and resolved tenant/project name.
pub(super) fn match_haystack(rule: &RuleRow, names: &HashMap<String, String>) -> String {
    let name = tenants::display_name(names, &rule.tenant_id);
    format!("{} {} {}", rule.rule_text, rule.scope, name)
}

/// Minimum interval between data refreshes.
const REFRESH_INTERVAL_MS: u128 = 5000;

/// Rules browser view state.
pub struct RuleBrowser {
    /// All rules from the last fetch (before page/global filtering).
    all_items: Vec<RuleRow>,
    /// Rules currently shown — `all_items` narrowed by the page + global filter.
    pub(super) items: Vec<RuleRow>,
    /// Index of the selected item.
    pub(super) selected: usize,
    /// Per-page narrowing filter (`f`); composes (AND) with the global filter.
    page_filter: FilterState,
    /// Compiled global filter pushed from the app.
    global_re: Option<Regex>,
    /// Whether the detail popup is open.
    detail_open: bool,
    /// When data was last refreshed.
    last_refresh: Option<Instant>,
    /// Cursor-jump search state (`/`).
    pub(super) search: SearchState,
    /// tenant_id → project name map for scope display.
    pub(super) names: HashMap<String, String>,
    /// Pending typed-name deletion confirm, if the modal is open.
    delete_confirm: Option<TypedConfirm>,
    /// Transient status message shown after an action.
    message: Option<String>,
}

impl RuleBrowser {
    pub fn new() -> Self {
        Self {
            all_items: Vec::new(),
            items: Vec::new(),
            selected: 0,
            page_filter: FilterState::new(),
            global_re: None,
            detail_open: false,
            last_refresh: None,
            search: SearchState::new(),
            names: HashMap::new(),
            delete_confirm: None,
            message: None,
        }
    }

    // ── Draw accessors (used by rules_draw.rs) ──────────────────────────────

    /// Slice of the currently visible items.
    pub(super) fn items_slice(&self) -> &[RuleRow] {
        &self.items
    }

    /// Index of the currently selected item.
    pub(super) fn selected_index(&self) -> usize {
        self.selected
    }

    /// Read-only reference to the search state (for prompt rendering).
    pub(super) fn search_ref(&self) -> &SearchState {
        &self.search
    }

    /// Read-only reference to the tenant-name map.
    pub(super) fn names_ref(&self) -> &HashMap<String, String> {
        &self.names
    }

    // ── Public API ──────────────────────────────────────────────────────────

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
        let names = &self.names;
        let filtered: Vec<RuleRow> = self
            .all_items
            .iter()
            .filter(|r| {
                let h = match_haystack(r, names);
                page.matches(&h) && filter::regex_matches(global, &h)
            })
            .cloned()
            .collect();
        self.items = filtered;
        self.selected = self.selected.min(self.items.len().saturating_sub(1));
    }

    /// Refresh data from SQLite if enough time has elapsed.
    pub fn on_tick(&mut self) {
        let should_refresh = self
            .last_refresh
            .map_or(true, |t| t.elapsed().as_millis() >= REFRESH_INTERVAL_MS);

        if should_refresh {
            self.all_items = fetch_rule_rows();
            self.names = tenants::name_map();
            let names = self.names.clone();
            self.all_items.sort_by(|a, b| {
                crate::tui::util::natural_cmp(
                    &rule_tenant_key(a, &names),
                    &rule_tenant_key(b, &names),
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
            .filter(|(_, r)| self.search.is_match(&match_haystack(r, &self.names)))
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

    // ── Delete confirm ──────────────────────────────────────────────────────

    /// Whether the typed-name delete confirmation modal is open.
    pub fn delete_confirm_open(&self) -> bool {
        self.delete_confirm.is_some()
    }

    /// Open a typed-name delete confirmation for the selected rule.
    ///
    /// The confirmation target name is the first line of the rule text, truncated
    /// to 40 characters, so the user types something recognisable.
    pub fn request_delete(&mut self) {
        if let Some(rule) = self.items.get(self.selected) {
            let label = rule
                .rule_text
                .lines()
                .next()
                .unwrap_or(&rule.rule_id)
                .chars()
                .take(40)
                .collect::<String>();
            self.delete_confirm = Some(TypedConfirm::new(label));
        }
    }

    /// Mutable reference to the pending delete confirm (for key input).
    pub fn delete_confirm_mut(&mut self) -> Option<&mut TypedConfirm> {
        self.delete_confirm.as_mut()
    }

    /// Return the rule_id of the selected item together with the typed confirm,
    /// consuming both. Returns `None` if the confirm is not open or input does
    /// not yet match.
    pub fn take_delete_if_confirmed(&mut self) -> Option<String> {
        if self.delete_confirm.as_ref().map_or(false, |c| c.matches()) {
            self.delete_confirm = None;
            self.items
                .get(self.selected)
                .map(|rule| rule.rule_id.clone())
        } else {
            if let Some(ref mut c) = self.delete_confirm {
                c.mark_rejected();
            }
            None
        }
    }

    /// Cancel and close the delete confirmation modal.
    pub fn cancel_delete(&mut self) {
        self.delete_confirm = None;
    }

    /// Set a transient status message shown in the summary bar.
    pub fn set_message(&mut self, msg: String) {
        self.message = Some(msg);
    }

    /// Force a data refresh on the next tick.
    pub fn force_refresh(&mut self) {
        self.last_refresh = None;
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

        // Summary bar
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
        // Typed-name deletion confirm modal (on top of everything)
        if let Some(ref tc) = self.delete_confirm {
            draw_typed_confirm(frame, frame.area(), tc);
        }
    }
}

/// Sort key for a rule: the global label for global rules, otherwise the
/// resolved project name. Used to group/sort rules by tenant.
fn rule_tenant_key(rule: &RuleRow, names: &HashMap<String, String>) -> String {
    if rule.scope == TENANT_GLOBAL || rule.tenant_id.is_empty() {
        TENANT_GLOBAL.to_string()
    } else {
        tenants::display_name(names, &rule.tenant_id)
    }
}

// Re-export draw helpers so tests (via `use super::*`) can reach them.
#[cfg(test)]
pub(super) use super::rules_draw::{format_short_date, truncate_str};

#[cfg(test)]
#[path = "rules_tests.rs"]
mod tests;

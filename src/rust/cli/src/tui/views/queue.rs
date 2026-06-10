//! Queue browser view with scrollable list, status filtering, and detail popup.
//!
//! Data is fetched from the local SQLite state database (read-only) and
//! refreshes on each tick event with a minimum 2-second interval.

use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::Frame;

use regex::Regex;

use super::confirm::{draw_action_confirm, ActionConfirm, SimpleConfirm};
use super::queue_data::{
    fetch_queue_detail, fetch_queue_rows, QueueDetail, QueueRow, StatusFilter,
};
use crate::tui::filter::{self, FilterState};
use crate::tui::search::SearchState;

/// Build the text a filter or search matches against for a queue row: the
/// shortened id, project, object path, item type, operation, and status.
pub(super) fn match_haystack(item: &QueueRow) -> String {
    format!(
        "{} {} {} {} {} {}",
        item.short_id, item.project, item.object, item.item_type, item.op, item.status
    )
}

/// Minimum interval between data refreshes.
const REFRESH_INTERVAL_MS: u128 = 2000;

/// Which queue action is awaiting confirmation for the selected item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueAction {
    Retry,
    Cancel,
    Remove,
}

/// Queue browser view state.
pub struct QueueBrowser {
    /// All rows from the last fetch (before page/global filtering).
    all_items: Vec<QueueRow>,
    /// Rows currently shown — `all_items` narrowed by the page + global filter.
    pub(super) items: Vec<QueueRow>,
    /// Index of the selected item in `items`.
    pub(super) selected: usize,
    /// Active SQL-side status filter (cycled with `s`).
    filter: StatusFilter,
    /// Per-page narrowing filter (`f`); composes (AND) with status and global.
    page_filter: FilterState,
    /// Compiled global filter pushed from the app; composes (AND) with the rest.
    global_re: Option<Regex>,
    /// Detail popup for the selected item, if open.
    pub(super) detail: Option<QueueDetail>,
    /// When data was last refreshed from SQLite.
    last_refresh: Option<Instant>,
    /// Cursor-jump search state (`/`).
    pub(super) search: SearchState,
    /// Pending action awaiting a `y`/`N` confirmation, if the modal is open.
    /// Captured at request time so a background refresh between opening the
    /// modal and confirming cannot retarget the action: retry/remove act on
    /// the `queue_id`, cancel acts tenant-wide on the `tenant_id`.
    pending: Option<(QueueAction, String, String)>, // (action, queue_id, tenant_id)
    /// Transient status message shown in the summary bar after an action.
    pub(super) message: Option<String>,
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
            pending: None,
            message: None,
        }
    }

    pub fn search_mut(&mut self) -> &mut SearchState {
        &mut self.search
    }

    pub fn search_active(&self) -> bool {
        self.search.active
    }

    // ── Draw accessors (used by queue_draw.rs) ──────────────────────────────

    /// The filter label string for the status bar.
    pub(super) fn filter_label(&self) -> &'static str {
        self.filter.label()
    }

    /// Number of currently visible items.
    pub(super) fn item_count(&self) -> usize {
        self.items.len()
    }

    /// Read-only reference to the page filter (for prompt rendering).
    pub(super) fn page_filter_ref(&self) -> &FilterState {
        &self.page_filter
    }

    /// Read-only reference to the search state (for prompt rendering).
    pub(super) fn search_ref(&self) -> &SearchState {
        &self.search
    }

    /// Number of items matching the current search pattern.
    pub(super) fn search_match_count(&self) -> usize {
        self.search_matches().len()
    }

    /// Optional reference to the transient status message.
    pub(super) fn message_ref(&self) -> Option<&str> {
        self.message.as_deref()
    }

    /// Index of the currently selected item.
    pub(super) fn selected_index(&self) -> usize {
        self.selected
    }

    /// Slice of the currently visible items.
    pub(super) fn items_slice(&self) -> &[QueueRow] {
        &self.items
    }

    // ── Action confirm ──────────────────────────────────────────────────────

    /// Whether the action-confirmation modal is open.
    pub fn confirm_open(&self) -> bool {
        self.pending.is_some()
    }

    /// Build the [`ActionConfirm`] to display from the current pending action.
    pub fn action_confirm(&self) -> Option<ActionConfirm> {
        self.pending.as_ref().map(|(action, _, _)| {
            let (verb, target) = match action {
                QueueAction::Retry => ("Retry", "selected queue item"),
                QueueAction::Cancel => ("Cancel pending items for", "selected project"),
                QueueAction::Remove => ("Remove", "selected queue item"),
            };
            ActionConfirm::Simple(SimpleConfirm {
                verb: verb.to_string(),
                target: target.to_string(),
            })
        })
    }

    /// Open a confirmation modal for a queue action on the selected item,
    /// capturing both its queue_id and tenant_id.
    pub fn request_action(&mut self, action: QueueAction) {
        if let Some(item) = self.items.get(self.selected) {
            self.pending = Some((action, item.queue_id.clone(), item.tenant_id.clone()));
        }
    }

    /// Take the pending action (action kind + queue_id + tenant_id), clearing
    /// the modal. Returns `None` if no action is pending.
    pub fn take_action(&mut self) -> Option<(QueueAction, String, String)> {
        self.pending.take()
    }

    /// Cancel and close the action-confirmation modal.
    pub fn cancel_action(&mut self) {
        self.pending = None;
    }

    /// Set a transient status message shown in the summary bar.
    pub fn set_message(&mut self, msg: String) {
        self.message = Some(msg);
    }

    /// Force a data refresh on the next tick (after a state-changing action).
    pub fn force_refresh(&mut self) {
        self.last_refresh = None;
    }

    // ── Filter / page filter ────────────────────────────────────────────────

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
    pub(super) fn search_matches(&self) -> Vec<usize> {
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
        if let Some(ref confirm) = self.action_confirm() {
            draw_action_confirm(frame, frame.area(), confirm);
        }
    }
}

// Re-export draw helpers so tests (via `use super::*`) can reach them.
#[cfg(test)]
pub(super) use super::queue_draw::{age_color, status_color, truncate_str};

#[cfg(test)]
pub(crate) use ratatui::style::Color;

#[cfg(test)]
#[path = "queue_tests.rs"]
mod tests;

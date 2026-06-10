//! Project browser view with scrollable list, active/inactive indicators,
//! and a detail popup showing watch folders and queue breakdown.
//!
//! Data is fetched from the local SQLite state database (read-only) and
//! refreshes on each tick event with a minimum 5-second interval.

use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::Frame;

use regex::Regex;

use super::confirm::{
    draw_action_confirm, draw_toggle_confirm, ActionConfirm, SimpleConfirm, ToggleConfirm,
};
use super::file_list::{handle_popup_key, FileListAction, FileListState, LibraryMode};
use super::file_list_data::fetch_file_entries;
use super::projects_data::{fetch_project_detail, fetch_project_rows, ProjectDetail, ProjectRow};
use crate::tui::filter::{self, FilterState};
use crate::tui::search::SearchState;

/// Build the text a filter or search matches against for a project row: name,
/// path, and current branch.
pub(super) fn match_haystack(item: &ProjectRow) -> String {
    format!("{} {} {}", item.name, item.display_path, item.branch)
}

/// Minimum interval between data refreshes (projects change infrequently).
const REFRESH_INTERVAL_MS: u128 = 5000;

/// Project browser view state.
pub struct ProjectBrowser {
    /// All projects from the last fetch (before page/global filtering).
    all_items: Vec<ProjectRow>,
    /// Projects currently shown — `all_items` narrowed by page + global filter.
    pub(super) items: Vec<ProjectRow>,
    /// Index of the selected item in `items`.
    pub(super) selected: usize,
    /// Per-page narrowing filter (`f`); composes (AND) with the global filter.
    page_filter: FilterState,
    /// Compiled global filter pushed from the app.
    global_re: Option<Regex>,
    /// Detail popup for the selected project, if open.
    pub(super) detail: Option<ProjectDetail>,
    /// File-list tab and content overlay state for the detail popup.
    pub(in crate::tui::views) file_list: FileListState,
    /// When data was last refreshed from SQLite.
    last_refresh: Option<Instant>,
    /// Cursor-jump search state (`/`).
    pub(super) search: SearchState,
    /// Pending tracking-toggle confirmation, if the modal is open.
    confirm: Option<ToggleConfirm>,
    /// Pending nudge (rescan) confirmation, if the modal is open. Captures
    /// (watch_id, display name) at request time so a periodic refresh that
    /// rebuilds/reorders `items` cannot retarget the rescan.
    nudge_confirm: Option<(String, String)>, // (watch_id, name)
    /// Transient status message (e.g. toggle result), shown in the header.
    pub(super) message: Option<String>,
}

impl ProjectBrowser {
    /// Create a new, empty project browser. Data loads on the first tick.
    pub fn new() -> Self {
        Self {
            all_items: Vec::new(),
            items: Vec::new(),
            selected: 0,
            page_filter: FilterState::new(),
            global_re: None,
            detail: None,
            file_list: FileListState::new(),
            last_refresh: None,
            search: SearchState::new(),
            confirm: None,
            nudge_confirm: None,
            message: None,
        }
    }

    // ── Draw accessors (used by projects_draw.rs) ───────────────────────────

    /// Slice of the currently visible items.
    pub(super) fn items_slice(&self) -> &[ProjectRow] {
        &self.items
    }

    /// Index of the currently selected item.
    pub(super) fn selected_index(&self) -> usize {
        self.selected
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
        let filtered: Vec<ProjectRow> = self
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

    /// Whether any confirmation modal (toggle or nudge) is open.
    pub fn confirm_open(&self) -> bool {
        self.confirm.is_some() || self.nudge_confirm.is_some()
    }

    /// Whether specifically the nudge-confirmation modal is open.
    pub fn nudge_confirm_open(&self) -> bool {
        self.nudge_confirm.is_some()
    }

    /// Open a `y`/`N` nudge confirmation for the selected project, capturing
    /// its watch_id and name so a refresh cannot retarget the rescan.
    pub fn request_nudge(&mut self) {
        if let Some(item) = self.items.get(self.selected) {
            self.nudge_confirm = Some((item.watch_id.clone(), item.name.clone()));
        }
    }

    /// Build the [`ActionConfirm`] for the nudge modal from the captured target.
    pub fn nudge_action_confirm(&self) -> Option<ActionConfirm> {
        self.nudge_confirm.as_ref().map(|(_, name)| {
            ActionConfirm::Simple(SimpleConfirm {
                verb: "Rescan".to_string(),
                target: name.clone(),
            })
        })
    }

    /// Take the nudge confirmation (clears the modal) and return the tenant_id
    /// captured when the modal opened.
    pub fn take_nudge(&mut self) -> Option<String> {
        self.nudge_confirm.take().map(|(watch_id, _)| watch_id)
    }

    /// Cancel the nudge confirmation modal.
    pub fn cancel_nudge(&mut self) {
        self.nudge_confirm = None;
    }

    /// Open a confirmation to toggle tracking for the selected project.
    pub fn request_toggle(&mut self) {
        if let Some(item) = self.items.get(self.selected) {
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

    /// Refresh data from SQLite if enough time has elapsed.
    pub fn on_tick(&mut self) {
        let should_refresh = self
            .last_refresh
            .map_or(true, |t| t.elapsed().as_millis() >= REFRESH_INTERVAL_MS);

        if should_refresh {
            self.all_items = fetch_project_rows();
            // Active projects first, then by natural (case-insensitive) name.
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

    /// Open the detail popup for the currently selected project.
    ///
    /// Loads the metadata and pre-fetches the file list for the Files tab so
    /// the first tab switch is instant.
    pub fn open_detail(&mut self) {
        if let Some(item) = self.items.get(self.selected) {
            self.detail = fetch_project_detail(&item.watch_id);
            self.file_list.reset();
            // Pre-load files so the tab is ready immediately.
            let entries = fetch_file_entries(&item.watch_id);
            self.file_list.load(entries);
        }
    }

    /// Close the detail popup and reset the file-list state.
    pub fn close_detail(&mut self) {
        self.detail = None;
        self.file_list.reset();
    }

    /// Route a key event to the file-list state machine while the detail popup
    /// is open. Returns the action the caller should take.
    ///
    /// Projects are not libraries, so `d` on a file is a no-op.
    pub fn handle_popup_key(&mut self, key: crossterm::event::KeyCode) -> FileListAction {
        handle_popup_key(&mut self.file_list, key, LibraryMode::NotLibrary)
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
    pub(super) fn search_matches(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, item)| self.search.is_match(&match_haystack(item)))
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
        if let Some(ref confirm) = self.confirm {
            draw_toggle_confirm(frame, frame.area(), confirm);
        }
        if let Some(ref nudge) = self.nudge_action_confirm() {
            draw_action_confirm(frame, frame.area(), nudge);
        }
    }
}

// Re-export popup helpers so tests (via `super::*`) can reach them.
#[cfg(test)]
pub(crate) use super::projects_popup::{status_color, truncate_str};

#[cfg(test)]
#[path = "projects_tests.rs"]
mod tests;

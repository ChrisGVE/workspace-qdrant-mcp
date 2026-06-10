//! Library browser view with scrollable list and detail popup.
//!
//! Data is fetched from the local SQLite state database (read-only) and
//! refreshes on each tick event with a minimum 5-second interval.

use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::Frame;

use regex::Regex;

use super::confirm::{
    draw_action_confirm, draw_toggle_confirm, draw_typed_confirm, ActionConfirm, SimpleConfirm,
    ToggleConfirm, TypedConfirm,
};
use super::file_list::{handle_popup_key, FileListAction, FileListState, LibraryMode};
use super::file_list_data::fetch_file_entries;
use super::libraries_data::{fetch_library_detail, fetch_library_rows, LibraryDetail, LibraryRow};
use crate::tui::filter::{self, FilterState};
use crate::tui::search::SearchState;

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
    /// File-list tab and content overlay state for the detail popup.
    pub(in crate::tui::views) file_list: FileListState,
    /// When data was last refreshed from SQLite.
    last_refresh: Option<Instant>,
    /// Cursor-jump search state (`/`).
    search: SearchState,
    /// Pending tracking-toggle confirmation, if the modal is open.
    confirm: Option<ToggleConfirm>,
    /// Pending nudge (rescan) confirmation, if the modal is open. Captures
    /// (tag, display name) at request time so a periodic refresh that
    /// rebuilds/reorders `items` cannot retarget the rescan.
    nudge_confirm: Option<(String, String)>, // (tag, name)
    /// Pending typed-name book-removal confirm (incremental libs only).
    book_remove_confirm: Option<(TypedConfirm, String)>, // (confirm, abs_path)
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
            file_list: FileListState::new(),
            last_refresh: None,
            search: SearchState::new(),
            confirm: None,
            nudge_confirm: None,
            book_remove_confirm: None,
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

    /// Whether any confirmation modal (toggle, nudge, or book-removal) is open.
    pub fn confirm_open(&self) -> bool {
        self.confirm.is_some() || self.nudge_confirm.is_some() || self.book_remove_confirm.is_some()
    }

    /// Whether specifically the nudge-confirmation modal is open.
    pub fn nudge_confirm_open(&self) -> bool {
        self.nudge_confirm.is_some()
    }

    /// Open a `y`/`N` nudge confirmation for the selected library, capturing
    /// its tag and name so a refresh cannot retarget the rescan.
    pub fn request_nudge(&mut self) {
        if let Some(item) = self.items.get(self.selected) {
            self.nudge_confirm = Some((item.tag.clone(), item.name.clone()));
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
        self.nudge_confirm.take().map(|(tag, _)| tag)
    }

    /// Cancel the nudge confirmation modal.
    pub fn cancel_nudge(&mut self) {
        self.nudge_confirm = None;
    }

    // ── Book removal (incremental libraries) ────────────────────────────────

    /// Whether the book-removal confirm modal is open.
    pub fn book_remove_confirm_open(&self) -> bool {
        self.book_remove_confirm.is_some()
    }

    /// Open a typed-name confirm for removing a library book at `abs_path`.
    ///
    /// The target name shown in the confirm is the file's base name (the last
    /// path component), which is what the user types after `Delete `.
    pub fn request_book_remove(&mut self, abs_path: String) {
        let file_name = abs_path
            .trim_end_matches('/')
            .rsplit('/')
            .find(|s| !s.is_empty())
            .unwrap_or(&abs_path)
            .to_string();
        self.book_remove_confirm = Some((TypedConfirm::new(file_name), abs_path));
    }

    /// Mutable reference to the [`TypedConfirm`] inside the book-removal modal.
    pub fn book_remove_confirm_mut(&mut self) -> Option<&mut TypedConfirm> {
        self.book_remove_confirm.as_mut().map(|(tc, _)| tc)
    }

    /// If the typed input matches, consume the confirm and return the abs_path.
    /// If the input does not match, mark it rejected and return `None`.
    pub fn take_book_remove_if_confirmed(&mut self) -> Option<String> {
        let matched = self
            .book_remove_confirm
            .as_ref()
            .map_or(false, |(tc, _)| tc.matches());
        if matched {
            self.book_remove_confirm.take().map(|(_, path)| path)
        } else {
            if let Some((tc, _)) = &mut self.book_remove_confirm {
                tc.mark_rejected();
            }
            None
        }
    }

    /// Cancel the book-removal confirm modal.
    pub fn cancel_book_remove(&mut self) {
        self.book_remove_confirm = None;
    }

    /// Return the [`LibraryMode`] for the currently selected library.
    ///
    /// Used to gate `d` (remove book) to incremental libraries only.
    pub fn selected_library_mode(&self) -> LibraryMode {
        match self.items.get(self.selected) {
            Some(item) if item.mode == "incremental" => LibraryMode::Incremental,
            Some(_) => LibraryMode::Sync,
            None => LibraryMode::NotLibrary,
        }
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

    /// Open the detail popup for the currently selected library.
    ///
    /// Pre-fetches the file list for the Files tab so the first tab switch is
    /// instant. Libraries currently have 0 rows in the live DB — this is
    /// code-complete but unverified until a library exists.
    pub fn open_detail(&mut self) {
        if let Some(item) = self.items.get(self.selected) {
            self.detail = fetch_library_detail(&item.watch_id);
            self.file_list.reset();
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
    /// is open.
    ///
    /// Passes the [`LibraryMode`] of the selected library so that the `d` key
    /// is only accepted for incremental libraries.  If the file-list returns
    /// [`FileListAction::RequestBookRemove`] the caller must route back through
    /// [`Self::request_book_remove`] / [`Self::cancel_book_remove`].
    pub fn handle_popup_key(&mut self, key: crossterm::event::KeyCode) -> FileListAction {
        let mode = self.selected_library_mode();
        handle_popup_key(&mut self.file_list, key, mode)
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
        if let Some(ref nudge) = self.nudge_action_confirm() {
            draw_action_confirm(frame, frame.area(), nudge);
        }
        if let Some((ref tc, _)) = self.book_remove_confirm {
            draw_typed_confirm(frame, frame.area(), tc);
        }
    }

    /// Draw the header bar above the table.
    fn draw_header_bar(&self, frame: &mut Frame, area: Rect) {
        super::libraries_draw::draw_header_bar(
            frame,
            area,
            self.items.len(),
            &self.page_filter,
            &self.search,
            self.search_matches().len(),
            self.message.as_deref(),
        );
    }

    /// Draw the scrollable table of library items.
    fn draw_table(&self, frame: &mut Frame, area: Rect) {
        super::libraries_draw::draw_table(frame, area, &self.items, self.selected, &self.search);
    }
}

#[cfg(test)]
#[path = "libraries_tests.rs"]
mod tests;

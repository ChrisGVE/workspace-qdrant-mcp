//! Per-view key handlers for Queue, Projects, and Libraries list views.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::super::filter::FilterAction;
use super::super::search::SearchAction;
use super::super::views::file_list::FileListAction;
use super::App;

impl App {
    /// Handle queue-specific keys. Returns true if the key was consumed.
    pub(super) fn handle_queue_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let (half, full) = self.nav_steps();
        let browser = self.queue_browser();

        if browser.page_filter_active() {
            if ctrl && key.code == KeyCode::Char('c') {
                browser.clear_page_filter();
                return true;
            }
            if matches!(
                browser.page_filter_mut().handle_key(key.code),
                FilterAction::Applied | FilterAction::Cancelled
            ) {
                browser.recompute_visible();
            }
            return true;
        }

        if browser.search_active() {
            if browser.search_mut().handle_key(key.code) == SearchAction::Confirmed {
                browser.search_first();
            }
            return true;
        }

        if browser.detail_open() {
            if key.code == KeyCode::Esc {
                browser.close_detail();
            }
            return true;
        }

        match key.code {
            KeyCode::Char('/') => {
                browser.search_mut().activate();
                true
            }
            KeyCode::Char('j') | KeyCode::Down => {
                browser.select_next();
                true
            }
            KeyCode::Char('k') | KeyCode::Up => {
                browser.select_prev();
                true
            }
            KeyCode::Char('d') if ctrl => {
                browser.page_down(half);
                true
            }
            KeyCode::Char('u') if ctrl => {
                browser.page_up(half);
                true
            }
            KeyCode::Char('f') if ctrl => {
                browser.page_down(full);
                true
            }
            KeyCode::Char('b') if ctrl => {
                browser.page_up(full);
                true
            }
            KeyCode::PageUp => {
                browser.page_up(full);
                true
            }
            KeyCode::PageDown => {
                browser.page_down(full);
                true
            }
            KeyCode::Char('n') => {
                browser.search_next();
                true
            }
            KeyCode::Char('N') => {
                browser.search_prev();
                true
            }
            KeyCode::Char('g') => {
                browser.jump_first();
                true
            }
            KeyCode::Char('G') => {
                browser.jump_last();
                true
            }
            KeyCode::Enter => {
                browser.open_detail();
                true
            }
            KeyCode::Esc => {
                browser.close_detail();
                true
            }
            KeyCode::Char('s') => {
                browser.cycle_filter();
                true
            }
            KeyCode::Char('f') => {
                browser.page_filter_mut().activate();
                true
            }
            _ => false,
        }
    }

    /// Handle project-specific keys. Returns true if the key was consumed.
    pub(super) fn handle_project_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let (half, full) = self.nav_steps();
        let browser = self.project_browser();

        // The page-filter prompt captures all input while active.
        if browser.page_filter_active() {
            if ctrl && key.code == KeyCode::Char('c') {
                browser.clear_page_filter();
                return true;
            }
            if matches!(
                browser.page_filter_mut().handle_key(key.code),
                FilterAction::Applied | FilterAction::Cancelled
            ) {
                browser.recompute_visible();
            }
            return true;
        }

        // When search is active, delegate all keys to search
        if browser.search_active() {
            if browser.search_mut().handle_key(key.code) == SearchAction::Confirmed {
                browser.search_first();
            }
            return true;
        }

        // When the toggle-confirmation modal is open, y/Enter confirms and runs
        // the daemon-side enable/disable; n/Esc cancels.
        if browser.confirm_open() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                    if let Some((watch_id, enable)) = browser.take_confirm() {
                        let msg = super::super::commands::set_watch_enabled(&watch_id, enable);
                        browser.set_message(msg);
                        browser.force_refresh();
                    }
                }
                _ => browser.cancel_confirm(),
            }
            return true;
        }

        // When the detail popup is open, route all keys through the file-list
        // state machine. It handles Tab/Shift+Tab (tab switching), j/k/g/G
        // (file cursor), Enter (content overlay), and Esc (close overlay or
        // popup). Keys not consumed by the state machine are silently swallowed
        // while the popup is open so they don't leak to global bindings.
        if browser.detail_open() {
            match browser.handle_popup_key(key.code) {
                FileListAction::ClosePopup => browser.close_detail(),
                FileListAction::Consumed | FileListAction::NotConsumed => {}
            }
            return true;
        }

        match key.code {
            KeyCode::Char('/') => {
                browser.search_mut().activate();
                true
            }
            KeyCode::Char('j') | KeyCode::Down => {
                browser.select_next();
                true
            }
            KeyCode::Char('k') | KeyCode::Up => {
                browser.select_prev();
                true
            }
            KeyCode::Char('d') if ctrl => {
                browser.page_down(half);
                true
            }
            KeyCode::Char('u') if ctrl => {
                browser.page_up(half);
                true
            }
            KeyCode::Char('f') if ctrl => {
                browser.page_down(full);
                true
            }
            KeyCode::Char('b') if ctrl => {
                browser.page_up(full);
                true
            }
            KeyCode::PageUp => {
                browser.page_up(full);
                true
            }
            KeyCode::PageDown => {
                browser.page_down(full);
                true
            }
            KeyCode::Char('n') => {
                browser.search_next();
                true
            }
            KeyCode::Char('N') => {
                browser.search_prev();
                true
            }
            KeyCode::Char('g') => {
                browser.jump_first();
                true
            }
            KeyCode::Char('G') => {
                browser.jump_last();
                true
            }
            KeyCode::Char('t') => {
                browser.request_toggle();
                true
            }
            KeyCode::Char('f') => {
                browser.page_filter_mut().activate();
                true
            }
            KeyCode::Enter => {
                browser.open_detail();
                true
            }
            KeyCode::Esc => {
                browser.close_detail();
                true
            }
            _ => false,
        }
    }

    /// Handle library-specific keys. Returns true if the key was consumed.
    pub(super) fn handle_library_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let (half, full) = self.nav_steps();
        let browser = self.library_browser();

        if browser.page_filter_active() {
            if ctrl && key.code == KeyCode::Char('c') {
                browser.clear_page_filter();
                return true;
            }
            if matches!(
                browser.page_filter_mut().handle_key(key.code),
                FilterAction::Applied | FilterAction::Cancelled
            ) {
                browser.recompute_visible();
            }
            return true;
        }

        if browser.search_active() {
            if browser.search_mut().handle_key(key.code) == SearchAction::Confirmed {
                browser.search_first();
            }
            return true;
        }

        // Toggle-confirmation modal: y/Enter confirms, anything else cancels.
        if browser.confirm_open() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                    if let Some((watch_id, enable)) = browser.take_confirm() {
                        let msg = super::super::commands::set_watch_enabled(&watch_id, enable);
                        browser.set_message(msg);
                        browser.force_refresh();
                    }
                }
                _ => browser.cancel_confirm(),
            }
            return true;
        }

        // Detail popup: same tab/file-list routing as the Projects view.
        if browser.detail_open() {
            match browser.handle_popup_key(key.code) {
                FileListAction::ClosePopup => browser.close_detail(),
                FileListAction::Consumed | FileListAction::NotConsumed => {}
            }
            return true;
        }

        match key.code {
            KeyCode::Char('/') => {
                browser.search_mut().activate();
                true
            }
            KeyCode::Char('j') | KeyCode::Down => {
                browser.select_next();
                true
            }
            KeyCode::Char('k') | KeyCode::Up => {
                browser.select_prev();
                true
            }
            KeyCode::Char('d') if ctrl => {
                browser.page_down(half);
                true
            }
            KeyCode::Char('u') if ctrl => {
                browser.page_up(half);
                true
            }
            KeyCode::Char('f') if ctrl => {
                browser.page_down(full);
                true
            }
            KeyCode::Char('b') if ctrl => {
                browser.page_up(full);
                true
            }
            KeyCode::PageUp => {
                browser.page_up(full);
                true
            }
            KeyCode::PageDown => {
                browser.page_down(full);
                true
            }
            KeyCode::Char('n') => {
                browser.search_next();
                true
            }
            KeyCode::Char('N') => {
                browser.search_prev();
                true
            }
            KeyCode::Char('g') => {
                browser.jump_first();
                true
            }
            KeyCode::Char('G') => {
                browser.jump_last();
                true
            }
            KeyCode::Char('t') => {
                browser.request_toggle();
                true
            }
            KeyCode::Char('f') => {
                browser.page_filter_mut().activate();
                true
            }
            KeyCode::Enter => {
                browser.open_detail();
                true
            }
            KeyCode::Esc => {
                browser.close_detail();
                true
            }
            _ => false,
        }
    }
}

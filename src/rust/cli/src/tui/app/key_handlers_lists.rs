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

        // Action confirmation modal (retry/cancel/remove): y/Enter confirms,
        // anything else cancels.
        if browser.confirm_open() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                    if let Some((action, queue_id)) = browser.take_action() {
                        use super::super::views::queue::QueueAction;
                        let msg = match action {
                            QueueAction::Retry => super::super::commands::queue_retry(&queue_id),
                            QueueAction::Remove => super::super::commands::queue_remove(&queue_id),
                            QueueAction::Cancel => {
                                // queue_id holds queue_id; resolve tenant_id
                                // from the items list before the action.
                                super::super::commands::queue_cancel(&queue_id)
                            }
                        };
                        browser.set_message(msg);
                        browser.force_refresh();
                    }
                }
                _ => browser.cancel_action(),
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
            // Queue actions: require confirmation before executing.
            KeyCode::Char('r') => {
                browser.request_action(super::super::views::queue::QueueAction::Retry);
                true
            }
            KeyCode::Char('c') => {
                browser.request_action(super::super::views::queue::QueueAction::Cancel);
                true
            }
            KeyCode::Char('x') => {
                browser.request_action(super::super::views::queue::QueueAction::Remove);
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

        // Nudge confirm (y/N rescan) takes priority over toggle confirm.
        if browser.nudge_confirm_open() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                    if let Some(tenant_id) = browser.take_nudge() {
                        let msg = super::super::commands::project_nudge(&tenant_id);
                        browser.set_message(msg);
                        browser.force_refresh();
                    }
                }
                _ => browser.cancel_nudge(),
            }
            return true;
        }

        // Toggle-confirmation modal: y/Enter confirms and runs enable/disable.
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
        // Projects have no library mode, so RequestBookRemove is never returned.
        if browser.detail_open() {
            match browser.handle_popup_key(key.code) {
                FileListAction::ClosePopup => browser.close_detail(),
                FileListAction::Consumed
                | FileListAction::NotConsumed
                | FileListAction::RequestBookRemove(_) => {}
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
            KeyCode::Char('r') => {
                browser.request_nudge();
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

        // Book-removal typed confirm: each key is forwarded to the TypedConfirm
        // input buffer; Enter attempts confirmation, Esc cancels.
        if browser.book_remove_confirm_open() {
            match key.code {
                KeyCode::Char(c) => {
                    if let Some(tc) = browser.book_remove_confirm_mut() {
                        tc.push_char(c);
                    }
                }
                KeyCode::Backspace => {
                    if let Some(tc) = browser.book_remove_confirm_mut() {
                        tc.pop_char();
                    }
                }
                KeyCode::Enter => {
                    if let Some(abs_path) = browser.take_book_remove_if_confirmed() {
                        let msg = super::super::commands::library_book_remove(&abs_path);
                        browser.set_message(msg);
                        browser.force_refresh();
                    }
                }
                KeyCode::Esc => browser.cancel_book_remove(),
                _ => {}
            }
            return true;
        }

        // Nudge confirm (y/N rescan) takes priority over toggle confirm.
        if browser.nudge_confirm_open() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                    if let Some(tenant_id) = browser.take_nudge() {
                        let msg = super::super::commands::library_nudge(&tenant_id);
                        browser.set_message(msg);
                        browser.force_refresh();
                    }
                }
                _ => browser.cancel_nudge(),
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

        // Detail popup: route keys through file-list state machine.
        // For libraries, `d` on a file in the Files tab may produce
        // RequestBookRemove — route it to the book-removal confirm flow.
        if browser.detail_open() {
            let action = browser.handle_popup_key(key.code);
            match action {
                FileListAction::ClosePopup => browser.close_detail(),
                FileListAction::RequestBookRemove(path) => {
                    if path == "__sync_blocked__" {
                        browser.set_message(
                            "Sync library books are managed by the watcher; not removable here"
                                .to_string(),
                        );
                    } else {
                        browser.request_book_remove(path);
                    }
                }
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
            KeyCode::Char('r') => {
                browser.request_nudge();
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

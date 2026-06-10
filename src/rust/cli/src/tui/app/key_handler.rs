//! Key event handling and tick dispatch for the TUI application.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::super::filter::FilterAction;
use super::super::search::SearchAction;
use super::super::views::dashboard::FocusedCell;
use super::{App, View};

impl App {
    /// Handle periodic tick events for live-updating views.
    pub(super) fn on_tick(&mut self) {
        // Re-push the global filter each tick so it also applies after switching
        // views and after a live data refresh (cheap; the list is ≤200 rows).
        self.push_global_filter();
        match self.current_view {
            View::Dashboard => {
                self.dashboard().on_tick();
            }
            View::Queue => {
                self.queue_browser().on_tick();
            }
            View::Projects => {
                self.project_browser().on_tick();
            }
            View::Libraries => {
                self.library_browser().on_tick();
            }
            View::Rules => {
                self.rule_browser().on_tick();
            }
            View::Scratchpad => {
                self.scratchpad_browser().on_tick();
            }
            View::Service => {
                self.service_view().on_tick();
            }
            View::Logs => {
                self.log_viewer().on_tick();
            }
        }
    }

    /// Handle a key event with global bindings.
    pub(super) fn handle_key(&mut self, key: KeyEvent) {
        // Help overlay captures Esc and ? only
        if self.show_help {
            match key.code {
                KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('q') => {
                    self.show_help = false;
                }
                _ => {}
            }
            return;
        }

        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

        // The global-filter prompt captures all input while active. Ctrl+C
        // clears the filter (rather than quitting); Enter applies, Esc cancels.
        if self.global_filter_active() {
            if ctrl && key.code == KeyCode::Char('c') {
                self.global_filter_mut().clear();
                self.push_global_filter();
                return;
            }
            match self.global_filter_mut().handle_key(key.code) {
                FilterAction::Applied | FilterAction::Cancelled => self.push_global_filter(),
                FilterAction::Editing => {}
            }
            return;
        }

        // Shift+F opens the global filter from any view.
        if key.code == KeyCode::Char('F') && !ctrl {
            self.global_filter_mut().activate();
            return;
        }

        // Delegate keys to the dashboard when on Dashboard view
        if self.current_view == View::Dashboard {
            if self.handle_dashboard_key(key) {
                return;
            }
        }

        // Delegate keys to the queue browser when on Queue view
        if self.current_view == View::Queue {
            if self.handle_queue_key(key) {
                return;
            }
        }

        // Delegate keys to the project browser when on Projects view
        if self.current_view == View::Projects {
            if self.handle_project_key(key) {
                return;
            }
        }

        // Delegate keys to the library browser when on Libraries view
        if self.current_view == View::Libraries {
            if self.handle_library_key(key) {
                return;
            }
        }

        // Delegate keys to the rule browser when on Rules view
        if self.current_view == View::Rules {
            if self.handle_rule_key(key) {
                return;
            }
        }

        // Delegate keys to the scratchpad browser when on Scratchpad view
        if self.current_view == View::Scratchpad {
            if self.handle_scratchpad_key(key) {
                return;
            }
        }

        // Delegate keys to the service view
        if self.current_view == View::Service {
            if self.handle_service_key(key) {
                return;
            }
        }

        // Delegate scrolling keys to the log viewer when on Logs view
        if self.current_view == View::Logs {
            if self.handle_log_key(key) {
                return;
            }
        }

        self.handle_global_key(key);
    }

    /// Handle dashboard-specific keys. Returns true if the key was consumed.
    fn handle_dashboard_key(&mut self, key: KeyEvent) -> bool {
        let dash = self.dashboard();

        // When popup is open, handle popup keys
        if dash.popup_open() {
            match key.code {
                KeyCode::Esc => dash.close_popup(),
                KeyCode::Char('j') | KeyCode::Down => dash.popup_scroll_down(),
                KeyCode::Char('k') | KeyCode::Up => dash.popup_scroll_up(),
                _ => {}
            }
            return true;
        }

        // Cell focus shortcuts
        match key.code {
            KeyCode::Char('p') | KeyCode::Char('P') => {
                dash.focused = FocusedCell::Projects;
                true
            }
            KeyCode::Char('l') | KeyCode::Char('L') => {
                dash.focused = FocusedCell::Libraries;
                true
            }
            KeyCode::Char('s') | KeyCode::Char('S') => {
                dash.focused = FocusedCell::Scratchpad;
                true
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                dash.focused = FocusedCell::Rules;
                true
            }
            KeyCode::Char('a') | KeyCode::Char('A') => {
                dash.focused = FocusedCell::ActiveProjects;
                true
            }
            KeyCode::Char('e') | KeyCode::Char('E') => {
                dash.focused = FocusedCell::Errors;
                true
            }
            // Navigation within focused cell
            KeyCode::Char('j') | KeyCode::Down if dash.focused != FocusedCell::None => {
                dash.select_next();
                true
            }
            KeyCode::Char('k') | KeyCode::Up if dash.focused != FocusedCell::None => {
                dash.select_prev();
                true
            }
            KeyCode::Enter if dash.focused != FocusedCell::None => {
                dash.open_popup();
                true
            }
            KeyCode::Esc => {
                dash.focused = FocusedCell::None;
                true
            }
            _ => false,
        }
    }

    /// Handle queue-specific keys. Returns true if the key was consumed.
    fn handle_queue_key(&mut self, key: KeyEvent) -> bool {
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
    fn handle_project_key(&mut self, key: KeyEvent) -> bool {
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

        // When detail popup is open, only Esc closes it
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
    fn handle_library_key(&mut self, key: KeyEvent) -> bool {
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

    /// Handle rule-specific keys. Returns true if the key was consumed.
    fn handle_rule_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let (half, full) = self.nav_steps();
        let browser = self.rule_browser();

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

    /// Handle scratchpad-specific keys. Returns true if the key was consumed.
    fn handle_scratchpad_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let (half, full) = self.nav_steps();
        let browser = self.scratchpad_browser();

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
            match key.code {
                KeyCode::Esc => {
                    browser.close_detail();
                    return true;
                }
                KeyCode::Char('j') | KeyCode::Down => {
                    browser.scroll_detail_down();
                    return true;
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    browser.scroll_detail_up();
                    return true;
                }
                KeyCode::PageDown => {
                    browser.page_down(10);
                    return true;
                }
                KeyCode::PageUp => {
                    browser.page_up(10);
                    return true;
                }
                _ => return true,
            }
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

    /// Handle service-specific keys. Returns true if the key was consumed.
    fn handle_service_key(&mut self, key: KeyEvent) -> bool {
        match key.code {
            KeyCode::Char('p') | KeyCode::Char('P') => {
                let msg = super::super::commands::pause_watchers();
                self.service_view().last_message = Some(msg);
                true
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                let msg = super::super::commands::resume_watchers();
                self.service_view().last_message = Some(msg);
                true
            }
            _ => false,
        }
    }

    /// Handle log-specific keys. Returns true if the key was consumed.
    fn handle_log_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let (half, full) = self.nav_steps();
        let viewer = self.log_viewer();

        // When the entry popup is open, keys scroll or close it.
        if viewer.popup_open() {
            match key.code {
                KeyCode::Esc => viewer.close_popup(),
                KeyCode::Char('j') | KeyCode::Down => viewer.popup_scroll_down(),
                KeyCode::Char('k') | KeyCode::Up => viewer.popup_scroll_up(),
                _ => {}
            }
            return true;
        }

        // When the search prompt is active, route input to it.
        if viewer.search_active() {
            if viewer.search_mut().handle_key(key.code) == SearchAction::Confirmed {
                viewer.search_first();
            }
            return true;
        }

        match key.code {
            KeyCode::Char('/') => {
                viewer.search_mut().activate();
                true
            }
            KeyCode::Char('n') => {
                viewer.search_next();
                true
            }
            KeyCode::Char('N') => {
                viewer.search_prev();
                true
            }
            KeyCode::Char('j') | KeyCode::Down => {
                viewer.scroll_down();
                true
            }
            KeyCode::Char('k') | KeyCode::Up => {
                viewer.scroll_up();
                true
            }
            KeyCode::Char('d') if ctrl => {
                viewer.page_down(half);
                true
            }
            KeyCode::Char('u') if ctrl => {
                viewer.page_up(half);
                true
            }
            KeyCode::Char('f') if ctrl => {
                viewer.page_down(full);
                true
            }
            KeyCode::Char('b') if ctrl => {
                viewer.page_up(full);
                true
            }
            KeyCode::Enter => {
                viewer.open_popup();
                true
            }
            KeyCode::Char('g') => {
                viewer.jump_first();
                true
            }
            KeyCode::Char('G') => {
                viewer.jump_last();
                true
            }
            KeyCode::Esc => {
                viewer.scroll_to_bottom();
                true
            }
            KeyCode::PageUp => {
                viewer.page_up(full);
                true
            }
            KeyCode::PageDown => {
                viewer.page_down(full);
                true
            }
            _ => false,
        }
    }

    /// Handle global key bindings (quit, view switching, help).
    fn handle_global_key(&mut self, key: KeyEvent) {
        match key.code {
            // Quit
            KeyCode::Char('q') => self.running = false,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.running = false;
            }

            // View switching by number
            KeyCode::Char('1') => self.current_view = View::Dashboard,
            KeyCode::Char('2') => self.current_view = View::Queue,
            KeyCode::Char('3') => self.current_view = View::Projects,
            KeyCode::Char('4') => self.current_view = View::Libraries,
            KeyCode::Char('5') => self.current_view = View::Rules,
            KeyCode::Char('6') => self.current_view = View::Scratchpad,
            KeyCode::Char('7') => self.current_view = View::Service,
            KeyCode::Char('8') => self.current_view = View::Logs,

            // Tab navigation
            KeyCode::Tab => self.current_view = self.current_view.next(),
            KeyCode::BackTab => self.current_view = self.current_view.prev(),

            // Help
            KeyCode::Char('?') => self.show_help = true,

            _ => {}
        }
    }
}

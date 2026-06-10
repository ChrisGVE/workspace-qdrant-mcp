//! Key event handling and tick dispatch for the TUI application.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::super::filter::FilterAction;
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
            View::Graph => {
                self.graph_view().on_tick();
            }
            View::Search => {
                self.search_view().on_tick();
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

        // Delegate keys to the graph view when on Graph view
        if self.current_view == View::Graph {
            if self.handle_graph_key(key) {
                return;
            }
        }

        // Delegate keys to the search page when on Search view
        if self.current_view == View::Search {
            if self.handle_search_key(key) {
                return;
            }
        }

        self.handle_global_key(key);
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
            KeyCode::Char('9') => self.current_view = View::Graph,
            KeyCode::Char('0') => self.current_view = View::Search,

            // Tab navigation
            KeyCode::Tab => self.current_view = self.current_view.next(),
            KeyCode::BackTab => self.current_view = self.current_view.prev(),

            // Help
            KeyCode::Char('?') => self.show_help = true,

            _ => {}
        }
    }
}

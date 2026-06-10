//! TUI application core — event loop and state management.

mod key_handler;
mod render;

use std::time::Duration;

use super::event::{Event, EventHandler};
use super::filter::FilterState;
use super::terminal;
use super::views::dashboard::Dashboard;
use super::views::graph::GraphView;
use super::views::libraries::LibraryBrowser;
use super::views::logs::LogViewer;
use super::views::projects::ProjectBrowser;
use super::views::queue::QueueBrowser;
use super::views::rules::RuleBrowser;
use super::views::scratchpad::ScratchpadBrowser;
use super::views::service::ServiceView;

/// Active view in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Dashboard,
    Queue,
    Projects,
    Libraries,
    Rules,
    Scratchpad,
    Service,
    Logs,
    Graph,
}

impl View {
    /// All views in tab order.
    const ALL: [View; 9] = [
        View::Dashboard,
        View::Queue,
        View::Projects,
        View::Libraries,
        View::Rules,
        View::Scratchpad,
        View::Service,
        View::Logs,
        View::Graph,
    ];

    fn label(self) -> &'static str {
        match self {
            View::Dashboard => "Dashboard",
            View::Queue => "Queue",
            View::Projects => "Projects",
            View::Libraries => "Libraries",
            View::Rules => "Rules",
            View::Scratchpad => "Scratchpad",
            View::Service => "Service",
            View::Logs => "Logs",
            View::Graph => "Graph",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|v| *v == self).unwrap_or(0)
    }

    fn from_index(i: usize) -> Self {
        Self::ALL.get(i).copied().unwrap_or(View::Dashboard)
    }

    fn next(self) -> Self {
        Self::from_index((self.index() + 1) % Self::ALL.len())
    }

    fn prev(self) -> Self {
        let len = Self::ALL.len();
        Self::from_index((self.index() + len - 1) % len)
    }
}

/// TUI application state.
pub struct App {
    /// Whether the app is still running.
    pub running: bool,
    /// Currently active view.
    pub current_view: View,
    /// Whether the help overlay is visible.
    pub show_help: bool,
    /// Dashboard view state (lazily initialized on first Dashboard view).
    dashboard: Option<Dashboard>,
    /// Queue browser state (lazily initialized on first Queue view).
    queue_browser: Option<QueueBrowser>,
    /// Project browser state (lazily initialized on first Projects view).
    project_browser: Option<ProjectBrowser>,
    /// Library browser state (lazily initialized on first Libraries view).
    library_browser: Option<LibraryBrowser>,
    /// Rule browser state (lazily initialized on first Rules view).
    rule_browser: Option<RuleBrowser>,
    /// Scratchpad browser state (lazily initialized on first Scratchpad view).
    scratchpad_browser: Option<ScratchpadBrowser>,
    /// Service view state (lazily initialized on first Service view).
    service_view: Option<ServiceView>,
    /// Log viewer state (lazily initialized on first Logs view).
    log_viewer: Option<LogViewer>,
    /// Graph view state (lazily initialized on first Graph view).
    graph_view: Option<GraphView>,
    /// Height of the main content area from the last frame, used to size
    /// half/full-screen navigation. Updated during `draw`.
    content_height: std::cell::Cell<u16>,
    /// Global narrowing filter (`F`), applied across every list view in
    /// addition to each view's own page filter (`f`).
    global_filter: FilterState,
}

impl App {
    /// Create a new app instance.
    pub fn new(_daemon_addr: String) -> Self {
        Self {
            running: true,
            current_view: View::Dashboard,
            show_help: false,
            dashboard: None,
            queue_browser: None,
            project_browser: None,
            library_browser: None,
            rule_browser: None,
            scratchpad_browser: None,
            service_view: None,
            log_viewer: None,
            graph_view: None,
            content_height: std::cell::Cell::new(0),
            global_filter: FilterState::new(),
        }
    }

    /// Whether the global-filter prompt is currently capturing input.
    pub(super) fn global_filter_active(&self) -> bool {
        self.global_filter.active
    }

    /// Mutable access to the global filter, for the key handler.
    pub(super) fn global_filter_mut(&mut self) -> &mut FilterState {
        &mut self.global_filter
    }

    /// Read-only access to the global filter, for the status-bar indicator.
    pub(super) fn global_filter(&self) -> &FilterState {
        &self.global_filter
    }

    /// Push the compiled global filter into the current view's browser so its
    /// list re-narrows. Called on every tick (cheap; ≤200 rows) so a global
    /// filter set on one view also applies after switching to another.
    pub(super) fn push_global_filter(&mut self) {
        let re = self.global_filter.regex();
        match self.current_view {
            View::Queue => self.queue_browser().set_global_filter(re),
            View::Projects => self.project_browser().set_global_filter(re),
            View::Libraries => self.library_browser().set_global_filter(re),
            View::Rules => self.rule_browser().set_global_filter(re),
            View::Scratchpad => self.scratchpad_browser().set_global_filter(re),
            View::Dashboard | View::Service | View::Logs | View::Graph => {}
        }
    }

    /// Half- and full-screen navigation step sizes, derived from the last
    /// rendered content height (minus a small allowance for borders/header).
    pub(super) fn nav_steps(&self) -> (usize, usize) {
        let full = (self.content_height.get().saturating_sub(3) as usize).max(1);
        let half = (full / 2).max(1);
        (half, full)
    }

    /// Return a mutable reference to the dashboard, creating it if needed.
    fn dashboard(&mut self) -> &mut Dashboard {
        self.dashboard.get_or_insert_with(Dashboard::new)
    }

    /// Return a mutable reference to the queue browser, creating it if needed.
    fn queue_browser(&mut self) -> &mut QueueBrowser {
        self.queue_browser.get_or_insert_with(QueueBrowser::new)
    }

    /// Return a mutable reference to the project browser, creating it if needed.
    fn project_browser(&mut self) -> &mut ProjectBrowser {
        self.project_browser.get_or_insert_with(ProjectBrowser::new)
    }

    /// Return a mutable reference to the library browser, creating it if needed.
    fn library_browser(&mut self) -> &mut LibraryBrowser {
        self.library_browser.get_or_insert_with(LibraryBrowser::new)
    }

    /// Return a mutable reference to the rule browser, creating it if needed.
    fn rule_browser(&mut self) -> &mut RuleBrowser {
        self.rule_browser.get_or_insert_with(RuleBrowser::new)
    }

    /// Return a mutable reference to the scratchpad browser, creating it if needed.
    fn scratchpad_browser(&mut self) -> &mut ScratchpadBrowser {
        self.scratchpad_browser
            .get_or_insert_with(ScratchpadBrowser::new)
    }

    /// Return a mutable reference to the service view, creating it if needed.
    fn service_view(&mut self) -> &mut ServiceView {
        self.service_view.get_or_insert_with(ServiceView::new)
    }

    /// Return a mutable reference to the log viewer, creating it if needed.
    fn log_viewer(&mut self) -> &mut LogViewer {
        self.log_viewer.get_or_insert_with(LogViewer::new)
    }

    /// Return a mutable reference to the graph view, creating it if needed.
    fn graph_view(&mut self) -> &mut GraphView {
        self.graph_view.get_or_insert_with(GraphView::new)
    }

    /// Main run loop: setup terminal, handle events, render, cleanup.
    pub fn run(&mut self) -> anyhow::Result<()> {
        let mut term = terminal::init()?;
        let events = EventHandler::new(Duration::from_millis(250));

        while self.running {
            term.draw(|frame| self.draw(frame))?;

            match events.next() {
                Ok(Event::Key(key)) => self.handle_key(key),
                Ok(Event::Resize) => {} // ratatui handles resize on next draw
                Ok(Event::Tick) => self.on_tick(),
                Err(_) => self.running = false,
            }
        }

        terminal::restore()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

    #[test]
    fn app_starts_on_dashboard() {
        let app = App::new("http://127.0.0.1:50051".into());
        assert!(app.running);
        assert_eq!(app.current_view, View::Dashboard);
        assert!(!app.show_help);
        assert!(app.log_viewer.is_none());
        assert!(app.dashboard.is_none());
        assert!(app.queue_browser.is_none());
        assert!(app.project_browser.is_none());
        assert!(app.library_browser.is_none());
        assert!(app.rule_browser.is_none());
        assert!(app.scratchpad_browser.is_none());
        assert!(app.service_view.is_none());
        assert!(app.graph_view.is_none());
    }

    #[test]
    fn nav_steps_half_and_full() {
        let app = App::new("addr".into());
        // Default content height 0 → clamped to a minimum of 1.
        assert_eq!(app.nav_steps(), (1, 1));
        // height 23 → full = 23-3 = 20, half = 10.
        app.content_height.set(23);
        assert_eq!(app.nav_steps(), (10, 20));
    }

    #[test]
    fn view_next_wraps() {
        assert_eq!(View::Dashboard.next(), View::Queue);
        // Graph is now the last tab; it wraps back to Dashboard.
        assert_eq!(View::Graph.next(), View::Dashboard);
        assert_eq!(View::Libraries.next(), View::Rules);
        assert_eq!(View::Service.next(), View::Logs);
        assert_eq!(View::Logs.next(), View::Graph);
    }

    #[test]
    fn view_prev_wraps() {
        // Graph is the last tab; Dashboard.prev() wraps to Graph.
        assert_eq!(View::Dashboard.prev(), View::Graph);
        assert_eq!(View::Queue.prev(), View::Dashboard);
        assert_eq!(View::Rules.prev(), View::Libraries);
        assert_eq!(View::Graph.prev(), View::Logs);
    }

    #[test]
    fn handle_key_quit() {
        let mut app = App::new("addr".into());
        app.handle_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
        assert!(!app.running);
    }

    #[test]
    fn handle_key_ctrl_c_quit() {
        let mut app = App::new("addr".into());
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(!app.running);
    }

    #[test]
    fn handle_key_number_switches_view() {
        let mut app = App::new("addr".into());
        app.handle_key(KeyEvent::new(KeyCode::Char('3'), KeyModifiers::NONE));
        assert_eq!(app.current_view, View::Projects);
    }

    #[test]
    fn handle_key_number_switches_to_new_views() {
        let mut app = App::new("addr".into());
        app.handle_key(KeyEvent::new(KeyCode::Char('5'), KeyModifiers::NONE));
        assert_eq!(app.current_view, View::Rules);
        app.handle_key(KeyEvent::new(KeyCode::Char('6'), KeyModifiers::NONE));
        assert_eq!(app.current_view, View::Scratchpad);
        app.handle_key(KeyEvent::new(KeyCode::Char('7'), KeyModifiers::NONE));
        assert_eq!(app.current_view, View::Service);
        app.handle_key(KeyEvent::new(KeyCode::Char('8'), KeyModifiers::NONE));
        assert_eq!(app.current_view, View::Logs);
    }

    #[test]
    fn handle_key_tab_cycles() {
        let mut app = App::new("addr".into());
        assert_eq!(app.current_view, View::Dashboard);
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.current_view, View::Queue);
    }

    #[test]
    fn handle_key_help_toggle() {
        let mut app = App::new("addr".into());
        assert!(!app.show_help);
        app.handle_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE));
        assert!(app.show_help);
        // While help is shown, q closes help instead of quitting
        app.handle_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
        assert!(!app.show_help);
        assert!(app.running); // did not quit
    }

    #[test]
    fn view_label_and_index() {
        assert_eq!(View::Dashboard.label(), "Dashboard");
        assert_eq!(View::Queue.index(), 1);
        assert_eq!(View::from_index(4), View::Rules);
        assert_eq!(View::from_index(7), View::Logs);
        // Graph is tab 9 (index 8).
        assert_eq!(View::from_index(8), View::Graph);
        assert_eq!(View::Graph.label(), "Graph");
        assert_eq!(View::from_index(99), View::Dashboard);
        // 9-view count.
        assert_eq!(View::ALL.len(), 9);
    }

    #[test]
    fn log_viewer_lazy_init() {
        let mut app = App::new("addr".into());
        assert!(app.log_viewer.is_none());
        // Switching to Logs view and pressing j should initialize the viewer
        app.current_view = View::Logs;
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.log_viewer.is_some());
    }

    #[test]
    fn log_scroll_keys_do_not_quit() {
        let mut app = App::new("addr".into());
        app.current_view = View::Logs;
        // j/k should scroll, not trigger global bindings
        app.handle_key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE));
        assert!(app.running);
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.running);
    }

    #[test]
    fn on_tick_initializes_log_viewer_on_logs_view() {
        let mut app = App::new("addr".into());
        app.current_view = View::Logs;
        app.on_tick();
        assert!(app.log_viewer.is_some());
    }

    #[test]
    fn on_tick_initializes_dashboard_on_dashboard_view() {
        let mut app = App::new("addr".into());
        assert!(app.dashboard.is_none());
        app.current_view = View::Dashboard;
        app.on_tick();
        assert!(app.dashboard.is_some());
    }

    #[test]
    fn dashboard_lazy_init() {
        let mut app = App::new("addr".into());
        assert!(app.dashboard.is_none());
        let _ = app.dashboard();
        assert!(app.dashboard.is_some());
    }

    #[test]
    fn queue_browser_lazy_init() {
        let mut app = App::new("addr".into());
        assert!(app.queue_browser.is_none());
        let _ = app.queue_browser();
        assert!(app.queue_browser.is_some());
    }

    #[test]
    fn project_browser_lazy_init() {
        let mut app = App::new("addr".into());
        assert!(app.project_browser.is_none());
        let _ = app.project_browser();
        assert!(app.project_browser.is_some());
    }

    #[test]
    fn library_browser_lazy_init() {
        let mut app = App::new("addr".into());
        assert!(app.library_browser.is_none());
        let _ = app.library_browser();
        assert!(app.library_browser.is_some());
    }

    #[test]
    fn rule_browser_lazy_init() {
        let mut app = App::new("addr".into());
        assert!(app.rule_browser.is_none());
        let _ = app.rule_browser();
        assert!(app.rule_browser.is_some());
    }

    #[test]
    fn scratchpad_browser_lazy_init() {
        let mut app = App::new("addr".into());
        assert!(app.scratchpad_browser.is_none());
        let _ = app.scratchpad_browser();
        assert!(app.scratchpad_browser.is_some());
    }

    #[test]
    fn service_view_lazy_init() {
        let mut app = App::new("addr".into());
        assert!(app.service_view.is_none());
        let _ = app.service_view();
        assert!(app.service_view.is_some());
    }

    #[test]
    fn rule_keys_initialize_browser() {
        let mut app = App::new("addr".into());
        app.current_view = View::Rules;
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.rule_browser.is_some());
        assert!(app.running);
    }

    #[test]
    fn scratchpad_keys_initialize_browser() {
        let mut app = App::new("addr".into());
        app.current_view = View::Scratchpad;
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.scratchpad_browser.is_some());
        assert!(app.running);
    }

    #[test]
    fn on_tick_initializes_rules_on_rules_view() {
        let mut app = App::new("addr".into());
        app.current_view = View::Rules;
        app.on_tick();
        assert!(app.rule_browser.is_some());
    }

    #[test]
    fn on_tick_initializes_scratchpad_on_scratchpad_view() {
        let mut app = App::new("addr".into());
        app.current_view = View::Scratchpad;
        app.on_tick();
        assert!(app.scratchpad_browser.is_some());
    }

    #[test]
    fn on_tick_initializes_service_on_service_view() {
        let mut app = App::new("addr".into());
        app.current_view = View::Service;
        app.on_tick();
        assert!(app.service_view.is_some());
    }

    #[test]
    fn queue_keys_initialize_browser() {
        let mut app = App::new("addr".into());
        app.current_view = View::Queue;
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.queue_browser.is_some());
    }

    #[test]
    fn project_keys_initialize_browser() {
        let mut app = App::new("addr".into());
        app.current_view = View::Projects;
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.project_browser.is_some());
    }

    #[test]
    fn library_keys_initialize_browser() {
        let mut app = App::new("addr".into());
        app.current_view = View::Libraries;
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.library_browser.is_some());
    }

    #[test]
    fn project_keys_do_not_quit() {
        let mut app = App::new("addr".into());
        app.current_view = View::Projects;
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.running);
        app.handle_key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE));
        assert!(app.running);
    }

    #[test]
    fn library_keys_do_not_quit() {
        let mut app = App::new("addr".into());
        app.current_view = View::Libraries;
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.running);
        app.handle_key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE));
        assert!(app.running);
    }

    #[test]
    fn on_tick_initializes_projects_on_projects_view() {
        let mut app = App::new("addr".into());
        app.current_view = View::Projects;
        app.on_tick();
        assert!(app.project_browser.is_some());
    }

    #[test]
    fn on_tick_initializes_libraries_on_libraries_view() {
        let mut app = App::new("addr".into());
        app.current_view = View::Libraries;
        app.on_tick();
        assert!(app.library_browser.is_some());
    }

    #[test]
    fn switch_to_libraries_via_number_key() {
        let mut app = App::new("addr".into());
        app.handle_key(KeyEvent::new(KeyCode::Char('4'), KeyModifiers::NONE));
        assert_eq!(app.current_view, View::Libraries);
    }

    #[test]
    fn queue_keys_do_not_quit() {
        let mut app = App::new("addr".into());
        app.current_view = View::Queue;
        // j/k/f should not trigger quit
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.running);
        app.handle_key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE));
        assert!(app.running);
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE));
        assert!(app.running);
    }

    #[test]
    fn queue_status_cycles_via_s_key() {
        let mut app = App::new("addr".into());
        app.current_view = View::Queue;
        app.handle_key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE));
        // After one s press, the status filter should cycle from All to Pending.
        let browser = app.queue_browser.as_ref().unwrap();
        assert_eq!(
            browser.filter(),
            super::super::views::queue_data::StatusFilter::Pending
        );
    }

    #[test]
    fn queue_f_opens_page_filter_prompt() {
        let mut app = App::new("addr".into());
        app.current_view = View::Queue;
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE));
        assert!(app.queue_browser.as_ref().unwrap().page_filter_active());
    }

    #[test]
    fn shift_f_opens_global_filter_prompt() {
        let mut app = App::new("addr".into());
        app.current_view = View::Projects;
        app.handle_key(KeyEvent::new(KeyCode::Char('F'), KeyModifiers::SHIFT));
        assert!(app.global_filter_active());
    }

    #[test]
    fn on_tick_initializes_queue_on_queue_view() {
        let mut app = App::new("addr".into());
        app.current_view = View::Queue;
        app.on_tick();
        assert!(app.queue_browser.is_some());
    }

    #[test]
    fn graph_view_lazy_init() {
        let mut app = App::new("addr".into());
        assert!(app.graph_view.is_none());
        let _ = app.graph_view();
        assert!(app.graph_view.is_some());
    }

    #[test]
    fn on_tick_initializes_graph_on_graph_view() {
        let mut app = App::new("addr".into());
        app.current_view = View::Graph;
        app.on_tick();
        assert!(app.graph_view.is_some());
    }

    #[test]
    fn handle_key_9_switches_to_graph() {
        let mut app = App::new("addr".into());
        app.handle_key(KeyEvent::new(KeyCode::Char('9'), KeyModifiers::NONE));
        assert_eq!(app.current_view, View::Graph);
    }

    #[test]
    fn graph_keys_initialize_view() {
        let mut app = App::new("addr".into());
        app.current_view = View::Graph;
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.graph_view.is_some());
        assert!(app.running);
    }
}

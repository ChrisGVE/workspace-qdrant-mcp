//! TUI application core — event loop and state management.

mod key_handler;
mod key_handlers_lists;
mod key_handlers_misc;
mod key_handlers_simple;
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
#[path = "app_tests.rs"]
mod tests;

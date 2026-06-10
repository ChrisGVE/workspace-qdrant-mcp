//! Unit tests for the TUI application core.
//!
//! Extracted into a separate file to keep `app.rs` under the 500-line limit.

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
    assert!(app.search_view.is_none());
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
    // Search is now the last tab; it wraps back to Dashboard.
    assert_eq!(View::Search.next(), View::Dashboard);
    assert_eq!(View::Libraries.next(), View::Rules);
    assert_eq!(View::Service.next(), View::Logs);
    assert_eq!(View::Logs.next(), View::Graph);
    assert_eq!(View::Graph.next(), View::Search);
}

#[test]
fn view_prev_wraps() {
    // Search is the last tab; Dashboard.prev() wraps to Search.
    assert_eq!(View::Dashboard.prev(), View::Search);
    assert_eq!(View::Queue.prev(), View::Dashboard);
    assert_eq!(View::Rules.prev(), View::Libraries);
    assert_eq!(View::Graph.prev(), View::Logs);
    assert_eq!(View::Search.prev(), View::Graph);
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
    // Search is tab 10 (index 9, key '0').
    assert_eq!(View::from_index(9), View::Search);
    assert_eq!(View::Search.label(), "Search");
    assert_eq!(View::from_index(99), View::Dashboard);
    // 10-view count.
    assert_eq!(View::ALL.len(), 10);
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
        crate::tui::views::queue_data::StatusFilter::Pending
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

#[test]
fn search_view_lazy_init() {
    let mut app = App::new("addr".into());
    assert!(app.search_view.is_none());
    let _ = app.search_view();
    assert!(app.search_view.is_some());
}

#[test]
fn on_tick_initializes_search_on_search_view() {
    let mut app = App::new("addr".into());
    app.current_view = View::Search;
    app.on_tick();
    assert!(app.search_view.is_some());
}

#[test]
fn handle_key_0_switches_to_search() {
    let mut app = App::new("addr".into());
    app.handle_key(KeyEvent::new(KeyCode::Char('0'), KeyModifiers::NONE));
    assert_eq!(app.current_view, View::Search);
}

#[test]
fn search_keys_initialize_view() {
    let mut app = App::new("addr".into());
    app.current_view = View::Search;
    app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
    assert!(app.search_view.is_some());
    assert!(app.running);
}

#[test]
fn search_mode_keys_do_not_switch_global_view() {
    // Keys 1-3 on Search page switch search sub-mode, not the global view.
    let mut app = App::new("addr".into());
    app.current_view = View::Search;
    app.handle_key(KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE));
    assert_eq!(app.current_view, View::Search);
    app.handle_key(KeyEvent::new(KeyCode::Char('2'), KeyModifiers::NONE));
    assert_eq!(app.current_view, View::Search);
    app.handle_key(KeyEvent::new(KeyCode::Char('3'), KeyModifiers::NONE));
    assert_eq!(app.current_view, View::Search);
}

#[test]
fn search_slash_opens_prompt() {
    let mut app = App::new("addr".into());
    app.current_view = View::Search;
    app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE));
    assert!(app.search_view.as_ref().unwrap().prompt.active);
}

#[test]
fn search_i_opens_prompt() {
    let mut app = App::new("addr".into());
    app.current_view = View::Search;
    app.handle_key(KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE));
    assert!(app.search_view.as_ref().unwrap().prompt.active);
}

#[test]
fn search_esc_cancels_prompt() {
    let mut app = App::new("addr".into());
    app.current_view = View::Search;
    // Open prompt
    app.handle_key(KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE));
    assert!(app.search_view.as_ref().unwrap().prompt.active);
    // Esc cancels it
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    assert!(!app.search_view.as_ref().unwrap().prompt.active);
    assert!(app.running);
}

#[test]
fn search_tenant_cycling_keys() {
    let mut app = App::new("addr".into());
    app.current_view = View::Search;
    // Both [ and ] should be consumed (not fall through to global)
    app.handle_key(KeyEvent::new(KeyCode::Char('['), KeyModifiers::NONE));
    assert!(app.search_view.is_some());
    assert!(app.running);
    app.handle_key(KeyEvent::new(KeyCode::Char(']'), KeyModifiers::NONE));
    assert!(app.running);
}

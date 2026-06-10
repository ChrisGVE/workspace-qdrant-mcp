//! Per-view key handlers for Dashboard, Service, Logs, and Graph views.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::super::search::SearchAction;
use super::super::views::dashboard::FocusedCell;
use super::super::views::graph::GraphMode;
use super::App;

impl App {
    /// Handle dashboard-specific keys. Returns true if the key was consumed.
    pub(super) fn handle_dashboard_key(&mut self, key: KeyEvent) -> bool {
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

    /// Handle service-specific keys. Returns true if the key was consumed.
    pub(super) fn handle_service_key(&mut self, key: KeyEvent) -> bool {
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
    pub(super) fn handle_log_key(&mut self, key: KeyEvent) -> bool {
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

    /// Handle graph-specific keys. Returns true if the key was consumed.
    ///
    /// Keys 1–5 switch the graph mode (Stats/PageRank/Communities/Betweenness/Impact).
    /// `[`/`]` cycle the active tenant. `i` opens the Impact symbol prompt.
    /// Standard j/k/g/G/^d/^u/^f/^b navigation applies in list modes.
    /// Enter toggles community member expansion in Communities mode.
    pub(super) fn handle_graph_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let (half, full) = self.nav_steps();
        let view = self.graph_view();

        // Impact prompt captures all input while active.
        if view.impact_prompt.active {
            match key.code {
                KeyCode::Esc => {
                    view.impact_prompt.cancel();
                    return true;
                }
                KeyCode::Enter => {
                    if let Some(sym) = view.impact_prompt.confirm() {
                        view.trigger_impact(sym);
                    }
                    return true;
                }
                KeyCode::Backspace => {
                    view.impact_prompt.backspace();
                    return true;
                }
                KeyCode::Char(c) => {
                    view.impact_prompt.push_char(c);
                    return true;
                }
                _ => return true,
            }
        }

        match key.code {
            // Mode switching: keys 1-5 select graph sub-modes.
            KeyCode::Char(c @ '1'..='5') => {
                let digit = c as u8 - b'0';
                if let Some(mode) = GraphMode::from_key(digit) {
                    view.set_mode(mode);
                    return true;
                }
                false
            }

            // Tenant cycling
            KeyCode::Char('[') => {
                view.prev_tenant();
                true
            }
            KeyCode::Char(']') => {
                view.next_tenant();
                true
            }

            // Impact query prompt
            KeyCode::Char('i') => {
                view.set_mode(GraphMode::Impact);
                view.impact_prompt.activate();
                true
            }

            // List navigation
            KeyCode::Char('j') | KeyCode::Down => {
                view.select_next();
                true
            }
            KeyCode::Char('k') | KeyCode::Up => {
                view.select_prev();
                true
            }
            KeyCode::Char('d') if ctrl => {
                view.page_down(half);
                true
            }
            KeyCode::Char('u') if ctrl => {
                view.page_up(half);
                true
            }
            KeyCode::Char('f') if ctrl => {
                view.page_down(full);
                true
            }
            KeyCode::Char('b') if ctrl => {
                view.page_up(full);
                true
            }
            KeyCode::PageUp => {
                view.page_up(full);
                true
            }
            KeyCode::PageDown => {
                view.page_down(full);
                true
            }
            KeyCode::Char('g') => {
                view.jump_first();
                true
            }
            KeyCode::Char('G') => {
                view.jump_last();
                true
            }

            // Enter: expand/collapse community members in Communities mode.
            KeyCode::Enter => {
                view.toggle_community_expand();
                true
            }

            // Esc: close community popup if open, otherwise fall through.
            KeyCode::Esc => {
                if view.expanded_community.is_some() {
                    view.expanded_community = None;
                    return true;
                }
                false
            }

            _ => false,
        }
    }
}

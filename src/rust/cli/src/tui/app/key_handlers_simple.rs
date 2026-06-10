//! Per-view key handlers for Rules and Scratchpad list views.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::super::filter::FilterAction;
use super::super::search::SearchAction;
use super::App;

impl App {
    /// Handle rule-specific keys. Returns true if the key was consumed.
    pub(super) fn handle_rule_key(&mut self, key: KeyEvent) -> bool {
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
    pub(super) fn handle_scratchpad_key(&mut self, key: KeyEvent) -> bool {
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
}

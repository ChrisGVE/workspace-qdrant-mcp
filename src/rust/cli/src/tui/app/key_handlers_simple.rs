//! Per-view key handlers for Rules, Scratchpad, and Search page views.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::super::filter::FilterAction;
use super::super::search::SearchAction;
use super::super::views::search_page::SearchMode;
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

        // Typed-name delete confirm: each key goes to the TypedConfirm buffer.
        // Enter attempts confirmation; Esc cancels.
        if browser.delete_confirm_open() {
            match key.code {
                KeyCode::Char(c) => {
                    if let Some(tc) = browser.delete_confirm_mut() {
                        tc.push_char(c);
                    }
                }
                KeyCode::Backspace => {
                    if let Some(tc) = browser.delete_confirm_mut() {
                        tc.pop_char();
                    }
                }
                KeyCode::Enter => {
                    if let Some(rule_id) = browser.take_delete_if_confirmed() {
                        let msg = super::super::commands::rule_delete(&rule_id);
                        browser.set_message(msg);
                        browser.force_refresh();
                    }
                }
                KeyCode::Esc => browser.cancel_delete(),
                _ => {}
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
            KeyCode::Char('d') => {
                browser.request_delete();
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

    /// Handle Search page keys. Returns true if the key was consumed.
    ///
    /// Keys 1–4 switch the search mode (Grep/Exact/Graph/Semantic), consuming
    /// them so they never fall through to the global view-switch handler. `i`
    /// or `/` opens the query prompt. `[`/`]` cycle tenants.
    /// Standard j/k/g/G/^d/^u/^f/^b navigation applies in the results list.
    /// Enter opens the result preview; Esc closes it or the prompt.
    pub(super) fn handle_search_key(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let (half, full) = self.nav_steps();
        let view = self.search_view();

        // Query prompt captures all input while active.
        if view.prompt.active {
            match key.code {
                KeyCode::Esc => {
                    view.prompt.cancel();
                    return true;
                }
                KeyCode::Enter => {
                    if let Some(query) = view.prompt.confirm() {
                        view.dispatch_query(query);
                    }
                    return true;
                }
                KeyCode::Backspace => {
                    view.prompt.backspace();
                    return true;
                }
                KeyCode::Char(c) => {
                    view.prompt.push_char(c);
                    return true;
                }
                _ => return true,
            }
        }

        // Preview popup: Esc closes it; all other keys scroll or close.
        if view.preview_open {
            match key.code {
                KeyCode::Esc | KeyCode::Enter => {
                    view.close_preview();
                    return true;
                }
                KeyCode::Char('j') | KeyCode::Down => {
                    view.select_next();
                    return true;
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    view.select_prev();
                    return true;
                }
                _ => return true,
            }
        }

        match key.code {
            // Mode switching: keys 1-4 select search sub-modes.
            KeyCode::Char(c @ '1'..='4') => {
                let digit = c as u8 - b'0';
                if let Some(mode) = SearchMode::from_key(digit) {
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

            // Open query prompt
            KeyCode::Char('i') | KeyCode::Char('/') => {
                view.prompt.activate();
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

            // Enter: open result preview.
            KeyCode::Enter => {
                view.open_preview();
                true
            }

            // Esc: close preview if open (already handled above), else fall through.
            KeyCode::Esc => false,

            _ => false,
        }
    }
}

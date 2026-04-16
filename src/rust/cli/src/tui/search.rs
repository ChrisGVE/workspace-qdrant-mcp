//! Reusable search/filter state for TUI list views.
//!
//! Provides a text input mode activated by '/' that filters items
//! by case-insensitive substring match. Esc cancels, Enter confirms.

use crossterm::event::KeyCode;

/// Search mode state for list views.
#[derive(Debug, Clone)]
pub struct SearchState {
    /// Whether search input mode is active.
    pub active: bool,
    /// Current search query text.
    pub query: String,
    /// Last confirmed query (persists after Enter).
    pub confirmed_query: String,
}

impl SearchState {
    pub fn new() -> Self {
        Self {
            active: false,
            query: String::new(),
            confirmed_query: String::new(),
        }
    }

    /// Activate search mode.
    pub fn activate(&mut self) {
        self.active = true;
        self.query = self.confirmed_query.clone();
    }

    /// Handle a key event while search is active.
    /// Returns true if the key was consumed.
    pub fn handle_key(&mut self, code: KeyCode) -> bool {
        if !self.active {
            return false;
        }

        match code {
            KeyCode::Esc => {
                self.active = false;
                self.query.clear();
                self.confirmed_query.clear();
                true
            }
            KeyCode::Enter => {
                self.active = false;
                self.confirmed_query = self.query.clone();
                true
            }
            KeyCode::Backspace => {
                self.query.pop();
                true
            }
            KeyCode::Char(c) => {
                self.query.push(c);
                true
            }
            _ => true, // consume all keys while active
        }
    }

    /// Returns the active filter query (live while typing, confirmed after Enter).
    pub fn filter_query(&self) -> &str {
        if self.active {
            &self.query
        } else {
            &self.confirmed_query
        }
    }

    /// Whether there's an active filter (either typing or confirmed).
    pub fn has_filter(&self) -> bool {
        !self.filter_query().is_empty()
    }

    /// Check if a text matches the current filter (case-insensitive substring).
    pub fn matches(&self, text: &str) -> bool {
        let q = self.filter_query();
        if q.is_empty() {
            return true;
        }
        text.to_lowercase().contains(&q.to_lowercase())
    }
}

impl Default for SearchState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_has_no_filter() {
        let s = SearchState::new();
        assert!(!s.active);
        assert!(!s.has_filter());
        assert!(s.matches("anything"));
    }

    #[test]
    fn typing_filters_live() {
        let mut s = SearchState::new();
        s.activate();
        assert!(s.active);
        s.handle_key(KeyCode::Char('f'));
        s.handle_key(KeyCode::Char('o'));
        s.handle_key(KeyCode::Char('o'));
        assert_eq!(s.filter_query(), "foo");
        assert!(s.matches("foobar"));
        assert!(!s.matches("baz"));
    }

    #[test]
    fn enter_confirms_query() {
        let mut s = SearchState::new();
        s.activate();
        s.handle_key(KeyCode::Char('t'));
        s.handle_key(KeyCode::Char('e'));
        s.handle_key(KeyCode::Enter);
        assert!(!s.active);
        assert_eq!(s.confirmed_query, "te");
        assert!(s.has_filter());
        assert!(s.matches("test"));
    }

    #[test]
    fn esc_clears_everything() {
        let mut s = SearchState::new();
        s.activate();
        s.handle_key(KeyCode::Char('x'));
        s.handle_key(KeyCode::Esc);
        assert!(!s.active);
        assert!(!s.has_filter());
    }

    #[test]
    fn backspace_removes_char() {
        let mut s = SearchState::new();
        s.activate();
        s.handle_key(KeyCode::Char('a'));
        s.handle_key(KeyCode::Char('b'));
        s.handle_key(KeyCode::Backspace);
        assert_eq!(s.query, "a");
    }

    #[test]
    fn case_insensitive_match() {
        let mut s = SearchState::new();
        s.activate();
        s.handle_key(KeyCode::Char('F'));
        s.handle_key(KeyCode::Char('O'));
        assert!(s.matches("foobar"));
        assert!(s.matches("FOOBAR"));
        assert!(s.matches("Foo"));
    }

    #[test]
    fn reactivate_preserves_confirmed() {
        let mut s = SearchState::new();
        s.activate();
        s.handle_key(KeyCode::Char('x'));
        s.handle_key(KeyCode::Enter);
        assert_eq!(s.confirmed_query, "x");

        s.activate();
        assert_eq!(s.query, "x"); // reloads confirmed
    }
}

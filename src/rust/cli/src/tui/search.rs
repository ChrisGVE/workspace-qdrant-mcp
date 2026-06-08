//! Incremental regex search for TUI list views.
//!
//! Activated by `/`: the typed text is a regex pattern (compiled
//! case-insensitively). Search does not narrow the list — it moves the cursor
//! to matching rows. After confirming with Enter, `n`/`N` jump to the
//! next/previous match (wrapping). Invalid patterns are handled gracefully:
//! the pattern is marked invalid and simply matches nothing.

use crossterm::event::KeyCode;
use ratatui::style::Style;
use ratatui::text::Span;
use regex::Regex;

use crate::tui::theme;

/// Outcome of feeding a key to an active search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchAction {
    /// Key consumed; still editing.
    Editing,
    /// Enter pressed — the pattern is confirmed (jump to the first match).
    Confirmed,
    /// Esc pressed — search cancelled and cleared.
    Cancelled,
}

/// Regex search state for a list view.
#[derive(Debug, Clone, Default)]
pub struct SearchState {
    /// Whether the search input prompt is active.
    pub active: bool,
    /// Live input text while typing.
    pub query: String,
    /// Last confirmed pattern.
    confirmed: String,
    /// Compiled pattern (from `confirmed`); `None` when empty or invalid.
    regex: Option<Regex>,
    /// True when the last confirmed pattern failed to compile.
    pub invalid: bool,
}

impl SearchState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin editing, seeding the input with the last confirmed pattern.
    pub fn activate(&mut self) {
        self.active = true;
        self.query = self.confirmed.clone();
    }

    /// Feed a key to the active prompt. Returns the resulting action.
    pub fn handle_key(&mut self, code: KeyCode) -> SearchAction {
        if !self.active {
            return SearchAction::Editing;
        }
        match code {
            KeyCode::Esc => {
                self.active = false;
                self.query.clear();
                self.confirmed.clear();
                self.regex = None;
                self.invalid = false;
                SearchAction::Cancelled
            }
            KeyCode::Enter => {
                self.active = false;
                self.confirmed = self.query.clone();
                self.compile();
                SearchAction::Confirmed
            }
            KeyCode::Backspace => {
                self.query.pop();
                SearchAction::Editing
            }
            KeyCode::Char(c) => {
                self.query.push(c);
                SearchAction::Editing
            }
            _ => SearchAction::Editing,
        }
    }

    /// Compile the confirmed pattern case-insensitively.
    fn compile(&mut self) {
        if self.confirmed.is_empty() {
            self.regex = None;
            self.invalid = false;
            return;
        }
        match Regex::new(&format!("(?i){}", self.confirmed)) {
            Ok(re) => {
                self.regex = Some(re);
                self.invalid = false;
            }
            Err(_) => {
                self.regex = None;
                self.invalid = true;
            }
        }
    }

    /// Whether there is a confirmed, non-empty search pattern.
    pub fn has_query(&self) -> bool {
        !self.confirmed.is_empty()
    }

    /// The confirmed pattern text.
    pub fn confirmed(&self) -> &str {
        &self.confirmed
    }

    /// Whether `text` matches the confirmed pattern. Always false when there is
    /// no pattern or the pattern is invalid.
    pub fn is_match(&self, text: &str) -> bool {
        self.regex.as_ref().is_some_and(|r| r.is_match(text))
    }
}

/// Build the status spans for a search prompt: the live input while typing,
/// or a match-count / invalid-pattern indicator once confirmed.
pub fn prompt_spans(state: &SearchState, match_count: usize) -> Vec<Span<'static>> {
    if state.active {
        return vec![
            Span::styled("  /", theme::search_style()),
            Span::styled(state.query.clone(), theme::search_style()),
            Span::styled("\u{2588}", theme::search_style()),
        ];
    }
    if !state.has_query() {
        return Vec::new();
    }
    if state.invalid {
        return vec![Span::styled(
            "  [invalid regex]",
            Style::default().fg(theme::COLOR_ERROR),
        )];
    }
    let fg = if match_count == 0 {
        theme::COLOR_WARNING
    } else {
        theme::COLOR_ACCENT
    };
    vec![Span::styled(
        format!(
            "  /{}/ {} match{}",
            state.confirmed(),
            match_count,
            if match_count == 1 { "" } else { "es" }
        ),
        Style::default().fg(fg),
    )]
}

/// Index of the first match strictly after `current`, wrapping to the first.
pub fn next_index(matches: &[usize], current: usize) -> Option<usize> {
    if matches.is_empty() {
        return None;
    }
    matches
        .iter()
        .find(|&&i| i > current)
        .or_else(|| matches.first())
        .copied()
}

/// Index of the last match strictly before `current`, wrapping to the last.
pub fn prev_index(matches: &[usize], current: usize) -> Option<usize> {
    if matches.is_empty() {
        return None;
    }
    matches
        .iter()
        .rev()
        .find(|&&i| i < current)
        .or_else(|| matches.last())
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_has_no_query() {
        let s = SearchState::new();
        assert!(!s.active);
        assert!(!s.has_query());
        assert!(!s.is_match("anything"));
    }

    #[test]
    fn typing_then_confirm_compiles_regex() {
        let mut s = SearchState::new();
        s.activate();
        for c in "fo+".chars() {
            assert_eq!(s.handle_key(KeyCode::Char(c)), SearchAction::Editing);
        }
        assert_eq!(s.handle_key(KeyCode::Enter), SearchAction::Confirmed);
        assert!(!s.active);
        assert!(s.has_query());
        assert!(s.is_match("foobar"));
        assert!(s.is_match("FOO")); // case-insensitive
        assert!(!s.is_match("bar"));
    }

    #[test]
    fn invalid_regex_marked_and_matches_nothing() {
        let mut s = SearchState::new();
        s.activate();
        for c in "fo(".chars() {
            s.handle_key(KeyCode::Char(c));
        }
        s.handle_key(KeyCode::Enter);
        assert!(s.invalid);
        assert!(!s.is_match("foo"));
    }

    #[test]
    fn esc_clears_everything() {
        let mut s = SearchState::new();
        s.activate();
        s.handle_key(KeyCode::Char('x'));
        assert_eq!(s.handle_key(KeyCode::Esc), SearchAction::Cancelled);
        assert!(!s.active);
        assert!(!s.has_query());
    }

    #[test]
    fn backspace_edits() {
        let mut s = SearchState::new();
        s.activate();
        s.handle_key(KeyCode::Char('a'));
        s.handle_key(KeyCode::Char('b'));
        s.handle_key(KeyCode::Backspace);
        assert_eq!(s.query, "a");
    }

    #[test]
    fn reactivate_preserves_confirmed() {
        let mut s = SearchState::new();
        s.activate();
        s.handle_key(KeyCode::Char('x'));
        s.handle_key(KeyCode::Enter);
        s.activate();
        assert_eq!(s.query, "x");
    }

    #[test]
    fn next_and_prev_wrap() {
        let m = vec![2usize, 5, 9];
        assert_eq!(next_index(&m, 0), Some(2));
        assert_eq!(next_index(&m, 2), Some(5));
        assert_eq!(next_index(&m, 9), Some(2)); // wrap
        assert_eq!(prev_index(&m, 9), Some(5));
        assert_eq!(prev_index(&m, 2), Some(9)); // wrap
        assert_eq!(next_index(&[], 0), None);
    }
}

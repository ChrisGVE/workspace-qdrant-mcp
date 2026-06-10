//! Narrowing filters for TUI list views.
//!
//! Two filter scopes share this one type:
//!
//! - **Page filter** (`f`): sticky per view, lives on each browser.
//! - **Global filter** (`F`): one filter applied across every list view, lives
//!   on the [`App`](crate::tui::app).
//!
//! Unlike `/` search — which only moves the cursor to matching rows without
//! hiding anything — a filter *narrows* the list: rows that do not match are
//! removed from view. The typed text is a regex compiled case-insensitively.
//! Global and page filters compose with AND (a row must match both), and on the
//! Queue view they further compose with the SQL-side status filter.
//!
//! An empty or invalid pattern matches everything, so a typo never blanks the
//! list out from under the user; the prompt indicator flags the invalid regex
//! instead.

use crossterm::event::KeyCode;
use ratatui::style::Style;
use ratatui::text::Span;
use regex::Regex;

use crate::tui::theme;

/// Outcome of feeding a key to an active filter prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterAction {
    /// Key consumed; still editing.
    Editing,
    /// Enter pressed — the pattern is confirmed and the list re-narrowed.
    Applied,
    /// Esc pressed — editing cancelled, the previously confirmed filter kept.
    Cancelled,
}

/// Regex narrowing-filter state for a list view.
#[derive(Debug, Clone, Default)]
pub struct FilterState {
    /// Whether the filter input prompt is active.
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

impl FilterState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin editing, seeding the input with the last confirmed pattern.
    pub fn activate(&mut self) {
        self.active = true;
        self.query = self.confirmed.clone();
    }

    /// Feed a key to the active prompt. Returns the resulting action. Esc keeps
    /// the previously confirmed filter (only editing is cancelled); use
    /// [`clear`](Self::clear) to remove the filter entirely.
    pub fn handle_key(&mut self, code: KeyCode) -> FilterAction {
        if !self.active {
            return FilterAction::Editing;
        }
        match code {
            KeyCode::Esc => {
                self.active = false;
                self.query.clear();
                FilterAction::Cancelled
            }
            KeyCode::Enter => {
                self.active = false;
                self.confirmed = self.query.clone();
                self.compile();
                FilterAction::Applied
            }
            KeyCode::Backspace => {
                self.query.pop();
                FilterAction::Editing
            }
            KeyCode::Char(c) => {
                self.query.push(c);
                FilterAction::Editing
            }
            _ => FilterAction::Editing,
        }
    }

    /// Clear the filter entirely (pattern and compiled regex). Used by the
    /// Ctrl+C binding on the filter prompt.
    pub fn clear(&mut self) {
        self.active = false;
        self.query.clear();
        self.confirmed.clear();
        self.regex = None;
        self.invalid = false;
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

    /// Whether there is a confirmed, non-empty filter pattern.
    pub fn has_filter(&self) -> bool {
        !self.confirmed.is_empty()
    }

    /// The confirmed pattern text.
    pub fn confirmed(&self) -> &str {
        &self.confirmed
    }

    /// A clone of the compiled regex, for pushing a global filter down into the
    /// browsers. `None` when empty or invalid.
    pub fn regex(&self) -> Option<Regex> {
        self.regex.clone()
    }

    /// Whether `text` passes the filter. An empty or invalid pattern passes
    /// everything (a typo never blanks the list); otherwise the regex must
    /// match.
    pub fn matches(&self, text: &str) -> bool {
        match &self.regex {
            Some(re) => re.is_match(text),
            None => true,
        }
    }
}

/// Whether `text` passes an optional pre-compiled filter regex (the global
/// filter pushed into a browser). `None` passes everything.
pub fn regex_matches(re: &Option<Regex>, text: &str) -> bool {
    match re {
        Some(r) => r.is_match(text),
        None => true,
    }
}

/// Build the status spans for a filter prompt. `label` names the scope (e.g.
/// "Filter" or "Global"). While editing it shows the live input; once confirmed
/// it shows the active pattern (or an invalid-pattern marker).
pub fn prompt_spans(state: &FilterState, label: &str) -> Vec<Span<'static>> {
    if state.active {
        return vec![
            Span::styled(format!("  {label}: "), theme::search_style()),
            Span::styled(state.query.clone(), theme::search_style()),
            Span::styled("\u{2588}", theme::search_style()),
        ];
    }
    if !state.has_filter() {
        return Vec::new();
    }
    if state.invalid {
        return vec![Span::styled(
            format!("  {label}: [invalid regex]"),
            Style::default().fg(theme::COLOR_ERROR),
        )];
    }
    vec![Span::styled(
        format!("  {label}: {}", state.confirmed()),
        Style::default().fg(theme::COLOR_ACCENT),
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_passes_everything() {
        let f = FilterState::new();
        assert!(!f.active);
        assert!(!f.has_filter());
        assert!(f.matches("anything"));
        assert!(f.regex().is_none());
    }

    #[test]
    fn typing_then_apply_narrows() {
        let mut f = FilterState::new();
        f.activate();
        for c in "ru".chars() {
            assert_eq!(f.handle_key(KeyCode::Char(c)), FilterAction::Editing);
        }
        assert_eq!(f.handle_key(KeyCode::Enter), FilterAction::Applied);
        assert!(!f.active);
        assert!(f.has_filter());
        assert!(f.matches("main.rs is RUst")); // case-insensitive
        assert!(!f.matches("python.py"));
        assert!(f.regex().is_some());
    }

    #[test]
    fn invalid_regex_passes_everything() {
        let mut f = FilterState::new();
        f.activate();
        for c in "fo(".chars() {
            f.handle_key(KeyCode::Char(c));
        }
        f.handle_key(KeyCode::Enter);
        assert!(f.invalid);
        // Invalid pattern must not blank the list out.
        assert!(f.matches("foo"));
        assert!(f.matches("bar"));
        assert!(f.regex().is_none());
    }

    #[test]
    fn esc_keeps_confirmed_filter() {
        let mut f = FilterState::new();
        f.activate();
        for c in "abc".chars() {
            f.handle_key(KeyCode::Char(c));
        }
        f.handle_key(KeyCode::Enter);
        assert!(f.matches("xabcx"));
        // Re-open, type junk, cancel — the confirmed filter survives.
        f.activate();
        f.handle_key(KeyCode::Char('z'));
        assert_eq!(f.handle_key(KeyCode::Esc), FilterAction::Cancelled);
        assert!(!f.active);
        assert!(f.has_filter());
        assert!(f.matches("xabcx"));
        assert!(!f.matches("nope"));
    }

    #[test]
    fn clear_removes_filter() {
        let mut f = FilterState::new();
        f.activate();
        f.handle_key(KeyCode::Char('a'));
        f.handle_key(KeyCode::Enter);
        assert!(f.has_filter());
        f.clear();
        assert!(!f.has_filter());
        assert!(!f.active);
        assert!(f.matches("anything"));
    }

    #[test]
    fn backspace_edits() {
        let mut f = FilterState::new();
        f.activate();
        f.handle_key(KeyCode::Char('a'));
        f.handle_key(KeyCode::Char('b'));
        f.handle_key(KeyCode::Backspace);
        assert_eq!(f.query, "a");
    }

    #[test]
    fn reactivate_seeds_confirmed() {
        let mut f = FilterState::new();
        f.activate();
        f.handle_key(KeyCode::Char('x'));
        f.handle_key(KeyCode::Enter);
        f.activate();
        assert_eq!(f.query, "x");
    }

    #[test]
    fn handle_key_inactive_is_noop() {
        let mut f = FilterState::new();
        assert_eq!(f.handle_key(KeyCode::Char('a')), FilterAction::Editing);
        assert!(f.query.is_empty());
    }

    #[test]
    fn regex_matches_helper() {
        let re = Regex::new("(?i)foo").ok();
        assert!(regex_matches(&re, "FOObar"));
        assert!(!regex_matches(&re, "bar"));
        assert!(regex_matches(&None, "anything"));
    }

    #[test]
    fn prompt_spans_editing_and_confirmed() {
        let mut f = FilterState::new();
        assert!(prompt_spans(&f, "Filter").is_empty());
        f.activate();
        f.handle_key(KeyCode::Char('a'));
        assert!(!prompt_spans(&f, "Filter").is_empty()); // editing shows prompt
        f.handle_key(KeyCode::Enter);
        let spans = prompt_spans(&f, "Filter");
        assert!(!spans.is_empty()); // confirmed shows indicator
    }
}

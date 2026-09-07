//! The dialog slot — the one row between the status block and the list, and everything it holds.
//!
//! v0.1 has nowhere to put a search: it appends `/yml/ 0 matches` to the end of its status line,
//! beside `[s] status  [f] filter`, and by the time a filter is on as well that line reads
//! `Status: Done  (0 items)  [s] status  [f] filter  Filter: svg  /yml/ 0 matches` — five facts
//! and two conversations in one row, told apart by nothing. So this screen spends a row.
//!
//! One row, always present, blank when there is nothing to say. Reserved rather than inserted:
//! a row that appeared when `/` was pressed would push the whole list down one line, and the row
//! a reader was looking at would move under them at the moment they started looking for
//! something.
//!
//! # Two halves, and they belong to different kinds of thing
//!
//! **Left: the conversation.** Typing a term, or the term that was accepted and what it found.
//! It comes and goes.
//!
//! **Right: the settings.** `type P · status failed`. Chris did not say where the selectors are
//! shown; a knob whose position is only in the reader's memory is a knob they will forget they
//! turned, and *"why is the list empty"* is the question that follows. So they are drawn, at the
//! right end of the row, on every one of its states — and omitted entirely when they are `All`,
//! because a selector at its default is not a setting anybody made.
//!
//! *Supervisor's ruling on where they go, not yet Chris's.*

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use super::state::{Dialog, Kind, Status};
use crate::tokens;
use crate::widgets::edit_field::{caret_spans, Edit};

/// Columns between one fact on this row and the next — the same three
/// [`crate::tokens::key_hints`] puts between two hints, and the same
/// [`crate::panes::status_block`] puts between two groups.
const GAP: &str = "   ";

/// The prompts, spelled once each. Chris's own words, and the shared `term/regex` half is why
/// they sit together: the two dialogs differ in one verb, and a reader should see that they do.
const SEARCH_PROMPT: &str = "search term/regex: ";
const FILTER_PROMPT: &str = "filter term/regex: ";
const SEARCH_ON: &str = "search on: ";
const FILTER_ON: &str = "filter on: ";
/// What both settled dialogs end with. A dialog that did not say how it ends is one the reader
/// has to guess their way out of — §6's rule for a modal, and a dialog owes the same.
const CANCEL: &str = "Esc to cancel";

/// The row under the status block: a dialog on the left, the selectors on the right.
pub struct DialogSlot {
    dialog: Dialog,
    kind: Option<Kind>,
    status: Option<Status>,
}

impl DialogSlot {
    pub fn new(dialog: Dialog, kind: Option<Kind>, status: Option<Status>) -> Self {
        Self {
            dialog,
            kind,
            status,
        }
    }

    /// The settings half: `type P · status failed`, label muted and value normal, and nothing at
    /// all when both are `All`.
    ///
    /// The label is muted because it never changes and the value is what a reader is checking;
    /// the same split the status block draws between a glyph's word and its count.
    fn selectors(&self) -> Vec<Span<'static>> {
        let mut shown: Vec<(&str, String)> = Vec::new();
        if let Some(kind) = self.kind {
            shown.push(("type", kind.letter().to_string()));
        }
        if let Some(status) = self.status {
            shown.push(("status", status.label().to_string()));
        }
        let mut spans = Vec::new();
        for (at, (label, value)) in shown.into_iter().enumerate() {
            if at > 0 {
                spans.push(Span::styled(" · ", tokens::muted_style()));
            }
            spans.push(Span::styled(format!("{label} "), tokens::muted_style()));
            spans.push(Span::styled(value, tokens::normal_style()));
        }
        spans
    }

    /// A settled dialog: what it is, the term, what it found, and how to leave.
    fn settled(prompt: &str, term: &str, found: String) -> Vec<Span<'static>> {
        vec![
            Span::styled(prompt.to_string(), tokens::muted_style()),
            Span::styled(term.to_string(), tokens::normal_style()),
            Span::styled(GAP, tokens::muted_style()),
            Span::styled(found, tokens::normal_style()),
            Span::styled(GAP, tokens::muted_style()),
            Span::styled(CANCEL.to_string(), tokens::muted_style()),
        ]
    }

    /// The conversation half, and how many columns of fill the typing field wants after it.
    ///
    /// The fill is returned rather than drawn here because only [`Widget::render`] knows where
    /// the selectors start, and the field runs up to them — Chris: it fills *"the rest of the
    /// row"*. A field that stopped at the end of its own text would read as a word on a coloured
    /// background rather than as somewhere to type.
    fn conversation(&self) -> (Vec<Span<'static>>, bool) {
        match &self.dialog {
            Dialog::Idle => (Vec::new(), false),
            Dialog::SearchInput(term) => (Self::typing(SEARCH_PROMPT, term), true),
            Dialog::FilterInput(term) => (Self::typing(FILTER_PROMPT, term), true),
            Dialog::SearchOn { term, hit, hits } => (
                Self::settled(SEARCH_ON, term, format!("{hit}/{hits}")),
                false,
            ),
            Dialog::FilterOn { term, rows } => (
                Self::settled(FILTER_ON, term, format!("{rows} rows")),
                false,
            ),
        }
    }

    /// A dialog being typed into: the prompt, then the crate's own edit-in-place field.
    ///
    /// [`crate::widgets::edit_field`] draws the caret, so this screen and the config table are
    /// the same field in two places rather than two fields that look alike. Insert mode with the
    /// caret past the last character is where typing leaves it, which is the state a still frame
    /// of somebody typing should be in.
    fn typing(prompt: &str, term: &str) -> Vec<Span<'static>> {
        let mut spans = vec![Span::styled(prompt.to_string(), tokens::muted_style())];
        spans.extend(caret_spans(
            &Edit::insert(term),
            tokens::normal_style().bg(tokens::edit_bg()),
        ));
        spans
    }
}

fn width(spans: &[Span<'static>]) -> usize {
    spans.iter().map(|s| s.content.chars().count()).sum()
}

impl Widget for DialogSlot {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }
        let (mut left, fills) = self.conversation();
        let right = self.selectors();
        let slack = (area.width as usize).saturating_sub(width(&left) + width(&right));

        if fills {
            // The field runs to where the settings begin, so there is somewhere to type into
            // rather than a coloured word. One clear column before the settings, so the two
            // halves do not touch.
            let pad = slack.saturating_sub(usize::from(!right.is_empty()));
            left.push(Span::styled(
                " ".repeat(pad),
                tokens::normal_style().bg(tokens::edit_bg()),
            ));
            left.push(Span::raw(" ".repeat(slack - pad)));
        } else {
            left.push(Span::raw(" ".repeat(slack)));
        }
        left.extend(right);
        Paragraph::new(Line::from(left)).render(Rect { height: 1, ..area }, buf);
    }
}

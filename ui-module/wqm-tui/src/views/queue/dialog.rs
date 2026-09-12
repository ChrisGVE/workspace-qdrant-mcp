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
//! # Two conversations, then the settings
//!
//! **Left: the conversations.** The search and the filter are independent (Chris, 2026-09-07:
//! *"I was thinking they were mutually exclusive but they are not"*), so the row holds both
//! when both are held: whichever was opened first on the left — the order they arrived in,
//! which is the order a reader remembers — the second to its right, three spaces between. One
//! alone sits on the left, as it always did. Each leaves by its own door, and each settled
//! conversation says so beside itself: `Esc` takes the search, `f` takes the filter.
//!
//! **Right: the settings.** `op update · status failed`. Chris did not say where the selectors
//! are shown; a knob whose position is only in the reader's memory is a knob they will forget
//! they turned, and *"why is the list empty"* is the question that follows. So they are drawn, at
//! the right end of the row, on every one of its states — and omitted entirely when they are
//! `All`, because a selector at its default is not a setting anybody made.
//!
//! *Supervisor's ruling on where they go, not yet Chris's.*

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use super::state::{Filter, First, QueueState, Search};
use crate::tokens;
use crate::widgets::edit_field::{caret_spans, Edit};

/// Columns between one fact on this row and the next — the same three
/// [`crate::tokens::key_hints`] puts between two hints, and the same
/// [`crate::panes::status_block`] puts between two groups. It is also the three between the
/// two conversations, so the row reads as a sequence of facts at one spacing rather than two.
const GAP: &str = "   ";

/// The prompts, spelled once each. Chris's own words, and the shared `term/regex` half is why
/// they sit together: the two dialogs differ in one verb, and a reader should see that they do.
const SEARCH_PROMPT: &str = "search term/regex: ";
const FILTER_PROMPT: &str = "filter term/regex: ";
const SEARCH_ON: &str = "search on: ";
const FILTER_ON: &str = "filter on: ";
/// What each settled conversation ends with. A dialog that did not say how it ends is one the
/// reader has to guess their way out of — §6's rule for a modal, and a dialog owes the same.
/// The two differ, because the two conversations leave by different doors: Esc takes the
/// search, and the filter is taken by `f` — the key that opened it.
const SEARCH_EXIT: &str = "Esc to cancel";
const FILTER_EXIT: &str = "f to clear";

/// The row under the status block: the conversations on the left, the selectors on the right.
///
/// Built from the whole [`QueueState`] rather than from the slots passed in one by one, because
/// the row's left half is a fact about TWO slots — which is held, and which was opened first —
/// and a caller hand-feeding the pieces would be a second place that could feed them in the
/// wrong order.
pub struct DialogSlot<'a> {
    state: &'a QueueState,
}

impl<'a> DialogSlot<'a> {
    pub fn new(state: &'a QueueState) -> Self {
        Self { state }
    }

    /// The settings half: `op update · status failed`, label muted and value normal, and
    /// nothing at all when both are `All`.
    ///
    /// The label is muted because it never changes and the value is what a reader is checking;
    /// the same split the status block draws between a glyph's word and its count.
    fn selectors(&self) -> Vec<Span<'static>> {
        let mut shown: Vec<(&str, String)> = Vec::new();
        if let Some(op) = self.state.op {
            shown.push(("op", op.label().to_string()));
        }
        if let Some(status) = self.state.status {
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
        spans.extend(self.count());
        spans
    }

    /// How many rows are selected, at the right-hand end of the row (Chris, 20260907,
    /// ruling 10: *"count right-aligned on the dialog row"*).
    ///
    /// **Rightmost of the right-hand group, after the selectors.** A selector is a setting that
    /// stays until it is changed; a count changes on every press, and a number that moves the
    /// two settled words beside it every time a row is picked is a wobble a reader cannot stop
    /// noticing. Last, it moves nothing.
    ///
    /// Absent at zero rather than reading `0 selected`: the whole slot is silent when nothing
    /// has been said to it, and an empty selection is nothing said.
    fn count(&self) -> Vec<Span<'static>> {
        let selected = self.state.selection.len();
        if selected == 0 {
            return Vec::new();
        }
        let mut spans = Vec::new();
        if self.state.op.is_some() || self.state.status.is_some() {
            spans.push(Span::styled(" · ", tokens::muted_style()));
        }
        spans.push(Span::styled(
            crate::format::grouped(selected as u64),
            ratatui::style::Style::default().fg(tokens::selected()),
        ));
        spans.push(Span::styled(" selected", tokens::muted_style()));
        spans
    }

    /// A settled conversation: what it is, the term, what it found, and how to leave.
    fn settled(prompt: &str, term: &str, found: String, exit: &str) -> Vec<Span<'static>> {
        vec![
            Span::styled(prompt.to_string(), tokens::muted_style()),
            Span::styled(term.to_string(), tokens::normal_style()),
            Span::styled(GAP, tokens::muted_style()),
            Span::styled(found, tokens::normal_style()),
            Span::styled(GAP, tokens::muted_style()),
            Span::styled(exit.to_string(), tokens::muted_style()),
        ]
    }

    /// The search conversation, and whether it wants a typing field after it.
    fn search(search: &Search) -> (Vec<Span<'static>>, bool) {
        match search {
            Search::Input(term) => (Self::typing(SEARCH_PROMPT, term), true),
            Search::On { term, hit, hits } => (
                Self::settled(SEARCH_ON, term, format!("{hit}/{hits}"), SEARCH_EXIT),
                false,
            ),
        }
    }

    /// The filter conversation, likewise.
    fn filter(filter: &Filter) -> (Vec<Span<'static>>, bool) {
        match filter {
            Filter::Input(term) => (Self::typing(FILTER_PROMPT, term), true),
            Filter::On { term, rows } => (
                Self::settled(FILTER_ON, term, format!("{rows} rows"), FILTER_EXIT),
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

/// The fill a typing field is drawn with — the "somewhere to type" half of the field.
fn field_fill(columns: usize) -> Span<'static> {
    Span::styled(
        " ".repeat(columns),
        tokens::normal_style().bg(tokens::edit_bg()),
    )
}

impl Widget for DialogSlot<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }
        // The two conversations in the order they were opened: the elder on the left, the
        // second to its right. [`First`] is only read when both are held — with one, that one
        // sits on the left whatever the marker says.
        let (left, right) = match self.state.first {
            First::Search => (
                self.state.search.as_ref().map(Self::search),
                self.state.filter.as_ref().map(Self::filter),
            ),
            First::Filter => (
                self.state.filter.as_ref().map(Self::filter),
                self.state.search.as_ref().map(Self::search),
            ),
        };
        let settings = self.selectors();
        let both = left.is_some() && right.is_some();
        let left_typing = left.as_ref().is_some_and(|(_, typing)| *typing);
        let right_typing = right.as_ref().is_some_and(|(_, typing)| *typing);

        // The slack, and how much of it each typing field runs for. A field runs to whatever
        // follows it — the other conversation, or the settings — because a field that stopped
        // at the end of its own text would read as a word on a coloured background rather than
        // as somewhere to type. When BOTH are being typed into at once they share the slack,
        // the elder taking the odd column.
        let slack = (area.width as usize).saturating_sub(
            left.as_ref().map_or(0, |(spans, _)| width(spans))
                + right.as_ref().map_or(0, |(spans, _)| width(spans))
                + usize::from(both) * GAP.chars().count()
                + width(&settings),
        );
        let mut left_fill = 0;
        let mut right_fill = 0;
        match (left_typing, right_typing) {
            (true, true) => {
                left_fill = slack / 2;
                right_fill = slack - left_fill;
            }
            (true, false) => left_fill = slack,
            (false, true) => right_fill = slack,
            (false, false) => {}
        }
        // One clear column before the settings, so a field and a setting never touch — the
        // same reservation the single-conversation row made, on whichever field is last.
        if !settings.is_empty() {
            if right_typing {
                right_fill = right_fill.saturating_sub(1);
            } else if right.is_none() && left_typing {
                left_fill = left_fill.saturating_sub(1);
            }
        }

        let mut row: Vec<Span<'static>> = Vec::new();
        if let Some(spans) = left.as_ref() {
            row.extend(spans.0.iter().cloned());
            if left_fill > 0 {
                row.push(field_fill(left_fill));
            }
            if both {
                row.push(Span::styled(GAP, tokens::muted_style()));
            }
        }
        if let Some(spans) = right.as_ref() {
            row.extend(spans.0.iter().cloned());
            if right_fill > 0 {
                row.push(field_fill(right_fill));
            }
        }
        let used = width(&row) + width(&settings);
        row.push(Span::raw(
            " ".repeat((area.width as usize).saturating_sub(used)),
        ));
        row.extend(settings);
        Paragraph::new(Line::from(row)).render(Rect { height: 1, ..area }, buf);
    }
}

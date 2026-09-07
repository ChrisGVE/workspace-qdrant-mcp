//! What the Queue tab is showing, and the one function that turns it into rows.
//!
//! Every frame of this tab is [`QueueState`] plus the fixture, and every guard reads the same
//! [`project`] the frames do. That is the whole point of the file: a test that re-implemented
//! "selector, then filter, then page" would be checking one copy of the rule against another,
//! and would agree with the screen exactly as long as both copies were wrong together.
//!
//! # Search and filter are mutually exclusive because [`Dialog`] is one value
//!
//! Chris's two dialogs do different things — a filter reloads the list, a search moves the
//! cursor within it — and starting one must abandon the other. That is not a rule anybody
//! applies here; it is the shape of the type. A struct with `search: Option<..>` beside
//! `filter: Option<..>` could hold both at once, and then somebody would have to remember.
//!
//! # The selectors are NOT part of the dialog, and Esc says so
//!
//! `t` and `s` narrow the buffer; `/` and `f` open a dialog over it. Esc leaves the dialog and
//! leaves the selectors exactly where they were ([`QueueState::escape`]) — because a selector is
//! a *setting* the reader made, and a dialog is a *conversation* they are in. Losing a setting
//! by pressing Escape out of an unrelated conversation is the failure this separation prevents.

use super::fixture::QueueRow;
use crate::panes::cell::Sort;
use crate::panes::list::LIST_PAGE;
use crate::panes::status_block::QUEUE_LABELS;

/// What the one-letter `T` column shows, and what the `t` selector cycles through.
///
/// The four canonical collections of ADR-001, one letter each, exactly as v0.1 abbreviates
/// them. Not sourced from `wqm_common::names::Collection`: these are the letters v0.1 draws in a
/// one-column field, and the collection names are words — the day they should be the same thing
/// is a decision, not an inference.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Project,
    Library,
    Scratchpad,
    Rules,
}

impl Kind {
    /// The cycle order Chris gave: `All → P → L → S → R → All`.
    pub const CYCLE: [Kind; 4] = [Kind::Project, Kind::Library, Kind::Scratchpad, Kind::Rules];

    /// The letter the `T` column and the selector both show. One producer, so the column and
    /// the knob above it cannot spell the same thing two ways.
    pub const fn letter(self) -> &'static str {
        match self {
            Kind::Project => "P",
            Kind::Library => "L",
            Kind::Scratchpad => "S",
            Kind::Rules => "R",
        }
    }
}

/// Where a queued item has got to. The three v0.1 shows, and the three the status block counts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Pending,
    InProgress,
    Failed,
}

impl Status {
    pub const CYCLE: [Status; 3] = [Status::Pending, Status::InProgress, Status::Failed];

    /// The word the `Status` column shows — read out of
    /// [`crate::panes::status_block::QUEUE_LABELS`] rather than spelled again here, so the
    /// column in the middle of the screen and the counts at the top of it say `in progress` the
    /// same way. v0.1 writes the column as `in_progress` and the block as `in progress`; the
    /// block's spelling wins, because it is the one a reader reads as English.
    pub const fn label(self) -> &'static str {
        match self {
            Status::Pending => QUEUE_LABELS[0],
            Status::InProgress => QUEUE_LABELS[1],
            Status::Failed => QUEUE_LABELS[2],
        }
    }

    /// The hue the word carries — the same three the status block gives its own counts, so
    /// *failed* is the same colour wherever the screen says it.
    pub const fn hue(self) -> fn() -> ratatui::style::Color {
        match self {
            Status::Pending => crate::tokens::degraded,
            Status::InProgress => crate::tokens::in_flight,
            Status::Failed => crate::tokens::offline,
        }
    }
}

/// What the dialog slot — the blank row under the status block — is holding.
///
/// Five states and no sixth: `Idle` is the row with nothing on it but the selectors, and the
/// other four are the two conversations in their two halves, typing and settled.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Dialog {
    Idle,
    SearchInput(String),
    /// A search that has been accepted: which hit the cursor is on (1-based) and how many there
    /// are, counted over the BUFFERED rows rather than over the visible ones — a `3/17` that
    /// meant "three of the seventeen you can see" would change when the terminal was resized.
    SearchOn {
        term: String,
        hit: usize,
        hits: usize,
    },
    FilterInput(String),
    /// A filter that has been accepted, and how many rows came back.
    FilterOn {
        term: String,
        rows: usize,
    },
}

impl Dialog {
    /// The term this dialog is carrying, whichever half it is in. `None` when idle.
    pub fn term(&self) -> Option<&str> {
        match self {
            Dialog::Idle => None,
            Dialog::SearchInput(term) | Dialog::FilterInput(term) => Some(term),
            Dialog::SearchOn { term, .. } | Dialog::FilterOn { term, .. } => Some(term),
        }
    }

    /// Whether this dialog NARROWS the list, as opposed to moving the cursor within it. The
    /// whole difference between the two conversations, asked once.
    fn filters(&self) -> bool {
        matches!(self, Dialog::FilterInput(_) | Dialog::FilterOn { .. })
    }
}

/// Everything the Queue tab is showing that is not the data.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct QueueState {
    pub dialog: Dialog,
    pub kind: Option<Kind>,
    pub status: Option<Status>,
    pub sort: Option<Sort>,
    pub cursor: usize,
}

impl Default for QueueState {
    fn default() -> Self {
        Self {
            dialog: Dialog::Idle,
            kind: None,
            status: None,
            sort: None,
            cursor: 0,
        }
    }
}

impl QueueState {
    /// `/` pressed: the search dialog opens carrying whatever term is already in hand.
    ///
    /// Chris: pressing `/` again is the input state *"with the term pre-loaded"*. Re-typing a
    /// regex you can already see on the screen is the thing this saves, and it is why the term
    /// is taken from the dialog rather than reset.
    pub fn open_search(&self) -> Self {
        Self {
            dialog: Dialog::SearchInput(self.dialog.term().unwrap_or_default().to_string()),
            ..self.clone()
        }
    }

    /// `f` pressed. Same shape, and — see the module docs — the same value, so opening this
    /// abandons any search by construction.
    pub fn open_filter(&self) -> Self {
        Self {
            dialog: Dialog::FilterInput(self.dialog.term().unwrap_or_default().to_string()),
            ..self.clone()
        }
    }

    /// Esc: leave the dialog, and leave the selectors alone. See the module docs.
    pub fn escape(&self) -> Self {
        Self {
            dialog: Dialog::Idle,
            ..self.clone()
        }
    }

    /// `t`: the next type that has rows, or back to All. See [`cycle`].
    pub fn next_kind(&self, buffer: &[QueueRow]) -> Self {
        Self {
            kind: cycle(&Kind::CYCLE, self.kind, |k| {
                buffer.iter().any(|row| row.kind == *k)
            }),
            ..self.clone()
        }
    }

    /// `s`: the next status that has rows, or back to All.
    pub fn next_status(&self, buffer: &[QueueRow]) -> Self {
        Self {
            status: cycle(&Status::CYCLE, self.status, |s| {
                buffer.iter().any(|row| row.status == *s)
            }),
            ..self.clone()
        }
    }
}

/// The next value in a selector's cycle, skipping every value `has_rows` says nothing.
///
/// **The skip is measured against the whole buffer, not against what the other selector has
/// left.** Two selectors that each hid values the other had emptied would make the cycle depend
/// on the order they were pressed in, and a knob whose stops move is a knob nobody can learn. It
/// does mean the two together can land on an empty list — `type L, status failed` — which is a
/// state the screen can honestly draw (`No data`) rather than one it must prevent.
///
/// *Supervisor's ruling on the reading of "no rows in the buffer", not yet Chris's.*
fn cycle<T: Copy + PartialEq>(
    values: &[T],
    current: Option<T>,
    has_rows: impl Fn(&T) -> bool,
) -> Option<T> {
    let from = match current {
        None => 0,
        Some(value) => values.iter().position(|v| *v == value).map_or(0, |i| i + 1),
    };
    values[from..].iter().find(|value| has_rows(value)).copied()
}

/// Whether a term matches a row.
///
/// **Tenant, Object, Type and Op — and deliberately not Status or Size or Age.** Those three are
/// what the selectors and the sort keys are for; a search that also matched `failed` would make
/// `/failed` and `s` two ways to do one thing that disagree about what happens to the cursor.
///
/// Case-insensitive substring. Chris's label says *"search term/regex"* and v0.1 does take a
/// regex; this crate is a storyboard with no regex engine and adding one to draw a still frame
/// would be a dependency bought for nothing. Every term the frames use is a literal, for which
/// the two agree. **Deviation, flagged rather than papered over.**
pub fn matches(row: &QueueRow, term: &str) -> bool {
    if term.is_empty() {
        return true;
    }
    let term = term.to_lowercase();
    [row.tenant, row.object, row.item, row.op]
        .iter()
        .any(|field| field.to_lowercase().contains(&term))
}

/// The rows the screen shows, from the buffer and the state: **selector → filter → page**.
///
/// One function, called by the frames and by every guard. The order is the one the rules
/// require and not an arbitrary one: the selectors narrow the buffer *in place* (Chris: they
/// *"filter the buffer"*), a filter then **reloads** — which is why the page is taken after it
/// and not before — and the sort is applied last, by [`crate::panes::list::ListPane::sorted`],
/// because sorting is about the order of what survived rather than about what survived.
///
/// A search does NOT appear here. It moves the cursor within the rows a filter already chose;
/// see [`hits`].
pub fn project<'a>(buffer: &'a [QueueRow], state: &QueueState) -> Vec<&'a QueueRow> {
    buffer
        .iter()
        .filter(|row| state.kind.is_none_or(|kind| row.kind == kind))
        .filter(|row| state.status.is_none_or(|status| row.status == status))
        .filter(|row| {
            !state.dialog.filters() || matches(row, state.dialog.term().unwrap_or_default())
        })
        .take(LIST_PAGE)
        .collect()
}

/// Which of the projected rows a search term hits, as indices into the projection.
///
/// Over the whole projection rather than over the visible window, so `3/17` is a fact about the
/// list and not about the terminal. See [`Dialog::SearchOn`].
pub fn hits(rows: &[&QueueRow], term: &str) -> Vec<usize> {
    if term.is_empty() {
        return Vec::new();
    }
    rows.iter()
        .enumerate()
        .filter(|(_, row)| matches(row, term))
        .map(|(at, _)| at)
        .collect()
}

//! What the Queue tab is showing, and the one function that turns it into rows.
//!
//! Every frame of this tab is [`QueueState`] plus the fixture, and every guard reads the same
//! [`project`] the frames do. That is the whole point of the file: a test that re-implemented
//! "selector, then filter, then page" would be checking one copy of the rule against another,
//! and would agree with the screen exactly as long as both copies were wrong together.
//!
//! # Search and filter are two slots, and both can be held at once
//!
//! Chris, 2026-09-07: *"I was thinking they were mutually exclusive but they are not"*. This
//! file used to disagree by construction: one `Dialog` value, so opening either conversation
//! abandoned the other, and no reader could hold a narrowed list and a moving cursor at the
//! same time. [`QueueState`] now carries a [`Search`] slot beside a [`Filter`] slot, and each
//! survives the other being opened, accepted and cleared. The two do different jobs and
//! always did — a filter reloads the list, a search moves the cursor within it — which is
//! the whole case for letting them coexist: narrow to the rows you mean, then find your way
//! among them.
//!
//! **And they leave by different doors.** Esc clears the search — typing or settled — and
//! nothing else: the search is the thing that moves (the cursor, on `n` and `N`), and a
//! reader pressing Esc is telling it to stop. The filter is a narrowing the reader built with
//! `f`, and `f` is what unbuilds it ([`QueueState::toggle_filter`]): one key, one meaning, in
//! both directions. The selectors are spared both keys — a setting survives every
//! conversation, because losing a setting by walking out of an unrelated conversation is the
//! failure this separation exists to prevent.

use super::fixture::QueueRow;
use crate::motion::Motion;
use crate::panes::cell::Sort;
use crate::panes::list::LIST_PAGE;
use crate::panes::status_block::QUEUE_LABELS;

/// What the one-letter `T` column shows.
///
/// **NOT contract-bound (UIQ pending).** The four names are ADR-001's, but these one-letter
/// abbreviations are v0.1's drawing of them and nothing on the wire spells them this way.
///
/// The four canonical collections of ADR-001, one letter each, exactly as v0.1 abbreviates
/// them. Not sourced from `wqm_common::names::Collection`: these are the letters v0.1 draws in a
/// one-column field, and the collection names are words — the day they should be the same thing
/// is a decision, not an inference.
///
/// The type **selector** is gone (2026-09-07) and the column stays: the free-text filter
/// covers narrowing by it, so a reader who wants one type types the word. The value remains
/// per-row, so the column is still a fact about each row rather than a constant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Project,
    Library,
    Scratchpad,
    Rules,
}

impl Kind {
    /// The letter the `T` column shows. One producer, so the column cannot spell the same
    /// thing two ways.
    pub const fn letter(self) -> &'static str {
        match self {
            Kind::Project => "P",
            Kind::Library => "L",
            Kind::Scratchpad => "S",
            Kind::Rules => "R",
        }
    }
}

/// What the queue will do with an item — the `Op` column, and what the `o` selector cycles
/// through.
///
/// **NOT contract-bound (UIQ pending).** Four operations is v0.1's vocabulary; the contract
/// may name more or fewer, and the cycle will have to follow it when it does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    Add,
    Update,
    Delete,
    Scan,
}

impl Op {
    /// The cycle order: `All → add → update → delete → scan → All`.
    pub const CYCLE: [Op; 4] = [Op::Add, Op::Update, Op::Delete, Op::Scan];

    /// The lowercase word the `Op` column shows. Lowercase because that is v0.1's own
    /// spelling of an operation — `update`, `add`, read off the captured screen rather than
    /// re-cased here. One producer, so the column and the selector cannot spell the same
    /// operation two ways.
    pub const fn label(self) -> &'static str {
        match self {
            Op::Add => "add",
            Op::Update => "update",
            Op::Delete => "delete",
            Op::Scan => "scan",
        }
    }
}

/// Where a queued item has got to. The three v0.1 shows, and the three the status block counts.
///
/// **NOT contract-bound (UIQ pending).** Three states is what the captured screen has; whether
/// the queue really has exactly three is the contract's answer to give.
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

/// The search slot: the conversation that moves the cursor within the rows the filter chose.
///
/// Two states and no third: typing a term, or accepted with the hit the cursor is on and how
/// many there are. The counts are a fact about the projection as it stood when Enter was
/// pressed — recomputing them when the filter changes underneath is the live screen's job, not
/// a still frame's.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Search {
    /// A term being typed.
    Input(String),
    /// A search that has been accepted: which hit the cursor is on (1-based) and how many
    /// there are, counted over the BUFFERED rows rather than over the visible ones — a `3/17`
    /// that meant "three of the seventeen you can see" would change when the terminal was
    /// resized. Counted over the rows the filter left, when one is on: the search cannot move
    /// the cursor onto a row the reader has narrowed away.
    On {
        term: String,
        hit: usize,
        hits: usize,
    },
}

impl Search {
    /// The term this slot carries, whichever state it is in.
    pub fn term(&self) -> &str {
        match self {
            Search::Input(term) => term,
            Search::On { term, .. } => term,
        }
    }
}

/// The filter slot: the conversation that narrows the rows.
///
/// Two states, matching [`Search`] shape for shape: typing a term — already narrowing, which
/// is v0.1's own live behaviour — or accepted with the row count read off the reload.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Filter {
    /// A term being typed, narrowing as it grows.
    Input(String),
    /// A filter that has been accepted, and how many rows came back.
    On {
        term: String,
        rows: usize,
    },
}

impl Filter {
    /// The term this slot carries, whichever state it is in.
    pub fn term(&self) -> &str {
        match self {
            Filter::Input(term) => term,
            Filter::On { term, .. } => term,
        }
    }
}

/// Which of the two conversations was opened first — the one drawn on the left of the row
/// when both are held (see [`super::dialog`]).
///
/// Rewritten only when a slot OPENS out of nothing: the slot already held is the elder and
/// keeps the left, and a slot reopening after Esc or `f` is a newcomer like any other. Not
/// rewritten when a slot clears — with one slot left there is nothing to order, and the stale
/// value decides nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum First {
    Search,
    Filter,
}

/// Everything the Queue tab is showing that is not the data.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct QueueState {
    /// The conversation that moves the cursor. See [`Search`].
    pub search: Option<Search>,
    /// The conversation that narrows the rows. See [`Filter`].
    pub filter: Option<Filter>,
    /// Which of the two was opened first. See [`First`].
    pub first: First,
    /// The `o` selector.
    pub op: Option<Op>,
    /// The `s` selector.
    pub status: Option<Status>,
    pub sort: Option<Sort>,
    pub cursor: usize,
}

impl Default for QueueState {
    fn default() -> Self {
        Self {
            search: None,
            filter: None,
            // Arbitrary until the second conversation opens — `/` is the likelier first
            // press, so the default names the search. A value that cannot decide anything
            // until it is written again is not a value worth wrapping in an Option.
            first: First::Search,
            op: None,
            status: None,
            sort: None,
            cursor: 0,
        }
    }
}

impl QueueState {
    /// The `first` a slot should record as it opens out of nothing.
    ///
    /// An opening slot is the newcomer by definition, so the answer is the OTHER conversation
    /// whenever one is already held, and the opening slot itself otherwise — the
    /// elder-on-the-left rule ([`First`]) stated once for the two keys that open things
    /// rather than twice, once in each.
    fn opened_first(&self, opening: First, other: First) -> First {
        let held = match other {
            First::Search => self.search.is_some(),
            First::Filter => self.filter.is_some(),
        };
        if held {
            other
        } else {
            opening
        }
    }

    /// `/` pressed: the search input opens carrying whatever term the search slot already
    /// holds.
    ///
    /// Chris: pressing `/` again is the input state *"with the term pre-loaded"*. Re-typing a
    /// regex you can already see on the screen is the thing this saves. The FILTER's term is
    /// deliberately not carried over: the two slots are independent, and a search pre-loaded
    /// with somebody else's narrowing is a search the reader did not ask for. The filter
    /// itself is not touched — see the module docs.
    pub fn open_search(&self) -> Self {
        let reopening = self.search.is_some();
        Self {
            search: Some(Search::Input(
                self.search
                    .as_ref()
                    .map(|search| search.term())
                    .unwrap_or_default()
                    .to_string(),
            )),
            first: if reopening {
                self.first
            } else {
                self.opened_first(First::Search, First::Filter)
            },
            ..self.clone()
        }
    }

    /// `f` pressed — one key, both directions. With no filter, the input opens; with a
    /// filter accepted, the filter goes.
    ///
    /// The input opens EMPTY, unlike `/`'s pre-load: a filter this method cleared left no
    /// term worth carrying, and the reader who wants it back can retype it into a field that
    /// is already narrowing. While the input is being typed into, `f` is a letter going into
    /// the term rather than a command, so the method leaves that conversation alone.
    ///
    /// See the module docs for why clearing the filter is `f`'s job rather than Esc's.
    pub fn toggle_filter(&self) -> Self {
        match self.filter.as_ref() {
            None => Self {
                filter: Some(Filter::Input(String::new())),
                first: self.opened_first(First::Filter, First::Search),
                ..self.clone()
            },
            Some(Filter::Input(_)) => self.clone(),
            Some(Filter::On { .. }) => Self {
                filter: None,
                ..self.clone()
            },
        }
    }

    /// Enter on a search: count the hits over the projection and put the cursor on the first.
    ///
    /// **The counts are computed, never stated.** A frame that wrote `hits: 4` beside a
    /// projection holding five would be a screen telling its reader something false, and no
    /// guard reading the same frame could tell. So `3/17` and the cursor both come out of
    /// [`hits`], and a guard checks the drawn numbers against it rather than against a
    /// literal.
    ///
    /// A term nobody matches gives `0/0` and leaves the cursor where it was: there is no first
    /// hit to move to, and moving to row one would look like a hit.
    pub fn accept_search(&self, buffer: &[QueueRow]) -> Self {
        let term = self
            .search
            .as_ref()
            .map(|search| search.term())
            .unwrap_or_default()
            .to_string();
        let found = hits(&project(buffer, self), &term);
        Self {
            cursor: found.first().copied().unwrap_or(self.cursor),
            search: Some(Search::On {
                hit: usize::from(!found.is_empty()),
                hits: found.len(),
                term,
            }),
            ..self.clone()
        }
    }

    /// Enter on a filter: the list reloads as the first page of what matched, and the count is
    /// read off that reload rather than stated. The cursor goes back to the top, because the
    /// row it was on may not have survived.
    ///
    /// The search, if one is held, is untouched — its counts stay the facts they were when
    /// its own Enter was pressed.
    pub fn accept_filter(&self, buffer: &[QueueRow]) -> Self {
        let term = self
            .filter
            .as_ref()
            .map(|filter| filter.term())
            .unwrap_or_default()
            .to_string();
        let settled = Self {
            filter: Some(Filter::On {
                term: term.clone(),
                rows: 0,
            }),
            cursor: 0,
            ..self.clone()
        };
        Self {
            filter: Some(Filter::On {
                rows: project(buffer, &settled).len(),
                term,
            }),
            ..settled
        }
    }

    /// Esc: the search leaves — typing or settled — and everything else stays. See the module
    /// docs: the search is the moving thing, and the filter unbuilds with the key that built
    /// it.
    pub fn escape(&self) -> Self {
        Self {
            search: None,
            ..self.clone()
        }
    }

    /// `o`: the next operation that has rows, or back to All. See [`cycle`].
    pub fn next_op(&self, buffer: &[QueueRow]) -> Self {
        Self {
            op: cycle(&Op::CYCLE, self.op, |op| {
                buffer.iter().any(|row| row.op == *op)
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

    /// Apply a motion to the cursor, `count` times, clamped to the projection.
    ///
    /// The list half of the shared movement model ([`crate::motion`]): a [`Motion`] and a count
    /// arrive, and this turns them into the cursor the list will draw. `nos` are the projection's
    /// invariant `No` values in drawn order — one per row — so the cursor's range is
    /// `0..nos.len()` and [`Motion::Row`] can find the row a number names rather than guess a
    /// drawn position. `page` is how many rows the list shows at once: the step
    /// [`Motion::PageDown`] and [`Motion::PageUp`] take.
    ///
    /// The count is passed even for the motions that ignore it — [`Motion::Top`],
    /// [`Motion::Bottom`] and [`Motion::Row`] — because [`crate::motion::Prefix::key`] returns it
    /// uniformly and only the repeating motions consume it. A [`Motion::Row`] whose number is not
    /// in the projection leaves the cursor where it was: there is no row to move to, and moving to
    /// the nearest neighbour would look like the row was found.
    pub fn moved(&self, motion: Motion, count: usize, page: usize, nos: &[u16]) -> Self {
        let last = nos.len().saturating_sub(1);
        let cursor = match motion {
            Motion::Up => self.cursor.saturating_sub(count),
            Motion::Down => self.cursor.saturating_add(count).min(last),
            Motion::PageUp => self.cursor.saturating_sub(count.saturating_mul(page)),
            Motion::PageDown => self.cursor.saturating_add(count.saturating_mul(page)).min(last),
            Motion::Top => 0,
            Motion::Bottom => last,
            Motion::Row(n) => nos
                .iter()
                .position(|&no| usize::from(no) == n)
                .unwrap_or(self.cursor),
        };
        Self {
            cursor,
            ..self.clone()
        }
    }
}

/// The next value in a selector's cycle, skipping every value `has_rows` says nothing.
///
/// **The skip is measured against the whole buffer, not against what the other selector has
/// left.** Two selectors that each hid values the other had emptied would make the cycle depend
/// on the order they were pressed in, and a knob whose stops move is a knob nobody can learn. It
/// does mean the two together can land on an empty list — `op update, status failed` — which is
/// a state the screen can honestly draw (`No data`) rather than one it must prevent.
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
/// **Tenant, Object, Type and Op — and deliberately not Status or Size or Age.** Those are
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
    [row.tenant, row.object, row.item, row.op.label()]
        .iter()
        .any(|field| field.to_lowercase().contains(&term))
}

/// The rows the screen shows, from the buffer and the state: **selector → filter → page**, in
/// the buffer's own order until a sort — or the default order below — moves them.
///
/// One function, called by the frames and by every guard. The order of the three stages is the
/// one the rules require and not an arbitrary one: the selectors narrow the buffer *in place*
/// (Chris: they *"filter the buffer"*), a filter then **reloads** — which is why the page is
/// taken after it and not before — and the sort is applied last, by
/// [`crate::panes::list::ListPane::sorted`], because sorting is about the order of what
/// survived rather than about what survived.
///
/// **The default order puts `in progress` first.** With no sort chosen, the rows the queue is
/// actively working on lead and everything else keeps its buffer order — promoted WITHIN the
/// page rather than before it, so the page is still the one a reload would fetch and the
/// promotion only reorders what survived. A chosen sort replaces the default outright: a
/// reader who has asked for an order gets that order, not that order with a silent shuffle in
/// front of it.
///
/// A search does NOT appear here. It moves the cursor within the rows a filter already chose;
/// see [`hits`].
pub fn project<'a>(buffer: &'a [QueueRow], state: &QueueState) -> Vec<&'a QueueRow> {
    let mut rows: Vec<&'a QueueRow> = buffer
        .iter()
        .filter(|row| state.op.is_none_or(|op| row.op == op))
        .filter(|row| state.status.is_none_or(|status| row.status == status))
        .filter(|row| match state.filter.as_ref() {
            Some(filter) => matches(row, filter.term()),
            None => true,
        })
        .take(LIST_PAGE)
        .collect();
    if state.sort.is_none() {
        // Stable, so the in-progress rows keep the buffer's order among themselves and so do
        // the rest — a partition, spelled as the sort it is implemented by.
        rows.sort_by_key(|row| row.status != Status::InProgress);
    }
    rows
}

/// Which of the projected rows a search term hits, as indices into the projection.
///
/// Over the whole projection rather than over the visible window, so `3/17` is a fact about the
/// list and not about the terminal. See [`Search::On`].
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

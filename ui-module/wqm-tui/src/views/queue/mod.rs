//! The Queue — tab 2, and the first list this crate draws.
//!
//! Chris, 2026-09-07: *"let's work on the Queue tab, we'll get inspiration from the current wqm
//! tui"*. The capture that inspiration was read off is kept verbatim beside this crate at
//! `design-notes/V01-QUEUE-CAPTURE.txt`, so the frame can be compared with the thing it
//! reproduces rather than with a memory of it.
//!
//! Rows top to bottom: the constant top ([`crate::views::top::ConstantTop`] — app bar, frame
//! rule, status block, and the rule that closes it), the **dialog slot**, the list's column
//! header, the rows, the foot rule, the foot. Nothing else, and in particular **no frame**:
//! Chris, 2026-09-07, *"no frame around the table, valid for all views"*. v0.1 draws its queue
//! inside a `┌ Queue ┐` box that VL §6 already forbids — a box means a modal or a toast — and
//! that was spending two columns and two rows to repeat what the tab bar says.
//!
//! # What this screen is, in one sentence
//!
//! A [`crate::panes::list::ListPane`] over the captured buffer, and a [`QueueState`] saying what
//! has been done to it. Every frame is that pair; every guard reads the same
//! [`frames::pane`] the frames do.
//!
//! # The dialog slot is a reserved row, not an inserted one
//!
//! See [`dialog`]. It is blank most of the time, and it costs one row of list, and that is the
//! cheaper of the two mistakes: a row that appeared on `/` would push the list down at exactly
//! the moment the reader started looking for something in it.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    widgets::Widget,
};

use crate::panes::list::NAV_HELP;
use crate::panes::status_block::StatusBlock;
use crate::tokens::Health;
use crate::views::top::{self, ConstantTop};
use crate::widgets::chrome::{inset, StatusLine};
use crate::widgets::modal::Modal;

pub mod dialog;
pub mod fixture;
pub mod frames;
#[cfg(feature = "tui-pantry")]
pub mod ingredient;
pub mod state;
#[cfg(test)]
mod tests;

pub use state::{Filter, First, Kind, Op, QueueState, Search, Status};

/// The Queue's index in [`crate::widgets::tab_bar::TabBar::storyboard_tabs`] — tab 2.
pub const QUEUE_TAB: usize = 1;

/// Rows the dialog slot takes: one, always.
pub const DIALOG_ROWS: u16 = 1;

/// The keys that move the data cursor, spelled as the Dashboard spells them. One producer would
/// be better; two screens is not yet enough to know whether this is one word or two rules.
const NAVIGATE_KEYS: &str = "↓↑/jk";

/// **Every key this screen has already spoken for.** A column's sort key must not be one of them
/// — Chris's requirement from the Dashboard (2026-09-07), and it binds here for the same reason:
/// a lit letter promises that pressing it sorts, and a letter that also does something else
/// makes that promise false.
///
/// One list rather than a rule applied from memory, so the day a key is bound the collision is a
/// red test rather than a surprise on the screen.
///
/// `Enter` and `Esc` are bound too and are not in here, because they are not letters — a column
/// cannot collide with them, and a `char` list is the wrong shape to say so. `^D`/`^U`/`^F`/`^B`
/// are likewise out: a control chord is not a letter either, and nothing lights one.
///
/// **The guard compares case-INSENSITIVELY**, and this screen is why. `n` and `N` are two
/// different bindings here (next hit, previous hit), but
/// [`crate::widgets::chrome::keyed_spans`] finds a sort key in a title without regard to case —
/// so a column offering `N` would light the `n` in its own name and a reader would press the one
/// that moves the search.
pub const QUEUE_BOUND_KEYS: [char; 14] = [
    '/', // open search
    'n', // next hit
    'N', // previous hit
    'o', // operation selector
    's', // status selector
    'f', // filter — opens it, and clears it once it is on
    'r', // retry
    'c', // cancel
    'x', // remove
    'q', // quit
    '?', // help
    'j', // cursor down
    'k', // cursor up
    'h', // reserved: left, the vim pair `j k h l` is bound as a set
];

/// The two extra letters the vim pair claims, so the constant above is the whole story.
pub const QUEUE_BOUND_EXTRA: [char; 1] = ['l'];

/// The Queue tab, composed.
pub struct Queue {
    state: QueueState,
    status: StatusBlock,
    modal: Option<Modal>,
    /// Whether the page is drawn beneath a modal, with no modal on top of it. The Dashboard's
    /// own frame: it is how the ruling that the page goes quiet is judged at all.
    under_modal: bool,
}

impl Queue {
    pub fn new(state: QueueState, status: StatusBlock) -> Self {
        Self {
            state,
            status,
            modal: None,
            under_modal: false,
        }
    }

    /// Open a modal over this page. The page beneath goes quiet on its own — see the Dashboard's
    /// [`crate::views::dashboard::Dashboard::under_modal`] for why the scope is held at the view
    /// and nowhere below it.
    pub fn modal(mut self, modal: Modal) -> Self {
        self.modal = Some(modal);
        self.under_modal = true;
        self
    }

    /// The page as it looks beneath a modal, with the modal itself left undrawn.
    pub fn under_modal(mut self, under: bool) -> Self {
        self.under_modal = under;
        self
    }

    /// The keys the foot offers, in the order Chris gave them.
    ///
    /// **An empty list offers two hints and no more.** Chris, 2026-09-07: on an empty list the
    /// foot is `? Help · q Quit` alone. Every other key on this screen acts on a row — retry it,
    /// cancel it, remove it, search for it — and a hint is a promise that the key does
    /// something.
    ///
    /// Two of the rest are conditional for the same reason. `↓↑/jk Navigate` needs somewhere to
    /// navigate to, so it wants more than one row. `n/N Next/Prev` moves between search hits, so
    /// it appears only while a search is on — and disappears again on Esc.
    ///
    /// *The ORDER is the supervisor's ruling, not yet Chris's.* It runs: move, then find, then
    /// narrow, then act, then leave — so the keys that only look at the list come before the
    /// three that change it.
    pub fn hints(&self) -> Vec<(&'static str, &'static str)> {
        let rows = frames::pane(&self.state).len();
        if rows == 0 {
            return vec![("?", "Help"), ("q", "Quit")];
        }
        let mut hints = Vec::new();
        if rows > 1 {
            hints.push((NAVIGATE_KEYS, "Navigate"));
        }
        hints.push(("/", "Search"));
        if matches!(self.state.search, Some(Search::On { .. })) {
            hints.push(("n/N", "Next/Prev"));
        }
        for pair in [
            ("f", "Filter"),
            ("o", "Op"),
            ("s", "Status"),
            ("r", "Retry"),
            ("c", "Cancel"),
            ("x", "Remove"),
            ("?", "Help"),
            ("q", "Quit"),
        ] {
            hints.push(pair);
        }
        hints
    }

    /// Every key this view has, in the order the help lists them.
    ///
    /// Exposed rather than built inline so a guard can read what the window DECLARES rather than
    /// scraping it back out of a rendered box — where `r` and `c` are indistinguishable from the
    /// hundreds of `r`s and `c`s in the list behind it.
    ///
    /// The paging chords come from [`NAV_HELP`] (Chris, 2026-09-07: *"shown only in the help…
    /// valid for all lists including the dashboard"*), so the day the Dashboard grows a help
    /// modal the two say the same words. `?` and `q` are in here too: the foot offers them, and
    /// a window that claimed to list every key while omitting the two on every screen would be
    /// the one place a reader could not check.
    pub fn help_keys() -> Vec<(&'static str, &'static str)> {
        vec![
            ("↓↑ / j k", "Move the cursor"),
            NAV_HELP[0],
            NAV_HELP[1],
            ("Enter", "Open the item, or load the next page"),
            ("/  n N", "Search; next and previous hit"),
            ("f", "Filter the list; again clears it"),
            ("o", "Cycle the operation"),
            ("s", "Cycle the status"),
            ("r  c  x", "Retry, cancel, remove"),
            ("Esc", "Leave the search"),
            ("?", "This window"),
            ("q", "Quit"),
        ]
    }

    /// The help modal: [`Queue::help_keys`], laid out, over the quietened page.
    pub fn help() -> Modal {
        let keys = Self::help_keys();
        // The key column is as wide as the widest key plus two, MEASURED rather than stated: the
        // paging chords are the longest entries and they come from another module, so a number
        // written here would be a number that silently stopped fitting.
        let column = keys
            .iter()
            .map(|(key, _)| key.chars().count())
            .max()
            .unwrap_or(0)
            + 2;
        let mut body: Vec<String> = keys
            .iter()
            .map(|(key, what)| format!("{key:<column$}{what}"))
            .collect();
        body.push(String::new());
        body.push("The selectors survive Esc; the search leaves on Esc, the filter on f.".to_string());
        Modal::with_body("Queue — keys", body).action("Esc", "close")
    }
}

impl Widget for Queue {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let hints = self.hints();
        let pane = frames::pane(&self.state);
        // Held for the whole page, exactly as the Dashboard holds it: the top, the slot, the
        // list and the foot are all "beneath the modal", and each reads its colours from the
        // tokens while this is alive.
        let quiet = self.under_modal.then(crate::tokens::ModalScope::enter);
        let body = ConstantTop::new(QUEUE_TAB)
            .status(self.status)
            .draw(area, buf);
        if body.height <= DIALOG_ROWS + top::FOOT_ROWS {
            return;
        }

        dialog::DialogSlot::new(&self.state).render(inset(top::row(body, 0)), buf);

        let below = Rect {
            y: body.y + DIALOG_ROWS,
            height: body.height - DIALOG_ROWS,
            ..body
        };
        let (list_area, foot_row) = top::foot(below, buf);
        pane.render(inset(list_area), buf);

        let mut status = StatusLine::new();
        for (key, label) in hints {
            status = status.hint(key, label);
        }
        status.render(inset(foot_row), buf);

        // The modal last and outside the scope: it is the thing in focus, so it keeps its own
        // colours while everything under it has already been drawn quiet.
        drop(quiet);
        if let Some(modal) = self.modal {
            let at = modal.rect(area);
            modal.render(at, buf);
        }
    }
}

/// The roll-up this tab's status block states, from the same inputs the block is built from.
///
/// The foot no longer repeats it (Chris, 2026-09-07): the block at the top is the permanent
/// detailed health, and a summary of it at the foot was redundant. Exposed so a frame cannot
/// build the block's headline apart from the parts under it.
pub fn overall(daemon: Health, entries: &[Health]) -> Health {
    crate::panes::status_block::rollup(daemon, entries)
}

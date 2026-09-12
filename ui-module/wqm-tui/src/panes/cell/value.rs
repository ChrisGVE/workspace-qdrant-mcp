//! What one value in a table row can be, and how it is drawn.
//!
//! Split out of [`super::table`] when the Queue tab needed two shapes the Dashboard never did.
//! The division is the same one [`super::sort`] already draws: `table` is about *columns* —
//! how wide they are, where their headers sit, which one carries the sort mark — and this is
//! about *values*, which is a separate vocabulary that both [`super::CellTable`] and
//! [`crate::panes::list::ListPane`] read from. Two tables sharing a value type is what stops
//! the Dashboard's cells and the Queue's list from disagreeing about what a number looks like.
//!
//! # Every variant here exists because a column could not be honest without it
//!
//! [`Cell::Text`] and [`Cell::Num`] are the two the Dashboard needed. [`Cell::Queue`] came from
//! the triple v0.1 writes as `2'635/0/0` — three facts, so three hues. The Queue tab added the
//! other two, and both are about the same failure: a value whose **printed form and its order
//! are different things**. `64.6 KB` sorts above `4.0 MB` as text and below it as a size, and
//! `12s ago` sorts above `1m ago` as text and below it as an age — so [`Cell::Measured`] carries
//! the magnitude beside the words rather than asking a comparator to parse them back out.
//! [`Cell::Tinted`] is the other half: a word whose hue is the fact (`failed` is not the same
//! kind of news as `pending`), which a plain [`Cell::Text`] has nowhere to put.

use ratatui::{
    style::{Color, Style},
    text::Span,
};

use crate::format::{count_span, grouped};
use crate::tokens;

/// `text` cut to `width`, ending in `…` when it did not fit.
///
/// Counted in CHARACTERS — the mistake this crate has already made once — and the ellipsis
/// takes one of them, so the result is never wider than the column it was measured against.
pub fn fit(text: &str, width: u16) -> String {
    let width = width as usize;
    if text.chars().count() <= width || width == 0 {
        return text.to_string();
    }
    let mut out: String = text.chars().take(width.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// `text` cut to `width` from the LEFT, beginning with `…` when it did not fit.
///
/// The Queue's `Object` column is a file path, and v0.1 shortens it this way for a reason worth
/// keeping: the end of a path is the file, and the file is what a reader is looking for.
/// `…of-book-indexing/sources/chicago.pdf` names something; `/Users/chris/dev/projects/open-b…`
/// names nothing at all, thirty-seven columns in.
pub fn fit_left(text: &str, width: u16) -> String {
    let width = width as usize;
    let count = text.chars().count();
    if count <= width || width == 0 {
        return text.to_string();
    }
    let keep = width.saturating_sub(1);
    let tail: String = text.chars().skip(count - keep).collect();
    format!("…{tail}")
}

/// Which end of a value is dropped when it does not fit its column.
///
/// A property of the COLUMN rather than of the value: whether the head or the tail of a string
/// carries its meaning is a fact about what the column holds, and every value in one column
/// answers it the same way.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Elide {
    /// The default everywhere: names read left to right, so the tail is what goes.
    #[default]
    Right,
    /// File paths, and nothing else so far — see [`fit_left`].
    Left,
}

impl Elide {
    /// `text` cut to `width` from whichever end this says.
    pub fn fit(self, text: &str, width: u16) -> String {
        match self {
            Elide::Right => fit(text, width),
            Elide::Left => fit_left(text, width),
        }
    }
}

/// One value in a row.
pub enum Cell {
    Text(String),
    Num(u64),
    /// The queue triple v0.1 writes as `2'635/0/0` — three figures that are three different
    /// facts, so they carry three different hues rather than being one string.
    Queue {
        pending: u64,
        in_flight: u64,
        failed: u64,
    },
    /// A magnitude printed in human units and ordered by its true one: `64.6 KB` shown,
    /// `66_150` compared. Both halves are carried because neither can be derived from the
    /// other without a parser, and a parser is a second place for the units to be defined.
    ///
    /// `shown` may be empty — v0.1 leaves the size blank on a delete, which has no size — and
    /// an empty one orders as `order`, which the caller sets to zero. A blank that sorted as
    /// text would file every delete between `903.5 KB` and `961 B`.
    Measured {
        shown: String,
        order: u64,
    },
    /// A word whose hue IS the fact. `pending`, `in progress` and `failed` are the three the
    /// Queue tab has, and they take the same three hues the status block's own counts take
    /// ([`crate::panes::status_block`]), so the block at the top of the screen and the column
    /// in the middle of it say the same thing in the same colour.
    ///
    /// The hue is a function rather than a [`Color`] because a colour read at construction time
    /// is a colour read outside [`crate::tokens::ModalScope`] — the frame would keep its full
    /// vibrance under a modal while every neutral around it went quiet.
    Tinted {
        text: String,
        hue: fn() -> Color,
    },
}

impl Cell {
    /// The spans this value is drawn with, for a column `width` columns wide, shortened from
    /// whichever end `elide` names.
    ///
    /// A queue triple is several spans on purpose: its three numbers mean waiting, moving and
    /// lost, and a single-coloured `2'635/0/0` would throw away the only thing that
    /// distinguishes them.
    ///
    /// **Text elides to a shorter name; a figure that does not fit is replaced outright.** An
    /// elided name is still recognisable and the `…` says it was shortened. There is no such
    /// thing as a shortened number — `11'236` cut to five cells is `11'23`, a different value
    /// with nothing to mark it — so a figure too wide for its column is drawn as `…` and the
    /// column width is treated as the defect it is. A [`Cell::Measured`] is a figure by that
    /// test: `4.0 MB` cut to five cells is `4.0 M`, which is not a rounder number, it is a
    /// unitless one.
    ///
    /// The zero rule is the queue triple's alone: `count_span` mutes a zero because *no work
    /// waiting* is not news. A plain figure column keeps its zeros at the normal rung — v0.1's
    /// `Pts` column was all zeros, and muting them would have made the column disappear rather
    /// than recede. (That column has since been dropped altogether, 2026-09-07: a field that is
    /// always zero is better removed than styled.)
    pub fn spans(&self, width: u16, elide: Elide, recede: bool) -> Vec<Span<'static>> {
        // A table that has receded behind the live one is dull in its ENTIRETY (Chris,
        // 20260912, ruling 3) — the hues go with the text, because a status column still
        // painting red beside a grey one is the one thing on the receded table still claiming
        // to be live. One branch here rather than a rule at each call site: the cell is what
        // knows how many spans it has and which of them carry hues.
        if recede {
            return vec![Span::styled(
                elide.fit(&self.plain(), width),
                tokens::muted_style(),
            )];
        }
        match self {
            Cell::Text(text) => vec![Span::styled(elide.fit(text, width), tokens::table_row_style())],
            Cell::Tinted { text, hue } => vec![Span::styled(
                elide.fit(text, width),
                Style::default().fg(hue()),
            )],
            // A figure that does not fit becomes `…` — NEVER a clipped one. Right-aligning
            // `11'236` into five cells renders `11'23`, which is not a truncated number, it is
            // a DIFFERENT number, displayed with no mark to say so. `…` says "there is a value
            // here and it did not fit", which is the only honest thing a too-narrow column can
            // say. The real fix is always the column width, and
            // `views::dashboard::tests` guards that every frame's figure columns are wide
            // enough — this is the net under that guard, not a substitute for it.
            Cell::Num(value) => {
                vec![Span::styled(
                    too_wide(grouped(*value), width),
                    tokens::table_row_style(),
                )]
            }
            Cell::Measured { shown, .. } => {
                vec![Span::styled(
                    too_wide(shown.clone(), width),
                    tokens::table_row_style(),
                )]
            }
            Cell::Queue {
                pending,
                in_flight,
                failed,
            } => vec![
                count_span(*pending, tokens::degraded),
                Span::styled("/", tokens::muted_style()),
                count_span(*in_flight, tokens::in_flight),
                Span::styled("/", tokens::muted_style()),
                count_span(*failed, tokens::offline),
            ],
        }
    }

    /// The value as one plain string — what a receded table draws, where nothing is coloured
    /// and the queue triple's three counts are one grey figure like any other.
    fn plain(&self) -> String {
        match self {
            Cell::Text(text) | Cell::Tinted { text, .. } => text.clone(),
            Cell::Num(value) => grouped(*value),
            Cell::Measured { shown, .. } => shown.clone(),
            Cell::Queue {
                pending,
                in_flight,
                failed,
            } => format!(
                "{}/{}/{}",
                grouped(*pending),
                grouped(*in_flight),
                grouped(*failed)
            ),
        }
    }

    /// How wide this value wants to be, before any column has a say.
    ///
    /// Exists so a guard can ask "does every figure fit its column" of the DATA rather than of
    /// a rendered screen. Read off the render, a figure's ellipsis is indistinguishable from a
    /// name's — both are `…` — and the guard that matters is about numbers only.
    pub fn natural_width(&self) -> usize {
        match self {
            Cell::Text(text) | Cell::Tinted { text, .. } => text.chars().count(),
            Cell::Num(value) => grouped(*value).chars().count(),
            Cell::Measured { shown, .. } => shown.chars().count(),
            Cell::Queue {
                pending,
                in_flight,
                failed,
            } => {
                grouped(*pending).chars().count()
                    + grouped(*in_flight).chars().count()
                    + grouped(*failed).chars().count()
                    + 2
            }
        }
    }

    /// Whether this value is a figure — the kind that must never be shortened.
    pub fn is_figure(&self) -> bool {
        matches!(
            self,
            Cell::Num(_) | Cell::Queue { .. } | Cell::Measured { .. }
        )
    }
}

/// A figure that does not fit its column, replaced by `…` rather than cut. See [`Cell::spans`].
fn too_wide(text: String, width: u16) -> String {
    if text.chars().count() > width as usize {
        "…".to_string()
    } else {
        text
    }
}

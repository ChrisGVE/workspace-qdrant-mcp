//! The constant service status — three rows under the top rule, on every tab but Service.
//!
//! Chris, 20260906: the top of the screen is the same on tabs 1–9, and it answers *"is the
//! system alright?"* before anything a tab has to say. So this pane is not the Service hub's
//! band cut down; it is a **different question**. [`crate::panes::status_band`] shows the
//! federation in detail because the Service hub is where you go to look at it. This shows the
//! roll-up because you are not looking at it — you are on the Queue tab, and you need to know
//! whether to believe what is on the screen.
//!
//! That is why the Service tab does not carry it (see [`crate::views::shell`]): the band and
//! the block would be the same claim, twice, with two chances to disagree.
//!
//! # Three rows, each answering a different question
//!
//! 1. **Is it up, what is it, and how old is this?** — the roll-up glyph, `Service status`,
//!    the version, and the freshness right-flushed. Four facts, and no detail.
//! 2. **Which part?** — four equal columns, glyph and label, in a fixed order.
//! 3. **Is work moving?** — the queue's three counts.
//!
//! # The title takes colour only when it is bad news
//!
//! §4 is explicit that on a status line *only the glyph carries colour*, and that stands: a
//! healthy `Service status` is bold at the normal rung and nothing more. But §4's other half —
//! the must-see rule — says the critical datum *"is pushed forward with bold + the state
//! colour"*, and a system that is degraded is exactly that datum. So the hue arrives only when
//! there is something to see, which is what keeps a healthy screen quiet (Chris agreed,
//! 20260906).
//!
//! # `in progress` is `secondary`, and that is not a free choice
//!
//! Chris asked for blue. Blue is what `info` is in every bundled theme, and §10 reserves `info`
//! for the selector **absolutely** — a blue count would be the one thing §3 forbids: a
//! selector-hued mark that is not a selection. [`crate::tokens::secondary`] is the free role
//! that reads blue-ish without being that one. Flagged to Chris rather than decided quietly.
//!
//! # Collapse is on or off, never gradual
//!
//! A block that shed one row at a time would put the queue counts on some widths and not
//! others, and a user cannot learn a layout that has four shapes. So there are two: the whole
//! block, or its first row. See [`Collapse`] for the two triggers.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens::{self, Health};
use crate::widgets::chrome::{inset, Freshness, Rule, MARGIN};

/// The four parts of the system this block names, **in this order**.
///
/// UI constants for this scaffold (Chris chose these words, 20260906) — they are not N8 names
/// and nothing on the wire spells them. One array rather than four literals so the order is a
/// property of the module instead of a convention four render calls agree to keep, and so the
/// widest of them — which is what the column arithmetic below is sized against — is computed
/// from the words themselves rather than remembered.
pub const ENTRY_LABELS: [&str; 4] = ["daemon", "vector db", "graph db", "search db"];

/// The widest label, in columns. ASCII by construction, so bytes and columns agree; a label
/// with a non-ASCII character would need `Span::width` and a runtime constant.
const fn widest_label() -> u16 {
    let mut i = 0;
    let mut widest = 0;
    while i < ENTRY_LABELS.len() {
        let len = ENTRY_LABELS[i].len();
        if len > widest {
            widest = len;
        }
        i += 1;
    }
    widest as u16
}

/// The narrowest column that can still hold `glyph + space + the widest label`.
///
/// Below it the four columns cannot be equal AND hold their contents, which is the width
/// trigger in [`Collapse`]. Derived rather than chosen: a fifth entry, or a longer word,
/// moves this number without anyone editing it.
pub const MIN_COLUMN: u16 = widest_label() + 2;

/// Rows the whole block occupies: three of content plus the rule that closes it.
pub const ROWS_FULL: u16 = 4;

/// Rows the collapsed block occupies: the roll-up line plus the same closing rule.
pub const ROWS_COLLAPSED: u16 = 2;

/// The floor a view keeps for its own content before this block gives way.
///
/// **A tolerance, and Chris has not set it.** Sixteen is a placeholder, and this is the whole
/// of its reasoning: a tab body worth showing is a heading, a dozen rows and a foot, and below
/// that the constant furniture is eating the screen it is supposed to be introducing. The
/// arithmetic it produces is the part to argue with — with the two rows above the block, the
/// full shape survives a 24-row terminal (`24 − 2 ≥ 4 + 16`) and collapses on a 20-row one.
///
/// It is a *parameter* of [`Collapse::decide`] rather than a constant read inside it precisely
/// because the number is the view's to state: a screen whose content is a one-line summary can
/// afford the full block where a screen holding a table cannot.
pub const MIN_CONTENT_ROWS: u16 = 16;

/// The block's rows, by index. Load-bearing: [`StatusBlock::render`] places each row from its
/// own constant rather than from the order of a layout array, so the number a test asserts a
/// row's position with is the number the renderer used to put it there.
const STATUS_ROW: u16 = 0;
const ENTRIES_ROW: u16 = 1;
const QUEUE_ROW: u16 = 2;

/// Words the queue row is spelled with. Named so the tests can reach a count by its label
/// rather than by its digits — see `tests::style_after`.
const PENDING: &str = "pending ";
const IN_PROGRESS: &str = "in progress ";
const FAILED: &str = "failed ";

/// Columns between one group on a row and the next. Wide enough that `queue` and `pending 123`
/// read as two things; the same gap [`crate::tokens::key_hints`] puts between two hints.
const GAP: &str = "   ";

/// The three counts the queue reports, and how the queue itself is.
///
/// The health is carried beside the counts rather than derived from them: "failed 2" is not
/// automatically degraded — whether two failures matter is the daemon's judgement, not this
/// pane's, and inventing a threshold here would be this crate deciding a tolerance.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Queue {
    pub pending: u64,
    pub in_progress: u64,
    pub failed: u64,
    pub health: Health,
}

/// **INTERIM** — the overall health this block shows, pending `CR-057` §12.
///
/// Chris, 20260906: *"the daemon down is not equal to the other services being down, the
/// overall service is however obviously degraded"*. So an unreachable daemon rolls up to
/// [`Health::Degraded`], never to [`Health::Offline`] — and neither does anything else: this
/// function has no path to `Offline` at all, which is the whole of what makes it interim.
///
/// It deliberately **disagrees** with [`crate::health::SystemHealth::rollup`], which follows §7
/// and reports the master's own word. Two rules, two surfaces, both written down: §7's answers
/// *"what is the daemon saying"* on the Service hub, this one answers *"can I trust this
/// screen"* on every other tab.
///
/// `entries` is the non-daemon list; the daemon is the first parameter because it is the one
/// the rule treats differently.
pub fn rollup(daemon: Health, entries: &[Health]) -> Health {
    if daemon != Health::Healthy || entries.iter().any(|h| *h != Health::Healthy) {
        Health::Degraded
    } else {
        Health::Healthy
    }
}

/// Whether four equal columns can each hold `glyph + space + the widest label` at this width.
///
/// The SSOT for the width trigger: [`Collapse::decide`] and [`StatusBlock::render`] both ask
/// here, so the height a view reserves and the shape the block draws cannot disagree.
pub fn columns_align(width: u16) -> bool {
    let content = width.saturating_sub(MARGIN * 2);
    content / ENTRY_LABELS.len() as u16 >= MIN_COLUMN
}

/// The block's two shapes. On or off — see the module docs for why there is no third.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Collapse {
    Full,
    Collapsed,
}

impl Collapse {
    /// Which shape fits in `area`, leaving `content_rows_needed` for whatever sits beneath.
    ///
    /// `area` is the region the block AND the content below it share — a view's remaining
    /// space, not the block's own slice, which does not exist until this has answered.
    pub fn decide(area: Rect, content_rows_needed: u16) -> Collapse {
        if !columns_align(area.width) || area.height < ROWS_FULL + content_rows_needed {
            Collapse::Collapsed
        } else {
            Collapse::Full
        }
    }

    const fn rows(self) -> u16 {
        match self {
            Collapse::Full => ROWS_FULL,
            Collapse::Collapsed => ROWS_COLLAPSED,
        }
    }
}

/// One row of a block, by its index. Never a `Layout`: the row constants above are the
/// positions, and a layout array would be a second statement of the same order.
fn line(area: Rect, n: u16) -> Rect {
    Rect {
        y: area.y + n,
        height: 1,
        ..area
    }
}

/// The constant service status, three rows and a rule.
pub struct StatusBlock {
    overall: Health,
    version: String,
    freshness: Freshness,
    /// One health per [`ENTRY_LABELS`] slot, positionally. An array rather than a list of
    /// pairs because the labels are fixed: a caller cannot reorder them, mislabel one, or hand
    /// over three.
    entries: [Health; ENTRY_LABELS.len()],
    queue: Queue,
}

impl StatusBlock {
    pub fn new(
        overall: Health,
        version: impl Into<String>,
        freshness: Freshness,
        entries: [Health; ENTRY_LABELS.len()],
        queue: Queue,
    ) -> Self {
        Self {
            overall,
            version: version.into(),
            freshness,
            entries,
            queue,
        }
    }

    pub fn overall(mut self, overall: Health) -> Self {
        self.overall = overall;
        self
    }

    pub fn queue(mut self, queue: Queue) -> Self {
        self.queue = queue;
        self
    }

    /// How many rows to reserve for this block, given the space it shares with the content.
    ///
    /// A view asks this before it lays anything out. The block then decides the same thing
    /// again from the slice it is handed, and the two agree because both go through
    /// [`columns_align`] and [`ROWS_FULL`] rather than through two copies of the arithmetic.
    pub fn rows_for(area: Rect, content_rows_needed: u16) -> u16 {
        Collapse::decide(area, content_rows_needed).rows()
    }

    /// Row 1 — the roll-up, the version, and how old all of it is.
    fn status_row(&self, area: Rect, buf: &mut Buffer) {
        let mut left = vec![
            Span::styled(self.overall.glyph(), Style::default().fg(self.overall.color())),
            Span::raw(" "),
            // §4: quiet while healthy, pushed forward with bold + the state colour when not.
            Span::styled(
                "Service status",
                Style::default()
                    .fg(match self.overall {
                        Health::Healthy => tokens::normal(),
                        other => other.color(),
                    })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(GAP, tokens::muted_style()),
            Span::styled(self.version.clone(), tokens::muted_style()),
        ];

        let right = self.freshness.span();
        let used: usize = left
            .iter()
            .chain(std::iter::once(&right))
            .map(|s| s.content.chars().count())
            .sum();
        // The freshness is the half that changes, so a row that cannot hold both loses the gap
        // rather than the age — the same trade `title_bar` makes.
        left.push(Span::raw(
            " ".repeat((area.width as usize).saturating_sub(used).max(1)),
        ));
        left.push(right);
        Paragraph::new(Line::from(left)).render(area, buf);
    }

    /// Row 2 — four equal columns, glyph coloured and label muted.
    fn entries_row(&self, area: Rect, buf: &mut Buffer) {
        let column = area.width / ENTRY_LABELS.len() as u16;
        for (i, (label, health)) in ENTRY_LABELS.iter().zip(self.entries).enumerate() {
            let cell = Rect {
                x: area.x + column * i as u16,
                width: column,
                ..area
            };
            Paragraph::new(Line::from(vec![
                Span::styled(health.glyph(), Style::default().fg(health.color())),
                Span::styled(format!(" {label}"), tokens::muted_style()),
            ]))
            .render(cell, buf);
        }
    }

    /// Row 3 — is work moving, waiting, or lost.
    fn queue_row(&self, area: Rect, buf: &mut Buffer) {
        let count = |value: u64, hue: fn() -> ratatui::style::Color| {
            // A count of nothing is not news, so it recedes to its own label's rung.
            let style = if value == 0 {
                tokens::muted_style()
            } else {
                Style::default().fg(hue())
            };
            Span::styled(value.to_string(), style)
        };

        Paragraph::new(Line::from(vec![
            Span::styled(
                self.queue.health.glyph(),
                Style::default().fg(self.queue.health.color()),
            ),
            Span::styled(format!(" queue{GAP}{PENDING}"), tokens::muted_style()),
            count(self.queue.pending, tokens::degraded),
            Span::styled(format!("{GAP}{IN_PROGRESS}"), tokens::muted_style()),
            count(self.queue.in_progress, tokens::secondary),
            Span::styled(format!("{GAP}{FAILED}"), tokens::muted_style()),
            count(self.queue.failed, tokens::offline),
        ]))
        .render(area, buf);
    }
}

impl Widget for StatusBlock {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }
        // Decided from the slice actually handed over, through the same two facts
        // `rows_for` consults — so a view that reserved four rows gets four rows of block.
        let collapsed = area.height < ROWS_FULL || !columns_align(area.width);

        self.status_row(inset(line(area, STATUS_ROW)), buf);
        if !collapsed {
            self.entries_row(inset(line(area, ENTRIES_ROW)), buf);
            self.queue_row(inset(line(area, QUEUE_ROW)), buf);
        }
        // Edge to edge: §6's rules underline the whole screen, so this one is not inset. It is
        // the LAST row of whichever shape was drawn, which is what makes the block close at the
        // same place in both — the seam a screen is read down does not move with the content.
        let rows = if collapsed { ROWS_COLLAPSED } else { ROWS_FULL };
        Rule::internal().render(line(area, rows - 1), buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient;

#[cfg(test)]
mod tests;

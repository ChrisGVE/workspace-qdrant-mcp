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
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
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

/// What the queue row says, on the SAME four columns row 2 uses (Chris, 20260906, after
/// seeing the block in the pantry: the two rows were on unrelated grids and it read as two
/// unrelated rows). Column 0 is the queue itself; these three take columns 1–3.
pub const QUEUE_LABELS: [&str; 3] = ["pending", "in progress", "failed"];

/// Column 0's word — the one the glyph belongs to, because the glyph reports on the queue and
/// not on any one of its counts.
pub const QUEUE_HEAD: &str = "queue";

/// Cells a count is right-aligned into: enough for `9 999 999`, grouped.
///
/// A fixed field rather than a measured one so that every count on the row ends on the same
/// relative cell whatever its magnitude — a right edge that moved with the number would undo
/// the alignment this whole row exists for.
pub const COUNT_WIDTH: u16 = 9;

/// The widest label on either row, in columns. ASCII by construction, so bytes and columns
/// agree; a label with a non-ASCII character would need `Span::width` and a runtime constant.
///
/// Both rows, not just row 2: `in progress` is eleven columns against `vector db`'s nine, and
/// sizing the grid to the entries alone is what would let the queue row overflow it.
const fn widest_label() -> u16 {
    let mut widest = QUEUE_HEAD.len();
    let mut i = 0;
    while i < ENTRY_LABELS.len() {
        if ENTRY_LABELS[i].len() > widest {
            widest = ENTRY_LABELS[i].len();
        }
        i += 1;
    }
    let mut j = 0;
    while j < QUEUE_LABELS.len() {
        if QUEUE_LABELS[j].len() > widest {
            widest = QUEUE_LABELS[j].len();
        }
        j += 1;
    }
    widest as u16
}

/// The narrowest column at which the grid still works — **for both rows**.
///
/// Two constraints, and the second dominates:
///
/// 1. A column holds `glyph + space + the widest label`: `widest_label() + 2`.
/// 2. A count is right-aligned into the slack LEFT of its own column, occupying
///    `[origin − COUNT_WIDTH + 1, origin]`. The previous column's label ends at
///    `origin_prev + widest_label() + 1`, so one clear blank cell between them needs
///    `column ≥ widest_label() + COUNT_WIDTH + 2`.
///
/// The second is what this constant is *for*, and it **raised** the number from 11 to 23 when
/// the queue row moved onto the entry grid (Chris sanctioned the raise: *"if it does not,
/// raise `MIN_COLUMN` to guarantee it and say so"*). The visible cost is the collapse
/// threshold: a screen narrower than `2 · MARGIN + 4 · MIN_COLUMN` takes the short block,
/// where before it took the full one and drew a row nobody could line up.
///
/// # It is held at `+ 3` where `+ 2` would now do, deliberately
///
/// Moving the count field one cell right (Chris, 20260906) bought a cell of clearance back, so
/// the requirement above is one lower than this constant. Lowering it would move the collapse
/// threshold from 96 columns to 92 — a change to what a real terminal shows, decided by an
/// unrelated alignment tweak. The extra cell stays until the threshold is looked at on its own
/// terms, and the clearance it buys is two cells rather than one.
pub const MIN_COLUMN: u16 = widest_label() + COUNT_WIDTH + 3;

/// The widest a column is allowed to get.
///
/// Chris, 20260906: on a very wide terminal the four columns spread until they are hard to
/// read — four glyphs separated by thirty blank cells stop being a row and become four
/// unrelated things. Past this the grid packs LEFT from the margin and the remaining width
/// stays empty, which is the honest answer: the information did not get bigger, so neither
/// should the space it occupies.
///
/// **A tolerance, and Chris has not set it.** `MIN_COLUMN + 8` is the placeholder: eight cells
/// of slack above the narrowest legal column is enough to keep a full-width count comfortably
/// clear of the label to its left without the row starting to drift apart. The number to argue
/// with is what it produces — the storyboard's 125 columns sit just under it and are unaffected,
/// and a 200-column terminal is capped rather than spread.
pub const MAX_COLUMN: u16 = MIN_COLUMN + 8;

/// A cap below the floor is not a tight grid, it is a `clamp` that panics — and it would panic
/// on the first frame drawn, not in a test. Held at COMPILE time because both operands are
/// constants: a runtime guard for a fact the compiler already knows is a test that can only
/// ever pass.
const _: () = assert!(
    MIN_COLUMN <= MAX_COLUMN,
    "MAX_COLUMN is below MIN_COLUMN: the grid has no legal width"
);

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
    even_share(width) >= MIN_COLUMN
}

/// What the width would give each column if nothing constrained it.
///
/// Separate from [`column_width`] on purpose, and the separation is load-bearing:
/// [`columns_align`] must ask the UNCLAMPED question. Clamping raises a too-narrow column up
/// to [`MIN_COLUMN`], so an alignment test asking the clamped value would always be satisfied
/// and the width trigger in [`Collapse`] would never fire.
fn even_share(area_width: u16) -> u16 {
    area_width.saturating_sub(MARGIN * 2) / ENTRY_LABELS.len() as u16
}

/// The width of one of the four columns at this screen width — the even share, capped.
///
/// One function so both rows ask the same question. It is also the only place the grid's
/// arithmetic exists, which is what lets a test state a column position as an expression
/// rather than measure it off the other row.
pub fn column_width(area_width: u16) -> u16 {
    even_share(area_width).clamp(MIN_COLUMN, MAX_COLUMN)
}

/// A count, grouped in threes with a **plain** space: `1 240`, `9 999 999`.
///
/// Chris's standing rule for an isolated number. Plain rather than thin or narrow because
/// `U+2009`/`U+202F` are not width-1 in every terminal, and a separator whose width depends on
/// the emulator changes the cell a right-aligned number ends on — which is the one property
/// this row is built around.
pub fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(digit);
    }
    out
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
    fn entries_row(&self, area: Rect, buf: &mut Buffer, column: u16) {
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

    /// Row 3 — is work moving, waiting, or lost, on row 2's own grid.
    ///
    /// Each label starts where the label above it starts, and each count is right-aligned into
    /// the slack to its left, its units digit landing **on** its column's glyph position. So
    /// the numbers grow leftward, away from the words they belong to; no label ever moves
    /// because a count gained a digit; and a count's last cell sits directly under the disc
    /// above it, which is the column the eye is already following.
    ///
    /// Chris, 20260906: *"given the symbol, right-align the number on the left"* — the
    /// symbol's slot is where the number ends. The first cut ended one cell earlier, which
    /// left two blanks before the label and made each count read as belonging to the column on
    /// its left rather than to the word beside it.
    fn queue_row(&self, area: Rect, buf: &mut Buffer, column: u16) {
        Paragraph::new(Line::from(vec![
            Span::styled(
                self.queue.health.glyph(),
                Style::default().fg(self.queue.health.color()),
            ),
            Span::styled(format!(" {QUEUE_HEAD}"), tokens::muted_style()),
        ]))
        .render(
            Rect {
                width: column.min(area.width),
                ..area
            },
            buf,
        );

        let counts = [
            (self.queue.pending, tokens::degraded as fn() -> Color),
            (self.queue.in_progress, tokens::secondary as fn() -> Color),
            (self.queue.failed, tokens::offline as fn() -> Color),
        ];

        for (i, ((value, hue), label)) in counts.iter().zip(QUEUE_LABELS).enumerate() {
            let origin = column * (i as u16 + 1);
            // A count of nothing is not news, so it recedes to its own label's rung.
            let style = if *value == 0 {
                tokens::muted_style()
            } else {
                Style::default().fg(hue())
            };
            Paragraph::new(Line::from(Span::styled(grouped(*value), style)))
                .alignment(Alignment::Right)
                .render(
                    Rect {
                        x: area.x + (origin + 1).saturating_sub(COUNT_WIDTH),
                        width: COUNT_WIDTH.min(origin + 1),
                        ..area
                    },
                    buf,
                );

            let label_x = origin + 2;
            if label_x < area.width {
                Paragraph::new(Line::from(Span::styled(label, tokens::muted_style()))).render(
                    Rect {
                        x: area.x + label_x,
                        width: area.width - label_x,
                        ..area
                    },
                    buf,
                );
            }
        }
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
            // One column width, computed once and handed to both rows: two rows that each
            // worked it out for themselves is exactly how they came to be on different grids.
            let column = column_width(area.width);
            self.entries_row(inset(line(area, ENTRIES_ROW)), buf, column);
            self.queue_row(inset(line(area, QUEUE_ROW)), buf, column);
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

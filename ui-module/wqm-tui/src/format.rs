//! How a figure is written down, anywhere on this surface.
//!
//! One module because the rules are the surface's, not any one pane's. [`grouped`] arrived in
//! [`crate::panes::status_block`] when the queue row was the only thing rendering a count, with
//! a note saying it belonged elsewhere the moment a second consumer appeared. The Dashboard's
//! cells are that second consumer (20260907), so it moved rather than being copied — two
//! number styles on one surface is the failure this exists to prevent, and it is a failure
//! nobody notices, because each screen looks internally consistent.
//!
//! [`count_span`] is here for the same reason: "muted at zero, otherwise the stated hue" is a
//! rule about what a figure MEANS, and the queue row and the Dashboard's queue columns must
//! not each have their own opinion of it.

use ratatui::style::{Color, Style};
use ratatui::text::Span;

use crate::tokens;

/// What separates one group of three digits from the next — see [`grouped`].
///
/// Named rather than inlined so the one character the whole rule is about is greppable, and so
/// a test can assert against the constant instead of retyping a literal that would then agree
/// with itself.
pub const GROUP_SEPARATOR: char = '\'';

/// A count, grouped in threes with an **ASCII apostrophe**: `1'240`, `9'999'999`.
///
/// Chris's standing rule for an isolated number anywhere in this TUI, set 20260907. Swiss
/// style, and `U+0027` specifically — **not** `U+2019` (the typographic right single quote,
/// which is what a word processor substitutes) and no longer a space.
///
/// # Three separators have been considered and two are wrong
///
/// A **thin or narrow space** (`U+2009`, `U+202F`) is what typography asks for and is unusable
/// here: neither is width-1 in every terminal, and a separator whose width depends on the
/// emulator changes the cell a right-aligned number ends on — which is the one property the
/// queue row is built around. A **plain space** was the first answer and survives that test,
/// but it makes one figure look like two: `1 240 pending` reads as a count of 1 beside a count
/// of 240 until the eye resolves it. The apostrophe is width-1 like the space and *binds*
/// rather than separates, so the number reads as one thing.
///
/// `U+2019` would look right and behave badly: it is width-1 in most terminals and not all,
/// and it is what an autocorrecting editor produces from `'`, so a figure copied off the screen
/// and back into a config would not round-trip. The guard names both rejects.
///
/// # This is the TUI's rule, not this pane's
///
/// It lives here because the queue row is the only thing that currently renders a count. **Any
/// future figure — a graph's node count, a tag tally, a byte size — calls this rather than
/// formatting its own**, or the surface acquires two number styles and nobody notices which
/// screen has which. When a second consumer arrives this belongs in a module of its own.
pub fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(GROUP_SEPARATOR);
        }
        out.push(digit);
    }
    out
}

/// A figure as it should be drawn: **muted when it is zero**, and the stated hue otherwise.
///
/// A count of nothing is not news, so it recedes to the same rung as the label beside it. The
/// hue is passed in rather than decided here because *which* hue depends on what the figure
/// counts — waiting work is `warning`, work in flight is [`tokens::in_flight`], failures are
/// `error` — and this module has no business knowing that. It knows only that zero is quiet.
pub fn count_span(value: u64, hue: fn() -> Color) -> Span<'static> {
    let style = if value == 0 {
        tokens::muted_style()
    } else {
        Style::default().fg(hue())
    };
    Span::styled(grouped(value), style)
}

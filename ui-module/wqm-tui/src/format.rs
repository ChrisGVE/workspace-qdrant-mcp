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

/// The unit field's width: two cells, so `B` occupies as much as `KB` and the space before it
/// lands in the same column on every row.
///
/// Chris, 2026-09-07: sizes are *"right-aligned … aligned on the space"*. Right-alignment alone
/// cannot do that when the unit is sometimes one character and sometimes two — `169 B` and
/// `31 KB` right-aligned put their spaces one column apart — so the unit is padded here, where
/// the figure is written, rather than at each column that draws one.
const UNIT_WIDTH: usize = 2;

/// The steps, largest first, with the divisor each one measures in.
const UNITS: [(u64, &str); 4] = [
    (1024 * 1024 * 1024, "GB"),
    (1024 * 1024, "MB"),
    (1024, "KB"),
    (1, "B"),
];

/// A byte count as this surface writes it: **an integer and a unit**, `31 KB`, `4 MB`, `169 B`.
///
/// Chris, 2026-09-07: *"integer + unit"*, no decimal — a queue row is read down a column for its
/// order of magnitude, and `30.8 KB` spends two cells saying what `31 KB` says. Rounded rather
/// than truncated: `1023 B` is nearer a kilobyte than nothing, and a column that always rounded
/// down would report `1.9 MB` as `1 MB`.
///
/// **`None` is not zero.** A queue item whose size is not yet known draws nothing at all, which
/// is what v0.1 does and what the captured buffer holds; a file that really is empty draws
/// `0 B`. The two are different facts and the column shows the difference.
///
/// The unit is padded to [`UNIT_WIDTH`], so every figure this returns is the same width and a
/// right-aligned column lines up the space between the number and the unit.
pub fn size(bytes: Option<u64>) -> String {
    let Some(bytes) = bytes else {
        return String::new();
    };
    let (divisor, unit) = UNITS
        .iter()
        .copied()
        // The largest unit this figure reaches at least one whole of — and `B` catches
        // everything below a kilobyte, itself included.
        .find(|(divisor, _)| bytes >= *divisor)
        .unwrap_or((1, "B"));
    let scaled = (bytes as f64 / divisor as f64).round() as u64;
    format!("{scaled} {unit:<UNIT_WIDTH$}")
}

/// An age as this surface writes it: **the coarsest unit that still says something**, and no
/// `ago` — `12s`, `1m`, `3h`, `2d`.
///
/// Chris, 2026-09-07: the tables print `1m` / `3h` / `2d`, *"no `ago`"*, aligned on the unit.
/// The unit is one character and the column is right-aligned, so it lands in the last cell
/// whatever the number's width — nothing needs padding here, unlike [`size`].
///
/// The word `ago` belongs to a SENTENCE rather than to a figure: the title bar's freshness line
/// reads *"updated 4s ago"* and adds the word itself
/// ([`crate::widgets::chrome::freshness::format_age`]), which is why that line and this column
/// can share one producer without sharing a phrasing.
pub fn age(seconds: u64) -> String {
    match seconds {
        0..=59 => format!("{seconds}s"),
        60..=3_599 => format!("{}m", seconds / 60),
        3_600..=86_399 => format!("{}h", seconds / 3_600),
        _ => format!("{}d", seconds / 86_400),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Chris, 2026-09-07: an integer and a unit, and nothing between them but one space.
    #[test]
    fn a_size_is_an_integer_and_a_unit() {
        assert_eq!(size(Some(169)).trim_end(), "169 B");
        assert_eq!(size(Some(31_539)).trim_end(), "31 KB");
        assert_eq!(size(Some(4_194_304)).trim_end(), "4 MB");
        assert_eq!(size(Some(2_147_483_648)).trim_end(), "2 GB");
        assert!(
            !size(Some(31_539)).contains('.'),
            "a decimal spends two cells saying what the integer says"
        );
    }

    /// Rounded, not truncated — a column that always rounded down would call 1.9 MB one.
    #[test]
    fn a_size_rounds_to_the_nearest_whole_unit() {
        assert_eq!(size(Some(1_992_294)).trim_end(), "2 MB");
        assert_eq!(size(Some(1023)).trim_end(), "1023 B");
    }

    /// An unknown size and a size of zero are different facts, and the column shows both.
    #[test]
    fn no_size_draws_nothing_and_an_empty_file_draws_zero() {
        assert_eq!(size(None), "");
        assert_eq!(size(Some(0)).trim_end(), "0 B");
    }

    /// Every figure is the same width, so a right-aligned column puts every space in one place.
    ///
    /// The property, not three examples: `B` is one character where `KB` is two, and the whole
    /// point of padding here is that the difference never reaches the column.
    #[test]
    fn every_size_is_the_same_width_after_its_number() {
        for bytes in [0u64, 1, 999, 1024, 999_999, 5_000_000, 9_000_000_000] {
            let drawn = size(Some(bytes));
            let (number, unit) = drawn.split_once(' ').expect("one space, always");
            assert_eq!(
                unit.chars().count(),
                UNIT_WIDTH,
                "{drawn:?} pads its unit to {UNIT_WIDTH}"
            );
            assert!(
                number.chars().all(|c| c.is_ascii_digit()),
                "{drawn:?} writes a bare integer before the space"
            );
        }
    }

    /// The coarsest unit that still says something, and no `ago` (Chris, 2026-09-07).
    #[test]
    fn an_age_is_a_bare_figure_in_its_coarsest_unit() {
        assert_eq!(age(0), "0s");
        assert_eq!(age(59), "59s");
        assert_eq!(age(60), "1m");
        assert_eq!(age(3_599), "59m");
        assert_eq!(age(3_600), "1h");
        assert_eq!(age(86_399), "23h");
        assert_eq!(age(86_400), "1d");
        assert_eq!(age(172_800), "2d");
        for seconds in [0, 59, 60, 3_600, 86_400, 1_000_000] {
            assert!(!age(seconds).contains("ago"), "{}", age(seconds));
        }
    }

    /// The unit is the last character, so a right-aligned column aligns on it with no padding —
    /// which is why [`age`] pads nothing and [`size`] must.
    #[test]
    fn an_age_ends_in_its_unit() {
        for seconds in [0, 90, 7_200, 300_000] {
            let drawn = age(seconds);
            let last = drawn.chars().last().expect("a unit");
            assert!(last.is_ascii_alphabetic(), "{drawn:?}");
            assert!(
                drawn[..drawn.len() - 1].chars().all(|c| c.is_ascii_digit()),
                "{drawn:?} is digits then one unit letter"
            );
        }
    }
}

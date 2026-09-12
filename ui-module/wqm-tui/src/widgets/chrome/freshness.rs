//! How old a screen's readings are — the one datum every constant-top element carries.
//!
//! It lived inside [`crate::widgets::chrome::title_bar`] while the title bar was the only
//! thing that showed it. [`crate::panes::status_block`] shows the same value in the same
//! treatment on line 1 of every screen, and a second copy of "muted while fresh, the degraded
//! hue once it is not" is a second copy that can disagree with the first — so the type and its
//! formatting moved here and both callers read them from one place.
//!
//! `title_bar` re-exports both names, so nothing that already spelled
//! `chrome::Freshness` or `title_bar::format_age` had to change.

use std::time::Duration;

use ratatui::{
    style::Style,
    text::Span,
};

use crate::tokens;

/// How old the screen's readings are, and how old they are allowed to get.
///
/// §4: *"Freshness/staleness is right-aligned, muted; past its SLA it turns `[yellow]stale
/// …`"*. Both halves of that comparison are carried, so [`Freshness::is_stale`] is a
/// measurement rather than a claim — a frame reading `updated 18m ago` in muted grey under a
/// one-minute SLA is not constructible.
///
/// **The SLA itself is not this crate's to set** — §7 leaves the freshness SLA open (OQ-6),
/// which is exactly why it is a parameter here instead of a constant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Freshness {
    age: Duration,
    sla: Duration,
}

impl Freshness {
    pub const fn new(age: Duration, sla: Duration) -> Self {
        Self { age, sla }
    }

    pub fn is_stale(&self) -> bool {
        self.age > self.sla
    }

    /// The right-aligned span: muted while fresh, and the degraded hue once it is not.
    ///
    /// `pub(crate)` rather than private since the move: the status block right-flushes exactly
    /// this span on its own first row, and rebuilding it there is how the two treatments drift.
    pub(crate) fn span(&self) -> Span<'static> {
        if self.is_stale() {
            Span::styled(
                format!("stale — {} ago", format_age(self.age)),
                Style::default().fg(tokens::Health::Degraded.color()),
            )
        } else {
            Span::styled(
                format!("updated {} ago", format_age(self.age)),
                tokens::muted_style(),
            )
        }
    }
}

/// An age in the coarsest unit that still says something: `4s`, `18m`, `2h`, `3d`.
///
/// Coarse on purpose. The number is read peripherally to answer *"is this recent?"*, and a
/// second of precision on an eighteen-minute age answers a question nobody asked.
///
/// A [`Duration`]-shaped door onto [`crate::format::age`], which is the surface's one age
/// producer (20260912). The freshness line adds `ago` because it is writing a sentence; a table
/// column prints the bare figure. One rule for how long is written, two phrasings around it.
pub fn format_age(age: Duration) -> String {
    crate::format::age(age.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_age_is_named_in_the_coarsest_unit_that_still_says_something() {
        assert_eq!(format_age(Duration::from_secs(0)), "0s");
        assert_eq!(format_age(Duration::from_secs(59)), "59s");
        assert_eq!(format_age(Duration::from_secs(60)), "1m");
        assert_eq!(format_age(Duration::from_secs(3_599)), "59m");
        assert_eq!(format_age(Duration::from_secs(3_600)), "1h");
        assert_eq!(format_age(Duration::from_secs(86_399)), "23h");
        assert_eq!(format_age(Duration::from_secs(86_400)), "1d");
    }
}

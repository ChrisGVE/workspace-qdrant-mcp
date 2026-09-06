//! The title bar — the screen's name, with how old its readings are right-aligned against it.
//!
//! [`Freshness`] and [`format_age`] are re-exported rather than owned: they moved to
//! [`super::freshness`] when [`crate::panes::status_block`] needed the same treatment on line
//! 1 of every screen. Every path that already named them through this module still resolves.

use std::time::Duration;

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Modifier,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens;

pub use super::freshness::{format_age, Freshness};

/// The screen's name on its own line, with the freshness right-aligned against it (§6).
pub struct TitleBar {
    title: String,
    freshness: Option<Freshness>,
}

impl TitleBar {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            freshness: None,
        }
    }

    pub fn freshness(mut self, freshness: Freshness) -> Self {
        self.freshness = Some(freshness);
        self
    }
}

impl Widget for TitleBar {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }

        // Bold at the normal rung — the same treatment a config group header carries, which
        // is this crate's established way of saying "a structural name, not a datum". Strong
        // is reserved for the one value that must be seen, and a title is never that.
        let title = Span::styled(
            self.title.clone(),
            tokens::normal_style().add_modifier(Modifier::BOLD),
        );

        let mut spans = vec![title];
        if let Some(freshness) = self.freshness {
            let right = freshness.span();
            let used = self.title.chars().count() + right.content.chars().count();
            // A title and a freshness that together outrun the line lose the gap, not the
            // freshness: the age is the half that changes, so it is the half worth keeping.
            let gap = (area.width as usize).saturating_sub(used).max(1);
            spans.push(Span::raw(" ".repeat(gap)));
            spans.push(right);
        }

        Paragraph::new(Line::from(spans)).render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::Health;
    use crate::widgets::chrome::test_support::{render, row, style_at, Restore};

    #[test]
    fn staleness_is_the_comparison_and_cannot_be_stated_against_it() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let sla = Duration::from_secs(60);
        let fresh = Freshness::new(Duration::from_secs(4), sla);
        let stale = Freshness::new(Duration::from_secs(61), sla);
        assert!(!fresh.is_stale() && stale.is_stale());

        // The word and the hue both follow the comparison — there is no third input either
        // could have been set from.
        let fresh_line = row(&render(TitleBar::new("Service").freshness(fresh)), 0);
        assert!(fresh_line.contains("updated 4s ago"), "{fresh_line}");
        assert!(!fresh_line.contains("stale"), "{fresh_line}");

        let stale_buf = render(TitleBar::new("Service").freshness(stale));
        let stale_line = row(&stale_buf, 0);
        assert!(stale_line.contains("stale — 1m ago"), "{stale_line}");
        let x = stale_line.chars().position(|c| c == 's').expect("the word") as u16;
        assert_eq!(
            style_at(&stale_buf, x).fg,
            Some(Health::Degraded.color()),
            "past its SLA the freshness turns the degraded hue"
        );
    }

    #[test]
    fn the_freshness_is_flush_with_the_right_edge() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let line = row(
            &render(TitleBar::new("Service").freshness(Freshness::new(
                Duration::from_secs(4),
                Duration::from_secs(60),
            ))),
            0,
        );
        assert!(
            !line.ends_with(' '),
            "right-aligned means the last cell is used: {line:?}"
        );
        assert!(line.starts_with("Service "), "{line:?}");
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "title",
            ty: "String",
            description: "The view's own name — never a datum, so it is bold at the normal rung",
        },
        PropInfo {
            name: "freshness",
            ty: "Option<Freshness>",
            description: "Age and SLA together; the word STALE is the comparison, never a flag",
        },
    ];

    struct Variant {
        name: &'static str,
        description: &'static str,
        build: fn() -> TitleBar,
    }

    impl Ingredient for Variant {
        fn group(&self) -> &str {
            "Title Bar"
        }
        fn name(&self) -> &str {
            self.name
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::chrome::title_bar"
        }
        fn description(&self) -> &str {
            self.description
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let top = Rect { height: 1, ..area };
            (self.build)().render(top, buf);
        }
    }

    const MINUTE: Duration = Duration::from_secs(60);

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Variant {
                name: "Default",
                description: "A screen whose readings have no age to report — the title alone",
                build: || TitleBar::new("Service"),
            }),
            Box::new(Variant {
                name: "Fresh",
                description: "Inside its SLA: muted, right-aligned, and nothing asks to be looked at",
                build: || TitleBar::new("Service").freshness(Freshness::new(Duration::from_secs(4), MINUTE)),
            }),
            Box::new(Variant {
                name: "Stale",
                description: "Past its SLA: the same slot turns the degraded hue — does it beat the title without beating the content?",
                build: || TitleBar::new("Service").freshness(Freshness::new(Duration::from_secs(1_100), MINUTE)),
            }),
            Box::new(Variant {
                name: "Long Title",
                description: "Title and age together outrun the line: the gap collapses to one cell and the age survives",
                build: || {
                    TitleBar::new("Service — collections, daemon, queue and watchers")
                        .freshness(Freshness::new(Duration::from_secs(1_100), MINUTE))
                },
            }),
        ]
    }
}

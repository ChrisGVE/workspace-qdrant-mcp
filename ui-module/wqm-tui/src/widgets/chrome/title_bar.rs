//! The title bar — the screen's name, with how old its readings are right-aligned against it.

use std::time::Duration;

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
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
    fn span(&self) -> Span<'static> {
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
pub fn format_age(age: Duration) -> String {
    let secs = age.as_secs();
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3_599 => format!("{}m", secs / 60),
        3_600..=86_399 => format!("{}h", secs / 3_600),
        _ => format!("{}d", secs / 86_400),
    }
}

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
    fn an_age_is_named_in_the_coarsest_unit_that_still_says_something() {
        assert_eq!(format_age(Duration::from_secs(0)), "0s");
        assert_eq!(format_age(Duration::from_secs(59)), "59s");
        assert_eq!(format_age(Duration::from_secs(60)), "1m");
        assert_eq!(format_age(Duration::from_secs(3_599)), "59m");
        assert_eq!(format_age(Duration::from_secs(3_600)), "1h");
        assert_eq!(format_age(Duration::from_secs(86_399)), "23h");
        assert_eq!(format_age(Duration::from_secs(86_400)), "1d");
    }

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

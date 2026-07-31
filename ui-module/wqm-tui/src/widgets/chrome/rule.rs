//! The horizontal rule — §6's only zone divider, in §2's two weights.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens;

/// What a rule is drawn with. One glyph wide per column, so a rule's width is its cell count.
pub const RULE: &str = "─";

/// Which of §2's two structural greys a rule carries.
///
/// The pair is not decoration. §2 puts the frame rules *lighter* than the internal ones on
/// purpose — the outer pair underlines the screen, the inner ones divide within it — so a
/// screen drawn with one grey loses the difference between its edge and its seams.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Weight {
    /// The top and bottom rules that underline the screen's frame.
    Frame,
    /// A separator between two zones inside the screen.
    Internal,
}

/// A horizontal rule spanning its area's width — §6's only zone divider.
pub struct Rule {
    weight: Weight,
}

impl Rule {
    pub const fn new(weight: Weight) -> Self {
        Self { weight }
    }

    pub const fn frame() -> Self {
        Self::new(Weight::Frame)
    }

    pub const fn internal() -> Self {
        Self::new(Weight::Internal)
    }

    fn colour(&self) -> Color {
        match self.weight {
            Weight::Frame => tokens::rule_frame(),
            Weight::Internal => tokens::rule_internal(),
        }
    }
}

impl Widget for Rule {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }
        Paragraph::new(Line::from(Span::styled(
            RULE.repeat(area.width as usize),
            Style::default().fg(self.colour()),
        )))
        .render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widgets::chrome::test_support::{render, row, style_at, Restore, AREA};

    #[test]
    fn a_rule_spans_its_whole_width_and_the_frame_is_lighter_than_a_seam() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let frame = render(Rule::frame());
        assert_eq!(
            row(&frame, 0),
            RULE.repeat(AREA.width as usize),
            "a rule divides the whole zone or it is not a divider"
        );

        // §2 puts the frame pair lighter than the internal ones. Measured against the
        // background, because "lighter" inverts on a light theme and "further from the
        // terminal's own background" does not (§6.26).
        let seam = render(Rule::internal());
        let bg = tokens::endpoints().background;
        let distance = |c: Color| match c {
            Color::Rgb(r, g, b) => {
                let d = |x: u8, y: u8| (x as f32 - y as f32).powi(2);
                (d(r, bg.r) + d(g, bg.g) + d(b, bg.b)).sqrt()
            }
            other => panic!("expected RGB under Derived + TrueColor, got {other:?}"),
        };
        let frame_fg = style_at(&frame, 0).fg.expect("a rule is coloured");
        let seam_fg = style_at(&seam, 0).fg.expect("a rule is coloured");
        assert_ne!(frame_fg, seam_fg, "the two weights must be distinguishable");
        assert!(
            distance(frame_fg) > distance(seam_fg),
            "the frame rule stands further off the base than a seam: {frame_fg:?} vs {seam_fg:?}"
        );
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use ratatui::layout::{Constraint, Layout};
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[PropInfo {
        name: "weight",
        ty: "Weight",
        description: "Frame (the screen's own edge) or Internal (a seam between two zones)",
    }];

    struct Single(Weight, &'static str, &'static str);

    impl Ingredient for Single {
        fn group(&self) -> &str {
            "Rule"
        }
        fn name(&self) -> &str {
            self.1
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::chrome::rule"
        }
        fn description(&self) -> &str {
            self.2
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let top = Rect { height: 1, ..area };
            Rule::new(self.0).render(top, buf);
        }
    }

    /// Both weights on one frame, because §2's rule is about the *pair*.
    ///
    /// A single rule can only be judged against a remembered one, and a remembered colour is
    /// not a measurement. Two of the same widget is not a composition — it is the smallest
    /// frame in which "the frame is lighter than the seam" is a thing the eye can check.
    struct Pair;

    impl Ingredient for Pair {
        fn group(&self) -> &str {
            "Rule"
        }
        fn name(&self) -> &str {
            "Frame over Seam"
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::chrome::rule"
        }
        fn description(&self) -> &str {
            "Both weights, one above the other — the only frame in which the difference is visible rather than remembered"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let rows = Layout::vertical([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .split(area);
            Rule::frame().render(rows[0], buf);
            Rule::internal().render(rows[2], buf);
            Rule::frame().render(rows[3], buf);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Single(
                Weight::Frame,
                "Frame",
                "The screen's own top and bottom rule — the lighter of the two greys",
            )),
            Box::new(Single(
                Weight::Internal,
                "Internal",
                "A seam between two zones — recessive, so the screen's edge stays the strongest line",
            )),
            Box::new(Pair),
        ]
    }
}

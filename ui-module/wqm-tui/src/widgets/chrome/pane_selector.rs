//! The sub-screen selector — the tab bar's mechanism, one level down.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens;

/// The Service hub's sub-screen selector — §3's *"identical inverse block"*.
///
/// The same mechanism as the tab bar, minus the number: a tab's leading digit is a jump hint
/// and there is no digit to jump to here. Sharing the mechanism is goal 3 — if it is a cyan
/// block, it is what you have selected, on every screen and at every level.
pub struct PaneSelector {
    panes: Vec<String>,
    active: usize,
}

impl PaneSelector {
    pub fn new(panes: Vec<String>, active: usize) -> Self {
        Self { panes, active }
    }
}

impl Widget for PaneSelector {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut spans: Vec<Span> = Vec::new();
        for (i, pane) in self.panes.iter().enumerate() {
            if i > 0 {
                spans.push(Span::raw("  "));
            }
            if i == self.active {
                spans.push(Span::styled(
                    format!(" {pane} "),
                    tokens::inverted(tokens::selector()),
                ));
            } else {
                spans.push(Span::styled(pane.clone(), tokens::muted_style()));
            }
        }
        Paragraph::new(Line::from(spans)).render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widgets::chrome::test_support::{render, row, style_at, Restore, AREA};

    #[test]
    fn exactly_one_pane_is_an_inverse_block() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(PaneSelector::new(vec!["Config".into(), "Logs".into()], 0));
        let selector = tokens::selector();
        let filled: Vec<u16> = (0..AREA.width)
            .filter(|x| style_at(&buf, *x).bg == Some(selector))
            .collect();

        // " Config " — one space each side inside the block, per §3.
        assert_eq!(
            filled.len(),
            8,
            "the inverse block is the selected pane only"
        );
        let line = row(&buf, 0);
        assert!(line.starts_with(" Config   Logs"), "{line:?}");

        // The unselected pane recedes rather than carrying a second block.
        let logs_x = line.find("Logs").expect("both panes are drawn") as u16;
        assert_eq!(style_at(&buf, logs_x).fg, Some(tokens::muted()));
        assert_ne!(style_at(&buf, logs_x).bg, Some(selector));
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "panes",
            ty: "Vec<String>",
            description: "The sub-screens this view owns, in the order they are offered",
        },
        PropInfo {
            name: "active",
            ty: "usize",
            description: "Index of the selected pane — the one inverse block on the line",
        },
    ];

    struct Variant {
        name: &'static str,
        description: &'static str,
        active: usize,
    }

    impl Ingredient for Variant {
        fn group(&self) -> &str {
            "Pane Selector"
        }
        fn name(&self) -> &str {
            self.name
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::chrome::pane_selector"
        }
        fn description(&self) -> &str {
            self.description
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let top = Rect { height: 1, ..area };
            PaneSelector::new(vec!["Config".into(), "Logs".into()], self.active).render(top, buf);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Variant {
                name: "Default",
                description: "§4.1's Config↔Logs toggle, Config selected — the same cyan block a tab carries",
                active: 0,
            }),
            Box::new(Variant {
                name: "Second Selected",
                description: "The block away from the left edge: it has to read as a selection, not as a highlight of the first word",
                active: 1,
            }),
        ]
    }
}

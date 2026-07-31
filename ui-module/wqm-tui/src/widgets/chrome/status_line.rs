//! The merged status-and-help line at the foot of the screen.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::health::Rollup;
use crate::tokens;
use crate::widgets::config_table::EditMode;

/// The merged status-and-help line at the foot of the screen (§6).
///
/// Left: the edit-mode indicator when there is one, then §7's single rollup dot. Right: the
/// keys available for what is selected. §4 caps the whole line — *"never more vibrant than
/// the content"* — so only the health glyph carries a hue, and §3 keeps the mode indicator
/// on weight alone because cyan belongs to the selector.
pub struct StatusLine {
    rollup: Rollup,
    mode: Option<EditMode>,
    hints: Vec<(String, String)>,
}

impl StatusLine {
    pub fn new(rollup: Rollup) -> Self {
        Self {
            rollup,
            mode: None,
            hints: Vec::new(),
        }
    }

    /// The vim mode an edit-in-place is in, if one is open. Taken from the table that owns
    /// the edit rather than restated, so the caret and the indicator cannot disagree.
    pub fn mode(mut self, mode: Option<EditMode>) -> Self {
        self.mode = mode;
        self
    }

    pub fn hint(mut self, key: impl Into<String>, label: impl Into<String>) -> Self {
        self.hints.push((key.into(), label.into()));
        self
    }

    fn left(&self) -> Vec<Span<'static>> {
        let mut spans: Vec<Span<'static>> = Vec::new();
        if let Some(mode) = self.mode {
            spans.push(mode.indicator_span());
            spans.push(Span::raw("  "));
        }
        spans.push(Span::styled(
            self.rollup.health.glyph(),
            Style::default().fg(self.rollup.health.color()),
        ));
        spans.push(Span::styled(
            format!(" {}", self.rollup.label),
            tokens::muted_style(),
        ));
        spans
    }
}

impl Widget for StatusLine {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }

        let left = self.left();
        let left_width: usize = left.iter().map(|s| s.content.chars().count()).sum();
        let hints_width = tokens::key_hints_width(&self.hints);

        let mut spans = left;
        // One clear cell between the two halves is the minimum that still reads as two
        // halves. Below that the hints go entirely: the status is what the line is for, and
        // half a hint row is noise rather than help.
        if hints_width > 0 && left_width + 1 + hints_width <= area.width as usize {
            let gap = area.width as usize - left_width - hints_width;
            spans.push(Span::raw(" ".repeat(gap)));
            spans.extend(tokens::key_hints(&self.hints));
        }

        Paragraph::new(Line::from(spans)).render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::Health;
    use crate::widgets::chrome::test_support::{render, row, style_at, Restore, AREA};
    use ratatui::style::Modifier;

    #[test]
    fn the_status_line_carries_colour_on_the_glyph_and_nowhere_else() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(
            StatusLine::new(Rollup {
                health: Health::Degraded,
                label: "1 degraded".into(),
            })
            .hint("↵", "edit"),
        );

        assert_eq!(
            style_at(&buf, 0).fg,
            Some(Health::Degraded.color()),
            "the glyph carries the state"
        );
        // Every other painted cell is a neutral. §4: the status line is never more vibrant
        // than the content.
        let hues = [
            Health::Healthy.color(),
            Health::Degraded.color(),
            Health::Offline.color(),
            tokens::selector(),
        ];
        for x in 1..AREA.width {
            let fg = style_at(&buf, x).fg;
            assert!(
                fg.is_none_or(|c| !hues.contains(&c)),
                "column {x} carries a reserved hue"
            );
        }
    }

    #[test]
    fn the_hints_are_flush_right_and_the_rollup_outlives_them() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let line = StatusLine::new(Rollup {
            health: Health::Healthy,
            label: "healthy".into(),
        })
        .hint("j/k", "move")
        .hint("↵", "edit");

        let wide = row(&render(line), 0);
        assert!(wide.starts_with("● healthy"), "{wide:?}");
        assert!(
            wide.ends_with("↵ edit"),
            "the hints sit against the right edge: {wide:?}"
        );

        // Narrow enough that the two halves would overlap. The hints go; the status stays.
        let mut narrow_buf = Buffer::empty(Rect {
            x: 0,
            y: 0,
            width: 14,
            height: 1,
        });
        StatusLine::new(Rollup {
            health: Health::Healthy,
            label: "healthy".into(),
        })
        .hint("j/k", "move")
        .hint("↵", "edit")
        .render(narrow_buf.area, &mut narrow_buf);
        let narrow: String = (0..14)
            .map(|x| narrow_buf.cell((x, 0)).expect("cell in area").symbol())
            .collect();
        assert!(narrow.starts_with("● healthy"), "{narrow:?}");
        assert!(
            !narrow.contains("move"),
            "half a hint row is not help: {narrow:?}"
        );
    }

    #[test]
    fn the_edit_indicator_appears_only_while_an_edit_is_open() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let rollup = Rollup {
            health: Health::Healthy,
            label: "healthy".into(),
        };

        let idle = row(&render(StatusLine::new(rollup.clone())), 0);
        assert!(
            !idle.contains("INSERT") && !idle.contains("NORMAL"),
            "{idle:?}"
        );

        let editing = render(StatusLine::new(rollup).mode(Some(EditMode::Insert)));
        let line = row(&editing, 0);
        assert!(line.starts_with("-- INSERT --"), "{line:?}");
        // §3: bold, no hue — cyan is the selector's and this must not read as a selection.
        assert!(style_at(&editing, 0).add_modifier.contains(Modifier::BOLD));
        assert_ne!(style_at(&editing, 0).fg, Some(tokens::selector()));
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use crate::tokens::Health;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "rollup",
            ty: "Rollup",
            description: "§7's single dot: the whole system's health in one glyph and one phrase",
        },
        PropInfo {
            name: "mode",
            ty: "Option<EditMode>",
            description:
                "Taken from the table that owns the edit, so caret and indicator cannot disagree",
        },
        PropInfo {
            name: "hints",
            ty: "Vec<(String, String)>",
            description:
                "Key hints for what is selected — dropped whole when the line is too narrow",
        },
    ];

    struct Variant {
        name: &'static str,
        description: &'static str,
        build: fn() -> StatusLine,
    }

    impl Ingredient for Variant {
        fn group(&self) -> &str {
            "Status Line"
        }
        fn name(&self) -> &str {
            self.name
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::chrome::status_line"
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

    fn rollup(health: Health, label: &str) -> Rollup {
        Rollup {
            health,
            label: label.into(),
        }
    }

    fn hinted(health: Health, label: &str) -> StatusLine {
        StatusLine::new(rollup(health, label))
            .hint("j/k", "move")
            .hint("↵", "edit")
            .hint("q", "quit")
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Variant {
                name: "Healthy",
                description: "Nothing is wrong: one green dot, and the rest of the line is hints",
                build: || hinted(Health::Healthy, "healthy"),
            }),
            Box::new(Variant {
                name: "Degraded",
                description: "The one hue on the line. §4 caps it: never more vibrant than the content above",
                build: || hinted(Health::Degraded, "1 degraded"),
            }),
            Box::new(Variant {
                name: "Offline",
                description: "The loudest this line ever gets — and it is still one glyph",
                build: || hinted(Health::Offline, "vector store offline"),
            }),
            Box::new(Variant {
                name: "Editing",
                description: "An edit is open: the mode indicator leads, on weight alone, because cyan belongs to the selector",
                build: || {
                    StatusLine::new(rollup(Health::Healthy, "healthy"))
                        .mode(Some(EditMode::Insert))
                        .hint("esc", "normal")
                        .hint("↵", "accept")
                },
            }),
        ]
    }
}

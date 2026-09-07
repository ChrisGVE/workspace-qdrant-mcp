//! The merged status-and-help line at the foot of the screen.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens;
use crate::widgets::config_table::EditMode;

/// The merged status-and-help line at the foot of the screen (§6).
///
/// Left: the edit-mode indicator when there is one. Right: the keys available for what is
/// selected. The roll-up dot that once led the left half is gone (Chris, 2026-09-07): the
/// status block at the top of every tab is the permanent detailed health, so the foot
/// repeating a summary of it was redundant. §4's cap — *"never more vibrant than the
/// content"* — is therefore absolute now: the line carries no hue at all, and §3 keeps the
/// mode indicator on weight alone because cyan belongs to the selector.
pub struct StatusLine {
    mode: Option<EditMode>,
    /// Each hint, and whether it survives a foot too narrow for all of them. See [`ALWAYS`].
    hints: Vec<(String, String, bool)>,
}

/// The two hints every screen ends with — and the two that stay when the rest cannot fit.
///
/// Chris, 2026-09-07, on the Queue tab's foot: when the hints do not fit the width, render
/// `? Help · q Quit` alone, *"the help modal carries them all"*. The rule is about **every**
/// view, so it lives here and not in one of them, and [`StatusLine::hint`] applies it by
/// comparing what it is handed against this list. A view therefore gets the behaviour without
/// being able to forget it, and there is no second place where the pair is spelled.
///
/// Why these two and not a count of whatever fits: a foot that shed hints one at a time would
/// offer a different set at every terminal width, and a reader would have to discover which
/// keys their window happens to be showing. Two keys, always the same two, and one of them opens
/// the window that lists the rest.
pub const ALWAYS: [(&str, &str); 2] = [("?", "Help"), ("q", "Quit")];

impl StatusLine {
    pub fn new() -> Self {
        Self {
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

    /// Offer a key. A hint that is one of [`ALWAYS`] is marked as surviving a narrow foot; every
    /// other one goes when the line cannot hold them all.
    pub fn hint(mut self, key: impl Into<String>, label: impl Into<String>) -> Self {
        let (key, label) = (key.into(), label.into());
        let always = ALWAYS
            .iter()
            .any(|(k, l)| *k == key.as_str() && *l == label.as_str());
        self.hints.push((key, label, always));
        self
    }

    /// The hints to draw, given `room` columns for them: all of them, or [`ALWAYS`]'s alone, or
    /// none. Three steps rather than a search for the widest fitting subset — see [`ALWAYS`].
    fn fitting(&self, room: usize) -> Vec<(&str, &str)> {
        let all: Vec<(&str, &str)> = self
            .hints
            .iter()
            .map(|(key, label, _)| (key.as_str(), label.as_str()))
            .collect();
        if tokens::key_hints_width(&all) <= room {
            return all;
        }
        let always: Vec<(&str, &str)> = self
            .hints
            .iter()
            .filter(|(_, _, always)| *always)
            .map(|(key, label, _)| (key.as_str(), label.as_str()))
            .collect();
        if !always.is_empty() && tokens::key_hints_width(&always) <= room {
            return always;
        }
        Vec::new()
    }

    /// The left half: the edit-mode indicator when there is one, and nothing else. The health
    /// dot that used to follow it is gone (Chris, 2026-09-07) — see the struct docs — so a
    /// foot with no edit open has no left half at all.
    fn left(&self) -> Vec<Span<'static>> {
        match self.mode {
            Some(mode) => vec![mode.indicator_span(), Span::raw("  ")],
            None => Vec::new(),
        }
    }
}

impl Default for StatusLine {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for StatusLine {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }

        let left = self.left();
        let left_width: usize = left.iter().map(|s| s.content.chars().count()).sum();
        // One clear cell between the two halves is the minimum that still reads as two halves —
        // and with no edit open there is only one half, so the hints get the whole row. Below
        // that the hints fall back to [`ALWAYS`] and then go entirely: half a hint row is noise
        // rather than help.
        let room = (area.width as usize).saturating_sub(left_width + usize::from(!left.is_empty()));
        let hints = self.fitting(room);
        let hints_width = tokens::key_hints_width(&hints);

        let mut spans = left;
        if hints_width > 0 {
            let gap = area.width as usize - left_width - hints_width;
            spans.push(Span::raw(" ".repeat(gap)));
            spans.extend(tokens::key_hints(&hints));
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

    /// The roll-up dot is gone (Chris, 2026-09-07) and §4's cap went absolute with it: not one
    /// cell of the foot carries a hue, health's included. The status block at the top of every
    /// tab is the permanent detailed health — the foot repeating a summary of it was redundant,
    /// and a foot that restated it in colour was the one place the cap had an exception.
    #[test]
    fn the_foot_carries_no_hue_anywhere_not_even_for_health() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(
            StatusLine::new()
                .mode(Some(EditMode::Insert))
                .hint("↵", "edit"),
        );

        // §4: the status line is never more vibrant than the content — and now it is not
        // vibrant at all.
        let hues = [
            Health::Healthy.color(),
            Health::Degraded.color(),
            Health::Offline.color(),
            tokens::selector(),
        ];
        for x in 0..AREA.width {
            let fg = style_at(&buf, x).fg;
            assert!(
                fg.is_none_or(|c| !hues.contains(&c)),
                "column {x} carries a reserved hue"
            );
        }
    }

    #[test]
    fn the_hints_are_flush_right_and_go_whole_rather_than_halved() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let line = StatusLine::new().hint("j/k", "move").hint("↵", "edit");

        let wide = row(&render(line), 0);
        assert!(
            wide.ends_with("↵ edit"),
            "the hints sit against the right edge: {wide:?}"
        );

        // Narrow enough that the hints would have to overlap or truncate each other. They go
        // entirely rather than be shown halved: half a hint row is not help. With no status to
        // keep, the row is blank.
        let mut narrow_buf = Buffer::empty(Rect {
            x: 0,
            y: 0,
            width: 14,
            height: 1,
        });
        StatusLine::new()
            .hint("j/k", "move")
            .hint("↵", "edit")
            .render(narrow_buf.area, &mut narrow_buf);
        let narrow: String = (0..14)
            .map(|x| narrow_buf.cell((x, 0)).expect("cell in area").symbol())
            .collect();
        assert!(
            narrow.trim().is_empty(),
            "half a hint row is not help: {narrow:?}"
        );
        assert!(!narrow.contains("move"), "{narrow:?}");
    }

    /// Too narrow for the whole row, the foot falls back to [`ALWAYS`] rather than to nothing.
    ///
    /// Three widths in one guard because the rule is a ladder and a ladder fails between its
    /// rungs: wide enough for everything, wide enough for the two, and too narrow even for those.
    #[test]
    fn a_foot_too_narrow_for_every_hint_keeps_the_two_that_open_the_rest() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        fn line(width: u16) -> String {
            let mut buf = Buffer::empty(Rect {
                x: 0,
                y: 0,
                width,
                height: 1,
            });
            StatusLine::new()
                .hint("/", "Search")
                .hint("f", "Filter")
                .hint("x", "Remove")
                .hint("?", "Help")
                .hint("q", "Quit")
                .render(buf.area, &mut buf);
            (0..width)
                .map(|x| buf.cell((x, 0)).expect("cell in area").symbol())
                .collect()
        }

        let wide = line(60);
        assert!(wide.contains("/ Search") && wide.ends_with("? Help   q Quit"), "{wide:?}");

        // Room for the two, and not for `/ Search` beside them.
        let narrow = line(30);
        assert!(narrow.ends_with("? Help   q Quit"), "{narrow:?}");
        for absent in ["Search", "Filter", "Remove"] {
            assert!(!narrow.contains(absent), "{absent:?} survived: {narrow:?}");
        }

        // And below even that, the row is blank: half a hint row is not help.
        let tiny = line(14);
        assert!(
            tiny.trim().is_empty(),
            "too narrow for the two, so nothing at all: {tiny:?}"
        );
    }

    #[test]
    fn the_edit_indicator_appears_only_while_an_edit_is_open() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let idle = row(&render(StatusLine::new()), 0);
        assert!(
            !idle.contains("INSERT") && !idle.contains("NORMAL"),
            "{idle:?}"
        );

        let editing = render(StatusLine::new().mode(Some(EditMode::Insert)));
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
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
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

    /// The row with no edit open: hints alone, flush right. The health dot that once led
    /// this line is gone (Chris, 2026-09-07) — the status block at the top of every tab is
    /// the permanent detailed health, so the foot repeating a summary of it was redundant.
    fn idle() -> StatusLine {
        StatusLine::new()
            .hint("j/k", "move")
            .hint("↵", "edit")
            .hint("q", "quit")
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Variant {
                name: "Idle",
                description: "No edit open: the keys for what is selected, flush right — and nothing else",
                build: idle,
            }),
            Box::new(Variant {
                name: "Editing",
                description: "An edit is open: the mode indicator leads, on weight alone, because cyan belongs to the selector",
                build: || {
                    StatusLine::new()
                        .mode(Some(EditMode::Insert))
                        .hint("esc", "normal")
                        .hint("↵", "accept")
                },
            }),
        ]
    }
}

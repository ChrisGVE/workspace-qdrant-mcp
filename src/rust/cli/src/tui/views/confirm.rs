//! Shared confirmation modal for tracking toggles on the Projects and
//! Libraries views.
//!
//! Toggling `watch_folders.enabled` is reversible, so an explicit `y`/`N`
//! confirmation (rather than a typed exact-name confirm) is the proportionate
//! guard: it prevents an accidental keystroke from pausing indexing while
//! keeping the action one keypress away once intended.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

/// Pending request to toggle tracking for a single watch folder.
#[derive(Debug, Clone)]
pub struct ToggleConfirm {
    /// Watch folder ID the toggle targets.
    pub watch_id: String,
    /// Human-readable name shown in the prompt.
    pub name: String,
    /// Target state: `true` enables tracking, `false` disables it.
    pub enable: bool,
}

impl ToggleConfirm {
    /// The action verb for the prompt ("enable" / "disable").
    pub fn verb(&self) -> &'static str {
        if self.enable {
            "enable"
        } else {
            "disable"
        }
    }
}

/// Build a centered `Tracked?` table cell: green `Yes` when tracking is
/// enabled, red `No` when disabled. Centered within the 9-wide column shared by
/// the Projects and Libraries views.
pub fn tracked_cell(enabled: bool) -> Span<'static> {
    let (label, color) = if enabled {
        ("Yes", Color::Green)
    } else {
        ("No", Color::Red)
    };
    Span::styled(format!("{label:^9}"), Style::default().fg(color))
}

/// Draw a centered confirmation overlay for a pending tracking toggle.
pub fn draw_toggle_confirm(frame: &mut Frame, area: Rect, confirm: &ToggleConfirm) {
    let width = 56u16.min(area.width.saturating_sub(4));
    let height = 7u16.min(area.height.saturating_sub(2));
    let x = (area.width.saturating_sub(width)) / 2;
    let y = (area.height.saturating_sub(height)) / 2;
    let popup = Rect::new(x, y, width, height);

    frame.render_widget(Clear, popup);

    // Disabling stops indexing, so paint it as the attention-grabbing action.
    let accent = if confirm.enable {
        Color::Green
    } else {
        Color::Yellow
    };

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!("{} tracking", confirm.verb()),
                Style::default().fg(accent).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" for "),
            Span::styled(
                confirm.name.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw("?"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(
                "y",
                Style::default().fg(accent).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" confirm    ", Style::default().fg(Color::Gray)),
            Span::styled("n / Esc", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(" cancel", Style::default().fg(Color::Gray)),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Confirm ")
        .title_style(Style::default().add_modifier(Modifier::BOLD))
        .border_style(Style::default().fg(accent))
        .style(Style::default().bg(Color::Black));

    frame.render_widget(Paragraph::new(lines).block(block), popup);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verb_reflects_target_state() {
        let enable = ToggleConfirm {
            watch_id: "w1".into(),
            name: "proj".into(),
            enable: true,
        };
        assert_eq!(enable.verb(), "enable");
        let disable = ToggleConfirm {
            watch_id: "w1".into(),
            name: "proj".into(),
            enable: false,
        };
        assert_eq!(disable.verb(), "disable");
    }
}

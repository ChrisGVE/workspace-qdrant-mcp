//! Shared confirmation modals for TUI actions.
//!
//! Three confirmation kinds are provided:
//! - [`ToggleConfirm`]: a simple `y`/`N` modal for reversible tracking toggles.
//! - [`SimpleConfirm`]: a `y`/`N` modal for any non-destructive action (queue
//!   retry, cancel, remove; project/library nudge).
//! - [`TypedConfirm`]: an input-field modal for destructive actions (delete
//!   rule, remove library book). Requires the user to type exactly
//!   `Delete <name>` before the action is enabled.
//!
//! Simple and typed confirms are wrapped in [`ActionConfirm`] so views can
//! hold a single `Option<ActionConfirm>` for all pending actions.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

// ─── Tracking toggle (existing) ─────────────────────────────────────────────

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

// ─── Simple y/N confirm ──────────────────────────────────────────────────────

/// Pending non-destructive action awaiting a simple `y`/`N` confirm.
#[derive(Debug, Clone)]
pub struct SimpleConfirm {
    /// Short verb shown in the prompt (e.g. "Retry", "Cancel", "Remove").
    pub verb: String,
    /// Human-readable target name shown in the prompt.
    pub target: String,
}

// ─── Typed-name confirm ──────────────────────────────────────────────────────

/// Pending destructive action awaiting a typed-name confirmation.
///
/// The user must type `Delete <name>` (case-sensitive, exact) to enable the
/// action. Any other input keeps the modal open.
#[derive(Debug, Clone)]
pub struct TypedConfirm {
    /// The human-readable name the user must type (without the `Delete ` prefix).
    pub name: String,
    /// Current input buffer (what the user has typed so far).
    pub input: String,
    /// Whether the last Enter press was rejected (non-matching input).
    pub rejected: bool,
}

impl TypedConfirm {
    /// Create a new typed confirm for the given object name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            input: String::new(),
            rejected: false,
        }
    }

    /// Append a character to the input buffer.
    pub fn push_char(&mut self, c: char) {
        self.input.push(c);
        self.rejected = false;
    }

    /// Remove the last character from the input buffer.
    pub fn pop_char(&mut self) {
        self.input.pop();
        self.rejected = false;
    }

    /// Whether the current input exactly matches `Delete <name>`.
    pub fn matches(&self) -> bool {
        typed_confirm_matches(&self.name, &self.input)
    }

    /// Mark this confirm as rejected (wrong input, Enter pressed).
    pub fn mark_rejected(&mut self) {
        self.rejected = true;
    }
}

/// Check whether a typed-confirm input matches the required `Delete <name>`
/// string (case-sensitive, exact).
///
/// This is the single source of truth for the typed-confirm acceptance rule.
pub fn typed_confirm_matches(name: &str, input: &str) -> bool {
    input == format!("Delete {name}")
}

// ─── Unified action confirm ──────────────────────────────────────────────────

/// A pending action confirm for a non-destructive action (`y`/`N` prompt).
///
/// Views hold an `Option<ActionConfirm>` for actions that require a simple
/// yes/no confirmation (queue retry/cancel/remove, project/library nudge).
/// Destructive actions (rule delete, library book remove) use [`TypedConfirm`]
/// directly rather than wrapping it here.
#[derive(Debug, Clone)]
pub enum ActionConfirm {
    Simple(SimpleConfirm),
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

// ─── Drawing helpers ─────────────────────────────────────────────────────────

/// Compute the centered popup area within `area`.
fn centered_popup(area: Rect, width: u16, height: u16) -> Rect {
    let w = width.min(area.width.saturating_sub(4));
    let h = height.min(area.height.saturating_sub(2));
    let x = (area.width.saturating_sub(w)) / 2;
    let y = (area.height.saturating_sub(h)) / 2;
    Rect::new(x, y, w, h)
}

/// Draw a simple `y`/`N` confirmation modal for a non-destructive action.
pub fn draw_simple_confirm(frame: &mut Frame, area: Rect, confirm: &SimpleConfirm) {
    let popup = centered_popup(area, 58, 7);
    frame.render_widget(Clear, popup);

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(
                confirm.verb.clone(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                confirm.target.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw("?"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled(
                "y",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
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
        .border_style(Style::default().fg(Color::Cyan))
        .style(Style::default().bg(Color::Black));

    frame.render_widget(Paragraph::new(lines).block(block), popup);
}

/// Draw a typed-name confirmation modal for a destructive action.
///
/// Shows the required string (`Delete <name>`) as a reference, an input
/// field with the current buffer, and a rejection hint when the user pressed
/// Enter with a non-matching input.
pub fn draw_typed_confirm(frame: &mut Frame, area: Rect, confirm: &TypedConfirm) {
    let popup = centered_popup(area, 64, 9);
    frame.render_widget(Clear, popup);

    let required = format!("Delete {}", confirm.name);
    let ready = confirm.matches();
    let field_fg = if ready { Color::Green } else { Color::Yellow };

    let mut lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Type to confirm: "),
            Span::styled(required.clone(), Style::default().fg(Color::Gray)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  > "),
            Span::styled(
                format!("{}_", confirm.input),
                Style::default().fg(field_fg).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
    ];

    if confirm.rejected {
        lines.push(Line::from(Span::styled(
            format!("  Must match exactly: {required}"),
            Style::default().fg(Color::Red),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            "  Enter: confirm (when text matches)   Esc: cancel",
            Style::default().fg(Color::DarkGray),
        )));
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Confirm Deletion ")
        .title_style(Style::default().add_modifier(Modifier::BOLD))
        .border_style(Style::default().fg(Color::Red))
        .style(Style::default().bg(Color::Black));

    frame.render_widget(Paragraph::new(lines).block(block), popup);
}

/// Draw whichever action-confirm modal is pending.
pub fn draw_action_confirm(frame: &mut Frame, area: Rect, confirm: &ActionConfirm) {
    match confirm {
        ActionConfirm::Simple(c) => draw_simple_confirm(frame, area, c),
    }
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

    // ── ToggleConfirm ────────────────────────────────────────────────────────

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

    // ── typed_confirm_matches ────────────────────────────────────────────────

    #[test]
    fn typed_confirm_exact_match_accepted() {
        assert!(typed_confirm_matches("my-rule", "Delete my-rule"));
    }

    #[test]
    fn typed_confirm_case_sensitive_rejected() {
        // Capital D is required; lowercase should not match.
        assert!(!typed_confirm_matches("my-rule", "delete my-rule"));
        // Name casing must also match exactly.
        assert!(!typed_confirm_matches("My-Rule", "Delete my-rule"));
    }

    #[test]
    fn typed_confirm_wrong_name_rejected() {
        assert!(!typed_confirm_matches("rule-a", "Delete rule-b"));
    }

    #[test]
    fn typed_confirm_partial_input_rejected() {
        assert!(!typed_confirm_matches("my-rule", "Delete my-rul"));
        assert!(!typed_confirm_matches("my-rule", "Delete"));
        assert!(!typed_confirm_matches("my-rule", ""));
    }

    #[test]
    fn typed_confirm_trailing_space_rejected() {
        // Extra whitespace must not sneak through.
        assert!(!typed_confirm_matches("my-rule", "Delete my-rule "));
    }

    // ── TypedConfirm state ───────────────────────────────────────────────────

    #[test]
    fn typed_confirm_starts_empty_and_unrejected() {
        let c = TypedConfirm::new("foo");
        assert_eq!(c.input, "");
        assert!(!c.rejected);
        assert!(!c.matches());
    }

    #[test]
    fn typed_confirm_push_pop_char() {
        let mut c = TypedConfirm::new("foo");
        c.push_char('D');
        c.push_char('e');
        assert_eq!(c.input, "De");
        c.pop_char();
        assert_eq!(c.input, "D");
    }

    #[test]
    fn typed_confirm_push_clears_rejected_flag() {
        let mut c = TypedConfirm::new("foo");
        c.rejected = true;
        c.push_char('x');
        assert!(!c.rejected);
    }

    #[test]
    fn typed_confirm_matches_when_fully_typed() {
        let mut c = TypedConfirm::new("bar");
        for ch in "Delete bar".chars() {
            c.push_char(ch);
        }
        assert!(c.matches());
    }

    #[test]
    fn typed_confirm_mark_rejected_sets_flag() {
        let mut c = TypedConfirm::new("foo");
        c.mark_rejected();
        assert!(c.rejected);
    }
}

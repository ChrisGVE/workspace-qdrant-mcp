//! Destructive command confirmation guard.
//!
//! Two-step confirmation for dangerous operations:
//! 1. User types the exact identifier (e.g., "user/repo" or "project/name")
//! 2. User confirms with "yes" or cancels with Esc
//!
//! This forces the user to think before committing to destructive actions.

use crossterm::event::KeyCode;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::tui::theme;

/// Guard dialog state.
#[derive(Debug, Clone)]
pub enum GuardState {
    /// Not active.
    Inactive,
    /// Step 1: User must type the exact identifier to confirm.
    AwaitingIdentifier {
        /// The action being confirmed (e.g., "Delete project").
        action: String,
        /// The expected identifier the user must type.
        expected: String,
        /// What the user has typed so far.
        input: String,
    },
    /// Step 2: User must type "yes" to confirm.
    AwaitingConfirmation {
        /// The action being confirmed.
        action: String,
        /// The confirmed identifier.
        identifier: String,
        /// What the user has typed so far ("y", "ye", "yes").
        input: String,
    },
    /// Action was confirmed.
    Confirmed { action: String, identifier: String },
}

impl GuardState {
    /// Start a new guard dialog.
    pub fn start(action: String, expected_identifier: String) -> Self {
        GuardState::AwaitingIdentifier {
            action,
            expected: expected_identifier,
            input: String::new(),
        }
    }

    /// Whether the guard dialog is active.
    pub fn is_active(&self) -> bool {
        !matches!(self, GuardState::Inactive | GuardState::Confirmed { .. })
    }

    /// Whether the action was confirmed.
    pub fn is_confirmed(&self) -> bool {
        matches!(self, GuardState::Confirmed { .. })
    }

    /// Handle a key event. Returns true if consumed.
    pub fn handle_key(&mut self, code: KeyCode) -> bool {
        match self {
            GuardState::AwaitingIdentifier {
                action,
                expected,
                input,
            } => match code {
                KeyCode::Esc => {
                    *self = GuardState::Inactive;
                    true
                }
                KeyCode::Backspace => {
                    input.pop();
                    true
                }
                KeyCode::Enter => {
                    if input == expected {
                        let a = action.clone();
                        let id = input.clone();
                        *self = GuardState::AwaitingConfirmation {
                            action: a,
                            identifier: id,
                            input: String::new(),
                        };
                    }
                    // Wrong input: do nothing (user keeps typing)
                    true
                }
                KeyCode::Char(c) => {
                    input.push(c);
                    true
                }
                _ => true,
            },
            GuardState::AwaitingConfirmation {
                action,
                identifier,
                input,
            } => match code {
                KeyCode::Esc => {
                    *self = GuardState::Inactive;
                    true
                }
                KeyCode::Backspace => {
                    input.pop();
                    true
                }
                KeyCode::Enter => {
                    if input == "yes" {
                        let a = action.clone();
                        let id = identifier.clone();
                        *self = GuardState::Confirmed {
                            action: a,
                            identifier: id,
                        };
                    } else {
                        *self = GuardState::Inactive;
                    }
                    true
                }
                KeyCode::Char(c) => {
                    input.push(c);
                    true
                }
                _ => true,
            },
            _ => false,
        }
    }

    /// Draw the guard dialog overlay.
    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let (title, lines) = match self {
            GuardState::AwaitingIdentifier {
                action,
                expected,
                input,
            } => {
                let matches = input == expected;
                let input_style = if matches {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(theme::COLOR_FG)
                };

                (
                    format!(" {} — Confirm Identity ", action),
                    vec![
                        Line::from(""),
                        Line::from(Span::styled(
                            format!("  Type the exact identifier to confirm:"),
                            Style::default().fg(theme::COLOR_MUTED),
                        )),
                        Line::from(Span::styled(
                            format!("  Expected: {expected}"),
                            Style::default().fg(theme::COLOR_DIM),
                        )),
                        Line::from(""),
                        Line::from(vec![
                            Span::styled("  > ", Style::default().fg(theme::COLOR_ACCENT)),
                            Span::styled(input.clone(), input_style),
                            Span::styled("\u{2588}", Style::default().fg(theme::COLOR_ACCENT)),
                        ]),
                        Line::from(""),
                        Line::from(Span::styled(
                            "  Press Enter when correct, Esc to cancel",
                            Style::default().fg(theme::COLOR_DIM),
                        )),
                    ],
                )
            }
            GuardState::AwaitingConfirmation {
                action,
                identifier,
                input,
            } => (
                format!(" {} — Final Confirmation ", action),
                vec![
                    Line::from(""),
                    Line::from(Span::styled(
                        format!("  {action} \"{identifier}\"?"),
                        Style::default()
                            .fg(theme::COLOR_ERROR)
                            .add_modifier(Modifier::BOLD),
                    )),
                    Line::from(""),
                    Line::from(Span::styled(
                        "  Type \"yes\" to confirm:",
                        Style::default().fg(theme::COLOR_MUTED),
                    )),
                    Line::from(vec![
                        Span::styled("  > ", Style::default().fg(theme::COLOR_ACCENT)),
                        Span::raw(input.clone()),
                        Span::styled("\u{2588}", Style::default().fg(theme::COLOR_ACCENT)),
                    ]),
                    Line::from(""),
                    Line::from(Span::styled(
                        "  Press Enter to confirm, Esc to cancel",
                        Style::default().fg(theme::COLOR_DIM),
                    )),
                ],
            ),
            _ => return,
        };

        let popup_width = 60u16.min(area.width.saturating_sub(4));
        let popup_height = 10u16.min(area.height.saturating_sub(4));
        let x = (area.width.saturating_sub(popup_width)) / 2;
        let y = (area.height.saturating_sub(popup_height)) / 2;
        let popup_area = Rect::new(x, y, popup_width, popup_height);

        frame.render_widget(Clear, popup_area);
        let popup = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .title_style(
                    Style::default()
                        .fg(theme::COLOR_ERROR)
                        .add_modifier(Modifier::BOLD),
                )
                .style(theme::popup_style()),
        );
        frame.render_widget(popup, popup_area);
    }

    /// Reset to inactive.
    pub fn reset(&mut self) {
        *self = GuardState::Inactive;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_creates_awaiting_identifier() {
        let g = GuardState::start("Delete".into(), "user/repo".into());
        assert!(g.is_active());
        assert!(!g.is_confirmed());
    }

    #[test]
    fn esc_cancels_at_any_step() {
        let mut g = GuardState::start("Delete".into(), "foo".into());
        g.handle_key(KeyCode::Esc);
        assert!(!g.is_active());
    }

    #[test]
    fn correct_identifier_advances_to_confirmation() {
        let mut g = GuardState::start("Delete".into(), "foo".into());
        g.handle_key(KeyCode::Char('f'));
        g.handle_key(KeyCode::Char('o'));
        g.handle_key(KeyCode::Char('o'));
        g.handle_key(KeyCode::Enter);
        assert!(matches!(g, GuardState::AwaitingConfirmation { .. }));
    }

    #[test]
    fn wrong_identifier_stays_at_step_1() {
        let mut g = GuardState::start("Delete".into(), "foo".into());
        g.handle_key(KeyCode::Char('b'));
        g.handle_key(KeyCode::Char('a'));
        g.handle_key(KeyCode::Char('r'));
        g.handle_key(KeyCode::Enter);
        assert!(matches!(g, GuardState::AwaitingIdentifier { .. }));
    }

    #[test]
    fn typing_yes_confirms() {
        let mut g = GuardState::start("Delete".into(), "foo".into());
        // Step 1
        for c in "foo".chars() {
            g.handle_key(KeyCode::Char(c));
        }
        g.handle_key(KeyCode::Enter);
        // Step 2
        for c in "yes".chars() {
            g.handle_key(KeyCode::Char(c));
        }
        g.handle_key(KeyCode::Enter);
        assert!(g.is_confirmed());
    }

    #[test]
    fn typing_no_cancels_at_step_2() {
        let mut g = GuardState::start("Delete".into(), "foo".into());
        for c in "foo".chars() {
            g.handle_key(KeyCode::Char(c));
        }
        g.handle_key(KeyCode::Enter);
        for c in "no".chars() {
            g.handle_key(KeyCode::Char(c));
        }
        g.handle_key(KeyCode::Enter);
        assert!(!g.is_active());
        assert!(!g.is_confirmed());
    }

    #[test]
    fn backspace_works() {
        let mut g = GuardState::start("Delete".into(), "ab".into());
        g.handle_key(KeyCode::Char('a'));
        g.handle_key(KeyCode::Char('x'));
        g.handle_key(KeyCode::Backspace);
        g.handle_key(KeyCode::Char('b'));
        g.handle_key(KeyCode::Enter);
        assert!(matches!(g, GuardState::AwaitingConfirmation { .. }));
    }

    #[test]
    fn reset_clears() {
        let mut g = GuardState::start("Delete".into(), "foo".into());
        g.reset();
        assert!(!g.is_active());
    }
}

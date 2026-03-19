//! TUI application core — event loop and state management.

use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs};
use ratatui::Frame;

use super::event::{Event, EventHandler};
use super::terminal;

/// Active view in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Dashboard,
    Queue,
    Projects,
    Libraries,
    Logs,
}

impl View {
    /// All views in tab order.
    const ALL: [View; 5] = [
        View::Dashboard,
        View::Queue,
        View::Projects,
        View::Libraries,
        View::Logs,
    ];

    fn label(self) -> &'static str {
        match self {
            View::Dashboard => "Dashboard",
            View::Queue => "Queue",
            View::Projects => "Projects",
            View::Libraries => "Libraries",
            View::Logs => "Logs",
        }
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|v| *v == self).unwrap_or(0)
    }

    fn from_index(i: usize) -> Self {
        Self::ALL.get(i).copied().unwrap_or(View::Dashboard)
    }

    fn next(self) -> Self {
        Self::from_index((self.index() + 1) % Self::ALL.len())
    }

    fn prev(self) -> Self {
        let len = Self::ALL.len();
        Self::from_index((self.index() + len - 1) % len)
    }
}

/// TUI application state.
pub struct App {
    /// Whether the app is still running.
    pub running: bool,
    /// Currently active view.
    pub current_view: View,
    /// Whether the help overlay is visible.
    pub show_help: bool,
    /// Daemon gRPC address for data fetching.
    pub daemon_addr: String,
}

impl App {
    /// Create a new app instance.
    pub fn new(daemon_addr: String) -> Self {
        Self {
            running: true,
            current_view: View::Dashboard,
            show_help: false,
            daemon_addr,
        }
    }

    /// Main run loop: setup terminal, handle events, render, cleanup.
    pub fn run(&mut self) -> anyhow::Result<()> {
        let mut term = terminal::init()?;
        let events = EventHandler::new(Duration::from_millis(250));

        while self.running {
            term.draw(|frame| self.draw(frame))?;

            match events.next() {
                Ok(Event::Key(key)) => self.handle_key(key),
                Ok(Event::Resize(_, _)) => {} // ratatui handles resize on next draw
                Ok(Event::Tick) => {}         // placeholder for future live updates
                Err(_) => self.running = false,
            }
        }

        terminal::restore()?;
        Ok(())
    }

    /// Handle a key event with global bindings.
    fn handle_key(&mut self, key: KeyEvent) {
        // Help overlay captures Esc and ? only
        if self.show_help {
            match key.code {
                KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('q') => {
                    self.show_help = false;
                }
                _ => {}
            }
            return;
        }

        match key.code {
            // Quit
            KeyCode::Char('q') => self.running = false,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.running = false;
            }

            // View switching by number
            KeyCode::Char('1') => self.current_view = View::Dashboard,
            KeyCode::Char('2') => self.current_view = View::Queue,
            KeyCode::Char('3') => self.current_view = View::Projects,
            KeyCode::Char('4') => self.current_view = View::Libraries,
            KeyCode::Char('5') => self.current_view = View::Logs,

            // Tab navigation
            KeyCode::Tab => self.current_view = self.current_view.next(),
            KeyCode::BackTab => self.current_view = self.current_view.prev(),

            // Help
            KeyCode::Char('?') => self.show_help = true,

            // Delegate to view-specific handlers (future)
            _ => {}
        }
    }

    /// Render the current state.
    fn draw(&self, frame: &mut Frame) {
        let chunks = Layout::vertical([
            Constraint::Length(3), // tab bar
            Constraint::Min(1),    // main content
            Constraint::Length(1), // status bar
        ])
        .split(frame.area());

        // Tab bar
        let titles: Vec<Line> = View::ALL
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let style = if *v == self.current_view {
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                Line::from(Span::styled(format!(" {} {} ", i + 1, v.label()), style))
            })
            .collect();

        let tabs = Tabs::new(titles)
            .select(self.current_view.index())
            .divider(Span::raw("│"))
            .block(
                Block::default()
                    .borders(Borders::BOTTOM)
                    .title(" wqm ")
                    .title_style(Style::default().add_modifier(Modifier::BOLD)),
            );
        frame.render_widget(tabs, chunks[0]);

        // Main content area — placeholder per view
        let content = Paragraph::new(format!(
            "{} view\n\nThis view will be implemented in a future update.",
            self.current_view.label()
        ))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(content, chunks[1]);

        // Status bar
        let status = Paragraph::new(Line::from(vec![
            Span::styled(" q", Style::default().fg(Color::Yellow)),
            Span::raw(" quit  "),
            Span::styled("Tab", Style::default().fg(Color::Yellow)),
            Span::raw(" switch  "),
            Span::styled("?", Style::default().fg(Color::Yellow)),
            Span::raw(" help  "),
            Span::styled("1-5", Style::default().fg(Color::Yellow)),
            Span::raw(" jump"),
        ]))
        .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(status, chunks[2]);

        // Help overlay
        if self.show_help {
            self.draw_help_overlay(frame);
        }
    }

    /// Draw the help overlay centered on screen.
    fn draw_help_overlay(&self, frame: &mut Frame) {
        let area = frame.area();
        let help_width = 50u16.min(area.width.saturating_sub(4));
        let help_height = 14u16.min(area.height.saturating_sub(4));

        let x = (area.width.saturating_sub(help_width)) / 2;
        let y = (area.height.saturating_sub(help_height)) / 2;

        let help_area = ratatui::layout::Rect::new(x, y, help_width, help_height);

        // Clear the area
        frame.render_widget(ratatui::widgets::Clear, help_area);

        let help_text = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  q / Ctrl+C  ", Style::default().fg(Color::Yellow)),
                Span::raw("Quit"),
            ]),
            Line::from(vec![
                Span::styled("  Tab         ", Style::default().fg(Color::Yellow)),
                Span::raw("Next view"),
            ]),
            Line::from(vec![
                Span::styled("  Shift+Tab   ", Style::default().fg(Color::Yellow)),
                Span::raw("Previous view"),
            ]),
            Line::from(vec![
                Span::styled("  1-5         ", Style::default().fg(Color::Yellow)),
                Span::raw("Jump to view"),
            ]),
            Line::from(vec![
                Span::styled("  j/k         ", Style::default().fg(Color::Yellow)),
                Span::raw("Navigate list (future)"),
            ]),
            Line::from(vec![
                Span::styled("  Enter       ", Style::default().fg(Color::Yellow)),
                Span::raw("Show detail (future)"),
            ]),
            Line::from(vec![
                Span::styled("  /           ", Style::default().fg(Color::Yellow)),
                Span::raw("Filter/search (future)"),
            ]),
            Line::from(vec![
                Span::styled("  Esc / ?     ", Style::default().fg(Color::Yellow)),
                Span::raw("Close this help"),
            ]),
            Line::from(""),
        ];

        let help = Paragraph::new(help_text).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Help ")
                .title_style(Style::default().add_modifier(Modifier::BOLD))
                .style(Style::default().bg(Color::Black)),
        );
        frame.render_widget(help, help_area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_starts_on_dashboard() {
        let app = App::new("http://127.0.0.1:50051".into());
        assert!(app.running);
        assert_eq!(app.current_view, View::Dashboard);
        assert!(!app.show_help);
    }

    #[test]
    fn view_next_wraps() {
        assert_eq!(View::Dashboard.next(), View::Queue);
        assert_eq!(View::Logs.next(), View::Dashboard);
    }

    #[test]
    fn view_prev_wraps() {
        assert_eq!(View::Dashboard.prev(), View::Logs);
        assert_eq!(View::Queue.prev(), View::Dashboard);
    }

    #[test]
    fn handle_key_quit() {
        let mut app = App::new("addr".into());
        app.handle_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
        assert!(!app.running);
    }

    #[test]
    fn handle_key_ctrl_c_quit() {
        let mut app = App::new("addr".into());
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL));
        assert!(!app.running);
    }

    #[test]
    fn handle_key_number_switches_view() {
        let mut app = App::new("addr".into());
        app.handle_key(KeyEvent::new(KeyCode::Char('3'), KeyModifiers::NONE));
        assert_eq!(app.current_view, View::Projects);
    }

    #[test]
    fn handle_key_tab_cycles() {
        let mut app = App::new("addr".into());
        assert_eq!(app.current_view, View::Dashboard);
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.current_view, View::Queue);
    }

    #[test]
    fn handle_key_help_toggle() {
        let mut app = App::new("addr".into());
        assert!(!app.show_help);
        app.handle_key(KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE));
        assert!(app.show_help);
        // While help is shown, q closes help instead of quitting
        app.handle_key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
        assert!(!app.show_help);
        assert!(app.running); // did not quit
    }

    #[test]
    fn view_label_and_index() {
        assert_eq!(View::Dashboard.label(), "Dashboard");
        assert_eq!(View::Queue.index(), 1);
        assert_eq!(View::from_index(4), View::Logs);
        assert_eq!(View::from_index(99), View::Dashboard);
    }
}

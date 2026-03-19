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
use super::views::dashboard::Dashboard;
use super::views::logs::LogViewer;

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
    /// Dashboard view state (lazily initialized on first Dashboard view).
    dashboard: Option<Dashboard>,
    /// Log viewer state (lazily initialized on first Logs view).
    log_viewer: Option<LogViewer>,
}

impl App {
    /// Create a new app instance.
    pub fn new(daemon_addr: String) -> Self {
        Self {
            running: true,
            current_view: View::Dashboard,
            show_help: false,
            daemon_addr,
            dashboard: None,
            log_viewer: None,
        }
    }

    /// Return a mutable reference to the dashboard, creating it if needed.
    fn dashboard(&mut self) -> &mut Dashboard {
        self.dashboard.get_or_insert_with(Dashboard::new)
    }

    /// Return a mutable reference to the log viewer, creating it if needed.
    fn log_viewer(&mut self) -> &mut LogViewer {
        self.log_viewer.get_or_insert_with(LogViewer::new)
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
                Ok(Event::Tick) => self.on_tick(),
                Err(_) => self.running = false,
            }
        }

        terminal::restore()?;
        Ok(())
    }

    /// Handle periodic tick events for live-updating views.
    fn on_tick(&mut self) {
        match self.current_view {
            View::Dashboard => {
                self.dashboard().on_tick();
            }
            View::Logs => {
                self.log_viewer().on_tick();
            }
            _ => {}
        }
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

        // Delegate scrolling keys to the log viewer when on Logs view
        if self.current_view == View::Logs {
            match key.code {
                KeyCode::Char('j') | KeyCode::Down => {
                    self.log_viewer().scroll_down();
                    return;
                }
                KeyCode::Char('k') | KeyCode::Up => {
                    self.log_viewer().scroll_up();
                    return;
                }
                KeyCode::Char('G') => {
                    self.log_viewer().scroll_to_bottom();
                    return;
                }
                KeyCode::PageUp => {
                    // Use a reasonable default page size; actual height
                    // is not available here, so use 20 as an approximation.
                    self.log_viewer().page_up(20);
                    return;
                }
                KeyCode::PageDown => {
                    self.log_viewer().page_down(20);
                    return;
                }
                _ => {}
            }
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
            .divider(Span::raw("|"))
            .block(
                Block::default()
                    .borders(Borders::BOTTOM)
                    .title(" wqm ")
                    .title_style(Style::default().add_modifier(Modifier::BOLD)),
            );
        frame.render_widget(tabs, chunks[0]);

        // Main content area
        match self.current_view {
            View::Dashboard => {
                if let Some(dash) = &self.dashboard {
                    dash.draw(frame, chunks[1]);
                } else {
                    draw_loading(frame, chunks[1], "Dashboard");
                }
            }
            View::Logs => {
                if let Some(viewer) = &self.log_viewer {
                    viewer.draw(frame, chunks[1]);
                } else {
                    draw_loading(frame, chunks[1], "Logs");
                }
            }
            _ => {
                let content = Paragraph::new(format!(
                    "{} view\n\nThis view will be implemented in a future update.",
                    self.current_view.label()
                ))
                .alignment(Alignment::Center)
                .style(Style::default().fg(Color::DarkGray));
                frame.render_widget(content, chunks[1]);
            }
        }

        // Status bar
        let status_spans = if self.current_view == View::Logs {
            vec![
                Span::styled(" q", Style::default().fg(Color::Yellow)),
                Span::raw(" quit  "),
                Span::styled("j/k", Style::default().fg(Color::Yellow)),
                Span::raw(" scroll  "),
                Span::styled("G", Style::default().fg(Color::Yellow)),
                Span::raw(" bottom  "),
                Span::styled("Tab", Style::default().fg(Color::Yellow)),
                Span::raw(" switch  "),
                Span::styled("?", Style::default().fg(Color::Yellow)),
                Span::raw(" help"),
            ]
        } else {
            vec![
                Span::styled(" q", Style::default().fg(Color::Yellow)),
                Span::raw(" quit  "),
                Span::styled("Tab", Style::default().fg(Color::Yellow)),
                Span::raw(" switch  "),
                Span::styled("?", Style::default().fg(Color::Yellow)),
                Span::raw(" help  "),
                Span::styled("1-5", Style::default().fg(Color::Yellow)),
                Span::raw(" jump"),
            ]
        };
        let status =
            Paragraph::new(Line::from(status_spans)).style(Style::default().fg(Color::DarkGray));
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
                Span::raw("Scroll up/down (Logs)"),
            ]),
            Line::from(vec![
                Span::styled("  G           ", Style::default().fg(Color::Yellow)),
                Span::raw("Jump to bottom (Logs)"),
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

/// Draw a "Loading..." placeholder for a view that has not been initialized yet.
fn draw_loading(frame: &mut Frame, area: ratatui::layout::Rect, view_name: &str) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {} ", view_name))
        .title_style(Style::default().add_modifier(Modifier::BOLD));
    let p = Paragraph::new(format!("Loading {}...", view_name.to_lowercase()))
        .style(Style::default().fg(Color::DarkGray))
        .block(block);
    frame.render_widget(p, area);
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
        assert!(app.log_viewer.is_none());
        assert!(app.dashboard.is_none());
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

    #[test]
    fn log_viewer_lazy_init() {
        let mut app = App::new("addr".into());
        assert!(app.log_viewer.is_none());
        // Switching to Logs view and pressing j should initialize the viewer
        app.current_view = View::Logs;
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.log_viewer.is_some());
    }

    #[test]
    fn log_scroll_keys_do_not_quit() {
        let mut app = App::new("addr".into());
        app.current_view = View::Logs;
        // j/k should scroll, not trigger global bindings
        app.handle_key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE));
        assert!(app.running);
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE));
        assert!(app.running);
    }

    #[test]
    fn on_tick_initializes_log_viewer_on_logs_view() {
        let mut app = App::new("addr".into());
        app.current_view = View::Logs;
        app.on_tick();
        assert!(app.log_viewer.is_some());
    }

    #[test]
    fn on_tick_initializes_dashboard_on_dashboard_view() {
        let mut app = App::new("addr".into());
        assert!(app.dashboard.is_none());
        app.current_view = View::Dashboard;
        app.on_tick();
        assert!(app.dashboard.is_some());
    }

    #[test]
    fn dashboard_lazy_init() {
        let mut app = App::new("addr".into());
        assert!(app.dashboard.is_none());
        let _ = app.dashboard();
        assert!(app.dashboard.is_some());
    }
}

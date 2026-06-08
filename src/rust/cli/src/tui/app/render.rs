//! Rendering / drawing functions for the TUI application.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use super::super::theme;
use super::{App, View};

impl App {
    /// Render the current state.
    pub(super) fn draw(&self, frame: &mut Frame) {
        // Check alarm state from service view
        let alarm = self
            .service_view
            .as_ref()
            .map_or(false, |sv| sv.alarm_active());

        let chunks = Layout::vertical([
            Constraint::Length(1), // tab bar (single line)
            Constraint::Min(1),    // main content
            Constraint::Length(1), // status bar
        ])
        .split(frame.area());

        // Apply alarm background tint to entire frame if service down
        if alarm {
            let alarm_bg = Paragraph::new("").style(theme::alarm_style());
            frame.render_widget(alarm_bg, frame.area());
        }

        self.content_height.set(chunks[1].height);

        self.draw_tab_bar(frame, chunks[0]);
        self.draw_main_content(frame, chunks[1]);
        self.draw_status_bar(frame, chunks[2], alarm);

        if self.show_help {
            self.draw_help_overlay(frame);
        }
    }

    /// Draw the bottom status bar with contextual hints and alarm indicator.
    fn draw_status_bar(&self, frame: &mut Frame, area: Rect, alarm: bool) {
        let mut spans = Vec::new();

        // Alarm indicator
        if alarm {
            spans.push(Span::styled(
                " ▲ SERVICE DOWN ",
                Style::default()
                    .fg(theme::COLOR_FG)
                    .bg(theme::COLOR_ERROR)
                    .add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::raw(" "));
        }

        // Contextual hints based on current view
        let hints = match self.current_view {
            View::Dashboard => "p/l/s/r/a/e Focus cell  Enter Detail  ? Help  q Quit",
            View::Queue => "j/k Navigate  f Filter  Enter Detail  ? Help  q Quit",
            View::Projects | View::Libraries => "j/k Navigate  Enter Detail  ? Help  q Quit",
            View::Rules => "j/k Navigate  Enter Detail  ? Help  q Quit",
            View::Scratchpad => "j/k Navigate  Enter Detail (j/k scroll)  ? Help  q Quit",
            View::Service => "p Pause  r Resume  ? Help  q Quit",
            View::Logs => "j/k Move  Enter View  G/Esc Live  ? Help  q Quit",
        };

        spans.push(Span::styled(format!(" {hints}"), theme::status_bar_style()));

        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Draw the tab bar: "Workspace Qdrant MCP" bold, then tabs 1-5.
    fn draw_tab_bar(&self, frame: &mut Frame, area: Rect) {
        let mut spans = vec![Span::styled(
            " Workspace Qdrant MCP ",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )];

        for (i, v) in View::ALL.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(" | ", Style::default().fg(Color::Gray)));
            }
            // Number always yellow
            spans.push(Span::styled(
                format!("{} ", i + 1),
                Style::default().fg(Color::Yellow),
            ));
            // Label: inverted if active, gray if inactive
            if *v == self.current_view {
                spans.push(Span::styled(
                    v.label().to_string(),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                spans.push(Span::styled(
                    v.label().to_string(),
                    Style::default().fg(Color::Gray),
                ));
            }
        }

        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Draw the main content area for the active view.
    fn draw_main_content(&self, frame: &mut Frame, area: Rect) {
        match self.current_view {
            View::Dashboard => {
                if let Some(dash) = &self.dashboard {
                    dash.draw(frame, area);
                } else {
                    draw_loading(frame, area, "Dashboard");
                }
            }
            View::Queue => {
                if let Some(browser) = &self.queue_browser {
                    browser.draw(frame, area);
                } else {
                    draw_loading(frame, area, "Queue");
                }
            }
            View::Projects => {
                if let Some(browser) = &self.project_browser {
                    browser.draw(frame, area);
                } else {
                    draw_loading(frame, area, "Projects");
                }
            }
            View::Libraries => {
                if let Some(browser) = &self.library_browser {
                    browser.draw(frame, area);
                } else {
                    draw_loading(frame, area, "Libraries");
                }
            }
            View::Rules => {
                if let Some(browser) = &self.rule_browser {
                    browser.draw(frame, area);
                } else {
                    draw_loading(frame, area, "Rules");
                }
            }
            View::Scratchpad => {
                if let Some(browser) = &self.scratchpad_browser {
                    browser.draw(frame, area);
                } else {
                    draw_loading(frame, area, "Scratchpad");
                }
            }
            View::Service => {
                if let Some(view) = &self.service_view {
                    view.draw(frame, area);
                } else {
                    draw_loading(frame, area, "Service");
                }
            }
            View::Logs => {
                if let Some(viewer) = &self.log_viewer {
                    viewer.draw(frame, area);
                } else {
                    draw_loading(frame, area, "Logs");
                }
            }
        }
    }

    /// Draw the help overlay centered on screen.
    fn draw_help_overlay(&self, frame: &mut Frame) {
        let area = frame.area();
        let help_width = 55u16.min(area.width.saturating_sub(4));
        let help_height = 20u16.min(area.height.saturating_sub(4));

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
                Span::styled("  1-8         ", Style::default().fg(Color::Yellow)),
                Span::raw("Jump to view"),
            ]),
            Line::from(vec![
                Span::styled("  j/k         ", Style::default().fg(Color::Yellow)),
                Span::raw("Navigate (Queue, Projects, Libraries, Logs)"),
            ]),
            Line::from(vec![
                Span::styled("  ^d/^u ^f/^b ", Style::default().fg(Color::Yellow)),
                Span::raw("Half / full page down/up"),
            ]),
            Line::from(vec![
                Span::styled("  Enter       ", Style::default().fg(Color::Yellow)),
                Span::raw("Open detail (Queue, Projects, Libraries)"),
            ]),
            Line::from(vec![
                Span::styled("  f           ", Style::default().fg(Color::Yellow)),
                Span::raw("Cycle status filter (Queue)"),
            ]),
            Line::from(vec![
                Span::styled("  G           ", Style::default().fg(Color::Yellow)),
                Span::raw("Jump to bottom (Logs)"),
            ]),
            Line::from(vec![
                Span::styled("  Esc         ", Style::default().fg(Color::Yellow)),
                Span::raw("Close popup / help"),
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
fn draw_loading(frame: &mut Frame, area: Rect, view_name: &str) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {} ", view_name))
        .title_style(Style::default().add_modifier(Modifier::BOLD));
    let p = Paragraph::new(format!("Loading {}...", view_name.to_lowercase()))
        .style(Style::default().fg(Color::DarkGray))
        .block(block);
    frame.render_widget(p, area);
}

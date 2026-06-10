//! Detail popup rendering for the project browser.
//!
//! This module contains the `draw_detail_popup` method and all of its private
//! rendering helpers, extracted to keep `projects.rs` under the 500-line limit.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use super::file_list::{draw_content_overlay, draw_file_list_tab, draw_tab_bar, PopupTab};
use super::projects::ProjectBrowser;
use super::projects_data::{build_status_text, format_local_time, ProjectDetail};

impl ProjectBrowser {
    /// Draw a centered detail popup overlay with Detail / Files tabs.
    pub(super) fn draw_detail_popup(&self, frame: &mut Frame, area: Rect, detail: &ProjectDetail) {
        let popup_w = 76u16.min(area.width.saturating_sub(4));
        let popup_h = 28u16.min(area.height.saturating_sub(4));
        let x = (area.width.saturating_sub(popup_w)) / 2;
        let y = (area.height.saturating_sub(popup_h)) / 2;
        let popup_area = Rect::new(x, y, popup_w, popup_h);

        frame.render_widget(Clear, popup_area);

        // Outer block provides the border and title.
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Project Detail ")
            .title_style(Style::default().add_modifier(Modifier::BOLD))
            .style(Style::default().bg(Color::Black));
        let inner = block.inner(popup_area);
        frame.render_widget(block, popup_area);

        // Split inner area: tab bar (1 row) + content (rest).
        let sections = Layout::vertical([Constraint::Length(1), Constraint::Min(3)]).split(inner);

        draw_tab_bar(frame, sections[0], self.file_list.tab);

        match self.file_list.tab {
            PopupTab::Detail => self.draw_detail_content(frame, sections[1], detail),
            PopupTab::Files => draw_file_list_tab(frame, sections[1], &self.file_list),
        }

        // Content overlay sits on top of everything else inside the popup.
        if self.file_list.content_open() {
            draw_content_overlay(frame, popup_area, &self.file_list);
        }
    }

    /// Draw the metadata content for the Detail tab.
    pub(super) fn draw_detail_content(
        &self,
        frame: &mut Frame,
        area: Rect,
        detail: &ProjectDetail,
    ) {
        let mut lines = vec![
            detail_line("Name", &detail.name),
            detail_line("Path", &detail.display_path),
            detail_line("Watch ID", &detail.watch_id),
            detail_line("Tenant ID", &truncate_str(&detail.tenant_id, 40)),
            detail_line("Collection", &detail.collection),
            Line::from(""),
            detail_line("Status", &build_status_text(detail)),
        ];

        if let Some(ref url) = detail.git_remote_url {
            // Rows written before #126 may still carry URL credentials.
            let sanitized = wqm_common::git_url::sanitize_git_remote_url(url);
            lines.push(detail_line("Git Remote", &truncate_str(&sanitized, 50)));
        }

        lines.push(Line::from(""));
        lines.push(detail_line(
            "Created",
            &format_local_time(&detail.created_at),
        ));
        lines.push(detail_line(
            "Updated",
            &format_local_time(&detail.updated_at),
        ));
        if let Some(ref scan) = detail.last_scan {
            lines.push(detail_line("Last Scan", &format_local_time(scan)));
        }

        self.append_queue_breakdown(&mut lines, detail);
        self.append_sub_watches(&mut lines, detail);

        frame.render_widget(Paragraph::new(lines), area);
    }

    /// Append queue breakdown section to popup lines.
    pub(super) fn append_queue_breakdown(
        &self,
        lines: &mut Vec<Line<'static>>,
        detail: &ProjectDetail,
    ) {
        if detail.queue_by_status.is_empty() {
            return;
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Queue Breakdown:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        for (status, count) in &detail.queue_by_status {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("    {:<14} ", status),
                    Style::default().fg(status_color(status)),
                ),
                Span::raw(count.to_string()),
            ]));
        }
    }

    /// Append sub-watch folders section to popup lines.
    pub(super) fn append_sub_watches(
        &self,
        lines: &mut Vec<Line<'static>>,
        detail: &ProjectDetail,
    ) {
        if detail.sub_watches.is_empty() {
            return;
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Watch Folders:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        for sw in &detail.sub_watches {
            lines.push(Line::from(Span::styled(
                format!("    {}", truncate_str(sw, 55)),
                Style::default().fg(Color::Gray),
            )));
        }
    }
}

/// Map a status string to a display color.
pub(crate) fn status_color(status: &str) -> Color {
    match status {
        "done" => Color::Green,
        "pending" => Color::Yellow,
        "in_progress" => Color::Blue,
        "failed" => Color::Red,
        _ => Color::Reset,
    }
}

/// Build a key-value detail line.
fn detail_line(key: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {:<14} ", key), Style::default().fg(Color::Gray)),
        Span::raw(value.to_string()),
    ])
}

/// Truncate a string to `max_len` characters, appending "..." if truncated.
pub(crate) fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

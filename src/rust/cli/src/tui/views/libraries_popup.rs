//! Detail popup rendering for the library browser.
//!
//! This module contains the `draw_detail_popup` method and all of its private
//! rendering helpers, extracted to keep `libraries.rs` under the 500-line limit.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use super::file_list::{draw_content_overlay, draw_file_list_tab, draw_tab_bar, PopupTab};
use super::libraries::LibraryBrowser;
use super::libraries_data::{status_label, LibraryDetail};

impl LibraryBrowser {
    /// Draw a centered detail popup overlay with Detail / Files tabs.
    pub(super) fn draw_detail_popup(&self, frame: &mut Frame, area: Rect, detail: &LibraryDetail) {
        let popup_w = 76u16.min(area.width.saturating_sub(4));
        let popup_h = 28u16.min(area.height.saturating_sub(4));
        let x = (area.width.saturating_sub(popup_w)) / 2;
        let y = (area.height.saturating_sub(popup_h)) / 2;
        let popup_area = Rect::new(x, y, popup_w, popup_h);

        frame.render_widget(Clear, popup_area);

        // Outer block provides the border and title.
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Library Detail ")
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
        detail: &LibraryDetail,
    ) {
        let status = status_label(detail.enabled, detail.is_active);
        let paused = if detail.is_paused { "yes" } else { "no" };
        let archived = if detail.is_archived { "yes" } else { "no" };
        let symlinks = if detail.follow_symlinks { "yes" } else { "no" };
        let cleanup = if detail.cleanup_on_disable {
            "yes"
        } else {
            "no"
        };

        let name = detail
            .display_path
            .trim_end_matches('/')
            .rsplit('/')
            .find(|s| !s.is_empty())
            .unwrap_or(&detail.tag);

        let mut lines = vec![
            detail_line("Name", name),
            detail_line("Tag", &detail.tag),
            detail_line("Watch ID", &detail.watch_id),
            detail_line("Path", &detail.display_path),
            Line::from(""),
            detail_line("Status", status),
            detail_line("Mode", &detail.mode),
            detail_line("Documents", &detail.doc_count.to_string()),
            Line::from(""),
            detail_line("Paused", paused),
            detail_line("Archived", archived),
            detail_line("Symlinks", symlinks),
            detail_line("Cleanup", cleanup),
            Line::from(""),
            detail_line("Created", &format_local_time(&detail.created_at)),
            detail_line("Updated", &format_local_time(&detail.updated_at)),
        ];

        if let Some(ref scan) = detail.last_scan {
            lines.push(detail_line("Last Scan", &format_local_time(scan)));
        }
        if let Some(ref activity) = detail.last_activity_at {
            lines.push(detail_line("Last Active", &format_local_time(activity)));
        }

        frame.render_widget(Paragraph::new(lines), area);
    }
}

/// Build a key-value detail line.
fn detail_line(key: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {:<14} ", key), Style::default().fg(Color::Gray)),
        Span::raw(value.to_string()),
    ])
}

/// Format a UTC timestamp for local display.
fn format_local_time(utc_str: &str) -> String {
    wqm_common::timestamp_fmt::format_local(utc_str)
}

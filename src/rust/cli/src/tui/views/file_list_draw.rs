//! Rendering for the file-list popup tab and content overlay.
//!
//! Split out of `file_list.rs` (which owns the state and key handling) to keep
//! both files under the line limit. These are free functions that take the
//! popup [`FileListState`] by reference and draw into the popup area.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Row, Table};
use ratatui::Frame;

use super::file_list::{FileListState, PopupTab};
use crate::tui::theme;
use crate::tui::util::{scroll_offset, truncate_path, visible_rows};

/// Draw the tab bar (1-row strip at top of popup). Active tab = bold + cyan.
pub fn draw_tab_bar(frame: &mut Frame, area: Rect, active: PopupTab) {
    let detail_style = if active == PopupTab::Detail {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
            .add_modifier(Modifier::UNDERLINED)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let files_style = if active == PopupTab::Files {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
            .add_modifier(Modifier::UNDERLINED)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let spans = vec![
        Span::raw(" "),
        Span::styled(" Detail ", detail_style),
        Span::styled("  ", Style::default().fg(Color::DarkGray)),
        Span::styled(" Files ", files_style),
        Span::styled(
            "  Tab/Shift+Tab to switch",
            Style::default().fg(Color::DarkGray),
        ),
    ];
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Draw the file list table inside the Files tab content area.
pub fn draw_file_list_tab(frame: &mut Frame, inner: Rect, state: &FileListState) {
    // Split vertically: hint bar (1 row) + table (rest).
    let sections = Layout::vertical([Constraint::Length(1), Constraint::Min(3)]).split(inner);
    let hint_area = sections[0];
    let table_area = sections[1];

    // Key hint.
    let hint = Paragraph::new(Line::from(vec![Span::styled(
        "Enter: view content   j/k: navigate   g/G: first/last   Esc: close",
        Style::default().fg(Color::DarkGray),
    )]));
    frame.render_widget(hint, hint_area);

    if state.files.is_empty() {
        let msg =
            Paragraph::new("No tracked files found").style(Style::default().fg(Color::DarkGray));
        frame.render_widget(msg, table_area);
        return;
    }

    // Table: path, size, chunks.
    let header = Row::new(vec!["File", "Size", "Chunks"])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

    // Chrome: borders 2 + header 1 + header margin 1.
    let visible = visible_rows(table_area.height, 4);
    let offset = scroll_offset(state.file_cursor, visible);

    // Fixed: size 10, chunks 8; separator gaps 2; path fills remainder.
    let path_w = (table_area.width as usize)
        .saturating_sub(10 + 8 + 2 + 2)
        .max(10);

    let rows: Vec<Row> = state
        .files
        .iter()
        .enumerate()
        .skip(offset)
        .take(visible)
        .map(|(i, entry)| {
            let row_style = if i == state.file_cursor {
                theme::selected_row_style()
            } else {
                Style::default()
            };

            let size_str = match entry.size {
                Some(b) => super::service_data::format_bytes(b),
                None => "\u{2014}".to_string(), // em dash for "unknown"
            };

            Row::new(vec![
                Span::styled(
                    truncate_path(&entry.relative_path, path_w),
                    Style::default().fg(Color::White),
                ),
                Span::styled(format!("{:>9}", size_str), Style::default().fg(Color::Cyan)),
                Span::styled(
                    format!("{:>7}", crate::tui::util::fmt_count(entry.chunk_count)),
                    Style::default().fg(Color::Cyan),
                ),
            ])
            .style(row_style)
        })
        .collect();

    let widths = [
        Constraint::Min(10),    // relative path (flex)
        Constraint::Length(10), // size
        Constraint::Length(8),  // chunks
    ];

    let block = Block::default().borders(Borders::ALL).title(" Files ");
    let table = Table::new(rows, widths).header(header).block(block);
    frame.render_widget(table, table_area);
}

/// Draw the content overlay (full-popup panel) for the selected file.
///
/// `state.content` holds pre-rendered `Vec<Line<'static>>` produced by
/// [`crate::tui::render::content::render_for_path`] at overlay-open time.
/// The draw path clones the lines and hands them to `Paragraph::new` — no
/// re-parsing happens here.
pub fn draw_content_overlay(frame: &mut Frame, area: Rect, state: &FileListState) {
    let Some(ref rendered_lines) = state.content else {
        return;
    };

    // Use almost the full popup area, leaving a 1-cell margin on each side.
    let overlay_w = area.width.saturating_sub(2);
    let overlay_h = area.height.saturating_sub(2);
    let overlay_area = Rect::new(area.x + 1, area.y + 1, overlay_w, overlay_h);

    frame.render_widget(Clear, overlay_area);

    let filename = state
        .files
        .get(state.file_cursor)
        .map(|e| e.relative_path.as_str())
        .unwrap_or("file");

    let inner_h = overlay_h.saturating_sub(2) as usize;

    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" {filename} "))
        .title_style(Style::default().add_modifier(Modifier::BOLD))
        .style(Style::default().bg(Color::Black))
        .title_bottom(Line::from(Span::styled(
            " j/k: scroll  Esc: back ",
            Style::default().fg(Color::DarkGray),
        )));

    let scroll_pos = state.content_scroll.min(
        rendered_lines
            .len()
            .saturating_sub(inner_h)
            .try_into()
            .unwrap_or(u16::MAX),
    );

    let para = Paragraph::new(rendered_lines.clone())
        .block(block)
        .scroll((scroll_pos, 0));

    frame.render_widget(para, overlay_area);
}

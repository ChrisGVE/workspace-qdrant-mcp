//! Popup rendering functions for dashboard popups.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use crate::output::style::home_to_tilde;

use super::dashboard_popups::{ErrorDetail, FileDetailRow, FileStatus, NoteEntry, PopupState};

/// Draw the active popup overlay on the full frame area.
pub fn draw_popup(frame: &mut Frame, area: Rect, popup: &PopupState) {
    let popup_w = (area.width * 4 / 5).min(area.width.saturating_sub(4));
    let popup_h = (area.height * 4 / 5).min(area.height.saturating_sub(4));
    let x = (area.width.saturating_sub(popup_w)) / 2;
    let y = (area.height.saturating_sub(popup_h)) / 2;
    let popup_area = Rect::new(x, y, popup_w, popup_h);

    frame.render_widget(Clear, popup_area);

    match popup {
        PopupState::Project {
            name,
            is_active,
            files,
            scroll,
        } => draw_project_popup(frame, popup_area, name, *is_active, files, *scroll),
        PopupState::Library {
            name,
            files,
            scroll,
        } => draw_library_popup(frame, popup_area, name, files, *scroll),
        PopupState::Scratchpad {
            name,
            notes,
            scroll,
        } => draw_notes_popup(frame, popup_area, name, "Scratchpad", notes, *scroll),
        PopupState::Rules {
            name,
            notes,
            scroll,
        } => draw_notes_popup(frame, popup_area, name, "Rules", notes, *scroll),
        PopupState::Error(detail) => draw_error_popup(frame, popup_area, detail),
    }
}

fn draw_project_popup(
    frame: &mut Frame,
    area: Rect,
    name: &str,
    is_active: bool,
    files: &[FileDetailRow],
    scroll: usize,
) {
    let active_indicator = if is_active { " (active)" } else { "" };
    let title = format!(" {}{} ", name, active_indicator);
    let inner_h = area.height.saturating_sub(3) as usize;
    let mut lines = vec![Line::from("")];

    if files.is_empty() {
        lines.push(Line::from(Span::styled(
            "  No tracked files",
            Style::default().fg(Color::Gray),
        )));
    } else {
        lines.push(file_header_line(true));
        for file in files.iter().skip(scroll).take(inner_h.saturating_sub(2)) {
            lines.push(file_row_line(file, true));
        }
    }

    render_popup_block(frame, area, &title, lines);
}

fn draw_library_popup(
    frame: &mut Frame,
    area: Rect,
    name: &str,
    files: &[FileDetailRow],
    scroll: usize,
) {
    let title = format!(" {} ", name);
    let inner_h = area.height.saturating_sub(3) as usize;
    let mut lines = vec![Line::from("")];

    if files.is_empty() {
        lines.push(Line::from(Span::styled(
            "  No tracked files",
            Style::default().fg(Color::Gray),
        )));
    } else {
        lines.push(file_header_line(false));
        for file in files.iter().skip(scroll).take(inner_h.saturating_sub(2)) {
            lines.push(file_row_line(file, false));
        }
    }

    render_popup_block(frame, area, &title, lines);
}

fn draw_notes_popup(
    frame: &mut Frame,
    area: Rect,
    name: &str,
    kind: &str,
    notes: &[NoteEntry],
    scroll: usize,
) {
    let title = format!(" {} — {} ", name, kind);
    let inner_h = area.height.saturating_sub(3) as usize;
    let mut lines = vec![Line::from("")];

    if notes.is_empty() {
        lines.push(Line::from(Span::styled(
            "  No entries",
            Style::default().fg(Color::Gray),
        )));
    } else {
        for note in notes.iter().skip(scroll).take(inner_h.saturating_sub(1)) {
            let circle = status_circle(note.status);
            let preview = truncate(&note.content, area.width.saturating_sub(8) as usize);
            lines.push(Line::from(vec![
                Span::raw("  "),
                circle,
                Span::raw(" "),
                Span::raw(preview),
            ]));
        }
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .title_style(Style::default().add_modifier(Modifier::BOLD))
        .style(Style::default().bg(Color::Black));

    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .block(block),
        area,
    );
}

fn draw_error_popup(frame: &mut Frame, area: Rect, detail: &ErrorDetail) {
    let title = format!(" {} ", detail.collection_label);

    let mut lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  {}", detail.collection_label),
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
    ];

    for err_line in detail.error_message.lines() {
        lines.push(Line::from(Span::styled(
            format!("  {}", err_line),
            Style::default().fg(Color::Red),
        )));
    }

    lines.push(Line::from(""));

    let fields: Vec<(&str, String)> = vec![
        ("Type", detail.item_type.clone()),
        ("Operation", detail.op.clone()),
        ("Tenant", detail.tenant_id.clone()),
        (
            "File",
            detail
                .file_path
                .as_deref()
                .map(home_to_tilde)
                .unwrap_or("-".into()),
        ),
        ("Created", detail.created_at.clone()),
        ("Updated", detail.updated_at.clone()),
        ("Retries", detail.retry_count.to_string()),
    ];

    let label_width = fields.iter().map(|(k, _)| k.len()).max().unwrap_or(0);

    for (i, (label, value)) in fields.iter().enumerate() {
        let style = if i % 2 == 1 {
            Style::default().add_modifier(Modifier::REVERSED)
        } else {
            Style::default()
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("  {:<width$}  ", label, width = label_width),
                Style::default().fg(Color::Gray),
            ),
            Span::styled(value.clone(), style),
        ]));
    }

    render_popup_block(frame, area, &title, lines);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn render_popup_block(frame: &mut Frame, area: Rect, title: &str, lines: Vec<Line<'static>>) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title.to_string())
        .title_style(Style::default().add_modifier(Modifier::BOLD))
        .style(Style::default().bg(Color::Black));
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn file_header_line(with_prefix: bool) -> Line<'static> {
    let style = Style::default().fg(Color::Gray);
    let mut spans = vec![Span::styled("  ", style)];
    if with_prefix {
        spans.push(Span::styled(format!("{:<16} ", "Workspace"), style));
    }
    spans.push(Span::styled(format!("{:<20} ", "Path"), style));
    spans.push(Span::styled(format!("{:<20} ", "File"), style));
    spans.push(Span::styled(format!("{:>6} ", "Chunks"), style));
    spans.push(Span::styled("Status", style));
    Line::from(spans)
}

fn file_row_line(file: &FileDetailRow, with_prefix: bool) -> Line<'static> {
    let circle = status_circle(file.status);
    let mut spans = vec![Span::raw("  ")];
    if with_prefix {
        spans.push(Span::raw(format!("{:<16} ", truncate(&file.prefix, 16))));
    }
    spans.push(Span::raw(format!("{:<20} ", truncate(&file.rel_path, 20))));
    spans.push(Span::raw(format!("{:<20} ", truncate(&file.filename, 20))));
    spans.push(Span::styled(
        format!("{:>6} ", file.chunk_count),
        Style::default(),
    ));
    spans.push(circle);
    Line::from(spans)
}

fn status_circle(status: FileStatus) -> Span<'static> {
    match status {
        FileStatus::UpToDate => Span::styled("●", Style::default().fg(Color::Green)),
        FileStatus::Pending => Span::styled("●", Style::default().fg(Color::Yellow)),
        FileStatus::InProgress => Span::styled("●", Style::default().fg(Color::Blue)),
        FileStatus::Errored => Span::styled("●", Style::default().fg(Color::Red)),
        FileStatus::ErroredNoVersion => Span::styled(
            "◆",
            Style::default()
                .fg(Color::Red)
                .add_modifier(Modifier::REVERSED),
        ),
        FileStatus::Missing => Span::styled("○", Style::default().fg(Color::Gray)),
    }
}

fn truncate(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        s.to_string()
    } else if max > 3 {
        let t: String = s.chars().take(max - 3).collect();
        format!("{}...", t)
    } else {
        s.chars().take(max).collect()
    }
}

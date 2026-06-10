//! Drawing helpers for the queue browser view.
//!
//! Extracted from `queue.rs` to keep that file under the 500-line limit.
//! All drawing methods are implemented on `QueueBrowser` here and the free
//! helper functions that only the draw path needs live here too.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Row, Table};
use ratatui::Frame;

use super::queue::QueueBrowser;
use super::queue_data::QueueDetail;
use super::service_data::format_bytes;
use crate::tui::filter;
use crate::tui::theme;
use crate::tui::util::{truncate_end, truncate_path};

impl QueueBrowser {
    /// Draw the filter status bar above the table.
    pub(super) fn draw_filter_bar(&self, frame: &mut Frame, area: Rect) {
        let status_label = self.filter_label();
        let count = self.item_count();

        let mut spans = vec![
            Span::styled(" Status: ", Style::default().fg(Color::Gray)),
            Span::styled(
                status_label,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  ({count} items)"),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                "  [s] status  [f] filter",
                Style::default().fg(Color::DarkGray),
            ),
        ];
        spans.extend(filter::prompt_spans(self.page_filter_ref(), "Filter"));
        spans.extend(crate::tui::search::prompt_spans(
            self.search_ref(),
            self.search_match_count(),
        ));
        if let Some(msg) = self.message_ref() {
            spans.push(Span::styled(
                format!("  {msg}"),
                Style::default().fg(Color::Yellow),
            ));
        }

        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Draw the scrollable table of queue items.
    pub(super) fn draw_table(&self, frame: &mut Frame, area: Rect) {
        let header = Row::new(vec![
            "ID", "T", "Tenant", "Object", "Type", "Op", "Status", "Size", "Age",
        ])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

        // Fixed columns: ID 10, T 1, Tenant 20, Type 8, Op 8, Status 12,
        // Size 9 (right-aligned), Age 10. Object flexes to fill the rest so
        // the filename is the focus.
        let widths = [
            Constraint::Length(10),
            Constraint::Length(1),
            Constraint::Length(20),
            Constraint::Min(20),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(12),
            Constraint::Length(9),
            Constraint::Length(10),
        ];
        // Fixed widths (excl. Object) + 8 inter-column gaps + 2 borders.
        let object_w = (area.width as usize)
            .saturating_sub(10 + 1 + 20 + 8 + 8 + 12 + 9 + 10 + 8 + 2)
            .max(12);

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Queue ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        // Chrome = top+bottom borders (2) + header row (1) + header margin (1).
        let inner_height = crate::tui::util::visible_rows(area.height, 4);
        let offset = crate::tui::util::scroll_offset(self.selected_index(), inner_height);

        let visible_rows: Vec<Row> = self
            .items_slice()
            .iter()
            .enumerate()
            .skip(offset)
            .take(inner_height)
            .map(|(i, item)| {
                let matched = self.search_ref().has_query()
                    && self.search_ref().is_match(&match_haystack_item(item));
                let row_style = if i == self.selected_index() {
                    theme::selected_row_style()
                } else if matched {
                    theme::search_match_style()
                } else {
                    Style::default()
                };
                let fg = status_color(&item.status);

                let kind_fg = match item.kind {
                    'L' => theme::COLOR_INFO,
                    'P' => theme::COLOR_ACCENT,
                    _ => theme::COLOR_DIM,
                };

                // Spans set only fg; the row's background (cursor) shows through.
                Row::new(vec![
                    Span::styled(item.short_id.clone(), Style::default().fg(Color::Cyan)),
                    Span::styled(item.kind.to_string(), Style::default().fg(kind_fg)),
                    Span::raw(truncate_end(&item.project, 20)),
                    Span::raw(truncate_path(&item.object, object_w)),
                    Span::raw(item.item_type.clone()),
                    Span::raw(item.op.clone()),
                    Span::styled(item.status.clone(), Style::default().fg(fg)),
                    Span::raw(format!(
                        "{:>9}",
                        item.size.map(format_bytes).unwrap_or_default()
                    )),
                    Span::styled(
                        item.age.clone(),
                        Style::default().fg(age_color(&item.status, &item.age)),
                    ),
                ])
                .style(row_style)
            })
            .collect();

        let table = Table::new(visible_rows, widths).header(header).block(block);
        frame.render_widget(table, area);

        if self.items_slice().is_empty() {
            let inner = area.inner(ratatui::layout::Margin::new(2, 3));
            let msg =
                Paragraph::new("No queue items found").style(Style::default().fg(Color::DarkGray));
            frame.render_widget(msg, inner);
        }
    }

    /// Draw a centered detail popup overlay.
    pub(super) fn draw_detail_popup(&self, frame: &mut Frame, area: Rect, detail: &QueueDetail) {
        let popup_width = 70u16.min(area.width.saturating_sub(4));
        let popup_height = 22u16.min(area.height.saturating_sub(4));

        let x = (area.width.saturating_sub(popup_width)) / 2;
        let y = (area.height.saturating_sub(popup_height)) / 2;
        let popup_area = Rect::new(x, y, popup_width, popup_height);

        frame.render_widget(Clear, popup_area);

        let mut lines = vec![
            detail_line("Queue ID", &detail.queue_id),
            detail_line("Idemp. Key", &truncate_str(&detail.idempotency_key, 40)),
            Line::from(""),
            detail_line("Project", &detail.project),
            detail_line("Tenant ID", &truncate_str(&detail.tenant_id, 40)),
            detail_line("Object", &detail.object),
            Line::from(""),
            detail_line("Type", &detail.item_type),
            detail_line("Operation", &detail.op),
            detail_line("Collection", &detail.collection),
            detail_line("Status", &detail.status),
            detail_line("Retries", &detail.retry_count.to_string()),
            Line::from(""),
            detail_line("Created", &format_local_time(&detail.created_at)),
            detail_line("Updated", &format_local_time(&detail.updated_at)),
        ];

        if let Some(ref err) = detail.error_message {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("  Error: {}", truncate_str(err, 50)),
                Style::default().fg(Color::Red),
            )));
        }

        let payload_preview = truncate_str(&detail.payload_json, 55);
        lines.push(Line::from(""));
        lines.push(detail_line("Payload", &payload_preview));

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Queue Item Detail ")
            .title_style(Style::default().add_modifier(Modifier::BOLD))
            .style(Style::default().bg(Color::Black));

        let popup = Paragraph::new(lines).block(block);
        frame.render_widget(popup, popup_area);
    }
}

/// Build the match haystack for a single queue row item (used within this module).
fn match_haystack_item(item: &super::queue_data::QueueRow) -> String {
    format!(
        "{} {} {} {} {} {}",
        item.short_id, item.project, item.object, item.item_type, item.op, item.status
    )
}

/// Map a status string to a display color.
pub(super) fn status_color(status: &str) -> Color {
    match status {
        "done" => Color::Green,
        "pending" => Color::Yellow,
        "in_progress" => Color::Blue,
        "failed" => Color::Red,
        _ => Color::Reset,
    }
}

/// Color the Age column by queue health.
pub(super) fn age_color(status: &str, age: &str) -> Color {
    if status == "failed" {
        return Color::Red;
    }
    let stale = age.contains("h ago") || age.contains("d ago");
    if stale && (status == "pending" || status == "in_progress") {
        Color::Yellow
    } else {
        Color::Gray
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

/// Truncate a string to `max_len` characters, appending "..." if truncated.
pub(super) fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

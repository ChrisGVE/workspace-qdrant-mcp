//! Drawing helpers for the library browser view.
//!
//! Extracted from `libraries.rs` to keep that file under the 500-line limit.
//! All functions are free functions that accept the data they need directly, so
//! there is no circular dependency between this module and `libraries.rs`.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};
use ratatui::Frame;

use super::confirm::tracked_cell;
use super::libraries_data::LibraryRow;
use crate::tui::filter::{self, FilterState};
use crate::tui::search::SearchState;
use crate::tui::theme;
use crate::tui::util::truncate_path;

/// Draw the header bar above the library table.
///
/// Parameters mirror the fields of `LibraryBrowser` that the header needs.
pub(super) fn draw_header_bar(
    frame: &mut Frame,
    area: Rect,
    item_count: usize,
    page_filter: &FilterState,
    search: &SearchState,
    search_match_count: usize,
    message: Option<&str>,
) {
    let mut spans = vec![
        Span::styled(" Libraries: ", Style::default().fg(Color::Gray)),
        Span::styled(
            item_count.to_string(),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            if item_count == 1 {
                " library"
            } else {
                " libraries"
            },
            Style::default().fg(Color::DarkGray),
        ),
    ];
    spans.extend(filter::prompt_spans(page_filter, "Filter"));
    spans.extend(crate::tui::search::prompt_spans(search, search_match_count));
    if let Some(msg) = message {
        spans.push(Span::styled(
            format!("  {msg}"),
            Style::default().fg(Color::Yellow),
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Draw the scrollable table of library items.
///
/// `selected` is the row index within `items` that has the cursor.
/// `search` is used to highlight matched rows.
pub(super) fn draw_table(
    frame: &mut Frame,
    area: Rect,
    items: &[LibraryRow],
    selected: usize,
    search: &SearchState,
) {
    let header = Row::new(vec!["Name", "Path", "Docs", "Tracked?", "Mode"])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

    let widths = [
        Constraint::Length(24), // name (+ folded P:<project> marker)
        Constraint::Min(30),    // path (flex)
        Constraint::Length(8),  // docs
        Constraint::Length(9),  // tracked? (centered Yes/No)
        Constraint::Length(13), // mode
    ];
    // Path flexes; keep the trailing path (filename) visible on truncation.
    // Fixed: name 24, docs 8, tracked 9, mode 13; 4 gaps + borders.
    let path_w = (area.width as usize)
        .saturating_sub(24 + 8 + 9 + 13 + 4 + 2)
        .max(20);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Libraries ")
        .title_style(Style::default().add_modifier(Modifier::BOLD));

    // Chrome = top+bottom borders (2) + header row (1) + header margin (1).
    let inner_height = crate::tui::util::visible_rows(area.height, 4);
    let offset = crate::tui::util::scroll_offset(selected, inner_height);

    let match_haystack = |item: &LibraryRow| {
        format!(
            "{} {} {} {}",
            item.name,
            item.tag,
            item.display_path,
            item.source.as_deref().unwrap_or("")
        )
    };

    let visible_rows: Vec<Row> = items
        .iter()
        .enumerate()
        .skip(offset)
        .take(inner_height)
        .map(|(i, item)| {
            let matched = search.has_query() && search.is_match(&match_haystack(item));
            let row_style = if i == selected {
                theme::selected_row_style()
            } else if matched {
                theme::search_match_style()
            } else {
                Style::default()
            };
            // Bold marks an active library (mirrors the projects view).
            let name_style = if item.is_active {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Cyan)
            };
            // Fold the project-derived marker into the Name cell.
            let (name_cell, mode) = match &item.source {
                Some(src) => (
                    Line::from(vec![
                        Span::styled(item.name.clone(), name_style),
                        Span::raw(" "),
                        Span::styled(src.clone(), Style::default().fg(Color::Magenta)),
                    ]),
                    "sync".to_string(),
                ),
                None => (
                    Line::from(Span::styled(item.name.clone(), name_style)),
                    item.mode.clone(),
                ),
            };
            let cells: Vec<Cell> = vec![
                Cell::from(name_cell),
                Cell::from(truncate_path(&item.display_path, path_w)),
                Cell::from(Span::styled(
                    format!("{:>7}", crate::tui::util::fmt_count(item.doc_count as i64)),
                    Style::default().fg(Color::Cyan),
                )),
                Cell::from(tracked_cell(item.enabled)),
                Cell::from(mode),
            ];
            Row::new(cells).style(row_style)
        })
        .collect();

    let table = Table::new(visible_rows, widths).header(header).block(block);
    frame.render_widget(table, area);

    if items.is_empty() {
        let inner = area.inner(ratatui::layout::Margin::new(2, 3));
        let msg =
            Paragraph::new("No libraries configured").style(Style::default().fg(Color::DarkGray));
        frame.render_widget(msg, inner);
    }
}

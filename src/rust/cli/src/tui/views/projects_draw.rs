//! Table and summary-bar drawing helpers for the project browser.
//!
//! Extracted from `projects.rs` to keep that file under the 500-line limit.
//! All drawing methods are implemented on `ProjectBrowser` here.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};
use ratatui::Frame;

use super::confirm::tracked_cell;
use super::projects::ProjectBrowser;
use super::projects_data::ProjectRow;
use crate::tui::filter;
use crate::tui::theme;
use crate::tui::util::{truncate_end, truncate_path};

impl ProjectBrowser {
    /// Draw the summary bar above the table.
    pub(super) fn draw_summary_bar(&self, frame: &mut Frame, area: Rect) {
        let total = self.items_slice().len();
        let active = self.items_slice().iter().filter(|p| p.is_active).count();

        let mut spans = vec![
            Span::styled(" Projects: ", Style::default().fg(Color::Gray)),
            Span::styled(
                total.to_string(),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  Active: ", Style::default().fg(Color::Gray)),
            Span::styled(
                active.to_string(),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  Inactive: {}", total - active),
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

    /// Draw the scrollable table of projects.
    pub(super) fn draw_table(&self, frame: &mut Frame, area: Rect) {
        let header = Row::new(vec![
            "", "Name", "Tracked?", "Path", "Branch", "Docs", "Queue",
        ])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .bottom_margin(1);

        let widths = [
            Constraint::Length(2),  // status indicator
            Constraint::Length(22), // name
            Constraint::Length(9),  // tracked? (centered Yes/No)
            Constraint::Min(30),    // path
            Constraint::Length(16), // current branch
            Constraint::Length(8),  // doc count
            Constraint::Length(8),  // queue count
        ];

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Projects ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        // Chrome = top+bottom borders (2) + header row (1) + header margin (1).
        let inner_height = crate::tui::util::visible_rows(area.height, 4);
        let offset = crate::tui::util::scroll_offset(self.selected_index(), inner_height);

        // Path flexes; compute its width so truncation keeps the trailing path.
        // Fixed: indicator 2, name 22, tracked 9, branch 16, docs 8, queue 8;
        // 6 inter-column gaps + 2 borders.
        let path_w = (area.width as usize)
            .saturating_sub(2 + 22 + 9 + 16 + 8 + 8 + 6 + 2)
            .max(20);

        let visible_rows: Vec<Row> = self
            .items_slice()
            .iter()
            .enumerate()
            .skip(offset)
            .take(inner_height)
            .map(|(i, item)| self.render_row(i, item, path_w))
            .collect();

        let table = Table::new(visible_rows, widths).header(header).block(block);
        frame.render_widget(table, area);

        if self.items_slice().is_empty() {
            let inner = area.inner(ratatui::layout::Margin::new(2, 3));
            let msg = Paragraph::new("No registered projects found")
                .style(Style::default().fg(Color::DarkGray));
            frame.render_widget(msg, inner);
        }
    }

    /// Render a single table row for a project.
    ///
    /// The cursor is the row's base style; spans set only `fg` so the highlight
    /// background shows through across the whole line.
    pub(super) fn render_row(
        &self,
        index: usize,
        item: &ProjectRow,
        path_w: usize,
    ) -> Row<'static> {
        let matched = self.search_ref().has_query()
            && self
                .search_ref()
                .is_match(&super::projects::match_haystack(item));
        let row_style = if index == self.selected_index() {
            theme::selected_row_style()
        } else if matched {
            theme::search_match_style()
        } else {
            Style::default()
        };
        // Active rows are bold (the only use of bold in the list); inactive rows
        // use a legible Gray rather than the previous near-invisible DarkGray.
        let indicator = if item.is_active {
            Span::styled("\u{25cf} ", Style::default().fg(Color::Green))
        } else {
            Span::styled("\u{25cb} ", Style::default().fg(Color::Gray))
        };
        let name_style = if item.is_active {
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        let path_fg = if item.is_active {
            Color::Gray
        } else {
            Color::DarkGray
        };
        let queue_fg = if item.queue_count > 0 {
            Color::Yellow
        } else {
            Color::DarkGray
        };

        let branch_fg = if item.is_active {
            Color::Magenta
        } else {
            Color::DarkGray
        };
        Row::new(vec![
            indicator,
            Span::styled(truncate_end(&item.name, 22), name_style),
            tracked_cell(item.enabled),
            Span::styled(
                truncate_path(&item.display_path, path_w),
                Style::default().fg(path_fg),
            ),
            Span::styled(
                truncate_end(&item.branch, 16),
                Style::default().fg(branch_fg),
            ),
            Span::styled(
                format!("{:>7}", crate::tui::util::fmt_count(item.doc_count)),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled(
                format!("{:>7}", crate::tui::util::fmt_count(item.queue_count)),
                Style::default().fg(queue_fg),
            ),
        ])
        .style(row_style)
    }
}

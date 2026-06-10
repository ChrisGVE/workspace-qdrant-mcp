//! Table and detail-popup drawing helpers for the rules browser.
//!
//! Extracted from `rules.rs` to keep that file under the 500-line limit.
//! All drawing methods are implemented on `RuleBrowser` here.

use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Row, Table};
use ratatui::Frame;
use wqm_common::constants::TENANT_GLOBAL;

use super::rules::RuleBrowser;
use super::rules_data::RuleRow;
use crate::data::tenants;
use crate::tui::render::content::render_markdown;
use crate::tui::theme;

impl RuleBrowser {
    /// Draw the rules table (below the summary block).
    pub(super) fn render_rules_table(&self, frame: &mut Frame, area: Rect) {
        let header =
            Row::new(vec!["  Tenant", "Rule Text", "Updated"]).style(theme::table_header_style());

        // Chrome = top+bottom borders (2) + header row (1). No header margin.
        let visible_height = crate::tui::util::visible_rows(area.height, 3);
        let start = crate::tui::util::scroll_offset(self.selected_index(), visible_height);

        let rows: Vec<Row> = self
            .items_slice()
            .iter()
            .enumerate()
            .skip(start)
            .take(visible_height)
            .map(|(i, rule)| {
                let scope_display = scope_label(rule, self.names_ref());
                let text_preview = truncate_str(&rule.rule_text, 60);
                let updated = format_short_date(&rule.updated_at);
                let matched = self.search_ref().has_query()
                    && self
                        .search_ref()
                        .is_match(&super::rules::match_haystack(rule, self.names_ref()));
                let style = if i == self.selected_index() {
                    theme::selected_row_style()
                } else if matched {
                    theme::search_match_style()
                } else {
                    Style::default()
                };
                Row::new(vec![format!("  {}", scope_display), text_preview, updated]).style(style)
            })
            .collect();

        let widths = [
            Constraint::Length(16),
            Constraint::Min(30),
            Constraint::Length(12),
        ];

        let table = Table::new(rows, widths)
            .header(header)
            .block(Block::default().borders(Borders::ALL));

        frame.render_widget(table, area);
    }

    /// Draw the detail popup overlay for the selected rule.
    pub(super) fn draw_detail_popup(&self, frame: &mut Frame, area: Rect, rule: &RuleRow) {
        let popup_width = (area.width - 4).min(80);
        let popup_height = (area.height - 4).min(30);
        let x = (area.width.saturating_sub(popup_width)) / 2;
        let y = (area.height.saturating_sub(popup_height)) / 2;
        let popup_area = Rect::new(x, y, popup_width, popup_height);

        frame.render_widget(Clear, popup_area);

        let scope_display = if rule.scope == TENANT_GLOBAL {
            TENANT_GLOBAL.to_string()
        } else if rule.tenant_id.is_empty() {
            rule.scope.clone()
        } else {
            tenants::display_name(self.names_ref(), &rule.tenant_id)
        };

        let mut lines = vec![
            Line::from(vec![
                Span::styled("  ID:       ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&rule.rule_id),
            ]),
            Line::from(vec![
                Span::styled("  Scope:    ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&scope_display),
            ]),
            Line::from(vec![
                Span::styled("  Created:  ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&rule.created_at),
            ]),
            Line::from(vec![
                Span::styled("  Updated:  ", Style::default().fg(theme::COLOR_MUTED)),
                Span::raw(&rule.updated_at),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                "  Rule Text:",
                Style::default()
                    .fg(theme::COLOR_FG)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        // Render rule text as Markdown, word-wrapped to the popup inner width.
        let text_width = popup_width.saturating_sub(6) as usize;
        lines.extend(render_markdown(&rule.rule_text, text_width));

        let popup = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Rule Detail ")
                .title_style(Style::default().add_modifier(Modifier::BOLD))
                .style(theme::popup_style()),
        );

        frame.render_widget(popup, popup_area);
    }
}

/// Compute the scope display label for a rule row.
pub(super) fn scope_label(
    rule: &RuleRow,
    names: &std::collections::HashMap<String, String>,
) -> String {
    if rule.scope == TENANT_GLOBAL {
        TENANT_GLOBAL.to_string()
    } else if rule.tenant_id.is_empty() {
        rule.scope.clone()
    } else {
        truncate_str(&tenants::display_name(names, &rule.tenant_id), 14)
    }
}

/// Truncate a string to `max_len` characters, adding ellipsis if needed.
pub(super) fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(1)).collect();
        format!("{}\u{2026}", truncated)
    }
}

/// Format an ISO datetime string to short date (YYYY-MM-DD).
pub(super) fn format_short_date(s: &str) -> String {
    if s.len() >= 10 {
        s[..10].to_string()
    } else {
        s.to_string()
    }
}

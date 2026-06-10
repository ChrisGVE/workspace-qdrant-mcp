//! Primary rendering for the Search TUI page.
//!
//! Located at: `src/rust/cli/src/tui/views/search_render.rs`
//!
//! Contains the `draw` entry point and the per-mode render helpers for the
//! Grep, Exact, and Graph modes. The preview popup and the query prompt
//! overlay live in `search_render_detail.rs`.
//!
//! Neighbors: `search_page.rs` (state), `search_data.rs` (data model),
//! `search_render_detail.rs` (overlay renderers).

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};
use ratatui::Frame;

use super::search_data::{GraphRelatedNode, SearchMatch, SearchSnapshot};
use super::search_page::{SearchMode, SearchPageView};
use super::search_semantic::SemanticHit;
use crate::tui::theme;
use crate::tui::util::{scroll_offset, truncate_end, truncate_path, visible_rows};

impl SearchPageView {
    /// Render the Search page into `area`.
    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let snap = self.read_snapshot();

        let sections = Layout::vertical([
            Constraint::Length(2), // top bar (tenant + mode selector)
            Constraint::Min(1),    // content
        ])
        .split(area);

        self.draw_top_bar(frame, sections[0], &snap);
        self.draw_content(frame, sections[1], &snap);

        // Query prompt overlay (shown when the user pressed i or /)
        if self.prompt.active {
            self.draw_query_prompt(frame, frame.area());
        }

        // Result preview popup (shown when Enter is pressed on a result)
        if self.preview_open {
            self.draw_preview(frame, frame.area(), &snap);
        }
    }

    /// Draw the two-line top bar: tenant line + mode selector.
    pub(super) fn draw_top_bar(&self, frame: &mut Frame, area: Rect, snap: &SearchSnapshot) {
        let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).split(area);

        // Line 1: active tenant name + loading/error indicator
        let tenant_label = self
            .active_tenant()
            .map_or("(no tenants registered)".to_string(), |t| {
                format!(" Tenant: {}", t.name)
            });

        let status_span = if snap.loading {
            Span::styled("  [searching…]", Style::default().fg(theme::COLOR_WARNING))
        } else if let Some(ref e) = snap.last_error {
            Span::styled(
                format!("  [error: {}]", truncate_end(e, 60)),
                Style::default().fg(theme::COLOR_ERROR),
            )
        } else if snap.total > 0 {
            let ms_suffix = if snap.query_time_ms > 0 {
                format!(" in {}ms", snap.query_time_ms)
            } else {
                String::new()
            };
            let trunc = if snap.truncated { " (truncated)" } else { "" };
            Span::styled(
                format!(
                    "  {} result{}{}{}",
                    snap.total,
                    if snap.total == 1 { "" } else { "s" },
                    ms_suffix,
                    trunc
                ),
                Style::default().fg(theme::COLOR_ACCENT),
            )
        } else {
            Span::raw("")
        };

        let cycle_hint = if self.tenants.len() > 1 {
            Span::styled(
                "  [ ] ] cycle tenant",
                Style::default().fg(theme::COLOR_DIM),
            )
        } else {
            Span::raw("")
        };

        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    tenant_label,
                    Style::default()
                        .fg(theme::COLOR_FG)
                        .add_modifier(Modifier::BOLD),
                ),
                status_span,
                cycle_hint,
            ])),
            rows[0],
        );

        // Line 2: mode selector (1 Grep | 2 Exact | 3 Graph) + last query
        let mut spans = Vec::new();
        for (i, m) in SearchMode::ALL.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(" | ", Style::default().fg(Color::Gray)));
            }
            spans.push(Span::styled(
                format!("{} ", m.number()),
                Style::default().fg(Color::Yellow),
            ));
            if *m == self.mode {
                spans.push(Span::styled(
                    m.label(),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                spans.push(Span::styled(m.label(), Style::default().fg(Color::Gray)));
            }
        }

        // Show the last submitted query as a reminder
        if self.prompt.has_query() {
            spans.push(Span::styled(
                "   query: ",
                Style::default().fg(theme::COLOR_MUTED),
            ));
            spans.push(Span::styled(
                truncate_end(&self.prompt.last_submitted, 50),
                Style::default().fg(theme::COLOR_FG),
            ));
        } else {
            spans.push(Span::styled(
                "   i or / to search",
                Style::default().fg(theme::COLOR_DIM),
            ));
        }

        frame.render_widget(Paragraph::new(Line::from(spans)), rows[1]);
    }

    /// Route content rendering to the appropriate mode helper.
    pub(super) fn draw_content(&self, frame: &mut Frame, area: Rect, snap: &SearchSnapshot) {
        match self.mode {
            SearchMode::Grep | SearchMode::Exact => {
                self.draw_text_results(frame, area, &snap.matches, self.mode.label())
            }
            SearchMode::Graph => self.draw_graph_results(frame, area, &snap.graph_nodes),
            SearchMode::Semantic => self.draw_semantic_results(frame, area, &snap.semantic_hits),
        }
    }

    /// Render Semantic mode results as a scrollable table (#125).
    pub(super) fn draw_semantic_results(
        &self,
        frame: &mut Frame,
        area: Rect,
        hits: &[SemanticHit],
    ) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Semantic Results ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        if hits.is_empty() {
            let hint = if self.prompt.has_query() {
                "No results. Try rephrasing the query."
            } else {
                "Press i or / to enter a natural-language query, then Enter to run it."
            };
            let p = Paragraph::new(hint)
                .style(Style::default().fg(theme::COLOR_DIM))
                .block(block);
            frame.render_widget(p, area);
            return;
        }

        let header = Row::new(vec!["SCORE", "COLL", "LOCATOR", "SNIPPET"])
            .style(theme::table_header_style())
            .bottom_margin(1);

        let inner_height = visible_rows(area.height, 4);
        let offset = scroll_offset(self.selected, inner_height);

        let snippet_w = (area.width as usize)
            .saturating_sub(7 + 11 + 32 + 3 + 2)
            .max(10);

        let rows: Vec<Row> = hits
            .iter()
            .enumerate()
            .skip(offset)
            .take(inner_height)
            .map(|(i, h)| {
                let row_style = if i == self.selected {
                    theme::selected_row_style()
                } else {
                    Style::default()
                };
                Row::new(vec![
                    Span::styled(
                        format!("{:.3}", h.score),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::styled(
                        truncate_end(&h.collection, 11),
                        Style::default().fg(theme::COLOR_ACCENT),
                    ),
                    Span::styled(
                        truncate_path(&h.locator, 30),
                        Style::default().fg(theme::COLOR_MUTED),
                    ),
                    Span::raw(truncate_end(&h.snippet, snippet_w)),
                ])
                .style(row_style)
            })
            .collect();

        let widths = [
            Constraint::Length(7),
            Constraint::Length(11),
            Constraint::Length(32),
            Constraint::Min(10),
        ];
        let table = Table::new(rows, widths).header(header).block(block);
        frame.render_widget(table, area);
    }

    /// Render Grep or Exact results as a scrollable table.
    pub(super) fn draw_text_results(
        &self,
        frame: &mut Frame,
        area: Rect,
        matches: &[SearchMatch],
        title: &str,
    ) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" {} Results ", title))
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        if matches.is_empty() {
            let hint = if self.prompt.has_query() {
                "No results. Try a different query."
            } else {
                "Press i or / to enter a search query, then Enter to run it."
            };
            let p = Paragraph::new(hint)
                .style(Style::default().fg(theme::COLOR_DIM))
                .block(block);
            frame.render_widget(p, area);
            return;
        }

        let header = Row::new(vec!["FILE:LINE", "CONTENT"])
            .style(theme::table_header_style())
            .bottom_margin(1);

        // Chrome: 2 borders + 1 header + 1 header margin = 4
        let inner_height = visible_rows(area.height, 4);
        let offset = scroll_offset(self.selected, inner_height);

        // Dynamic content column width: total minus file:line column (35) + separators (3)
        let content_w = (area.width as usize).saturating_sub(35 + 3).max(10);

        let rows: Vec<Row> = matches
            .iter()
            .enumerate()
            .skip(offset)
            .take(inner_height)
            .map(|(i, m)| {
                let row_style = if i == self.selected {
                    theme::selected_row_style()
                } else {
                    Style::default()
                };
                let file_line = format!("{}:{}", truncate_path(&m.file_path, 28), m.line_number);
                Row::new(vec![
                    Span::styled(file_line, Style::default().fg(theme::COLOR_MUTED)),
                    Span::raw(truncate_end(m.content.trim(), content_w)),
                ])
                .style(row_style)
            })
            .collect();

        let widths = [Constraint::Length(35), Constraint::Min(10)];
        let table = Table::new(rows, widths).header(header).block(block);
        frame.render_widget(table, area);
    }

    /// Render Graph mode results as a scrollable table.
    pub(super) fn draw_graph_results(
        &self,
        frame: &mut Frame,
        area: Rect,
        nodes: &[GraphRelatedNode],
    ) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Graph Results ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        if nodes.is_empty() {
            let hint = if self.prompt.has_query() {
                "No related nodes found. Try a symbol name (e.g. a function or struct)."
            } else {
                "Press i or / to enter a symbol name, then Enter to find related nodes."
            };
            let p = Paragraph::new(hint)
                .style(Style::default().fg(theme::COLOR_DIM))
                .block(block);
            frame.render_widget(p, area);
            return;
        }

        let header = Row::new(vec!["EDGE", "DEPTH", "SYMBOL", "TYPE", "FILE"])
            .style(theme::table_header_style())
            .bottom_margin(1);

        let inner_height = visible_rows(area.height, 4);
        let offset = scroll_offset(self.selected, inner_height);

        let file_w = (area.width as usize)
            .saturating_sub(8 + 7 + 24 + 12 + 3 + 2)
            .max(10);

        let rows: Vec<Row> = nodes
            .iter()
            .enumerate()
            .skip(offset)
            .take(inner_height)
            .map(|(i, n)| {
                let row_style = if i == self.selected {
                    theme::selected_row_style()
                } else {
                    Style::default()
                };
                let file = if n.file_path.is_empty() {
                    "(stub)".to_string()
                } else {
                    truncate_path(&n.file_path, file_w)
                };
                Row::new(vec![
                    Span::styled(
                        truncate_end(&n.edge_type, 8),
                        Style::default().fg(theme::COLOR_ACCENT),
                    ),
                    Span::styled(n.depth.to_string(), Style::default().fg(Color::Yellow)),
                    Span::raw(truncate_end(&n.symbol_name, 24)),
                    Span::styled(
                        truncate_end(&n.symbol_type, 12),
                        Style::default().fg(Color::Gray),
                    ),
                    Span::raw(file),
                ])
                .style(row_style)
            })
            .collect();

        let widths = [
            Constraint::Length(8),
            Constraint::Length(7),
            Constraint::Length(24),
            Constraint::Length(12),
            Constraint::Min(10),
        ];
        let table = Table::new(rows, widths).header(header).block(block);
        frame.render_widget(table, area);
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::search_data::{GraphRelatedNode, SearchMatch, SearchSnapshot};
    use super::super::search_page::{SearchMode, SearchPageView};

    #[test]
    fn results_len_grep_empty() {
        let v = SearchPageView::new();
        let snap = SearchSnapshot::default();
        assert_eq!(v.results_len(&snap), 0);
    }

    #[test]
    fn results_len_graph_empty() {
        let mut v = SearchPageView::new();
        v.mode = SearchMode::Graph;
        let snap = SearchSnapshot::default();
        assert_eq!(v.results_len(&snap), 0);
    }

    #[test]
    fn results_len_exact_counts_matches() {
        let mut v = SearchPageView::new();
        v.mode = SearchMode::Exact;
        let mut snap = SearchSnapshot::default();
        snap.matches = vec![SearchMatch {
            file_path: "a.rs".into(),
            line_number: 1,
            content: "x".into(),
            context_before: vec![],
            context_after: vec![],
        }];
        assert_eq!(v.results_len(&snap), 1);
    }

    #[test]
    fn results_len_graph_counts_nodes() {
        let mut v = SearchPageView::new();
        v.mode = SearchMode::Graph;
        let mut snap = SearchSnapshot::default();
        snap.graph_nodes = vec![
            GraphRelatedNode {
                symbol_name: "foo".into(),
                symbol_type: "fn".into(),
                file_path: "src/foo.rs".into(),
                edge_type: "CALLS".into(),
                depth: 1,
            },
            GraphRelatedNode {
                symbol_name: "bar".into(),
                symbol_type: "struct".into(),
                file_path: "src/bar.rs".into(),
                edge_type: "IMPORTS".into(),
                depth: 2,
            },
        ];
        assert_eq!(v.results_len(&snap), 2);
    }
}

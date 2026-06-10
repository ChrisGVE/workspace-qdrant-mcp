//! Primary rendering for the Graph TUI page.
//!
//! Located at: `src/rust/cli/src/tui/views/graph_render.rs`
//!
//! Contains the `draw` entry point and the per-mode render helpers for Stats,
//! PageRank/Betweenness (ranked list), and Communities table.
//! Community popup, Impact result, and Impact prompt are in `graph_render_detail.rs`.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};
use ratatui::Frame;

use super::graph::{GraphMode, GraphView};
use super::graph_data::{Community, GraphSnapshot, GraphStats, RankedSymbol};
use crate::tui::theme;
use crate::tui::util::{fmt_count, scroll_offset, truncate_end, truncate_path, visible_rows};

impl GraphView {
    /// Render the Graph view into `area`.
    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let snap = self.read_snapshot();

        let sections = Layout::vertical([
            Constraint::Length(2), // top bar (tenant + mode)
            Constraint::Min(1),    // content
        ])
        .split(area);

        self.draw_top_bar(frame, sections[0], &snap);
        self.draw_content(frame, sections[1], &snap);

        // Impact prompt overlay
        if self.impact_prompt.active {
            self.draw_impact_prompt(frame, frame.area());
        }
    }

    /// Draw the two-line top bar: tenant line + mode selector line.
    pub(super) fn draw_top_bar(&self, frame: &mut Frame, area: Rect, snap: &GraphSnapshot) {
        let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).split(area);

        // Line 1: active tenant name + loading indicator
        let tenant_label = self
            .active_tenant()
            .map_or("(no tenants registered)".to_string(), |t| {
                format!(" Tenant: {}", t.name)
            });
        let loading_span = if snap.loading {
            Span::styled("  [loading…]", Style::default().fg(theme::COLOR_WARNING))
        } else if let Some(ref e) = snap.last_error {
            Span::styled(
                format!("  [error: {}]", truncate_end(e, 60)),
                Style::default().fg(theme::COLOR_ERROR),
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
                loading_span,
                cycle_hint,
            ])),
            rows[0],
        );

        // Line 2: mode selector tabs (1 Stats | 2 PageRank | …)
        let mut spans = Vec::new();
        for (i, m) in GraphMode::ALL.iter().enumerate() {
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
        frame.render_widget(Paragraph::new(Line::from(spans)), rows[1]);
    }

    /// Route content rendering to the appropriate mode-specific helper.
    pub(super) fn draw_content(&self, frame: &mut Frame, area: Rect, snap: &GraphSnapshot) {
        match self.mode {
            GraphMode::Stats => self.draw_stats(frame, area, snap.stats.as_ref()),
            GraphMode::PageRank => self.draw_ranked_list(frame, area, &snap.pagerank, "PageRank"),
            GraphMode::Communities => self.draw_communities(frame, area, &snap.communities),
            GraphMode::Betweenness => {
                self.draw_ranked_list(frame, area, &snap.betweenness, "Betweenness")
            }
            GraphMode::Impact => self.draw_impact(frame, area, snap.impact.as_ref()),
        }
    }

    /// Render the Stats summary panel.
    pub(super) fn draw_stats(&self, frame: &mut Frame, area: Rect, stats: Option<&GraphStats>) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Graph Stats ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        let Some(s) = stats else {
            let p = Paragraph::new("No data — waiting for daemon…")
                .style(Style::default().fg(theme::COLOR_DIM))
                .block(block);
            frame.render_widget(p, area);
            return;
        };

        let mut lines = vec![
            Line::from(vec![
                Span::styled("  Nodes: ", Style::default().fg(theme::COLOR_MUTED)),
                Span::styled(
                    fmt_count(s.total_nodes as i64),
                    Style::default().fg(theme::COLOR_ACCENT),
                ),
                Span::styled("   Edges: ", Style::default().fg(theme::COLOR_MUTED)),
                Span::styled(
                    fmt_count(s.total_edges as i64),
                    Style::default().fg(theme::COLOR_ACCENT),
                ),
                Span::styled(
                    if s.query_time_ms > 0 {
                        format!("   ({}ms)", s.query_time_ms)
                    } else {
                        String::new()
                    },
                    Style::default().fg(theme::COLOR_DIM),
                ),
            ]),
            Line::from(""),
        ];

        if !s.nodes_by_type.is_empty() {
            lines.push(Line::from(Span::styled(
                "  Node types:",
                Style::default().fg(theme::COLOR_MUTED),
            )));
            for (t, count) in &s.nodes_by_type {
                lines.push(Line::from(vec![
                    Span::styled(format!("    {:<18}", t), Style::default().fg(Color::Gray)),
                    Span::styled(
                        fmt_count(*count as i64),
                        Style::default().fg(theme::COLOR_ACCENT),
                    ),
                ]));
            }
            lines.push(Line::from(""));
        }

        if !s.edges_by_type.is_empty() {
            lines.push(Line::from(Span::styled(
                "  Edge types:",
                Style::default().fg(theme::COLOR_MUTED),
            )));
            for (t, count) in &s.edges_by_type {
                lines.push(Line::from(vec![
                    Span::styled(format!("    {:<18}", t), Style::default().fg(Color::Gray)),
                    Span::styled(
                        fmt_count(*count as i64),
                        Style::default().fg(theme::COLOR_ACCENT),
                    ),
                ]));
            }
        }

        let p = Paragraph::new(lines).block(block);
        frame.render_widget(p, area);
    }

    /// Render PageRank or Betweenness as a scrollable table.
    pub(super) fn draw_ranked_list(
        &self,
        frame: &mut Frame,
        area: Rect,
        entries: &[RankedSymbol],
        title: &str,
    ) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" {} ", title))
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        if entries.is_empty() {
            let p = Paragraph::new("No data — run a reload (r) or wait for auto-refresh.")
                .style(Style::default().fg(theme::COLOR_DIM))
                .block(block);
            frame.render_widget(p, area);
            return;
        }

        let header = Row::new(vec!["SCORE", "SYMBOL", "TYPE", "FILE"])
            .style(theme::table_header_style())
            .bottom_margin(1);

        // Chrome: top+bottom borders (2) + header (1) + header margin (1) = 4
        let inner_height = visible_rows(area.height, 4);
        let offset = scroll_offset(self.selected, inner_height);

        // Dynamic file column width
        let file_w = (area.width as usize)
            .saturating_sub(10 + 28 + 14 + 3 + 2)
            .max(10);

        let rows: Vec<Row> = entries
            .iter()
            .enumerate()
            .skip(offset)
            .take(inner_height)
            .map(|(i, e)| {
                let row_style = if i == self.selected {
                    theme::selected_row_style()
                } else {
                    Style::default()
                };
                let file = if e.file_path.is_empty() {
                    "(stub)".to_string()
                } else {
                    truncate_path(&e.file_path, file_w)
                };
                Row::new(vec![
                    Span::styled(
                        format!("{:.6}", e.score),
                        Style::default().fg(theme::COLOR_ACCENT),
                    ),
                    Span::raw(truncate_end(&e.symbol, 28)),
                    Span::styled(
                        truncate_end(&e.symbol_type, 14),
                        Style::default().fg(Color::Gray),
                    ),
                    Span::raw(file),
                ])
                .style(row_style)
            })
            .collect();

        let widths = [
            Constraint::Length(10),
            Constraint::Length(28),
            Constraint::Length(14),
            Constraint::Min(10),
        ];

        let table = Table::new(rows, widths).header(header).block(block);
        frame.render_widget(table, area);
    }

    /// Render the Communities list; selected community can show members.
    pub(super) fn draw_communities(
        &self,
        frame: &mut Frame,
        area: Rect,
        communities: &[Community],
    ) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Communities ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        if communities.is_empty() {
            let p = Paragraph::new("No communities detected.")
                .style(Style::default().fg(theme::COLOR_DIM))
                .block(block);
            frame.render_widget(p, area);
            return;
        }

        let header = Row::new(vec!["ID", "MEMBERS", "TOP SYMBOLS"])
            .style(theme::table_header_style())
            .bottom_margin(1);

        let inner_height = visible_rows(area.height, 4);
        let offset = scroll_offset(self.selected, inner_height);

        let sym_w = (area.width as usize).saturating_sub(6 + 10 + 3 + 2).max(10);

        let rows: Vec<Row> = communities
            .iter()
            .enumerate()
            .skip(offset)
            .take(inner_height)
            .map(|(i, c)| {
                let row_style = if i == self.selected {
                    theme::selected_row_style()
                } else {
                    Style::default()
                };
                // Show first 3 symbol names as a preview
                let preview: Vec<&str> = c
                    .members
                    .iter()
                    .take(3)
                    .map(|(s, _, _)| s.as_str())
                    .collect();
                let sym_preview = truncate_end(&preview.join(", "), sym_w);
                Row::new(vec![
                    Span::styled(
                        format!("{}", c.id),
                        Style::default().fg(theme::COLOR_ACCENT),
                    ),
                    Span::raw(fmt_count(c.member_count as i64)),
                    Span::styled(sym_preview, Style::default().fg(Color::Gray)),
                ])
                .style(row_style)
            })
            .collect();

        let widths = [
            Constraint::Length(6),
            Constraint::Length(10),
            Constraint::Min(10),
        ];

        let table = Table::new(rows, widths).header(header).block(block);
        frame.render_widget(table, area);

        // If a community is expanded, show its member popup
        if let Some(exp_idx) = self.expanded_community {
            if let Some(c) = communities.get(exp_idx) {
                self.draw_community_popup(frame, frame.area(), c);
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::graph::{GraphMode, GraphView};
    use super::super::graph_data::{Community, GraphSnapshot, RankedSymbol};

    // ── list_len — tested here because list_len drives scroll offset in render ──

    #[test]
    fn list_len_stats_is_zero() {
        let v = GraphView::new();
        let snap = GraphSnapshot::default();
        assert_eq!(v.list_len(&snap), 0);
    }

    #[test]
    fn list_len_pagerank_matches_entries() {
        let mut v = GraphView::new();
        v.mode = GraphMode::PageRank;
        let mut snap = GraphSnapshot::default();
        snap.pagerank = vec![
            RankedSymbol {
                symbol: "a".into(),
                symbol_type: "fn".into(),
                score: 0.5,
                file_path: "f.rs".into(),
            },
            RankedSymbol {
                symbol: "b".into(),
                symbol_type: "fn".into(),
                score: 0.3,
                file_path: "g.rs".into(),
            },
        ];
        assert_eq!(v.list_len(&snap), 2);
    }

    #[test]
    fn list_len_communities_matches_entries() {
        let mut v = GraphView::new();
        v.mode = GraphMode::Communities;
        let mut snap = GraphSnapshot::default();
        snap.communities = vec![Community {
            id: 1,
            member_count: 3,
            members: vec![],
        }];
        assert_eq!(v.list_len(&snap), 1);
    }
}

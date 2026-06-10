//! Detail renderers for the Graph TUI page.
//!
//! Located at: `src/rust/cli/src/tui/views/graph_render_detail.rs`
//!
//! Contains the community-member popup, the Impact analysis result panel,
//! and the Impact symbol input prompt overlay.
//! Primary mode rendering lives in `graph_render.rs`.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use super::graph::GraphView;
use super::graph_data::{Community, ImpactResult};
use crate::tui::theme;
use crate::tui::util::{truncate_end, truncate_path};

impl GraphView {
    /// Centered popup showing all members of the selected community.
    pub(super) fn draw_community_popup(
        &self,
        frame: &mut Frame,
        area: Rect,
        community: &Community,
    ) {
        let popup_w = 70u16.min(area.width.saturating_sub(4));
        let popup_h = 22u16.min(area.height.saturating_sub(4));
        let x = (area.width.saturating_sub(popup_w)) / 2;
        let y = (area.height.saturating_sub(popup_h)) / 2;
        let popup_area = Rect::new(x, y, popup_w, popup_h);

        frame.render_widget(Clear, popup_area);

        let mut lines = vec![Line::from(Span::styled(
            format!(
                "  Community {} ({} members)",
                community.id, community.member_count
            ),
            Style::default()
                .fg(theme::COLOR_ACCENT)
                .add_modifier(Modifier::BOLD),
        ))];
        lines.push(Line::from(""));

        for (sym, typ, path) in &community.members {
            let file = if path.is_empty() {
                "(stub)"
            } else {
                path.as_str()
            };
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {:<28}", truncate_end(sym, 28)),
                    Style::default().fg(theme::COLOR_FG),
                ),
                Span::styled(
                    format!("{:<12}", truncate_end(typ, 12)),
                    Style::default().fg(Color::Gray),
                ),
                Span::raw(truncate_path(file, 20)),
            ]));
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Community Members ")
            .title_style(Style::default().add_modifier(Modifier::BOLD))
            .style(theme::popup_style());

        let p = Paragraph::new(lines).block(block);
        frame.render_widget(p, popup_area);
    }

    /// Render the Impact analysis result (or the "no result yet" hint).
    pub(super) fn draw_impact(&self, frame: &mut Frame, area: Rect, impact: Option<&ImpactResult>) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Impact Analysis ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        let hint_line = Line::from(vec![
            Span::styled("  Press ", Style::default().fg(theme::COLOR_DIM)),
            Span::styled("i", Style::default().fg(Color::Yellow)),
            Span::styled(
                " to query a symbol's impact.",
                Style::default().fg(theme::COLOR_DIM),
            ),
        ]);

        let Some(r) = impact else {
            let p = Paragraph::new(vec![Line::from(""), hint_line])
                .style(Style::default())
                .block(block);
            frame.render_widget(p, area);
            return;
        };

        let mut lines = vec![
            Line::from(vec![
                Span::styled("  Symbol: ", Style::default().fg(theme::COLOR_MUTED)),
                Span::styled(
                    format!(
                        "{} direct, {} transitive | {} files affected | {} total | {}ms",
                        r.direct.len(),
                        r.transitive.len(),
                        r.affected_files.len(),
                        r.total_impacted,
                        r.query_time_ms
                    ),
                    Style::default().fg(theme::COLOR_ACCENT),
                ),
            ]),
            Line::from(""),
        ];

        if !r.direct.is_empty() {
            lines.push(Line::from(Span::styled(
                "  Direct callers:",
                Style::default().fg(theme::COLOR_MUTED),
            )));
            for (sym, path) in &r.direct {
                let file = if path.is_empty() {
                    "(stub)"
                } else {
                    path.as_str()
                };
                lines.push(Line::from(vec![
                    Span::styled("    ", Style::default()),
                    Span::raw(truncate_end(sym, 32)),
                    Span::styled(
                        format!("  {}", truncate_path(file, 30)),
                        Style::default().fg(Color::Gray),
                    ),
                ]));
            }
            lines.push(Line::from(""));
        }

        if !r.transitive.is_empty() {
            lines.push(Line::from(Span::styled(
                "  Transitive callers:",
                Style::default().fg(theme::COLOR_MUTED),
            )));
            for (sym, path, dist) in &r.transitive {
                let file = if path.is_empty() {
                    "(stub)"
                } else {
                    path.as_str()
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("    [d={}] ", dist),
                        Style::default().fg(Color::DarkGray),
                    ),
                    Span::raw(truncate_end(sym, 28)),
                    Span::styled(
                        format!("  {}", truncate_path(file, 28)),
                        Style::default().fg(Color::Gray),
                    ),
                ]));
            }
            lines.push(Line::from(""));
        }

        if !r.affected_files.is_empty() {
            lines.push(Line::from(Span::styled(
                "  Affected files:",
                Style::default().fg(theme::COLOR_MUTED),
            )));
            for f in &r.affected_files {
                lines.push(Line::from(Span::styled(
                    format!("    {}", f),
                    Style::default().fg(Color::Gray),
                )));
            }
        }

        lines.push(Line::from(""));
        lines.push(hint_line);

        let p = Paragraph::new(lines).block(block);
        frame.render_widget(p, area);
    }

    /// Small centered text-input overlay for entering a symbol name.
    pub(super) fn draw_impact_prompt(&self, frame: &mut Frame, area: Rect) {
        let popup_w = 50u16.min(area.width.saturating_sub(4));
        let popup_h = 4u16;
        let x = (area.width.saturating_sub(popup_w)) / 2;
        let y = (area.height.saturating_sub(popup_h)) / 2;
        let popup_area = Rect::new(x, y, popup_w, popup_h);

        frame.render_widget(Clear, popup_area);

        let lines = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Symbol: ", Style::default().fg(theme::COLOR_MUTED)),
                Span::styled(
                    self.impact_prompt.query.clone(),
                    Style::default().fg(theme::COLOR_ACCENT),
                ),
                Span::styled("\u{2588}", theme::search_style()), // block cursor
            ]),
        ];

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Impact Query (Enter=run, Esc=cancel) ")
            .title_style(Style::default().add_modifier(Modifier::BOLD))
            .style(theme::popup_style());

        let p = Paragraph::new(lines).block(block);
        frame.render_widget(p, popup_area);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::graph::{GraphMode, GraphView, ImpactPrompt};

    // ── ImpactPrompt ──────────────────────────────────────────────────────

    #[test]
    fn impact_prompt_activate_clears_query() {
        let mut p = ImpactPrompt::default();
        p.query = "old".into();
        p.activate();
        assert!(p.active);
        assert!(p.query.is_empty());
    }

    #[test]
    fn impact_prompt_cancel_deactivates() {
        let mut p = ImpactPrompt::default();
        p.activate();
        p.query = "foo".into();
        p.cancel();
        assert!(!p.active);
        assert!(p.query.is_empty());
    }

    #[test]
    fn impact_prompt_push_char_builds_query() {
        let mut p = ImpactPrompt::default();
        p.activate();
        p.push_char('f');
        p.push_char('o');
        p.push_char('o');
        assert_eq!(p.query, "foo");
    }

    #[test]
    fn impact_prompt_backspace_removes_last_char() {
        let mut p = ImpactPrompt::default();
        p.activate();
        for c in "foo".chars() {
            p.push_char(c);
        }
        p.backspace();
        assert_eq!(p.query, "fo");
    }

    #[test]
    fn impact_prompt_confirm_returns_symbol_and_deactivates() {
        let mut p = ImpactPrompt::default();
        p.activate();
        for c in "parse_config".chars() {
            p.push_char(c);
        }
        let sym = p.confirm();
        assert_eq!(sym, Some("parse_config".to_string()));
        assert!(!p.active);
        assert_eq!(p.last_submitted, "parse_config");
        assert!(p.query.is_empty());
    }

    #[test]
    fn impact_prompt_confirm_whitespace_only_returns_none() {
        let mut p = ImpactPrompt::default();
        p.activate();
        p.push_char(' ');
        let sym = p.confirm();
        assert_eq!(sym, None);
        assert!(!p.active);
    }

    // ── GraphView mode-switch no-op (render-adjacent) ─────────────────────

    #[test]
    fn switch_to_impact_mode_from_stats() {
        let mut v = GraphView::new();
        v.set_mode(GraphMode::Impact);
        assert_eq!(v.mode, GraphMode::Impact);
        assert_eq!(v.selected, 0);
    }
}

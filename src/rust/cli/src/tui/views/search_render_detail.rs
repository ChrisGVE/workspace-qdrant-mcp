//! Detail renderers for the Search TUI page.
//!
//! Located at: `src/rust/cli/src/tui/views/search_render_detail.rs`
//!
//! Contains two overlay renderers:
//! - `draw_query_prompt` — centered input bar for entering a search query.
//! - `draw_preview`      — full-content popup for the selected result.
//!
//! Primary mode rendering (results table, top bar) lives in `search_render.rs`.
//!
//! Neighbors: `search_page.rs` (state), `search_render.rs` (primary render),
//! `search_data.rs` (data model), `graph_render_detail.rs` (parallel for Graph page).

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use super::search_data::{GraphRelatedNode, SearchMatch, SearchSnapshot};
use super::search_page::{SearchMode, SearchPageView};
use crate::tui::render::content;
use crate::tui::theme;
use crate::tui::util::truncate_end;

impl SearchPageView {
    /// Render the query-input prompt overlay (activated by `i` or `/`).
    ///
    /// Appears as a single-line bar at vertical center, spanning most of the
    /// terminal width, so it stays readable on narrow terminals.
    pub(super) fn draw_query_prompt(&self, frame: &mut Frame, area: Rect) {
        let prompt_w = area.width.saturating_sub(4).min(80).max(30);
        let prompt_h = 3u16;
        let x = (area.width.saturating_sub(prompt_w)) / 2;
        let y = (area.height.saturating_sub(prompt_h)) / 2;
        let prompt_area = Rect::new(x, y, prompt_w, prompt_h);

        frame.render_widget(Clear, prompt_area);

        let mode_tag = match self.mode {
            SearchMode::Grep => "Grep",
            SearchMode::Exact => "Exact",
            SearchMode::Graph => "Graph symbol",
            SearchMode::Semantic => "Semantic",
        };

        let hint = format!(" {} query (Enter to run, Esc to cancel): ", mode_tag);
        let cursor = "\u{2588}"; // block cursor character

        let content_line = Line::from(vec![
            Span::styled(hint, Style::default().fg(theme::COLOR_MUTED)),
            Span::styled(
                self.prompt.query.clone(),
                Style::default().fg(theme::COLOR_FG),
            ),
            Span::styled(cursor, Style::default().fg(theme::COLOR_ACCENT)),
        ]);

        let p = Paragraph::new(content_line).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Search ")
                .title_style(
                    Style::default()
                        .fg(theme::COLOR_ACCENT)
                        .add_modifier(Modifier::BOLD),
                )
                .style(Style::default().bg(Color::Black)),
        );
        frame.render_widget(p, prompt_area);
    }

    /// Render the file-content (or graph-node detail) preview popup.
    ///
    /// For Grep/Exact: reads the matched file from disk and renders it via
    /// the content renderer with the match line highlighted. For Graph: shows
    /// traversal details for the selected node.
    ///
    /// The popup is 80% of the terminal width/height, centered.
    pub(super) fn draw_preview(&self, frame: &mut Frame, area: Rect, snap: &SearchSnapshot) {
        let popup_w = ((area.width as u32 * 4 / 5) as u16)
            .min(area.width.saturating_sub(4))
            .max(40);
        let popup_h = ((area.height as u32 * 4 / 5) as u16)
            .min(area.height.saturating_sub(4))
            .max(10);
        let x = (area.width.saturating_sub(popup_w)) / 2;
        let y = (area.height.saturating_sub(popup_h)) / 2;
        let popup_area = Rect::new(x, y, popup_w, popup_h);

        frame.render_widget(Clear, popup_area);

        match self.mode {
            SearchMode::Grep | SearchMode::Exact => {
                self.draw_text_preview(frame, popup_area, &snap.matches);
            }
            SearchMode::Graph => {
                self.draw_graph_preview(frame, popup_area, &snap.graph_nodes);
            }
            SearchMode::Semantic => {
                self.draw_semantic_preview(frame, popup_area, &snap.semantic_hits);
            }
        }
    }

    /// Preview popup for a Semantic mode result (#125).
    ///
    /// Shows the stored chunk content for the selected hit — the content
    /// came back with the Qdrant payload, so no disk read is needed.
    fn draw_semantic_preview(
        &self,
        frame: &mut Frame,
        area: Rect,
        hits: &[super::search_semantic::SemanticHit],
    ) {
        let Some(h) = hits.get(self.selected) else {
            return;
        };

        let title = format!(
            " [{:.3}] {} ({}) ",
            h.score,
            truncate_end(&h.locator, 45),
            h.collection
        );
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .title_style(Style::default().add_modifier(Modifier::BOLD))
            .style(Style::default().bg(Color::Black));

        let inner_w = area.width.saturating_sub(2) as usize;
        let max_lines = area.height.saturating_sub(4) as usize;

        let mut lines: Vec<Line<'static>> = Vec::new();
        for chunk_line in h.content.lines().take(max_lines) {
            lines.extend(content::render_plain(chunk_line, inner_w));
        }

        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "(no content)",
                Style::default().fg(theme::COLOR_DIM),
            )));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            " Esc to close",
            Style::default().fg(theme::COLOR_DIM),
        )));

        let p = Paragraph::new(lines).block(block);
        frame.render_widget(p, area);
    }

    /// Preview popup for a Grep or Exact search result.
    ///
    /// Renders the matching line plus surrounding context lines from the
    /// snapshot. A full disk read is intentionally avoided in the TUI loop;
    /// the 2-line context captured during the gRPC fetch is shown instead,
    /// with the matching line highlighted in the accent color.
    fn draw_text_preview(&self, frame: &mut Frame, area: Rect, matches: &[SearchMatch]) {
        let Some(m) = matches.get(self.selected) else {
            return;
        };

        let title = format!(
            " {} : line {} ",
            truncate_end(&m.file_path, 50),
            m.line_number
        );
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .title_style(Style::default().add_modifier(Modifier::BOLD))
            .style(Style::default().bg(Color::Black));

        let inner_w = area.width.saturating_sub(2) as usize;

        // Build lines: context_before → match line (highlighted) → context_after
        let mut lines: Vec<Line<'static>> = Vec::new();

        for ctx in &m.context_before {
            lines.extend(content::render_plain(ctx, inner_w));
        }

        // Match line: use the content renderer for syntax, then re-style
        // the first resulting line with the accent foreground to highlight it.
        let match_lines = content::render_plain(m.content.trim(), inner_w);
        for (i, line) in match_lines.into_iter().enumerate() {
            if i == 0 {
                lines.push(restyle_line_fg(line, theme::COLOR_ACCENT));
            } else {
                lines.push(line);
            }
        }

        for ctx in &m.context_after {
            lines.extend(content::render_plain(ctx, inner_w));
        }

        if lines.is_empty() {
            lines.push(Line::from(Span::styled(
                "(no content)",
                Style::default().fg(theme::COLOR_DIM),
            )));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            " Esc to close",
            Style::default().fg(theme::COLOR_DIM),
        )));

        let p = Paragraph::new(lines).block(block);
        frame.render_widget(p, area);
    }

    /// Preview popup for a Graph mode result.
    ///
    /// Shows traversal details for the selected related node: symbol, type,
    /// edge relationship, depth, and file path.
    fn draw_graph_preview(&self, frame: &mut Frame, area: Rect, nodes: &[GraphRelatedNode]) {
        let Some(n) = nodes.get(self.selected) else {
            return;
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Related Node ")
            .title_style(Style::default().add_modifier(Modifier::BOLD))
            .style(Style::default().bg(Color::Black));

        let kv_style = Style::default().fg(theme::COLOR_MUTED);
        let val_style = Style::default().fg(theme::COLOR_FG);

        let file_display = if n.file_path.is_empty() {
            "(stub — no source location)".to_string()
        } else {
            n.file_path.clone()
        };

        let lines = vec![
            Line::from(""),
            Line::from(vec![
                Span::styled("  Symbol : ", kv_style),
                Span::styled(
                    n.symbol_name.clone(),
                    val_style.add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Type   : ", kv_style),
                Span::styled(n.symbol_type.clone(), Style::default().fg(Color::Gray)),
            ]),
            Line::from(vec![
                Span::styled("  Edge   : ", kv_style),
                Span::styled(
                    n.edge_type.clone(),
                    Style::default().fg(theme::COLOR_ACCENT),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Depth  : ", kv_style),
                Span::styled(n.depth.to_string(), Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::styled("  File   : ", kv_style),
                Span::styled(file_display, Style::default().fg(Color::Gray)),
            ]),
            Line::from(""),
            Line::from(Span::styled(
                " Esc to close",
                Style::default().fg(theme::COLOR_DIM),
            )),
        ];

        let p = Paragraph::new(lines).block(block);
        frame.render_widget(p, area);
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Re-style every span in `line` to use `fg` as the foreground color,
/// preserving any background and modifier bits from the original style.
///
/// Used to highlight the matching line in the text preview without losing
/// any syntax-highlight structure the content renderer may have added.
fn restyle_line_fg(line: Line<'static>, fg: Color) -> Line<'static> {
    let spans: Vec<Span<'static>> = line
        .spans
        .into_iter()
        .map(|s| {
            let style = s.style.fg(fg);
            Span::styled(s.content, style)
        })
        .collect();
    Line::from(spans)
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use ratatui::style::Color;
    use ratatui::text::{Line, Span};

    use super::restyle_line_fg;

    #[test]
    fn restyle_line_fg_changes_foreground() {
        let original = Line::from(vec![Span::raw("hello")]);
        let restyled = restyle_line_fg(original, Color::Cyan);
        assert_eq!(restyled.spans.len(), 1);
        assert_eq!(restyled.spans[0].style.fg, Some(Color::Cyan));
    }

    #[test]
    fn restyle_line_fg_empty_line() {
        let original = Line::from(vec![]);
        let restyled = restyle_line_fg(original, Color::Red);
        assert!(restyled.spans.is_empty());
    }

    #[test]
    fn restyle_line_fg_preserves_content() {
        let original = Line::from(vec![Span::raw("fn main()")]);
        let restyled = restyle_line_fg(original, Color::Green);
        assert_eq!(restyled.spans[0].content, "fn main()");
    }

    #[test]
    fn restyle_line_fg_multiple_spans() {
        let original = Line::from(vec![Span::raw("a"), Span::raw("b"), Span::raw("c")]);
        let restyled = restyle_line_fg(original, Color::Yellow);
        assert_eq!(restyled.spans.len(), 3);
        for span in &restyled.spans {
            assert_eq!(span.style.fg, Some(Color::Yellow));
        }
    }
}

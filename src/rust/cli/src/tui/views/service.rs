//! Service view — live daemon/Qdrant health plus the key operational telemetry.
//!
//! Health is probed off-thread (see `service_data`); the render thread reads a
//! snapshot. SQLite-derived counters (queue depth, DLQ, indexed docs/chunks,
//! watcher state) refresh on each tick.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use super::service_data::{
    fetch_service_status, format_bytes, spawn_service_fetcher, ServiceLive, ServiceStatus,
};
use crate::tui::theme;

/// Minimum interval between SQLite status reads.
const REFRESH_INTERVAL_MS: u128 = 3000;

/// Service view state.
pub struct ServiceView {
    status: ServiceStatus,
    live: Arc<Mutex<ServiceLive>>,
    last_refresh: Option<Instant>,
    /// Last command result message (shown briefly).
    pub last_message: Option<String>,
}

impl ServiceView {
    pub fn new() -> Self {
        Self {
            status: ServiceStatus::default(),
            live: spawn_service_fetcher(),
            last_refresh: None,
            last_message: None,
        }
    }

    pub fn on_tick(&mut self) {
        let should_refresh = self
            .last_refresh
            .map_or(true, |t| t.elapsed().as_millis() >= REFRESH_INTERVAL_MS);

        if should_refresh {
            self.status = fetch_service_status();
            self.last_refresh = Some(Instant::now());
        }
    }

    /// Snapshot of the off-thread live signals.
    fn live(&self) -> ServiceLive {
        self.live.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Returns true if the daemon has been confirmed down (for alarm state).
    pub fn alarm_active(&self) -> bool {
        self.live().daemon_healthy == Some(false)
    }

    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let live = self.live();

        let rows = Layout::vertical([
            Constraint::Length(8), // daemon | qdrant
            Constraint::Length(8), // queue | index
            Constraint::Min(1),    // hints
        ])
        .split(area);

        let top = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[0]);
        let mid = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[1]);

        frame.render_widget(self.render_daemon_panel(&live), top[0]);
        frame.render_widget(self.render_qdrant_panel(&live), top[1]);
        frame.render_widget(self.render_queue_panel(), mid[0]);
        frame.render_widget(self.render_index_panel(), mid[1]);
        frame.render_widget(self.render_hints_panel(), rows[2]);
    }

    /// Render a tri-state health indicator span.
    fn health_indicator(healthy: Option<bool>) -> Span<'static> {
        match healthy {
            Some(true) => Span::styled(
                format!("{} Healthy", theme::GUTTER_SYNC),
                Style::default().fg(theme::COLOR_SUCCESS),
            ),
            Some(false) => Span::styled(
                format!("{} Unreachable", theme::GUTTER_REMOVE),
                Style::default().fg(theme::COLOR_ERROR),
            ),
            None => Span::styled("… probing", Style::default().fg(theme::COLOR_DIM)),
        }
    }

    fn render_daemon_panel(&self, live: &ServiceLive) -> Paragraph<'static> {
        let block_style = if live.daemon_healthy == Some(false) {
            theme::alarm_style()
        } else {
            Style::default()
        };
        let footprint = live
            .footprint_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "—".to_string());
        let lines = vec![
            kv("Status", Self::health_indicator(live.daemon_healthy)),
            kv(
                "Memory",
                Span::styled(footprint, Style::default().fg(theme::COLOR_ACCENT)),
            ),
            kv(
                "Schema",
                Span::raw(format!("v{}", self.status.schema_version)),
            ),
        ];
        panel(lines, " Daemon (memexd) ", block_style)
    }

    fn render_qdrant_panel(&self, live: &ServiceLive) -> Paragraph<'static> {
        let block_style = if live.qdrant_healthy == Some(false) {
            theme::alarm_style()
        } else {
            Style::default()
        };
        let lines = vec![
            kv("Status", Self::health_indicator(live.qdrant_healthy)),
            kv(
                "URL",
                Span::raw(crate::tui::util::truncate_path(&self.status.qdrant_url, 36)),
            ),
        ];
        panel(lines, " Qdrant ", block_style)
    }

    fn render_queue_panel(&self) -> Paragraph<'static> {
        let s = &self.status;
        let failed_fg = if s.queue_failed > 0 {
            theme::COLOR_ERROR
        } else {
            theme::COLOR_DIM
        };
        let dlq_fg = if s.dlq_count > 0 {
            theme::COLOR_ERROR
        } else {
            theme::COLOR_DIM
        };
        let lines = vec![
            kv(
                "Pending",
                Span::styled(
                    s.queue_pending.to_string(),
                    Style::default().fg(theme::COLOR_WARNING),
                ),
            ),
            kv(
                "In progress",
                Span::styled(
                    s.queue_in_progress.to_string(),
                    Style::default().fg(theme::COLOR_INFO),
                ),
            ),
            kv(
                "Failed",
                Span::styled(s.queue_failed.to_string(), Style::default().fg(failed_fg)),
            ),
            kv(
                "Dead-letter",
                Span::styled(s.dlq_count.to_string(), Style::default().fg(dlq_fg)),
            ),
        ];
        panel(lines, " Queue ", Style::default())
    }

    fn render_index_panel(&self) -> Paragraph<'static> {
        let s = &self.status;
        let paused_fg = if s.watchers_paused > 0 {
            theme::COLOR_WARNING
        } else {
            theme::COLOR_DIM
        };
        let lines = vec![
            kv(
                "Documents",
                Span::styled(
                    s.total_docs.to_string(),
                    Style::default().fg(theme::COLOR_ACCENT),
                ),
            ),
            kv(
                "Chunks",
                Span::styled(
                    s.total_chunks.to_string(),
                    Style::default().fg(theme::COLOR_ACCENT),
                ),
            ),
            kv(
                "Watchers",
                Span::styled(
                    format!("{} active", s.watchers_active),
                    Style::default().fg(theme::COLOR_SUCCESS),
                ),
            ),
            kv(
                "Paused",
                Span::styled(
                    s.watchers_paused.to_string(),
                    Style::default().fg(paused_fg),
                ),
            ),
        ];
        panel(lines, " Index ", Style::default())
    }

    fn render_hints_panel(&self) -> Paragraph<'static> {
        let mut spans: Vec<Span<'static>> = vec![
            Span::styled("  p ", Style::default().fg(theme::COLOR_ACCENT)),
            Span::styled("Pause watchers  ", Style::default().fg(theme::COLOR_DIM)),
            Span::styled("r ", Style::default().fg(theme::COLOR_ACCENT)),
            Span::styled("Resume watchers", Style::default().fg(theme::COLOR_DIM)),
        ];
        if let Some(ref msg) = self.last_message {
            spans.push(Span::styled(
                format!("  | {msg}"),
                Style::default().fg(theme::COLOR_WARNING),
            ));
        }
        Paragraph::new(Line::from(spans)).block(Block::default().borders(Borders::ALL))
    }
}

/// Build a key/value line: dimmed key, then the value span.
fn kv(key: &str, value: Span<'static>) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("  {key:<12} "),
            Style::default().fg(theme::COLOR_MUTED),
        ),
        value,
    ])
}

/// Wrap lines in a titled, bordered panel.
fn panel(lines: Vec<Line<'static>>, title: &str, block_style: Style) -> Paragraph<'static> {
    Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .title(title.to_string())
            .title_style(Style::default().add_modifier(Modifier::BOLD))
            .style(block_style),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_view_initializes() {
        let view = ServiceView::new();
        // No probe has completed yet, so the daemon is not yet confirmed down.
        assert!(!view.alarm_active());
        assert!(view.last_message.is_none());
    }

    #[test]
    fn health_indicator_states() {
        assert!(ServiceView::health_indicator(Some(true))
            .content
            .contains("Healthy"));
        assert!(ServiceView::health_indicator(Some(false))
            .content
            .contains("Unreachable"));
        assert!(ServiceView::health_indicator(None)
            .content
            .contains("probing"));
    }
}

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
    fetch_service_status, fetch_storage, format_bytes, spawn_service_fetcher, ServiceLive,
    ServiceStatus, StorageInfo,
};
use crate::tui::theme;
use crate::tui::util::fmt_count;

/// Minimum interval between SQLite status reads.
const REFRESH_INTERVAL_MS: u128 = 3000;

/// Service view state.
pub struct ServiceView {
    status: ServiceStatus,
    storage: StorageInfo,
    live: Arc<Mutex<ServiceLive>>,
    last_refresh: Option<Instant>,
    /// Last command result message (shown briefly).
    pub last_message: Option<String>,
}

impl ServiceView {
    pub fn new() -> Self {
        Self {
            status: ServiceStatus::default(),
            storage: StorageInfo::default(),
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
            self.storage = fetch_storage();
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
            Constraint::Length(8), // storage
            Constraint::Min(1),    // hints
        ])
        .split(area);

        let top = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[0]);
        let mid = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[1]);

        frame.render_widget(self.render_daemon_panel(&live), top[0]);
        frame.render_widget(self.render_qdrant_panel(&live), top[1]);
        frame.render_widget(self.render_queue_panel(&live), mid[0]);
        frame.render_widget(self.render_index_panel(), mid[1]);
        frame.render_widget(self.render_storage_panel(), rows[2]);
        frame.render_widget(self.render_hints_panel(), rows[3]);
    }

    /// Render the storage panel: each database file's size, the total, and the
    /// free space on the volume that holds the data directory.
    fn render_storage_panel(&self) -> Paragraph<'static> {
        let st = &self.storage;
        let mut lines: Vec<Line<'static>> = Vec::new();

        // Right-align the size column: pad the label to a fixed width, then the
        // size to a fixed width so the digits line up under one another.
        const LABEL_W: usize = 16;
        const SIZE_W: usize = 11;
        let size_line = |label: &str, size: String, style: Style| -> Line<'static> {
            Line::from(vec![
                Span::styled(
                    format!("  {label:<LABEL_W$}"),
                    Style::default().fg(theme::COLOR_MUTED),
                ),
                Span::styled(format!("{size:>SIZE_W$}"), style),
            ])
        };

        if st.db_files.is_empty() {
            lines.push(Line::from(Span::styled(
                "  no databases found",
                Style::default().fg(theme::COLOR_DIM),
            )));
        } else {
            for f in &st.db_files {
                lines.push(size_line(
                    &f.name,
                    format_bytes(f.size),
                    Style::default().fg(theme::COLOR_ACCENT),
                ));
            }
            lines.push(size_line(
                "Total",
                format_bytes(st.total_db_bytes),
                Style::default()
                    .fg(theme::COLOR_FG)
                    .add_modifier(Modifier::BOLD),
            ));
        }

        let free = st
            .free_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "—".to_string());
        lines.push(size_line(
            "Free",
            free,
            Style::default().fg(theme::COLOR_SUCCESS),
        ));

        let title = if st.data_dir.is_empty() {
            " Storage ".to_string()
        } else {
            format!(" Storage — {} ", st.data_dir)
        };
        panel(lines, &title, Style::default())
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

    /// Render the functional queue-health verdict (#133 F9). A NEW render path,
    /// deliberately NOT `health_indicator` (which is probe-liveness): verdict
    /// `Unknown` is the daemon's cold-start ("learning baseline"), distinct from
    /// the probe-pending "… probing" (UX-7). Each state has a glyph AND a word, so
    /// Amber/Red are distinguishable without color (UX-1).
    fn verdict_indicator(verdict: Option<crate::output::ServiceStatus>) -> Span<'static> {
        use crate::output::ServiceStatus as Vs;
        match verdict {
            Some(Vs::Healthy) | Some(Vs::Active) => Span::styled(
                format!("{} healthy", theme::GUTTER_SYNC),
                Style::default().fg(theme::COLOR_SUCCESS),
            ),
            Some(Vs::Degraded) => Span::styled(
                format!("{} degraded", theme::GUTTER_WARNING),
                Style::default().fg(theme::COLOR_WARNING),
            ),
            Some(Vs::Unhealthy) => Span::styled(
                format!("{} unhealthy", theme::GUTTER_REMOVE),
                Style::default().fg(theme::COLOR_ERROR),
            ),
            Some(Vs::Unknown) | Some(Vs::Inactive) | None => {
                Span::styled("… learning baseline", Style::default().fg(theme::COLOR_DIM))
            }
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

    fn render_queue_panel(&self, live: &ServiceLive) -> Paragraph<'static> {
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
        let mut lines = vec![
            kv("Health", Self::verdict_indicator(live.queue_verdict)),
            kv(
                "Pending",
                Span::styled(
                    fmt_count(s.queue_pending),
                    Style::default().fg(theme::COLOR_WARNING),
                ),
            ),
            kv(
                "In progress",
                Span::styled(
                    fmt_count(s.queue_in_progress),
                    Style::default().fg(theme::COLOR_INFO),
                ),
            ),
            kv(
                "Failed",
                Span::styled(fmt_count(s.queue_failed), Style::default().fg(failed_fg)),
            ),
            kv(
                "Dead-letter (DLQ)",
                Span::styled(fmt_count(s.dlq_count), Style::default().fg(dlq_fg)),
            ),
        ];
        // Per-line attributed remediation beneath the counts (clipped to the
        // panel height); each line is already `[<rag> <culprit>] <text>`.
        for line in &live.queue_remediation {
            lines.push(Line::from(Span::styled(
                format!("  {line}"),
                Style::default().fg(theme::COLOR_DIM),
            )));
        }
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
                    fmt_count(s.total_docs),
                    Style::default().fg(theme::COLOR_ACCENT),
                ),
            ),
            kv(
                "Chunks",
                Span::styled(
                    fmt_count(s.total_chunks),
                    Style::default().fg(theme::COLOR_ACCENT),
                ),
            ),
            kv(
                "Watchers",
                Span::styled(
                    format!("{} active", fmt_count(s.watchers_active)),
                    Style::default().fg(theme::COLOR_SUCCESS),
                ),
            ),
            kv(
                "Paused",
                Span::styled(fmt_count(s.watchers_paused), Style::default().fg(paused_fg)),
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

    #[test]
    fn verdict_indicator_states() {
        use crate::output::ServiceStatus as Vs;
        // Cold-start renders "learning baseline" via the NEW path, NOT the
        // probe-pending "probing" (#133 F9/UX-7).
        let cold = ServiceView::verdict_indicator(Some(Vs::Unknown));
        assert!(cold.content.contains("learning baseline"));
        assert!(!cold.content.contains("probing"));
        assert!(ServiceView::verdict_indicator(None)
            .content
            .contains("learning baseline"));
        // Amber vs Red distinguished by word + glyph, not color alone (UX-1).
        let amber = ServiceView::verdict_indicator(Some(Vs::Degraded));
        let red = ServiceView::verdict_indicator(Some(Vs::Unhealthy));
        assert!(amber.content.contains("degraded"));
        assert!(red.content.contains("unhealthy"));
        assert_ne!(amber.content, red.content);
        assert!(ServiceView::verdict_indicator(Some(Vs::Healthy))
            .content
            .contains("healthy"));
    }
}

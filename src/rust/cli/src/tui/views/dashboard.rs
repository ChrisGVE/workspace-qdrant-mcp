//! Dashboard view showing system status, queue summary, active projects,
//! and recent errors.
//!
//! Data is fetched from the local SQLite state database (read-only) and
//! refreshes on each tick event (~250ms) with a minimum 1-second interval.

use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use crate::commands::queue::db::connect_readonly;

/// Minimum interval between data refreshes to avoid excessive SQLite reads.
const REFRESH_INTERVAL_MS: u128 = 1000;

/// Maximum number of recent errors to display.
const MAX_RECENT_ERRORS: usize = 5;

/// Maximum number of active projects to display in the panel.
const MAX_DISPLAYED_PROJECTS: usize = 8;

/// Snapshot of dashboard data fetched from SQLite.
#[derive(Debug, Clone)]
pub struct DashboardData {
    /// Whether the SQLite database was reachable.
    pub db_connected: bool,
    /// Queue totals by status.
    pub queue_pending: i64,
    pub queue_in_progress: i64,
    pub queue_done: i64,
    pub queue_failed: i64,
    /// Active projects: (tenant_id, path).
    pub active_projects: Vec<(String, String)>,
    /// Recent error messages from the unified queue.
    pub recent_errors: Vec<String>,
}

impl Default for DashboardData {
    fn default() -> Self {
        Self {
            db_connected: false,
            queue_pending: 0,
            queue_in_progress: 0,
            queue_done: 0,
            queue_failed: 0,
            active_projects: Vec::new(),
            recent_errors: Vec::new(),
        }
    }
}

impl DashboardData {
    /// Total items across all queue statuses.
    pub fn queue_total(&self) -> i64 {
        self.queue_pending + self.queue_in_progress + self.queue_done + self.queue_failed
    }
}

/// Dashboard view state.
pub struct Dashboard {
    /// Current snapshot of data.
    data: DashboardData,
    /// When the data was last refreshed.
    last_refresh: Option<Instant>,
}

impl Dashboard {
    /// Create a new, empty dashboard. Data is loaded on the first tick.
    pub fn new() -> Self {
        Self {
            data: DashboardData::default(),
            last_refresh: None,
        }
    }

    /// Refresh data from SQLite if enough time has elapsed.
    pub fn on_tick(&mut self) {
        let should_refresh = self
            .last_refresh
            .map_or(true, |t| t.elapsed().as_millis() >= REFRESH_INTERVAL_MS);

        if should_refresh {
            self.data = fetch_dashboard_data();
            self.last_refresh = Some(Instant::now());
        }
    }

    /// Render the dashboard into the given area.
    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let rows = Layout::vertical([Constraint::Length(7), Constraint::Min(5)]).split(area);

        let top_cols = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[0]);

        let bottom_cols =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(rows[1]);

        self.draw_status_panel(frame, top_cols[0]);
        self.draw_queue_panel(frame, top_cols[1]);
        self.draw_projects_panel(frame, bottom_cols[0]);
        self.draw_errors_panel(frame, bottom_cols[1]);
    }

    fn draw_status_panel(&self, frame: &mut Frame, area: Rect) {
        let (status_text, status_color) = if self.data.db_connected {
            ("connected", Color::Green)
        } else {
            ("unavailable", Color::Red)
        };

        let active_count = self.data.active_projects.len();
        let total_queue = self.data.queue_total();

        let lines = vec![
            Line::from(vec![
                Span::styled("  Database:   ", Style::default().fg(Color::Gray)),
                Span::styled(status_text, Style::default().fg(status_color)),
            ]),
            Line::from(vec![
                Span::styled("  Projects:   ", Style::default().fg(Color::Gray)),
                Span::raw(active_count.to_string()),
            ]),
            Line::from(vec![
                Span::styled("  Queue:      ", Style::default().fg(Color::Gray)),
                Span::raw(format!("{} items", total_queue)),
            ]),
            Line::from(vec![
                Span::styled("  Failed:     ", Style::default().fg(Color::Gray)),
                failed_span(self.data.queue_failed),
            ]),
        ];

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Status ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        frame.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn draw_queue_panel(&self, frame: &mut Frame, area: Rect) {
        let d = &self.data;

        let lines = vec![
            Line::from(vec![
                Span::styled("  Total:       ", Style::default().fg(Color::Gray)),
                Span::raw(d.queue_total().to_string()),
            ]),
            Line::from(vec![
                Span::styled("  Pending:     ", Style::default().fg(Color::Gray)),
                Span::styled(
                    d.queue_pending.to_string(),
                    Style::default().fg(Color::Yellow),
                ),
            ]),
            Line::from(vec![
                Span::styled("  In Progress: ", Style::default().fg(Color::Gray)),
                Span::styled(
                    d.queue_in_progress.to_string(),
                    Style::default().fg(Color::Blue),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Done:        ", Style::default().fg(Color::Gray)),
                Span::styled(d.queue_done.to_string(), Style::default().fg(Color::Green)),
            ]),
            Line::from(vec![
                Span::styled("  Failed:      ", Style::default().fg(Color::Gray)),
                failed_span(d.queue_failed),
            ]),
        ];

        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Queue ")
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        frame.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn draw_projects_panel(&self, frame: &mut Frame, area: Rect) {
        let projects = &self.data.active_projects;

        let mut lines: Vec<Line> = if projects.is_empty() {
            vec![Line::from(Span::styled(
                "  No active projects",
                Style::default().fg(Color::DarkGray),
            ))]
        } else {
            projects
                .iter()
                .take(MAX_DISPLAYED_PROJECTS)
                .map(|(tenant_id, path)| {
                    let short_id = truncate_id(tenant_id);
                    let display_path = abbreviate_home(path);
                    Line::from(vec![
                        Span::styled(format!("  {} ", short_id), Style::default().fg(Color::Cyan)),
                        Span::raw(display_path),
                    ])
                })
                .collect()
        };

        if projects.len() > MAX_DISPLAYED_PROJECTS {
            lines.push(Line::from(Span::styled(
                format!("  ... and {} more", projects.len() - MAX_DISPLAYED_PROJECTS),
                Style::default().fg(Color::DarkGray),
            )));
        }

        let title = format!(" Active Projects ({}) ", projects.len());
        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        frame.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn draw_errors_panel(&self, frame: &mut Frame, area: Rect) {
        let errors = &self.data.recent_errors;

        let lines: Vec<Line> = if errors.is_empty() {
            vec![Line::from(Span::styled(
                "  No recent errors",
                Style::default().fg(Color::DarkGray),
            ))]
        } else {
            errors
                .iter()
                .map(|msg| {
                    let truncated = truncate_str(msg, 60);
                    Line::from(Span::styled(
                        format!("  {truncated}"),
                        Style::default().fg(Color::Red),
                    ))
                })
                .collect()
        };

        let title = if errors.is_empty() {
            " Recent Errors ".to_string()
        } else {
            format!(" Recent Errors ({}) ", errors.len())
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .title(title)
            .title_style(Style::default().add_modifier(Modifier::BOLD));

        frame.render_widget(Paragraph::new(lines).block(block), area);
    }
}

/// Fetch all dashboard data from the SQLite state database.
fn fetch_dashboard_data() -> DashboardData {
    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(_) => return DashboardData::default(),
    };

    let mut data = DashboardData {
        db_connected: true,
        ..DashboardData::default()
    };

    fetch_queue_stats(&conn, &mut data);
    fetch_active_projects(&conn, &mut data);
    fetch_recent_errors(&conn, &mut data);

    data
}

/// Populate queue status counts from `unified_queue`.
fn fetch_queue_stats(conn: &rusqlite::Connection, data: &mut DashboardData) {
    let Ok(mut stmt) = conn.prepare("SELECT status, COUNT(*) FROM unified_queue GROUP BY status")
    else {
        return;
    };

    let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    }) else {
        return;
    };

    for row in rows.flatten() {
        let (status, count) = row;
        match status.as_str() {
            "pending" => data.queue_pending = count,
            "in_progress" => data.queue_in_progress = count,
            "done" => data.queue_done = count,
            "failed" => data.queue_failed = count,
            _ => {}
        }
    }
}

/// Populate active projects from `watch_folders`.
fn fetch_active_projects(conn: &rusqlite::Connection, data: &mut DashboardData) {
    let Ok(mut stmt) = conn.prepare(
        "SELECT tenant_id, path FROM watch_folders \
         WHERE is_active > 0 AND collection = 'projects' \
         ORDER BY last_activity_at DESC",
    ) else {
        return;
    };

    let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) else {
        return;
    };

    data.active_projects = rows.flatten().collect();
}

/// Populate recent error messages from failed queue items.
fn fetch_recent_errors(conn: &rusqlite::Connection, data: &mut DashboardData) {
    let Ok(mut stmt) = conn.prepare(
        "SELECT error_message FROM unified_queue \
         WHERE status = 'failed' AND error_message IS NOT NULL \
         ORDER BY updated_at DESC LIMIT ?1",
    ) else {
        return;
    };

    let Ok(rows) = stmt.query_map([MAX_RECENT_ERRORS as i64], |row| row.get::<_, String>(0)) else {
        return;
    };

    data.recent_errors = rows.flatten().collect();
}

/// Return a styled span for the failed count: red if non-zero, green otherwise.
fn failed_span(count: i64) -> Span<'static> {
    if count > 0 {
        Span::styled(count.to_string(), Style::default().fg(Color::Red))
    } else {
        Span::styled(count.to_string(), Style::default().fg(Color::Green))
    }
}

/// Truncate a tenant ID to the first 8 characters followed by ellipsis.
fn truncate_id(id: &str) -> String {
    if id.len() <= 12 {
        id.to_string()
    } else {
        format!("{}...", &id[..8])
    }
}

/// Truncate a string to `max_len` characters, appending "..." if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

/// Replace the home directory prefix with `~` for compact display.
fn abbreviate_home(path: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        let home_str = home.to_string_lossy();
        if let Some(rest) = path.strip_prefix(home_str.as_ref()) {
            return format!("~{rest}");
        }
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dashboard_data_default_is_disconnected() {
        let data = DashboardData::default();
        assert!(!data.db_connected);
        assert_eq!(data.queue_total(), 0);
        assert!(data.active_projects.is_empty());
        assert!(data.recent_errors.is_empty());
    }

    #[test]
    fn dashboard_data_queue_total() {
        let data = DashboardData {
            db_connected: true,
            queue_pending: 10,
            queue_in_progress: 3,
            queue_done: 100,
            queue_failed: 2,
            active_projects: Vec::new(),
            recent_errors: Vec::new(),
        };
        assert_eq!(data.queue_total(), 115);
    }

    #[test]
    fn truncate_id_short() {
        assert_eq!(truncate_id("abc123"), "abc123");
    }

    #[test]
    fn truncate_id_long() {
        let long_id = "abcdef0123456789abcdef";
        let result = truncate_id(long_id);
        assert_eq!(result, "abcdef01...");
    }

    #[test]
    fn truncate_str_no_truncation() {
        assert_eq!(truncate_str("short", 10), "short");
    }

    #[test]
    fn truncate_str_with_truncation() {
        let long = "a".repeat(40);
        let result = truncate_str(&long, 10);
        assert!(result.ends_with("..."));
        assert!(result.chars().count() <= 10);
    }

    #[test]
    fn abbreviate_home_replaces_prefix() {
        if let Some(home) = dirs::home_dir() {
            let path = format!("{}/projects/test", home.display());
            let result = abbreviate_home(&path);
            assert!(result.starts_with("~/"));
            assert!(result.contains("projects/test"));
        }
    }

    #[test]
    fn abbreviate_home_no_match() {
        let path = "/opt/something";
        assert_eq!(abbreviate_home(path), path);
    }

    #[test]
    fn failed_span_zero_is_green() {
        let span = failed_span(0);
        assert_eq!(span.style.fg, Some(Color::Green));
    }

    #[test]
    fn failed_span_nonzero_is_red() {
        let span = failed_span(5);
        assert_eq!(span.style.fg, Some(Color::Red));
    }

    #[test]
    fn dashboard_new_starts_without_data() {
        let dash = Dashboard::new();
        assert!(!dash.data.db_connected);
        assert!(dash.last_refresh.is_none());
    }
}

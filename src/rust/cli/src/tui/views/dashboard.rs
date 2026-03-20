//! Dashboard view — 2x4 interactive grid with live data.
//!
//! Row 1 (fixed 3 lines): Services health + Queue status
//! Rows 2-4 (scrollable): Projects, Libraries, Scratchpad, Rules,
//!                         Active Projects, Last Errors

use std::sync::{Arc, Mutex};
use std::time::Instant;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use super::dashboard_cells::ScrollableCell;
use super::dashboard_data::{
    fetch_dashboard_data, merge_async_data, spawn_async_fetcher, AsyncDashboardData, DashboardData,
    ServiceHealth,
};
use super::dashboard_grid;

const REFRESH_INTERVAL_MS: u128 = 1000;

/// Which cell is currently focused (rows 2-4 only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusedCell {
    None,
    Projects,
    Libraries,
    Scratchpad,
    Rules,
    ActiveProjects,
    Errors,
}

/// Dashboard view state.
pub struct Dashboard {
    data: DashboardData,
    async_shared: Arc<Mutex<AsyncDashboardData>>,
    async_snapshot: AsyncDashboardData,
    last_refresh: Option<Instant>,
    pub focused: FocusedCell,
    // Scroll state per cell
    pub cell_projects: ScrollableCell,
    pub cell_libraries: ScrollableCell,
    pub cell_scratchpad: ScrollableCell,
    pub cell_rules: ScrollableCell,
    pub cell_active: ScrollableCell,
    pub cell_errors: ScrollableCell,
}

impl Dashboard {
    pub fn new() -> Self {
        Self {
            data: DashboardData::default(),
            async_shared: spawn_async_fetcher(),
            async_snapshot: AsyncDashboardData::default(),
            last_refresh: None,
            focused: FocusedCell::None,
            cell_projects: ScrollableCell::new(),
            cell_libraries: ScrollableCell::new(),
            cell_scratchpad: ScrollableCell::new(),
            cell_rules: ScrollableCell::new(),
            cell_active: ScrollableCell::new(),
            cell_errors: ScrollableCell::new(),
        }
    }

    pub fn on_tick(&mut self) {
        let should_refresh = self
            .last_refresh
            .map_or(true, |t| t.elapsed().as_millis() >= REFRESH_INTERVAL_MS);

        if should_refresh {
            self.data = fetch_dashboard_data();

            // Read async data (non-blocking)
            if let Ok(guard) = self.async_shared.try_lock() {
                self.async_snapshot = guard.clone();
            }
            merge_async_data(&mut self.data, &self.async_snapshot);

            self.last_refresh = Some(Instant::now());
        }
    }

    /// Navigate down in the focused cell.
    pub fn select_next(&mut self) {
        let count = self.focused_row_count();
        match self.focused {
            FocusedCell::Projects => self.cell_projects.select_next(count),
            FocusedCell::Libraries => self.cell_libraries.select_next(count),
            FocusedCell::Scratchpad => self.cell_scratchpad.select_next(count),
            FocusedCell::Rules => self.cell_rules.select_next(count),
            FocusedCell::ActiveProjects => self.cell_active.select_next(count),
            FocusedCell::Errors => self.cell_errors.select_next(count),
            FocusedCell::None => {}
        }
    }

    /// Navigate up in the focused cell.
    pub fn select_prev(&mut self) {
        match self.focused {
            FocusedCell::Projects => self.cell_projects.select_prev(),
            FocusedCell::Libraries => self.cell_libraries.select_prev(),
            FocusedCell::Scratchpad => self.cell_scratchpad.select_prev(),
            FocusedCell::Rules => self.cell_rules.select_prev(),
            FocusedCell::ActiveProjects => self.cell_active.select_prev(),
            FocusedCell::Errors => self.cell_errors.select_prev(),
            FocusedCell::None => {}
        }
    }

    /// Row count for the currently focused cell.
    fn focused_row_count(&self) -> usize {
        match self.focused {
            FocusedCell::Projects => self.data.projects.len(),
            FocusedCell::Libraries => self.data.libraries.len(),
            FocusedCell::Scratchpad => self.data.scratchpad.len(),
            FocusedCell::Rules => self.data.rules.len(),
            FocusedCell::ActiveProjects => self.data.active_projects.len(),
            FocusedCell::Errors => self.data.errors.len(),
            FocusedCell::None => 0,
        }
    }

    /// Whether a popup is currently open.
    pub fn popup_open(&self) -> bool {
        false // TODO: Phase 3
    }

    pub fn close_popup(&mut self) {
        // TODO: Phase 3
    }

    /// Get the selected row's tenant_id for popup opening.
    pub fn selected_tenant(&self) -> Option<String> {
        match self.focused {
            FocusedCell::Projects => self
                .data
                .projects
                .get(self.cell_projects.selected)
                .map(|r| r.tenant_id.clone()),
            FocusedCell::Libraries => self
                .data
                .libraries
                .get(self.cell_libraries.selected)
                .map(|r| r.tenant_id.clone()),
            FocusedCell::Scratchpad => self
                .data
                .scratchpad
                .get(self.cell_scratchpad.selected)
                .map(|r| r.tenant_id.clone()),
            FocusedCell::Rules => self
                .data
                .rules
                .get(self.cell_rules.selected)
                .map(|r| r.tenant_id.clone()),
            FocusedCell::ActiveProjects => self
                .data
                .active_projects
                .get(self.cell_active.selected)
                .map(|r| r.tenant_id.clone()),
            FocusedCell::Errors => self
                .data
                .errors
                .get(self.cell_errors.selected)
                .map(|r| r.queue_id.clone()),
            FocusedCell::None => None,
        }
    }

    // ------------------------------------------------------------------
    // Drawing
    // ------------------------------------------------------------------

    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        // 4 rows: top fixed (3 lines), then 3 flexible rows
        let rows = Layout::vertical([
            Constraint::Length(3),
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
        ])
        .split(area);

        // Each row splits into 2 columns
        let top = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[0]);
        let row2 = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[1]);
        let row3 = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[2]);
        let row4 = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[3]);

        // Row 1: fixed status panels
        self.draw_services(frame, top[0]);
        self.draw_queue_status(frame, top[1]);

        // Row 2: Projects + Libraries
        self.draw_projects_cell(frame, row2[0]);
        self.draw_libraries_cell(frame, row2[1]);

        // Row 3: Scratchpad + Rules
        self.draw_scratchpad_cell(frame, row3[0]);
        self.draw_rules_cell(frame, row3[1]);

        // Row 4: Active Projects + Last Errors
        self.draw_active_cell(frame, row4[0]);
        self.draw_errors_cell(frame, row4[1]);
    }

    // ------------------------------------------------------------------
    // Row 1: Services (1,1)
    // ------------------------------------------------------------------

    fn draw_services(&self, frame: &mut Frame, area: Rect) {
        let health = &self.async_snapshot.health;
        let (circle, circle_color) = services_indicator(health);

        let lines = vec![
            Line::from(vec![
                Span::styled(format!(" {} ", circle), Style::default().fg(circle_color)),
                Span::styled(
                    "Services",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::raw("   "),
                health_span("qdrant", health.qdrant_healthy),
                Span::raw("  "),
                health_span("memexd", health.daemon_healthy),
            ]),
        ];

        frame.render_widget(Paragraph::new(lines), area);
    }

    // ------------------------------------------------------------------
    // Row 1: Queue Status (2,1)
    // ------------------------------------------------------------------

    fn draw_queue_status(&self, frame: &mut Frame, area: Rect) {
        let d = &self.data;
        let total = d.queue_pending + d.queue_in_progress + d.queue_failed;

        let lines = vec![
            Line::from(vec![
                Span::styled(
                    " Queue Status ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("({})", total),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::raw("   Pending: "),
                Span::styled(
                    d.queue_pending.to_string(),
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw("  In Progress: "),
                Span::styled(
                    d.queue_in_progress.to_string(),
                    Style::default().fg(Color::Blue),
                ),
                Span::raw("  Failed: "),
                Span::styled(d.queue_failed.to_string(), Style::default().fg(Color::Red)),
            ]),
        ];

        frame.render_widget(Paragraph::new(lines), area);
    }

    // ------------------------------------------------------------------
    // Rows 2-4: delegate to dashboard_grid module
    // ------------------------------------------------------------------

    fn draw_projects_cell(&self, frame: &mut Frame, area: Rect) {
        dashboard_grid::draw_projects(frame, area, &self.data, &self.cell_projects, self.focused);
    }

    fn draw_libraries_cell(&self, frame: &mut Frame, area: Rect) {
        dashboard_grid::draw_libraries(frame, area, &self.data, &self.cell_libraries, self.focused);
    }

    fn draw_scratchpad_cell(&self, frame: &mut Frame, area: Rect) {
        dashboard_grid::draw_scratchpad(
            frame,
            area,
            &self.data,
            &self.cell_scratchpad,
            self.focused,
        );
    }

    fn draw_rules_cell(&self, frame: &mut Frame, area: Rect) {
        dashboard_grid::draw_rules(frame, area, &self.data, &self.cell_rules, self.focused);
    }

    fn draw_active_cell(&self, frame: &mut Frame, area: Rect) {
        dashboard_grid::draw_active_projects(
            frame,
            area,
            &self.data,
            &self.cell_active,
            self.focused,
        );
    }

    fn draw_errors_cell(&self, frame: &mut Frame, area: Rect) {
        dashboard_grid::draw_errors(frame, area, &self.data, &self.cell_errors, self.focused);
    }
}

// ---------------------------------------------------------------------------
// Health display helpers
// ---------------------------------------------------------------------------

fn services_indicator(health: &ServiceHealth) -> (&'static str, Color) {
    match (health.qdrant_healthy, health.daemon_healthy) {
        (Some(true), Some(true)) => ("●", Color::Green),
        (Some(false), Some(false)) | (None, None) => ("●", Color::Red),
        _ => {
            // At least one unhealthy or unknown
            let any_down =
                health.qdrant_healthy == Some(false) || health.daemon_healthy == Some(false);
            if any_down {
                ("●", Color::Yellow)
            } else {
                ("●", Color::DarkGray) // still checking
            }
        }
    }
}

fn health_span<'a>(name: &'a str, healthy: Option<bool>) -> Span<'a> {
    let (indicator, color) = match healthy {
        Some(true) => ("●", Color::Green),
        Some(false) => ("●", Color::Red),
        None => ("○", Color::DarkGray),
    };
    Span::styled(
        format!("{} {}", indicator, name),
        Style::default().fg(color),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focused_cell_default_is_none() {
        let dash = Dashboard::new();
        assert_eq!(dash.focused, FocusedCell::None);
    }

    #[test]
    fn services_indicator_both_healthy() {
        let h = ServiceHealth {
            qdrant_healthy: Some(true),
            daemon_healthy: Some(true),
        };
        let (_, color) = services_indicator(&h);
        assert_eq!(color, Color::Green);
    }

    #[test]
    fn services_indicator_both_down() {
        let h = ServiceHealth {
            qdrant_healthy: Some(false),
            daemon_healthy: Some(false),
        };
        let (_, color) = services_indicator(&h);
        assert_eq!(color, Color::Red);
    }

    #[test]
    fn services_indicator_one_down() {
        let h = ServiceHealth {
            qdrant_healthy: Some(true),
            daemon_healthy: Some(false),
        };
        let (_, color) = services_indicator(&h);
        assert_eq!(color, Color::Yellow);
    }

    #[test]
    fn services_indicator_unknown() {
        let h = ServiceHealth {
            qdrant_healthy: None,
            daemon_healthy: None,
        };
        let (_, color) = services_indicator(&h);
        assert_eq!(color, Color::Red);
    }

    #[test]
    fn select_next_increments() {
        let mut dash = Dashboard::new();
        dash.focused = FocusedCell::Errors;
        // No data, should not panic
        dash.select_next();
        assert_eq!(dash.cell_errors.selected, 0);
    }

    #[test]
    fn select_prev_decrements() {
        let mut dash = Dashboard::new();
        dash.focused = FocusedCell::Projects;
        dash.cell_projects.selected = 2;
        dash.select_prev();
        assert_eq!(dash.cell_projects.selected, 1);
    }
}

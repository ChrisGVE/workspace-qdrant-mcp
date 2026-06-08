//! Dashboard grid cell drawing and column definitions.
//!
//! Each function draws one cell of the 2x4 dashboard grid.

use ratatui::layout::Rect;
use ratatui::Frame;

use super::dashboard::FocusedCell;
use super::dashboard_cells::{draw_cell, queue_cell, Align, CellRow, CellValue, ColDef};
use super::dashboard_data::DashboardData;
use crate::tui::util::fmt_count;

/// Column definitions for the Projects cell (1,2).
pub fn projects_cols() -> &'static [ColDef] {
    &[
        ColDef {
            header: "Name",
            align: Align::Left,
            flex: true,
        },
        ColDef {
            header: "Bch",
            align: Align::Right,
            flex: false,
        },
        ColDef {
            header: "Pts",
            align: Align::Right,
            flex: false,
        },
        ColDef {
            header: "Files",
            align: Align::Right,
            flex: false,
        },
        ColDef {
            header: "Queue",
            align: Align::Right,
            flex: false,
        },
    ]
}

pub fn libraries_cols() -> &'static [ColDef] {
    &[
        ColDef {
            header: "Name",
            align: Align::Left,
            flex: true,
        },
        ColDef {
            header: "Pts",
            align: Align::Right,
            flex: false,
        },
        ColDef {
            header: "Files",
            align: Align::Right,
            flex: false,
        },
        ColDef {
            header: "Queue",
            align: Align::Right,
            flex: false,
        },
        ColDef {
            header: "Sync",
            align: Align::Left,
            flex: false,
        },
    ]
}

pub fn scratchpad_cols() -> &'static [ColDef] {
    &[
        ColDef {
            header: "Scope",
            align: Align::Left,
            flex: true,
        },
        ColDef {
            header: "Notes",
            align: Align::Right,
            flex: false,
        },
        ColDef {
            header: "Queue",
            align: Align::Right,
            flex: false,
        },
    ]
}

pub fn rules_cols() -> &'static [ColDef] {
    &[
        ColDef {
            header: "Scope",
            align: Align::Left,
            flex: true,
        },
        ColDef {
            header: "Rules",
            align: Align::Right,
            flex: false,
        },
        ColDef {
            header: "Queue",
            align: Align::Right,
            flex: false,
        },
    ]
}

pub fn active_cols() -> &'static [ColDef] {
    &[
        ColDef {
            header: "Name",
            align: Align::Left,
            flex: true,
        },
        ColDef {
            header: "Branch",
            align: Align::Left,
            flex: true,
        },
        ColDef {
            header: "Files",
            align: Align::Right,
            flex: false,
        },
        ColDef {
            header: "Queue",
            align: Align::Right,
            flex: false,
        },
    ]
}

pub fn error_cols() -> &'static [ColDef] {
    &[
        ColDef {
            header: "Collection",
            align: Align::Left,
            flex: false,
        },
        ColDef {
            header: "Error",
            align: Align::Left,
            flex: true,
        },
    ]
}

// ---------------------------------------------------------------------------
// Cell draw functions
// ---------------------------------------------------------------------------

use super::dashboard_cells::ScrollableCell;

pub fn draw_projects(
    frame: &mut Frame,
    area: Rect,
    data: &DashboardData,
    cell: &ScrollableCell,
    focused: FocusedCell,
) {
    let rows: Vec<CellRow> = data
        .projects
        .iter()
        .map(|p| {
            vec![
                CellValue::Plain(p.name.clone()),
                CellValue::Plain(fmt_count(p.branch_count)),
                CellValue::Plain(fmt_count(p.qdrant_points as i64)),
                CellValue::Plain(fmt_count(p.tracked_files)),
                queue_cell(p.queue_pending, p.queue_in_progress, p.queue_failed),
            ]
        })
        .collect();

    draw_cell(
        frame,
        area,
        "Projects",
        Some(data.projects.len()),
        Some('P'),
        projects_cols(),
        &rows,
        cell,
        focused == FocusedCell::Projects,
    );
}

pub fn draw_libraries(
    frame: &mut Frame,
    area: Rect,
    data: &DashboardData,
    cell: &ScrollableCell,
    focused: FocusedCell,
) {
    let rows: Vec<CellRow> = data
        .libraries
        .iter()
        .map(|l| {
            vec![
                CellValue::Plain(l.name.clone()),
                CellValue::Plain(fmt_count(l.qdrant_points as i64)),
                CellValue::Plain(fmt_count(l.tracked_files)),
                queue_cell(l.queue_pending, l.queue_in_progress, l.queue_failed),
                CellValue::Plain(l.sync_mode.clone()),
            ]
        })
        .collect();

    draw_cell(
        frame,
        area,
        "Libraries",
        Some(data.libraries.len()),
        Some('L'),
        libraries_cols(),
        &rows,
        cell,
        focused == FocusedCell::Libraries,
    );
}

pub fn draw_scratchpad(
    frame: &mut Frame,
    area: Rect,
    data: &DashboardData,
    cell: &ScrollableCell,
    focused: FocusedCell,
) {
    let rows: Vec<CellRow> = data
        .scratchpad
        .iter()
        .map(|s| {
            vec![
                CellValue::Plain(s.name.clone()),
                CellValue::Plain(fmt_count(s.note_count as i64)),
                queue_cell(s.queue_pending, s.queue_in_progress, s.queue_failed),
            ]
        })
        .collect();

    draw_cell(
        frame,
        area,
        "Scratchpad",
        Some(data.scratchpad.len()),
        Some('S'),
        scratchpad_cols(),
        &rows,
        cell,
        focused == FocusedCell::Scratchpad,
    );
}

pub fn draw_rules(
    frame: &mut Frame,
    area: Rect,
    data: &DashboardData,
    cell: &ScrollableCell,
    focused: FocusedCell,
) {
    let rows: Vec<CellRow> = data
        .rules
        .iter()
        .map(|r| {
            vec![
                CellValue::Plain(r.name.clone()),
                CellValue::Plain(fmt_count(r.rule_count as i64)),
                queue_cell(r.queue_pending, r.queue_in_progress, r.queue_failed),
            ]
        })
        .collect();

    draw_cell(
        frame,
        area,
        "Rules",
        Some(data.rules.len()),
        Some('R'),
        rules_cols(),
        &rows,
        cell,
        focused == FocusedCell::Rules,
    );
}

pub fn draw_active_projects(
    frame: &mut Frame,
    area: Rect,
    data: &DashboardData,
    cell: &ScrollableCell,
    focused: FocusedCell,
) {
    let rows: Vec<CellRow> = data
        .active_projects
        .iter()
        .map(|a| {
            vec![
                CellValue::Plain(a.name.clone()),
                CellValue::Plain(a.branch.clone()),
                CellValue::Plain(fmt_count(a.tracked_files)),
                queue_cell(a.queue_pending, a.queue_in_progress, a.queue_failed),
            ]
        })
        .collect();

    draw_cell(
        frame,
        area,
        "Active Projects",
        Some(data.active_projects.len()),
        Some('A'),
        active_cols(),
        &rows,
        cell,
        focused == FocusedCell::ActiveProjects,
    );
}

pub fn draw_errors(
    frame: &mut Frame,
    area: Rect,
    data: &DashboardData,
    cell: &ScrollableCell,
    focused: FocusedCell,
) {
    let rows: Vec<CellRow> = data
        .errors
        .iter()
        .map(|e| {
            let tenant_label = format!("[{}] {}", e.collection_letter, e.tenant_name);
            vec![
                CellValue::Plain(tenant_label),
                CellValue::Plain(e.error_message.clone()),
            ]
        })
        .collect();

    draw_cell(
        frame,
        area,
        "Last Errors",
        None,
        Some('E'),
        error_cols(),
        &rows,
        cell,
        focused == FocusedCell::Errors,
    );
}

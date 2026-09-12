//! Width allocation shared by Dashboard cell tables and the Queue list.
//!
//! The fit is computed from every row, so scrolling cannot move a column.

use ratatui::layout::{Constraint, Rect};

use super::table::{Align, Column};
use super::value::Cell;

pub(crate) const DEFAULT_MIN_FLEX: u16 = 12;

/// The columns that survive, their widths in that order, and their uniform gap.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Fit {
    pub active: Vec<usize>,
    pub widths: Vec<u16>,
    pub gap: u16,
}

/// Spend width in rank order: floors and one-cell gaps, level truncated columns
/// together to their natural widths, then afford two-cell gaps everywhere, then
/// divide surplus by each column's natural-minus-floor demand. A remainder goes
/// to the flex column. If the flex stays below `min_flex`, drop the lowest-priority
/// fixed column (rightmost on a tie) and start again. Widths grow monotonically
/// except when the uniform gap flips from one to two: the flex can give back up
/// to one cell per gap at that threshold.
///
/// `first_data_column` is zero for a cell table and one for the Queue, whose
/// untitled row-number column is generated at render time rather than stored.
pub(crate) fn fit(
    columns: &[Column],
    rows: &[Vec<Cell>],
    width: u16,
    min_flex: u16,
    first_data_column: usize,
) -> Fit {
    let flex = columns.iter().position(|column| column.fixed().is_none());
    let natural: Vec<u16> = columns
        .iter()
        .enumerate()
        .map(|(at, column)| natural_width(column, rows, at.checked_sub(first_data_column)))
        .collect();
    let floors: Vec<u16> = columns
        .iter()
        .enumerate()
        .map(|(at, column)| {
            let figures = column.align == Align::Right
                || at.checked_sub(first_data_column).is_some_and(|cell_at| {
                    rows.iter()
                        .filter_map(|row| row.get(cell_at))
                        .any(Cell::is_figure)
                });
            let floor = if figures {
                natural[at]
            } else {
                natural[at].min(column.title.chars().count() as u16)
            };
            if Some(at) == flex {
                floor.max(min_flex)
            } else {
                floor
            }
        })
        .collect();
    let mut active: Vec<usize> = (0..columns.len()).collect();
    loop {
        let fitted = spend(&active, &natural, &floors, width, flex);
        let over_budget = fitted.widths.iter().map(|&w| w as usize).sum::<usize>()
            + fitted.active.len().saturating_sub(1) * fitted.gap as usize
            > width as usize;
        let too_narrow = flex.is_some()
            && (over_budget
                || flex.is_some_and(|at| {
                    fitted
                        .active
                        .iter()
                        .position(|&column| column == at)
                        .is_some_and(|position| fitted.widths[position] < min_flex)
                }));
        if !too_narrow || active.len() <= 1 {
            return fitted;
        }
        let drop = active
            .iter()
            .copied()
            .filter(|&at| Some(at) != flex)
            .min_by_key(|&at| (columns[at].priority, std::cmp::Reverse(at)))
            .expect("a fixed column remains to drop");
        active.retain(|&at| at != drop);
    }
}

fn natural_width(column: &Column, rows: &[Vec<Cell>], cell_at: Option<usize>) -> u16 {
    if let Constraint::Length(width) = column.width {
        return width;
    }
    let values = cell_at
        .into_iter()
        .flat_map(|at| rows.iter().filter_map(move |row| row.get(at)))
        .map(Cell::natural_width)
        .max()
        .unwrap_or(0);
    values
        .max(column.title.chars().count())
        .min(u16::MAX as usize) as u16
}

fn spend(
    active: &[usize],
    natural: &[u16],
    floors: &[u16],
    width: u16,
    flex: Option<usize>,
) -> Fit {
    let mut widths: Vec<u16> = active.iter().map(|&at| floors[at]).collect();
    let gaps = active.len().saturating_sub(1);
    let floor_need = widths.iter().map(|&w| w as usize).sum::<usize>() + gaps;
    let mut remaining = (width as usize).saturating_sub(floor_need);
    while remaining > 0 {
        let truncated: Vec<usize> = active
            .iter()
            .enumerate()
            .filter_map(|(position, &at)| (widths[position] < natural[at]).then_some(position))
            .collect();
        if truncated.is_empty() {
            break;
        }
        for position in truncated {
            if remaining == 0 {
                break;
            }
            widths[position] += 1;
            remaining -= 1;
        }
    }
    let all_natural = active
        .iter()
        .enumerate()
        .all(|(position, &at)| widths[position] >= natural[at]);
    let gap = if all_natural && remaining >= gaps && gaps > 0 {
        remaining -= gaps;
        2
    } else {
        1
    };
    if all_natural && gap == 1 {
        // A partial set of wide gaps is never drawn.
        if let Some(position) = active.iter().position(|&at| Some(at) == flex) {
            widths[position] += remaining as u16;
        }
    } else if all_natural && flex.is_some() {
        distribute_surplus(active, natural, floors, &mut widths, remaining, flex);
    }
    Fit {
        active: active.to_vec(),
        widths,
        gap,
    }
}

fn distribute_surplus(
    active: &[usize],
    natural: &[u16],
    floors: &[u16],
    widths: &mut [u16],
    remaining: usize,
    flex: Option<usize>,
) {
    let demands: Vec<usize> = active
        .iter()
        .map(|&at| natural[at].saturating_sub(floors[at]) as usize)
        .collect();
    let total: usize = demands.iter().sum();
    let mut assigned = 0;
    for (position, demand) in demands.iter().enumerate() {
        if let Some(extra) = (remaining * demand).checked_div(total) {
            widths[position] += extra as u16;
            assigned += extra;
        }
    }
    if let Some(position) = active.iter().position(|&at| Some(at) == flex) {
        widths[position] += (remaining - assigned) as u16;
    }
}

/// Place fitted columns without letting a rect cross the table's right edge.
pub(crate) fn laid_out(area: Rect, fit: &Fit) -> Vec<Rect> {
    let mut x = area.x;
    let right = area.x.saturating_add(area.width);
    fit.widths
        .iter()
        .map(|&width| {
            let rect = Rect {
                x,
                width: width.min(right.saturating_sub(x)),
                ..area
            };
            x = x.saturating_add(width).saturating_add(fit.gap).min(right);
            rect
        })
        .collect()
}

#[cfg(test)]
mod tests;

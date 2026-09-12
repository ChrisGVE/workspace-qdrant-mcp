//! Isolated cases for the shared affordability fit.

use super::*;

fn text(value: &str) -> Vec<Vec<Cell>> {
    vec![vec![Cell::Text(value.to_string())]]
}

fn shape(columns: &[Column], rows: &[Vec<Cell>], width: u16) -> (Vec<u16>, u16) {
    let result = fit(columns, rows, width, 0, 0);
    (result.widths, result.gap)
}

#[test]
fn floors_only() {
    let columns = [Column::flex("A"), Column::number("N", 3)];
    assert_eq!(shape(&columns, &text("abcdef"), 5), (vec![1, 3], 1));
}

#[test]
fn truncated_text_columns_level_from_their_floors_left_to_right() {
    let columns = [
        Column::text("A", 4),
        Column::text("BB", 6),
        Column::text("CCC", 9),
    ];
    assert_eq!(shape(&columns, &[], 15), (vec![4, 4, 5], 1));
}

#[test]
fn gaps_flip_together_only_after_every_column_is_natural() {
    let columns = [Column::flex("Name"), Column::text("B", 3)];
    let rows = text("abcdefgh");
    assert_eq!(shape(&columns, &rows, 12), (vec![8, 3], 1));
    assert_eq!(shape(&columns, &rows, 13), (vec![8, 3], 2));
}

#[test]
fn surplus_is_proportional_and_rounding_goes_to_flex() {
    let columns = [Column::flex("Name"), Column::text("B", 3)];
    let rows = text("abcdefgh");
    assert_eq!(shape(&columns, &rows, 20), (vec![13, 5], 2));
}

#[test]
fn a_table_without_flex_leaves_surplus_as_right_padding() {
    let columns = [Column::text("A", 4), Column::number("N", 3)];
    assert_eq!(shape(&columns, &[], 20), (vec![4, 3], 2));
}

#[test]
fn the_lowest_priority_fixed_column_drops_then_fit_restarts() {
    let columns = [
        Column::flex("Name"),
        Column::text("Branch", 10).priority(2),
        Column::number("Files", 5).priority(1),
        Column::number("Queue", 8).priority(0),
    ];
    let result = fit(&columns, &[], 17, 12, 0);
    assert_eq!(result.active, vec![0]);
    assert_eq!(result.widths, vec![17]);
}

#[test]
fn flex_natural_width_uses_every_row_even_when_only_one_is_visible() {
    let columns = [Column::flex("Name"), Column::number("N", 1)];
    let rows = vec![
        vec![Cell::Text("short".into()), Cell::Num(1)],
        vec![Cell::Text("a much longer name".into()), Cell::Num(2)],
    ];
    assert_eq!(shape(&columns, &rows, 20), (vec![18, 1], 1));
    assert_eq!(shape(&columns, &rows, 21), (vec![18, 1], 2));
}

#[test]
fn a_figure_value_prevents_truncation_even_in_a_left_aligned_column() {
    let columns = [Column::text("Value", 6), Column::flex("Name")];
    let rows = vec![vec![Cell::Num(123_456), Cell::Text("name".into())]];
    let fitted = fit(&columns, &rows, 18, 12, 0);
    assert_eq!(fitted.active, vec![1]);
    assert_eq!(fitted.widths, vec![18]);
}

#[test]
fn a_flex_figure_keeps_its_natural_width() {
    let mut figure = Column::flex("N");
    figure.align = Align::Right;
    let columns = [figure, Column::text("T", 1)];
    let rows = vec![vec![Cell::Num(123_456), Cell::Text("T".into())]];
    let fitted = fit(&columns, &rows, 9, 0, 0);
    assert_eq!(fitted.active, vec![0, 1]);
    assert_eq!(fitted.widths, vec![7, 1]);
}

#[test]
fn partial_wide_gap_budget_goes_to_flex_instead() {
    let columns = [
        Column::flex("Name"),
        Column::number("A", 1),
        Column::number("B", 1),
    ];
    let rows = text("Name");
    assert_eq!(shape(&columns, &rows, 9), (vec![5, 1, 1], 1));
}

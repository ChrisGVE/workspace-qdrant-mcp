//! Header rendering and focus guards for a Dashboard cell table.

use super::*;

/// A column header is WHITE, UPRIGHT and underlined from end to end, and the data under it is a
/// light grey (Chris, 20260912, rulings 1 and 2).
///
/// This replaces the 20260907 rule outright — same rung as the data, distinguished by italics —
/// which was tried and rejected: *"italic wasn't a good idea"*. Three things are pinned because
/// the failure of any one of them leaves a header that looks nearly right: the rung, the absence
/// of the slant, and the rule running under the GAPS as well as under the names. The last is why
/// this reads every column of the row rather than one painted cell: an underline applied per
/// span would pass a guard that only looked at letters.
#[test]
fn a_column_header_is_white_and_upright_over_a_light_grey_body() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let sample = CellTable::new(columns(), rows(1));
    let buf = render(CellPane::new("Projects", Some(1), sample), WIDE, TALL);

    // The header is the row under the heading, so line 1 — and every painted cell on it.
    let mut painted = 0;
    for x in 0..WIDE {
        let cell = buf.cell((x, 1)).expect("cell in area");
        if cell.symbol().trim().is_empty() {
            continue;
        }
        assert!(
            !cell.style().add_modifier.contains(Modifier::ITALIC),
            "column {x} ({:?}) of the header is still italic",
            cell.symbol()
        );
        assert!(
            !cell.style().add_modifier.contains(Modifier::BOLD),
            "column {x} ({:?}) of the header is bold — a header is not a heading",
            cell.symbol()
        );
        assert_eq!(
            cell.style().fg,
            Some(tokens::header()),
            "column {x} ({:?}) of the header is not white",
            cell.symbol()
        );
        painted += 1;
    }
    assert!(painted > 0, "the header drew nothing at all");

    // The rule runs under the whole row — the gaps between the names as well, which is the half
    // an underline applied per span would fail.
    let table = buf.cell((0, 1)).expect("cell in area").style();
    assert!(
        table.add_modifier.contains(Modifier::UNDERLINED),
        "the header row carries no rule"
    );
    let mut gaps = 0;
    for x in 0..WIDE {
        let cell = buf.cell((x, 1)).expect("cell in area");
        if !cell.symbol().trim().is_empty() {
            continue;
        }
        gaps += 1;
        assert!(
            cell.style().add_modifier.contains(Modifier::UNDERLINED),
            "the gap at column {x} breaks the rule under the header"
        );
    }
    assert!(gaps > 0, "the header has no gaps, so this proves nothing");
}

/// Plain data keeps its row hue beneath the brighter column header.
#[test]
fn plain_data_uses_the_row_rung_beneath_the_header() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();
    let sample = CellTable::new(columns(), rows(1));
    let name_right =
        crate::panes::cell::fit::fit(sample.columns(), sample.rows(), WIDE, 12, 0).widths[0];
    let buf = render(CellPane::new("Projects", Some(1), sample), WIDE, TALL);

    // The Name column is plain text; figure columns may carry their own hues.
    let mut body = 0;
    for x in 0..name_right {
        let cell = buf.cell((x, 2)).expect("cell in area");
        if cell.symbol().trim().is_empty() {
            continue;
        }
        body += 1;
        assert_eq!(
            cell.style().fg,
            Some(tokens::table_row()),
            "column {x} ({:?}) of the first row is not the row rung",
            cell.symbol()
        );
    }
    assert!(body > 0, "the first row drew nothing at all");
    assert_ne!(
        tokens::table_row(),
        tokens::header(),
        "the row rung and the header are the same colour, so none of this separates them"
    );
}

/// A cell's column header follows the cell's focus exactly as its heading does: the live cell's
/// header — and every header while no cell is focused — sits at the text rung, while a cell that
/// has receded behind the live one drops to muted.
#[test]
fn a_receded_cells_header_is_muted_while_the_live_cells_is_at_the_text_rung() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let live = render(
        CellPane::new("Projects", Some(1), CellTable::new(columns(), rows(1)))
            .placed(0, Attention::Zone(0)),
        WIDE,
        TALL,
    );
    let idle = render(
        CellPane::new("Projects", Some(1), CellTable::new(columns(), rows(1))),
        WIDE,
        TALL,
    );
    let receded = render(
        CellPane::new("Projects", Some(1), CellTable::new(columns(), rows(1)))
            .placed(0, Attention::Zone(1)),
        WIDE,
        TALL,
    );

    // The header is the row under the heading, so line 1. Every painted cell of the live and the
    // idle header is at the text rung; every painted cell of the receded header is muted.
    for (buf, rung, label) in [
        (&live, tokens::normal(), "live"),
        (&idle, tokens::normal(), "idle (no cell focused)"),
        (&receded, tokens::muted(), "receded"),
    ] {
        let mut painted = 0;
        for x in 0..WIDE {
            let cell = buf.cell((x, 1)).expect("cell in area");
            if cell.symbol().trim().is_empty() {
                continue;
            }
            assert_eq!(
                cell.style().fg,
                Some(rung),
                "the {label} header's column {x} ({:?}) is not at the {label} rung",
                cell.symbol()
            );
            painted += 1;
        }
        assert!(painted > 0, "the {label} header drew nothing at all");
    }
}

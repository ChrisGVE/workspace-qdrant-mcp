//! The rule that closes this screen's content, directly above its key-hint line.
//!
//! A sibling file rather than four more lines in `super`: `views::service` is already a long
//! way past this crate's size limit, and a guard about one row of the screen is exactly the
//! kind of thing that can live beside it instead of inside it.

use super::*;

/// R11 (Chris, 2026-09-07): the row directly above the key-hint line is a rule, on every
/// view that has a foot.
///
/// This screen already had a rule there — it is the one the Dashboard was made to match —
/// so what this guard adds is the WEIGHT and the single source of it. Both screens now draw
/// it through [`crate::views::top::foot_rule`], and a view that drew its own would be free
/// to pick the other of §2's two greys and leave the two feet looking different.
#[test]
fn the_row_above_the_key_hint_line_is_an_internal_rule_and_the_row_above_that_is_content() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::dark_truecolor();

    let buf = render(frames::base());
    let rule_row = AREA.height - crate::views::top::FOOT_ROWS;

    assert_eq!(
        row(&buf, rule_row),
        crate::widgets::chrome::rule::RULE.repeat(AREA.width as usize),
        "the row above the foot is a rule from edge to edge"
    );
    for x in 0..AREA.width {
        assert_eq!(
            buf.cell((x, rule_row)).expect("cell in area").style().fg,
            Some(tokens::rule_internal()),
            "column {x} of the foot rule is not the internal weight"
        );
    }

    let above = row(&buf, rule_row - 1);
    assert!(
        !above.contains(crate::widgets::chrome::rule::RULE),
        "the row above the foot rule is content, not more rule: {above:?}"
    );
    assert!(
        row(&buf, AREA.height - 1).contains("quit"),
        "the row below it is the key-hint line"
    );
}

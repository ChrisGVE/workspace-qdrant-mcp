//! One test per claim the record view makes — every field kind, both modes, both marks.

use super::*;
use crate::encoding::Encoding;
use crate::tokens::{ModalTint, Palette};

/// The viewport a framework window hands a view at 125×34: 92 columns, 17 rows.
const VIEW: Rect = Rect {
    x: 0,
    y: 0,
    width: 92,
    height: 17,
};

struct Restore(Palette, Encoding, ModalTint, f32);

impl Restore {
    fn mocha() -> Self {
        let restore = Restore(
            Palette::current(),
            Encoding::current(),
            ModalTint::current(),
            tokens::tint_strength(),
        );
        Palette::set(Palette::Bundled);
        Encoding::set(Encoding::TrueColor);
        tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());
        restore
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        Palette::set(self.0);
        Encoding::set(self.1);
        ModalTint::set(self.2);
        tokens::set_tint_strength(self.3);
    }
}

/// Every field kind the ruling names, in one record, so a change to the row arithmetic cannot
/// be green against a fixture that happens to exercise only the easy half.
fn record() -> Vec<FieldRow> {
    vec![
        FieldRow::new("Tenant", Value::Text("open-books".into())).read_only(),
        FieldRow::new("Chunk overlap", Value::Number("128".into())).reference("64"),
        FieldRow::new("Watch for changes", Value::Bool(true)).reference("yes"),
        FieldRow::new(
            "Operation",
            Value::Radio {
                choices: vec!["add".into(), "update".into(), "delete".into()],
                at: 1,
            },
        )
        .reference("update"),
        FieldRow::new(
            "Chunking",
            Value::Choice {
                choices: vec![
                    "tree-sitter/function".into(),
                    "fixed/512".into(),
                    "paragraph".into(),
                ],
                at: 0,
            },
        )
        .reference("fixed/512"),
        FieldRow::new(
            "Note",
            Value::Multi("Held back once already: the grammar download timed out.".into()),
        ),
    ]
}

fn draw(view: RecordView) -> Buffer {
    let mut buf = Buffer::empty(VIEW);
    view.render(VIEW, &mut buf);
    buf
}

fn text(buf: &Buffer, row: u16) -> String {
    (VIEW.x..VIEW.right())
        .filter_map(|x| buf.cell((x, row)).map(|c| c.symbol().to_string()))
        .collect()
}

fn bg_at(buf: &Buffer, x: u16, y: u16) -> Color {
    buf.cell((x, y)).expect("cell in view").bg
}

/// Where a field's value column starts, in the geometry the config table settled.
const VALUE_X: u16 = (GUTTER + W_LABEL) as u16;

/// The screen row the nth DATA row lands on when a reference column is present.
///
/// One, not zero: the reference header is chrome above the data and does not scroll, so every
/// assertion about a field in a three-column record is one row lower than its index. Spelled
/// once here because getting it wrong reads as a colour bug rather than as an off-by-one — it
/// is how the first cut of these tests came to claim the cursor block was missing.
const HEADER: u16 = 1;

// ---------------------------------------------------------------------------------------
// The field kinds
// ---------------------------------------------------------------------------------------

/// Each kind renders as the shape the ruling names it by, and the shapes are SHAPES — a tick
/// box and a radio button survive a terminal with no colour at all (r06 #8).
#[test]
fn every_field_kind_draws_the_shape_its_ruling_names() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    Encoding::set(Encoding::NoColor);
    let buf = draw(RecordView::new(record(), Mode::View { at: 0 }));

    assert!(text(&buf, 0).contains("open-books"), "single-line text");
    assert!(text(&buf, 1).contains("128"), "a number");
    assert!(text(&buf, 2).contains(TICKED), "a tick box, ticked");
    let radio = text(&buf, 3);
    assert!(radio.contains("( ) add"), "an unpicked radio: {radio:?}");
    assert!(radio.contains("(\u{25cf}) update"), "…and the picked one");
    assert!(
        radio.find("add") < radio.find("update"),
        "each label sits to the RIGHT of its own button: {radio:?}"
    );
    let choice = text(&buf, 4);
    assert!(choice.contains("tree-sitter/function"), "{choice:?}");
    assert!(
        choice.contains(OPENS),
        "a choice field says it opens a list"
    );
}

/// An unticked box is a DIFFERENT GLYPH, not a dimmer one — the whole point of drawing state
/// as shape.
#[test]
fn a_boolean_reads_the_same_with_no_colour_at_all() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    Encoding::set(Encoding::NoColor);
    let on = draw(RecordView::new(
        vec![FieldRow::new("W", Value::Bool(true))],
        Mode::View { at: 0 },
    ));
    let off = draw(RecordView::new(
        vec![FieldRow::new("W", Value::Bool(false))],
        Mode::View { at: 0 },
    ));
    assert!(text(&on, 0).contains(TICKED));
    assert!(text(&off, 0).contains(UNTICKED));
    assert_ne!(text(&on, 0), text(&off, 0));
}

/// A radio WRAPS rather than elides. A choice truncated away is a choice the reader can
/// neither make nor know about, which is the opposite of why few choices are drawn in place.
#[test]
fn a_radio_wraps_rather_than_losing_a_choice() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let choices: Vec<String> = ["reindex", "re-embed", "rescan", "reconcile"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let field = FieldRow::new("Operation", Value::Radio { choices, at: 0 }).reference("reindex");
    let narrow = Rect { width: 60, ..VIEW };
    let view = RecordView::new(vec![field], Mode::View { at: 0 }).reference(Reference::Band("DEF"));
    assert!(
        view.rows(narrow.width) > 1,
        "the fourth choice has to go somewhere"
    );

    let mut buf = Buffer::empty(narrow);
    RecordView::new(
        vec![FieldRow::new(
            "Operation",
            Value::Radio {
                choices: ["reindex", "re-embed", "rescan", "reconcile"]
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
                at: 0,
            },
        )
        .reference("reindex")],
        Mode::View { at: 0 },
    )
    .reference(Reference::Band("DEF"))
    .render(narrow, &mut buf);
    let all: String = (0..narrow.height)
        .map(|row| {
            (narrow.x..narrow.right())
                .filter_map(|x| buf.cell((x, row)).map(|c| c.symbol().to_string()))
                .collect::<String>()
        })
        .collect();
    for choice in ["reindex", "re-embed", "rescan", "reconcile"] {
        assert!(all.contains(choice), "{choice} was elided away: {all:?}");
    }
}

/// A multi-line value costs the rows it wraps to, and that is what the scrollbar counts.
#[test]
fn a_multi_line_field_costs_its_wrapped_rows() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let long = "the quick brown fox jumps over the lazy dog and keeps going well past the end \
                of any one line this column could offer it";
    let view = RecordView::new(
        vec![FieldRow::new("Note", Value::Multi(long.into()))],
        Mode::View { at: 0 },
    );
    assert!(view.rows(60) > 1, "a long note must wrap");
    let buf = draw(RecordView::new(
        vec![FieldRow::new("Note", Value::Multi(long.into()))],
        Mode::View { at: 0 },
    ));
    assert!(text(&buf, 0).contains("Note"), "the label is on row one");
    assert!(
        !text(&buf, 1).contains("Note"),
        "…and a continuation row repeats no label: {:?}",
        text(&buf, 1)
    );
    assert!(!text(&buf, 1).trim().is_empty(), "the wrap has to land");
}

/// Wrapping never produces a line wider than the column it was given.
#[test]
fn wrapping_respects_its_width() {
    for line in wrap("alpha beta gamma delta epsilon zeta eta theta", 12) {
        assert!(line.chars().count() <= 12, "{line:?} overflows");
    }
}

// ---------------------------------------------------------------------------------------
// The two modes, and their two marks
// ---------------------------------------------------------------------------------------

/// VIEW mode: the selected field takes the table's own cursor block — the same lavender fill a
/// list's cursor row takes, because Chris asked for "the same manner as for a table".
#[test]
fn view_mode_marks_the_selected_field_with_the_tables_own_block() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let buf = draw(RecordView::new(record(), Mode::View { at: 2 }));
    assert_eq!(bg_at(&buf, VALUE_X, 2), tokens::cursor_bg(), "the block");
    assert_ne!(bg_at(&buf, VALUE_X, 1), tokens::cursor_bg(), "and only it");
    assert!(
        !text(&buf, 2).contains(AT_ROW),
        "no glyph beside the block — that is the same statement twice"
    );
}

/// EDIT mode: the SET mark on every editable field, the POINT mark on the active one, both
/// underlined, and a read-only field wearing neither.
#[test]
fn edit_mode_marks_the_set_and_the_point_differently() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let buf = draw(RecordView::new(record(), Mode::Edit { at: 1, edit: None }));

    assert_eq!(bg_at(&buf, VALUE_X, 1), tokens::field::active_bg(), "point");
    assert_eq!(bg_at(&buf, VALUE_X, 2), tokens::field::editable_bg(), "set");
    assert_ne!(
        bg_at(&buf, VALUE_X, 0),
        tokens::field::editable_bg(),
        "a read-only field is not in the set"
    );
    assert!(
        text(&buf, 1).contains(AT_ROW),
        "the active row carries the mark: {:?}",
        text(&buf, 1)
    );
    // …and no row block anywhere, or the two modes would be a brightness comparison.
    for row in 0..6 {
        assert_ne!(
            bg_at(&buf, VALUE_X, row),
            tokens::cursor_bg(),
            "row {row} still wears the VIEW-mode block"
        );
    }
}

/// The SET mark survives an encoding that takes the ladder away, because it is not only a
/// fill — the underline is what r06 #8 requires and what `NO_COLOR` leaves standing.
#[test]
fn the_editable_set_survives_an_encoding_with_no_ladder() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    for encoding in [Encoding::NoColor, Encoding::Ansi16, Encoding::Ansi256] {
        Encoding::set(encoding);
        let buf = draw(RecordView::new(record(), Mode::Edit { at: 1, edit: None }));
        let editable = buf.cell((VALUE_X, 2)).expect("an editable field");
        let read_only = buf.cell((VALUE_X, 0)).expect("a read-only field");
        assert!(
            editable.modifier.contains(tokens::field::EDITABLE_MARK),
            "{encoding:?}: an editable field must be ruled"
        );
        assert!(
            !read_only.modifier.contains(tokens::field::EDITABLE_MARK),
            "{encoding:?}: a read-only field must not be"
        );
    }
}

/// The live caret is exactly three spans plus its pad — `caret_spans` is not generalised, and
/// the column arithmetic downstream relies on that.
#[test]
fn the_active_text_field_carries_the_crates_one_caret() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let buf = draw(RecordView::new(
        record(),
        Mode::Edit {
            at: 1,
            edit: Some(Edit::insert("256")),
        },
    ));
    let row = text(&buf, 1);
    assert!(row.contains("256"), "the value being typed: {row:?}");
    assert!(row.contains('\u{258f}'), "the insert caret: {row:?}");
    assert_eq!(
        caret_spans(&Edit::insert("256"), Style::default()).len(),
        3,
        "three spans, and it stays three"
    );
}

// ---------------------------------------------------------------------------------------
// The third column
// ---------------------------------------------------------------------------------------

/// The header row belongs to the third column alone — columns one and two have no title — so a
/// two-column record has no header row and gives it back to data.
#[test]
fn only_the_third_column_brings_a_header_row() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    // A fixture with no WRAPPING field, so the only thing that could move the row count is the
    // header. A record with a multi-line note does gain rows from a third column, because that
    // column narrows the one the note wraps into — a fact about wrapping, not about headers,
    // and it would make this test measure the wrong thing.
    let fixed = || {
        vec![
            FieldRow::new("Tenant", Value::Text("open-books".into())).read_only(),
            FieldRow::new("Chunk overlap", Value::Number("128".into())).reference("64"),
        ]
    };
    let plain = RecordView::new(fixed(), Mode::View { at: 0 });
    let with = RecordView::new(fixed(), Mode::View { at: 0 }).reference(Reference::Band("DEFAULT"));
    assert_eq!(plain.header_rows(), 0);
    assert_eq!(with.header_rows(), 1);
    // The header is chrome: it costs a row of VIEWPORT and never a row of SCROLL.
    assert_eq!(with.rows(VIEW.width), plain.rows(VIEW.width));
    assert_eq!(with.data_height(VIEW.height) + 1, VIEW.height);
    assert_eq!(plain.data_height(VIEW.height), VIEW.height);
}

/// **The header does not scroll.** The defect the overflow frame found: while it was the first
/// entry of the scrollable list, any offset past the top left the third column an unlabelled
/// band of values — and a default is not distinguishable from a pre-edit value by looking.
#[test]
fn the_header_survives_a_scroll() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    for offset in [0usize, 1, 3, 5] {
        let buf = draw(
            RecordView::new(record(), Mode::View { at: 0 })
                .reference(Reference::Band("DEFAULT"))
                .offset(offset),
        );
        assert!(
            text(&buf, 0).contains("DEFAULT"),
            "at offset {offset} the header is gone: {:?}",
            text(&buf, 0)
        );
    }
}

/// The band is a SURFACE and it runs the whole height, header row included — one region, told
/// apart by common region rather than by a hue nobody could name.
#[test]
fn the_band_is_a_surface_that_runs_the_whole_height() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let view =
        RecordView::new(record(), Mode::View { at: 0 }).reference(Reference::Band("DEFAULT"));
    let x = view.reference_x(VIEW);
    let buf =
        draw(RecordView::new(record(), Mode::View { at: 0 }).reference(Reference::Band("DEFAULT")));
    for row in 0..=3u16 {
        assert_eq!(
            bg_at(&buf, x, row),
            tokens::field::reference_bg(),
            "the band breaks on row {row}"
        );
    }
    // Arm A spends no surface at all, which is the difference the pantry pair shows.
    let plain =
        draw(RecordView::new(record(), Mode::View { at: 0 }).reference(Reference::Text("DEFAULT")));
    assert_ne!(bg_at(&plain, x, 1), tokens::field::reference_bg());
}

/// **The band comes off in EDIT mode.** While the window is editing, a background means one
/// thing and one thing only: you may type here.
#[test]
fn the_band_comes_off_while_the_window_is_editing() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let view =
        RecordView::new(record(), Mode::View { at: 0 }).reference(Reference::Band("DEFAULT"));
    let x = view.reference_x(VIEW);
    let editing = draw(
        RecordView::new(record(), Mode::Edit { at: 1, edit: None })
            .reference(Reference::Band("DEFAULT")),
    );
    assert_ne!(
        bg_at(&editing, x, 2),
        tokens::field::reference_bg(),
        "the reference band is still on in edit mode"
    );
}

/// The designer's round-1b call: the VIEW-mode block stops at the value column, so the band
/// survives on the very row where the reader most needs the comparison.
#[test]
fn the_cursor_block_stops_at_the_value_and_leaves_the_band_whole() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let probe =
        RecordView::new(record(), Mode::View { at: 1 }).reference(Reference::Band("DEFAULT"));
    let x = probe.reference_x(VIEW);

    let to_value = draw(
        RecordView::new(record(), Mode::View { at: 1 })
            .reference(Reference::Band("DEFAULT"))
            .cursor_extent(CursorExtent::ToValue),
    );
    assert_eq!(
        bg_at(&to_value, VALUE_X, HEADER + 1),
        tokens::cursor_bg(),
        "the field"
    );
    assert_eq!(
        bg_at(&to_value, x, HEADER + 1),
        tokens::field::reference_bg(),
        "…and the band is unbroken beside it"
    );

    let full = draw(
        RecordView::new(record(), Mode::View { at: 1 })
            .reference(Reference::Band("DEFAULT"))
            .cursor_extent(CursorExtent::FullRow),
    );
    assert_eq!(
        bg_at(&full, x, HEADER + 1),
        tokens::cursor_bg(),
        "arm A is the one that breaks the band — if it stopped, the A/B says nothing"
    );
}

/// With no third column there is nothing to leave whole, so the block runs the whole row.
#[test]
fn the_cursor_block_runs_the_whole_row_when_there_is_no_band() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let buf = draw(RecordView::new(record(), Mode::View { at: 1 }));
    assert_eq!(bg_at(&buf, VIEW.right() - 1, 1), tokens::cursor_bg());
}

/// A value that differs from its reference is marked by the CONFIG TABLE's rule, not by a
/// second one — derived from the two values, so the mark and the fact cannot disagree.
#[test]
fn a_changed_value_is_marked_by_the_config_tables_own_rule() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let changed = FieldRow::new("Chunk overlap", Value::Number("128".into())).reference("64");
    let same = FieldRow::new("Chunk overlap", Value::Number("64".into())).reference("64");
    assert!(changed.is_changed());
    assert!(!same.is_changed());

    // `at: 9` is off the end on purpose: no cursor block, so the only thing colouring these
    // two cells is the changed-value rule itself.
    let buf = draw(
        RecordView::new(vec![changed, same], Mode::View { at: 9 })
            .reference(Reference::Band("DEFAULT")),
    );
    let differs = buf.cell((VALUE_X, HEADER)).expect("the changed value");
    let matches = buf.cell((VALUE_X, HEADER + 1)).expect("the matching value");
    assert_eq!(differs.fg, tokens::strong(), "a differing value is loud");
    assert_eq!(matches.fg, tokens::normal(), "a matching one is not");
}

// ---------------------------------------------------------------------------------------
// The states the frames did not show
// ---------------------------------------------------------------------------------------

/// An empty record says `No data` — the word the table already uses, because one fact should
/// not have two spellings — and draws no header, because a header over no column is chrome for
/// data that is not there.
#[test]
fn an_empty_record_says_what_the_table_says() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let view = RecordView::new(Vec::new(), Mode::View { at: 0 }).reference(Reference::Band("DEF"));
    assert!(view.is_empty());
    assert_eq!(view.header_rows(), 0, "no header over no column");
    assert_eq!(view.rows(VIEW.width), 1, "the one line saying so");

    let buf = draw(
        RecordView::new(Vec::new(), Mode::View { at: 0 }).reference(Reference::Band("DEFAULT")),
    );
    assert!(text(&buf, 0).contains(EMPTY), "{:?}", text(&buf, 0));
    assert!(!text(&buf, 0).contains("DEFAULT"), "and no header above it");
}

/// A viewport too short for even the header draws nothing rather than a header with no data
/// under it.
#[test]
fn a_viewport_with_no_room_draws_nothing() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let tiny = Rect { height: 0, ..VIEW };
    let mut buf = Buffer::empty(VIEW);
    RecordView::new(record(), Mode::View { at: 0 }).render(tiny, &mut buf);
    assert_eq!(buf, Buffer::empty(VIEW));
}

/// Scrolling moves the data and only the data.
#[test]
fn the_offset_moves_the_window_onto_the_rows() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let top = draw(RecordView::new(record(), Mode::View { at: 0 }));
    let down = draw(RecordView::new(record(), Mode::View { at: 0 }).offset(2));
    assert!(top.area.height > 0);
    assert_eq!(
        text(&down, 0).trim(),
        text(&top, 2).trim(),
        "offset 2 must put the third row first"
    );
}

/// The three edit-mode arms are genuinely different frames, or the pantry A/B/C says nothing.
#[test]
fn the_three_background_schemes_are_three_different_frames() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    // The POINT (field 1) and the SET (field 2), both one row lower than their index because
    // the reference header sits above them.
    let at = |scheme| {
        let buf = draw(
            RecordView::new(record(), Mode::Edit { at: 1, edit: None })
                .scheme(scheme)
                .reference(Reference::Band("DEFAULT")),
        );
        (
            bg_at(&buf, VALUE_X, HEADER + 1),
            bg_at(&buf, VALUE_X, HEADER + 2),
        )
    };
    let neutral = at(Scheme::Neutral);
    let wash = at(Scheme::AccentWash);
    let derived = at(Scheme::SelectionDerived);
    assert_ne!(neutral, wash);
    assert_ne!(neutral, derived);
    assert_ne!(wash, derived);
    // Arm C's point IS the VIEW-mode block, which is precisely the objection to it.
    assert_eq!(derived.0, tokens::cursor_bg());
}

// ---------------------------------------------------------------------------------------
// The drop-down
// ---------------------------------------------------------------------------------------

/// The list is as wide as the cell it came out of and covers its own `▾`, so it reads as the
/// field OPENING rather than as a box that happens to be nearby.
#[test]
fn the_drop_down_is_at_least_as_wide_as_the_cell_it_came_from() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let anchor = Rect {
        x: VALUE_X,
        y: 4,
        width: 40,
        height: 1,
    };
    let choices: Vec<String> = ["fixed/512", "paragraph"]
        .iter()
        .map(|s| (*s).into())
        .collect();
    let list = DropDown::new(choices.clone(), 0, anchor);
    let rect = list.rect(VIEW);
    assert!(rect.width >= anchor.width, "narrower than its own cell");
    assert_eq!(rect.x, anchor.x, "and hanging from it");
    assert_eq!(rect.y, anchor.y + 1, "…directly below");
}

/// The cursor sits on the CURRENT value, wearing the data cursor's own block — inside this
/// list the highlighted line is the cursor, so no highlight is invented.
#[test]
fn the_drop_down_opens_with_the_cursor_on_the_current_value() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let anchor = Rect {
        x: VALUE_X,
        y: 2,
        width: 30,
        height: 1,
    };
    let choices: Vec<String> = ["tree-sitter/function", "fixed/512", "paragraph"]
        .iter()
        .map(|s| (*s).into())
        .collect();
    let list = DropDown::new(choices.clone(), 1, anchor);
    let rect = list.rect(VIEW);
    let mut buf = Buffer::empty(VIEW);
    DropDown::new(choices, 1, anchor).render(VIEW, &mut buf);

    let row_text = |row: u16| {
        (rect.x + 1..rect.right() - 1)
            .filter_map(|x| buf.cell((x, row)).map(|c| c.symbol().to_string()))
            .collect::<String>()
    };
    assert!(row_text(rect.y + 1).contains("tree-sitter/function"));
    assert!(row_text(rect.y + 2).contains("fixed/512"));
    assert_eq!(
        bg_at(&buf, rect.x + 1, rect.y + 2),
        tokens::cursor_bg(),
        "the cursor must be on the current value"
    );
    assert_ne!(bg_at(&buf, rect.x + 1, rect.y + 1), tokens::cursor_bg());
}

/// A long list scrolls inside the drop-down rather than growing past the window.
#[test]
fn a_long_drop_down_scrolls_to_keep_the_current_value_visible() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let anchor = Rect {
        x: 2,
        y: 1,
        width: 20,
        height: 1,
    };
    let choices: Vec<String> = (0..40).map(|n| format!("choice-{n}")).collect();
    let bounds = Rect {
        width: 60,
        height: 12,
        ..VIEW
    };
    let list = DropDown::new(choices.clone(), 39, anchor);
    let rect = list.rect(bounds);
    assert!(rect.height <= bounds.height, "the list must fit its bounds");

    let mut buf = Buffer::empty(bounds);
    DropDown::new(choices, 39, anchor).render(bounds, &mut buf);
    let mut all = String::new();
    for y in rect.y..rect.bottom() {
        for x in rect.x..rect.right() {
            if let Some(cell) = buf.cell((x, y)) {
                all.push_str(cell.symbol());
            }
        }
    }
    assert!(all.contains("choice-39"), "the current value must be shown");
}

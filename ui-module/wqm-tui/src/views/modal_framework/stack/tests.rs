//! Push, pop, the guard, the trail and the slide — one test per rule the stack carries.

use super::*;
use crate::encoding::Encoding;
use crate::panes::cell::{Cell, Column};
use crate::tokens::{ModalTint, Palette};
use crate::views::modal_framework::record::{FieldRow, Reference, Value};
use crate::views::modal_framework::table::Pin;
use crate::widgets::edit_field::Edit;

const SCREEN: Rect = Rect {
    x: 0,
    y: 0,
    width: 125,
    height: 34,
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

fn fields() -> Vec<FieldRow> {
    vec![
        FieldRow::new("Tenant", Value::Text("open-books".into())).read_only(),
        FieldRow::new("Chunk overlap", Value::Number("128".into())).reference("64"),
        FieldRow::new("Watch for changes", Value::Bool(true)).reference("yes"),
        FieldRow::new(
            "Note",
            Value::Multi("Held back once already: the grammar download timed out.".into()),
        ),
    ]
}

fn record_layer(mode: Mode) -> Layer {
    Layer::new(
        "reading_guide.py",
        "Queue item — reading_guide.py",
        View::Record(RecordState::new(fields(), mode).reference(Reference::Text("DEFAULT"))),
    )
    .hint("e", "Edit")
    .hint("\u{232b}", "Back")
}

fn table_layer() -> Layer {
    let columns = vec![
        Column::number("", 3),
        Column::text("Tenant", 20),
        Column::flex("Object"),
    ];
    let rows = vec![
        vec![
            Cell::Text("open-books".into()),
            Cell::Text("stage_b/reading_guide.py".into()),
        ],
        vec![
            Cell::Text("mnemosyne".into()),
            Cell::Text("src/recall.rs".into()),
        ],
    ];
    Layer::new(
        "Queue",
        "open-books — queue",
        View::Table(super::TableView::new(columns, rows).pinned(Pin::new(1, "open-books"))),
    )
    .hint("\u{21b5}", "Drill down")
}

fn stack() -> Stack {
    Stack::new(Layer::new(
        "Libraries",
        "Libraries",
        View::Record(RecordState::new(fields(), Mode::View { at: 0 })),
    ))
}

fn draw(stack: &Stack) -> Buffer {
    let mut buf = Buffer::empty(SCREEN);
    stack.render(Container::footprint(SCREEN), &mut buf);
    buf
}

fn inside(buf: &Buffer, row: u16) -> String {
    let rect = Container::footprint(SCREEN);
    (rect.x + 1..rect.right() - 1)
        .filter_map(|x| buf.cell((x, row)).map(|c| c.symbol().to_string()))
        .collect()
}

// ---------------------------------------------------------------------------------------
// Push and pop
// ---------------------------------------------------------------------------------------

/// A pop restores the parent **exactly as it was left** — its cursor, its scroll offset, its
/// dirty flag and its edits. The stack holds live state rather than pictures, and this is the
/// property that distinguishes the two.
#[test]
fn a_pop_restores_the_parent_exactly_as_it_was_left() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mut stack = stack();

    // Leave the root mid-edit, scrolled, and dirty.
    *stack.top_mut() = record_layer(Mode::Edit {
        at: 2,
        edit: Some(Edit::insert("256")),
    })
    .dirty(true);
    if let View::Record(record) = &mut stack.top_mut().view {
        record.offset = 2;
    }

    stack.push(table_layer());
    assert_eq!(stack.depth(), 2);
    // The child is its own view, with its own cursor — the parent's state is untouched.
    assert!(!stack.top().dirty);
    assert!(!stack.top().view.editing());

    assert_eq!(stack.pop(), Pop::Popped);
    assert_eq!(stack.depth(), 1);
    let parent = stack.top();
    assert!(
        parent.dirty,
        "the parent's dirty flag survived the round trip"
    );
    assert!(parent.view.editing(), "…and so did its edit mode");
    match &parent.view {
        View::Record(record) => {
            assert_eq!(record.offset, 2, "…and its scroll offset");
            assert_eq!(record.mode.at(), 2, "…and its cursor");
            match &record.mode {
                Mode::Edit {
                    edit: Some(edit), ..
                } => {
                    assert_eq!(edit.value(), "256", "…and the characters typed into it")
                }
                other => panic!("the live edit is gone: {other:?}"),
            }
        }
        View::Table(_) => panic!("the wrong view came back"),
    }
}

/// A window's last view IS the window, so the root cannot be popped away.
#[test]
fn the_root_view_cannot_be_popped() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mut stack = stack();
    assert_eq!(stack.pop(), Pop::AtRoot);
    assert_eq!(stack.depth(), 1);
}

/// **Backspace belongs to the editor first.** Inside edit mode it is a keystroke the field is
/// entitled to, and a stack that took it would delete a character's worth of navigation.
#[test]
fn backspace_is_held_by_the_editor_while_a_field_is_being_typed_into() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mut stack = stack();
    stack.push(record_layer(Mode::Edit {
        at: 1,
        edit: Some(Edit::insert("128")),
    }));

    assert_eq!(stack.pop(), Pop::HeldByEditor);
    assert_eq!(stack.depth(), 2, "nothing moved");
    assert!(!stack.guarded(), "and no guard opened either");

    // Leave edit mode, and the same key pops.
    if let View::Record(record) = &mut stack.top_mut().view {
        record.mode = Mode::View { at: 1 };
    }
    assert_eq!(stack.pop(), Pop::Popped);
    assert_eq!(stack.depth(), 1);
}

/// A refusal is named, not silent: *the editor has it* and *this is the root* are different
/// facts, and a caller that could not tell them apart could not report either.
#[test]
fn every_refusal_says_which_refusal_it_is() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mut root = stack();
    assert_eq!(root.pop(), Pop::AtRoot);

    let mut editing = stack();
    editing.push(record_layer(Mode::Edit { at: 0, edit: None }));
    assert_eq!(editing.pop(), Pop::HeldByEditor);

    let mut dirty = stack();
    dirty.push(record_layer(Mode::View { at: 0 }).dirty(true));
    assert_eq!(dirty.pop(), Pop::Guarded);
}

// ---------------------------------------------------------------------------------------
// The guard
// ---------------------------------------------------------------------------------------

/// A dirty pop asks rather than discarding, and asks on `Fill::Layer2` — which is the job §6
/// left that layer.
#[test]
fn a_pop_over_a_dirty_view_opens_the_confirm() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mut stack = stack();
    stack.push(record_layer(Mode::View { at: 1 }).dirty(true));

    assert_eq!(stack.pop(), Pop::Guarded);
    assert!(stack.guarded(), "the guard must be open");
    assert_eq!(stack.depth(), 2, "and nothing popped behind it");

    let buf = draw(&stack);
    let screen: String = (0..SCREEN.height)
        .map(|row| {
            (0..SCREEN.width)
                .filter_map(|x| buf.cell((x, row)).map(|c| c.symbol().to_string()))
                .collect::<String>()
        })
        .collect();
    assert!(
        screen.contains("Discard changes?"),
        "the guard is not drawn"
    );
    assert!(
        screen.contains("keep editing"),
        "and neither is its way out"
    );
}

/// The guard is centred on the WINDOW, not on the screen. A dialogue belongs to the thing it
/// is asking about, and the window sits below the page's own header — centring on the screen
/// puts the guard above the window's middle.
#[test]
fn the_guard_is_centred_on_the_window_it_is_asking_about() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let middle = |rect: Rect, of: Rect| {
        let above = rect.y - of.y;
        let below = of.bottom() - rect.bottom();
        above.abs_diff(below) <= 1
    };

    // Stated at BOTH sizes a frame is drawn at.
    //
    // **The old `assert_ne!` against a screen-centred guard is gone, and the ruling is why.**
    // Round 1's window was clamped below the page header, so centring on the window and
    // centring on the screen produced two different rects and the difference was the claim.
    // Item 0's window is the page inset by five on every side, which makes it exactly
    // concentric with the page — the two centres now coincide by construction, and asserting
    // they differ would be asserting the ruling is not in force.
    //
    // What still bites is containment: the guard belongs to the window it is asking about, so
    // it must sit wholly inside it. On a screen-centred guard that is a coincidence of the
    // sizes; here it is the property.
    for screen in [SCREEN, Rect::new(0, 0, 100, 30)] {
        let window = Container::footprint(screen);
        let on_window = discard_guard().rect(window);
        assert!(middle(on_window, window), "centred in the window");
        assert!(
            on_window.y >= window.y
                && on_window.x >= window.x
                && on_window.bottom() <= window.bottom()
                && on_window.right() <= window.right(),
            "the guard must sit wholly inside the window it is asking about: {on_window:?} in \
             {window:?}"
        );
    }
}

/// The two answers the guard offers, and they do what they say.
#[test]
fn the_guard_discards_and_pops_or_keeps_editing() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mut discard = stack();
    discard.push(record_layer(Mode::View { at: 0 }).dirty(true));
    assert_eq!(discard.pop(), Pop::Guarded);
    assert_eq!(discard.discard(), Pop::Popped);
    assert_eq!(discard.depth(), 1);
    assert!(!discard.guarded(), "the guard closes behind it");

    let mut keep = stack();
    keep.push(record_layer(Mode::View { at: 0 }).dirty(true));
    assert_eq!(keep.pop(), Pop::Guarded);
    keep.keep_editing();
    assert!(!keep.guarded());
    assert_eq!(keep.depth(), 2, "keeping must change nothing else");
    assert!(keep.top().dirty, "…including the edits it was asking about");
}

/// While the guard is open it owns the key: a second Backspace must not pop behind it.
#[test]
fn a_second_backspace_does_not_pop_behind_an_open_guard() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mut stack = stack();
    stack.push(record_layer(Mode::View { at: 0 }).dirty(true));
    assert_eq!(stack.pop(), Pop::Guarded);
    assert_eq!(stack.pop(), Pop::Guarded, "still asking");
    assert_eq!(stack.depth(), 2);
}

// ---------------------------------------------------------------------------------------
// The decoration the stack produces
// ---------------------------------------------------------------------------------------

/// The breadcrumb IS the stack, read out — which is what makes it start-dependent. The same
/// view reached two ways shows two trails, and nothing here consults a hierarchy.
#[test]
fn the_breadcrumb_is_the_stack_and_therefore_start_dependent() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mut from_libraries = stack();
    from_libraries.push(table_layer());
    assert_eq!(from_libraries.crumbs(), vec!["Libraries", "Queue"]);

    let mut from_queue = Stack::new(table_layer());
    from_queue.push(record_layer(Mode::View { at: 0 }));
    assert_eq!(from_queue.crumbs(), vec!["Queue", "reading_guide.py"]);

    // The trail on screen, in order, and with the chevron between.
    let buf = draw(&from_libraries);
    let rect = Container::footprint(SCREEN);
    let trail = inside(&buf, rect.y + 1);
    assert!(trail.trim().starts_with("Libraries"), "{trail:?}");
    assert!(trail.contains('\u{203a}'), "no chevron: {trail:?}");
}

/// A pop shortens the trail, so the breadcrumb can never claim a depth the stack does not
/// have.
#[test]
fn the_trail_shortens_with_the_stack() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mut stack = stack();
    stack.push(table_layer());
    stack.push(record_layer(Mode::View { at: 0 }));
    assert_eq!(stack.crumbs().len(), 3);
    assert_eq!(stack.pop(), Pop::Popped);
    assert_eq!(stack.crumbs().len(), 2);
    assert_eq!(stack.crumbs().last().map(String::as_str), Some("Queue"));
}

/// The fifth row appears because the TOP view mounts one, and goes away when a view that does
/// not is on top — the row is the composition's, and the composition is the top layer.
#[test]
fn the_fifth_row_follows_the_view_on_top() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let mut stack = stack();
    assert_eq!(
        stack.decoration().top_rows(),
        crate::widgets::modal_frame::TOP_ROWS,
        "a record mounts no search row"
    );

    stack.push(table_layer());
    assert_eq!(
        stack.decoration().top_rows(),
        crate::widgets::modal_frame::TOP_ROWS + 1,
        "a pinned table states its floor on the fifth row"
    );

    assert_eq!(stack.pop(), Pop::Popped);
    assert_eq!(
        stack.decoration().top_rows(),
        crate::widgets::modal_frame::TOP_ROWS,
        "and the row goes back to the view when the table leaves"
    );
}

/// The window says `-- EDIT --` while the top view is editing, and stops when it is not.
/// **Nothing on the title row says the window is editing.**
///
/// Round 1 put `-- EDIT --` there on a Nielsen #1 reading, and Chris ruled it out (item (b)):
/// the columnar change IS the indication. The test asserts BOTH halves, because a window that
/// had simply failed to enter edit mode would satisfy the first one on its own.
#[test]
fn no_banner_announces_edit_mode_and_the_window_is_editing_all_the_same() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);
    let mut stack = stack();
    stack.push(record_layer(Mode::Edit { at: 1, edit: None }));
    assert!(stack.top().view.editing(), "the frame is in edit mode");
    let editing = draw(&stack);
    let title_row = inside(&editing, rect.y + 3);
    assert!(
        !title_row.contains("EDIT"),
        "the title row still carries a mode banner: {title_row:?}"
    );

    if let View::Record(record) = &mut stack.top_mut().view {
        record.mode = Mode::View { at: 1 };
    }
    let viewing = draw(&stack);
    assert!(!inside(&viewing, rect.y + 3).contains("EDIT"));
}

/// The window does not resize between views — a record and a table put their top-left corner
/// on the same cell, which is what a fixed container is FOR.
#[test]
fn the_window_does_not_move_when_the_view_changes() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let corner = |buf: &Buffer| {
        (0..SCREEN.height)
            .flat_map(|y| (0..SCREEN.width).map(move |x| (x, y)))
            .find(|(x, y)| buf.cell((*x, *y)).map(|c| c.symbol()) == Some("\u{250c}"))
    };
    let mut stack = stack();
    let record = draw(&stack);
    stack.push(table_layer());
    let table = draw(&stack);
    assert_eq!(corner(&record), corner(&table), "the window moved");
}

// ---------------------------------------------------------------------------------------
// The slide
// ---------------------------------------------------------------------------------------

/// At the half-way point of a push BOTH views are on screen, which is the only thing a still
/// frame can say about a slide and the only thing worth putting to Chris.
#[test]
fn the_slide_shows_both_views_at_half_way() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let viewport = Rect::new(0, 0, 40, 3);
    let outgoing = scratch(viewport, |area, buf| {
        buf.set_string(
            area.x,
            area.y,
            "L".repeat(area.width as usize),
            Style::default(),
        );
    });
    let incoming = scratch(viewport, |area, buf| {
        buf.set_string(
            area.x,
            area.y,
            "R".repeat(area.width as usize),
            Style::default(),
        );
    });

    let row = |buf: &Buffer| {
        (0..viewport.width)
            .filter_map(|x| buf.cell((x, 0)).map(|c| c.symbol().to_string()))
            .collect::<String>()
    };

    let at = |t: f32| {
        let mut buf = Buffer::empty(viewport);
        slide(viewport, t, &outgoing, &incoming, &mut buf);
        row(&buf)
    };

    assert_eq!(at(0.0), "L".repeat(40), "t=0 is the outgoing view, whole");
    assert_eq!(at(1.0), "R".repeat(40), "t=1 is the incoming view, whole");

    let half = at(0.5);
    assert_eq!(
        half.matches('L').count(),
        20,
        "half of the record is leaving"
    );
    assert_eq!(
        half.matches('R').count(),
        20,
        "…and half the queue arriving"
    );
    assert!(
        half.starts_with('L') && half.ends_with('R'),
        "and it slides right-to-left: {half:?}"
    );
}

/// **Only the viewport moves.** The decoration is the window's, not the view's, and it is
/// already at the destination — a breadcrumb that slid with its content would be briefly
/// wrong, which is worse than one that arrives early.
#[test]
fn the_slide_moves_the_viewport_and_leaves_the_decoration_alone() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);
    let mut stack = stack();
    stack.push(table_layer());

    let mut buf = Buffer::empty(SCREEN);
    stack.render(rect, &mut buf);
    let trail_before = inside(&buf, rect.y + 1);
    let title_before = inside(&buf, rect.y + 3);

    let viewport = Container::new(stack.decoration()).viewport(rect);
    let outgoing = scratch(viewport, |area, buf| {
        buf.set_string(area.x, area.y, "OUTGOING", Style::default());
    });
    let incoming = scratch(viewport, |area, buf| {
        buf.set_string(area.x, area.y, "INCOMING", Style::default());
    });
    slide(viewport, 0.5, &outgoing, &incoming, &mut buf);

    assert_eq!(inside(&buf, rect.y + 1), trail_before, "the trail moved");
    assert_eq!(inside(&buf, rect.y + 3), title_before, "the title moved");
    // …and the viewport really did change, or this test proves nothing.
    let content = inside(&buf, viewport.y);
    assert!(
        content.contains("INCOMING") || content.contains("GOING"),
        "the viewport did not slide: {content:?}"
    );
}

/// The slide is clamped: a `t` outside `0..=1` is not a frame, and must not read past the end
/// of either buffer.
#[test]
fn the_slide_clamps_rather_than_reading_past_a_buffer() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let viewport = Rect::new(0, 0, 8, 1);
    let outgoing = scratch(viewport, |area, buf| {
        buf.set_string(area.x, area.y, "LLLLLLLL", Style::default());
    });
    let incoming = scratch(viewport, |area, buf| {
        buf.set_string(area.x, area.y, "RRRRRRRR", Style::default());
    });
    for t in [-2.0f32, -0.1, 1.1, 99.0] {
        let mut buf = Buffer::empty(viewport);
        slide(viewport, t, &outgoing, &incoming, &mut buf);
        let row: String = (0..8)
            .filter_map(|x| buf.cell((x, 0)).map(|c| c.symbol().to_string()))
            .collect();
        assert!(row == "LLLLLLLL" || row == "RRRRRRRR", "t={t}: {row:?}");
    }
}

// ---------------------------------------------------------------------------------------
// The scrollbar the stack asks for
// ---------------------------------------------------------------------------------------

/// A view that fits gets no scrollbar, and one that overflows gets one — measured against the
/// DATA rows, with the record's header excluded because it is chrome and does not scroll.
#[test]
fn the_scrollbar_appears_only_when_the_data_overflows() {
    let _serial = crate::global_state_lock();
    let _restore = Restore::mocha();
    let rect = Container::footprint(SCREEN);
    let viewport = Container::new(stack().decoration()).viewport(rect);

    let short = Stack::new(Layer::new(
        "x",
        "x",
        View::Record(RecordState::new(fields(), Mode::View { at: 0 })),
    ));
    assert!(
        short.top().view.scroll(viewport).total <= viewport.height as usize,
        "four fields must fit, or this test is measuring nothing"
    );

    let many: Vec<FieldRow> = (0..60)
        .map(|n| FieldRow::new(format!("Field {n}"), Value::Number(n.to_string())))
        .collect();
    let long = Stack::new(Layer::new(
        "x",
        "x",
        View::Record(RecordState::new(many, Mode::View { at: 0 })),
    ));
    assert!(long.top().view.scroll(viewport).total > viewport.height as usize);

    let with_bar = draw(&long);
    let bar_x = viewport.x + viewport.width - 1;
    let column: String = (viewport.y..viewport.y + viewport.height)
        .filter_map(|y| with_bar.cell((bar_x, y)).map(|c| c.symbol().to_string()))
        .collect();
    assert!(
        column.contains('\u{2590}') || column.contains('\u{2595}'),
        "no bar was drawn for a record that overflows: {column:?}"
    );
}

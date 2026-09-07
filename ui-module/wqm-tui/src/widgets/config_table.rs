//! The config table — the two highlight roles nothing else in this crate renders.
//!
//! VISUAL-LANGUAGE §3 specifies four highlight roles. The selector is the tab bar's and is
//! built ([`super::tab_bar`]); health is the store zone's and is built
//! ([`super::store_health`]). The remaining two — the **data cursor** and the **editing
//! cell** — are specified against a table of settable keys, and this is that table.
//!
//! Geometry and content follow the frame Chris reviewed at r06 (`tools/gen-config-frame.py`
//! in the storyboard tree), so the widget and the reviewed frame can be laid side by side.
//! What the frame expressed as flags this module **derives**, which is the one deliberate
//! departure — see [`Entry::is_changed`] and [`Focus`].

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens;
use crate::widgets::edit_field::caret_spans;

/// The edit-in-place field this table opened, now shared with every other surface that has one
/// — see [`crate::widgets::edit_field`]. Re-exported rather than moved out of sight: this is
/// still where a reader of the config table expects to find what an edit is.
pub use crate::widgets::edit_field::{Edit, EditMode};

/// Column geometry, carried over from the reviewed frame so the two agree cell for cell.
/// Padding is computed on the **visible** text and the style wrapped around the padded
/// field, never the other way round — a trailing space inside a styled span is what breaks
/// column alignment and bleeds a row background past the table.
const MARGIN: usize = 2;
const W_KEY: usize = 20;
const W_ACTUAL: usize = 26;
const W_DEFAULT: usize = 21;
/// How far a key is indented under its group header.
const INDENT: usize = 3;

/// The spelling of an absent value. It is the absence of a value rather than a value, so it
/// renders faint — a `<unset>` that read as normal body text would claim a setting is set.
pub const UNSET: &str = "<unset>";

/// One settable key: its human label, the value in force, and the value it would have had.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Entry {
    label: String,
    actual: String,
    default: String,
}

impl Entry {
    pub fn new(
        label: impl Into<String>,
        actual: impl Into<String>,
        default: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            actual: actual.into(),
            default: default.into(),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn actual(&self) -> &str {
        &self.actual
    }

    pub fn default_value(&self) -> &str {
        &self.default
    }

    /// §3: *"a differing actual value is `[bold]`, a matching one is normal"*.
    ///
    /// **Derived, never a flag.** The reviewed frame carried `changed` as a separate field
    /// beside the two values, which makes a frame showing `2000` against a default of `1500`
    /// in plain weight representable — a depiction of a state the rule forbids. Comparing the
    /// two values instead removes that frame from the language: the mark and the fact it
    /// marks cannot disagree.
    pub fn is_changed(&self) -> bool {
        self.actual != self.default
    }

    /// Whether the value in force is [`UNSET`].
    pub fn is_unset(&self) -> bool {
        self.actual == UNSET
    }
}

/// A display row: either a group header or a settable key.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Row {
    /// `qdrant`, `watcher`, … — bold at the margin, with no values of its own. Chris's r06
    /// mark #10: the key names were "too technical", so the dotted id is decomposed into a
    /// group header plus an indented human label.
    Group(String),
    Entry(Entry),
}

/// What the table's interaction state is, expressed so the states §3 forbids cannot be built.
///
/// Both indices count **entries**, not display rows, which is what makes a cursor on a group
/// header unrepresentable — a group is not a datum, so the data cursor has nothing to sit on
/// there. And an edit carries its own row, so *"the rest of the row stays the data cursor"*
/// is not a rule a caller can forget to honour: editing a row that is not the cursor row is
/// not a value of this type.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Focus {
    /// No row carries the cursor — the unfocused zone of §3, where nothing competes.
    None,
    /// The data cursor sits on the nth entry.
    Cursor(usize),
    /// Edit-in-place on the nth entry, which is therefore also the cursor row.
    Editing(usize, Edit),
}

/// The table of settable keys, carrying the data cursor and the editing cell.
pub struct ConfigTable {
    rows: Vec<Row>,
    focus: Focus,
    /// §3 asks the editing cell for a fill **and** an underline. The reviewed frame drew only
    /// the fill, so the pair is rendered as two pantry variants rather than settled by
    /// argument — this is the knob that produces them, not a runtime preference.
    underline_edit: bool,
}

impl ConfigTable {
    pub fn new(rows: Vec<Row>) -> Self {
        Self {
            rows,
            focus: Focus::None,
            underline_edit: true,
        }
    }

    pub fn focus(mut self, focus: Focus) -> Self {
        self.focus = focus;
        self
    }

    pub fn without_edit_underline(mut self) -> Self {
        self.underline_edit = false;
        self
    }

    /// The mode the status line should be showing, if any. One producer for the indicator.
    pub fn edit_mode(&self) -> Option<EditMode> {
        match &self.focus {
            Focus::Editing(_, edit) => Some(edit.mode()),
            _ => None,
        }
    }

    /// The entry the cursor is on, whether or not it is being edited.
    fn cursor_entry(&self) -> Option<usize> {
        match &self.focus {
            Focus::None => None,
            Focus::Cursor(n) => Some(*n),
            Focus::Editing(n, _) => Some(*n),
        }
    }

    fn edit_for(&self, entry: usize) -> Option<&Edit> {
        match &self.focus {
            Focus::Editing(n, edit) if *n == entry => Some(edit),
            _ => None,
        }
    }
}

/// The column header row: CAPS at the header rung — structure, not data (§2).
fn header_line() -> Line<'static> {
    let fields = format!(
        "{:<W_KEY$}{:<W_ACTUAL$}{:<W_DEFAULT$}",
        "KEY", "ACTUAL", "DEFAULT"
    );
    Line::from(vec![
        Span::raw(" ".repeat(MARGIN)),
        Span::styled(fields, Style::default().fg(tokens::header())),
    ])
}

/// Fit a field to its column: pad it out, or cut it back with an ellipsis.
///
/// **Padding alone is not enough, and the reviewed frame could not know that** — every label
/// in it fits. A longer one silently pushes ACTUAL and DEFAULT to the right *on that row
/// only*, so the table stops being a table exactly when a key name is interesting. The grid
/// is what §3 relies on to say "this cell, not that one", so the column wins and the text
/// yields; the canonical dotted id is shown in full in the detail pane, so nothing said here
/// is the only place it is said.
pub(crate) fn fit(text: &str, width: usize) -> String {
    let count = text.chars().count();
    if count <= width {
        return format!("{text:<width$}");
    }
    let kept: String = text.chars().take(width.saturating_sub(1)).collect();
    format!("{kept}…")
}

/// A group header: bold at the margin, no values of its own.
fn group_line(name: &str) -> Line<'static> {
    Line::from(vec![
        Span::raw(" ".repeat(MARGIN)),
        Span::styled(
            name.to_string(),
            tokens::normal_style().add_modifier(Modifier::BOLD),
        ),
    ])
}

/// The style an ACTUAL value is drawn in: weight when it differs from its default (§3),
/// faint when there is no value at all, normal otherwise.
fn actual_style(value: &str, default: &str) -> Style {
    if value == UNSET {
        tokens::faint_style()
    } else if value != default {
        tokens::strong_style()
    } else {
        tokens::normal_style()
    }
}

impl ConfigTable {
    /// The editing cell: the value cell only, filled lighter than the cursor tint so it
    /// reads as *"you are typing HERE"*, with the caret §3 gives the mode.
    ///
    /// Returns the spans and the cells they occupy — **measured from the spans rather than
    /// computed from the value's length**, because the two carets are different widths and a
    /// formula that assumed one of them would silently misalign the DEFAULT column.
    ///
    /// # The fill has no leading pad, and the reviewed frame did
    ///
    /// A pad on both sides makes the lighter fill read as a *cell* rather than as coloured
    /// text, which is why the r06 frame had one. It also pushed the value one column right of
    /// the ACTUAL header, and nothing inside this widget could see that: it takes a full
    /// screen, with the header directly above the row, for a one-column shift to be visible
    /// as a shift rather than as spacing. §6.23 already ruled this case — the column wins and
    /// the text yields — so the fill begins where the column begins and keeps its trailing
    /// cell. The value no longer moves when a row goes into edit.
    fn edit_cell(&self, edit: &Edit, default: &str) -> (Vec<Span<'static>>, usize) {
        let mut style = actual_style(edit.value(), default).bg(tokens::edit_bg());
        if self.underline_edit {
            style = style.add_modifier(Modifier::UNDERLINED);
        }
        let mut spans = caret_spans(edit, style);
        spans.push(Span::styled(" ", style));
        let width = spans.iter().map(|s| s.content.chars().count()).sum();
        (spans, width)
    }

    /// One settable key's row: marker, KEY, ACTUAL, DEFAULT.
    fn entry_line(&self, entry: &Entry, index: usize) -> Line<'static> {
        let on_cursor = self.cursor_entry() == Some(index);

        // The marker column is always present so the columns stay put whether or not the
        // row carries the cursor.
        let mut spans = vec![Span::raw(" ")];
        spans.push(if on_cursor {
            Span::styled("▸", Style::default().fg(tokens::cursor_mark()))
        } else {
            Span::raw(" ")
        });

        let label = format!("{:indent$}{}", "", entry.label, indent = INDENT);
        spans.push(Span::styled(fit(&label, W_KEY), tokens::normal_style()));

        match self.edit_for(index) {
            Some(edit) => {
                let (cell, width) = self.edit_cell(edit, &entry.default);
                spans.extend(cell);
                // The remainder of the ACTUAL field carries no background of its own, so it
                // falls back to the row's — the cursor tint, exactly as §3 asks.
                spans.push(Span::raw(" ".repeat(W_ACTUAL.saturating_sub(width))));
            }
            None => spans.push(Span::styled(
                fit(&entry.actual, W_ACTUAL),
                actual_style(&entry.actual, &entry.default),
            )),
        }

        spans.push(Span::styled(
            fit(&entry.default, W_DEFAULT),
            tokens::faint_style(),
        ));

        let line = Line::from(spans);
        if on_cursor {
            line.style(Style::default().bg(tokens::cursor_bg()))
        } else {
            line
        }
    }
}

impl Widget for ConfigTable {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut lines = vec![header_line()];
        let mut index = 0;

        for row in &self.rows {
            match row {
                Row::Group(name) => lines.push(group_line(name)),
                Row::Entry(entry) => {
                    lines.push(self.entry_line(entry, index));
                    index += 1;
                }
            }
        }

        Paragraph::new(lines).render(area, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "rows",
            ty: "Vec<Row>",
            description: "Group headers and settable keys, in display order",
        },
        PropInfo {
            name: "focus",
            ty: "Focus",
            description: "None | Cursor(entry) | Editing(entry, Edit) — indices count ENTRIES",
        },
    ];

    /// The reviewed frame's own keys, so a preview can be laid beside the r06 PNG.
    fn rows() -> Vec<Row> {
        vec![
            Row::Group("qdrant".into()),
            Row::Entry(Entry::new(
                "URL",
                "http://localhost:6333",
                "http://localhost:6333",
            )),
            Row::Entry(Entry::new("API key", UNSET, UNSET)),
            Row::Group("watcher".into()),
            Row::Entry(Entry::new("Debounce [ms]", "2000", "1500")),
            Row::Group("embedding".into()),
            Row::Entry(Entry::new("model", "all-MiniLM-L6-v2", "all-MiniLM-L6-v2")),
            Row::Entry(Entry::new("batch size", "32", "32")),
        ]
    }

    /// The debounce key — the one the frame edits — as an entry index.
    const DEBOUNCE: usize = 2;

    struct Variant(&'static str, &'static str, fn() -> ConfigTable);

    impl Ingredient for Variant {
        fn group(&self) -> &str {
            "Config Table"
        }
        fn name(&self) -> &str {
            self.0
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::config_table"
        }
        fn description(&self) -> &str {
            self.1
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            (self.2)().render(area, buf);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Variant(
                "Unfocused",
                "No row carries the cursor — §3's unfocused zone, where nothing competes for the eye",
                || ConfigTable::new(rows()),
            )),
            Box::new(Variant(
                "Data Cursor",
                "The cursor role: a subtle row tint plus ▸, deliberately NOT the selector's inverse block",
                || ConfigTable::new(rows()).focus(Focus::Cursor(DEBOUNCE)),
            )),
            Box::new(Variant(
                "Editing, Insert",
                "The third highlight role — a lighter cell inside the cursor row, caret `▏` between characters",
                || {
                    ConfigTable::new(rows())
                        .focus(Focus::Editing(DEBOUNCE, Edit::insert("2000")))
                },
            )),
            Box::new(Variant(
                "Editing, Normal",
                "vim normal: the caret is a reversed block ON a character. Never rendered before — judge it against Insert",
                || {
                    ConfigTable::new(rows())
                        .focus(Focus::Editing(DEBOUNCE, Edit::normal("2000", 1)))
                },
            )),
            Box::new(Variant(
                "Editing, no underline",
                "§3 asks for fill AND underline; the reviewed frame drew only the fill. This is that frame — compare with `Editing, Insert`",
                || {
                    ConfigTable::new(rows())
                        .focus(Focus::Editing(DEBOUNCE, Edit::insert("2000")))
                        .without_edit_underline()
                },
            )),
            Box::new(Variant(
                "Overlong Key",
                "A label wider than its column yields to the grid with `…` — does the truncation read as deliberate?",
                || {
                    ConfigTable::new(vec![
                        Row::Group("watcher".into()),
                        Row::Entry(Entry::new("Debounce [ms]", "2000", "1500")),
                        Row::Entry(Entry::new(
                            "Recursive descent depth limit",
                            "8",
                            "4",
                        )),
                    ])
                    .focus(Focus::Cursor(1))
                },
            )),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::Encoding;
    use crate::terminal::{Endpoints, Rgb};
    use crate::tokens::Palette;
    use ratatui::style::Color;

    struct Restore(Palette, Encoding, Endpoints);

    impl Restore {
        fn dark_truecolor() -> Self {
            let restore = Restore(Palette::current(), Encoding::current(), tokens::endpoints());
            Palette::set(Palette::Derived);
            Encoding::set(Encoding::TrueColor);
            tokens::set_endpoints(Endpoints {
                background: Rgb::new(0x1e, 0x1e, 0x2e),
                foreground: Rgb::new(0xcd, 0xd6, 0xf4),
            });
            restore
        }
    }

    impl Drop for Restore {
        fn drop(&mut self) {
            Palette::set(self.0);
            Encoding::set(self.1);
            tokens::set_endpoints(self.2);
        }
    }

    /// The reviewed frame's own content, so a test failure can be read against the PNG.
    fn rows() -> Vec<Row> {
        vec![
            Row::Group("watcher".into()),
            Row::Entry(Entry::new("Debounce [ms]", "2000", "1500")),
            Row::Group("qdrant".into()),
            Row::Entry(Entry::new(
                "URL",
                "http://localhost:6333",
                "http://localhost:6333",
            )),
            Row::Entry(Entry::new("API key", UNSET, UNSET)),
        ]
    }

    fn render(table: ConfigTable, width: u16, height: u16) -> Buffer {
        let area = Rect::new(0, 0, width, height);
        let mut buf = Buffer::empty(area);
        table.render(area, &mut buf);
        buf
    }

    fn row_text(buf: &Buffer, y: u16) -> String {
        (0..buf.area.width)
            .map(|x| buf.cell((x, y)).expect("cell in area").symbol().to_string())
            .collect()
    }

    fn cell_bg(buf: &Buffer, x: u16, y: u16) -> Color {
        buf.cell((x, y)).expect("cell in area").style().bg.unwrap()
    }

    /// Row 0 is the header, so display row `n + 1` is the nth content row.
    const FIRST_CONTENT_ROW: u16 = 1;

    #[test]
    fn the_data_cursor_tints_its_whole_row_and_marks_it() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(ConfigTable::new(rows()).focus(Focus::Cursor(0)), 80, 8);

        // The first ENTRY is display row 2 — the group header above it is not a datum and
        // cannot carry the cursor.
        let y = FIRST_CONTENT_ROW + 1;
        assert!(
            row_text(&buf, y).starts_with(" ▸"),
            "the §3 marker is missing"
        );
        assert_eq!(cell_bg(&buf, 10, y), tokens::cursor_bg());
        // …and the group header above it is untouched.
        assert_ne!(cell_bg(&buf, 10, FIRST_CONTENT_ROW), tokens::cursor_bg());
    }

    #[test]
    fn a_value_that_differs_from_its_default_is_bold_and_one_that_matches_is_not() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(ConfigTable::new(rows()), 80, 8);

        // `2000` against a default of `1500` — the changed row.
        let changed = FIRST_CONTENT_ROW + 1;
        let x = (MARGIN + W_KEY) as u16;
        assert!(
            buf.cell((x, changed))
                .unwrap()
                .style()
                .add_modifier
                .contains(Modifier::BOLD),
            "a differing value must carry weight (§3)"
        );

        // `http://localhost:6333` against the same default — unchanged.
        let same = FIRST_CONTENT_ROW + 3;
        assert!(
            !buf.cell((x, same))
                .unwrap()
                .style()
                .add_modifier
                .contains(Modifier::BOLD),
            "a matching value must not"
        );
    }

    #[test]
    fn an_unset_value_recedes_rather_than_reading_as_data() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(ConfigTable::new(rows()), 80, 8);
        let x = (MARGIN + W_KEY) as u16;
        let unset = FIRST_CONTENT_ROW + 4;
        assert_eq!(
            buf.cell((x, unset)).unwrap().style().fg,
            Some(tokens::faint())
        );
    }

    #[test]
    fn the_editing_cell_is_lighter_than_the_cursor_tint_it_sits_in() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(
            ConfigTable::new(rows()).focus(Focus::Editing(0, Edit::insert("2000"))),
            80,
            8,
        );
        let y = FIRST_CONTENT_ROW + 1;

        // The value cell carries the edit fill, and it begins exactly where the ACTUAL
        // column begins — the fill marks a cell of the grid rather than sitting one inside
        // it and pushing the value out.
        assert_eq!(cell_bg(&buf, (MARGIN + W_KEY) as u16, y), tokens::edit_bg());
        assert_eq!(
            cell_bg(&buf, (MARGIN + W_KEY + 1) as u16, y),
            tokens::edit_bg()
        );
        // … while the label and the default column stay the data cursor's (§3).
        assert_eq!(cell_bg(&buf, (MARGIN + 3) as u16, y), tokens::cursor_bg());
        assert_eq!(
            cell_bg(&buf, (MARGIN + W_KEY + W_ACTUAL + 1) as u16, y),
            tokens::cursor_bg()
        );
    }

    #[test]
    fn the_insert_caret_is_a_bar_and_the_normal_caret_is_a_block_on_a_character() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let insert = render(
            ConfigTable::new(rows()).focus(Focus::Editing(0, Edit::insert("2000"))),
            80,
            8,
        );
        let y = FIRST_CONTENT_ROW + 1;
        assert!(
            row_text(&insert, y).contains("2000▏"),
            "insert draws a bar after the caret position"
        );

        let normal = render(
            ConfigTable::new(rows()).focus(Focus::Editing(0, Edit::normal("2000", 0))),
            80,
            8,
        );
        assert!(
            !row_text(&normal, y).contains('▏'),
            "normal mode has no bar — the block IS the caret"
        );
        // Caret 0 is the first character of the value, and the value starts at the ACTUAL
        // column. This constant used to carry a `+ 1` — the fill's leading pad, which was
        // also the one-column shift the screen exposed.
        let on_char = (MARGIN + W_KEY) as u16;
        assert!(
            normal
                .cell((on_char, y))
                .unwrap()
                .style()
                .add_modifier
                .contains(Modifier::REVERSED),
            "normal mode reverses the character under the caret"
        );
    }

    #[test]
    fn the_mode_indicator_carries_weight_and_no_hue() {
        // Chris r06 #8: cyan is the selector's, so this may not use it — or any other hue.
        for mode in [EditMode::Insert, EditMode::Normal] {
            let span = mode.indicator_span();
            assert!(span.style.add_modifier.contains(Modifier::BOLD));
            assert_eq!(span.style.fg, None, "the indicator must not be coloured");
        }
    }

    #[test]
    fn the_columns_line_up_whatever_the_labels_are() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // A label longer than any in the reviewed frame, beside a very short one.
        let rows = vec![
            Row::Entry(Entry::new("a", "1", "1")),
            Row::Entry(Entry::new("an extremely long key name", "2", "2")),
        ];
        let buf = render(ConfigTable::new(rows), 80, 4);

        let default_col = (MARGIN + W_KEY + W_ACTUAL) as u16;
        for y in [FIRST_CONTENT_ROW, FIRST_CONTENT_ROW + 1] {
            let text = row_text(&buf, y);
            let at = text.chars().nth(default_col as usize).unwrap();
            assert!(
                at.is_ascii_digit(),
                "the DEFAULT column must start at the same cell on every row, found {at:?}"
            );
        }
    }

    #[test]
    fn a_row_that_goes_into_edit_keeps_its_value_under_the_actual_header() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // The relation is between the row and its OWN header, one row apart on the same
        // grid — which is why this was invisible until the table was rendered on a screen.
        // The header is read out of the buffer rather than computed, so a change to the
        // column arithmetic cannot make both sides agree on a wrong answer.
        let header = row_text(&render(ConfigTable::new(rows()), 80, 8), 0);
        let actual_col = header.find("ACTUAL").expect("the column header") as u16;
        let y = FIRST_CONTENT_ROW + 1;

        let quiet = row_text(
            &render(ConfigTable::new(rows()).focus(Focus::Cursor(0)), 80, 8),
            y,
        );
        let editing = row_text(
            &render(
                ConfigTable::new(rows()).focus(Focus::Editing(0, Edit::insert("2000"))),
                80,
                8,
            ),
            y,
        );

        for (label, text) in [("cursor", &quiet), ("editing", &editing)] {
            assert_eq!(
                text.chars().nth(actual_col as usize),
                Some('2'),
                "the {label} row's value starts under ACTUAL: {text:?}"
            );
        }
    }

    #[test]
    fn the_cursor_survives_an_encoding_that_refuses_colour() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();
        Encoding::set(Encoding::NoColor);

        let buf = render(ConfigTable::new(rows()).focus(Focus::Cursor(0)), 80, 8);
        let y = FIRST_CONTENT_ROW + 1;

        // §3 puts the structural signature first: with no fill available, the ▸ is the
        // whole difference between the cursor row and any other.
        assert!(row_text(&buf, y).starts_with(" ▸"));
    }
}

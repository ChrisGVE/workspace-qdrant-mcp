//! The **tabular record view** — one record as label + value rows.
//!
//! The second of the two view kinds a window can hold (the first is a table,
//! [`crate::panes::list::ListPane`], reused unchanged). Chris, 20:30: *"a list of vertically
//! aligned field name/label (label being a name that replaces an arcane field name, so the
//! user understand what it is), with their content for a given record"*.
//!
//! # It does not invent a geometry: it generalises the config table's
//!
//! [`crate::widgets::config_table`] already draws these three columns — a label, a value in
//! force, and a static third column carrying the value it would otherwise have had — at widths
//! Chris reviewed at r06. This view is that shape with the label generalised beyond a settings
//! key and the value generalised beyond text, so it takes [`config_table::W_KEY`],
//! [`config_table::W_DEFAULT`] and [`config_table::MARGIN`] rather than choosing its own. Two
//! widgets with two sets of numbers is how a config screen and a record window come to put
//! their values on different cells, in a way only a screenshot would ever show. The
//! changed-value mark is the same rule too — [`config_table::actual_style`], derived from the
//! two values rather than carried as a flag that could contradict them.
//!
//! # Two modes, two different marks, and they must not look alike
//!
//! Chris, 20:30: *"In view mode the currently selected field will be highlighted in the same
//! manner as for a table, in edit mode, one the editable field will show their background to
//! signify which are editable, and the active field will have a different background color."*
//!
//! | mode | what the reader is told | mechanism |
//! |---|---|---|
//! | VIEW | *this is the field you are reading* | the ROW takes [`tokens::cursor_bg`] — the same lavender block a table's cursor row takes (ruling 7), because he asked for "the same manner as for a table" |
//! | EDIT | *these are the fields you may change* | each editable VALUE CELL takes [`tokens::field::editable_bg`] plus [`tokens::field::EDITABLE_MARK`] — a SET mark |
//! | EDIT | *this is the one you are changing* | the active value cell takes [`tokens::field::active_bg`], the caret and a `▸` — a POINT mark, a whole emphasis step above the set |
//!
//! The row block and the cell backgrounds are **never both on screen**, which is what keeps
//! the two modes from being a brightness comparison a reader has to measure. In EDIT mode the
//! row block is gone and `▸` marks the active row; in VIEW mode there are no cell backgrounds
//! at all. The title row says which mode it is in words as well, because a mode visible only
//! as colour is invisible under `NO_COLOR` and to a reader who is not looking at a field.
//!
//! # No borders, anywhere
//!
//! Task 1 settles it: *"fields that wrap within their frame (frame might not be visible)"*,
//! and then the edit-mode description, which describes backgrounds and never a box. A
//! multi-line field's "frame" is its background extent — the fill IS the box, so drawing one
//! around it would be the same statement twice.
//!
//! # The third column is TEXT, and its header is CHROME
//!
//! Round 1 gave it a band of its own, reading *"in another color than the main window"* as a
//! surface. Chris ruled that out on 2026-09-14 — the column is informational and does not need
//! a region — so its values are [`tokens::faint`] on the window like everything else, and only
//! its header keeps the table-header style. What makes it readable is not a band but the
//! shorter ladder a window now uses ([`crate::tokens::WindowText`]).
//!
//! The header does **not scroll**, and that is a defect the designer found by rendering the
//! overflow frame: while it was the first entry of the scrollable list, any offset past the
//! top left the third column an unlabelled band of values — and a default is not
//! distinguishable from a pre-edit value by looking at it, which is the whole ambiguity the
//! header exists to remove. It is chrome, so it costs a viewport row and never a scroll row,
//! exactly as [`crate::panes::list::ListPane`] has always treated its own column-title row.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Clear, Paragraph, Widget},
};

use crate::panes::cell::EMPTY;
use crate::tokens;
use crate::widgets::config_table::{self, fit};
use crate::widgets::edit_field::{caret_spans_with, Caret, Edit};

#[cfg(test)]
mod tests;

/// The label column, the value column's neighbour, and the static third column — all three
/// taken from the config table rather than chosen again. See the module docs.
pub const W_LABEL: usize = config_table::W_KEY;
pub const W_REFERENCE: usize = config_table::W_DEFAULT;
/// The margin at the head of every row, where the EDIT-mode row mark goes.
pub const GUTTER: usize = config_table::MARGIN;
/// One cell between the value column and the reference band, so the band reads as a separate
/// sheet rather than as a fourth column of the same table.
const REFERENCE_GAP: usize = 1;
/// The narrowest a value column is allowed to get before the window is simply too narrow.
const MIN_VALUE: usize = 8;

/// A tick box, drawn as SHAPE so it survives a colourless terminal (r06 #8).
const TICKED: &str = "[\u{2713}]";
const UNTICKED: &str = "[ ]";
/// A radio button and its empty twin. The label goes to the RIGHT of each (Chris).
const PICKED: &str = "(\u{25cf})";
const UNPICKED: &str = "( )";
/// The mark that says a field opens a list — the affordance hue, because that is already what
/// *a thing you may press* wears.
const OPENS: &str = "\u{25be}";
/// The EDIT-mode row mark. VIEW mode has none: the block is the mark, and a glyph beside a
/// block is the same statement twice.
const AT_ROW: &str = "\u{25b8}";
/// Spelling of an absent reference value.
const NO_REFERENCE: &str = "\u{2014}";

/// What a field holds.
///
/// `Image` is named by the ruling and deferred by it (*"possibly image, but that's for future
/// development"*), so it is not here — a variant nothing can draw is a state the language can
/// express and the renderer cannot.
#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    /// Integer or float, already formatted by whoever knows the unit.
    Number(String),
    /// A single line of text.
    Text(String),
    /// A multi-line box, wrapped to the value column — here, because only the renderer knows
    /// how wide that column turned out to be.
    Multi(String),
    Bool(bool),
    /// Few choices, all shown in place, each label to the right of its button.
    ///
    /// **One model at two sizes with [`Value::Choice`]**, per the 21:08 ruling: the same choice
    /// set and the same pick semantics, differing only in presentation. The caller picks the
    /// presentation rather than the renderer inferring it from the count, because Chris called
    /// the four-choice threshold *"a rule of thumb but not universal"* — a renderer that
    /// branched on the count would have made it universal.
    Radio {
        choices: Vec<String>,
        at: usize,
    },
    /// The same radio group **stacked one per row** — item 3, and the caller's choice for the
    /// same reason the row form is.
    ///
    /// Chris: *"be prepared to show your radio button in a column, when selected the whole
    /// column gets the highlighting background, and changing the selection is done with up/down
    /// arrow or k/j."* So the two forms differ in three things at once — layout, highlight
    /// extent, and the keys that move within them — which is why it is a separate variant
    /// rather than a flag on [`Value::Radio`]: a flag would have left the keymap to be inferred
    /// somewhere else.
    RadioColumn {
        choices: Vec<String>,
        at: usize,
    },
    /// Many choices: one displayed value, `↵`/Space opens a [`DropDown`].
    Choice {
        choices: Vec<String>,
        at: usize,
    },
}

impl Value {
    /// The choices and the current pick, for the two variants that have them.
    pub fn choices(&self) -> Option<(&[String], usize)> {
        match self {
            Value::Radio { choices, at }
            | Value::RadioColumn { choices, at }
            | Value::Choice { choices, at } => Some((choices.as_slice(), *at)),
            _ => None,
        }
    }

    /// The text an editor would be handed for this field, or [`None`] where the field is not
    /// typed into at all.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Number(t) | Value::Text(t) | Value::Multi(t) => Some(t),
            _ => None,
        }
    }

    /// How many rows this value occupies at `width`. Only [`Value::Multi`] and a wrapping
    /// [`Value::Radio`] are ever more than one.
    pub fn rows(&self, width: usize) -> usize {
        match self {
            Value::Multi(text) => wrap(text, width).len().max(1),
            // A radio WRAPS rather than elides. A choice truncated away is a choice the reader
            // can neither make nor know about, and recognition over recall is the entire reason
            // few choices are drawn in place instead of behind a drop-down.
            Value::Radio { choices, at } => radio_rows(choices, *at, width).len().max(1),
            // One row per choice, always — that is what makes it a column. It does not wrap and
            // it does not elide, for the same reason the row form does not.
            Value::RadioColumn { choices, .. } => choices.len().max(1),
            _ => 1,
        }
    }
}

/// One row of the record: a human label, a value, an optional reference, and whether this
/// window lets the reader change it.
///
/// Named `FieldRow` and not `Field` because [`crate::editor::Field`] is the live editor under
/// a text input, and one crate with two `Field`s is one crate where a reader has to check
/// which one a signature means.
#[derive(Clone, PartialEq, Debug)]
pub struct FieldRow {
    label: String,
    value: Value,
    reference: Option<String>,
    editable: bool,
}

impl FieldRow {
    pub fn new(label: impl Into<String>, value: Value) -> Self {
        Self {
            label: label.into(),
            value,
            reference: None,
            editable: true,
        }
    }

    /// The third column's cell for this row — a default, or the pre-edit value.
    pub fn reference(mut self, text: impl Into<String>) -> Self {
        self.reference = Some(text.into());
        self
    }

    /// Editability is *"defined window by window"* (Chris), so it is declared per field by the
    /// composing view rather than inferred from the value's type.
    pub fn read_only(mut self) -> Self {
        self.editable = false;
        self
    }

    pub fn is_editable(&self) -> bool {
        self.editable
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    /// The value, to be changed by a keystroke.
    ///
    /// `pub(crate)` rather than `pub`: a field's value is edited through
    /// [`super::keys`], which knows which keys each kind answers to. A caller
    /// reaching in directly is a second place field semantics would live.
    pub(crate) fn value_mut(&mut self) -> &mut Value {
        &mut self.value
    }

    /// Whether this row's value differs from the reference beside it — the config table's own
    /// rule, so the two surfaces cannot come to mark a changed value differently.
    pub fn is_changed(&self) -> bool {
        match (self.value.as_text(), &self.reference) {
            (Some(value), Some(reference)) => config_table::is_changed(value, reference),
            _ => false,
        }
    }
}

/// Which mode the window is in, and where its attention is.
#[derive(Clone, PartialEq, Debug)]
pub enum Mode {
    /// Reading. `at` is the field under the data cursor.
    View { at: usize },
    /// Changing. `at` is the ACTIVE field; `edit` is its live snapshot while a text or numeric
    /// field is being typed into. A still frame builds the [`Edit`] directly; a live window
    /// derives one from [`crate::editor::Field::edit`].
    Edit { at: usize, edit: Option<Edit> },
}

impl Mode {
    pub fn at(&self) -> usize {
        match self {
            Mode::View { at } | Mode::Edit { at, .. } => *at,
        }
    }

    pub fn editing(&self) -> bool {
        matches!(self, Mode::Edit { .. })
    }
}

/// The edit-mode background pair, as arms so the choice is rendered rather than argued.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Scheme {
    /// **A+, the gate's call.** Two neutral rungs plus [`tokens::field::EDITABLE_MARK`]: the
    /// set lifts off the window, the point lifts off the set by more again, and the underline
    /// carries the set where the lift is too thin to — which on four of the fifteen bundled
    /// themes is always. Costs no hue. There is no arm without the underline: see
    /// [`tokens::field::SetMark`].
    #[default]
    Neutral,
    /// **B, the fallback.** The set is the layer fill washed toward `accent`; the point stays a
    /// neutral rung. Two different mechanisms, which is the appeal — and, inside an
    /// accent-tinted window, the same surface treatment twice, which is the objection. Its
    /// best case is a NEUTRAL window, which Chris has already called *"washed out and sad"*.
    AccentWash,
    /// **C, rejected.** The set is [`tokens::selection_bg`] and the point is the cursor block
    /// — which is the VIEW-mode mark, so the two modes come out looking identical. Kept
    /// reachable so the pantry can show that they do.
    SelectionDerived,
}

impl Scheme {
    fn set_bg(self, layer: Color) -> Color {
        match self {
            Scheme::Neutral => tokens::field::editable_bg(),
            Scheme::AccentWash => tokens::field::accent_wash(layer),
            Scheme::SelectionDerived => tokens::selection_bg(),
        }
    }

    fn point_bg(self) -> Color {
        match self {
            Scheme::Neutral | Scheme::AccentWash => tokens::field::active_bg(),
            Scheme::SelectionDerived => tokens::cursor_bg(),
        }
    }

    /// Arm C puts black-bold text on its point, because its point IS the cursor block.
    fn point_fg(self) -> Option<Color> {
        match self {
            Scheme::SelectionDerived => tokens::cursor_fg(),
            _ => None,
        }
    }
}

/// How the optional third column is drawn.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reference {
    /// No third column at all — and therefore no header row, which gives that row back to
    /// data. Columns one and two have no title (Chris), so a two-column record has no header.
    None,
    /// **A** — the column's values at [`tokens::faint`], on the window's own surface. Reads as
    /// "same colour, dimmer", which is emphasis rather than another colour.
    Text(&'static str),
}

impl Reference {
    fn header(self) -> Option<&'static str> {
        match self {
            Reference::None => None,
            Reference::Text(h) => Some(h),
        }
    }
}

/// One record, as label + value rows.
pub struct RecordView {
    fields: Vec<FieldRow>,
    mode: Mode,
    reference: Reference,
    scheme: Scheme,
    /// The layer fill this view sits on, so [`Scheme::AccentWash`] has something to blend.
    layer: Color,
    /// First visible DATA row — the container owns the scrollbar, this owns the window onto
    /// the rows. The header is not counted here; see the module docs.
    offset: usize,
    /// Who draws the caret. Under the conventional keymap the terminal does, so the view paints
    /// none — see [`crate::widgets::edit_field::Caret`].
    caret: Caret,
}

impl RecordView {
    pub fn new(fields: Vec<FieldRow>, mode: Mode) -> Self {
        Self {
            fields,
            mode,
            reference: Reference::None,
            scheme: Scheme::Neutral,
            layer: tokens::layer1_bg(),
            offset: 0,
            caret: Caret::default(),
        }
    }

    pub fn reference(mut self, reference: Reference) -> Self {
        self.reference = reference;
        self
    }

    pub fn scheme(mut self, scheme: Scheme) -> Self {
        self.scheme = scheme;
        self
    }

    pub fn layer(mut self, layer: Color) -> Self {
        self.layer = layer;
        self
    }

    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Which keymap's caret this view paints.
    pub fn caret(mut self, caret: Caret) -> Self {
        self.caret = caret;
        self
    }

    pub fn fields(&self) -> &[FieldRow] {
        &self.fields
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Rows of CHROME the view draws before its data: the reference column's header, when
    /// there is one and there is data under it.
    ///
    /// A header over no column is chrome for absent data, so an empty record draws none.
    pub fn header_rows(&self) -> usize {
        usize::from(self.reference.header().is_some() && !self.fields.is_empty())
    }

    /// The SCROLLABLE rows this record needs at `width` — what the container's scrollbar is a
    /// fraction of. The header is not among them: it is chrome and does not scroll.
    pub fn rows(&self, width: u16) -> usize {
        if self.fields.is_empty() {
            // The one line saying so. Counted, so a container never offers to scroll it.
            return 1;
        }
        let value_width = self.value_width(width);
        self.fields
            .iter()
            .map(|f| f.value.rows(value_width))
            .sum::<usize>()
    }

    /// The rows a view may put data on, inside a viewport of `height` rows.
    pub fn data_height(&self, height: u16) -> u16 {
        height.saturating_sub(self.header_rows() as u16)
    }

    fn has_reference(&self) -> bool {
        self.reference.header().is_some()
    }

    /// The value column's width inside a viewport `width` cells wide.
    ///
    /// Public because a live record has to know it off the render path: the anchor a drop-down
    /// opens out of is the value CELL, and the only thing that knows how wide that is, is the
    /// view that would have drawn it.
    pub fn value_width_at(&self, width: u16) -> usize {
        self.value_width(width)
    }

    fn value_width(&self, width: u16) -> usize {
        let reserved = GUTTER
            + W_LABEL
            + if self.has_reference() {
                W_REFERENCE + REFERENCE_GAP
            } else {
                0
            };
        (width as usize).saturating_sub(reserved).max(MIN_VALUE)
    }

}

/// One radio button and its label, as it is drawn.
fn radio_button(choice: &str, picked: bool) -> String {
    format!("{} {choice}", if picked { PICKED } else { UNPICKED })
}

/// One packed radio row, with the ACTIVE button bold and the rest at normal weight.
///
/// Item 3 asks for the weight to move with the selection, so the row cannot be one span. Three
/// spans — before, the active button, after — and the split is found by locating the active
/// button's rendered text inside the line this row actually holds. A button that is not on this
/// row (the group wrapped) leaves the line as one unweighted span, which is correct: there is no
/// active button here to embolden.
fn bolden_active(
    line: &str,
    choices: &[String],
    at: usize,
    width: usize,
    style: Style,
) -> Vec<Span<'static>> {
    let padded = fit(line, width);
    let active = choices
        .get(at)
        .map(|choice| radio_button(choice, true))
        .unwrap_or_default();
    let Some(start) = padded.find(&active).filter(|_| !active.is_empty()) else {
        return vec![Span::styled(padded, style)];
    };
    let end = start + active.len();
    vec![
        Span::styled(padded[..start].to_string(), style),
        Span::styled(
            padded[start..end].to_string(),
            style.add_modifier(Modifier::BOLD),
        ),
        Span::styled(padded[end..].to_string(), style),
    ]
}

/// The radio's buttons packed into rows no wider than `width`, never elided.
fn radio_rows(choices: &[String], at: usize, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for (i, choice) in choices.iter().enumerate() {
        let button = radio_button(choice, i == at);
        let candidate = if line.is_empty() {
            button.clone()
        } else {
            format!("{line}   {button}")
        };
        if candidate.chars().count() > width && !line.is_empty() {
            out.push(std::mem::take(&mut line));
            line = button;
        } else {
            line = candidate;
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

/// Break `text` into lines no wider than `width`, on spaces where it can.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        let candidate = if line.is_empty() {
            word.to_string()
        } else {
            format!("{line} {word}")
        };
        if candidate.chars().count() > width && !line.is_empty() {
            out.push(std::mem::take(&mut line));
            line = word.to_string();
        } else {
            line = candidate;
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

impl RecordView {
    /// The spans of one value cell, padded to `width` and wearing `style`.
    fn value_spans(
        &self,
        value: &Value,
        row: usize,
        width: usize,
        style: Style,
    ) -> Vec<Span<'static>> {
        match value {
            Value::Number(text) | Value::Text(text) => vec![Span::styled(fit(text, width), style)],
            Value::Multi(text) => {
                let lines = wrap(text, width);
                let line = lines.get(row).cloned().unwrap_or_default();
                vec![Span::styled(fit(&line, width), style)]
            }
            Value::Bool(on) => {
                let mark = if *on { TICKED } else { UNTICKED };
                let word = if *on { "yes" } else { "no" };
                vec![Span::styled(fit(&format!("{mark} {word}"), width), style)]
            }
            Value::Radio { choices, at } => {
                let lines = radio_rows(choices, *at, width);
                let line = lines.get(row).cloned().unwrap_or_default();
                // **The active button is BOLD** — item 3: *"when selected the radio button that
                // is active is bolden … and the newly selected gets bolden while the unselected
                // becomes normal."* Drawn as three spans so the weight lands on the active
                // button alone rather than on the whole packed row.
                bolden_active(&line, choices, *at, width, style)
            }
            // The column form. Its highlight is the whole column and is painted by the caller
            // (see `RecordView::render`), because a background that covered only the buttons
            // would be a highlight on the glyphs rather than on the field.
            Value::RadioColumn { choices, at } => {
                let line = choices
                    .get(row)
                    .map(|choice| radio_button(choice, row == *at))
                    .unwrap_or_default();
                let mut style = style;
                if row == *at {
                    style = style.add_modifier(Modifier::BOLD);
                }
                vec![Span::styled(fit(&line, width), style)]
            }
            Value::Choice { choices, at } => {
                let shown = choices.get(*at).cloned().unwrap_or_default();
                // The chevron sits at the right edge of the cell, so a column of choice fields
                // lines its affordances up rather than scattering them mid-row.
                let room = width.saturating_sub(2);
                vec![
                    Span::styled(fit(&shown, room), style),
                    Span::styled(OPENS, style.fg(tokens::field::affordance())),
                    Span::styled(" ", style),
                ]
            }
        }
    }

    /// The value cell's style for this field, under this mode and scheme.
    fn cell_style(&self, index: usize, field: &FieldRow) -> Style {
        // In VIEW mode the value carries only the changed-value weight; the row block is the
        // whole highlight, and a second one under it would be two marks for one fact.
        if !self.mode.editing() {
            return match &field.reference {
                Some(reference) => {
                    config_table::actual_style(field.value.as_text().unwrap_or_default(), reference)
                }
                None => tokens::normal_style(),
            };
        }
        if !field.editable {
            // Not editable, and it says so by staying on the window's own surface at the
            // metadata rung — there is nothing here to mistake for a slot.
            return tokens::muted_style();
        }
        if index == self.mode.at() {
            let mut style = tokens::normal_style().bg(self.scheme.point_bg());
            if let Some(fg) = self.scheme.point_fg() {
                style = style.fg(fg).add_modifier(Modifier::BOLD);
            }
            // **Item 3's black text.** The POINT fill is derived to carry it
            // (`tokens::field::FieldRungs`), so the foreground comes from the same place rather
            // than being spelled here — a cell that hard-coded black would be a cell that stayed
            // black on the themes where the derivation had to give it up.
            if self.scheme == Scheme::Neutral
                && tokens::field::FieldRungs::current() == tokens::field::FieldRungs::BlackText
            {
                style = style
                    .fg(tokens::field::active_fg())
                    .add_modifier(Modifier::BOLD);
            }
            return style;
        }
        // The SET mark. Its modifier is the encoding's business, not this widget's — see
        // `tokens::field::editable_modifier`.
        // The underline is `tokens::field::SetMark`'s call and nothing else's: Chris ruled it
        // out of the look, and it comes back only where the encoding collapses the ladder and
        // the fill stops saying anything. A flag here would have been a second place to decide
        // that, which is how a window comes to disagree with the token that owns the question.
        tokens::normal_style()
            .bg(self.scheme.set_bg(self.layer))
            .add_modifier(tokens::field::editable_modifier())
    }

    /// The reference column's header row — the only column that carries one.
    fn draw_header(&self, at: Rect, buf: &mut Buffer) {
        let Some(header) = self.reference.header() else {
            return;
        };
        let lead = at.width as usize - W_REFERENCE.min(at.width as usize);
        Paragraph::new(Line::from(vec![
            Span::raw(" ".repeat(lead)),
            Span::styled(fit(header, W_REFERENCE), tokens::header_style()),
        ]))
        .render(at, buf);
    }
}

impl Widget for RecordView {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() || area.height == 0 {
            return;
        }
        let value_width = self.value_width(area.width);

        if self.fields.is_empty() {
            // `No data` — the word the table already uses. One fact should not have two
            // spellings, and there is no header, because a header over no column is chrome
            // for data that is not there.
            Paragraph::new(Line::from(vec![
                Span::raw(" ".repeat(GUTTER)),
                Span::styled(EMPTY, tokens::faint_style()),
            ]))
            .render(Rect { height: 1, ..area }, buf);
            return;
        }

        let header_rows = self.header_rows() as u16;
        if header_rows > 0 {
            self.draw_header(Rect { height: 1, ..area }, buf);
        }
        let body = Rect {
            y: area.y + header_rows,
            height: area.height.saturating_sub(header_rows),
            ..area
        };
        if body.height == 0 {
            return;
        }

        // Every data row of the view, flattened: (field index, row within that field).
        let mut rows: Vec<(usize, usize)> = Vec::new();
        for (index, field) in self.fields.iter().enumerate() {
            for row in 0..field.value.rows(value_width) {
                rows.push((index, row));
            }
        }

        for (screen_row, (index, row)) in rows
            .iter()
            .skip(self.offset)
            .take(body.height as usize)
            .enumerate()
        {
            let line_rect = Rect {
                y: body.y + screen_row as u16,
                height: 1,
                ..body
            };

            let field = &self.fields[*index];
            let on_cursor = *index == self.mode.at();
            let mut spans: Vec<Span<'static>> = Vec::new();

            // The EDIT-mode row mark. VIEW mode has none — the block is the mark.
            spans.push(if self.mode.editing() && on_cursor && *row == 0 {
                Span::styled(
                    format!("{AT_ROW} "),
                    Style::default().fg(tokens::cursor_mark()),
                )
            } else {
                Span::raw(" ".repeat(GUTTER))
            });

            // A continuation row of a multi-line value repeats no label. Two cells short of
            // the column and then padded back out to it: a label long enough to be elided must
            // still leave a gap before its value, or the ellipsis reads as part of the value.
            let label = if *row == 0 { field.label.as_str() } else { "" };
            spans.push(Span::styled(
                format!("{:<W_LABEL$}", fit(label, W_LABEL - 2)),
                tokens::normal_style(),
            ));

            let style = self.cell_style(*index, field);
            // The live caret exists only on the active cell of a text or numeric field.
            let caret = match (&self.mode, field.editable) {
                (
                    Mode::Edit {
                        at,
                        edit: Some(edit),
                    },
                    true,
                ) if *at == *index => Some(edit),
                _ => None,
            };
            match caret {
                Some(edit) => {
                    // Three spans, always, plus one pad — `caret_spans` is not generalised and
                    // the column arithmetic downstream relies on that.
                    let mut cell = caret_spans_with(edit, style, self.caret);
                    let used: usize = cell.iter().map(|s| s.content.chars().count()).sum();
                    cell.push(Span::styled(
                        " ".repeat(value_width.saturating_sub(used)),
                        style,
                    ));
                    spans.extend(cell);
                }
                None => spans.extend(self.value_spans(&field.value, *row, value_width, style)),
            }

            if self.has_reference() {
                spans.push(Span::raw(" ".repeat(REFERENCE_GAP)));
                let text = if *row == 0 {
                    field
                        .reference
                        .clone()
                        .unwrap_or_else(|| NO_REFERENCE.into())
                } else {
                    String::new()
                };
                spans.push(Span::styled(fit(&text, W_REFERENCE), tokens::faint_style()));
            }

            Paragraph::new(Line::from(spans)).render(line_rect, buf);

            // VIEW mode's block, last and over the row — "the same manner as for a table"
            // means the same mechanism, which is a fill UNDER the row rather than a mark beside
            // it. How far it runs is [`CursorExtent`]'s.
            if on_cursor && !self.mode.editing() {
                // **The whole row** (Chris, item: full-row block confirmed). With the
                // third column's band gone there is nothing left for the block to stop short
                // of, so the extent is no longer a question a caller can be asked.
                let block = line_rect;
                buf.set_style(block, Style::default().bg(tokens::cursor_bg()));
                if let Some(fg) = tokens::cursor_fg() {
                    buf.set_style(block, Style::default().fg(fg).add_modifier(Modifier::BOLD));
                }
            }
        }
    }
}

/// The drop-down a [`Value::Choice`] field opens (Chris, 21:08: *"Drop down list, so A"*).
///
/// A list on the layer above the window — the same layer the discard guard uses — with the
/// cursor on the current value. It reuses the cursor block rather than inventing a highlight:
/// inside this list the highlighted line IS the data cursor.
pub struct DropDown {
    choices: Vec<String>,
    at: usize,
    /// The value cell it came out of, so it reads as the field opening.
    anchor: Rect,
    /// **Round 2's whole arm**, as one flag rather than three.
    ///
    /// Item 3 changes the list in three ways at once — the frame goes, the current value moves
    /// to the top with the rest sorted under it, and the cursor therefore starts on row 0 — and
    /// they are one decision, not three: sorting the list without moving the cursor would put
    /// the cursor on whatever sorted into the current value's old index, which is a different
    /// value. Three separate flags would let a caller build exactly that.
    item_three: bool,
    /// What the reader has typed, which **replaces** the value and narrows the list.
    filter: Option<String>,
}

/// The order item 3 asks for: *"The currently selected value should be the first on the list,
/// but the rest of the dropdown list should be sorted."*
///
/// Two different jobs in one list, and the order says so: the first row is *what this field is
/// now*, and everything under it is *what else it could be*, in the order a reader can search.
/// Sorting the current value in with the rest would make the reader hunt for the row they are
/// standing on.
pub fn current_first_then_sorted(choices: &[String], at: usize) -> Vec<String> {
    let Some(current) = choices.get(at) else {
        let mut rest = choices.to_vec();
        rest.sort();
        return rest;
    };
    let mut rest: Vec<String> = choices
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != at)
        .map(|(_, c)| c.clone())
        .collect();
    rest.sort();
    std::iter::once(current.clone()).chain(rest).collect()
}

/// Subsequence matching, case-insensitive — *"the list is reduced via fuzzy matching"*.
///
/// A subsequence rather than a substring, which is what "fuzzy" means to anyone who has used a
/// fuzzy finder: `tsf` finds `tree-sitter/function`. A substring match would refuse that and the
/// reader would conclude the filter is broken rather than that it is strict.
pub fn fuzzy_matches(choice: &str, term: &str) -> bool {
    let mut haystack = choice.chars().flat_map(char::to_lowercase);
    term.chars()
        .flat_map(char::to_lowercase)
        .all(|wanted| haystack.any(|c| c == wanted))
}

impl DropDown {
    pub fn new(choices: Vec<String>, at: usize, anchor: Rect) -> Self {
        Self {
            choices,
            at,
            anchor,
            item_three: false,
            filter: None,
        }
    }

    /// Item 3's list: no frame, current value first then sorted, cursor on the top row.
    pub fn item_three(mut self) -> Self {
        self.item_three = true;
        self
    }

    /// The reader's typed term, narrowing the list. Implies [`DropDown::item_three`] — a filter
    /// is one of its clauses and has no meaning in round 1's list.
    pub fn filter(mut self, term: impl Into<String>) -> Self {
        self.item_three = true;
        self.filter = Some(term.into());
        self
    }

    /// The rows this list shows, in order, under the filter in force.
    pub fn visible(&self) -> Vec<String> {
        if !self.item_three {
            return self.choices.clone();
        }
        let ordered = current_first_then_sorted(&self.choices, self.at);
        match &self.filter {
            None => ordered,
            Some(term) => ordered
                .into_iter()
                .filter(|choice| fuzzy_matches(choice, term))
                .collect(),
        }
    }

    /// Which visible row the cursor is on.
    ///
    /// Round 1 puts it on the current value wherever that sits in declaration order; item 3 puts
    /// the current value on top, so the two answers coincide in meaning and differ in number.
    fn cursor_row(&self) -> usize {
        if self.item_three {
            0
        } else {
            self.at
        }
    }

    /// **As wide as the cell it came out of**, floored there and grown only for a longer
    /// choice.
    ///
    /// The first cut was a fixed 28 cells and left the field's own `▾` peeking out beside an
    /// open list, saying *still closed*. Covering its own chevron is what makes the list read
    /// as the field opening rather than as a box that happens to be nearby.
    pub fn rect(&self, bounds: Rect) -> Rect {
        let rows = self.visible();
        let text = rows
            .iter()
            .map(|c| c.chars().count() as u16)
            .max()
            .unwrap_or(0);
        // The frame costs two cells each way; without it the list is its content plus the one
        // cell of lead every row carries. Measured from the SAME arm that draws, so a frameless
        // list is not silently given a bordered list's room.
        let padding = if self.item_three { 2 } else { 4 };
        let chrome = if self.item_three { 0 } else { 2 };
        let width = (text + padding).max(self.anchor.width).min(bounds.width);
        let height = (rows.len() as u16 + chrome).min(bounds.height.saturating_sub(2));
        Rect {
            x: self.anchor.x.min(bounds.right().saturating_sub(width)),
            y: (self.anchor.y + 1).min(bounds.bottom().saturating_sub(height)),
            width,
            height,
        }
    }
}

impl Widget for DropDown {
    fn render(self, bounds: Rect, buf: &mut Buffer) {
        let rect = self.rect(bounds);
        if rect.width < 4 || rect.height == 0 {
            return;
        }
        Clear.render(rect, buf);
        // The layer above the window, through the crate's one blend — so an open list is never
        // less tinted than the window it opened out of.
        let surface = Style::default().bg(tokens::modal_fill(tokens::layer2_bg()));
        let inner = if !self.item_three {
            Block::bordered()
                .border_style(Style::default().fg(tokens::modal_border()))
                .style(surface)
                .render(rect, buf);
            Rect {
                x: rect.x + 1,
                y: rect.y + 1,
                width: rect.width - 2,
                height: rect.height - 2,
            }
        } else {
            // Item 3: no frame. The list is told from the window by its own layer fill, which is
            // a step lighter, and by the fact that every OTHER field's highlight has gone — so a
            // box around it would be the third statement of a fact already made twice.
            Block::default().style(surface).render(rect, buf);
            rect
        };

        let rows = self.visible();
        // The cursor sits on the first row, which under the item-3 ordering IS the current
        // value — and, once a filter is typed, is the best match rather than a value that may
        // no longer be in the list at all.
        let cursor = self.cursor_row();
        // Scroll so the cursor is visible: a long list scrolls within the drop-down. Under item
        // 3 the cursor is row 0 and this is always 0, which is one of the things moving the
        // current value to the top buys — a list that opens at its own beginning.
        let first = cursor.saturating_sub(inner.height.saturating_sub(1) as usize);
        for (screen_row, choice) in rows
            .iter()
            .skip(first)
            .take(inner.height as usize)
            .enumerate()
        {
            let row = Rect {
                y: inner.y + screen_row as u16,
                height: 1,
                ..inner
            };
            Paragraph::new(Line::from(Span::styled(
                format!(" {}", fit(choice, inner.width as usize - 1)),
                tokens::normal_style(),
            )))
            .render(row, buf);
            if first + screen_row == cursor {
                buf.set_style(row, Style::default().bg(tokens::cursor_bg()));
                if let Some(fg) = tokens::cursor_fg() {
                    buf.set_style(row, Style::default().fg(fg).add_modifier(Modifier::BOLD));
                }
            }
        }
    }
}

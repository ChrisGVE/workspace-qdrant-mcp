//! Screen chrome — the furniture VISUAL-LANGUAGE §6 specifies and no widget owned.
//!
//! Every element in [`crate::widgets`] is a *zone's* content: a list of stores, a table of
//! keys, a floating window. A full screen needs a second vocabulary that belongs to none of
//! them — the rules that divide the zones, the title line, the sub-screen selector, the
//! merged status-and-help line at the foot. They are collected here because a screen is
//! where they are first needed and a second screen will need exactly the same ones.
//!
//! # Nothing here is a box
//!
//! §6 is explicit: zones are divided by horizontal rules, never by boxes, and **a box means
//! a modal or a toast**. So the chrome is rules, spacing and weight. That is also why the
//! module is small: most of the screen's structure is negative space, which costs no widget.
//!
//! # Every state that could be inconsistent is derived
//!
//! [`Attention`] is screen-level rather than per-zone, so *two* focused zones — or a dimmed
//! zone on a screen where nothing is focused — are not values the type can take.
//! [`Freshness`] holds the age and the SLA rather than a `stale` flag, so the word and the
//! number it describes cannot disagree. Both follow the rule the config table arrived at
//! (`config_table::Entry::is_changed`): a mark that can contradict the fact it marks is a
//! flag, and the comparison is the rule.

use std::time::Duration;

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::health::Rollup;
use crate::tokens;
use crate::widgets::config_table::EditMode;

/// What a rule is drawn with. One glyph wide per column, so a rule's width is its cell count.
const RULE: &str = "─";

/// The bar that marks the focused zone's heading (§3). No colour: the mark is structural, and
/// hue on this element would compete with the selector.
const FOCUS_BAR: &str = "▌";

/// Which of §2's two structural greys a rule carries.
///
/// The pair is not decoration. §2 puts the frame rules *lighter* than the internal ones on
/// purpose — the outer pair underlines the screen, the inner ones divide within it — so a
/// screen drawn with one grey loses the difference between its edge and its seams.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Weight {
    /// The top and bottom rules that underline the screen's frame.
    Frame,
    /// A separator between two zones inside the screen.
    Internal,
}

/// A horizontal rule spanning its area's width — §6's only zone divider.
pub struct Rule {
    weight: Weight,
}

impl Rule {
    pub const fn new(weight: Weight) -> Self {
        Self { weight }
    }

    pub const fn frame() -> Self {
        Self::new(Weight::Frame)
    }

    pub const fn internal() -> Self {
        Self::new(Weight::Internal)
    }

    fn colour(&self) -> Color {
        match self.weight {
            Weight::Frame => tokens::rule_frame(),
            Weight::Internal => tokens::rule_internal(),
        }
    }
}

impl Widget for Rule {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }
        Paragraph::new(Line::from(Span::styled(
            RULE.repeat(area.width as usize),
            Style::default().fg(self.colour()),
        )))
        .render(area, buf);
    }
}

/// How old the screen's readings are, and how old they are allowed to get.
///
/// §4: *"Freshness/staleness is right-aligned, muted; past its SLA it turns `[yellow]stale
/// …`"*. Both halves of that comparison are carried, so [`Freshness::is_stale`] is a
/// measurement rather than a claim — a frame reading `updated 18m ago` in muted grey under a
/// one-minute SLA is not constructible.
///
/// **The SLA itself is not this crate's to set** — §7 leaves the freshness SLA open (OQ-6),
/// which is exactly why it is a parameter here instead of a constant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Freshness {
    age: Duration,
    sla: Duration,
}

impl Freshness {
    pub const fn new(age: Duration, sla: Duration) -> Self {
        Self { age, sla }
    }

    pub fn is_stale(&self) -> bool {
        self.age > self.sla
    }

    /// The right-aligned span: muted while fresh, and the degraded hue once it is not.
    fn span(&self) -> Span<'static> {
        if self.is_stale() {
            Span::styled(
                format!("stale — {} ago", format_age(self.age)),
                Style::default().fg(tokens::Health::Degraded.color()),
            )
        } else {
            Span::styled(
                format!("updated {} ago", format_age(self.age)),
                tokens::muted_style(),
            )
        }
    }
}

/// An age in the coarsest unit that still says something: `4s`, `18m`, `2h`, `3d`.
///
/// Coarse on purpose. The number is read peripherally to answer *"is this recent?"*, and a
/// second of precision on an eighteen-minute age answers a question nobody asked.
pub fn format_age(age: Duration) -> String {
    let secs = age.as_secs();
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3_599 => format!("{}m", secs / 60),
        3_600..=86_399 => format!("{}h", secs / 3_600),
        _ => format!("{}d", secs / 86_400),
    }
}

/// The screen's name on its own line, with the freshness right-aligned against it (§6).
pub struct TitleLine {
    title: String,
    freshness: Option<Freshness>,
}

impl TitleLine {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            freshness: None,
        }
    }

    pub fn freshness(mut self, freshness: Freshness) -> Self {
        self.freshness = Some(freshness);
        self
    }
}

impl Widget for TitleLine {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }

        // Bold at the normal rung — the same treatment a config group header carries, which
        // is this crate's established way of saying "a structural name, not a datum". Strong
        // is reserved for the one value that must be seen, and a title is never that.
        let title = Span::styled(
            self.title.clone(),
            tokens::normal_style().add_modifier(Modifier::BOLD),
        );

        let mut spans = vec![title];
        if let Some(freshness) = self.freshness {
            let right = freshness.span();
            let used = self.title.chars().count() + right.content.chars().count();
            // A title and a freshness that together outrun the line lose the gap, not the
            // freshness: the age is the half that changes, so it is the half worth keeping.
            let gap = (area.width as usize).saturating_sub(used).max(1);
            spans.push(Span::raw(" ".repeat(gap)));
            spans.push(right);
        }

        Paragraph::new(Line::from(spans)).render(area, buf);
    }
}

/// The Service hub's sub-screen selector — §3's *"identical inverse block"*.
///
/// The same mechanism as the tab bar, minus the number: a tab's leading digit is a jump hint
/// and there is no digit to jump to here. Sharing the mechanism is goal 3 — if it is a cyan
/// block, it is what you have selected, on every screen and at every level.
pub struct PaneSelector {
    panes: Vec<String>,
    active: usize,
}

impl PaneSelector {
    pub fn new(panes: Vec<String>, active: usize) -> Self {
        Self { panes, active }
    }
}

impl Widget for PaneSelector {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let mut spans: Vec<Span> = Vec::new();
        for (i, pane) in self.panes.iter().enumerate() {
            if i > 0 {
                spans.push(Span::raw("  "));
            }
            if i == self.active {
                spans.push(Span::styled(
                    format!(" {pane} "),
                    tokens::inverted(tokens::selector()),
                ));
            } else {
                spans.push(Span::styled(pane.clone(), tokens::muted_style()));
            }
        }
        Paragraph::new(Line::from(spans)).render(area, buf);
    }
}

/// Which zone of the screen has the user's attention — a screen-level fact, deliberately.
///
/// §3 gives three treatments (focused, unfocused, and *"no zone focused"* where nothing is
/// dimmed at all), and the third is a property of the screen rather than of any zone. Making
/// it a per-zone flag would make two focused zones representable, and would make a screen
/// where one zone is dimmed and none is focused representable too — both are frames the rule
/// forbids. Passing the same [`Attention`] to every heading removes them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Attention {
    /// The default view: every zone normal, none dimmed, none accented.
    None,
    /// The nth zone is live; every other one recedes.
    Zone(usize),
}

/// The `▌` accent a zone carries when it is the live one, and nothing otherwise.
///
/// Public because the Service hub's lower zone is headed by a [`PaneSelector`] rather than by
/// a [`ZoneHeading`] — §4.1 puts the Config↔Logs toggle where the heading would be — and a
/// second copy of "which zone is accented" is how two zones end up accented at once.
///
/// **Open micro-question for Chris.** §3 writes the accent as a prefix (`▌ Heading`), which
/// shifts the heading text two columns to the right the moment a zone takes focus. A gutter —
/// two columns always reserved, the bar drawn into them — would hold the text still. The
/// prefix is what r02 says, so it is what is rendered; the jitter is visible in the
/// `Service — editing` frame beside the `Service` one.
pub fn accent(index: usize, attention: Attention) -> Option<Span<'static>> {
    match attention {
        Attention::Zone(live) if live == index => Some(Span::styled(
            format!("{FOCUS_BAR} "),
            tokens::normal_style().add_modifier(Modifier::BOLD),
        )),
        _ => None,
    }
}

/// A zone's heading, in the treatment [`Attention`] implies for it.
///
/// # The body is not dimmed, and that is a stated gap
///
/// §3 asks for the *body* of an unfocused zone to recede as well as its heading. The widgets
/// this screen composes have no muted mode — [`crate::widgets::store_health::StoreHealth`]
/// and the rest choose their own rungs — so only the heading carries the state today. The
/// default view (`Attention::None`) is unaffected, since nothing dims there; a frame with a
/// focused zone understates the contrast until the widgets grow the mode.
pub struct ZoneHeading {
    title: String,
    index: usize,
    attention: Attention,
}

impl ZoneHeading {
    pub fn new(title: impl Into<String>, index: usize, attention: Attention) -> Self {
        Self {
            title: title.into(),
            index,
            attention,
        }
    }

    fn spans(&self) -> Vec<Span<'static>> {
        let style = match self.attention {
            // §3's third row: the default view leaves every heading at the baseline.
            Attention::None => tokens::normal_style(),
            Attention::Zone(live) if live == self.index => {
                tokens::normal_style().add_modifier(Modifier::BOLD)
            }
            Attention::Zone(_) => tokens::muted_style(),
        };
        accent(self.index, self.attention)
            .into_iter()
            .chain(std::iter::once(Span::styled(self.title.clone(), style)))
            .collect()
    }
}

impl Widget for ZoneHeading {
    fn render(self, area: Rect, buf: &mut Buffer) {
        Paragraph::new(Line::from(self.spans())).render(area, buf);
    }
}

/// The merged status-and-help line at the foot of the screen (§6).
///
/// Left: the edit-mode indicator when there is one, then §7's single rollup dot. Right: the
/// keys available for what is selected. §4 caps the whole line — *"never more vibrant than
/// the content"* — so only the health glyph carries a hue, and §3 keeps the mode indicator
/// on weight alone because cyan belongs to the selector.
pub struct StatusLine {
    rollup: Rollup,
    mode: Option<EditMode>,
    hints: Vec<(String, String)>,
}

impl StatusLine {
    pub fn new(rollup: Rollup) -> Self {
        Self {
            rollup,
            mode: None,
            hints: Vec::new(),
        }
    }

    /// The vim mode an edit-in-place is in, if one is open. Taken from the table that owns
    /// the edit rather than restated, so the caret and the indicator cannot disagree.
    pub fn mode(mut self, mode: Option<EditMode>) -> Self {
        self.mode = mode;
        self
    }

    pub fn hint(mut self, key: impl Into<String>, label: impl Into<String>) -> Self {
        self.hints.push((key.into(), label.into()));
        self
    }

    fn left(&self) -> Vec<Span<'static>> {
        let mut spans: Vec<Span<'static>> = Vec::new();
        if let Some(mode) = self.mode {
            spans.push(mode.indicator_span());
            spans.push(Span::raw("  "));
        }
        spans.push(Span::styled(
            self.rollup.health.glyph(),
            Style::default().fg(self.rollup.health.color()),
        ));
        spans.push(Span::styled(
            format!(" {}", self.rollup.label),
            tokens::muted_style(),
        ));
        spans
    }
}

impl Widget for StatusLine {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.is_empty() {
            return;
        }

        let left = self.left();
        let left_width: usize = left.iter().map(|s| s.content.chars().count()).sum();
        let hints_width = tokens::key_hints_width(&self.hints);

        let mut spans = left;
        // One clear cell between the two halves is the minimum that still reads as two
        // halves. Below that the hints go entirely: the status is what the line is for, and
        // half a hint row is noise rather than help.
        if hints_width > 0 && left_width + 1 + hints_width <= area.width as usize {
            let gap = area.width as usize - left_width - hints_width;
            spans.push(Span::raw(" ".repeat(gap)));
            spans.extend(tokens::key_hints(&self.hints));
        }

        Paragraph::new(Line::from(spans)).render(area, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::Encoding;
    use crate::terminal::{Endpoints, Rgb};
    use crate::tokens::{Health, Palette};

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

    const AREA: Rect = Rect {
        x: 0,
        y: 0,
        width: 60,
        height: 1,
    };

    fn render(widget: impl Widget) -> Buffer {
        let mut buf = Buffer::empty(AREA);
        widget.render(AREA, &mut buf);
        buf
    }

    fn row(buf: &Buffer, y: u16) -> String {
        (0..AREA.width)
            .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
            .collect()
    }

    fn style_at(buf: &Buffer, x: u16) -> Style {
        buf.cell((x, 0)).expect("cell in area").style()
    }

    #[test]
    fn a_rule_spans_its_whole_width_and_the_frame_is_lighter_than_a_seam() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let frame = render(Rule::frame());
        assert_eq!(
            row(&frame, 0),
            RULE.repeat(AREA.width as usize),
            "a rule divides the whole zone or it is not a divider"
        );

        // §2 puts the frame pair lighter than the internal ones. Measured against the
        // background, because "lighter" inverts on a light theme and "further from the
        // terminal's own background" does not (§6.26).
        let seam = render(Rule::internal());
        let bg = tokens::endpoints().background;
        let distance = |c: Color| match c {
            Color::Rgb(r, g, b) => {
                let d = |x: u8, y: u8| (x as f32 - y as f32).powi(2);
                (d(r, bg.r) + d(g, bg.g) + d(b, bg.b)).sqrt()
            }
            other => panic!("expected RGB under Derived + TrueColor, got {other:?}"),
        };
        let frame_fg = style_at(&frame, 0).fg.expect("a rule is coloured");
        let seam_fg = style_at(&seam, 0).fg.expect("a rule is coloured");
        assert_ne!(frame_fg, seam_fg, "the two weights must be distinguishable");
        assert!(
            distance(frame_fg) > distance(seam_fg),
            "the frame rule stands further off the base than a seam: {frame_fg:?} vs {seam_fg:?}"
        );
    }

    #[test]
    fn an_age_is_named_in_the_coarsest_unit_that_still_says_something() {
        assert_eq!(format_age(Duration::from_secs(0)), "0s");
        assert_eq!(format_age(Duration::from_secs(59)), "59s");
        assert_eq!(format_age(Duration::from_secs(60)), "1m");
        assert_eq!(format_age(Duration::from_secs(3_599)), "59m");
        assert_eq!(format_age(Duration::from_secs(3_600)), "1h");
        assert_eq!(format_age(Duration::from_secs(86_399)), "23h");
        assert_eq!(format_age(Duration::from_secs(86_400)), "1d");
    }

    #[test]
    fn staleness_is_the_comparison_and_cannot_be_stated_against_it() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let sla = Duration::from_secs(60);
        let fresh = Freshness::new(Duration::from_secs(4), sla);
        let stale = Freshness::new(Duration::from_secs(61), sla);
        assert!(!fresh.is_stale() && stale.is_stale());

        // The word and the hue both follow the comparison — there is no third input either
        // could have been set from.
        let fresh_line = row(&render(TitleLine::new("Service").freshness(fresh)), 0);
        assert!(fresh_line.contains("updated 4s ago"), "{fresh_line}");
        assert!(!fresh_line.contains("stale"), "{fresh_line}");

        let stale_buf = render(TitleLine::new("Service").freshness(stale));
        let stale_line = row(&stale_buf, 0);
        assert!(stale_line.contains("stale — 1m ago"), "{stale_line}");
        let x = stale_line.chars().position(|c| c == 's').expect("the word") as u16;
        assert_eq!(
            style_at(&stale_buf, x).fg,
            Some(Health::Degraded.color()),
            "past its SLA the freshness turns the degraded hue"
        );
    }

    #[test]
    fn the_freshness_is_flush_with_the_right_edge() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let line = row(
            &render(TitleLine::new("Service").freshness(Freshness::new(
                Duration::from_secs(4),
                Duration::from_secs(60),
            ))),
            0,
        );
        assert!(
            !line.ends_with(' '),
            "right-aligned means the last cell is used: {line:?}"
        );
        assert!(line.starts_with("Service "), "{line:?}");
    }

    #[test]
    fn exactly_one_pane_is_an_inverse_block() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(PaneSelector::new(vec!["Config".into(), "Logs".into()], 0));
        let selector = tokens::selector();
        let filled: Vec<u16> = (0..AREA.width)
            .filter(|x| style_at(&buf, *x).bg == Some(selector))
            .collect();

        // " Config " — one space each side inside the block, per §3.
        assert_eq!(
            filled.len(),
            8,
            "the inverse block is the selected pane only"
        );
        let line = row(&buf, 0);
        assert!(line.starts_with(" Config   Logs"), "{line:?}");

        // The unselected pane recedes rather than carrying a second block.
        let logs_x = line.find("Logs").expect("both panes are drawn") as u16;
        assert_eq!(style_at(&buf, logs_x).fg, Some(tokens::muted()));
        assert_ne!(style_at(&buf, logs_x).bg, Some(selector));
    }

    #[test]
    fn a_screen_with_no_focused_zone_dims_nothing_and_accents_nothing() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // §3's third row. Two headings are rendered so the assertion is about the SCREEN:
        // with one heading, "nothing is dimmed" is vacuous — there is nothing to dim it
        // relative to.
        for index in 0..2 {
            let buf = render(ZoneHeading::new("Status", index, Attention::None));
            let line = row(&buf, 0);
            assert!(!line.contains(FOCUS_BAR), "no accent: {line:?}");
            assert_eq!(
                style_at(&buf, 0).fg,
                Some(tokens::normal()),
                "no zone dimmed on a screen with no focus"
            );
            assert!(!style_at(&buf, 0).add_modifier.contains(Modifier::BOLD));
        }
    }

    #[test]
    fn focusing_one_zone_accents_it_and_recedes_the_other() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let attention = Attention::Zone(1);

        let live = render(ZoneHeading::new("Config", 1, attention));
        let line = row(&live, 0);
        assert!(line.starts_with("▌ Config"), "{line:?}");
        assert!(
            style_at(&live, 0).add_modifier.contains(Modifier::BOLD),
            "the focused heading is bold"
        );

        let receded = render(ZoneHeading::new("Status", 0, attention));
        assert!(!row(&receded, 0).contains(FOCUS_BAR));
        assert_eq!(style_at(&receded, 0).fg, Some(tokens::muted()));
    }

    #[test]
    fn the_status_line_carries_colour_on_the_glyph_and_nowhere_else() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(
            StatusLine::new(Rollup {
                health: Health::Degraded,
                label: "1 degraded".into(),
            })
            .hint("↵", "edit"),
        );

        assert_eq!(
            style_at(&buf, 0).fg,
            Some(Health::Degraded.color()),
            "the glyph carries the state"
        );
        // Every other painted cell is a neutral. §4: the status line is never more vibrant
        // than the content.
        let hues = [
            Health::Healthy.color(),
            Health::Degraded.color(),
            Health::Offline.color(),
            tokens::selector(),
        ];
        for x in 1..AREA.width {
            let fg = style_at(&buf, x).fg;
            assert!(
                fg.is_none_or(|c| !hues.contains(&c)),
                "column {x} carries a reserved hue"
            );
        }
    }

    #[test]
    fn the_hints_are_flush_right_and_the_rollup_outlives_them() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let line = StatusLine::new(Rollup {
            health: Health::Healthy,
            label: "healthy".into(),
        })
        .hint("j/k", "move")
        .hint("↵", "edit");

        let wide = row(&render(line), 0);
        assert!(wide.starts_with("● healthy"), "{wide:?}");
        assert!(
            wide.ends_with("↵ edit"),
            "the hints sit against the right edge: {wide:?}"
        );

        // Narrow enough that the two halves would overlap. The hints go; the status stays.
        let mut narrow_buf = Buffer::empty(Rect {
            x: 0,
            y: 0,
            width: 14,
            height: 1,
        });
        StatusLine::new(Rollup {
            health: Health::Healthy,
            label: "healthy".into(),
        })
        .hint("j/k", "move")
        .hint("↵", "edit")
        .render(narrow_buf.area, &mut narrow_buf);
        let narrow: String = (0..14)
            .map(|x| narrow_buf.cell((x, 0)).expect("cell in area").symbol())
            .collect();
        assert!(narrow.starts_with("● healthy"), "{narrow:?}");
        assert!(
            !narrow.contains("move"),
            "half a hint row is not help: {narrow:?}"
        );
    }

    #[test]
    fn the_edit_indicator_appears_only_while_an_edit_is_open() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let rollup = Rollup {
            health: Health::Healthy,
            label: "healthy".into(),
        };

        let idle = row(&render(StatusLine::new(rollup.clone())), 0);
        assert!(
            !idle.contains("INSERT") && !idle.contains("NORMAL"),
            "{idle:?}"
        );

        let editing = render(StatusLine::new(rollup).mode(Some(EditMode::Insert)));
        let line = row(&editing, 0);
        assert!(line.starts_with("-- INSERT --"), "{line:?}");
        // §3: bold, no hue — cyan is the selector's and this must not read as a selection.
        assert!(style_at(&editing, 0).add_modifier.contains(Modifier::BOLD));
        assert_ne!(style_at(&editing, 0).fg, Some(tokens::selector()));
    }
}

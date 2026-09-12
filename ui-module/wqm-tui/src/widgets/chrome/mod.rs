//! Screen chrome — the furniture VISUAL-LANGUAGE §6 specifies and no zone owns.
//!
//! Most of [`crate::widgets`] is a *zone's* content: a list of stores, a table of keys, a
//! floating window. A full screen needs a second vocabulary that belongs to none of them —
//! the app bar at the very top, the rules that divide the zones, the title bar, the
//! sub-screen selector, the merged status-and-help line at the foot. They are collected here because a screen is where they
//! are first needed and a second screen will need exactly the same ones.
//!
//! # Chrome is a widget, and that is §16's word
//!
//! The composition model (Chris, 20260731) defines a widget as *"one element of the visual
//! language, **including the screen chrome**"*. So these five live under [`crate::widgets`]
//! with everything else atomic, and each is browsable in the pantry's Widgets tab. They were
//! previously inside `views::chrome`, unregistered and therefore unjudgeable — which is the
//! one thing a design instrument must never make of an element.
//!
//! They stay in one sub-module rather than six loose files because they are a family: a
//! screen takes all of them or none, and their shared vocabulary (§2's two rule weights, §3's
//! focus treatment) is stated once here.
//!
//! # Nothing here is a box
//!
//! §6 is explicit: zones are divided by horizontal rules, never by boxes, and **a box means
//! a modal or a toast**. So the chrome is rules, spacing and weight. That is also why the
//! family is small: most of a screen's structure is negative space, which costs no widget.
//!
//! # Every state that could be inconsistent is derived
//!
//! [`Attention`] is screen-level rather than per-zone, so *two* focused zones — or a dimmed
//! zone on a screen where nothing is focused — are not values the type can take.
//! [`Freshness`] holds the age and the SLA rather than a `stale` flag, so the word and the
//! number it describes cannot disagree. Both follow the rule the config table arrived at
//! ([`crate::widgets::config_table::Entry::is_changed`]): a mark that can contradict the fact
//! it marks is a flag, and the comparison is the rule.

pub mod app_bar;
pub mod freshness;
pub mod keyed;
pub mod pane_selector;
pub mod rule;
pub mod status_line;
pub mod title_bar;
pub mod zone_heading;

pub use app_bar::AppBar;
pub use keyed::keyed_spans;
pub use pane_selector::PaneSelector;
pub use rule::{Rule, Weight};
pub use status_line::StatusLine;
pub use title_bar::{format_age, Freshness, TitleBar};
pub use zone_heading::{accent, Attention, FocusMark, ZoneHeading};

/// Columns of quiet at each edge of a screen.
///
/// Here rather than in a view, because a **pane** needs it too: [`crate::panes`] renders a
/// zone's own heading and selector, and those start on the same column the title bar does. Two
/// constants would be two numbers to keep equal, and the failure is silent — a zone indented
/// one column further than the chrome above it reads as a wobble nobody can name.
///
/// [`crate::widgets::config_table`] keeps its own margin deliberately: it reaches this column
/// through its own arithmetic, which is what lets its KEY column line up under the store roles
/// without either widget knowing about the other.
pub const MARGIN: u16 = 2;

/// A row inset by [`MARGIN`] on both sides.
///
/// Everything except the rules, which underline the whole screen and so run edge to edge.
/// [`inset`], with one column of the LEFT margin handed back.
///
/// For a pane that draws a gutter mark beside its content — today the list's selection bar
/// ([`crate::panes::list::GUTTER`]). The mark then sits in the margin rather than in a column
/// of the table, so the content still begins on [`MARGIN`] and lines up with every other
/// screen. Chris removed a permanent two-column gutter on 2026-09-07 to regain the width; this
/// is what stops a selection quietly handing the bill back.
pub fn with_gutter(area: ratatui::layout::Rect) -> ratatui::layout::Rect {
    let inner = inset(area);
    ratatui::layout::Rect {
        x: inner.x - crate::panes::list::GUTTER,
        width: inner.width + crate::panes::list::GUTTER,
        ..inner
    }
}

pub fn inset(area: ratatui::layout::Rect) -> ratatui::layout::Rect {
    ratatui::layout::Rect {
        x: area.x + MARGIN,
        y: area.y,
        width: area.width.saturating_sub(MARGIN * 2),
        height: area.height,
    }
}

/// The shared scaffolding every chrome test needs: a known palette, a known encoding and
/// known endpoints, restored on drop.
///
/// One copy for the family rather than one per module. Each of these widgets renders a
/// single line against [`crate::tokens`], so they all need the same three process globals
/// pinned, and five transcriptions of the same setup is five chances for one of them to
/// drift into testing a different terminal than the others.
#[cfg(test)]
pub(crate) mod test_support {
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;
    use ratatui::style::Style;
    use ratatui::widgets::Widget;

    use crate::encoding::Encoding;
    use crate::terminal::{Endpoints, Rgb};
    use crate::tokens::{self, Palette};

    /// A wide, one-line area — the shape every chrome element is drawn into.
    pub const AREA: Rect = Rect {
        x: 0,
        y: 0,
        width: 60,
        height: 1,
    };

    pub struct Restore(Palette, Encoding, Endpoints);

    impl Restore {
        /// Mocha's endpoints under `Derived` + truecolor: the combination every measurement
        /// in this crate was taken against.
        pub fn dark_truecolor() -> Self {
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

    /// Render one widget into [`AREA`] and hand back the buffer.
    pub fn render(widget: impl Widget) -> Buffer {
        let mut buf = Buffer::empty(AREA);
        widget.render(AREA, &mut buf);
        buf
    }

    /// One row of a buffer as a string, so an assertion can be written about what is read.
    pub fn row(buf: &Buffer, y: u16) -> String {
        (0..buf.area.width)
            .map(|x| buf.cell((x, y)).expect("cell in area").symbol())
            .collect()
    }

    pub fn style_at(buf: &Buffer, x: u16) -> Style {
        buf.cell((x, 0)).expect("cell in area").style()
    }

    /// The neutral ladder, enumerated from [`crate::tokens`] itself.
    ///
    /// Every rung the vocabulary names, plus `Color::Reset` — which in a ratatui buffer is a
    /// cell nobody painted, i.e. the terminal's own ground. Read off the tokens rather than off
    /// a rendered frame, because a set derived from a render grows to fit whatever that render
    /// did, which is how a guard comes to bless the defect it was written against.
    ///
    /// Here rather than beside one screen's tests because **both** full screens are swept
    /// against it (VL §6), and two enumerations of one ladder is how the two come to disagree
    /// about which rungs exist.
    pub fn neutral_rungs() -> Vec<ratatui::style::Color> {
        vec![
            ratatui::style::Color::Reset,
            tokens::faint(),
            tokens::muted(),
            tokens::normal(),
            tokens::strong(),
            tokens::rule_frame(),
            tokens::rule_internal(),
            tokens::header(),
            tokens::cursor_bg(),
            tokens::cursor_mark(),
            tokens::edit_bg(),
            tokens::layer1_bg(),
            tokens::layer2_bg(),
        ]
    }

    /// Every cell of `buf` whose foreground or background is outside `neutrals`, reported as
    /// `(x, y, which, colour)` — enough to walk from a failure straight to the widget that
    /// painted it, which is what an instrument owes ([`coding.md#measurement`]: surface and
    /// localise, never classify).
    pub fn coloured_cells(
        buf: &Buffer,
        neutrals: &[ratatui::style::Color],
    ) -> Vec<(u16, u16, &'static str, ratatui::style::Color)> {
        let mut found = Vec::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                let style = buf.cell((x, y)).expect("cell in area").style();
                if let Some(fg) = style.fg.filter(|fg| !neutrals.contains(fg)) {
                    found.push((x, y, "fg", fg));
                }
                if let Some(bg) = style.bg.filter(|bg| !neutrals.contains(bg)) {
                    found.push((x, y, "bg", bg));
                }
            }
        }
        found
    }
}

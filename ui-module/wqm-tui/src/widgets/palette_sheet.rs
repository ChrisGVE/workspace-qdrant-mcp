//! Every role in the visual language, every state, one frame per palette.
//!
//! The other ingredients each isolate one element. This one does the opposite: it puts the
//! whole vocabulary on screen at once so the *relationships* can be judged — whether the
//! selector still beats an alarm, whether the cursor fill reads as distinct from the edit
//! fill, whether layer 2 floats above layer 1. Those are questions no single-widget frame
//! can answer, because each one is a comparison.
//!
//! # Why it composes the real widgets
//!
//! The tab row is a real [`TabBar`] and the health rows are a real [`StoreHealth`], not a
//! re-drawing of them. A sheet that re-implemented its subjects would drift from them, and
//! would then be a frame depicting a state the system does not produce — the failure this
//! crate exists to make impossible. Only the rungs and fills that *no* widget owns yet
//! (the emphasis ladder, the structural rules, the cursor and edit fills, the layer
//! backgrounds) are drawn from tokens directly, and each is labelled with the token that
//! produced it so a wrong colour is traceable to a wrong function.
//!
//! # Reading it
//!
//! The [`Palette`] under each section is set for the duration of that section only, and
//! restored afterwards, so one frame can show all three modes side by side. That is the
//! whole point: the trade-off between them is a comparison, and an argument about it is
//! worth less than the three sections stacked.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use wqm_common::names::Collection;

use crate::names;
use crate::tokens::{self, Health, Palette};
use crate::widgets::{
    store_health::{StoreHealth, StoreRow},
    tab_bar::{Tab, TabBar},
};

/// Rows one palette's section occupies: heading, gap, two tab rows, five sample rows, the
/// three health rows, and a trailing gap. Counted rather than estimated — at 11 the offline
/// store fell off the bottom, so the sheet silently showed two health states out of three.
const SECTION_HEIGHT: u16 = 13;
/// Row within a section where the health rows begin.
const HEALTH_ROW: u16 = 9;
/// Width of the left-hand label gutter, so every section's samples start on one column.
const GUTTER: usize = 11;

/// Which surface the sheet is drawn on. §6 gives a screen three of them, and a rung that
/// works on one can disappear on another.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Surface {
    /// A full screen: the terminal's own background, never repainted. What every frame in
    /// this crate rendered on until now — which meant §6's depth model was never exercised.
    Layer0,
    /// A modal floating over a full screen.
    Layer1,
    /// A modal over a modal, lighter again so the stack stays legible.
    Layer2,
}

impl Surface {
    /// The fill, or `None` for layer 0 — where "keep the terminal's background" means
    /// painting nothing at all, not painting something that resembles it.
    fn fill(self) -> Option<Color> {
        match self {
            Surface::Layer0 => None,
            Surface::Layer1 => Some(tokens::layer1_bg()),
            Surface::Layer2 => Some(tokens::layer2_bg()),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Surface::Layer0 => "layer 0 — terminal background",
            Surface::Layer1 => "layer 1 — modal",
            Surface::Layer2 => "layer 2 — modal over modal",
        }
    }
}

/// The conformance sheet for one or more palettes, on a chosen surface.
pub struct PaletteSheet {
    palettes: Vec<Palette>,
    surface: Surface,
}

impl PaletteSheet {
    pub fn new(palettes: Vec<Palette>) -> Self {
        Self {
            palettes,
            surface: Surface::Layer0,
        }
    }

    /// Draw on a modal background instead of the terminal's own.
    ///
    /// The fill is resolved per palette section, not once for the sheet: `layer1_bg()` is
    /// itself a rung, so under a different palette it is a different colour, and painting
    /// one palette's layer fill behind another's foregrounds would compare nothing.
    pub fn on(mut self, surface: Surface) -> Self {
        self.surface = surface;
        self
    }

    /// All three modes stacked — the frame the palette decision is actually taken from.
    pub fn all() -> Self {
        Self::new(Palette::ALL.to_vec())
    }

    /// One mode on its own, for looking at a single palette without the comparison.
    pub fn single(palette: Palette) -> Self {
        Self::new(vec![palette])
    }
}

/// A left-hand label, so every sample row says which part of §2/§3/§4 it is showing.
fn gutter(text: &str) -> Span<'static> {
    Span::styled(
        format!("{text:<GUTTER$}"),
        Style::default().fg(tokens::header()),
    )
}

/// A swatch: a run of filled cells followed by the token name that produced it.
fn swatch(fill: Color, name: &str) -> Vec<Span<'static>> {
    vec![
        Span::styled("     ", Style::default().bg(fill)),
        Span::styled(format!(" {name}   "), tokens::faint_style()),
    ]
}

/// §2's four emphasis rungs, in the order the ladder climbs.
fn emphasis_row() -> Line<'static> {
    Line::from(vec![
        gutter("EMPHASIS"),
        Span::styled("faint  ", tokens::faint_style()),
        Span::styled("muted  ", tokens::muted_style()),
        Span::styled("normal  ", tokens::normal_style()),
        Span::styled("strong", tokens::strong_style()),
        Span::styled("   ← grey-depth × weight, no hue", tokens::faint_style()),
    ])
}

/// §2's structural greys plus a CAPS header, which is structure rather than data.
fn structure_row() -> Line<'static> {
    Line::from(vec![
        gutter("STRUCTURE"),
        Span::styled("────────", Style::default().fg(tokens::rule_frame())),
        Span::styled(" frame   ", tokens::faint_style()),
        Span::styled("────────", Style::default().fg(tokens::rule_internal())),
        Span::styled(" internal   ", tokens::faint_style()),
        Span::styled("COLUMN", Style::default().fg(tokens::header())),
        Span::styled(" header", tokens::faint_style()),
    ])
}

/// §3's data cursor: a whole-row tint plus the `▸` mark, deliberately not an inverse block.
fn cursor_row() -> Line<'static> {
    let fill = Style::default().bg(tokens::cursor_bg());
    Line::from(vec![
        gutter("CURSOR"),
        Span::styled("▸ ", fill.fg(tokens::cursor_mark())),
        Span::styled("the row the data cursor is on", fill),
        Span::styled("  ← cursor_bg + cursor_mark", tokens::faint_style()),
    ])
}

/// §3's editing cell and both vim carets. The fill is lighter than the cursor's, which is
/// the only thing separating "you are here" from "you are typing here".
fn edit_row() -> Line<'static> {
    let cursor = Style::default().bg(tokens::cursor_bg());
    let edit = Style::default().bg(tokens::edit_bg());
    Line::from(vec![
        gutter("EDIT"),
        Span::styled("▸ ", cursor.fg(tokens::cursor_mark())),
        Span::styled("timeout   ", cursor),
        Span::styled("30s▏", edit),
        Span::styled(" insert    ", tokens::faint_style()),
        Span::styled("3", edit.add_modifier(Modifier::REVERSED)),
        Span::styled("0s", edit),
        Span::styled(" normal", tokens::faint_style()),
    ])
}

/// §6's two modal backgrounds. Layer 2 must read as lighter than layer 1 or the stack is
/// illegible; on a theme whose background is not near-black, both must still clear the base.
fn layers_row() -> Line<'static> {
    let mut spans = vec![gutter("LAYERS")];
    spans.extend(swatch(tokens::layer1_bg(), "layer1"));
    spans.extend(swatch(tokens::layer2_bg(), "layer2"));
    spans.push(Span::styled(
        "  ← must float above the terminal's own background",
        tokens::faint_style(),
    ));
    Line::from(spans)
}

/// The four tabs that exercise every selector state at once: a plain unselected tab, the
/// cyan selected block, and both alarm hues unselected.
fn alarm_tabs(active: usize) -> TabBar {
    TabBar::new(
        vec![
            Tab::new(1, "Dashboard"),
            Tab::for_collection(2, Collection::Libraries),
            // Labelled from the registry like any collection tab (`CR-038`); the alarm is
            // a separate axis and does not change where the word comes from.
            Tab::alarming(3, names::display(Collection::Rules), Health::Degraded),
            Tab::alarming(4, "Service", Health::Offline),
        ],
        active,
    )
}

/// The store rows, one per health state, so §4's glyph hues sit next to §3's selector.
fn health_rows() -> StoreHealth {
    StoreHealth::new(vec![
        StoreRow::bound("daemon", "memexd", Health::Healthy),
        StoreRow::bound("graph", "ladybug", Health::Degraded),
        StoreRow::bound("vector", "qdrant", Health::Offline),
    ])
}

/// Draws one palette's whole vocabulary into `area`, which must be [`SECTION_HEIGHT`] tall.
fn render_section(palette: Palette, surface: Surface, area: Rect, buf: &mut Buffer) {
    let row = |offset: u16| Rect {
        y: area.y + offset,
        height: 1,
        ..area
    };

    // Painted before anything else so every span below composites onto it. Layer 0 paints
    // nothing at all: "keeps the terminal background" has to mean the real one, or the
    // frame would be testing a stand-in for the surface rather than the surface.
    if let Some(fill) = surface.fill() {
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width {
                buf[(x, y)].set_bg(fill);
            }
        }
    }

    // The rule runs to the frame's edge whatever the label's length, so the three sections
    // line up; measuring the label is what keeps them from ending raggedly.
    let title = format!("── PALETTE: {} · {} ", palette.label(), surface.label());
    let heading = Line::from(vec![
        Span::styled(
            title.clone(),
            Style::default()
                .fg(tokens::rule_frame())
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "─".repeat((area.width as usize).saturating_sub(title.chars().count())),
            Style::default().fg(tokens::rule_frame()),
        ),
    ]);
    Paragraph::new(heading).render(row(0), buf);

    // Tabs get two rows: the selector on a plain tab, then the §4 override where the
    // selected tab owns an alarm and must invert in the alarm's hue, never in cyan.
    Paragraph::new(Line::from(gutter("TABS"))).render(row(2), buf);
    let tabs_area = Rect {
        x: area.x + GUTTER as u16,
        width: area.width.saturating_sub(GUTTER as u16),
        ..row(2)
    };
    alarm_tabs(0).render(tabs_area, buf);

    Paragraph::new(Line::from(gutter("  selected"))).render(row(3), buf);
    let alarm_area = Rect { ..tabs_area };
    alarm_tabs(3).render(
        Rect {
            y: row(3).y,
            ..alarm_area
        },
        buf,
    );

    Paragraph::new(emphasis_row()).render(row(4), buf);
    Paragraph::new(structure_row()).render(row(5), buf);
    Paragraph::new(cursor_row()).render(row(6), buf);
    Paragraph::new(edit_row()).render(row(7), buf);
    Paragraph::new(layers_row()).render(row(8), buf);

    Paragraph::new(Line::from(gutter("HEALTH"))).render(row(HEALTH_ROW), buf);
    health_rows().render(
        Rect {
            x: area.x + GUTTER as u16,
            width: area.width.saturating_sub(GUTTER as u16),
            y: row(HEALTH_ROW).y,
            height: 3.min(area.height.saturating_sub(HEALTH_ROW)),
        },
        buf,
    );
}

impl Widget for PaletteSheet {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let previous = Palette::current();
        for (i, palette) in self.palettes.iter().enumerate() {
            let top = area.y + i as u16 * SECTION_HEIGHT;
            if top >= area.y + area.height {
                break;
            }
            Palette::set(*palette);
            render_section(
                *palette,
                self.surface,
                Rect {
                    y: top,
                    height: SECTION_HEIGHT.min(area.y + area.height - top),
                    ..area
                },
                buf,
            );
        }
        Palette::set(previous);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[PropInfo {
        name: "palettes",
        ty: "Vec<Palette>",
        description: "Modes to stack; each section sets its palette for its rows only",
    }];

    struct Sheet(&'static str, &'static str, fn() -> PaletteSheet);

    impl Ingredient for Sheet {
        fn group(&self) -> &str {
            "Palette Sheet"
        }
        fn name(&self) -> &str {
            self.0
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::palette_sheet"
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
            Box::new(Sheet(
                "All Palettes",
                "Every role, every state, all three modes stacked — the frame the palette decision is taken from",
                PaletteSheet::all,
            )),
            Box::new(Sheet(
                "All Palettes on Layer 1",
                "The same sheet on a modal fill. A rung that works on the terminal's background can vanish here",
                || PaletteSheet::all().on(Surface::Layer1),
            )),
            Box::new(Sheet(
                "All Palettes on Layer 2",
                "And on a modal over a modal — the lighter fill, where the faint rungs are squeezed hardest",
                || PaletteSheet::all().on(Surface::Layer2),
            )),
            Box::new(Sheet(
                "Theme",
                "Theme slots only. Watch cursor-mark against muted, and the cursor fill against the background",
                || PaletteSheet::single(Palette::Theme),
            )),
            Box::new(Sheet(
                "Indexed",
                "The xterm greyscale ramp. Every rung survives; the greys are the only untinted thing on screen",
                || PaletteSheet::single(Palette::Indexed),
            )),
            Box::new(Sheet(
                "Derived",
                "Interpolated from the terminal's own background and foreground — rungs and hue both preserved",
                || PaletteSheet::single(Palette::Derived),
            )),
        ]
    }
}

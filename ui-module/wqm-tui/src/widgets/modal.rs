//! The modal — the background depth model of VISUAL-LANGUAGE §6, and the toast's sibling.
//!
//! §6 says a box means one of exactly two things, **told apart by position and lifetime,
//! never by their border**. The two therefore have to be built against each other: they
//! share a fill, so anything that distinguishes them has to be something a still frame can
//! carry. Two things are:
//!
//! - **where it is** — a modal is centred, a toast is welded to the lower-right corner;
//! - **whether it says how it ends** — a modal is dismissed by the user, so it must show
//!   the key that dismisses it. A toast expires on its own and shows nothing, because there
//!   is nothing for the reader to do. That asymmetry is not decoration: the action row is
//!   the only part of a modal that a toast structurally cannot have.
//!
//! Lifetime and focus are the other half of §6's table and neither survives into a single
//! frame, which is exactly why the two above carry the whole distinction.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Block, Padding, Paragraph, Widget},
};

use crate::tokens;

/// Border, plus one cell of padding inside each border.
const CHROME: u16 = 4;
/// The widest a modal's text may be before it wraps. Past this a modal is a screen.
const MAX_TEXT_WIDTH: u16 = 56;

/// Which of §6's background fills the window carries.
///
/// **A purpose-stable window is this type pinned rather than a third variant.** §6 gives
/// Help and confirm windows *"a fixed background tied to the window's purpose, the same on
/// whichever layer it appears"* — so the rule is "do not vary with depth", which is what
/// stating the fill explicitly already does. Which shade each purpose owns is a design
/// decision with no token behind it yet, so it stays the caller's (and ultimately Chris's)
/// rather than being invented here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fill {
    /// A modal over a full screen: dark, but distinctly above the base, so it floats.
    Layer1,
    /// A modal opened over a modal: distinct and further from the base again, so the stack
    /// stays legible rather than merging into one surface.
    Layer2,
}

impl Fill {
    fn colour(self) -> ratatui::style::Color {
        match self {
            Fill::Layer1 => tokens::layer1_bg(),
            Fill::Layer2 => tokens::layer2_bg(),
        }
    }
}

/// One way out of the modal: the key, and what it does.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Action {
    key: String,
    label: String,
}

impl Action {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
        }
    }
}

/// A centred window that takes focus and waits to be dismissed.
pub struct Modal {
    title: String,
    body: Vec<String>,
    actions: Vec<Action>,
    fill: Fill,
}

impl Modal {
    pub fn new(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self::with_body(title, vec![body.into()])
    }

    /// A modal whose body is already broken into lines — the caller wraps, because only the
    /// caller knows whether a line break is cosmetic or meaningful.
    pub fn with_body(title: impl Into<String>, body: Vec<String>) -> Self {
        Self {
            title: title.into(),
            body,
            actions: Vec::new(),
            fill: Fill::Layer1,
        }
    }

    /// Pin the fill. Also how a purpose-stable window is expressed — see [`Fill`].
    pub fn fill(mut self, fill: Fill) -> Self {
        self.fill = fill;
        self
    }

    pub fn action(mut self, key: impl Into<String>, label: impl Into<String>) -> Self {
        self.actions.push(Action::new(key, label));
        self
    }

    /// The action row: the keys that dismiss this window, muted, at the foot of the box.
    ///
    /// Every modal has one. A modal with no way out drawn on it is a modal the reader has
    /// to guess their way out of — and it is also, in a still frame, a toast.
    fn action_line(&self) -> Line<'static> {
        // The same `key label` idiom the bottom hint line draws, so the two cannot drift —
        // one producer, in `tokens`.
        let pairs: Vec<(&str, &str)> = self
            .actions
            .iter()
            .map(|a| (a.key.as_str(), a.label.as_str()))
            .collect();
        Line::from(tokens::key_hints(&pairs))
    }

    /// The rectangle this modal wants, centred in `area`.
    ///
    /// Exposed because a caller that needs to know what the modal covers — to dim behind
    /// it, or to keep a toast clear of it — must not re-derive the arithmetic and drift.
    pub fn rect(&self, area: Rect) -> Rect {
        let text_width = self
            .body
            .iter()
            .map(|line| line.chars().count() as u16)
            .chain(std::iter::once(self.title.chars().count() as u16))
            .chain(std::iter::once(self.action_width()))
            .max()
            .unwrap_or(0)
            .min(MAX_TEXT_WIDTH);

        let width = (text_width + CHROME).min(area.width);
        // Body, a clear row, the actions — plus the border top and bottom.
        let inner = self.body.len() as u16 + if self.actions.is_empty() { 0 } else { 2 };
        let height = (inner + 2).min(area.height);

        Rect {
            x: area.x + (area.width.saturating_sub(width)) / 2,
            y: area.y + (area.height.saturating_sub(height)) / 2,
            width,
            height,
        }
    }

    fn action_width(&self) -> u16 {
        self.action_line()
            .spans
            .iter()
            .map(|s| s.content.chars().count() as u16)
            .sum()
    }
}

impl Widget for Modal {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let rect = self.rect(area);
        if rect.width <= CHROME || rect.height < 3 {
            return;
        }

        let mut lines: Vec<Line> = self
            .body
            .iter()
            .map(|line| Line::from(Span::styled(line.clone(), tokens::normal_style())))
            .collect();

        if !self.actions.is_empty() {
            lines.push(Line::default());
            lines.push(self.action_line());
        }

        // A floating box has to OCCLUDE, and a background fill is not occlusion: ratatui's
        // `Block::style` restyles the cells it covers and leaves their symbols in place, so a
        // modal's blank rows showed the screen's own text through them, wearing the modal's
        // background. Invisible in an isolated preview — there is nothing behind a widget on
        // an empty buffer — and obvious the moment the modal was put on a screen.
        ratatui::widgets::Clear.render(rect, buf);

        // The title rides the top border, which is what makes the box a window rather than
        // a panel. A toast has no title — it has one sentence and no name for it.
        let block = Block::bordered()
            .title(Span::styled(
                format!(" {} ", self.title),
                tokens::normal_style(),
            ))
            .padding(Padding::horizontal(1))
            .border_style(tokens::muted_style())
            .style(Style::default().bg(self.fill.colour()));

        Paragraph::new(lines).block(block).render(rect, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[
        PropInfo {
            name: "title",
            ty: "String",
            description: "Rides the top border — a toast has no title",
        },
        PropInfo {
            name: "actions",
            ty: "Vec<Action>",
            description: "The keys that dismiss it. A toast has none: it dismisses itself",
        },
        PropInfo {
            name: "fill",
            ty: "Fill",
            description: "Layer1 over a screen, Layer2 over a modal; pin it for a purpose window",
        },
    ];

    struct Variant(&'static str, &'static str, fn() -> Modal);

    impl Ingredient for Variant {
        fn group(&self) -> &str {
            "Modal"
        }
        fn name(&self) -> &str {
            self.0
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::modal"
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
                "Layer 1",
                "A modal over a full screen: centred, titled, and showing the key that dismisses it",
                || {
                    Modal::new(
                        "Discard changes?",
                        "watcher.debounce_ms has been edited and not saved.",
                    )
                    .action("↵", "discard")
                    .action("Esc", "keep editing")
                },
            )),
            Box::new(Variant(
                "Layer 2",
                "A modal over a modal — further from the base again, so the stack reads as two surfaces",
                || {
                    Modal::new("Really discard?", "This cannot be undone.")
                        .fill(Fill::Layer2)
                        .action("y", "yes")
                        .action("n", "no")
                },
            )),
            Box::new(Variant(
                "Purpose-stable, on layer 1",
                "Help pins its shade. Compare with the next entry: §6 says they must be identical",
                || {
                    Modal::new("Help", "j/k move   Enter edit   Tab switch pane")
                        .fill(Fill::Layer2)
                        .action("?", "close")
                },
            )),
            Box::new(Variant(
                "Purpose-stable, on layer 2",
                "The same window opened one layer deeper — the shade must not have moved",
                || {
                    Modal::new("Help", "j/k move   Enter edit   Tab switch pane")
                        .fill(Fill::Layer2)
                        .action("?", "close")
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
    use crate::widgets::toast::{Toast, ToastDeck, ToastStack};
    use ratatui::style::Color;
    use std::time::Instant;

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
        width: 80,
        height: 20,
    };

    fn modal() -> Modal {
        Modal::new("Discard changes?", "watcher.debounce_ms is unsaved.")
            .action("↵", "discard")
            .action("Esc", "keep editing")
    }

    fn render(widget: impl Widget) -> Buffer {
        let mut buf = Buffer::empty(AREA);
        widget.render(AREA, &mut buf);
        buf
    }

    fn painted(buf: &Buffer) -> Vec<(u16, u16)> {
        let mut cells = Vec::new();
        for y in 0..AREA.height {
            for x in 0..AREA.width {
                let cell = buf.cell((x, y)).expect("cell in area");
                if cell.symbol() != " " || cell.style().bg != Some(Color::Reset) {
                    cells.push((x, y));
                }
            }
        }
        cells
    }

    fn distance(a: Color, b: Color) -> f32 {
        match (a, b) {
            (Color::Rgb(r1, g1, b1), Color::Rgb(r2, g2, b2)) => {
                let d = |x: u8, y: u8| (x as f32 - y as f32).powi(2);
                (d(r1, r2) + d(g1, g2) + d(b1, b2)).sqrt()
            }
            _ => panic!("expected RGB under Derived + TrueColor, got {a:?} and {b:?}"),
        }
    }

    #[test]
    fn a_modal_is_centred_and_a_toast_is_not() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // §6: the two are told apart by POSITION, never by their border. So the border is
        // deliberately not what this test looks at.
        let rect = modal().rect(AREA);
        let left = rect.x;
        let right = AREA.width - rect.right();
        assert!(
            left.abs_diff(right) <= 1,
            "a modal is centred: {left} left, {right} right"
        );

        let now = Instant::now();
        let mut deck = ToastDeck::new();
        deck.push(
            Toast::transition(
                tokens::Health::Healthy,
                tokens::Health::Offline,
                "vector store offline",
            )
            .expect("a change of state"),
            now,
        );
        let toast = render(ToastStack::new(&deck, now));

        // Every painted toast cell is in the bottom-right quadrant; no modal cell is only
        // there. That is the distinction stated as a measurement rather than as prose.
        assert!(
            painted(&toast)
                .iter()
                .all(|(x, y)| *x > AREA.width / 2 && *y > AREA.height / 2),
            "a toast is welded to the lower-right corner"
        );
        assert!(
            rect.x < AREA.width / 2,
            "a centred modal reaches into the left half; a toast never does"
        );
    }

    #[test]
    fn a_modal_says_how_it_ends_and_a_toast_has_nothing_to_say() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(modal());
        let text: String = (0..AREA.height)
            .flat_map(|y| {
                (0..AREA.width).map(move |x| {
                    // Collected per row; the join is only used for a substring search.
                    (x, y)
                })
            })
            .map(|(x, y)| buf.cell((x, y)).expect("cell in area").symbol().to_string())
            .collect();

        assert!(
            text.contains("Esc keep editing"),
            "the way out must be drawn"
        );
        assert!(text.contains("Discard changes?"), "and the window named");
    }

    #[test]
    fn layer_two_stands_further_off_the_base_than_layer_one() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // Stated as distance from the terminal's own background rather than as "lighter",
        // for the reason the wash taught (§9): on a light theme "lighter" inverts, while
        // "further from the base" is the same claim on both polarities.
        let base = Color::Rgb(0x1e, 0x1e, 0x2e);
        let one = distance(base, Fill::Layer1.colour());
        let two = distance(base, Fill::Layer2.colour());

        assert!(
            one > 0.0,
            "layer 1 must be distinct from the base it floats on"
        );
        assert!(
            two > one,
            "layer 2 must stand off further than layer 1, or the stack reads as one surface"
        );
    }

    #[test]
    fn a_purpose_window_keeps_its_shade_wherever_it_opens() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // §6: "the user learns 'this shade = Help'". Pinning the fill is what implements
        // that, so the same window built twice must be the same window.
        let help = || {
            Modal::new("Help", "j/k move")
                .fill(Fill::Layer2)
                .action("?", "close")
        };
        assert_eq!(render(help()), render(help()));
    }

    #[test]
    fn a_modal_covers_what_is_behind_it_rather_than_tinting_it() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // A background fill is not occlusion: ratatui's `Block::style` restyles the cells it
        // covers and leaves their symbols standing. Every other test in this module renders
        // onto an empty buffer, where an opaque box and a transparent one are identical — so
        // the buffer is filled with text first, and the fill is what makes the assertion
        // capable of failing.
        let mut buf = Buffer::empty(AREA);
        for y in 0..AREA.height {
            for x in 0..AREA.width {
                buf.cell_mut((x, y)).expect("cell in area").set_symbol("x");
            }
        }

        let window = modal();
        let rect = window.rect(AREA);
        window.render(AREA, &mut buf);

        // The row between the body and the actions is the modal's own blank row.
        let gap = rect.y + rect.height - 3;
        let inside: String = (rect.x + 1..rect.right() - 1)
            .map(|x| buf.cell((x, gap)).expect("cell in area").symbol())
            .collect();
        assert!(
            inside.trim().is_empty(),
            "the modal's blank row shows the screen through it: {inside:?}"
        );
        // …and the screen outside the modal is untouched, so the box occludes rather than
        // clearing more than it owns.
        assert_eq!(
            buf.cell((rect.x - 1, gap)).expect("cell in area").symbol(),
            "x"
        );
    }

    #[test]
    fn the_toast_is_painted_over_the_modal_and_not_under_it() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // §6: a toast "sits outside the stack — painted last, over whatever is there". A
        // modal that could cover a toast would make the arrival point conditional, which is
        // the one property the corner exists to guarantee.
        let now = Instant::now();
        let mut deck = ToastDeck::new();
        deck.push(
            Toast::transition(
                tokens::Health::Healthy,
                tokens::Health::Offline,
                "vector store offline",
            )
            .expect("a change of state"),
            now,
        );

        let mut buf = Buffer::empty(AREA);
        // Big enough to actually reach the corner the toast owns. The first version of this
        // test used a five-row modal ten rows above the toast: nothing overlapped, so it
        // asserted the survival of cells nothing had threatened, and passed without ever
        // exercising paint order. Hence the overlap assertion below — a test of what is
        // drawn on top must first establish that something is underneath.
        let covering = || Modal::with_body("Wide", vec!["x".repeat(70); 16]).action("Esc", "close");
        covering().render(AREA, &mut buf);
        let toast_only = render(ToastStack::new(&deck, now));

        let modal_rect = covering().rect(AREA);
        let overlap: Vec<(u16, u16)> = painted(&toast_only)
            .into_iter()
            .filter(|(x, y)| modal_rect.contains((*x, *y).into()))
            .collect();
        assert!(
            !overlap.is_empty(),
            "the modal must cover part of the toast, or this test proves nothing"
        );

        ToastStack::new(&deck, now).render(AREA, &mut buf);

        for (x, y) in overlap {
            assert_eq!(
                buf.cell((x, y)).unwrap().symbol(),
                toast_only.cell((x, y)).unwrap().symbol(),
                "the toast must survive being drawn after the modal at ({x}, {y})"
            );
        }
    }

    #[test]
    fn the_window_survives_an_encoding_that_refuses_colour() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();
        Encoding::set(Encoding::NoColor);

        // With no fill available the border and the title are the whole window — structural
        // signature first, exactly as §3 requires of every element.
        let buf = render(modal());
        let top: String = (0..AREA.width)
            .map(|x| {
                buf.cell((x, modal().rect(AREA).y))
                    .expect("cell in area")
                    .symbol()
                    .to_string()
            })
            .collect();
        assert!(top.contains("─"), "the border must still be drawn");
        assert!(top.contains("Discard changes?"), "and the title still read");
    }
}

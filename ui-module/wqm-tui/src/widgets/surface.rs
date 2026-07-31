//! The screen itself — layer 0, and the one condition that repaints it.
//!
//! Every full screen renders this first, before its tabs and zones. Normally it does
//! nothing at all: VISUAL-LANGUAGE §6 says layer 0 keeps the terminal's own background and is
//! never repainted, so [`Surface`] draws no cell and costs nothing.
//!
//! **While the daemon is unreachable it washes the whole area red** (Chris, 20260731) — every
//! tab, for as long as the condition lasts. The reason it is a screen-wide fill rather than an
//! alarm in one zone: when the liveness master is not answering, *every* reading on screen is
//! stale, so no zone is telling the truth and marking one of them would understate it.
//!
//! # Why the wash is not the whole signal
//!
//! Colour is r02 §3's reserved half, and [`crate::tokens::layer0_bg`] deliberately declines to
//! wash below truecolor — a fixed red fill is unreadable, and no substitute can know the
//! terminal's polarity. So the condition also carries a **structural** marker: one inverted
//! band on the bottom row, which survives `NO_COLOR` as reverse video exactly as the selector
//! does. On a truecolor terminal the two reinforce each other; on a weaker one the band is the
//! entire signal, and it is enough.
//!
//! The band sits on the bottom row because that is where §7 already puts the system rollup.
//! It is this module's addition rather than Chris's instruction, and it is cheap to drop if
//! he wants the wash alone.

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};

use crate::tokens::{self, Condition, Health};

/// What the band says while the daemon is unreachable.
///
/// The words are N49's own — `UnreachableReason::DaemonUnreachable` spells
/// `daemon_unreachable` on the wire, and [`crate::health`] derives this from it rather than
/// retyping it, so the screen and the wire cannot drift.
pub const UNREACHABLE_BAND: &str = "daemon unreachable";

/// Layer 0: the screen under everything.
pub struct Surface {
    condition: Condition,
}

impl Default for Surface {
    fn default() -> Self {
        Self::new()
    }
}

impl Surface {
    /// The surface for the condition currently in force.
    pub fn new() -> Self {
        Self {
            condition: Condition::current(),
        }
    }

    /// The surface for a stated condition — how a frame renders the wash without the process
    /// being in that state.
    pub fn with_condition(condition: Condition) -> Self {
        Self { condition }
    }
}

impl Surface {
    /// Rows a screen must reserve at its bottom for [`ConditionBand`] — 0 when nominal.
    ///
    /// The band and the wash are two widgets rather than one because they render at opposite
    /// ends of the pass: the wash goes down **first**, under everything, and the band goes
    /// down **last**, over nothing. Drawing both in one call put the band under the zones,
    /// which promptly overwrote it — caught in a `cargo pantry dump`, where the store rows sat
    /// on top of the words *daemon unreachable*.
    pub const fn reserved_rows(condition: Condition) -> u16 {
        match condition {
            Condition::Nominal => 0,
            Condition::DaemonUnreachable => 1,
        }
    }
}

impl Widget for Surface {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.condition == Condition::Nominal || area.is_empty() {
            return;
        }

        // The colour half — absent below truecolor, by design (see the module docs).
        if let Some(wash) = tokens::wash(self.condition) {
            for y in area.top()..area.bottom() {
                for x in area.left()..area.right() {
                    if let Some(cell) = buf.cell_mut((x, y)) {
                        cell.set_bg(wash);
                    }
                }
            }
        }
    }
}

/// The structural half of a condition: one inverted band, drawn last, in the row
/// [`Surface::reserved_rows`] asked the layout to keep.
pub struct ConditionBand {
    condition: Condition,
}

impl ConditionBand {
    pub fn new() -> Self {
        Self {
            condition: Condition::current(),
        }
    }

    pub fn with_condition(condition: Condition) -> Self {
        Self { condition }
    }
}

impl Default for ConditionBand {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for ConditionBand {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.condition == Condition::Nominal || area.is_empty() {
            return;
        }

        let label = format!(
            " {} {}  — every reading on screen is stale ",
            Health::Offline.glyph(),
            UNREACHABLE_BAND
        );
        let text = if (label.chars().count() as u16) <= area.width {
            label
        } else {
            // A narrow terminal keeps the glyph and the state; the explanation is what goes.
            format!(" {} {} ", Health::Offline.glyph(), UNREACHABLE_BAND)
        };
        Paragraph::new(Line::from(Span::styled(
            text,
            tokens::inverted(Health::Offline.color()),
        )))
        .render(area, buf);
    }
}

#[cfg(feature = "tui-pantry")]
pub mod ingredient {
    use super::*;
    use crate::widgets::{
        store_health::{StoreHealth, StoreRow},
        tab_bar::TabBar,
    };
    use ratatui::layout::{Constraint, Layout};
    use tui_pantry::{Ingredient, PropInfo};

    const PROPS: &[PropInfo] = &[PropInfo {
        name: "condition",
        ty: "Condition",
        description: "Nominal draws nothing; DaemonUnreachable washes the screen and bands it",
    }];

    /// A screen with something on it, so the wash is judged against real content rather than
    /// against an empty rectangle — which is the only way to see what it costs in contrast.
    ///
    /// **Forces `Palette::Derived` for the duration**, and restores whatever was in force
    /// after. The wash is RGB-only by construction ([`tokens::wash`]), so under the pantry's
    /// default `Theme` palette this frame would show the band and no wash at all — a preview
    /// of a mode the design is not authored in. `capture()` forces the same globals for the
    /// same reason.
    fn screen(condition: Condition, area: Rect, buf: &mut Buffer) {
        let previous = tokens::Palette::current();
        tokens::Palette::set(tokens::Palette::Derived);

        Surface::with_condition(condition).render(area, buf);

        let reserved = Surface::reserved_rows(condition);
        let [body, band] =
            Layout::vertical([Constraint::Min(0), Constraint::Length(reserved)]).areas(area);
        let [tabs, _gap, stores] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .areas(body);
        TabBar::standard(3).render(tabs, buf);
        // Through a dead daemon every component reading is UNKNOWN, not healthy — a frame
        // showing four green dots under an unreachable banner would depict a state the system
        // cannot produce, which is the one thing this crate promises not to do (§4).
        match condition {
            Condition::Nominal => StoreHealth::nominal().render(stores, buf),
            Condition::DaemonUnreachable => StoreHealth::new(vec![
                StoreRow::unreadable("daemon"),
                StoreRow::unreadable("vector"),
                StoreRow::unreadable("graph"),
                StoreRow::unreadable("relational"),
            ])
            .render(stores, buf),
        }

        // Last, over nothing — the ordering `reserved_rows` exists to make possible.
        ConditionBand::with_condition(condition).render(band, buf);

        tokens::Palette::set(previous);
    }

    struct Nominal;
    impl Ingredient for Nominal {
        fn group(&self) -> &str {
            "Surface"
        }
        fn name(&self) -> &str {
            "Nominal"
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::surface"
        }
        fn description(&self) -> &str {
            "The normal screen: layer 0 keeps the terminal's own background and is never repainted"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            screen(Condition::Nominal, area, buf);
        }
    }

    struct Unreachable;
    impl Ingredient for Unreachable {
        fn group(&self) -> &str {
            "Surface"
        }
        fn name(&self) -> &str {
            "Daemon Unreachable"
        }
        fn source(&self) -> &str {
            "wqm_tui::widgets::surface"
        }
        fn description(&self) -> &str {
            "The A/B against Nominal: the terminal's own background pulled toward red, plus the band"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            screen(Condition::DaemonUnreachable, area, buf);
        }
    }

    /// The candidate wash strengths, weakest first. The shipping value is
    /// [`tokens::WASH_MIX`]; this list exists so it is chosen by comparison rather than by
    /// argument.
    const CANDIDATE_MIXES: [f32; 5] = [0.06, 0.10, 0.14, 0.18, 0.22];

    struct WashStrengths;
    impl Ingredient for WashStrengths {
        fn group(&self) -> &str {
            "Surface"
        }
        fn name(&self) -> &str {
            "Wash Strengths"
        }
        fn source(&self) -> &str {
            "wqm_tui::tokens::wash_at"
        }
        fn description(&self) -> &str {
            "The tolerance, side by side: each strip carries real text, because what the wash costs is contrast"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let previous = tokens::Palette::current();
            tokens::Palette::set(tokens::Palette::Derived);

            for (i, mix) in CANDIDATE_MIXES.iter().enumerate() {
                let y = area.top() + i as u16;
                if y >= area.bottom() {
                    break;
                }
                let strip = Rect {
                    x: area.left(),
                    y,
                    width: area.width,
                    height: 1,
                };
                if let Some(colour) = tokens::wash_at(Condition::DaemonUnreachable, *mix) {
                    for x in strip.left()..strip.right() {
                        if let Some(cell) = buf.cell_mut((x, y)) {
                            cell.set_bg(colour);
                        }
                    }
                }
                // The same three rungs every screen is mostly made of, plus the glyph that has
                // to stay findable: if the wash costs anything, it costs it here.
                let marker = if (*mix - tokens::WASH_MIX).abs() < f32::EPSILON {
                    " ← current"
                } else {
                    ""
                };
                Paragraph::new(Line::from(vec![
                    Span::styled(format!(" {mix:.2}  "), tokens::strong_style()),
                    Span::styled("normal body text  ", tokens::normal_style()),
                    Span::styled("muted label  ", tokens::muted_style()),
                    Span::styled("faint default  ", tokens::faint_style()),
                    Span::styled(
                        Health::Offline.glyph(),
                        ratatui::style::Style::default().fg(Health::Offline.color()),
                    ),
                    Span::styled(marker, tokens::muted_style()),
                ]))
                .render(strip, buf);
            }

            tokens::Palette::set(previous);
        }
    }

    /// The same strips with the terminal's own endpoints **swapped**.
    ///
    /// A light theme is not a second design, it is the same one with the polarity reversed —
    /// so the cheapest honest test of "does this work on Latte?" is to run it against the
    /// user's own colours, inverted, rather than to hardcode somebody's theme here. If the
    /// wash is defined relatively it survives this frame unchanged in strength; if it was
    /// tuned for dark terminals it falls apart here.
    struct WashStrengthsInverted;
    impl Ingredient for WashStrengthsInverted {
        fn group(&self) -> &str {
            "Surface"
        }
        fn name(&self) -> &str {
            "Wash Strengths, polarity swapped"
        }
        fn source(&self) -> &str {
            "wqm_tui::tokens::wash_at"
        }
        fn description(&self) -> &str {
            "The light-theme check, built from YOUR endpoints reversed — no theme is hardcoded here"
        }
        fn props(&self) -> &[PropInfo] {
            PROPS
        }
        fn render(&self, area: Rect, buf: &mut Buffer) {
            let previous = tokens::endpoints();
            tokens::set_endpoints(crate::terminal::Endpoints {
                background: previous.foreground,
                foreground: previous.background,
            });
            WashStrengths.render(area, buf);
            tokens::set_endpoints(previous);
        }
    }

    pub fn ingredients() -> Vec<Box<dyn Ingredient>> {
        vec![
            Box::new(Nominal),
            Box::new(Unreachable),
            Box::new(WashStrengths),
            Box::new(WashStrengthsInverted),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::Encoding;
    use crate::terminal::{Endpoints, Rgb};
    use crate::tokens::Palette;

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

    /// A screen composed the way the contract says: the wash first, the band last, into the
    /// rows [`Surface::reserved_rows`] asked for. A test that rendered only one of the two
    /// would pass while the real composition failed — which is what happened before they were
    /// split.
    fn render(condition: Condition, width: u16, height: u16) -> Buffer {
        let area = Rect::new(0, 0, width, height);
        let mut buf = Buffer::empty(area);
        Surface::with_condition(condition).render(area, &mut buf);

        let reserved = Surface::reserved_rows(condition);
        let band = Rect {
            x: area.left(),
            y: area.bottom() - reserved,
            width: area.width,
            height: reserved,
        };
        ConditionBand::with_condition(condition).render(band, &mut buf);
        buf
    }

    #[test]
    fn a_nominal_screen_paints_nothing_at_all() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // §6: layer 0 keeps the terminal's own background. "Draws nothing" is the design, not
        // an optimisation — repainting it with our own black would override the user's theme.
        assert_eq!(
            render(Condition::Nominal, 40, 10),
            Buffer::empty(Rect::new(0, 0, 40, 10))
        );
    }

    #[test]
    fn the_wash_is_the_users_own_background_pulled_toward_red() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(Condition::DaemonUnreachable, 40, 10);
        let Some(ratatui::style::Color::Rgb(r, g, b)) = buf.cell((0, 0)).map(|c| c.bg) else {
            panic!("the wash is not RGB");
        };

        // Redder than the theme's background...
        assert!(r > 0x1e, "the wash did not move toward red");
        // ...and still recognisably that background: a dark theme stays dark, which is what
        // makes "dark or light red" answerable by the terminal rather than by us.
        assert!(r < 0x80, "the wash overrode the theme's polarity");
        assert!(
            g <= 0x1e && b <= 0x2e,
            "the wash brightened the other channels"
        );

        // Every cell above the band, not just the first — the condition is the whole
        // screen's, which is the difference between this and an alarm in one zone.
        for y in 0..9 {
            for x in 0..40 {
                assert_eq!(
                    buf.cell((x, y)).map(|c| c.bg),
                    Some(ratatui::style::Color::Rgb(r, g, b)),
                    "cell ({x},{y}) missed the wash"
                );
            }
        }
        // The band is the one row that is louder than the wash, on purpose: the wash says
        // "everything here is stale", the band says what is wrong.
        assert_eq!(
            buf.cell((0, 9)).map(|c| c.bg),
            Some(Health::Offline.color()),
            "the band lost its fill to the wash"
        );
    }

    #[test]
    fn a_light_theme_gets_a_light_wash() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();
        tokens::set_endpoints(Endpoints {
            background: Rgb::new(0xff, 0xff, 0xff),
            foreground: Rgb::new(0x20, 0x20, 0x20),
        });

        let Some(ratatui::style::Color::Rgb(_, g, b)) = render(Condition::DaemonUnreachable, 10, 3)
            .cell((0, 0))
            .map(|c| c.bg)
        else {
            panic!("the wash is not RGB");
        };
        // Pulled toward red from white is a pink/light red — the other channels stay high.
        assert!(g > 0xb0 && b > 0xb0, "a light theme was given a dark wash");
    }

    #[test]
    fn the_band_survives_an_encoding_that_refuses_colour() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();
        Encoding::set(Encoding::NoColor);

        let buf = render(Condition::DaemonUnreachable, 40, 6);
        let bottom: String = (0..40)
            .map(|x| buf.cell((x, 5)).unwrap().symbol().to_string())
            .collect();
        assert!(
            bottom.contains(UNREACHABLE_BAND),
            "the structural marker vanished with the colour: {bottom:?}"
        );
        assert!(
            buf.cell((1, 5))
                .unwrap()
                .modifier
                .contains(ratatui::style::Modifier::REVERSED),
            "the band must invert where it cannot colour"
        );
        // And with no colour there is no wash to fall back on, which is exactly why the band
        // is not optional.
        assert_eq!(
            buf.cell((0, 0)).map(|c| c.bg),
            Some(ratatui::style::Color::Reset)
        );
    }

    /// CIE76 perceptual distance, implemented here rather than pulled in as a dependency: it
    /// is fifteen lines and it exists to check one property of one constant.
    fn delta_e(a: (u8, u8, u8), b: (u8, u8, u8)) -> f32 {
        fn lab(c: (u8, u8, u8)) -> [f32; 3] {
            let lin = |v: u8| {
                let v = v as f32 / 255.0;
                if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            };
            let (r, g, b) = (lin(c.0), lin(c.1), lin(c.2));
            let (x, y, z) = (
                (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047,
                0.2126 * r + 0.7152 * g + 0.0722 * b,
                (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883,
            );
            let f = |t: f32| {
                if t > 0.008856 {
                    t.cbrt()
                } else {
                    7.787 * t + 16.0 / 116.0
                }
            };
            let (fx, fy, fz) = (f(x), f(y), f(z));
            [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
        }
        let (a, b) = (lab(a), lab(b));
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    #[test]
    fn the_wash_is_the_same_strength_on_a_light_theme_as_on_a_dark_one() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        // Two dark themes and two light ones. The point is that ONE constant serves both
        // polarities: the wash is defined against the terminal's own background, so a light
        // theme gets a pale tint that *darkens* while a dark theme gets one that *lightens* —
        // the inversion is a consequence of the definition, not a second code path.
        let themes: [(&str, (u8, u8, u8)); 4] = [
            ("mocha", (0x1e, 0x1e, 0x2e)),
            ("gruvbox dark", (0x28, 0x28, 0x28)),
            ("latte", (0xef, 0xf1, 0xf5)),
            ("solarized light", (0xfd, 0xf6, 0xe3)),
        ];

        for (name, bg) in themes {
            tokens::set_endpoints(Endpoints {
                background: Rgb::new(bg.0, bg.1, bg.2),
                // The foreground is irrelevant to the wash; give each theme a plausible one so
                // nothing else in the frame is nonsense.
                foreground: if bg.0 > 0x80 {
                    Rgb::new(0x4c, 0x4f, 0x69)
                } else {
                    Rgb::new(0xcd, 0xd6, 0xf4)
                },
            });
            let Some(ratatui::style::Color::Rgb(r, g, b)) =
                tokens::wash(Condition::DaemonUnreachable)
            else {
                panic!("{name}: no wash");
            };

            let distance = delta_e(bg, (r, g, b));
            // Measured across all four: 11.4–13.5 at mix 0.10, 16.0–18.9 at 0.14. The band is
            // deliberately wide and spans both — it pins "one constant works on both
            // polarities", never the constant itself, which is Chris's and moves with `OSC 4`.
            assert!(
                (8.0..=22.0).contains(&distance),
                "{name}: wash is ΔE {distance:.1} from the base — the tint no longer carries \
                 the same weight on this polarity"
            );
        }
    }

    #[test]
    fn a_narrow_screen_keeps_the_state_and_drops_the_explanation() {
        let _serial = crate::global_state_lock();
        let _restore = Restore::dark_truecolor();

        let buf = render(Condition::DaemonUnreachable, 24, 3);
        let bottom: String = (0..24)
            .map(|x| buf.cell((x, 2)).unwrap().symbol().to_string())
            .collect();
        assert!(bottom.contains(UNREACHABLE_BAND), "{bottom:?}");
        assert!(
            !bottom.contains("stale"),
            "the long form did not fit and was not dropped"
        );
    }
}

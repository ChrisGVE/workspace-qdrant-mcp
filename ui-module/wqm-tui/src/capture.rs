//! Headless pixel capture — a ratatui frame to a PNG, with no terminal at all.
//!
//! # This is the agent's instrument, not the normal output
//!
//! Chris, 20260730: **the PNG path is mostly for the agent.** `cargo pantry dump` and
//! `palette.sh` produce ANSI in a real terminal, and that is the normal output and the
//! surface a human reads. A capture exists so a session with no eyes can still judge a
//! frame — it does not replace them, and it is the less honest of the two about hue (see
//! below). When both are available, produce the ANSI.
//!
//! # What this closes
//!
//! §1 of `handover.md` exists because the `.mock` → `freeze` pipeline could not render
//! bold, so weight-based verdicts about the design language could not be trusted. Moving to
//! real ratatui removed the approximation from the *grid*; this module removes it from the
//! *pixels*. A frame captured here went through the same widgets, the same [`crate::tokens`]
//! and the same ratatui rasterisation as a frame on screen.
//!
//! It also sidesteps `config.claude#45`, where the `tui-test-harness` pixel backend is
//! blocked on an un-granted iTerm2 Automation permission. Nothing here asks the OS for
//! anything: no terminal, no window server, no GPU, no permission prompt. **It does not fix
//! that harness** — the harness screenshots *any* application's real window, and this only
//! renders frames our own process drew. For this crate's storyboard, that is the whole job.
//!
//! # Derived only, and that is enforced, not advised
//!
//! [`soft_ratatui`] 0.2.0 resolves `Color::Rgb` byte-exactly and resolves everything else
//! wrongly. Measured against 0.2.0 on 2026-07-30:
//!
//! | input | rendered | should be |
//! |---|---|---|
//! | `Color::Indexed(i)` | `Rgb(i*i, 2*i, i)` — arithmetic on the index, no palette lookup | the xterm 256 table |
//! | `Color::Cyan` | `#0000ff` (blue) | the terminal's cyan |
//! | `Color::LightMagenta` | `#8b008b`, *darker* than `Magenta`'s `#ff00ff` | the brighter of the pair |
//! | `Color::Reset` as a background | `#050179` (navy) | the terminal's background |
//!
//! Those are not roundings, they are the wrong colours. `Color::Indexed(232)` came out
//! `#40d0e8`, a cyan, where the greyscale ramp starts at `#080808`.
//!
//! [`Palette::Derived`] emits `Color::Rgb` for every neutral, so it is the one mode this
//! backend renders truthfully — and it is the settled default (`handover.md` §7.1) and the
//! mode the storyboard is authored in (§7.5). [`Palette::Indexed`] is *entirely* indexed
//! colour and would render as a cyan ramp; [`Palette::Theme`] would paint the reserved
//! selector hue blue, and cyan-means-selected is the one hue rule VISUAL-LANGUAGE §3 has.
//!
//! So [`capture`] **forces an RGB source — `Derived` unless one is already in force, since
//! §15's `Bundled` is RGB too — and `Encoding::TrueColor` with it, and restores both
//! afterwards.** A PNG that silently depicted the wrong palette is precisely the class of
//! artifact §1 exists to stop producing; the earlier one at least failed visibly, by not
//! rendering bold. The encoding has to be forced alongside the palette because a rung is
//! emitted in the lesser of the two ([`crate::tokens::family`]), so a process that had probed
//! a 16-colour terminal would degrade this capture's ladder back onto the slots the backend
//! mis-resolves.
//!
//! The hues still are not the user's: health green comes out `#006400` and yellow `#ffd700`,
//! because those are named slots too. Read a capture for *layout, weight, glyph and
//! neutral* fidelity; read a real terminal for hue.

use png::{BitDepth, ColorType, Encoder};
use ratatui::style::{Color, Style};
use ratatui::widgets::Block;
use ratatui::{Frame, Terminal};
use soft_ratatui::embedded_graphics_unicodefonts::{
    mono_8x13_atlas, mono_8x13_bold_atlas, mono_8x13_italic_atlas,
};
use soft_ratatui::{EmbeddedGraphics, SoftBackend};

use crate::encoding::{Encoding, Family};
use crate::tokens::{self, Palette};

/// Discards an error that cannot exist.
///
/// [`SoftBackend`]'s error type is [`Infallible`] — it writes into memory it owns, so there
/// is no failure to report. Matching on the uninhabited variant is total, which is why this
/// is not an `unwrap`: rasterisation genuinely has no failure path, and pretending it does
/// would put a variant in [`capture`]'s error type that no input can ever produce.
///
/// [`Infallible`]: std::convert::Infallible
fn infallible<T>(result: Result<T, std::convert::Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

/// The pixel size of one character cell under the bundled font.
///
/// Fixed because the font is a bitmap atlas rather than a scalable face: the same cell
/// grid produces the same pixels on every machine, which is what makes a capture
/// comparable to yesterday's capture. A TTF face (`soft_ratatui`'s `embedded-ttf`
/// feature) would render closer to the terminal's own JetBrains Mono and would cost
/// exactly that reproducibility, so it is deliberately not the default here.
pub const CELL: (usize, usize) = (8, 13);

/// Serialises the palette-forcing window in [`capture`].
///
/// [`Palette`] is process-global, so two concurrent captures would interleave their
/// force-and-restore and one would restore the other's temporary value. Found by
/// `capture_restores_the_palette_it_found` running beside the other capture tests, which is
/// exactly the shape a pantry that captured from a worker thread would hit.
static PALETTE_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Holds RGB output in force, and puts back what it found — on the way out of the scope, so
/// a panicking `draw` closure cannot leave the process in the capture palette.
///
/// **Both** globals have to be forced, not just the palette. A rung is emitted in the lesser
/// of the source and the encoding ([`crate::tokens::family`]), so an application that had
/// probed a 16-colour terminal would still degrade this capture's `Derived` ladder to theme
/// slots — which is the one thing the backend renders wrongly. Forcing the palette alone
/// would look correct in every test that never touched the encoding.
struct RgbOnly {
    palette: Palette,
    encoding: Encoding,
    _lock: std::sync::MutexGuard<'static, ()>,
}

impl RgbOnly {
    fn force() -> Self {
        // A panic inside a capture poisons the lock but leaves nothing inconsistent behind,
        // since the Drop below still ran. Recovering the guard keeps one bad frame from
        // disabling capture for the rest of the process.
        let lock = PALETTE_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let forced = Self {
            palette: Palette::current(),
            encoding: Encoding::current(),
            _lock: lock,
        };
        // An RGB source is the requirement; WHICH one is the caller's. Before §15 there was
        // only one (`Derived`), so "force RGB" and "force Derived" were the same instruction —
        // they stopped being the same the moment `Bundled` shipped, and forcing `Derived` from
        // then on would have captured the terminal's colours for a screen painted in the
        // theme's. So the RGB sources are left alone and only the others are overridden.
        if Palette::current().family() != Family::Rgb {
            Palette::set(Palette::Derived);
        }
        Encoding::set(Encoding::TrueColor);
        forced
    }
}

impl Drop for RgbOnly {
    fn drop(&mut self) {
        Palette::set(self.palette);
        Encoding::set(self.encoding);
    }
}

/// Renders `draw` into a `cols` × `rows` grid and returns the frame as PNG bytes.
///
/// An RGB palette is forced for the duration and restored after — see
/// the module docs for why that is a hard gate rather than a caller's choice. Concurrent
/// captures are serialised for the same reason.
///
/// The only failure mode is PNG encoding; rasterisation cannot fail (see [`infallible`]).
pub fn capture<F>(cols: u16, rows: u16, draw: F) -> Result<Vec<u8>, png::EncodingError>
where
    F: FnOnce(&mut Frame),
{
    let (width, height, rgba) = {
        let _rgb_only = RgbOnly::force();
        render_rgba(cols, rows, draw)
    };

    encode_png(width, height, &rgba)
}

/// The rasterisation half, split out so the palette guard above has one obvious scope.
fn render_rgba<F>(cols: u16, rows: u16, draw: F) -> (usize, usize, Vec<u8>)
where
    F: FnOnce(&mut Frame),
{
    let backend = SoftBackend::<EmbeddedGraphics>::new(
        cols,
        rows,
        mono_8x13_atlas(),
        // Both weights are passed because the whole reason this path exists is that the
        // previous renderer could not draw bold. Dropping them here would reintroduce the
        // defect in a new place.
        Some(mono_8x13_bold_atlas()),
        Some(mono_8x13_italic_atlas()),
    );
    let mut terminal = infallible(Terminal::new(backend));
    infallible(terminal.clear());
    infallible(terminal.draw(|frame| {
        // Layer 0 is "keep the terminal background and never repaint it" (VISUAL-LANGUAGE
        // §6), which is why `tokens` has no token for it. A capture has no terminal to keep,
        // and the backend fills unpainted cells with `#050179`, a navy that would sit under
        // every frame and make the neutral ladder unjudgeable. Painting the background we
        // were *told* about is the faithful reading of layer 0, not an addition to it.
        let background = tokens::endpoints().background;
        frame.render_widget(
            Block::default().style(Style::default().bg(Color::Rgb(
                background.r,
                background.g,
                background.b,
            ))),
            frame.area(),
        );
        draw(frame);
    }));

    let backend = terminal.backend();
    (
        backend.get_pixmap_width(),
        backend.get_pixmap_height(),
        backend.get_pixmap_data_as_rgba(),
    )
}

fn encode_png(width: usize, height: usize, rgba: &[u8]) -> Result<Vec<u8>, png::EncodingError> {
    let mut out = Vec::new();
    {
        let mut encoder = Encoder::new(&mut out, width as u32, height as u32);
        encoder.set_color(ColorType::Rgba);
        encoder.set_depth(BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(rgba)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::Paragraph;

    /// PNG's 8-byte signature. Checked rather than assumed, so a capture that produced
    /// something other than a PNG fails here instead of downstream in an image viewer.
    const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

    /// Serialises the tests in this module against each other, and against every other test
    /// that touches the process-global palette or encoding.
    ///
    /// [`PALETTE_GUARD`] makes each individual capture atomic, which is what a caller needs,
    /// but it cannot cover a *test* that sets the palette, captures, and then asserts —
    /// another test's capture may take the guard in the gap before the assertion. That is a
    /// property of the test, not of [`capture`], so it is fixed here rather than by widening
    /// the lock in the module. Always taken before [`PALETTE_GUARD`], so the order is
    /// consistent and cannot deadlock.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        crate::global_state_lock()
    }

    #[test]
    fn a_capture_is_a_png_of_the_expected_size() {
        let _serial = serial();
        let png = capture(10, 2, |frame| {
            frame.render_widget(Paragraph::new("hello"), frame.area());
        })
        .expect("capture");

        assert_eq!(&png[..8], &PNG_MAGIC, "not a PNG");

        // Width and height live in the IHDR chunk, immediately after the signature and the
        // 8-byte chunk header. Reading them proves the grid was honoured rather than
        // trusting the backend's word for it.
        let width = u32::from_be_bytes(png[16..20].try_into().unwrap());
        let height = u32::from_be_bytes(png[20..24].try_into().unwrap());
        assert_eq!(width as usize, 10 * CELL.0);
        assert_eq!(height as usize, 2 * CELL.1);
    }

    #[test]
    fn the_same_frame_captures_identically_twice() {
        let _serial = serial();
        // The property that makes a capture usable as a golden image. A bitmap atlas and a
        // software rasteriser should have nothing left to vary; this is what notices if a
        // future font or backend change quietly introduces something that does.
        let frame = || {
            capture(20, 1, |f| {
                f.render_widget(
                    Paragraph::new(Line::from(vec![
                        Span::styled("bold", Style::default().add_modifier(Modifier::BOLD)),
                        Span::styled(" rgb", Style::default().fg(Color::Rgb(0xcd, 0xd6, 0xf4))),
                    ])),
                    f.area(),
                );
            })
            .expect("capture")
        };
        assert_eq!(frame(), frame());
    }

    #[test]
    fn capture_restores_the_palette_and_encoding_it_found() {
        let _serial = serial();
        // The forcing in `capture` is a guard, not a mode switch: a pantry session that
        // captured a frame must not find its palette or its probed encoding changed
        // underneath it.
        for palette in Palette::ALL {
            for encoding in Encoding::ALL {
                Palette::set(palette);
                Encoding::set(encoding);
                let _ = capture(4, 1, |f| {
                    f.render_widget(Paragraph::new("x"), f.area());
                })
                .expect("capture");
                assert_eq!(Palette::current(), palette, "{palette:?} was not restored");
                assert_eq!(
                    Encoding::current(),
                    encoding,
                    "{encoding:?} was not restored"
                );
            }
        }
        Palette::set(Palette::Theme);
        Encoding::set(Encoding::TrueColor);
    }

    #[test]
    fn a_probed_sixteen_colour_terminal_does_not_reach_the_capture() {
        let _serial = serial();
        // The defect this catches is invisible without it: forcing only the palette leaves
        // `family()` at the encoding's ceiling, so an application that had probed a login
        // shell would capture the slot ladder — which `soft_ratatui` resolves by arithmetic
        // on the index rather than by lookup. The two captures must agree.
        let frame = || {
            capture(24, 1, |f| {
                f.render_widget(
                    Paragraph::new(Span::styled("neutral", tokens::muted_style())),
                    f.area(),
                )
            })
            .expect("capture")
        };

        Encoding::set(Encoding::TrueColor);
        let truecolor = frame();
        Encoding::set(Encoding::Ansi16);
        let sixteen = frame();
        Encoding::set(Encoding::TrueColor);

        assert_eq!(
            truecolor, sixteen,
            "the ambient encoding leaked into a capture"
        );
    }

    #[test]
    fn bold_and_regular_are_distinguishable_in_the_pixels() {
        let _serial = serial();
        // The original defect, pinned. `freeze` rendered bold and regular identically, which
        // is why weight-based verdicts could not be trusted (handover §1). Comparing two
        // captures of the same text is the measurement that would have caught it.
        let render = |modifier: Modifier| {
            capture(8, 1, move |f| {
                f.render_widget(
                    Paragraph::new(Span::styled(
                        "MMMM",
                        Style::default().add_modifier(modifier),
                    )),
                    f.area(),
                );
            })
            .expect("capture")
        };
        assert_ne!(
            render(Modifier::empty()),
            render(Modifier::BOLD),
            "bold rendered identically to regular — the freeze defect is back"
        );
    }
}

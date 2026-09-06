//! The pantry entry point.
//!
//! The terminal is asked for its background and foreground *here*, before `run!()` takes
//! the screen: `Palette::Derived` interpolates between those two colours, and the query has
//! to happen while the terminal is still in its normal mode and nothing else is competing
//! for its input. Rendering never queries, so a frame costs no round trip.
//!
//! A terminal that does not answer — or a piped stdout, which is every `cargo pantry dump`
//! — leaves the fallback endpoints in place, so derived frames still render; they render
//! the black-to-white ladder r02 was authored against rather than this terminal's.
//! `WQM_TUI_TERM_BG` / `WQM_TUI_TERM_FG` override both, which is how a derived frame is
//! captured reproducibly with no terminal to ask.
//!
//! # The encoding is probed here too, with one deliberate exception
//!
//! `encoding::detect` reports what stdout can carry, and a rung is then emitted in the lesser
//! of that and the palette. The exception is `NoTty`: `cargo pantry dump` writes ANSI *into a
//! pipe on purpose* — the escape sequences are the artifact, not an accident of rendering to
//! a terminal — so a probe that correctly reports "not a terminal" would strip exactly what
//! the caller asked for. This is the instrument, so the pipe keeps its colour and every other
//! row of the probe is honoured.
//!
//! To see a degradation rather than reason about one, force it:
//! `CLICOLOR_FORCE=ansi16 cargo pantry dump …`, or open the `Palette Reference` entries,
//! which render the encodings side by side.

use wqm_tui::encoding::{self, Encoding};

fn main() -> std::io::Result<()> {
    if let Some(endpoints) = wqm_tui::terminal::detect() {
        wqm_tui::tokens::set_endpoints(endpoints);
    }

    // §15's bundled theme. Static here on purpose: which theme a user gets is an N7
    // preference with nowhere to be written yet (`UIQ-009`/`UIQ-010`), and the pantry is not
    // where that decision belongs.
    //
    // **Catppuccin Mocha, on Chris's instruction 20260906.** It was Everforest from 20260801
    // — *"can you change the theme to Everforest so I can see with my own eyes"* — chosen as
    // the adversarial case for tinting `strong` toward `accent`, since Everforest's accent
    // (`#83C178`) and its `success` (`#A7C080`) are both green and sit ΔE 6.6 apart at
    // k = 0.55, the worst of the fifteen.
    //
    // That case has not gone away; it has moved somewhere better. `styles::strong` measures
    // Everforest directly, as data, and a *test* that names the adversarial theme is a
    // stronger instrument than a *harness* that happens to be set to it: it keeps measuring
    // whatever the pantry is painting with.
    //
    // What the harness was costing, meanwhile, was agreement. Mocha is the theme the crate's
    // tests pin and the theme every measurement in `tokens` was taken against, so the screen
    // Chris judged and the numbers this crate argues from were two different rooms. They are
    // one room now.
    //
    // ⚠ `pantry.toml`'s `[pantry.preview_backgrounds]` are a hand transcription of the three
    // layer colours THIS theme produces. Change this line and those three are silently wrong
    // — nothing fails, the previews simply sit on another theme's backdrop.
    wqm_tui::tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());

    Encoding::set(match encoding::detect() {
        // See the module docs: for a dump, the escape sequences are the whole output.
        Encoding::NoTty => Encoding::TrueColor,
        probed => probed,
    });

    tui_pantry::run!()
}

//! Round 2's frames as PNGs, and the one list both instruments read.
//!
//! ```bash
//! # the frame list, for the shell that dumps ANSI and cell grids
//! cargo run -p wqm-tui --features png-capture,tui-pantry --example mf_frames -- --list
//! # the PNGs
//! cargo run -p wqm-tui --features png-capture,tui-pantry --example mf_frames -- out/
//! ```
//!
//! # One list, two instruments, and they answer different questions
//!
//! Every frame below names a **pantry variant** rather than re-drawing anything, so the PNG and
//! the `.ans` come out of the same `Ingredient::render` and cannot disagree about what was
//! drawn. What they can disagree about is what they are able to SHOW, and that split is the
//! whole reason both exist:
//!
//! - **Colour is the PNG's.** A grid dump carries a colour as a number nobody can judge.
//! - **Glyphs and attributes are the dump's.** `png-capture` drops the `●` radio glyph (wqm#283),
//!   the powerline separator lives in the private-use area and renders as a blank in most text
//!   views, and no still image can show `SLOW_BLINK` at all. Every one of those is a fact about a
//!   cell, and a cell dump has it exactly.
//!
//! So a claim about hue is checked against a PNG and a claim about shape or attribute is checked
//! against the grid — and `views::modal_framework::shipping::tests` pins the ones that matter so
//! neither has to be read by eye to know the frame is right.
//!
//! # The theme has to be IN FORCE, not merely chosen
//!
//! `tokens::active_theme` answers `None` under every source but `Palette::Bundled`, so setting a
//! theme without setting the source renders the slot fallbacks: `modal_border` comes back
//! `Color::Magenta`, `Rgb::from_color` cannot read a slot, and `modal_fill` hands back the bare
//! layer — every tint frame identical and a magenta border on the rest. Round 1's equivalent
//! example set only the theme; both lines are here.

use std::path::PathBuf;

use wqm_tui::capture::capture;
use wqm_tui::terminal;
use wqm_tui::tokens;

/// `(file stem, pantry variant, cols, rows)`.
///
/// The storyboard screen is 125x34 and the floor Chris named is 100x30. Three entries render at
/// sizes of their own because the thing they show only exists there: the size model has to be
/// seen holding at a THIRD size, the minimum window only exists at the minimum, and the
/// too-small message only exists below it.
const FRAMES: &[(&str, &str, u16, u16)] = &[
    (
        "mf-00-max-125x34",
        "00 size — max window (dump at 125x34, 100x30, 200x40)",
        125,
        34,
    ),
    (
        "mf-01-max-100x30",
        "00 size — max window (dump at 125x34, 100x30, 200x40)",
        100,
        30,
    ),
    (
        "mf-02-max-200x40",
        "00 size — max window (dump at 125x34, 100x30, 200x40)",
        200,
        40,
    ),
    (
        "mf-03-min-44x20",
        "01 size — the minimum window (dump at 44x20)",
        44,
        20,
    ),
    (
        "mf-04-too-small-40x18",
        "02 size — screen too small (dump at 40x18)",
        40,
        18,
    ),
    (
        "mf-05-content-sized",
        "03 size — content-sized, not a drill-down (A)",
        125,
        34,
    ),
    (
        "mf-06-drilldown-both-bars",
        "04 size — drill-down at max, both bars (B)",
        125,
        34,
    ),
    (
        "mf-08-tint-held",
        "06 tint — blue 0.40, lightness held",
        125,
        34,
    ),
    (
        "mf-09-tint-held-solarized-dark",
        "07 tint — held, on Solarized Dark",
        125,
        34,
    ),
    (
        "mf-10-tint-held-latte",
        "08 tint — held, on Catppuccin Latte (light)",
        125,
        34,
    ),
    (
        "mf-11-view-black-cursor",
        "09 fields — view mode, black on the cursor row",
        125,
        34,
    ),
    (
        "mf-13-edit-fill-only",
        "11 fields — edit mode, fill alone",
        125,
        34,
    ),
    (
        "mf-12-tick-box",
        "10 fields — tick box, the two states side by side",
        125,
        34,
    ),
    (
        "mf-14-radio-row",
        "12 fields — radio on one row (A)",
        125,
        34,
    ),
    (
        "mf-15-radio-column",
        "13 fields — the same radio as a column (B)",
        125,
        34,
    ),
    (
        "mf-16-dropdown-open",
        "14 fields — drop-down open, no frame",
        125,
        34,
    ),
    (
        "mf-17-dropdown-filtered",
        "15 fields — drop-down, fuzzy filter typed",
        125,
        34,
    ),
    (
        "mf-18-caret-insert",
        "16 fields — single line, vim INSERT (bar, blinking)",
        125,
        34,
    ),
    (
        "mf-19-caret-normal",
        "17 fields — single line, vim NORMAL (block, steady)",
        125,
        34,
    ),
    (
        "mf-20-caret-conventional",
        "18 fields — single line, conventional (terminal caret)",
        125,
        34,
    ),
    (
        "mf-21-multiline-edit",
        "19 fields — multi-line being edited",
        125,
        34,
    ),
    (
        "mf-22-selection-A-neutral",
        "20 fields — selected text (A) neutral inversion",
        125,
        34,
    ),
    (
        "mf-23-selection-B-secondary",
        "21 fields — selected text (B) theme secondary",
        125,
        34,
    ),
    (
        "mf-24-typed-live",
        "22 fields — the same edit, driven by real keystrokes",
        125,
        34,
    ),
    (
        "mf-25-third-column-no-band",
        "23 readability — third column, no band",
        125,
        34,
    ),
    (
        "mf-26-no-headers",
        "24 readability — no headers at all",
        125,
        34,
    ),
    (
        "mf-27-crumb-depth1",
        "25 breadcrumb — depth 1",
        125,
        34,
    ),
    (
        "mf-28-crumb-depth2",
        "26 breadcrumb — depth 2",
        125,
        34,
    ),
    (
        "mf-29-crumb-depth3",
        "27 breadcrumb — depth 3",
        125,
        34,
    ),
    (
        "mf-30-crumb-depth3-plain",
        "28 breadcrumb — depth 3, plain (nerd-font fallback)",
        125,
        34,
    ),
    (
        "mf-31-no-edit-banner",
        "29 banner — view vs edit, no `-- EDIT --`",
        125,
        34,
    ),
    (
        "mf-32-confirm-over-window",
        "30 confirm — a window quietened under its own guard",
        125,
        34,
    ),
    (
        "mf-34-edge-spacing",
        "32 edge — spacing only",
        125,
        34,
    ),
    (
        "mf-35-no-color-set-mark",
        "33 degradation — the SET mark under NO_COLOR",
        125,
        34,
    ),
    (
        "mf-36-ansi16",
        "34 degradation — the same frame at ansi16",
        125,
        34,
    ),
    (
        "mf-37-floor-edit-100x30",
        "11 fields — edit mode, fill alone",
        100,
        30,
    ),
    (
        "mf-38-floor-drilldown-100x30",
        "04 size — drill-down at max, both bars (B)",
        100,
        30,
    ),
];

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());

    if arg == "--list" {
        for (stem, variant, cols, rows) in FRAMES {
            println!("{stem}\t{variant}\t{cols}x{rows}");
        }
        return;
    }

    if let Some(endpoints) = terminal::detect() {
        tokens::set_endpoints(endpoints);
    }
    // Both lines. See the module docs for what setting only the second produces.
    tokens::Palette::set(tokens::Palette::Bundled);
    tokens::set_theme(ratatui_themes::ThemeName::CatppuccinMocha.palette());

    let out = PathBuf::from(arg);
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("cannot write to {}: {e}", out.display());
        std::process::exit(1);
    }

    let ingredients = wqm_tui::views::modal_framework::shipping::ingredient::ingredients();
    for (stem, variant, cols, rows) in FRAMES {
        let Some(ingredient) = ingredients.iter().find(|i| i.name() == *variant) else {
            eprintln!("no pantry variant named {variant:?} — the list and the group disagree");
            std::process::exit(1);
        };
        let png = match capture(*cols, *rows, |frame| {
            let area = frame.area();
            ingredient.render(area, frame.buffer_mut());
        }) {
            Ok(png) => png,
            Err(e) => {
                eprintln!("{stem}: {e}");
                std::process::exit(1);
            }
        };
        let path = out.join(format!("{stem}.png"));
        if let Err(e) = std::fs::write(&path, &png) {
            eprintln!("{}: {e}", path.display());
            std::process::exit(1);
        }
        println!("{} ({} bytes)", path.display(), png.len());
    }
}

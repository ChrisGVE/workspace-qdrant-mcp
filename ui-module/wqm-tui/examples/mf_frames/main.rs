//! The modal framework's frames as PNGs, over the list the pantry group itself produces.
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
//! There is no list in this file. `views::modal_framework::index::frames` walks the group and
//! derives one, so a variant added, renamed or removed shows up here without anyone editing
//! anything — the failure that motivated it is in that module's docs.
//!
//! Every frame names a **pantry variant** rather than re-drawing anything, so the PNG and
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


fn main() {
    let arg = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());

    let frames = wqm_tui::views::modal_framework::index::frames();
    // Both ingredient modules are the one `Modal Framework` group.
    let ingredients: Vec<_> = wqm_tui::views::modal_framework::shipping::ingredient::ingredients()
        .into_iter()
        .chain(wqm_tui::views::modal_framework::ingredient::ingredients())
        .collect();
    if arg == "--list" {
        for frame in &frames {
            println!(
                "{}\t{}\t{}x{}",
                frame.stem, frame.variant, frame.cols, frame.rows
            );
        }
        return;
    }

    // The frames index, written from the same walk — so the file beside this one is a REPORT of
    // the group rather than a third list to keep in step with it.
    if arg == "--tsv" {
        print!("{}", wqm_tui::views::modal_framework::index::tsv());
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

    // `index::frames` walked the same ingredients to build the list, so a lookup here cannot
    // miss. It is still checked, because the alternative to an explicit exit is a silently
    // absent PNG.
    for frame in &frames {
        let Some(ingredient) = ingredients.iter().find(|i| i.name() == frame.variant) else {
            eprintln!(
                "no pantry variant named {:?} — the index and the group disagree",
                frame.variant
            );
            std::process::exit(1);
        };
        let png = match capture(frame.cols, frame.rows, |f| {
            let area = f.area();
            ingredient.render(area, f.buffer_mut());
        }) {
            Ok(png) => png,
            Err(e) => {
                eprintln!("{}: {e}", frame.stem);
                std::process::exit(1);
            }
        };
        let path = out.join(format!("{}.png", frame.stem));
        if let Err(e) = std::fs::write(&path, &png) {
            eprintln!("{}: {e}", path.display());
            std::process::exit(1);
        }
        println!("{} ({} bytes)", path.display(), png.len());
    }
}

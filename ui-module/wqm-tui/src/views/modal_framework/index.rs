//! **The frame index, derived from the pantry group** — one list, and nothing to keep in step.
//!
//! The PNG producer used to carry its own table of `(stem, variant, size)`. It looked up each
//! variant by name and exited when one was missing, which is the right failure — but it is a
//! failure, and it happened: renaming `32 edge — spacing only` in the group left the example
//! naming a variant that no longer existed, so it stopped after 32 frames and the five after it
//! (the `NO_COLOR` fallback, `ansi16`, the floor frames) were never rendered at all. A list that
//! has to be kept in step with another list eventually is not.
//!
//! So there is no second list. [`frames`] walks the group's own ingredients and derives an entry
//! per frame; the example renders what it returns, and `MF-ROUND2-FRAMES.tsv` is written from the
//! same walk. A variant added, renamed or removed in either ingredient module shows up in all
//! three without anyone touching anything.
//!
//! # What is derived, and the one thing that cannot be
//!
//! The **stem** is derived: the variant's own name, slugged. The **size** cannot be, because it
//! is a fact about what the frame is FOR rather than about what it is called — the size model has
//! to be seen holding at a third size, the minimum window only exists at the minimum, and the
//! too-small message only exists below it. [`SIZES`] is that table, keyed by the number the
//! variant's name already carries, and it is four lines rather than a parallel copy of the list.

use crate::views::modal_framework::{ingredient as states, shipping::ingredient as ruled};

/// The screen the storyboard is drawn at, and the floor Chris named.
pub const SCREEN: (u16, u16) = (125, 34);
pub const FLOOR: (u16, u16) = (100, 30);

/// The sizes a numbered variant renders at, where one screen is not the answer.
///
/// Keyed by the two-digit prefix the variant names already carry, so a renamed variant keeps its
/// sizes and a renumbered one is a deliberate change in one place.
///
/// - `00` — the size model has to be seen holding at a THIRD size, or "five columns each side"
///   reads as a coincidence of 125x34.
/// - `01` / `02` — the minimum window only exists at the minimum, and the too-small message only
///   below it. Rendering either at 125x34 would draw something that cannot happen.
/// - `04` / `11` — worth seeing at the floor as well: a drill-down with both bars and an edit
///   frame are the two that lose the most room there.
const SIZES: &[(&str, &[(u16, u16)])] = &[
    ("00", &[SCREEN, FLOOR, (200, 40)]),
    ("01", &[(44, 20)]),
    ("02", &[(40, 18)]),
    ("04", &[SCREEN, FLOOR]),
    ("11", &[SCREEN, FLOOR]),
];

/// One frame to render: what to call the file, which variant draws it, and how big.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Frame {
    pub stem: String,
    pub variant: String,
    pub cols: u16,
    pub rows: u16,
}

/// Every frame the group can produce, in the order the group lists them.
///
/// The two ingredient modules are one group (`Modal Framework`) and are walked as one: `ruled`
/// has a frame per clause Chris ruled on, `states` has the states a window enters that no clause
/// is about. Both are frames of the same thing.
pub fn frames() -> Vec<Frame> {
    let mut out = Vec::new();
    for ingredient in ruled::ingredients()
        .iter()
        .chain(states::ingredients().iter())
    {
        let variant = ingredient.name();
        let sizes = sizes_for(variant);
        let many = sizes.len() > 1;
        for (cols, rows) in sizes {
            let mut stem = format!("mf-{}", slug(variant));
            if many {
                // Only where a variant has more than one, so the common case keeps a clean name
                // and the exception says out loud which size it is.
                stem.push_str(&format!("-{cols}x{rows}"));
            }
            out.push(Frame {
                stem,
                variant: variant.to_string(),
                cols: *cols,
                rows: *rows,
            });
        }
    }
    out
}

/// The sizes this variant renders at — [`SIZES`] if its number is in there, one screen otherwise.
fn sizes_for(variant: &str) -> &'static [(u16, u16)] {
    let number = variant.split_whitespace().next().unwrap_or_default();
    SIZES
        .iter()
        .find(|(key, _)| *key == number)
        .map(|(_, sizes)| *sizes)
        .unwrap_or(&[SCREEN])
}

/// A variant's name as a file stem.
///
/// Parentheticals go, because they are asides for a reader of the list rather than part of what
/// the frame shows — `(dump at 125x34, 100x30, 200x40)` would otherwise put three sizes into the
/// name of a file that is one of them.
fn slug(variant: &str) -> String {
    let mut text = String::with_capacity(variant.len());
    let mut depth = 0usize;
    for c in variant.chars() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 => text.push(c),
            _ => {}
        }
    }
    let mut slug = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            slug.extend(c.to_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let slug = slug.trim_matches('-');
    // Long enough that no variant name in the group is cut today — the longest is 48 — and short
    // enough to stay a filename. Cut on a word boundary so a truncated stem never ends mid-word,
    // which is how two frames come to share a name; `no_two_frames_write_the_same_file` is what
    // catches it if a future name does collide past the cut.
    match slug.char_indices().nth(56) {
        None => slug.to_string(),
        Some((cut, _)) => slug[..slug[..cut].rfind('-').unwrap_or(cut)].to_string(),
    }
}

/// The comment block at the head of `MF-ROUND2-FRAMES.tsv`.
///
/// Here rather than in the file, because the file is generated: anything written into it by hand
/// is lost on the next regeneration, which is the property that makes it worth trusting.
const TSV_HEADER: &str = "\
# Modal-framework frames index. Tab-separated; `#` lines are comments.
#
# GENERATED — do not edit. Every row is derived from the `Modal Framework` pantry group, so the
# index cannot drift from the frames the way a hand-kept list did (see
# `views::modal_framework::index` for the failure that motivated it).
#
# REGENERATE
#   cargo run -q -p wqm-tui --features png-capture,tui-pantry --example mf_frames -- --tsv \\
#     > MF-ROUND2-FRAMES.tsv
#
# REPRODUCE A FRAME
#   PNG (all of them):  cargo run -p wqm-tui --features png-capture,tui-pantry \\
#                         --example mf_frames -- out/          -> out/<name>.png
#   grid / ANSI (one):  cargo pantry dump \"Modal Framework\" --variant \"<variant>\" \\
#                         --size <size> > <name>.ans
#   browse:             cargo pantry          (Views tab, group \"Modal Framework\")
#
# Both instruments render the SAME pantry variant, so they cannot disagree about what was
# drawn. Judge COLOUR from the PNG and GLYPHS/ATTRIBUTES from the grid — png-capture drops
# the `\u{25cf}` radio glyph (wqm#283), U+E0B0 is invisible in most text views, and no still image
# can show SLOW_BLINK.
#
# The theme is Catppuccin Mocha unless the variant says otherwise.
#
";

/// `MF-ROUND2-FRAMES.tsv`, in full.
///
/// It lives here rather than in the example so a test can compare the committed file against it.
/// That is the one drift this design still allows: the index cannot disagree with the group,
/// because it is derived from it — but a FILE checked into git can fall behind the code that
/// writes it, and nothing about being generated stops that.
pub fn tsv() -> String {
    let group = ruled::ingredients()
        .into_iter()
        .chain(states::ingredients())
        .collect::<Vec<_>>();
    let describe = |variant: &str| {
        group
            .iter()
            .find(|i| i.name() == variant)
            .map(|i| i.description().to_string())
            .unwrap_or_default()
    };
    let mut out = String::from(TSV_HEADER);
    out.push_str("name\tsize\tpantry variant\twhat it shows\n");
    for frame in frames() {
        out.push_str(&format!(
            "{}\t{}x{}\t{}\t{}\n",
            frame.stem,
            frame.cols,
            frame.rows,
            frame.variant,
            describe(&frame.variant)
        ));
    }
    out
}

#[cfg(test)]
mod tests;

//! **What is left to go wrong once the list stopped being a list.**
//!
//! The defect this module exists for: `32 edge — spacing only` was renamed in the group and the
//! PNG producer's own table was not, so the run stopped at that frame and the five after it — the
//! `NO_COLOR` fallback, `ansi16` and the two floor frames — were never rendered.
//!
//! The fix was structural rather than a test: [`super::frames`] derives the list FROM the group,
//! so a rename now flows straight through and the two cannot disagree at all. Which means the two
//! coverage tests below would pass today even if the rename were made again — they are proved
//! against the derivation, not against the bug, and that is worth saying out loud. They stay
//! because they are cheap and because they pin the invariant that a future shortcut (a hardcoded
//! entry, a filtered walk) would break.
//!
//! What can still drift is the FILE. `MF-ROUND2-FRAMES.tsv` is generated, and being generated
//! does not make a committed copy current — only regenerating it does. That is
//! [`the_committed_frames_index_is_what_the_generator_writes`], and unlike the two above it was
//! checked by making the defect and watching it fail.

use super::*;
use crate::views::modal_framework::{ingredient as states, shipping::ingredient as ruled};

fn group() -> Vec<String> {
    ruled::ingredients()
        .iter()
        .chain(states::ingredients().iter())
        .map(|i| i.name().to_string())
        .collect()
}

/// Every frame the index names is a variant the group has.
///
/// Structurally true now — see the module docs. It pins the derivation against a future shortcut,
/// which is a smaller claim than it looks like and should be read as one.
#[test]
fn every_indexed_frame_names_a_variant_the_group_has() {
    let group = group();
    for frame in frames() {
        assert!(
            group.contains(&frame.variant),
            "the index names {:?}, which the group does not have — the PNG run would stop here \
             and every frame after it would go unrendered",
            frame.variant
        );
    }
}

/// And every variant the group has is drawn at least once.
///
/// This is the direction that would fail SILENTLY if the walk were ever narrowed: a variant with
/// a pantry entry and no PNG produces no error, just a frame nobody ever sees.
#[test]
fn every_variant_in_the_group_is_drawn_at_least_once() {
    let drawn: Vec<String> = frames().into_iter().map(|f| f.variant).collect();
    for variant in group() {
        assert!(
            drawn.contains(&variant),
            "{variant:?} is in the group and in no frame, so it has a pantry entry and no PNG"
        );
    }
}

/// Two frames sharing a file name is one frame silently overwriting another.
///
/// The real risk is the slug, not the list: it drops parentheticals and truncates, so two
/// variants differing only in an aside or past the cut would collide and the second would win.
#[test]
fn no_two_frames_write_the_same_file() {
    let mut stems: Vec<String> = frames().into_iter().map(|f| f.stem).collect();
    stems.sort();
    let mut unique = stems.clone();
    unique.dedup();
    assert_eq!(
        stems, unique,
        "two frames write the same PNG, so one of them is not in the output at all"
    );
}

/// A stem is a filename: no spaces, no punctuation, nothing a shell has to be told about.
#[test]
fn a_stem_is_something_a_shell_never_has_to_quote() {
    for frame in frames() {
        assert!(
            frame
                .stem
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            "{:?} is not a plain filename",
            frame.stem
        );
        assert!(frame.stem.starts_with("mf-"), "{:?}", frame.stem);
        assert!(!frame.stem.ends_with('-'), "{:?}", frame.stem);
        assert!(!frame.stem.contains("--"), "{:?}", frame.stem);
    }
}

/// The sizes table earns its four lines: the three frames that exist only at their own size are
/// rendered there and nowhere else.
#[test]
fn the_frames_that_exist_only_at_one_size_are_rendered_only_there() {
    let sizes_of = |needle: &str| -> Vec<(u16, u16)> {
        frames()
            .into_iter()
            .filter(|f| f.variant.starts_with(needle))
            .map(|f| (f.cols, f.rows))
            .collect()
    };
    assert_eq!(sizes_of("00"), vec![SCREEN, FLOOR, (200, 40)]);
    assert_eq!(
        sizes_of("01"),
        vec![(44, 20)],
        "the minimum window only exists at the minimum"
    );
    assert_eq!(
        sizes_of("02"),
        vec![(40, 18)],
        "the too-small message only exists below the floor"
    );
    // …and a variant with nothing to say about size gets the one screen.
    assert_eq!(sizes_of("09"), vec![SCREEN]);
}

/// A frame rendered at more than one size says which size it is, and one rendered at a single
/// size does not carry a suffix nobody needs.
#[test]
fn only_a_multi_size_frame_carries_its_size_in_its_name() {
    for frame in frames() {
        let suffix = format!("-{}x{}", frame.cols, frame.rows);
        let multi = frames()
            .iter()
            .filter(|f| f.variant == frame.variant)
            .count()
            > 1;
        assert_eq!(
            frame.stem.ends_with(&suffix),
            multi,
            "{:?} at {}x{}",
            frame.stem,
            frame.cols,
            frame.rows
        );
    }
}

/// **The committed index matches what the code would write.**
///
/// This is the one drift the design still allows, and it is worth being explicit about why the
/// tests above cannot catch it: the index is DERIVED from the group, so renaming a variant flows
/// straight through and the two can never disagree. What can fall behind is the FILE — somebody
/// renames a variant, the group and the index agree perfectly, and `MF-ROUND2-FRAMES.tsv` still
/// says the old name because nobody re-ran the generator. Being generated does not make a file
/// current; regenerating it does.
///
/// The fix when this fails is in the failure message, because a diff of 73 lines is not a thing
/// anyone should be asked to apply by hand.
#[test]
fn the_committed_frames_index_is_what_the_generator_writes() {
    let committed = include_str!("../../../../MF-ROUND2-FRAMES.tsv");
    let generated = tsv();
    assert_eq!(
        committed,
        generated,
        "MF-ROUND2-FRAMES.tsv is behind the group. Regenerate it:\n  \
         cargo run -q -p wqm-tui --features png-capture,tui-pantry --example mf_frames -- --tsv \
         > MF-ROUND2-FRAMES.tsv"
    );
}
